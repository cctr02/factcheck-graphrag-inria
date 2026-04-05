#!/usr/bin/env python3
"""
build_evaluate.py - Build a formatted benchmark dataset for evaluate.

Why this script exists
----------------------
The evaluation benchmark files (queries.csv / documents.csv / relations.csv)
use their own sequential doc_ids that do NOT match the ids used inside the
knowledge graph.  The graph is built from english_fact_checks.csv and uses
*those* row ids (0, 1, 2, ...) as its doc_ids.

This script:
  1. Rebuilds the mapping   documents.csv doc_id -> english_fact_checks.csv id
     using exact title matching (first line of doc_text == title column).
  2. Filters queries to those whose ground-truth document is in the first
     --max-docs rows of english_fact_checks.csv (= the slice that was
     processed during the knowledge-graph build).
  3. Reformats each query from a raw social-media post into a clean factual
     claim statement using Ollama.
  4. Saves two CSV files usable directly by  cli.py evaluate :
       eval_queries_top<N>.csv   - (query_id, claim_text)
       eval_relations_top<N>.csv - (query_id, ef_doc_id)
     Plus a full id-mapping file for inspection:
       eval_id_map.csv

Usage
-----
  uv run build_evaluate.py --max-docs 10000
  uv run build_evaluate.py --max-docs 10000 --skip-llm   # bypass Ollama (for testing)
  uv run build_evaluate.py --help

Then run evaluation with the formatted files:
  uv run cli.py evaluate --graph-id <ID> \\
    --queries data/eval_queries_top10000.csv \\
    --relations data/eval_relations_top10000.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default paths / settings
# ---------------------------------------------------------------------------

_DEFAULT_QUERIES       = "data/queries.csv"
_DEFAULT_DOCUMENTS     = "data/documents.csv"
_DEFAULT_RELATIONS     = "data/relations.csv"
_DEFAULT_ENGLISH_FACTS = "data/english_fact_checks.csv"
_DEFAULT_OUTPUT_DIR    = "data"
_DEFAULT_OLLAMA_MODEL  = "llama3.1:latest"
_DEFAULT_OLLAMA_EP     = "http://localhost:11434/api/generate"

# ---------------------------------------------------------------------------
# Ollama LLM call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You extract factual claims from social-media posts. "
    "Return ONLY the core factual claim as a short, clear sentence.\n\n"
    "Rules:\n"
    "- If there is a named speaker with a direct quote, write: <Speaker> claimed \"<quote>\".\n"
    "- Remove all URLs, hashtags, @mentions, emojis, and social-media jargon.\n"
    "- Keep it to 1-2 sentences maximum.\n"
    "- Do NOT evaluate, comment, or prefix with 'False:'/'True:'.\n"
    "- Return ONLY the claim, nothing else.\n\n"
    "Examples:\n"
    "IN : '\" i\\'m not saving your woke brand \"\\n-Michael Jordan.\\nDo you support this? https://t.co/xxx'\n"
    "OUT: Michael Jordan claimed \"I'm not saving your woke brand.\"\n\n"
    "IN : '\"@electionsvic have announced a historic partnership with @wef to oversee and count the votes '\n"
    "OUT: The Victorian Electoral Commission announced a partnership with the World Economic Forum to oversee the 2022 state election votes.\n\n"
    "IN : '\"Biden Did Not Fall Asleep While Meeting with Maui Fire Victims, C-SPAN Video Confirms\" https://t.co/xxx'\n"
    "OUT: Biden did not fall asleep while meeting with Maui fire victims.\n\n"
    "IN : '\"...November\\'s snowfall across the Northern Hemisphere was running at rates exceeding a half-a-century average...\" https://t.co/xxx'\n"
    "OUT: November's snowfall across the Northern Hemisphere was running at rates exceeding a half-century average."
)


def _call_ollama(
    text: str,
    model: str = _DEFAULT_OLLAMA_MODEL,
    endpoint: str = _DEFAULT_OLLAMA_EP,
    timeout: int = 60,
) -> str:
    """Reformat a social-media post as a factual claim via Ollama."""
    payload = {
        "model": model,
        "prompt": f"Transform this social-media post into a factual claim:\n\n{text}",
        "system": _SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        tqdm.write(f"  [WARN] Ollama timeout, keeping original text.", file=sys.stderr)
        return text
    except Exception as exc:
        tqdm.write(f"  [WARN] Ollama error: {exc}, keeping original text.", file=sys.stderr)
        return text


# ---------------------------------------------------------------------------
# doc_id mapping: documents.csv  ->  english_fact_checks.csv
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Light normalisation: strip, collapse whitespace, lower-case."""
    return " ".join(s.lower().split())


def build_id_mapping(
    documents_df: pd.DataFrame,
    ef_csv_path: str,
) -> tuple:
    """
    Map each documents.csv doc_id to its english_fact_checks.csv id.

    Matching strategy
    -----------------
    The text in documents.csv appears to be  title + "\\n" + body  from
    english_fact_checks.csv.

    Step 1 - title match (O(1)):
        First line of doc_text  ==  english_fact_checks.title

    Step 2 - body-prefix match (O(1)):
        First 300 chars of doc_text (after the first line)
        ==  first 300 chars of english_fact_checks.body  (stripped)

    Both comparisons use light normalisation (lower-case + whitespace collapse)
    to absorb minor encoding differences.

    Returns
    -------
    id_map    : dict  {documents_doc_id (int) -> ef_id (int)}
    unmatched : list of documents_doc_ids that could not be matched
    """
    print("  Loading english_fact_checks.csv (id + title + body) ...", flush=True)
    ef_df = pd.read_csv(
        ef_csv_path,
        header=0,
        dtype={"id": str},
        usecols=["id", "title", "body"],
    )
    ef_df["id"] = pd.to_numeric(ef_df["id"], errors="coerce").fillna(-1).astype(int)

    # --- build lookup tables ---
    title_to_ef: dict[str, int] = {}
    body_to_ef:  dict[str, int] = {}
    for row in tqdm(
        ef_df.itertuples(index=False),
        total=len(ef_df),
        desc="  Building ef lookup",
        leave=False,
    ):
        ef_id = int(row.id)
        if ef_id < 0:
            continue
        t = _normalize(str(row.title or ""))
        if t:
            title_to_ef.setdefault(t, ef_id)
        b = _normalize(str(row.body or ""))[:300]
        if b:
            body_to_ef.setdefault(b, ef_id)

    print(
        f"  ef lookup ready: {len(title_to_ef):,} titles, {len(body_to_ef):,} body prefixes",
        flush=True,
    )

    # --- match each documents.csv row ---
    id_map:    dict[int, int] = {}
    unmatched: list[int]      = []

    for row in tqdm(
        documents_df.itertuples(index=False),
        total=len(documents_df),
        desc="  Matching documents",
        leave=False,
    ):
        doc_id   = int(row.doc_id)
        doc_text = str(row.doc_text or "").strip()

        parts      = doc_text.split("\n", 1)
        first_line = _normalize(parts[0])
        body_part  = _normalize(parts[1] if len(parts) > 1 else "")[:300]

        # Strategy 1: title
        if first_line in title_to_ef:
            id_map[doc_id] = title_to_ef[first_line]
            continue

        # Strategy 2: body prefix
        if body_part and body_part in body_to_ef:
            id_map[doc_id] = body_to_ef[body_part]
            continue

        # Strategy 3: doc_text prefix as body (in case there's no separate title line)
        full_prefix = _normalize(doc_text)[:300]
        if full_prefix in body_to_ef:
            id_map[doc_id] = body_to_ef[full_prefix]
            continue

        unmatched.append(doc_id)

    return id_map, unmatched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a formatted evaluation set for cli.py evaluate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--max-docs", type=int, default=10_000,
        help=(
            "Only keep queries whose ground-truth doc is among the first N rows "
            "of english_fact_checks.csv (same slice used during graph build). "
            "Default: 10 000"
        ),
    )
    ap.add_argument("--queries",       default=_DEFAULT_QUERIES,       help=f"(default: {_DEFAULT_QUERIES})")
    ap.add_argument("--documents",     default=_DEFAULT_DOCUMENTS,     help=f"(default: {_DEFAULT_DOCUMENTS})")
    ap.add_argument("--relations",     default=_DEFAULT_RELATIONS,     help=f"(default: {_DEFAULT_RELATIONS})")
    ap.add_argument("--english-facts", default=_DEFAULT_ENGLISH_FACTS, help=f"(default: {_DEFAULT_ENGLISH_FACTS})")
    ap.add_argument("--output-dir",    default=_DEFAULT_OUTPUT_DIR,    help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--ollama-model",    default=_DEFAULT_OLLAMA_MODEL)
    ap.add_argument("--ollama-endpoint", default=_DEFAULT_OLLAMA_EP)
    ap.add_argument(
        "--skip-llm", action="store_true",
        help="Skip Ollama formatting (keep raw query text - useful for testing).",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------ load
    print("=" * 60)
    print("BUILD EVALUATE - Loading benchmark files")
    print("=" * 60)
    print(f"  Queries   : {args.queries}")
    print(f"  Documents : {args.documents}")
    print(f"  Relations : {args.relations}")
    print(f"  EF CSV    : {args.english_facts}")
    print(f"  max-docs  : {args.max_docs:,}")

    queries_df   = pd.read_csv(args.queries,   header=0, dtype=str)
    documents_df = pd.read_csv(args.documents, header=0, dtype=str)
    relations_df = pd.read_csv(args.relations, header=0, dtype=str)

    queries_df.columns   = ["query_id", "query_text"]
    documents_df.columns = ["doc_id",   "doc_text"]
    relations_df.columns = ["query_id", "doc_id"]

    query_text_map: dict[str, str] = {
        str(r.query_id): str(r.query_text)
        for r in queries_df.itertuples(index=False)
    }
    print(
        f"\n  {len(queries_df):,} queries | {len(documents_df):,} documents | "
        f"{len(relations_df):,} relations\n"
    )

    # ------------------------------------------------------ id-mapping
    print("Building doc-id mapping (documents.csv -> english_fact_checks.csv) ...")
    id_map, unmatched = build_id_mapping(documents_df, args.english_facts)

    total_docs = len(documents_df)
    print(
        f"\n  Matched   : {len(id_map):,} / {total_docs:,} "
        f"({len(id_map)/total_docs*100:.1f}%)"
    )
    if unmatched:
        print(f"  Unmatched : {len(unmatched):,}  (will be skipped)")

    # ------------------------------------------------------ filter
    def _norm_id(val: str) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return -1

    valid: list[tuple[str, str, int]] = []   # (query_id, query_text, ef_doc_id)
    skipped_no_map  = 0
    skipped_too_far = 0

    for r in relations_df.itertuples(index=False):
        qid     = str(r.query_id)
        docs_id = _norm_id(str(r.doc_id))
        if docs_id < 0:
            continue
        if docs_id not in id_map:
            skipped_no_map += 1
            continue
        ef_id = id_map[docs_id]
        if ef_id >= args.max_docs:
            skipped_too_far += 1
            continue
        if qid not in query_text_map:
            continue
        valid.append((qid, query_text_map[qid], ef_id))

    # Remove duplicates (same query_id appearing multiple times)
    seen_qids: set[str] = set()
    deduped: list[tuple[str, str, int]] = []
    for item in valid:
        if item[0] not in seen_qids:
            seen_qids.add(item[0])
            deduped.append(item)
    valid = deduped

    print(f"\nFiltered queries (GT doc id < {args.max_docs:,}): {len(valid):,}")
    print(f"  Skipped (doc not mapped)       : {skipped_no_map:,}")
    print(f"  Skipped (doc outside max-docs) : {skipped_too_far:,}")

    if not valid:
        print(
            "\n[ERROR] No valid queries found after filtering.\n"
            "  - Try increasing --max-docs (current graph was built with ~10 000 docs).\n"
            "  - Check that --english-facts points to the same source used during build.\n"
            "  - Run with --skip-llm first to test the mapping without Ollama."
        )
        sys.exit(1)

    # ------------------------------------------------------ reformat with LLM
    formatted: list[tuple[str, str, int]] = []   # (query_id, claim, ef_doc_id)

    if args.skip_llm:
        print("\n--skip-llm: using raw query text (no Ollama call).")
        formatted = list(valid)
    else:
        print(
            f"\nFormatting {len(valid):,} queries with Ollama "
            f"({args.ollama_model}) ..."
        )
        print(
            "  Tip: interrupt with Ctrl-C at any time; re-run with --skip-llm to bypass LLM.\n"
        )
        for qid, qtxt, efid in tqdm(valid, desc="Formatting", unit="query"):
            claim = _call_ollama(qtxt, args.ollama_model, args.ollama_endpoint)
            formatted.append((qid, claim, efid))

    # ------------------------------------------------------ save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    suffix        = f"_top{args.max_docs}"
    queries_out   = out / f"eval_queries{suffix}.csv"
    relations_out = out / f"eval_relations{suffix}.csv"
    idmap_out     = out / f"eval_id_map.csv"

    # eval_queries_top<N>.csv  - same format as queries.csv
    with open(queries_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["0", "1"])
        for qid, claim, _ in formatted:
            w.writerow([qid, claim])

    # eval_relations_top<N>.csv  - same format as relations.csv but with ef_doc_id
    with open(relations_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["0", "1"])
        for qid, _, efid in formatted:
            w.writerow([qid, efid])

    # eval_id_map.csv  - full mapping for inspection
    with open(idmap_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["documents_doc_id", "ef_doc_id"])
        for docs_id, ef_id in sorted(id_map.items()):
            w.writerow([docs_id, ef_id])

    print(f"\n{'='*60}")
    print(f"Saved {len(formatted):,} formatted queries:")
    print(f"  Queries   : {queries_out}")
    print(f"  Relations : {relations_out}")
    print(f"  ID map    : {idmap_out}")
    print(f"\nNext step - run evaluation:")
    print(f"  uv run cli.py evaluate --graph-id <GRAPH_ID> \\")
    print(f"    --queries {queries_out} \\")
    print(f"    --relations {relations_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
