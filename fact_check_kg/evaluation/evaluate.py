"""
Retrieval Evaluation Benchmark.

Measures how well the hybrid retrieval system surfaces the ground-truth
fact-checking document for each social-media post (query).

Data format (all CSVs have numeric column names '0' and '1'):
    queries.csv   : 0=query_id,  1=query_text
    documents.csv : 0=doc_id,    1=doc_text
    relations.csv : 0=query_id,  1=doc_id   (ground-truth alignments)

Only queries whose ground-truth document is present in the loaded graph are
evaluated; the others are reported as out-of-corpus.

Metrics:
    Hit@K  (K=1,5,10,50) — fraction of queries where target doc is in top-K
    MRR    — Mean Reciprocal Rank  (1/rank; 0 when doc not found)
    Mean / Median Rank — average position of the relevant document (found only)
    Per-strategy Hit@10 — individual strategy analysis via max-pooled doc scores
"""

import csv
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Ollama query-formatting helpers
# ---------------------------------------------------------------------------

_DEFAULT_OLLAMA_MODEL   = "llama3.1:latest"
_DEFAULT_OLLAMA_EP      = "http://localhost:11434/api/generate"
_DEFAULT_QUERY_CACHE    = "data/eval_queries_formatted_cache.csv"

_FORMAT_SYSTEM_PROMPT = (
    "You extract factual claims from social-media posts. "
    "Return ONLY the core factual claim as a short, clear sentence.\n\n"
    "Rules:\n"
    "- If there is a named speaker with a direct quote, write: <Speaker> claimed \"<quote>\".\n"
    "- Remove all URLs, hashtags, @mentions, emojis, and social-media jargon.\n"
    "- Keep it to 1-2 sentences maximum.\n"
    "- Do NOT evaluate, comment, or prefix with 'False:'/'True:'.\n"
    "- Return ONLY the claim, nothing else.\n\n"
    "Examples:\n"
    "IN : '\"i\\'m not saving your woke brand \"\\n-Michael Jordan.\\nDo you support this?'\n"
    "OUT: Michael Jordan claimed \"I'm not saving your woke brand.\"\n\n"
    "IN : 'Biden Did Not Fall Asleep While Meeting with Maui Fire Victims'\n"
    "OUT: Biden did not fall asleep while meeting with Maui fire victims."
)


def _call_ollama(
    text: str,
    model: str = _DEFAULT_OLLAMA_MODEL,
    endpoint: str = _DEFAULT_OLLAMA_EP,
    timeout: int = 60,
) -> str:
    """Reformat a social-media post as a factual claim via Ollama. Returns original on error."""
    payload = {
        "model": model,
        "prompt": f"Transform this social-media post into a factual claim:\n\n{text}",
        "system": _FORMAT_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip() or text
    except Exception:
        return text


def _load_query_cache(cache_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """Load query_id -> formatted_query from CSV cache. Returns empty dict if file missing."""
    p = Path(cache_path)
    if not p.exists():
        return {}
    cache: Dict[str, str] = {}
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = row.get("query_id", "").strip()
                fmt = row.get("formatted_query", "").strip()
                if qid:
                    cache[qid] = fmt
        if logger:
            logger.info(f"Query cache loaded: {len(cache)} entries from {cache_path}")
    except Exception as exc:
        if logger:
            logger.warning(f"Could not load query cache ({exc}), starting fresh.")
    return cache


def _save_query_cache(cache_path: str, cache: Dict[str, str]) -> None:
    """Save query_id -> formatted_query dict to CSV cache."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "formatted_query"])
        for qid, fmt in sorted(cache.items()):
            w.writerow([qid, fmt])


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_benchmark(
    queries_path: str,
    documents_path: str,
    relations_path: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three benchmark CSVs and normalise column names.

    Returns DataFrames with columns:
        queries_df   : query_id (str), query_text (str)
        documents_df : doc_id (str),   doc_text (str)
        relations_df : query_id (str), doc_id (str)
    """
    if logger:
        logger.info("Loading benchmark CSVs ...")

    queries_df   = pd.read_csv(queries_path,   header=0, dtype=str)
    documents_df = pd.read_csv(documents_path, header=0, dtype=str)
    relations_df = pd.read_csv(relations_path, header=0, dtype=str)

    # Original column names are the strings "0" and "1"
    queries_df.columns   = ['query_id', 'query_text']
    documents_df.columns = ['doc_id',   'doc_text']
    relations_df.columns = ['query_id', 'doc_id']

    if logger:
        logger.info(f"  Queries   : {len(queries_df)}")
        logger.info(f"  Documents : {len(documents_df)}")
        logger.info(f"  Relations : {len(relations_df)}")

    return queries_df, documents_df, relations_df


def get_graph_doc_ids(graph) -> set:
    """
    Return the set of doc_id strings present in the graph.
    Reads the 'doc_id' attribute of every node typed 'doc'.
    """
    return {
        str(data['doc_id'])
        for _, data in graph.nodes(data=True)
        if data.get('type') == 'document' and data.get('doc_id') is not None
    }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    retriever,
    queries_path: str,
    documents_path: str,
    relations_path: str,
    top_k_eval: Optional[List[int]] = None,
    max_queries: Optional[int] = None,
    max_rank: int = 10,
    output_dir: str = "outputs/evaluation",
    graph_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    num_workers: int = 1,
    query_cache_path: str = _DEFAULT_QUERY_CACHE,
    ollama_model: str = _DEFAULT_OLLAMA_MODEL,
    ollama_endpoint: str = _DEFAULT_OLLAMA_EP,
    skip_llm: bool = False,
) -> Dict:
    """
    Run the full retrieval evaluation benchmark.

    For each (query, ground-truth doc) pair whose doc is present in the graph,
    retrieval is run and the rank of the target document is recorded.

    Document-level scoring uses softmax-pooling (implemented in
    HybridRetriever._aggregate_to_documents): strongly-scoring chunks dominate
    while multiple good chunks still contribute.

    Args:
        retriever:      Initialised HybridRetriever instance.
        queries_path:   Path to queries.csv.
        documents_path: Path to documents.csv.
        relations_path: Path to relations.csv.
        top_k_eval:     K values for Hit@K (default: [1, 5, 10, 50]).
        max_queries:    Cap on the number of queries to evaluate (None = all).
        max_rank:       How many top documents to search for the target (default 500).
                        `retriever.top_docs` is temporarily set to this value.
        output_dir:     Directory for report files.
        graph_id:       Graph identifier used in output filenames.
        logger:         Logger instance.

    Returns:
        Evaluation report dictionary (also written to JSON + TXT).
    """
    if top_k_eval is None:
        top_k_eval = [1, 5, 10]
    if logger is None:
        logger = logging.getLogger(__name__)

    eval_start = time.time()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – load benchmark data
    # ------------------------------------------------------------------
    queries_df, documents_df, relations_df = load_benchmark(
        queries_path, documents_path, relations_path, logger=logger
    )

    query_texts: Dict[str, str] = {
        str(row['query_id']): str(row['query_text'])
        for _, row in queries_df.iterrows()
    }

    # ------------------------------------------------------------------
    # Step 2 – identify which doc_ids are in the graph
    # ------------------------------------------------------------------
    graph_doc_ids = get_graph_doc_ids(retriever.graph)
    logger.info(f"Documents in graph: {len(graph_doc_ids)}")

    # ------------------------------------------------------------------
    # Step 3 – filter relations to in-graph docs
    # ------------------------------------------------------------------
    # Strip any floating-point suffix that pandas may add (e.g. "422.0" -> "422")
    def _norm_id(val: str) -> str:
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val).strip()

    valid_pairs: List[Tuple[str, str]] = []
    for _, row in relations_df.iterrows():
        qid = _norm_id(row['query_id'])
        did = _norm_id(row['doc_id'])
        if did in graph_doc_ids and qid in query_texts:
            valid_pairs.append((qid, did))

    n_total_relations = len(relations_df)
    logger.info(f"Total relations        : {n_total_relations}")
    logger.info(f"Valid pairs (in graph) : {len(valid_pairs)}")
    logger.info(f"Coverage               : {len(valid_pairs)/n_total_relations*100:.1f}%")

    if max_queries:
        valid_pairs = valid_pairs[:max_queries]
        logger.info(f"Limiting evaluation to {max_queries} queries")

    # ------------------------------------------------------------------
    # Step 3.5 – format ONLY the queries that will actually be evaluated
    # ------------------------------------------------------------------
    if skip_llm:
        logger.info("--skip-llm: using raw query text (no Ollama formatting).")
    else:
        eval_qids = {qid for qid, _ in valid_pairs}
        cache = _load_query_cache(query_cache_path, logger)
        to_format = [(qid, query_texts[qid]) for qid in eval_qids if qid not in cache]
        if to_format:
            logger.info(
                f"Formatting {len(to_format)} queries with Ollama ({ollama_model}) "
                f"({len(eval_qids) - len(to_format)} already cached) ..."
            )
            for qid, qtxt in tqdm(to_format, desc="Formatting queries", unit="query"):
                cache[qid] = _call_ollama(qtxt, ollama_model, ollama_endpoint)
            _save_query_cache(query_cache_path, cache)
            logger.info(f"  Cache updated: {len(cache)} total entries -> {query_cache_path}")
        else:
            logger.info(
                f"Query formatting: all {len(eval_qids)} eval queries already in cache."
            )
        # Replace raw text with formatted versions for eval queries only
        for qid in eval_qids:
            if qid in cache:
                query_texts[qid] = cache[qid]

    # ------------------------------------------------------------------
    # Step 4 – override top_docs for evaluation depth
    # Retrieve at least 50 docs so top-50 display is meaningful,
    # but ranks beyond max_rank are still treated as "not found" for metrics.
    # ------------------------------------------------------------------
    display_depth = max(max_rank, 50)
    original_top_docs = retriever.top_docs
    retriever.top_docs = display_depth
    logger.info(f"retriever.top_docs temporarily set to {display_depth} (max_rank={max_rank})")

    # ------------------------------------------------------------------
    # Step 5 – run retrieval for each query
    # ------------------------------------------------------------------
    strategy_keys = [
        'S1_semantic',
        'S2_topic_dist',
        'S3_thematic',
        'S4_entity',
        'S5_claim_doc',
        'S6_claim_topic',
        'S7_claim_keyphrase',
    ]

    # ------------------------------------------------------------------
    # Pre-build doc_id → first-chunk text extract (for per-query log)
    # ------------------------------------------------------------------
    doc_id_to_text: Dict[str, str] = {}
    for n in retriever.chunk_nodes:
        did = str(retriever.graph.nodes[n].get('doc_id', ''))
        if did and did not in doc_id_to_text:
            doc_id_to_text[did] = retriever.graph.nodes[n].get('text', '')
    logger.info(f"Text index built for {len(doc_id_to_text)} documents (for log display)")

    # ------------------------------------------------------------------
    # Per-query worker (shared by sequential and parallel paths)
    # ------------------------------------------------------------------
    def _eval_one(pair_with_index):
        """Evaluate one query. Only reads from the retriever — thread-safe."""
        idx, (query_id, target_doc_id) = pair_with_index
        query_text = query_texts[query_id]
        t0 = time.time()
        try:
            ret = retriever.retrieve(query_text)
        except Exception as exc:
            elapsed = time.time() - t0
            return idx, {
                'query_id': query_id,
                'query_text': query_text,
                'target_doc_id': target_doc_id,
                'gt_doc_text': doc_id_to_text.get(target_doc_id, ''),
                'rank': None,
                'per_strategy_ranks': {k: None for k in strategy_keys},
                'error': str(exc),
                'retrieval_time_s': round(elapsed, 3),
                'top50_full_doc_ids': [],
                'top50_s1_doc_ids': [],
                'top1_full_doc_id': None,
                'top1_s1_doc_id': None,
                'top1_full_text': '',
                'top1_s1_text': '',
            }, elapsed, str(exc)

        elapsed = time.time() - t0
        first = ret[0] if ret else {}

        doc_results = first.get('doc_results', [])
        _raw_rank = next(
            (j for j, d in enumerate(doc_results, 1)
             if _norm_id(d['doc_id']) == target_doc_id),
            None,
        )
        # Ranks beyond max_rank count as "not found" for metrics
        rank = _raw_rank if (_raw_rank is not None and _raw_rank <= max_rank) else None

        per_strat_docs = first.get('per_strategy_doc_results', {})
        per_strategy_ranks = {}
        for sk in strategy_keys:
            _raw = next(
                (j for j, d in enumerate(per_strat_docs.get(sk, []), 1)
                 if _norm_id(d['doc_id']) == target_doc_id),
                None,
            )
            per_strategy_ranks[sk] = _raw if (_raw is not None and _raw <= max_rank) else None
        # Top-1 doc info for both pipelines (for report display)
        top1_full_doc_id = _norm_id(doc_results[0]['doc_id']) if doc_results else None
        s1_docs = per_strat_docs.get('S1_semantic', [])
        top1_s1_doc_id = _norm_id(s1_docs[0]['doc_id']) if s1_docs else None

        return idx, {
            'query_id': query_id,
            'query_text': query_text,
            'target_doc_id': target_doc_id,
            'gt_doc_text': doc_id_to_text.get(target_doc_id, ''),
            'rank': rank,
            'per_strategy_ranks': per_strategy_ranks,
            'retrieval_time_s': round(elapsed, 3),
            'top50_full_doc_ids': [d['doc_id'] for d in doc_results[:50]],
            'top50_s1_doc_ids':   [d['doc_id'] for d in s1_docs[:50]],
            'top1_full_doc_id': top1_full_doc_id,
            'top1_s1_doc_id': top1_s1_doc_id,
            'top1_full_text': doc_id_to_text.get(top1_full_doc_id, '') if top1_full_doc_id else '',
            'top1_s1_text': doc_id_to_text.get(top1_s1_doc_id, '') if top1_s1_doc_id else '',
            '_raw_ret': ret,   # kept for miss-analysis export; stripped before JSON save
        }, elapsed, None

    # ------------------------------------------------------------------
    # Pre-init KeyBERT before spawning threads (lazy-init is not thread-safe)
    # ------------------------------------------------------------------
    if num_workers > 1 and getattr(retriever, 'eta', 0.0) > 0.0 \
            and getattr(retriever, 'keybert_model', None) is None:
        logger.info("Pre-initialising KeyBERT to avoid race condition in parallel evaluation ...")
        from keybert import KeyBERT
        retriever.keybert_model = KeyBERT(model=retriever.embedding_model)

    mode_str = f"{num_workers} threads" if num_workers > 1 else "sequential"
    logger.info(f"\nRunning evaluation on {len(valid_pairs)} queries ({mode_str}) ...")

    indexed_pairs = list(enumerate(valid_pairs, start=1))
    # Pre-allocated list keeps original order regardless of completion order
    results_per_query: List[Optional[Dict]] = [None] * len(valid_pairs)
    completed = 0

    def _on_result(idx, result, elapsed, err_msg):
        """Record one finished query (always called from the main thread)."""
        nonlocal completed
        results_per_query[idx - 1] = result
        completed += 1
        if err_msg:
            tqdm.write(
                f"[WARNING] Query {result['query_id']} failed in {elapsed:.2f}s: {err_msg}"
            )

    # Silence ALL output from retrieve() during the evaluation loop.
    # propagate=False ensures nothing leaks to the parent "evaluate" logger
    # (which has 3 handlers: main log, daily log, console) — only tqdm is visible.
    _quiet = logging.getLogger(retriever.logger.name + ".eval_quiet")
    _quiet.setLevel(logging.CRITICAL + 1)   # above CRITICAL → drops everything
    _quiet.propagate = False                 # no propagation to parent handlers
    _quiet.handlers.clear()                  # no handlers of its own either
    _orig_retriever_logger = retriever.logger
    retriever.logger = _quiet

    try:
        if num_workers <= 1:
            # --- Sequential ---
            for item in tqdm(indexed_pairs, desc="Evaluating", unit="query"):
                _on_result(*_eval_one(item))
        else:
            # --- Parallel (thread pool) ---
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with tqdm(total=len(valid_pairs), desc="Evaluating", unit="query") as pbar:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_eval_one, item): item[0]
                        for item in indexed_pairs
                    }
                    for future in as_completed(futures):
                        _on_result(*future.result())
                        pbar.update(1)
    finally:
        retriever.logger = _orig_retriever_logger

    results_per_query = [r for r in results_per_query if r is not None]

    # Restore original value
    retriever.top_docs = original_top_docs

    # ------------------------------------------------------------------
    # Step 5b – save individual retrieval result files for misses
    # A "miss" = the target doc is NOT at rank 1 in BOTH pipelines.
    # Files go to outputs/retrieval_results/ in the same format as the
    # regular `retrieve` CLI command, named  results_{graph_id}_eval_q{qid}.
    # ------------------------------------------------------------------
    from fact_check_kg.retrieval.retrieval_results import RetrievalResultsManager
    miss_results_manager = RetrievalResultsManager(logger=logger)
    misses_for_save = [
        r for r in results_per_query
        if not r.get('error')
        and not (r.get('rank') == 1
                 and r.get('per_strategy_ranks', {}).get('S1_semantic') == 1)
    ]
    logger.info(
        f"Saving {len(misses_for_save)} miss retrieval result files "
        f"to {miss_results_manager.results_dir} ..."
    )
    for r in misses_for_save:
        raw_ret = r.get('_raw_ret') or []
        if not raw_ret:
            continue
        first_chunk = raw_ret[0]
        try:
            miss_results_manager.save_results(
                claim=r['query_text'],
                results=raw_ret,
                alpha=getattr(retriever, 'alpha', 0.0),
                beta=getattr(retriever, 'beta', 0.0),
                gamma=getattr(retriever, 'gamma', 0.0),
                delta=getattr(retriever, 'delta', 0.0),
                epsilon=getattr(retriever, 'epsilon', 0.0),
                zeta=getattr(retriever, 'zeta', 0.0),
                eta=getattr(retriever, 'eta', 0.0),
                graph_id=graph_id,
                custom_name=f"eval_q{r['query_id']}",
                doc_results=first_chunk.get('doc_results'),
                per_strategy_doc_results=first_chunk.get('per_strategy_doc_results'),
                matched_claims_s5=first_chunk.get('matched_claims_s5'),
                matched_claims_s6=first_chunk.get('matched_claims_s6'),
                matched_claims_s7=first_chunk.get('matched_claims_s7'),
                detected_themes=first_chunk.get('detected_themes'),
                extracted_entities=first_chunk.get('entity_info_s4'),
            )
        except Exception as exc:
            logger.warning(f"  Could not save retrieval result for query {r['query_id']}: {exc}")

    # Strip internal field before JSON serialisation
    for r in results_per_query:
        r.pop('_raw_ret', None)

    total_eval_time = time.time() - eval_start
    retrieval_times = [
        r['retrieval_time_s'] for r in results_per_query
        if 'retrieval_time_s' in r
    ]

    # ------------------------------------------------------------------
    # Step 6 – aggregate metrics
    # ------------------------------------------------------------------
    error_count   = sum(1 for r in results_per_query if r.get('error'))
    n_valid       = len(results_per_query) - error_count
    all_ranks     = [r['rank'] for r in results_per_query if r.get('rank') is not None and not r.get('error')]
    not_found_cnt = sum(1 for r in results_per_query if r.get('rank') is None and not r.get('error'))

    # Hit@K
    hits = {
        f'hit_at_{k}': (sum(1 for rk in all_ranks if rk <= k) / n_valid if n_valid else 0.0)
        for k in top_k_eval
    }

    # MRR  (not-found contributes 0)
    mrr = sum(1.0 / rk for rk in all_ranks) / n_valid if n_valid else 0.0

    # Rank stats on found docs only
    mean_rank   = float(np.mean(all_ranks))   if all_ranks else 0.0
    median_rank = float(np.median(all_ranks)) if all_ranks else 0.0

    # Per-strategy Hit@10
    per_strategy_hit_at_10: Dict[str, float] = {}
    for sk in strategy_keys:
        strat_ranks = [
            r['per_strategy_ranks'].get(sk)
            for r in results_per_query
            if not r.get('error') and r['per_strategy_ranks'].get(sk) is not None
        ]
        per_strategy_hit_at_10[sk] = (
            sum(1 for rk in strat_ranks if rk <= 10) / n_valid if n_valid else 0.0
        )

    # S1-only baseline (classic semantic RAG — no KG traversal)
    s1_ranks = [
        r['per_strategy_ranks']['S1_semantic']
        for r in results_per_query
        if not r.get('error') and r.get('per_strategy_ranks', {}).get('S1_semantic') is not None
    ]
    s1_not_found = sum(
        1 for r in results_per_query
        if not r.get('error') and r.get('per_strategy_ranks', {}).get('S1_semantic') is None
    )
    s1_hits = {
        f'hit_at_{k}': (sum(1 for rk in s1_ranks if rk <= k) / n_valid if n_valid else 0.0)
        for k in top_k_eval
    }
    s1_mrr         = sum(1.0 / rk for rk in s1_ranks) / n_valid if n_valid else 0.0
    s1_mean_rank   = float(np.mean(s1_ranks))   if s1_ranks else 0.0
    s1_median_rank = float(np.median(s1_ranks)) if s1_ranks else 0.0

    # Rank distribution
    def _bucket(rk: int) -> str:
        if rk == 1:     return '1'
        elif rk <= 5:   return '2-5'
        elif rk <= 10:  return '6-10'
        else:           return f'11-{max_rank}'

    rank_dist: Dict[str, int] = defaultdict(int)
    rank_dist['not_found'] = not_found_cnt
    for rk in all_ranks:
        rank_dist[_bucket(rk)] += 1

    s1_rank_dist: Dict[str, int] = defaultdict(int)
    s1_rank_dist['not_found'] = s1_not_found
    for rk in s1_ranks:
        s1_rank_dist[_bucket(rk)] += 1

    # ------------------------------------------------------------------
    # Step 7 – build report
    # ------------------------------------------------------------------
    report = {
        'metadata': {
            'timestamp':         datetime.now().strftime('%Y%m%d_%H%M%S'),
            'graph_id':          graph_id,
            'queries_path':      queries_path,
            'documents_path':    documents_path,
            'relations_path':    relations_path,
            'max_rank_searched': max_rank,
            'max_queries':       max_queries,
        },
        'coverage': {
            'total_relations':         n_total_relations,
            'valid_pairs_in_graph':    len(valid_pairs),
            'coverage_pct':            round(len(valid_pairs) / n_total_relations * 100, 2),
            'evaluated':               n_valid,
            'errors':                  error_count,
        },
        'metrics': {
            **{k: round(v, 4) for k, v in hits.items()},
            'mrr':             round(mrr, 4),
            'mean_rank':       round(mean_rank, 2),
            'median_rank':     round(median_rank, 1),
            'found_count':     len(all_ranks),
            'not_found_count': not_found_cnt,
            'not_found_pct':   round(not_found_cnt / n_valid * 100, 2) if n_valid else 0.0,
        },
        'per_strategy_hit_at_10': {k: round(v, 4) for k, v in per_strategy_hit_at_10.items()},
        'rank_distribution': dict(rank_dist),
        's1_baseline': {
            **{k: round(v, 4) for k, v in s1_hits.items()},
            'mrr':             round(s1_mrr, 4),
            'mean_rank':       round(s1_mean_rank, 2),
            'median_rank':     round(s1_median_rank, 1),
            'found_count':     len(s1_ranks),
            'not_found_count': s1_not_found,
            'not_found_pct':   round(s1_not_found / n_valid * 100, 2) if n_valid else 0.0,
            'rank_distribution': dict(s1_rank_dist),
        },
        'timing': {
            'total_eval_s':      round(total_eval_time, 2),
            'mean_retrieval_s':  round(float(np.mean(retrieval_times)), 3)   if retrieval_times else 0.0,
            'median_retrieval_s':round(float(np.median(retrieval_times)), 3) if retrieval_times else 0.0,
            'min_retrieval_s':   round(float(np.min(retrieval_times)), 3)    if retrieval_times else 0.0,
            'max_retrieval_s':   round(float(np.max(retrieval_times)), 3)    if retrieval_times else 0.0,
        },
        'per_query_results': results_per_query,
    }

    # ------------------------------------------------------------------
    # Step 8 – save reports
    # ------------------------------------------------------------------
    ts   = report['metadata']['timestamp']
    stem = f"eval_{graph_id}_{ts}" if graph_id else f"eval_{ts}"
    out  = Path(output_dir)

    json_path = out / f"{stem}.json"
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    txt_path = out / f"{stem}.txt"
    with open(txt_path, 'w', encoding='utf-8') as fh:
        fh.write(_format_report(report, top_k_eval, max_rank))

    # ------------------------------------------------------------------
    # Step 8b – save "pipeline top-1 not in S1 top-10" report
    # Lists every query where our hybrid pipeline's top-1 doc is NOT
    # among the top-10 results of plain semantic (S1) retrieval.
    # For each such case: query text, top-1 doc, its rank in S1.
    # ------------------------------------------------------------------
    pipeline_only_top1: List[Dict] = []
    for r in results_per_query:
        if r.get('error'):
            continue
        top1_full = r.get('top1_full_doc_id')
        if not top1_full:
            continue
        s1_ids = r.get('top50_s1_doc_ids', [])
        if top1_full in s1_ids[:10]:
            continue
        try:
            s1_rank = s1_ids.index(top1_full) + 1  # 1-based, within top-50
        except ValueError:
            s1_rank = None  # not found in top-50 of S1
        pipeline_only_top1.append({
            'query_id':        r['query_id'],
            'query_text':      r['query_text'],
            'top1_full_doc_id': top1_full,
            'top1_full_text':  r.get('top1_full_text', ''),
            's1_rank':         s1_rank,
        })

    pipeline_only_path = out / f"{stem}_pipeline_only_top1.txt"
    with open(pipeline_only_path, 'w', encoding='utf-8') as fh:
        fh.write("=" * 70 + "\n")
        fh.write("PIPELINE TOP-1 NOT IN S1 SEMANTIC TOP-10\n")
        fh.write(f"Queries where the hybrid pipeline's top-1 doc is absent from\n")
        fh.write(f"the top-10 of plain semantic (S1) retrieval.\n")
        fh.write(f"Total cases: {len(pipeline_only_top1)}\n")
        fh.write("=" * 70 + "\n\n")
        for i, entry in enumerate(pipeline_only_top1, 1):
            s1_rank_str = str(entry['s1_rank']) if entry['s1_rank'] is not None else f">50"
            fh.write(f"--- Case {i} / {len(pipeline_only_top1)}  (query_id={entry['query_id']}) ---\n")
            fh.write(f"QUERY      : {entry['query_text']}\n")
            fh.write(f"TOP-1 DOC  : [doc_id={entry['top1_full_doc_id']}]\n")
            doc_preview = entry['top1_full_text'][:400].replace('\n', ' ')
            fh.write(f"             {doc_preview}\n")
            fh.write(f"S1 RANK    : {s1_rank_str}\n")
            fh.write("\n")

    logger.info(f"  Pipeline-only top-1 : {pipeline_only_path}  ({len(pipeline_only_top1)} cases)")

    logger.info(f"\nEvaluation complete in {total_eval_time:.1f}s.")
    logger.info(f"  JSON : {json_path}")
    logger.info(f"  Text : {txt_path}")
    _log_summary(report, top_k_eval, logger)

    return report


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_STRATEGY_LABELS = {
    'S1_semantic':        'S1  Semantic',
    'S2_topic_dist':      'S2  Topic Distribution',
    'S3_thematic':        'S3  Thematic',
    'S4_entity':          'S4  Entity',
    'S5_claim_doc':       'S5  Claim-Document',
    'S6_claim_topic':     'S6  Claim-Topic',
    'S7_claim_keyphrase': 'S7  Claim-Keyphrase',
}


def _format_report(report: Dict, top_k_eval: List[int], max_rank: int) -> str:
    m  = report['metrics']
    c  = report['coverage']
    rd = report['rank_distribution']
    lines = []

    lines += [
        "=" * 70,
        "KNOWLEDGE GRAPH RETRIEVAL EVALUATION REPORT",
        "=" * 70,
        f"Graph ID       : {report['metadata'].get('graph_id', 'N/A')}",
        f"Timestamp      : {report['metadata']['timestamp']}",
        f"Max rank depth : {max_rank}",
        "",
        "CORPUS COVERAGE",
        "-" * 70,
        f"  Total relations           : {c['total_relations']}",
        f"  Pairs with doc in graph   : {c['valid_pairs_in_graph']}  ({c['coverage_pct']:.1f}%)",
        f"  Evaluated (no error)      : {c['evaluated']}",
        f"  Errors (skipped)          : {c['errors']}",
        "",
        "OVERALL METRICS  (document-level, softmax-pooling over chunks)",
        "-" * 70,
    ]

    for k in top_k_eval:
        key = f'hit_at_{k}'
        v   = m.get(key, 0.0)
        bar = '#' * int(v * 40)
        lines.append(f"  Hit@{k:<4}       :  {v:.4f}  ({v*100:5.1f}%)  {bar}")

    lines += [
        f"  MRR            :  {m['mrr']:.4f}",
        f"  Mean Rank      :  {m['mean_rank']:.1f}",
        f"  Median Rank    :  {m['median_rank']:.1f}",
        f"  Found          :  {m['found_count']} / {c['evaluated']}",
        f"  Not found      :  {m['not_found_count']}  ({m['not_found_pct']:.1f}%)",
        "",
        "PER-STRATEGY  Hit@10  (document-level, softmax-pooling)",
        "-" * 70,
    ]

    for sk, label in _STRATEGY_LABELS.items():
        v   = report['per_strategy_hit_at_10'].get(sk, 0.0)
        bar = '#' * int(v * 40)
        lines.append(f"  {label:<28}:  {v:.4f}  ({v*100:5.1f}%)  {bar}")

    lines += ["", "RANK DISTRIBUTION", "-" * 70]
    total = c['evaluated']
    _display_buckets = ['1', '2-5', '6-10', 'not_found'] if max_rank <= 10 else \
                       ['1', '2-5', '6-10', f'11-{max_rank}', 'not_found']
    for bucket in _display_buckets:
        count = rd.get(bucket, 0)
        pct   = count / total * 100 if total else 0.0
        lines.append(f"  Rank {bucket:<14} :  {count:4d}  ({pct:5.1f}%)")

    t = report.get('timing', {})
    lines += [
        "", "TIMING", "-" * 70,
        f"  Total evaluation time  :  {t.get('total_eval_s', 0):.1f}s",
        f"  Mean retrieval / query :  {t.get('mean_retrieval_s', 0):.3f}s",
        f"  Median retrieval       :  {t.get('median_retrieval_s', 0):.3f}s",
        f"  Min / Max retrieval    :  {t.get('min_retrieval_s', 0):.3f}s / {t.get('max_retrieval_s', 0):.3f}s",
    ]

    # ------------------------------------------------------------------
    # Comparison: S1 Semantic (classic RAG) vs Full Pipeline
    # ------------------------------------------------------------------
    s1 = report.get('s1_baseline', {})
    if s1:
        def _fmt_delta(full_val: float, s1_val: float, higher_is_better: bool = True) -> str:
            delta = full_val - s1_val
            if higher_is_better:
                sign = '+' if delta >= 0 else ''
                tag  = ' ▲' if delta > 0 else (' ▼' if delta < 0 else '  ')
            else:
                sign = '+' if delta <= 0 else ''
                tag  = ' ▲' if delta < 0 else (' ▼' if delta > 0 else '  ')
            return f"{sign}{delta:+.4f}{tag}"

        lines += ["", "=" * 70,
                  "S1 SEMANTIC (classic RAG)  vs  FULL PIPELINE",
                  "=" * 70,
                  f"  {'Metric':<20}  {'S1 Semantic':>12}  {'Full Pipeline':>13}  {'Delta (full−S1)':>18}",
                  "  " + "-" * 67]
        for k in top_k_eval:
            key  = f'hit_at_{k}'
            s1v  = s1.get(key, 0.0)
            fullv = m.get(key, 0.0)
            bar_s1   = '#' * int(s1v  * 20)
            bar_full = '#' * int(fullv * 20)
            lines.append(
                f"  Hit@{k:<4}             {s1v:>12.4f}  {fullv:>13.4f}  "
                f"{_fmt_delta(fullv, s1v):>18}"
            )
        lines.append(
            f"  {'MRR':<20}  {s1.get('mrr', 0):>12.4f}  {m['mrr']:>13.4f}  "
            f"{_fmt_delta(m['mrr'], s1.get('mrr', 0)):>18}"
        )
        lines.append(
            f"  {'Mean Rank':<20}  {s1.get('mean_rank', 0):>12.1f}  {m['mean_rank']:>13.1f}  "
            f"{'(lower is better)':>18}"
        )
        lines.append(
            f"  {'Median Rank':<20}  {s1.get('median_rank', 0):>12.1f}  {m['median_rank']:>13.1f}  "
            f"{'':>18}"
        )
        lines.append(
            f"  {'Found':<20}  {s1.get('found_count', 0):>12}  {m['found_count']:>13}  "
            f"{'out of ' + str(c['evaluated']):>18}"
        )
        lines.append(
            f"  {'Not found':<20}  {s1.get('not_found_count', 0):>12}  {m['not_found_count']:>13}  "
            f"{'':>18}"
        )

        lines += ["", "  RANK DISTRIBUTION COMPARISON",
                  f"  {'Bucket':<14}  {'S1 Semantic':>12}  {'Full Pipeline':>13}"]
        _cmp_buckets = ['1', '2-5', '6-10', 'not_found'] if max_rank <= 10 else \
                       ['1', '2-5', '6-10', f'11-{max_rank}', 'not_found']
        for bucket in _cmp_buckets:
            s1c   = s1.get('rank_distribution', {}).get(bucket, 0)
            fullc = rd.get(bucket, 0)
            total_ev = c['evaluated']
            lines.append(
                f"  Rank {bucket:<9}  {s1c:>12}  {fullc:>13}"
            )

    # ------------------------------------------------------------------
    # PER-QUERY DETAILS — two sections:
    #   A) Successes: at least one pipeline returned GT at rank 1
    #      → detailed view with top-1 doc texts for both pipelines
    #   B) Misses:    neither pipeline returned GT at rank 1
    #      → brief header only
    # ------------------------------------------------------------------
    def _txt(s: str, n: int = 250) -> str:
        """One-line preview of a text."""
        s = s.replace('\n', ' ').replace('\r', '').strip()
        return (s[:n] + '...') if len(s) > n else s

    all_queries = report.get('per_query_results', [])
    successes = [r for r in all_queries
                 if r.get('rank') == 1 and r.get('per_strategy_ranks', {}).get('S1_semantic') == 1]
    misses    = [r for r in all_queries
                 if r not in successes]

    lines += ["", "=" * 70,
              f"PER-QUERY DETAILS",
              "=" * 70]

    # ---- A) Successes ----
    lines += ["",
              f"-- FOUND AT RANK 1  (full=1 AND S1=1) : {len(successes)} queries --"]
    for i, r in enumerate(successes, start=1):
        qid     = r.get('query_id', '?')
        gt_doc  = r.get('target_doc_id', '?')
        rank    = r.get('rank')
        s1_rank = r.get('per_strategy_ranks', {}).get('S1_semantic')
        elapsed = r.get('retrieval_time_s', 0.0)
        q_text  = r.get('query_text', '')
        err     = r.get('error')

        rank_str = str(rank)    if rank    is not None else 'NOT FOUND'
        s1_str   = str(s1_rank) if s1_rank is not None else 'NOT FOUND'

        lines.append(f"\n[S{i:03d}] Query {qid}  |  full: {rank_str}  |  S1: {s1_str}  |  {elapsed:.2f}s")
        lines.append(f"  Query  : {_txt(q_text, 120)}")
        lines.append(f"  GT doc : {gt_doc}  |  {_txt(r.get('gt_doc_text', ''), 200)}")
        if err:
            lines.append(f"  ERROR  : {err}")
            continue

        # Top-1 full pipeline
        t1f_id  = r.get('top1_full_doc_id', '')
        t1f_txt = r.get('top1_full_text', '')
        marker  = ' [=GT]' if t1f_id == gt_doc else ''
        lines.append(f"  TOP-1 FULL  [doc {t1f_id}]{marker} : {_txt(t1f_txt, 250)}")

        # Top-1 S1 (RAG)
        t1s_id  = r.get('top1_s1_doc_id', '')
        t1s_txt = r.get('top1_s1_text', '')
        marker  = ' [=GT]' if t1s_id == gt_doc else ''
        lines.append(f"  TOP-1 S1    [doc {t1s_id}]{marker} : {_txt(t1s_txt, 250)}")

        # Top-50 lists
        full50 = r.get('top50_full_doc_ids', [])
        s1_50  = r.get('top50_s1_doc_ids', [])
        def _label50(ids, gt):
            return ' '.join(f"[{d}]" if d == gt else d for d in ids)
        if full50:
            lines.append(f"  Top50 FULL : {_label50(full50, gt_doc)}")
        if s1_50:
            lines.append(f"  Top50 S1   : {_label50(s1_50, gt_doc)}")

    # ---- B) Misses ----
    lines += ["", "",
              f"-- NOT BOTH AT RANK 1 (full!=1 or S1!=1) : {len(misses)} queries --"]
    for i, r in enumerate(misses, start=1):
        qid     = r.get('query_id', '?')
        gt_doc  = r.get('target_doc_id', '?')
        rank    = r.get('rank')
        s1_rank = r.get('per_strategy_ranks', {}).get('S1_semantic')
        elapsed = r.get('retrieval_time_s', 0.0)
        q_text  = r.get('query_text', '')
        err     = r.get('error')

        rank_str = str(rank)    if rank    is not None else 'NOT FOUND'
        s1_str   = str(s1_rank) if s1_rank is not None else 'NOT FOUND'

        lines.append(f"\n[M{i:03d}] Query {qid}  |  full: {rank_str}  |  S1: {s1_str}  |  {elapsed:.2f}s")
        lines.append(f"  Query  : {_txt(q_text, 120)}")
        lines.append(f"  GT doc : {gt_doc}  |  {_txt(r.get('gt_doc_text', ''), 150)}")
        if err:
            lines.append(f"  ERROR  : {err}")
            continue

        # Top-50 lists for misses (useful for diagnosing why it failed)
        full50 = r.get('top50_full_doc_ids', [])
        s1_50  = r.get('top50_s1_doc_ids', [])
        def _label50(ids, gt):
            return ' '.join(f"[{d}]" if d == gt else d for d in ids)
        if full50:
            lines.append(f"  Top50 FULL : {_label50(full50, gt_doc)}")
        if s1_50:
            lines.append(f"  Top50 S1   : {_label50(s1_50, gt_doc)}")

    lines += ["", "=" * 70]
    return "\n".join(lines)


def _log_summary(report: Dict, top_k_eval: List[int], logger: logging.Logger):
    m  = report['metrics']
    c  = report['coverage']
    s1 = report.get('s1_baseline', {})
    logger.info("\n" + "=" * 65)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 65)
    logger.info(f"  Evaluated : {c['evaluated']}  |  Coverage: {c['coverage_pct']:.1f}%")
    logger.info(f"  {'Metric':<12}  {'S1 (RAG)':>10}  {'Full Pipeline':>14}  {'Delta':>8}")
    logger.info(f"  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*8}")
    for k in top_k_eval:
        key  = f'hit_at_{k}'
        full = m.get(key, 0.0)
        base = s1.get(key, 0.0) if s1 else full
        delta = full - base
        sign  = '+' if delta >= 0 else ''
        logger.info(f"  Hit@{k:<8}  {base:>10.4f}  {full:>14.4f}  {sign}{delta:>+.4f}")
    full_mrr = m['mrr']
    base_mrr = s1.get('mrr', full_mrr) if s1 else full_mrr
    delta_mrr = full_mrr - base_mrr
    logger.info(f"  {'MRR':<12}  {base_mrr:>10.4f}  {full_mrr:>14.4f}  {delta_mrr:>+8.4f}")
    logger.info(f"  {'Mean Rank':<12}  {s1.get('mean_rank', m['mean_rank']) if s1 else m['mean_rank']:>10.1f}  {m['mean_rank']:>14.1f}")
    logger.info(f"  {'Median Rank':<12}  {s1.get('median_rank', m['median_rank']) if s1 else m['median_rank']:>10.1f}  {m['median_rank']:>14.1f}")
    t = report.get('timing', {})
    logger.info(f"  Total eval time        : {t.get('total_eval_s', 0):.1f}s")
    logger.info(f"  Mean retrieval / query : {t.get('mean_retrieval_s', 0):.3f}s")
    logger.info(f"  Min / Max retrieval    : {t.get('min_retrieval_s', 0):.3f}s / {t.get('max_retrieval_s', 0):.3f}s")
    logger.info("=" * 65)
