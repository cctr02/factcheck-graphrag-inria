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

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


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
        if data.get('type') == 'doc' and data.get('doc_id') is not None
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
    max_rank: int = 500,
    output_dir: str = "outputs/evaluation",
    graph_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Run the full retrieval evaluation benchmark.

    For each (query, ground-truth doc) pair whose doc is present in the graph,
    retrieval is run and the rank of the target document is recorded.

    Document-level scoring uses max-pooling (already implemented in
    HybridRetriever._aggregate_to_documents): the doc score equals the maximum
    chunk score for any chunk of that document.  This is length-unbiased.

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
        top_k_eval = [1, 5, 10, 50]
    if logger is None:
        logger = logging.getLogger(__name__)

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
    # Step 4 – override top_docs for evaluation depth
    # ------------------------------------------------------------------
    original_top_docs = retriever.top_docs
    retriever.top_docs = max_rank
    logger.info(f"retriever.top_docs temporarily set to {max_rank}")

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

    logger.info(f"\nRunning evaluation on {len(valid_pairs)} queries ...")
    results_per_query: List[Dict] = []

    for i, (query_id, target_doc_id) in enumerate(
        tqdm(valid_pairs, desc="Evaluating", unit="query"), start=1
    ):
        query_text = query_texts[query_id]
        try:
            ret = retriever.retrieve(query_text)
        except Exception as exc:
            logger.warning(f"  Query {query_id} failed: {exc}")
            results_per_query.append({
                'query_id': query_id,
                'target_doc_id': target_doc_id,
                'rank': None,
                'per_strategy_ranks': {k: None for k in strategy_keys},
                'error': str(exc),
            })
            continue

        first = ret[0] if ret else {}

        # --- overall rank (final hybrid score) ---
        doc_results = first.get('doc_results', [])
        rank = next(
            (j for j, d in enumerate(doc_results, 1) if _norm_id(d['doc_id']) == target_doc_id),
            None
        )

        # --- per-strategy rank ---
        per_strat_docs = first.get('per_strategy_doc_results', {})
        per_strategy_ranks: Dict[str, Optional[int]] = {}
        for sk in strategy_keys:
            strat_list = per_strat_docs.get(sk, [])
            per_strategy_ranks[sk] = next(
                (j for j, d in enumerate(strat_list, 1) if _norm_id(d['doc_id']) == target_doc_id),
                None
            )

        results_per_query.append({
            'query_id': query_id,
            'target_doc_id': target_doc_id,
            'rank': rank,
            'per_strategy_ranks': per_strategy_ranks,
        })

        if i % 20 == 0 or i == len(valid_pairs):
            found_so_far = sum(1 for r in results_per_query if r.get('rank') is not None)
            logger.info(f"  Progress: {i}/{len(valid_pairs)} | Found: {found_so_far}/{i}")

    # Restore original value
    retriever.top_docs = original_top_docs

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

    # Rank distribution
    def _bucket(rk: int) -> str:
        if rk == 1:         return '1'
        elif rk <= 5:       return '2-5'
        elif rk <= 10:      return '6-10'
        elif rk <= 20:      return '11-20'
        elif rk <= 50:      return '21-50'
        else:               return f'51-{max_rank}'

    rank_dist: Dict[str, int] = defaultdict(int)
    rank_dist['not_found'] = not_found_cnt
    for rk in all_ranks:
        rank_dist[_bucket(rk)] += 1

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

    logger.info(f"\nEvaluation complete.")
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
        "OVERALL METRICS  (document-level, max-pooling over chunks)",
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
        "PER-STRATEGY  Hit@10  (document-level, max-pooling)",
        "-" * 70,
    ]

    for sk, label in _STRATEGY_LABELS.items():
        v   = report['per_strategy_hit_at_10'].get(sk, 0.0)
        bar = '#' * int(v * 40)
        lines.append(f"  {label:<28}:  {v:.4f}  ({v*100:5.1f}%)  {bar}")

    lines += ["", "RANK DISTRIBUTION", "-" * 70]
    total = c['evaluated']
    for bucket in ['1', '2-5', '6-10', '11-20', '21-50', f'51-{max_rank}', 'not_found']:
        count = rd.get(bucket, 0)
        pct   = count / total * 100 if total else 0.0
        lines.append(f"  Rank {bucket:<14} :  {count:4d}  ({pct:5.1f}%)")

    lines += ["", "=" * 70]
    return "\n".join(lines)


def _log_summary(report: Dict, top_k_eval: List[int], logger: logging.Logger):
    m = report['metrics']
    c = report['coverage']
    logger.info("\n" + "=" * 55)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Evaluated : {c['evaluated']}  |  Coverage: {c['coverage_pct']:.1f}%")
    for k in top_k_eval:
        v = m.get(f'hit_at_{k}', 0.0)
        logger.info(f"  Hit@{k:<4}  : {v:.4f}  ({v*100:.1f}%)")
    logger.info(f"  MRR       : {m['mrr']:.4f}")
    logger.info(f"  Mean Rank : {m['mean_rank']:.1f}  |  Median: {m['median_rank']:.1f}")
    logger.info("=" * 55)
