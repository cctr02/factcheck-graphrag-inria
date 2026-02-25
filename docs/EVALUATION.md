# Evaluation Documentation

This document describes the benchmark evaluation system that measures how well the knowledge graph retrieval pipeline surfaces the correct fact-checking article for a given social-media post.

## Table of Contents

1. [Overview](#overview)
2. [Input Data](#input-data)
3. [Filtering: Corpus Coverage](#filtering-corpus-coverage)
4. [Evaluation Pipeline](#evaluation-pipeline)
5. [Metrics](#metrics)
6. [Output Files](#output-files)
7. [CLI Usage](#cli-usage)

---

## Overview

The evaluation answers a single core question:

> **Given a social-media post (query), does our retrieval system return the correct fact-checking article in the top results?**

The ground truth is a set of pre-aligned `(post, fact-check article)` pairs. For each pair whose article is present in the graph, the post is run through the full hybrid retrieval pipeline, and the rank of the correct article in the returned document list is recorded.

---

## Input Data

Three CSV files are required (all with string column names `"0"` and `"1"`):

| File | Column `"0"` | Column `"1"` | Description |
|------|-------------|-------------|-------------|
| `queries.csv` | `query_id` | `query_text` | Social-media posts |
| `documents.csv` | `doc_id` | `doc_text` | Fact-checking articles |
| `relations.csv` | `query_id` | `doc_id` | Ground-truth alignments |

**Default paths:** `data/queries.csv`, `data/documents.csv`, `data/relations.csv`

**ID normalisation:** IDs may be stored as integers or floats by pandas (`"422"` or `"422.0"`). All IDs are normalised with:

$$\text{norm\_id}(v) = \text{str}\!\bigl(\text{int}(\text{float}(v))\bigr)$$

---

## Filtering: Corpus Coverage

Not all ground-truth pairs can be evaluated — the graph only contains a **subset of documents** (the first `max_documents` articles processed during build).

```
Total relations in relations.csv
        │
        ▼
Filter: is doc_id present in graph?  ──No──▶  out-of-corpus (skipped, reported)
        │ Yes
        ▼
Filter: does query_id have a text?   ──No──▶  skipped
        │ Yes
        ▼
    valid_pairs  →  evaluated
```

The **coverage** metric reports what fraction of ground-truth pairs can actually be evaluated:

$$\text{coverage} = \frac{|\text{valid pairs}|}{|\text{total relations}|} \times 100\%$$

A higher coverage means the graph was built from more documents.

---

## Evaluation Pipeline

For each valid `(query_text, target_doc_id)` pair:

### Step 1 — Temporary depth override

`retriever.top_docs` is set to `max_rank` (default 500) for the duration of the evaluation, so that documents ranked far down the list can still be found. It is restored to its original value after evaluation.

### Step 2 — Retrieval

The full hybrid retrieval pipeline is called:

```python
ret = retriever.retrieve(query_text)
```

This runs all 7 strategies (S1–S7) and aggregates scores to document level via **max-pooling**:

$$\text{doc\_score}_X(d) = \max_{\text{chunk} \in d}\; S_X(\text{chunk})$$

Documents are then ranked by their final hybrid score:

$$\text{score}(d) = \alpha\,S_1 + \beta\,S_2 + \gamma\,S_3 + \delta\,S_4 + \varepsilon\,S_5 + \zeta\,S_6 + \eta\,S_7$$

### Step 3 — Rank lookup

The rank of `target_doc_id` is searched in two lists:

| List | Description |
|------|-------------|
| `doc_results` | Overall ranking by hybrid score |
| `per_strategy_doc_results[Sx]` | Per-strategy ranking for each of the 7 strategies |

```
doc_results = [doc_0, doc_1, doc_2, ..., doc_499]
                 rank=1   rank=2  rank=3

If target_doc_id == doc_7.doc_id  →  rank = 8
If target_doc_id not found        →  rank = None  (not found within max_rank)
```

---

## Metrics

### Hit@K

Fraction of queries for which the target document appears within the top $K$ results:

$$\text{Hit@}K = \frac{|\{q \in Q : \text{rank}(q) \leq K\}|}{|Q|}$$

Computed for $K \in \{1, 5, 10, 50\}$.

| Metric | What it means |
|--------|--------------|
| Hit@1 | Correct article is the very first result |
| Hit@5 | Correct article is in the top 5 |
| Hit@10 | Correct article is in the top 10 |
| Hit@50 | Correct article is in the top 50 |

### MRR — Mean Reciprocal Rank

Rewards systems that rank the correct document **higher**. A document at rank 1 contributes 1.0, at rank 2 contributes 0.5, at rank 10 contributes 0.1. Not-found queries contribute 0.

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(q)}$$

where $\text{rank}(q) = +\infty$ (contribution = 0) when the document is not found.

MRR is more sensitive than Hit@K to whether the correct article is at rank 1 vs rank 10.

### Mean Rank / Median Rank

Average and median position of the correct document, **computed only over found queries** (queries where rank ≠ None):

$$\text{Mean Rank} = \frac{1}{|Q_\text{found}|} \sum_{q \in Q_\text{found}} \text{rank}(q)$$

A low mean rank indicates the system consistently surfaces the correct article near the top.

### Per-Strategy Hit@10

The same Hit@10 metric applied **independently to each strategy's document ranking**. This isolates each strategy's individual contribution, without the weighting of the hybrid score.

$$\text{Hit@10}_{S_x} = \frac{|\{q : \text{rank}_{S_x}(q) \leq 10\}|}{|Q|}$$

Useful to identify which strategy is most effective for this corpus.

### Rank Distribution

The found queries are grouped into rank buckets to visualise the shape of the distribution:

| Bucket | Interpretation |
|--------|----------------|
| 1 | Perfect retrieval |
| 2–5 | Very good |
| 6–10 | Good (within a standard top-10) |
| 11–20 | Marginal |
| 21–50 | Weak |
| 51–`max_rank` | Poor |
| not_found | Not found within `max_rank` documents |

---

## Output Files

Results are saved to `outputs/evaluation/` with the naming pattern:

```
eval_{graph_id}_{timestamp}.json   ← full machine-readable report
eval_{graph_id}_{timestamp}.txt    ← human-readable report
```

### JSON structure

```json
{
  "metadata": {
    "graph_id": "20260206_182357",
    "timestamp": "20260225_143012",
    "max_rank_searched": 500
  },
  "coverage": {
    "total_relations": 2000,
    "valid_pairs_in_graph": 850,
    "coverage_pct": 42.5,
    "evaluated": 850,
    "errors": 0
  },
  "metrics": {
    "hit_at_1":  0.1200,
    "hit_at_5":  0.3100,
    "hit_at_10": 0.4400,
    "hit_at_50": 0.6700,
    "mrr":       0.2150,
    "mean_rank": 18.3,
    "median_rank": 9.0,
    "found_count": 612,
    "not_found_count": 238,
    "not_found_pct": 28.0
  },
  "per_strategy_hit_at_10": {
    "S1_semantic":        0.3800,
    "S2_topic_dist":      0.1200,
    "S3_thematic":        0.2100,
    "S4_entity":          0.1500,
    "S5_claim_doc":       0.4100,
    "S6_claim_topic":     0.0800,
    "S7_claim_keyphrase": 0.0500
  },
  "rank_distribution": {
    "1": 102, "2-5": 161, "6-10": 111,
    "11-20": 98, "21-50": 140, "51-500": 0,
    "not_found": 238
  },
  "per_query_results": [
    { "query_id": "42", "target_doc_id": "317", "rank": 3,
      "per_strategy_ranks": { "S1_semantic": 2, "S2_topic_dist": 45, ... } },
    ...
  ]
}
```

### Text report example

```
======================================================================
KNOWLEDGE GRAPH RETRIEVAL EVALUATION REPORT
======================================================================
Graph ID       : 20260206_182357
Timestamp      : 20260225_143012
Max rank depth : 500

CORPUS COVERAGE
----------------------------------------------------------------------
  Total relations           : 2000
  Pairs with doc in graph   : 850  (42.5%)
  Evaluated (no error)      : 850
  Errors (skipped)          : 0

OVERALL METRICS  (document-level, max-pooling over chunks)
----------------------------------------------------------------------
  Hit@1          :  0.1200  ( 12.0%)  ####
  Hit@5          :  0.3100  ( 31.0%)  ############
  Hit@10         :  0.4400  ( 44.0%)  #################
  Hit@50         :  0.6700  ( 67.0%)  ##########################
  MRR            :  0.2150
  Mean Rank      :  18.3
  Median Rank    :  9.0
  Found          :  612 / 850
  Not found      :  238  (28.0%)

PER-STRATEGY  Hit@10  (document-level, max-pooling)
----------------------------------------------------------------------
  S1  Semantic              :  0.3800  ( 38.0%)  ###############
  S2  Topic Distribution    :  0.1200  ( 12.0%)  ####
  S3  Thematic              :  0.2100  ( 21.0%)  ########
  S4  Entity                :  0.1500  ( 15.0%)  ######
  S5  Claim-Document        :  0.4100  ( 41.0%)  ################
  S6  Claim-Topic           :  0.0800  (  8.0%)  ###
  S7  Claim-Keyphrase       :  0.0500  (  5.0%)  ##

RANK DISTRIBUTION
----------------------------------------------------------------------
  Rank 1              :   102  ( 12.0%)
  Rank 2-5            :   161  ( 18.9%)
  Rank 6-10           :   111  ( 13.1%)
  Rank 11-20          :    98  ( 11.5%)
  Rank 21-50          :   140  ( 16.5%)
  Rank 51-500         :     0  (  0.0%)
  Rank not_found      :   238  ( 28.0%)
======================================================================
```

---

## CLI Usage

```bash
# Standard run (all defaults)
python cli.py evaluate --graph-id 20260206_182357

# Quick test on 50 queries
python cli.py evaluate --graph-id 20260206_182357 --max-queries 50

# Deeper search (up to rank 1000)
python cli.py evaluate --graph-id 20260206_182357 --max-rank 1000

# Custom CSV paths
python cli.py evaluate --graph-id 20260206_182357 \
  --queries path/to/queries.csv \
  --documents path/to/documents.csv \
  --relations path/to/relations.csv

# All options
python cli.py evaluate --help
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--graph-id` | *(required)* | Timestamp ID of the graph (e.g. `20260206_182357`) |
| `--queries` | `data/queries.csv` | Path to queries CSV |
| `--documents` | `data/documents.csv` | Path to documents CSV |
| `--relations` | `data/relations.csv` | Path to relations CSV |
| `--max-queries` | all | Limit number of evaluated queries (useful for quick tests) |
| `--max-rank` | 500 | How far down the ranked list to search for the target document |
| `--log-level` | INFO | Verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
