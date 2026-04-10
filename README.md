# Knowledge Graph Fact-Checking System

A knowledge graph-based document retrieval system for fact-checking, combining topic modeling (BERTopic), named entity recognition (spaCy), keyphrase extraction (KeyBERT), and a 7-strategy hybrid retriever.

---

## Table of Contents

1. [Repository Purpose](#1-repository-purpose)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Installation and Environment Setup](#4-installation-and-environment-setup)
5. [Configuration](#5-configuration)
6. [Initialization](#6-initialization)
7. [How to Use `cli.py`](#7-how-to-use-clipy)
8. [CLI Commands Reference](#8-cli-commands-reference)
9. [Inputs and Outputs](#9-inputs-and-outputs)
10. [Module Reference](#10-module-reference)
11. [Experimental: `build_evaluate.py`](#11-experimental-build_evaluatepy)
12. [Typical Usage Examples](#12-typical-usage-examples)
13. [Current Limitations and Known Issues](#13-current-limitations-and-known-issues)

---

## 1. Repository Purpose

This repository implements the retrieval component of a fact-checking pipeline. Given a textual claim, the system retrieves the most relevant fact-checking articles from a knowledge graph built over a corpus of fact-check documents.

The theoretical background (graph construction choices, retrieval strategies, evaluation methodology) is documented in a separate academic report. This README focuses exclusively on the **code, setup, and usage**.

---

## 2. High-Level Architecture

### Entry Point

**Almost all operations go through `cli.py`**, which is the single entry point for all user-facing functionality. The only exception is `build_evaluate.py` (see [Section 11](#11-experimental-build_evaluatepy)).

### Execution Flow

```
User
 └─► cli.py  (Click CLI - dispatches all commands)
      ├─► build       ─► fact_check_kg/build/build_graph.py
      │                    ├─► data_loader.py      (load & chunk documents)
      │                    ├─► graph_builder.py    (NER, BERTopic, KeyBERT, graph construction)
      │                    └─► checkpoint_manager.py (resumable builds)
      │
      ├─► retrieve    ─► fact_check_kg/retrieval/retrieval.py (HybridRetriever)
      │                    ├─► retrieval_semantic.py       (S1: embedding similarity)
      │                    ├─► retrieval_topic.py          (S2: topic distribution)
      │                    ├─► retrieval_thematic.py       (S3: theme graph BFS)
      │                    ├─► retrieval_entity.py         (S4: entity graph BFS)
      │                    ├─► retrieval_claims.py         (S5: claim-document)
      │                    ├─► retrieval_claim_topic.py    (S6: claim-topic)
      │                    └─► retrieval_claim_keyphrase.py (S7: claim-keyphrase)
      │
      ├─► evaluate    ─► fact_check_kg/evaluation/evaluate.py
      ├─► visualize   ─► fact_check_kg/analysis/visualize_graph.py
      ├─► stats       ─► fact_check_kg/analysis/graph_manager.py
      ├─► list-graphs ─► fact_check_kg/analysis/graph_manager.py
      ├─► relabel     ─► fact_check_kg/analysis/relabel_topics.py
      ├─► config      ─► fact_check_kg/core/config_loader.py
      └─► init        ─► (creates directories / PostgreSQL tables)
```

### Graph Structure

The knowledge graph is a **multi-level NetworkX graph** with the following node types:

| Node Type | ID Format | Description |
|-----------|-----------|-------------|
| `document` | `doc::{id}` | Source fact-check article |
| `chunk` | `chunk::{doc_id}::{idx}` | Sentence-aware text segment |
| `theme` | `theme::{topic_id}` | BERTopic-discovered topic |
| `entity` | `entity::{TYPE}::{text}` | Named entity (PERSON, ORG, GPE, LOC, DATE, EVENT, NORP) |
| `keyphrase` | `keyphrase::{text}` | KeyBERT-extracted phrase |
| `claim` | `claim::{id}` | Fact-checking claim |

### Edge Types

| Edge | Direction | Weight |
|------|-----------|--------|
| `CONTAINS` | doc → chunk | 1.0 |
| `HAS_TOPIC` | chunk → theme | BERTopic probability |
| `HAS_KEYPHRASE` | chunk → keyphrase | KeyBERT score |
| `MENTIONS` | chunk → entity | σ(TF-IDF / 3.0) |
| `CO_OCCURS` | entity ↔ entity | σ(PMI / 2.0) |
| `SEMANTIC_SIM` | entity/theme ↔ entity/theme | cosine similarity (≥ 0.8) |
| `CO_PRESENCE` | theme ↔ theme | count × 1/√(freq₁ × freq₂) |
| `TEMPORAL_DIST` | DATE ↔ DATE | exp(−0.15 × \|year₁ − year₂\|) |
| `REVIEWS` | claim → document | 1.0 |
| `SIMILAR_CLAIM` | claim ↔ claim | embedding cosine similarity |

### Retrieval

The `HybridRetriever` fuses **7 independent strategies** using a weighted sum:

```
score_final = α·S1 + β·S2 + γ·S3 + δ·S4 + ε·S5 + ζ·S6 + η·S7
```

| Strategy | Weight | Description |
|----------|--------|-------------|
| S1 – Semantic | α = 0.50 | FAISS embedding cosine similarity |
| S2 – Topic Distribution | β = 0.03 | Jensen-Shannon divergence on topic vectors |
| S3 – Thematic | γ = 0.02 | BFS through theme graph (CO_PRESENCE, SEMANTIC_SIM) |
| S4 – Entity | δ = 0.05 | BFS through entity graph (CO_OCCURS, SEMANTIC_SIM) |
| S5 – Claim-Document | ε = 0.15 | Semantically similar claims → their linked documents |
| S6 – Claim-Topic | ζ = 0.13 | Claim-based topic distribution traversal |
| S7 – Claim-KeyPhrase | η = 0.12 | Claim-based keyphrase matching |

All weights must sum to 1.0 (validated at startup). Weights are configured in `config/config_retrieval.yaml`.

---

## 3. Repository Structure

```
PRL/
├── cli.py                          # Main entry point for ALL operations
├── build_evaluate.py               # [EXPERIMENTAL] Benchmark dataset preparation
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # Pip-compatible dependency list
│
├── config/                         # YAML configuration files
│   ├── config.yaml                 # Common settings (paths, models, DB, logging)
│   ├── config_build.yaml           # Graph construction parameters
│   └── config_retrieval.yaml       # Retrieval weights, thresholds, visualization
│
├── fact_check_kg/                  # Main Python package
│   ├── __init__.py
│   ├── core/                       # Infrastructure (config, logging, embeddings, DB)
│   │   ├── config_loader.py        # Three-layer YAML config system
│   │   ├── logger.py               # Rotating file logger with ANSI colors
│   │   ├── embedding_cache.py      # FAISS vector index (save/load/search)
│   │   └── db_connector.py         # PostgreSQL graph storage
│   ├── build/                      # Graph construction
│   │   ├── build_graph.py          # Build orchestrator (called by cli.py build)
│   │   ├── graph_builder.py        # Core construction engine (BERTopic, NER, KeyBERT)
│   │   ├── data_loader.py          # CSV loading + sentence-aware chunking
│   │   ├── checkpoint_manager.py   # Resumable build checkpoints
│   │   ├── graph_nodes_chunk_docs.py   # Document/chunk/keyphrase node builders
│   │   ├── graph_nodes_entity.py       # Entity extraction and CO_OCCURS edges
│   │   ├── graph_nodes_theme.py        # Topic modeling and theme nodes
│   │   ├── graph_nodes_claims.py       # Claim nodes and REVIEWS edges
│   │   ├── llm_labeling.py             # Batched LLM topic labeling with auto-tuning
│   │   └── topic_visualization.py      # Topic keyword analysis utilities
│   ├── retrieval/                  # Hybrid retrieval system
│   │   ├── retrieval.py            # HybridRetriever (7-strategy orchestrator)
│   │   ├── retrieval_semantic.py       # Strategy 1: embedding similarity
│   │   ├── retrieval_topic.py          # Strategy 2: topic distribution (JS divergence)
│   │   ├── retrieval_thematic.py       # Strategy 3: theme graph BFS
│   │   ├── retrieval_entity.py         # Strategy 4: entity graph BFS
│   │   ├── retrieval_claims.py         # Strategy 5: claim-document
│   │   ├── retrieval_claim_topic.py    # Strategy 6: claim-topic traversal
│   │   ├── retrieval_claim_keyphrase.py # Strategy 7: claim-keyphrase
│   │   ├── retrieval_propagation.py    # Graph traversal and relevance propagation
│   │   └── retrieval_results.py        # Result formatting, ranking, and export
│   ├── analysis/                   # Graph analysis and visualization
│   │   ├── graph_manager.py        # List/load/manage graph versions
│   │   ├── visualize_graph.py      # Pyvis interactive HTML visualization
│   │   ├── inspect_topics.py       # Topic inspection utilities
│   │   └── relabel_topics.py       # Post-build LLM topic relabeling
│   └── evaluation/                 # Benchmark evaluation
│       └── evaluate.py             # Hit@K, MRR, per-strategy evaluation runner
│
├── utils/                          # Utility functions
│   ├── io_utils.py                 # CSV loading and data utilities
│   └── check_device_usage.py       # GPU/CPU monitoring
│
├── data/                           # Input data (not committed)
│   └── english_fact_checks.csv     # Source fact-check corpus
│
└── outputs/                        # Generated artifacts (not committed)
    ├── graphs/                     # Graph pickle files + metadata JSON
    ├── models/                     # BERTopic models + topic probability arrays
    ├── embeddings/                 # FAISS indexes
    ├── visualizations/             # Interactive HTML files
    ├── evaluation/                 # Evaluation reports
    ├── retrieval_results/          # Per-query retrieval outputs
    ├── checkpoints/                # Build resumption data
    └── logs/                       # Rotating log files
```

---

## 4. Installation and Environment Setup

### Prerequisites

- **Python 3.10+** (tested on 3.10/3.11)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (recommended)
- **Ollama** (required for LLM-based topic labeling during `build`)
- **PostgreSQL** (optional — pickle storage works without it)
- **CUDA-capable GPU** (optional but strongly recommended for large builds)

### Install Dependencies

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Download Required Models

```bash
# spaCy NER model (required)
uv run python -m spacy download en_core_web_sm

# SentenceTransformers embedding model (downloaded automatically on first use)
# Model: all-MiniLM-L6-v2
```

### Ollama Setup (Required for Graph Build)

Topic labels are generated via a local LLM served by Ollama. Without it, the build will fail at the labeling step unless `llm_labeling` is disabled in config.

```bash
# Install Ollama
# Linux/macOS:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: https://ollama.com/download/windows

# Pull the model
ollama pull llama3.1:latest

# Start the server (keep running in a separate terminal)
ollama serve
```

The Ollama endpoint and model name are configurable in `config/config.yaml` (`models.llm_model`, `models.llm_endpoint`).

### PostgreSQL Setup (Optional)

PostgreSQL is used as a secondary storage backend. The system falls back to pickle-only storage if the database is unreachable.

```bash
# Initialize the database tables
uv run python cli.py init --postgres
```

Configure the connection in `config/config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  database: knowledge_graph
  user: postgres
  password: your_password
```

---

## 5. Configuration

The configuration system uses **three YAML files** in `config/`, loaded with a deep-merge strategy. `config.yaml` provides common defaults; the build and retrieval configs extend it.

| File | Used by | Contents |
|------|---------|----------|
| `config/config.yaml` | All commands | Paths, model names, database, logging, performance |
| `config/config_build.yaml` | `build` command | Chunking, topic modeling, entity extraction, LLM labeling |
| `config/config_retrieval.yaml` | `retrieve`, `evaluate`, `visualize` | Strategy weights, thresholds, top-k, visualization |

### Key Build Parameters (`config_build.yaml`)

```yaml
graph_construction:
  max_documents: 10000       # Default document limit
  chunk_size: 500            # Characters per chunk
  chunk_overlap: 100         # Overlap between consecutive chunks
  min_topic_size: 10         # Minimum documents per BERTopic cluster
  nr_topics: null            # null = auto-determine number of topics
  co_occurrence_threshold: 2 # Minimum entity co-occurrences to create an edge
  semantic_similarity_threshold: 0.8  # Minimum cosine similarity for SEMANTIC_SIM edges
  topic_probability_threshold: 0.15   # Minimum BERTopic probability for HAS_TOPIC edges
```

### Key Retrieval Parameters (`config_retrieval.yaml`)

```yaml
retrieval:
  alpha: 0.50    # S1: semantic (embedding similarity)
  beta:  0.03    # S2: topic distribution
  gamma: 0.02    # S3: thematic graph BFS
  delta: 0.05    # S4: entity graph BFS
  epsilon: 0.15  # S5: claim-document
  zeta:  0.13    # S6: claim-topic traversal
  eta:   0.12    # S7: claim-keyphrase
  top_k: 10      # Number of chunks returned per query
  top_docs: 50   # Number of documents considered during retrieval
```

All seven weights must sum to exactly 1.0 (validated on load).

---

## 6. Initialization

Run once after cloning and installing dependencies:

```bash
# Create all output directories
uv run python cli.py init --dirs

# (Optional) Initialize PostgreSQL tables
uv run python cli.py init --postgres
```

---

## 7. How to Use `cli.py`

`cli.py` is the **single entry point** for all graph operations. Run it with:

```bash
uv run python cli.py <command> [options]
```

Get help at any level:

```bash
uv run python cli.py --help
uv run python cli.py build --help
uv run python cli.py retrieve --help
```

### Typical Workflow

```
1. init       — create output directories (once)
2. build      — build the knowledge graph from CSV data
3. list-graphs — find the graph ID of the newly built graph
4. retrieve   — query the graph with a claim
5. evaluate   — run benchmark evaluation (optional)
6. visualize  — generate an HTML visualization (optional)
```

---

## 8. CLI Commands Reference

### `build` — Build the Knowledge Graph

Reads documents from `data/english_fact_checks.csv`, chunks them, runs BERTopic, NER, KeyBERT, labels topics with an LLM, and saves all outputs.

```bash
uv run python cli.py build [OPTIONS]

Options:
  --max-docs INTEGER    Maximum documents to process (default: from config_build.yaml)
  --log-level TEXT      DEBUG | INFO | WARNING | ERROR  (default: INFO)
  --resume              Resume from last checkpoint (after an interruption)
```

**Examples:**

```bash
# Build with config default (e.g., 10,000 docs)
uv run python cli.py build

# Build with only 500 documents (useful for testing)
uv run python cli.py build --max-docs 500

# Resume an interrupted build
uv run python cli.py build --resume
```

**Outputs** (all named with the same timestamp `GRAPH_ID`):

| File | Location | Description |
|------|----------|-------------|
| `graph_{GRAPH_ID}.gpickle` | `outputs/graphs/` | NetworkX graph |
| `metadata_{GRAPH_ID}.json` | `outputs/graphs/` | Build config + statistics |
| `topic_model_{GRAPH_ID}.pkl` | `outputs/models/` | BERTopic model |
| `topic_probs_{GRAPH_ID}.npz` | `outputs/models/` | Chunk topic distributions |
| `embeddings_{GRAPH_ID}.faiss` | `outputs/embeddings/` | FAISS index |
| `embeddings_{GRAPH_ID}.pkl` | `outputs/embeddings/` | FAISS metadata |

The `GRAPH_ID` is a timestamp in `YYYYMMDD_HHMMSS` format (e.g., `20260112_143022`). Use it to reference this specific graph in all other commands.

---

### `list-graphs` — List Available Graphs

```bash
uv run python cli.py list-graphs
```

Scans `outputs/graphs/metadata_*.json` and displays all available graphs with their IDs, document counts, and node/edge statistics.

---

### `retrieve` — Retrieve Relevant Chunks for a Claim

```bash
uv run python cli.py retrieve "CLAIM TEXT" [OPTIONS]

Options:
  --graph-id TEXT       Graph ID to use (required unless only one graph exists)
  --top-k INTEGER       Number of chunks to return (default: from config)
  --top-n INTEGER       Number of results to display in detail (default: 5)
  --alpha FLOAT         Override S1 weight
  --beta FLOAT          Override S2 weight
  --gamma FLOAT         Override S3 weight
  --delta FLOAT         Override S4 weight
  --epsilon FLOAT       Override S5 weight
  --zeta FLOAT          Override S6 weight
  --eta FLOAT           Override S7 weight
  --name TEXT           Custom name for the output files (e.g., "test1")
  --save-results/--no-save-results   Save output files (default: save)
  --use-postgres        Load graph from PostgreSQL instead of pickle
```

**Examples:**

```bash
uv run python cli.py retrieve "Bill Gates created COVID-19." --graph-id 20260112_143022

uv run python cli.py retrieve "The moon landing was faked." \
    --graph-id 20260112_143022 --top-k 20 --top-n 10

# Override strategy weights (must still sum to 1.0)
uv run python cli.py retrieve "Vaccines cause autism." \
    --graph-id 20260112_143022 --alpha 0.8 --beta 0.0 --gamma 0.0 \
    --delta 0.0 --epsilon 0.2 --zeta 0.0 --eta 0.0
```

**Output files** (saved to `outputs/retrieval_results/`):

- `results_{GRAPH_ID}_1.json` / `results_{GRAPH_ID}_1.txt` — auto-incremented
- `results_{GRAPH_ID}_{name}.json` / `.txt` — when `--name` is specified

Each result file contains: per-chunk scores for all 7 strategies, document-level rankings, matched claims/entities/themes, and the full text of all retrieved chunks.

---

### `stats` — Display Graph Statistics

```bash
uv run python cli.py stats [--graph-id GRAPH_ID] [--detailed]
```

Prints node and edge counts broken down by type. Uses the latest graph if `--graph-id` is omitted.

---

### `visualize` — Generate HTML Visualization

```bash
uv run python cli.py visualize [OPTIONS]

Options:
  --mode TEXT       full | node | stats  (default: stats)
  --node TEXT       Node ID for "node" mode (e.g., "entity::PERSON::trump")
  --graph-id TEXT   Graph ID to visualize
  --max-nodes INT   Maximum nodes to render in "full" mode
  --output TEXT     Output HTML file path (auto-generated if omitted)
  --use-postgres    Load from PostgreSQL
```

**Modes:**

- `stats` — displays pre-computed statistics to the console (fast, no file saved)
- `full` — renders the entire graph as an interactive HTML file (slow for large graphs)
- `node` — renders a subgraph centered on a specific node

**Examples:**

```bash
# Show statistics
uv run python cli.py visualize --mode stats --graph-id 20260112_143022

# Full graph visualization
uv run python cli.py visualize --mode full --graph-id 20260112_143022

# Subgraph centered on a named entity
uv run python cli.py visualize --mode node \
    --node "entity::PERSON::trump" --graph-id 20260112_143022

# Subgraph centered on a chunk
uv run python cli.py visualize --mode node \
    --node "chunk::0::5" --graph-id 20260112_143022
```

Output files are saved to `outputs/visualizations/`.

---

### `evaluate` — Run Benchmark Evaluation

```bash
uv run python cli.py evaluate --graph-id GRAPH_ID [OPTIONS]

Options:
  --queries PATH        Queries CSV (columns: query_id, query_text)
  --documents PATH      Documents CSV (columns: doc_id, doc_text)
  --relations PATH      Relations CSV (columns: query_id, doc_id)
  --max-queries INT     Limit number of evaluated queries
  --max-rank INT        Maximum rank to search for the target document (default: 10)
  --num-workers INT     Parallel evaluation threads (default: 1)
  --skip-llm            Use raw query text, skip Ollama reformatting
  --query-cache PATH    CSV cache for formatted queries (avoids re-running LLM)
  --ollama-model TEXT   Ollama model for query formatting (default: llama3.1:latest)
```

**Metrics reported:** Hit@1, Hit@5, Hit@10, Hit@50, MRR, mean rank, median rank, per-strategy Hit@10, rank distribution histogram.

**Example:**

```bash
uv run python cli.py evaluate \
    --graph-id 20260112_143022 \
    --queries data/eval_queries_top10000.csv \
    --relations data/eval_relations_top10000.csv \
    --max-queries 500 \
    --skip-llm
```

Results are saved as JSON and text reports in `outputs/evaluation/`.

---

### `relabel` — Relabel Topics with LLM

Re-runs LLM-based topic labeling on an existing graph without rebuilding it.

```bash
uv run python cli.py relabel [--llm-model MODEL]
```

---

### `config` — Inspect Configuration

```bash
uv run python cli.py config --show      # Print current merged configuration as YAML
uv run python cli.py config --validate  # Validate config files (check for errors)
```

---

### `init` — Initialize the System

```bash
uv run python cli.py init --dirs      # Create all required output directories
uv run python cli.py init --postgres  # Initialize PostgreSQL database tables
```

---

## 9. Inputs and Outputs

### Required Input Files

| File | Description |
|------|-------------|
| `data/english_fact_checks.csv` | Fact-check article corpus. Must have columns: `id`, `title`, `body`, `language`, `review_date`. |
| `data/claim_reviews.csv` (optional) | Claim data. Used to build claim nodes. Columns: `claim_id`, `claim`, `fact_check_id`. |

### For Evaluation (produced by `build_evaluate.py`)

| File | Description |
|------|-------------|
| `data/eval_queries_top{N}.csv` | Formatted claim queries. Columns: `query_id`, `claim_text`. |
| `data/eval_relations_top{N}.csv` | Ground-truth links. Columns: `query_id`, `ef_doc_id`. |

### Generated Outputs

| Directory | Contents |
|-----------|----------|
| `outputs/graphs/` | `graph_{ID}.gpickle`, `metadata_{ID}.json` |
| `outputs/models/` | `topic_model_{ID}.pkl`, `topic_probs_{ID}.npz` |
| `outputs/embeddings/` | `embeddings_{ID}.faiss`, `embeddings_{ID}.pkl` |
| `outputs/retrieval_results/` | `results_{ID}_{N}.json`, `results_{ID}_{N}.txt` |
| `outputs/evaluation/` | Evaluation reports (JSON + text) |
| `outputs/visualizations/` | Interactive HTML files |
| `outputs/checkpoints/` | Intermediate build state (for `--resume`) |
| `outputs/logs/` | Rotating log files (10 MB max, 5 backups) |

---

## 10. Module Reference

### `cli.py`

Central command dispatcher. All user-facing operations start here. Handles argument parsing (via Click), model loading, graph loading, and delegates to the relevant submodules. The `build` and `visualize` commands launch subprocesses; the remaining commands run Python functions directly.

---

### `fact_check_kg/core/config_loader.py`

Three-layer YAML configuration system.

- Loads `config.yaml` (base), deep-merges with `config_build.yaml` or `config_retrieval.yaml`
- Parses each section into a typed `dataclass` (e.g., `PathsConfig`, `RetrievalConfig`, `GraphConstructionConfig`)
- Validates constraints (e.g., retrieval weights sum to 1.0, batch size bounds)
- Provides singleton accessors: `get_build_config()`, `get_retrieval_config()`, `get_config()`

---

### `fact_check_kg/core/logger.py`

Centralized logging system.

- `get_logger(name)` — returns a logger with rotating file handlers and optional ANSI-colored console output
- Log files are written to `outputs/logs/`, rotate at 10 MB with 5 backups
- Per-module log levels configurable in `config.yaml`

---

### `fact_check_kg/core/embedding_cache.py`

FAISS-based vector index for fast chunk similarity search.

- Stores normalized embeddings in an inner-product FAISS index (equivalent to cosine similarity)
- GPU acceleration if `faiss-gpu` is available, otherwise falls back to CPU
- `build_index(chunk_ids, embeddings)` — builds the index from scratch
- `save(output_dir, graph_id)` / `load(output_dir, graph_id)` — serialize to/from disk
- `search(query_embedding, k)` — returns top-k most similar chunk IDs and scores

---

### `fact_check_kg/core/db_connector.py`

PostgreSQL storage backend for graphs.

- Stores graphs as edge tables (`source`, `predicate`, `target`, `weight`, `attributes`)
- Stores node metadata in a separate table with JSONB attributes
- Each graph is stored in its own timestamped set of tables
- `save_graph(graph, graph_id)` — batch-inserts nodes and edges
- `load_graph(graph_id)` — reconstructs a NetworkX graph from the database
- Falls back gracefully if the database is unreachable

---

### `fact_check_kg/build/build_graph.py`

Build orchestrator. Called as a subprocess by `cli.py build`.

1. Parses CLI arguments (`--max-docs`, `--log-level`, `--resume`)
2. Loads and chunks documents via `data_loader.py`
3. Loads and filters claims
4. Initializes and runs `KnowledgeGraphBuilder`
5. Saves all outputs with a consistent timestamp as the graph ID
6. Attempts to save to PostgreSQL (optional); continues with pickle-only on failure

---

### `fact_check_kg/build/graph_builder.py`

Core graph construction engine.

**Class `KnowledgeGraphBuilder`:**

- Initializes SentenceTransformer (embeddings), spaCy (NER), KeyBERT (keyphrases), and LLM labeler
- `build_graph(chunks, ...)` — orchestrates the full multi-level graph construction:
  1. Creates document and chunk nodes
  2. Trains BERTopic on chunk texts, creates theme nodes and `HAS_TOPIC` edges
  3. Labels topics using batched LLM calls (`llm_labeling.py`)
  4. Extracts named entities, creates entity nodes and `MENTIONS` edges (TF-IDF weighted)
  5. Builds entity-entity relationships: `CO_OCCURS` (PMI), `SEMANTIC_SIM` (cosine), `TEMPORAL_DIST`
  6. Extracts keyphrases and creates `HAS_KEYPHRASE` edges
  7. Creates theme-theme `CO_PRESENCE` and `SEMANTIC_SIM` edges
  8. Creates claim nodes and `REVIEWS` edges; builds `SIMILAR_CLAIM` edges between claims

---

### `fact_check_kg/build/data_loader.py`

Loads documents from CSV and splits them into sentence-aware chunks.

- `load_and_chunk_documents(csv_path, chunk_size, chunk_overlap, max_docs)` — main loader
  - Reads `english_fact_checks.csv`, filters by language, auto-detects title/body columns
  - Splits text into chunks of `chunk_size` characters without cutting mid-sentence
  - Returns a list of chunk dicts (each with `doc_id`, `chunk_idx`, `text`) and document metadata
- `load_claims(csv_path)` — loads `claim_reviews.csv` and returns claims + claim-to-doc mapping

---

### `fact_check_kg/build/checkpoint_manager.py`

Enables resumable builds.

- Saves intermediate results (graph, embeddings, model state) at defined checkpoints
- `get_last_completed()` — finds the last successfully saved checkpoint
- `validate_config(config, max_docs)` — verifies the config has not changed since the checkpoint
- Used by `build_graph.py` to support the `--resume` flag

---

### `fact_check_kg/build/llm_labeling.py`

Batched LLM-based topic label generation via Ollama.

- Sends groups of BERTopic topics (with keywords) to the LLM and parses short labels
- **Auto-tuning**: dynamically adjusts batch size based on success/failure history
  - Increases batch size after a `success_streak_threshold` of consecutive successes
  - Uses `binary_split` or `single_fallback` strategy on failure
- Falls back to keyword-based labels if the LLM is unavailable
- All parameters configurable in `config_build.yaml` under `llm_batch_labeling`

---

### `fact_check_kg/retrieval/retrieval.py`

**Class `HybridRetriever`** — the main retrieval engine.

- Initialized with the graph, topic model, embedding model, spaCy pipeline, and strategy weights
- Pre-computes chunk embeddings (loads from FAISS cache or recomputes) and topic distributions
- `retrieve(claim_text)` — runs all 7 strategies, fuses scores, and returns ranked chunk results
- Results include per-strategy scores, matched entities/themes/claims, and document-level rankings (using softmax pooling with configurable sharpness `softmax_beta`)

---

### `fact_check_kg/retrieval/retrieval_results.py`

Result formatting and export.

- `display_results_console(claim, results, top_n)` — pretty-prints top results to terminal
- `create_multiple_rankings(results)` — ranks by total score, per-component dominance, and document-level aggregation
- `generate_statistics(results)` — score distribution and component dominance analysis
- `save_results(claim, results, ...)` — exports JSON and text files to `outputs/retrieval_results/`

---

### `fact_check_kg/analysis/graph_manager.py`

Graph version management.

- `list_available_graphs()` — scans `outputs/graphs/metadata_*.json`
- `get_latest_graph_id()` — returns the most recently created graph ID
- `load_graph_by_id(graph_id)` — loads a specific graph (pickle, with optional PostgreSQL fallback)
- `display_available_graphs()` — prints a formatted table of all graphs

---

### `fact_check_kg/analysis/visualize_graph.py`

Interactive HTML graph visualization using Pyvis.

- `full` mode: renders the entire graph (configurable max nodes)
- `node` mode: renders a depth-limited subgraph around a specific node
- `analyze_graph_stats()`: prints graph statistics from a saved file or loaded graph
- Node and edge colors are configurable in `config/config_retrieval.yaml`

---

### `fact_check_kg/evaluation/evaluate.py`

Benchmark evaluation runner.

- `run_evaluation(retriever, queries_path, ...)` — main evaluation loop
  - Loads benchmark files, filters queries to in-graph documents
  - Optionally reformats raw social-media queries to factual claims via Ollama
  - For each query: runs retrieval, finds the rank of the ground-truth document
  - Computes Hit@1/5/10/50, MRR, mean/median rank, per-strategy Hit@10
  - Saves JSON + text reports to `outputs/evaluation/`
- Supports multi-threaded evaluation (`num_workers` parameter)
- Query results are cached to avoid re-running LLM formatting

---

### `utils/io_utils.py`

CSV loading utilities.

- `load_corpus_from_csv(path, ...)` — loads title + body columns, filters by language, returns texts + doc IDs
- `load_claims_from_csv(path, ...)` — loads claim reviews, maps to documents, deduplicates

---

## 11. Experimental: `build_evaluate.py`

> **Status: Work-in-progress.** This script is experimental and was not fully finalized due to time constraints. It already produces the intended benchmark output but has not been thoroughly tested across all edge cases.

`build_evaluate.py` is the **only script not called through `cli.py`**. It is run directly from the repository root.

### Purpose

The raw benchmark dataset (from an external source) uses document IDs from `data/documents.csv` that **do not match** the IDs used in the knowledge graph (which are row indices in `data/english_fact_checks.csv`). This script bridges that gap.

### What It Does

1. Builds a mapping from `documents.csv` IDs → `english_fact_checks.csv` IDs (via title/body matching)
2. Filters queries to those whose ground-truth document falls within the `--max-docs` slice used during graph construction
3. Optionally reformats raw social-media queries into clean factual claim statements using Ollama
4. Saves ready-to-use benchmark files for `cli.py evaluate`

### Outputs

| File | Description |
|------|-------------|
| `data/eval_queries_top{N}.csv` | Reformatted queries (query_id, claim_text) |
| `data/eval_relations_top{N}.csv` | Ground-truth links using graph doc IDs (query_id, ef_doc_id) |
| `data/eval_id_map.csv` | Full ID mapping (for debugging) |

### Usage

```bash
# Build benchmark files for a graph built on 10,000 documents
uv run python build_evaluate.py --max-docs 10000

# Skip Ollama reformatting (useful for testing the ID mapping)
uv run python build_evaluate.py --max-docs 10000 --skip-llm

# Then run evaluation
uv run python cli.py evaluate \
    --graph-id 20260112_143022 \
    --queries data/eval_queries_top10000.csv \
    --relations data/eval_relations_top10000.csv
```

### Known Limitations

- The title/body matching heuristic achieves approximately 80–90% match rate; unmatched documents are silently skipped.
- LLM reformatting via Ollama can be slow for large query sets (hundreds of thousands of queries). The script does not cache results across runs (unlike the caching in `cli.py evaluate`).
- Has not been fully validated with datasets other than the development benchmark.

---

## 12. Typical Usage Examples

### Build a Small Test Graph

```bash
uv run python cli.py init --dirs
uv run python cli.py build --max-docs 200
uv run python cli.py list-graphs
# Note the GRAPH_ID from the output (e.g., 20260112_143022)
```

### Retrieve Chunks for a Claim

```bash
uv run python cli.py retrieve \
    "The COVID-19 vaccine contains microchips." \
    --graph-id 20260112_143022 \
    --top-k 10 \
    --name microchip_test
# Results saved to outputs/retrieval_results/results_20260112_143022_microchip_test.json
```

### Visualize a Named Entity Neighborhood

```bash
uv run python cli.py visualize \
    --mode node \
    --node "entity::ORG::who" \
    --graph-id 20260112_143022
# Opens outputs/visualizations/graph_node_entity__ORG__who_20260112_143022.html
```

### Run an Evaluation Benchmark

```bash
# Step 1: prepare benchmark files (one-time, experimental)
uv run python build_evaluate.py --max-docs 10000 --skip-llm

# Step 2: run evaluation
uv run python cli.py evaluate \
    --graph-id 20260112_143022 \
    --queries data/eval_queries_top10000.csv \
    --relations data/eval_relations_top10000.csv \
    --max-queries 200 \
    --skip-llm
```

### Resume an Interrupted Build

```bash
uv run python cli.py build --resume
```

---

## 13. Current Limitations and Known Issues

### Graph Construction

- **Build time**: Building a graph over 10,000 documents takes several hours on CPU. A GPU is strongly recommended for embeddings, spaCy NER, and UMAP.
- **Memory**: Large graphs (10k+ documents) require significant RAM (~8–16 GB). Peak usage occurs during BERTopic training.
- **LLM dependency**: Topic labeling requires Ollama to be running. If Ollama is unavailable or times out, the build may fail or produce keyword-only labels.
- **Checkpoint granularity**: Checkpoints are saved at predefined build steps. Resuming from a checkpoint inside a long step (e.g., entity extraction) is not supported.

### Retrieval

- **Strategy weights are global**: Weights cannot be varied per-query without using the `--alpha/--beta/...` CLI overrides.
- **Strategies S2–S4 have low weight by default**: The current default configuration is heavily biased toward S1 (semantic, α=0.50) and S5 (claim-document, ε=0.15). The thematic (S3, γ=0.02) and topic distribution (S2, β=0.03) strategies contribute minimally.
- **Cold start**: The retriever pre-computes topic distributions and claim embeddings on initialization, which takes 10–30 seconds depending on graph size.

### Evaluation

- **`build_evaluate.py` is experimental**: The document ID mapping relies on string matching and may fail for documents with encoding issues or missing titles.
- **Evaluation scope**: The `evaluate` command only tests documents that are in the graph. If the graph was built on a subset of the corpus, queries referencing documents outside that subset are silently skipped.

### Storage

- **Pickle size**: A graph built on 10,000 documents produces a `.gpickle` file of approximately 500 MB–2 GB depending on entity and chunk counts.
- **PostgreSQL is optional**: The system operates fully without PostgreSQL. The dual-storage (pickle + PostgreSQL) is implemented but PostgreSQL is not required for any feature currently.

### General

- **Python version**: Tested on Python 3.10/3.11. Compatibility with Python 3.12+ has not been verified.
- **Windows paths**: The project was developed and tested on Windows. Unix paths (forward slashes) are used in configuration files and should work cross-platform via `pathlib`.
- **Old README was inaccurate**: The previous README described 4 retrieval strategies with different weights. The current implementation has 7 strategies. All documentation in this file reflects the actual code.
