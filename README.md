# Knowledge Graph for Fact-Checking

A fact-checking system based on a multi-level knowledge graph combining topic modeling (BERTopic), entity extraction (spaCy), and hybrid retrieval.

---

## Quick Start

```bash
# 1. Install dependencies
uv pip install -r requirements.txt
uv run python -m spacy download en_core_web_sm

# 2. Initialize system
uv run python cli.py init --dirs

# 3. Build a knowledge graph
uv run python cli.py build --max-docs 100

# 4. List available graphs
uv run python cli.py list-graphs

# 5. Retrieve relevant chunks for a claim
uv run python cli.py retrieve "Your claim here" --graph-id GRAPH_ID
```

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Graph Construction](#2-graph-construction)
3. [Retrieval Strategies](#3-retrieval-strategies)
4. [CLI Commands](#4-cli-commands)
5. [Configuration](#5-configuration)
6. [Installation](#6-installation)

---

## 1. Architecture Overview

### Graph Structure

| Node Type | Format | Description |
|-----------|--------|-------------|
| Document | `doc::{id}` | Source fact-check articles |
| Chunk | `chunk::{doc_id}::{idx}` | Sentence-based text segments |
| Theme | `theme::{topic_id}` | BERTopic-discovered topics |
| Entity | `entity::{type}::{text}` | Named entities (PERSON, ORG, GPE, LOC, DATE, EVENT, NORP) |

### Edge Types

| Edge | Connection | Weight |
|------|------------|--------|
| CONTAINS | doc → chunk | 1.0 |
| HAS_TOPIC | chunk → theme | BERTopic probability |
| MENTIONS | chunk → entity | Sigmoid(TF-IDF) |
| CO_OCCURS | entity ↔ entity | Sigmoid(PMI) |
| CO_PRESENCE | theme ↔ theme | Count × InvFreq |
| SEMANTIC_SIM | entity ↔ entity, theme ↔ theme | Cosine similarity (≥0.8) |
| TEMPORAL_DIST | year ↔ year | Exponential decay |

### Models Used

- **Embeddings**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **NER**: `en_core_web_sm` (spaCy)
- **Topic Modeling**: BERTopic
- **LLM** (optional): `llama3.1:latest` via Ollama

---

## 2. Graph Construction

### Chunking Strategy

Documents are split into sentence-based chunks (500 chars, 100 chars overlap):
- Sentences are never split mid-sentence
- Overlap maintains context between chunks

### Entity Extraction

Entities are extracted via spaCy NER. DATE entities are normalized to years only (e.g., "January 15, 2020" → "2020").

### Topic Modeling

Themes are discovered via BERTopic with keyword processing:
1. Extract top 15 keywords
2. Stem keywords (merge morphological variants)
3. Remove redundant n-grams
4. Keep top 10 deduplicated keywords

Labels can be generated via LLM (default) or from keywords (`--no-llm-labels`).

### Edge Weights

| Edge Type | Formula |
|-----------|---------|
| HAS_TOPIC | `P(topic\|chunk)` from BERTopic |
| MENTIONS | `σ(normalized_TF × IDF / 3.0)` |
| CO_OCCURS | `σ(PMI / 2.0)` |
| CO_PRESENCE | `count × 1/√(freq₁ × freq₂)` |
| SEMANTIC_SIM | `cos(emb₁, emb₂)` if ≥ 0.8 |
| TEMPORAL_DIST | `exp(-0.15 × \|year₁ - year₂\|)` |

---

## 3. Retrieval Strategies

The system uses 4 complementary strategies fused by weighted average:

### Strategy 1: Semantic (α = 0.3)
Embedding similarity between claim and chunks via FAISS-cached vectors.

### Strategy 2: Thematic (β = 0.2)
BFS through theme graph (CO_PRESENCE, SEMANTIC_SIM edges) to score chunks via HAS_TOPIC edges.

### Strategy 3: Entity-based (γ = 0.3)
Extract entities from claim, BFS through entity graph (CO_OCCURS, SEMANTIC_SIM edges), score via MENTIONS edges.

### Strategy 4: Topic Distribution (δ = 0.2)
Jensen-Shannon divergence between claim and chunk topic distributions.

### Fusion Formula

```
score_final = α×S_semantic + β×S_thematic + γ×S_entity + δ×S_topic_dist
```

---

## 4. CLI Commands

### Build a Graph

```bash
# Default (with LLM labels)
uv run python cli.py build --max-docs 100

# Fast (keyword labels)
uv run python cli.py build --max-docs 100 --no-llm-labels

# With PostgreSQL storage
uv run python cli.py build --max-docs 100 --use-postgres
```

### List Graphs

```bash
uv run python cli.py list-graphs
```

### Retrieve

```bash
uv run python cli.py retrieve "claim text" --graph-id GRAPH_ID --top-k 10
```

### Statistics

```bash
uv run python cli.py stats --graph-id GRAPH_ID
uv run python cli.py stats --graph-id GRAPH_ID --detailed
```

### Visualization

```bash
# Statistics display
uv run python cli.py visualize --mode stats --graph-id GRAPH_ID

# Full graph (interactive HTML)
uv run python cli.py visualize --mode full --graph-id GRAPH_ID

# Node-centered subgraph
uv run python cli.py visualize --mode node --node "entity::PERSON::trump" --graph-id GRAPH_ID
```

### Relabel Topics

```bash
uv run python cli.py relabel --llm-model llama3.1:latest
```

### Configuration

```bash
uv run python cli.py config --show
uv run python cli.py config --validate
```

---

## 5. Configuration

Three-layer YAML configuration system in `config/`:

| File | Purpose |
|------|---------|
| `config.yaml` | Common settings (paths, models, database, logging) |
| `config_build.yaml` | Build parameters (chunking, topics, entities, LLM) |
| `config_retrieval.yaml` | Retrieval weights, thresholds, visualization |

### Key Settings

```yaml
# config_build.yaml
graph:
  chunk_size: 500
  chunk_overlap: 100
  min_topic_size: 25

# config_retrieval.yaml
retrieval:
  alpha: 0.3  # semantic weight
  beta: 0.2   # thematic weight
  gamma: 0.3  # entity weight
  delta: 0.2  # topic distribution weight
```

---

## 6. Installation

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- Ollama (optional, for LLM topic labels)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd PRL

# Install dependencies
uv pip install -r requirements.txt

# Download spaCy model
uv run python -m spacy download en_core_web_sm

# Initialize directories
uv run python cli.py init --dirs
```

### Ollama Setup (Optional)

Required only for LLM-generated topic labels:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh  # Linux/macOS
# Windows: https://ollama.com/download/windows

# Pull model
ollama pull llama3.1:latest
ollama serve
```

### PostgreSQL Setup (Optional)

```bash
uv run python cli.py init --postgres
```

Configure connection in `config/config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  database: knowledge_graph
  user: postgres
  password: your_password
```

---

## Project Structure

```
PRL/
├── cli.py                    # Main CLI entry point
├── config/                   # YAML configuration files
│   ├── config.yaml          # Common settings
│   ├── config_build.yaml    # Build parameters
│   └── config_retrieval.yaml # Retrieval settings
├── fact_check_kg/           # Main module
│   ├── build_graph.py       # Build orchestration
│   ├── graph_builder.py     # Core graph construction
│   ├── retrieval.py         # Hybrid retrieval (4 strategies)
│   ├── data_loader.py       # CSV loading + chunking
│   ├── config_loader.py     # YAML config loading
│   ├── embedding_cache.py   # FAISS embedding cache
│   ├── graph_manager.py     # Graph listing/loading
│   ├── db_connector.py      # PostgreSQL connector
│   ├── visualize_graph.py   # Pyvis visualization
│   ├── relabel_topics.py    # LLM topic relabeling
│   └── logger.py            # Logging system
├── utils/                   # Utility functions
│   └── io_utils.py         # CSV loading
├── data/                    # Input data
│   └── english_fact_checks.csv
├── outputs/                 # Generated files
│   ├── graphs/             # Graph files (.gpickle, .faiss, .pkl)
│   ├── statistics/         # Statistics JSON
│   ├── visualizations/     # HTML visualizations
│   └── retrieval_results/  # Query results
├── requirements.txt
└── pyproject.toml
```

---

## Output Files

After building a graph with ID `GRAPH_ID`:

| File | Description |
|------|-------------|
| `graph_{ID}.gpickle` | NetworkX graph |
| `topic_model_{ID}.pkl` | BERTopic model |
| `embeddings_{ID}.faiss` | FAISS index |
| `embeddings_{ID}.pkl` | FAISS metadata |
| `metadata_{ID}.json` | Build configuration |
| `general_stats_{ID}.json` | Node/edge counts |
| `topic_stats_{ID}.json` | Topic details |
| `entity_stats_{ID}.json` | Entity statistics |

---

