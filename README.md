# Knowledge Graph Fact-Checking System

A comprehensive fact-checking system based on knowledge graphs, using machine learning and natural language processing to verify claims against a database of fact-checks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Development](#development)
- [Contributing](#contributing)

## Overview

This project implements a fact-checking system using a multi-level knowledge graph to analyze and verify claims. The system combines several techniques:

- **Named Entity Recognition** (spaCy) to identify people, places, organizations
- **Topic Modeling** (BERTopic) to discover themes in documents
- **Semantic Embeddings** (SentenceTransformers) for textual similarity
- **Knowledge Graph** (NetworkX) to model relationships
- **PostgreSQL Database** for scalable storage

### Use Cases

- Automatic verification of claims against existing fact-checks
- Discovery of patterns in misinformation
- Analysis of narrative propagation
- Identification of recurring entities and themes in fake news

## Features

### Graph Construction

- **Multi-level**: Documents → Chunks → Themes & Entities
- **Rich Relations**: CO_OCCURS, SEMANTIC_SIM, RELATED_TO, HAS_TOPIC, MENTIONS
- **Scalable**: PostgreSQL support for large-scale graphs
- **Configurable**: All parameters via config.yaml

### Hybrid Retrieval

- **Semantic**: Embedding similarity (alpha = 0.4)
- **Thematic**: Via BERTopic and graph traversal (beta = 0.3)
- **Entity-based**: Entity matching and graph paths (gamma = 0.3)

### Visualizations

- **Full Graph**: Filterable overview
- **Entity-Centered**: Local exploration
- **Hierarchical**: Organization by levels
- **Detailed Statistics**: Metrics and analytics

### Unified CLI

Command-line interface for all operations:
```bash
python cli.py build      # Build the graph
python cli.py visualize  # Visualize the graph
python cli.py retrieve   # Search for chunks
python cli.py stats      # Display statistics
python cli.py relabel    # Relabel themes with LLM
```

## Architecture

### Knowledge Graph Structure

```
Documents (fact-checks)
    ↓ CONTAINS
Chunks (text segments)
    ↓ HAS_TOPIC          ↓ MENTIONS
Themes (BERTopic)    Entities (NER)
    ↓ RELATED_TO         ↓ CO_OCCURS / SEMANTIC_SIM
    └────────────────────┘
```

### Node Types

1. **Document**: A complete fact-check
2. **Chunk**: Text segment (500 characters by default)
3. **Theme**: Theme discovered by BERTopic
4. **Entity**: Named entity (PERSON, ORG, GPE, etc.)

### Edge Types

1. **CONTAINS**: Document → Chunk
2. **HAS_TOPIC**: Chunk → Theme (weighted by probability)
3. **MENTIONS**: Chunk → Entity
4. **RELATED_TO**: Entity ↔ Theme (weighted by accumulated probabilities)
5. **CO_OCCURS**: Entity ↔ Entity (weighted by co-occurrence count)
6. **SEMANTIC_SIM**: Entity ↔ Entity (weighted by cosine similarity)

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL 12+ (optional, for scalable storage)
- Ollama (optional, for LLM-based theme relabeling)

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/PRL.git
cd PRL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm

# Initialize structure
python cli.py init --dirs

# (Optional) Initialize PostgreSQL
python cli.py init --postgres
```

### PostgreSQL Installation

```bash
# Install PostgreSQL
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS with Homebrew
brew install postgresql

# Windows: Download from https://www.postgresql.org/download/

# Create database
createdb knowledge_graph

# Update config.yaml with your connection parameters
# database:
#   host: localhost
#   port: 5432
#   database: knowledge_graph
#   user: postgres
#   password: your_password
```

## Configuration

All parameters are centralized in `config.yaml`:

```yaml
# Paths
paths:
  data_dir: "data"
  outputs_dir: "outputs"
  logs_dir: "logs"

# Graph construction
graph_construction:
  max_documents: 100
  chunk_size: 500
  chunk_overlap: 100
  min_topic_size: 10

# Models
models:
  embedding_model: "all-MiniLM-L6-v2"
  ner_model: "en_core_web_sm"

# Retrieval
retrieval:
  alpha: 0.4  # Semantic weight
  beta: 0.3   # Thematic weight
  gamma: 0.3  # Entity weight
  top_k: 10

# Database
database:
  host: "localhost"
  port: 5432
  database: "knowledge_graph"
```

See [config.yaml](config.yaml) for all available parameters.

## Usage

### Quick Start

```bash
# 1. Build the graph
python cli.py build --max-docs 100

# 2. Visualize the graph
python cli.py visualize --mode stats

# 3. Search for a claim
python cli.py retrieve "Photo from Gaza shared on Facebook"

# 4. Export the graph
python cli.py export --format json
```

### Graph Construction

```bash
# Build with pickle (default mode)
python cli.py build --max-docs 100 --chunk-size 500

# Build with PostgreSQL
python cli.py build --use-postgres --max-docs 1000

# Build with detailed logs
python cli.py build --log-level DEBUG
```

### Visualization

```bash
# Graph statistics
python cli.py visualize --mode stats

# Full graph view
python cli.py visualize --mode full --max-nodes 200

# Entity-centered view
python cli.py visualize --mode entity --entity "gaza"

# Hierarchical view
python cli.py visualize --mode hierarchical --entity "facebook"
```

### Retrieval

```bash
# Search with a claim
python cli.py retrieve "Trump met Macron in France to discuss NATO"

# Search with predefined test claims
python cli.py retrieve --test

# Adjust weights
python cli.py retrieve "Vaccine causes autism" --alpha 0.5 --beta 0.3 --gamma 0.2

# Use PostgreSQL
python cli.py retrieve --use-postgres --test
```

### Other Operations

```bash
# Relabel themes with LLM (requires Ollama)
python cli.py relabel --use-postgres

# Show configuration
python cli.py config --show

# Validate configuration
python cli.py config --validate
```

## Project Structure

```
PRL/
├── cli.py                      # Centralized CLI
├── config.yaml                 # Main configuration
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── fact_check_kg/             # Main package
│   ├── __init__.py
│   ├── logger.py              # Logging system
│   ├── config_loader.py       # Configuration loader
│   ├── config.py              # Legacy config (compatibility)
│   │
│   ├── data_loader.py         # Data loading and chunking
│   ├── graph_builder.py       # Graph construction
│   ├── db_connector.py        # PostgreSQL connector
│   │
│   ├── retrieval.py           # Hybrid retrieval system
│   ├── visualize_graph.py     # Standard visualizations
│   ├── visualize_hierarchical.py  # Hierarchical visualizations
│   │
│   ├── build_graph.py         # Build script
│   ├── test_retrieval.py      # Retrieval test script
│   ├── inspect_topics.py      # Topic inspection
│   └── relabel_topics.py      # LLM-based relabeling
│
├── data/                      # Data files
│   └── english_fact_checks.csv
│
├── outputs/                   # Generated outputs
│   ├── graphs/               # Graphs (pickle/gpickle)
│   ├── visualizations/       # HTML files
│   ├── models/               # Saved models
│   └── exports/              # Exports (JSON, CSV, etc.)
│
├── logs/                      # Log files
│   ├── cli.log
│   ├── graph_builder.log
│   └── retrieval.log
│
├── docs/                     # Documentation
│   ├── QUICKSTART.md
│   ├── ARCHITECTURE.md
│   ├── POSTGRES_MIGRATION.md
│   └── VISUALIZATION_GUIDE.md
│
├── factcheckbureau/          # External code (unchanged)
│
└── utils/                    # Utilities
    ├── helpers.py
    ├── io_utils.py
    └── logger.py
```

## Documentation

### Main Modules

#### graph_builder.py

Builds the multi-level graph:

```python
from fact_check_kg.graph_builder import KnowledgeGraphBuilder
from fact_check_kg.config_loader import get_config

config = get_config()
builder = KnowledgeGraphBuilder(
    embedding_model=config.models.embedding_model,
    ner_model=config.models.ner_model
)

graph = builder.build_graph(chunks)
builder.save_to_postgres()  # or save as pickle
```

#### retrieval.py

Hybrid retrieval system:

```python
from fact_check_kg.retrieval import HybridRetriever

retriever = HybridRetriever(
    graph=graph,
    topic_model=topic_model,
    embedding_model=embedding_model,
    nlp=nlp,
    alpha=0.4, beta=0.3, gamma=0.3
)

results = retriever.retrieve("Your claim here")
```

#### logger.py

Centralized logging system:

```python
from fact_check_kg.logger import get_logger, log_section

logger = get_logger(__name__)
log_section(logger, "PROCESSING STARTED")
logger.info("Processing data...")
```

### Storage Formats

#### Pickle (NetworkX)

- Format: `.gpickle`
- Advantages: Fast, simple
- Disadvantages: Doesn't scale, single file

#### PostgreSQL

- Tables: `graph` (edges), `nodes` (attributes)
- Advantages: Scalable, SQL queries, concurrent access
- Disadvantages: More complex setup

```sql
-- Edge table
CREATE TABLE graph (
    source TEXT,
    predicate TEXT,
    target TEXT,
    weight FLOAT
);

-- Node table
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT,
    attributes JSONB
);
```

## Development

### Testing

```bash
# Unit tests
pytest tests/

# Integration tests
python cli.py build --max-docs 10
python cli.py retrieve --test

# Validation
python cli.py config --validate
```

### Adding Features

1. Modify `config.yaml` if needed
2. Add code in `fact_check_kg/`
3. Add command in `cli.py` if relevant
4. Update documentation
5. Test with `pytest`

### Logging

All modules use the centralized logging system:

```python
from fact_check_kg.logger import get_logger

logger = get_logger(__name__)
logger.info("Info message")
logger.debug("Debug message")
logger.error("Error message")
```

Logs are saved in `logs/` with automatic rotation.

## Advanced Examples

### Build Custom Graph

```python
from fact_check_kg.config_loader import get_config
from fact_check_kg.graph_builder import KnowledgeGraphBuilder
from fact_check_kg.data_loader import load_and_chunk_documents

config = get_config()

# Load data
chunks, n_docs = load_and_chunk_documents(
    csv_path=config.paths.csv_file,
    chunk_size=300,
    chunk_overlap=50,
    max_docs=50
)

# Build graph
builder = KnowledgeGraphBuilder(
    embedding_model="all-MiniLM-L6-v2",
    ner_model="en_core_web_lg",  # More powerful model
    co_occur_threshold=3,
    semantic_sim_threshold=0.7
)

graph = builder.build_graph(chunks, min_topic_size=15, nr_topics=20)
builder.save_to_postgres(host="localhost", database="my_kg")
```

### Custom Retrieval

```python
from fact_check_kg.retrieval import HybridRetriever

# Create retriever
retriever = HybridRetriever(
    graph=graph,
    topic_model=topic_model,
    embedding_model=embedding_model,
    nlp=nlp,
    alpha=0.5,  # More weight on semantic
    beta=0.2,
    gamma=0.3,
    top_k=20
)

# Search
results = retriever.retrieve("My custom claim")

# Display results
for i, result in enumerate(results[:5], 1):
    print(f"\n{i}. Score: {result['score_total']:.3f}")
    print(f"   Text: {result['text'][:200]}...")
    print(f"   Semantic: {result['score_sem']:.3f}")
    print(f"   Thematic: {result['score_theme']:.3f}")
    print(f"   Entity: {result['score_entity']:.3f}")
```

## Performance

### Typical Metrics

- **Construction**: ~5-10 min for 100 documents
- **Retrieval**: <1s per query
- **Graph Size**: ~1000 nodes, ~5000 edges for 100 documents

### Optimization

```yaml
# In config.yaml
performance:
  batch_size: 1000      # Batch for PostgreSQL
  n_jobs: -1            # Use all cores
  max_memory_mb: 4096   # Memory limit
  enable_cache: true
```

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Check installation
pip list | grep sentence-transformers
pip list | grep spacy

# Reinstall if needed
pip install -r requirements.txt
```

### Issue: PostgreSQL connection failed

```bash
# Check if PostgreSQL is running
sudo service postgresql status

# Check credentials in config.yaml
python cli.py config --validate
```

### Issue: Out of memory

```bash
# Reduce max_documents or chunk_size
python cli.py build --max-docs 50 --chunk-size 300
```

## Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create a branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- **Your Name** - *Initial work*

## Acknowledgments

- BERTopic for topic modeling
- spaCy for NER
- SentenceTransformers for embeddings
- NetworkX for graph manipulation
- PostgreSQL for scalable storage

## Contact

For questions, contact: your.email@example.com

---

**Note**: This project is under active development. Documentation will be updated regularly.
