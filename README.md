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
- **Rich Relations**: CO_OCCURS, SEMANTIC_SIM, HAS_TOPIC, MENTIONS, CONTAINS
- **Scalable**: PostgreSQL support for large-scale graphs
- **Configurable**: All parameters via config.yaml
- **Automatic Statistics**: Comprehensive stats generated after each build

### Hybrid Retrieval

- **Semantic**: Embedding similarity (alpha = 0.4)
- **Thematic**: Via BERTopic and graph traversal (beta = 0.3)
- **Entity-based**: Entity matching and graph paths (gamma = 0.3)
- **Detailed Scoring**: Score breakdowns showing contribution of each component
- **Multiple Rankings**: Results ranked by different criteria (semantic, thematic, entity, balance)
- **Statistical Analysis**: Mean, min, max scores and component dominance analysis

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
                         ↓ CO_OCCURS / SEMANTIC_SIM
                     Entities (related)
```

Note: The RELATED_TO edge between entities and themes has been removed as this information is already captured through the HAS_TOPIC relationship from chunks.

### Node Types

1. **Document**: A complete fact-check
2. **Chunk**: Text segment (500 characters by default)
3. **Theme**: Theme discovered by BERTopic
4. **Entity**: Named entity (PERSON, ORG, GPE, etc.)

### Edge Types

1. **CONTAINS**: Document → Chunk
2. **HAS_TOPIC**: Chunk → Theme (weighted by probability)
3. **MENTIONS**: Chunk → Entity
4. **CO_OCCURS**: Entity ↔ Entity (weighted by co-occurrence count)
5. **SEMANTIC_SIM**: Entity ↔ Entity (weighted by cosine similarity)

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
# Search with a claim (includes detailed scoring)
python cli.py retrieve "Trump met Macron in France to discuss NATO"

# Search with predefined test claims
python cli.py retrieve --test

# Adjust weights and display parameters
python cli.py retrieve "Vaccine causes autism" --alpha 0.5 --beta 0.3 --gamma 0.2 --top-n 10

# Use PostgreSQL
python cli.py retrieve --use-postgres --test

# Display more results with detailed breakdowns
python cli.py retrieve "Gaza photo" --top-k 20 --top-n 5

# Disable automatic result saving
python cli.py retrieve "Election fraud" --no-save-results
```

The retrieval command now provides:
- **Detailed score breakdowns** for each result showing semantic, thematic, and entity contributions
- **Multiple rankings**: Results sorted by total score, semantic score, thematic score, entity score, balance, and max component
- **Statistical analysis**: Mean, min, max scores for each component
- **Component dominance**: Shows which scoring component is dominant in results
- **Automatic saving**: Results saved to `outputs/retrieval_results/` in JSON and text formats

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
│   ├── graph_builder.py       # Graph construction (with statistics)
│   ├── db_connector.py        # PostgreSQL connector
│   │
│   ├── retrieval.py           # Hybrid retrieval system
│   ├── retrieval_results.py   # Results manager with detailed scoring
│   ├── visualize_graph.py     # Standard visualizations
│   ├── visualize_hierarchical.py  # Hierarchical visualizations
│   │
│   ├── build_graph.py         # Build script
│   ├── test_retrieval.py      # Retrieval test script (enhanced)
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
│   ├── exports/              # Exports (JSON, CSV, etc.)
│   ├── statistics/           # Graph statistics (JSON/TXT)
│   └── retrieval_results/    # Retrieval results with detailed scoring
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

### Custom Retrieval with Detailed Scoring

```python
from fact_check_kg.retrieval import HybridRetriever
from fact_check_kg.retrieval_results import RetrievalResultsManager

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

# Create results manager for enhanced display
results_manager = RetrievalResultsManager()

# Search
results = retriever.retrieve("My custom claim")

# Display with detailed breakdowns
results_manager.display_results_console("My custom claim", results, top_n=5)

# Generate multiple rankings
rankings = results_manager.create_multiple_rankings(results)
print("\nTop 3 by semantic score:")
for i, result in enumerate(rankings['by_semantic_score'][:3], 1):
    print(f"  {i}. {result['chunk_id']} - Semantic: {result['score_sem']:.3f}")

# Save results with all rankings
json_path, text_path = results_manager.save_results(
    "My custom claim", results, alpha=0.5, beta=0.2, gamma=0.3, rankings=rankings
)
print(f"Results saved to {json_path}")
```

## New Features

### Automatic Graph Statistics

When building a graph, comprehensive statistics are automatically generated and saved to `outputs/statistics/`:

**Generated Files:**
- `general_stats_{timestamp}.json` - Node and edge counts by type
- `topic_stats_{timestamp}.json` - Topic information with keywords and document counts
- `entity_stats_{timestamp}.json` - Entity statistics with top entities by degree
- `summary_{timestamp}.txt` - Human-readable summary report

**Statistics Include:**
- Total nodes and edges
- Breakdown by node type (Document, Chunk, Theme, Entity)
- Breakdown by edge type (CONTAINS, HAS_TOPIC, MENTIONS, CO_OCCURS, SEMANTIC_SIM)
- Topics with their representative words and document counts
- Most connected entities
- Distribution metrics

### Enhanced Retrieval Scoring

The retrieval system now provides detailed score breakdowns and multiple ranking methods:

**Score Breakdown:**
Each result shows:
- Total score (weighted combination)
- Semantic score (embedding similarity)
- Thematic score (topic matching)
- Entity score (entity matching)
- Contribution percentage of each component
- Dominant component identification

**Multiple Rankings:**
Results can be sorted by:
1. **Total Score** - Default weighted combination
2. **Semantic Score** - Best embedding matches
3. **Thematic Score** - Best topic matches
4. **Entity Score** - Best entity matches
5. **Balance** - Most balanced across all components
6. **Max Component** - Highest individual component score

**Statistical Analysis:**
- Mean, min, max, median for each score type
- Component dominance analysis (which component dominates results)
- Score distribution insights

**Output Formats:**
- Console display with colored output
- JSON format for programmatic use
- Text format for human review
- Automatic saving to `outputs/retrieval_results/`

### Example Output

```
CLAIM: Photo from Gaza 2008 shared on Facebook claiming Kashmir violence
================================================================================

Top 5 Results:

--- Result #1 ---
Chunk: chunk_doc_1_2
Document: doc_1

Scores:
  Total:     0.7234  [DOMINANT: Semantic (52.3%)]
  Semantic:  0.8123  (contribution: 52.3%)
  Thematic:  0.6892  (contribution: 28.6%)
  Entity:    0.5687  (contribution: 19.1%)

Explanation:
  - Semantic: High similarity (0.81) - text closely matches query embedding
  - Thematic: Moderate match (0.69) - shares similar topics
  - Entity: Moderate match (0.57) - some common entities (Gaza, Facebook)

Text preview:
  A photo from the 2008 Gaza conflict was shared on Facebook with misleading...

STATISTICS
================================================================================
Score Averages:
  Total: 0.6234 (min: 0.4123, max: 0.7234)
  Semantic: 0.7123 (min: 0.5234, max: 0.8123)
  Thematic: 0.5892 (min: 0.3456, max: 0.6892)
  Entity: 0.5123 (min: 0.2789, max: 0.5687)

Component Dominance:
  Semantic: 7 (70.0%)
  Thematic: 2 (20.0%)
  Entity: 1 (10.0%)
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
