# CLI Guide - Knowledge Graph Fact-Checking System

Complete guide for using the Command-Line Interface (CLI) of the Knowledge Graph Fact-Checking System.

## Table of Contents

- [Installation](#installation)
- [General Usage](#general-usage)
- [Commands](#commands)
  - [init](#init---initialize-system)
  - [build](#build---build-knowledge-graph)
  - [visualize](#visualize---visualize-graph)
  - [retrieve](#retrieve---search-claims)
  - [stats](#stats---display-statistics)
  - [relabel](#relabel---relabel-topics)
  - [config](#config---manage-configuration)
- [Complete Workflow Examples](#complete-workflow-examples)
- [Troubleshooting](#troubleshooting)

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Download required spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## General Usage

The CLI uses a command-based structure:

```bash
python cli.py [COMMAND] [OPTIONS]
```

To see all available commands:

```bash
python cli.py --help
```

To see help for a specific command:

```bash
python cli.py [COMMAND] --help
```

## Commands

### init - Initialize System

Initialize the system by creating necessary directories and database tables.

#### Arguments

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dirs` | flag | False | Create directory structure |
| `--postgres` | flag | False | Initialize PostgreSQL database |

#### What it does

- `--dirs`: Creates all required directories:
  - `logs/` - Log files
  - `outputs/` - Main output directory
  - `outputs/graphs/` - Saved graphs (pickle/gpickle)
  - `outputs/visualizations/` - HTML visualizations
  - `outputs/models/` - Saved models
  - `outputs/exports/` - Exported data
  - `outputs/statistics/` - Graph statistics
  - `outputs/retrieval_results/` - Retrieval results with detailed scoring

- `--postgres`: Creates PostgreSQL database tables:
  - `graph` table - Stores edges (source, predicate, target, weight)
  - `nodes` table - Stores node attributes (JSONB format)

#### Configuration Used

- `config.paths.*` - All directory paths
- `config.database.*` - PostgreSQL connection parameters

#### Examples

```bash
# Create all directories
python cli.py init --dirs

# Initialize PostgreSQL database
python cli.py init --postgres

# Do both at once
python cli.py init --dirs --postgres
```

---

### build - Build Knowledge Graph

Build the multi-level knowledge graph from CSV data.

#### Arguments

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-docs` | int | 100 | Maximum number of documents to process |
| `--chunk-size` | int | 500 | Size of text chunks in characters |
| `--chunk-overlap` | int | 100 | Overlap between chunks in characters |
| `--use-postgres` | flag | False | Save to PostgreSQL instead of pickle |
| `--log-level` | choice | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

#### What it does

1. Loads documents from CSV file (`data/english_fact_checks.csv`)
2. Chunks documents into smaller text segments
3. Extracts entities using spaCy NER
4. Discovers topics using BERTopic
5. Builds multi-level graph:
   - Level 1: Documents → Chunks (CONTAINS)
   - Level 2: Chunks → Themes (HAS_TOPIC)
   - Level 3: Chunks → Entities (MENTIONS)
   - Relations: Entity ↔ Entity (CO_OCCURS, SEMANTIC_SIM)
6. Generates comprehensive statistics automatically
7. Saves graph to pickle or PostgreSQL

#### Configuration Used

- `config.paths.csv_file` - Input CSV file path
- `config.graph_construction.*` - All construction parameters
- `config.models.*` - Model names (embedding, NER)
- `config.database.*` - PostgreSQL connection (if --use-postgres)

#### Examples

```bash
# Build with default parameters (100 docs, pickle format)
python cli.py build

# Build with custom parameters
python cli.py build --max-docs 50 --chunk-size 300 --chunk-overlap 50

# Build and save to PostgreSQL
python cli.py build --max-docs 200 --use-postgres

# Build with debug logging
python cli.py build --max-docs 10 --log-level DEBUG

# Full production build
python cli.py build --max-docs 1000 --use-postgres --log-level INFO
```

---

### visualize - Visualize Graph

Generate interactive HTML visualizations of the knowledge graph.

#### Arguments

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | choice | stats | Visualization mode (full, entity, hierarchical, stats) |
| `--entity` | string | None | Entity name for entity/hierarchical mode |
| `--max-nodes` | int | 200 | Maximum nodes to display (full mode) |
| `--output` | string | auto | Output HTML file path (auto-generated if not specified) |
| `--use-postgres` | flag | False | Load graph from PostgreSQL |
| `--graph-file` | string | None | Path to graph pickle file |

#### Visualization Modes

**1. stats** - Display graph statistics
- Shows node/edge counts by type
- Lists top entities
- No HTML output, prints to console

**2. full** - Complete graph overview
- Shows all nodes up to `--max-nodes` limit
- Color-coded by node type
- Interactive physics simulation
- Filterable by node type

**3. entity** - Entity-centered view
- Focus on a specific entity and its neighborhood
- Shows direct connections (1-hop)
- Useful for exploring entity relationships
- Requires `--entity` parameter

**4. hierarchical** - Hierarchical layout
- Multi-level view centered on entity
- Shows: entity → chunks → themes/related entities
- Organized by graph levels
- Requires `--entity` parameter

#### Configuration Used

- `config.visualization.*` - Visualization settings
- `config.paths.visualizations_dir` - Output directory
- `config.export.timestamp_format` - File naming
- `config.database.*` - PostgreSQL connection (if --use-postgres)

#### Examples

```bash
# Display statistics
python cli.py visualize --mode stats

# Full graph visualization (limited to 200 nodes)
python cli.py visualize --mode full

# Full graph with more nodes
python cli.py visualize --mode full --max-nodes 500

# Entity-centered view
python cli.py visualize --mode entity --entity "Gaza"

# Hierarchical view around an entity
python cli.py visualize --mode hierarchical --entity "Facebook"

# Use PostgreSQL as source
python cli.py visualize --mode full --use-postgres

# Custom output path
python cli.py visualize --mode entity --entity "Trump" --output my_viz.html

# Use specific pickle file
python cli.py visualize --mode full --graph-file outputs/graphs/my_graph.gpickle
```

---

### retrieve - Search Claims

Search for relevant chunks using hybrid retrieval with detailed scoring.

#### Arguments

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CLAIM` | string | None | Claim text to verify (positional argument) |
| `--alpha` | float | 0.4 | Weight for semantic similarity score |
| `--beta` | float | 0.3 | Weight for thematic similarity score |
| `--gamma` | float | 0.3 | Weight for entity-based score |
| `--top-k` | int | 10 | Number of results to retrieve |
| `--top-n` | int | 5 | Number of top results to display in detail |
| `--use-postgres` | flag | False | Load graph from PostgreSQL |
| `--test` | flag | False | Run with test claims from config |
| `--save-results` / `--no-save-results` | flag | True | Save results to files |

#### What it does

1. Loads knowledge graph and topic model
2. Extracts claim embedding (semantic)
3. Identifies claim topics (thematic)
4. Extracts claim entities (entity-based)
5. Performs hybrid retrieval:
   - **Semantic**: Cosine similarity with chunk embeddings
   - **Thematic**: Topic overlap via BERTopic
   - **Entity**: Graph traversal (direct + co-occurrence)
6. Combines scores: `total = α×semantic + β×thematic + γ×entity`
7. Displays detailed results with:
   - Score breakdown per component
   - Contribution percentages
   - Dominant component identification
   - Multiple alternative rankings
   - Statistical analysis
8. Saves results to `outputs/retrieval_results/` (JSON + text)

#### Score Weights Explained

- **alpha (α)**: Weight for semantic similarity
  - Higher = prioritize text similarity
  - Good for: finding similar wording

- **beta (β)**: Weight for thematic similarity
  - Higher = prioritize topic matching
  - Good for: finding same subject matter

- **gamma (γ)**: Weight for entity matching
  - Higher = prioritize named entities (people, places, orgs)
  - Good for: finding mentions of specific entities

**Note**: α + β + γ should equal 1.0 for normalized scores

#### Configuration Used

- `config.retrieval.*` - All retrieval parameters
- `config.models.*` - Model names
- `config.testing.test_claims` - Test claims (if --test)
- `config.graph_construction.entity_types` - Entity types to extract

#### Examples

```bash
# Search with a claim (default weights: α=0.4, β=0.3, γ=0.3)
python cli.py retrieve "Photo from Gaza 2008 shared on Facebook"

# Run with test claims from config
python cli.py retrieve --test

# Prioritize semantic similarity
python cli.py retrieve "Vaccine causes autism" --alpha 0.6 --beta 0.2 --gamma 0.2

# Prioritize entity matching
python cli.py retrieve "Trump met Macron" --alpha 0.2 --beta 0.2 --gamma 0.6

# Balanced weights
python cli.py retrieve "Election fraud claims" --alpha 0.33 --beta 0.33 --gamma 0.34

# Get more results
python cli.py retrieve "Climate change" --top-k 20 --top-n 10

# Use PostgreSQL and don't save results
python cli.py retrieve "Fake news" --use-postgres --no-save-results

# Production search with detailed output
python cli.py retrieve "COVID-19 vaccine safety" --alpha 0.4 --beta 0.3 --gamma 0.3 --top-k 15 --top-n 5
```

---

### stats - Display Statistics

Display comprehensive statistics about the knowledge graph.

#### Arguments

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--use-postgres` | flag | False | Load graph from PostgreSQL |
| `--detailed` | flag | False | Show detailed statistics (currently not used) |

#### What it does

Displays:
- Total number of nodes and edges
- Node counts by type (Document, Chunk, Theme, Entity)
- Edge counts by type (CONTAINS, HAS_TOPIC, MENTIONS, CO_OCCURS, SEMANTIC_SIM)
- Top entities by degree (most connected)
- Graph density and connectivity metrics

#### Configuration Used

- `config.database.*` - PostgreSQL connection (if --use-postgres)

#### Examples

```bash
# Display stats from pickle file
python cli.py stats

# Display stats from PostgreSQL
python cli.py stats --use-postgres

# Detailed statistics (future feature)
python cli.py stats --detailed
```

---

### relabel - Relabel Topics

Relabel topic themes using a Large Language Model (requires Ollama).

#### Arguments

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--use-postgres` | flag | False | Use PostgreSQL storage |
| `--llm-model` | string | llama3.1:latest | LLM model to use for relabeling |

#### What it does

1. Loads knowledge graph and topic model
2. For each topic, extracts representative words
3. Sends words to LLM with prompt: "Generate a concise 2-4 word label"
4. Updates topic names in the graph
5. Saves updated graph

**Requirements**: Ollama must be running locally with the specified model installed.

#### Configuration Used

- `config.models.llm_model` - Default LLM model
- `config.models.llm_endpoint` - Ollama API endpoint
- `config.models.llm_temperature` - Generation temperature
- `config.models.llm_max_tokens` - Max tokens for label

#### Examples

```bash
# Relabel with default model (llama3.1:latest)
python cli.py relabel

# Use different model
python cli.py relabel --llm-model "mistral:latest"

# Relabel graph in PostgreSQL
python cli.py relabel --use-postgres

# Use custom model and PostgreSQL
python cli.py relabel --use-postgres --llm-model "llama2:13b"
```

**Note**: Make sure Ollama is running:
```bash
ollama serve
ollama pull llama3.1:latest
```

---

### config - Manage Configuration

Display or validate the system configuration.

#### Arguments

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--show` | flag | False | Show current configuration |
| `--validate` | flag | False | Validate configuration file |

#### What it does

**--show**: Displays the complete configuration in YAML format, including:
- All paths
- Graph construction parameters
- Model names
- Database settings
- Retrieval weights
- Visualization settings
- Logging configuration

**--validate**: Checks that:
- config.yaml is valid YAML
- All required sections exist
- Configuration can be loaded successfully
- Reports which sections are valid

#### Examples

```bash
# Show current configuration
python cli.py config --show

# Validate configuration
python cli.py config --validate

# Do both
python cli.py config --show --validate
```

---

## Complete Workflow Examples

### Example 1: Quick Start (Pickle Storage)

```bash
# 1. Initialize directories
python cli.py init --dirs

# 2. Build graph with 100 documents
python cli.py build --max-docs 100

# 3. View statistics
python cli.py stats

# 4. Visualize full graph
python cli.py visualize --mode full --max-nodes 200

# 5. Search for a claim
python cli.py retrieve "Photo from Gaza shared on Facebook"

# 6. Run test claims
python cli.py retrieve --test
```

### Example 2: Production Setup (PostgreSQL)

```bash
# 1. Initialize everything
python cli.py init --dirs --postgres

# 2. Build large graph in PostgreSQL
python cli.py build --max-docs 1000 --use-postgres

# 3. View statistics
python cli.py stats --use-postgres

# 4. Relabel topics with LLM
python cli.py relabel --use-postgres

# 5. Visualize hierarchical view
python cli.py visualize --mode hierarchical --entity "Trump" --use-postgres

# 6. Search with custom weights
python cli.py retrieve "Election fraud" --alpha 0.5 --beta 0.3 --gamma 0.2 --use-postgres --top-k 20
```

### Example 3: Development and Testing

```bash
# 1. Validate configuration
python cli.py config --validate

# 2. Build small test graph
python cli.py build --max-docs 10 --log-level DEBUG

# 3. Check statistics
python cli.py stats

# 4. Test different retrieval weights
python cli.py retrieve --test --alpha 0.5 --beta 0.25 --gamma 0.25
python cli.py retrieve --test --alpha 0.3 --beta 0.4 --gamma 0.3
python cli.py retrieve --test --alpha 0.2 --beta 0.2 --gamma 0.6

# 5. Visualize entity neighborhoods
python cli.py visualize --mode entity --entity "Facebook"
python cli.py visualize --mode entity --entity "Gaza"
```

### Example 4: Analysis Workflow

```bash
# 1. Build graph
python cli.py build --max-docs 200

# 2. Generate visualizations
python cli.py visualize --mode full --output analysis/full_graph.html
python cli.py visualize --mode hierarchical --entity "vaccine" --output analysis/vaccine_view.html

# 3. Search multiple claims and save results
python cli.py retrieve "Vaccine causes autism" --top-n 10
python cli.py retrieve "5G causes COVID-19" --top-n 10
python cli.py retrieve "Election was stolen" --top-n 10

# 4. Check all saved results in outputs/retrieval_results/
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'fact_check_kg'`
```bash
# Solution: Run from project root
cd /path/to/PRL
python cli.py [command]
```

**Issue**: `FileNotFoundError: data/english_fact_checks.csv`
```bash
# Solution: Check data file exists
ls data/english_fact_checks.csv

# Or update config.yaml with correct path
```

**Issue**: `psycopg2.OperationalError: could not connect to server`
```bash
# Solution: Start PostgreSQL and create database
sudo service postgresql start
psql -U postgres -c "CREATE DATABASE knowledge_graph;"

# Or update config.yaml with correct credentials
```

**Issue**: `spacy.errors.E050: Can't find model 'en_core_web_sm'`
```bash
# Solution: Download spaCy model
python -m spacy download en_core_web_sm
```

**Issue**: Ollama connection error in relabel command
```bash
# Solution: Start Ollama server
ollama serve

# In another terminal, pull model
ollama pull llama3.1:latest
```

**Issue**: Out of memory during build
```bash
# Solution: Reduce max_docs or chunk_size
python cli.py build --max-docs 50 --chunk-size 300
```

### Getting Help

```bash
# See all commands
python cli.py --help

# See command-specific help
python cli.py build --help
python cli.py retrieve --help
python cli.py visualize --help

# Check configuration
python cli.py config --show
python cli.py config --validate
```

### Logs

All operations create log files in `logs/`:
- `logs/cli.log` - CLI operations
- `logs/graph_builder.log` - Graph construction
- `logs/retrieval.log` - Retrieval operations

Check logs for detailed error messages:
```bash
tail -f logs/cli.log
```

## Summary of All CLI Commands

| Command | Purpose | Main Use Case |
|---------|---------|---------------|
| `init` | Initialize system | First-time setup |
| `build` | Build knowledge graph | Create/update graph |
| `visualize` | Generate visualizations | Explore graph structure |
| `retrieve` | Search for claims | Fact-checking queries |
| `stats` | Display statistics | Monitor graph metrics |
| `relabel` | Relabel topics with LLM | Improve topic readability |
| `config` | Manage configuration | Check/validate settings |

## Configuration File

All default values come from `config.yaml`. Edit this file to change system behavior without command-line arguments.

Key sections:
- `paths` - All file/directory paths
- `graph_construction` - Building parameters
- `models` - Model names and settings
- `database` - PostgreSQL connection
- `retrieval` - Search weights and parameters
- `visualization` - Display settings
- `testing` - Test claims

See [config.yaml](../config.yaml) for full configuration options.
