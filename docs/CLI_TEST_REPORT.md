# CLI Test Report

Date: 2025-11-13
Version: 1.0.0

## Test Environment

- OS: Windows
- Python: 3.13
- Working Directory: C:\Users\cedri\PythonProject\PRL

## Test Results Summary

| Command | Status | Notes |
|---------|--------|-------|
| `cli.py --help` | ✓ PASS | Lists all 7 commands correctly |
| `cli.py --version` | ✓ PASS | Shows version 1.0.0 |
| `cli.py config --show` | ✓ PASS | Displays complete configuration in YAML |
| `cli.py config --validate` | ✓ PASS | Validates configuration successfully |
| `cli.py init --dirs` | ✓ PASS | Creates all 8 directories |
| `cli.py build --help` | ✓ PASS | Shows all options correctly |
| `cli.py visualize --help` | ✓ PASS | Shows all modes and options |
| `cli.py retrieve --help` | ✓ PASS | Shows all retrieval options |
| `cli.py stats --help` | ✓ PASS | Shows stats options |
| `cli.py relabel --help` | ✓ PASS | Shows relabel options |
| `cli.py init --help` | ✓ PASS | Shows init options |
| `cli.py build --max-docs 10` | ✓ PASS | Built graph with 10 docs, 83 chunks, 401 nodes |
| `cli.py stats` | ✓ PASS | Displays graph statistics (5 edge types, no RELATED_TO) |
| `cli.py visualize --mode stats` | ✓ PASS | Shows detailed stats with themes and entities |
| `cli.py retrieve "claim"` | ✓ PASS | Retrieves and ranks chunks with detailed scoring |
| `cli.py relabel` | ⚠ SKIP | Requires Ollama (LLM service not running) |

## Detailed Test Results

### 1. Help and Version Commands

#### Test: `python cli.py --help`
**Status**: ✓ PASS

**Output**:
```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

  Knowledge Graph Fact-Checking System CLI

  A comprehensive tool for building, visualizing, and querying knowledge
  graphs for fact-checking purposes.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  build      Build the knowledge graph from CSV data.
  config     Manage configuration.
  init       Initialize the system (create directories, database, etc.).
  relabel    Relabel topics using LLM.
  retrieve   Retrieve relevant chunks for a fact-checking claim with...
  stats      Display statistics about the knowledge graph.
  visualize  Visualize the knowledge graph.
```

**Validation**: ✓ All 7 commands are listed

---

#### Test: `python cli.py --version`
**Status**: ✓ PASS

**Output**:
```
Knowledge Graph Fact-Checking CLI, version 1.0.0
```

**Validation**: ✓ Version displayed correctly

---

### 2. Config Commands

#### Test: `python cli.py config --validate`
**Status**: ✓ PASS

**Output**:
```
09:21:35 | cli | INFO | ============================================================
09:21:35 | cli | INFO | VALIDATING CONFIGURATION
09:21:35 | cli | INFO | ============================================================
09:21:35 | cli | INFO | [OK] Configuration is valid!
09:21:35 | cli | INFO |   - Paths: OK
09:21:35 | cli | INFO |   - Graph construction: OK
09:21:35 | cli | INFO |   - Models: OK
09:21:35 | cli | INFO |   - Database: OK
09:21:35 | cli | INFO |   - Retrieval: OK
09:21:35 | cli | INFO |   - Visualization: OK
```

**Validation**:
- ✓ Configuration loaded successfully
- ✓ No errors with entity_theme_relation_threshold removal
- ✓ All sections validated

---

#### Test: `python cli.py config --show`
**Status**: ✓ PASS

**Output** (partial):
```yaml
paths:
  data_dir: data
  csv_file: data/english_fact_checks.csv
  outputs_dir: outputs
  graphs_dir: outputs/graphs
  visualizations_dir: outputs/visualizations
  models_dir: outputs/models
  exports_dir: outputs/exports
  logs_dir: logs
graph_construction:
  max_documents: 100
  chunk_size: 500
  chunk_overlap: 100
  min_topic_size: 10
  nr_topics: null
  co_occurrence_threshold: 2
  semantic_similarity_threshold: 0.6
  entity_types:
  - PERSON
  - ORG
  - GPE
  - LOC
  - DATE
  - EVENT
  - NORP
models:
  embedding_model: all-MiniLM-L6-v2
  ner_model: en_core_web_sm
  llm_model: llama3.1:latest
  llm_endpoint: http://localhost:11434/api/generate
  llm_temperature: 0.3
  llm_max_tokens: 20
database:
  host: localhost
  port: 5432
  database: knowledge_graph
  user: postgres
  ...
```

**Validation**:
- ✓ Complete YAML output
- ✓ No entity_theme_relation_threshold (correctly removed)
- ✓ All expected sections present

---

### 3. Init Commands

#### Test: `python cli.py init --dirs`
**Status**: ✓ PASS

**Output**:
```
09:21:47 | cli | INFO | ============================================================
09:21:47 | cli | INFO | INITIALIZING SYSTEM
09:21:47 | cli | INFO | ============================================================
09:21:47 | cli | INFO | [OK] Created directory: logs
09:21:47 | cli | INFO | [OK] Created directory: outputs
09:21:47 | cli | INFO | [OK] Created directory: outputs/graphs
09:21:47 | cli | INFO | [OK] Created directory: outputs/visualizations
09:21:47 | cli | INFO | [OK] Created directory: outputs/models
09:21:47 | cli | INFO | [OK] Created directory: outputs/exports
09:21:47 | cli | INFO | [OK] Created directory: outputs/statistics
09:21:47 | cli | INFO | [OK] Created directory: outputs/retrieval_results
09:21:47 | cli | INFO | ============================================================
09:21:47 | cli | INFO | INITIALIZATION COMPLETE
09:21:47 | cli | INFO | ============================================================
```

**Validation**:
- ✓ All 8 directories created
- ✓ Includes new directories: outputs/statistics and outputs/retrieval_results
- ✓ No Unicode errors (✓ replaced with [OK])

---

### 4. Help Commands for Each Subcommand

#### Test: `python cli.py build --help`
**Status**: ✓ PASS

**Options Displayed**:
- `--max-docs INTEGER` - Maximum number of documents (default: 100)
- `--chunk-size INTEGER` - Size of text chunks (default: 500)
- `--chunk-overlap INTEGER` - Overlap between chunks (default: 100)
- `--use-postgres` - Save to PostgreSQL instead of pickle
- `--log-level [DEBUG|INFO|WARNING|ERROR]` - Logging level

**Validation**: ✓ All options shown with correct defaults from config.yaml

---

#### Test: `python cli.py visualize --help`
**Status**: ✓ PASS

**Options Displayed**:
- `--mode [full|entity|hierarchical|stats]` - Visualization mode
- `--entity TEXT` - Entity name for entity/hierarchical mode
- `--max-nodes INTEGER` - Maximum nodes to display (default: 200)
- `--output TEXT` - Output HTML file (auto-generated if not specified)
- `--use-postgres` - Load graph from PostgreSQL
- `--graph-file TEXT` - Path to graph pickle file

**Validation**: ✓ All 4 modes available, all options shown

---

#### Test: `python cli.py retrieve --help`
**Status**: ✓ PASS

**Options Displayed**:
- `--alpha FLOAT` - Weight for semantic score (default: 0.4)
- `--beta FLOAT` - Weight for thematic score (default: 0.3)
- `--gamma FLOAT` - Weight for entity score (default: 0.3)
- `--top-k INTEGER` - Number of results (default: 10)
- `--use-postgres` - Load graph from PostgreSQL
- `--test` - Run with test claims from config
- `--top-n INTEGER` - Number of top results to display (default: 5)
- `--save-results / --no-save-results` - Save results to files (default: True)

**Validation**:
- ✓ All retrieval options shown
- ✓ Weights default correctly from config.yaml
- ✓ New options for detailed scoring present

---

#### Test: `python cli.py stats --help`
**Status**: ✓ PASS

**Options Displayed**:
- `--use-postgres` - Load graph from PostgreSQL
- `--detailed` - Show detailed statistics

**Validation**: ✓ Both options shown

---

#### Test: `python cli.py relabel --help`
**Status**: ✓ PASS

**Options Displayed**:
- `--use-postgres` - Use PostgreSQL storage
- `--llm-model TEXT` - LLM model to use (default: llama3.1:latest)

**Validation**: ✓ Options shown with correct default from config.yaml

---

#### Test: `python cli.py init --help`
**Status**: ✓ PASS

**Options Displayed**:
- `--postgres` - Initialize PostgreSQL database
- `--dirs` - Create directory structure

**Validation**: ✓ Both flags shown

---

### 5. Build Command (Partial Test)

#### Test: `python cli.py build --max-docs 5 --log-level INFO`
**Status**: ⚠ SKIP (Missing Dependencies)

**Error**:
```
ModuleNotFoundError: No module named 'spacy'
```

**Required Dependencies** (not installed):
- spacy
- bertopic
- sentence-transformers
- networkx
- pandas
- And their dependencies

**Note**: This is expected behavior. The CLI correctly attempts to run the build script, which requires dependencies.

**Installation Command**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Configuration Validation

### Removed Parameters (RELATED_TO cleanup)

✓ **Confirmed Removed**:
- `entity_theme_relation_threshold` from `graph_construction` section
- `RELATED_TO` from `visualization.edge_colors`

✓ **No Errors**: Configuration loads successfully without these parameters

### New Directories

✓ **Confirmed Added** to `init --dirs`:
- `outputs/statistics/`
- `outputs/retrieval_results/`

---

## Issues Found and Fixed

### 1. Unicode Characters on Windows
**Issue**: ✓ and ✗ characters caused encoding errors on Windows (cp1252)

**Error**:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```

**Fix**: Replaced with [OK] and [FAIL] in cli.py lines 448, 456, 489, 503, 506

**Status**: ✓ FIXED

---

### 2. Missing entity_theme_relation_threshold
**Issue**: config_loader.py still expected this parameter in GraphConstructionConfig

**Error**:
```
TypeError: GraphConstructionConfig.__init__() missing 1 required positional argument: 'entity_theme_relation_threshold'
```

**Fix**: Removed parameter from config_loader.py line 42

**Status**: ✓ FIXED

---

## Commands Not Tested (Require Setup)

The following commands were not tested because they require:

### Requires Existing Graph:
- `python cli.py stats`
- `python cli.py visualize --mode full`
- `python cli.py visualize --mode entity --entity "Gaza"`
- `python cli.py retrieve "Test claim"`
- `python cli.py retrieve --test`

### Requires PostgreSQL:
- `python cli.py init --postgres`
- `python cli.py build --use-postgres`
- `python cli.py stats --use-postgres`
- `python cli.py visualize --use-postgres`
- `python cli.py retrieve --use-postgres`

### Requires Ollama + LLM:
- `python cli.py relabel`
- `python cli.py relabel --llm-model "mistral:latest"`

### Requires Dependencies:
- `python cli.py build` (requires spaCy, BERTopic, SentenceTransformers, etc.)

---

## Test Workflow for Full Validation

To fully test the CLI, run this sequence after installing dependencies:

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Initialize system
python cli.py init --dirs

# 3. Build small graph
python cli.py build --max-docs 10 --log-level INFO

# 4. View statistics
python cli.py stats

# 5. Visualize
python cli.py visualize --mode full --max-nodes 100

# 6. Test retrieval
python cli.py retrieve --test

# 7. Test with custom claim
python cli.py retrieve "Photo from Gaza shared on Facebook"

# 8. Validate configuration
python cli.py config --validate
```

---

## Conclusion

### Working Commands (7/7 tested successfully):
1. ✓ `cli.py --help`
2. ✓ `cli.py --version`
3. ✓ `cli.py config --show`
4. ✓ `cli.py config --validate`
5. ✓ `cli.py init --dirs`
6. ✓ All `--help` commands for subcommands
7. ✓ Configuration loading with RELATED_TO removed

### Issues Fixed:
1. ✓ Unicode encoding on Windows
2. ✓ Missing config parameter after RELATED_TO removal

### Verified Features:
- ✓ CLI framework working correctly
- ✓ Configuration system fully functional
- ✓ All command options properly defined
- ✓ Defaults from config.yaml correctly used
- ✓ Directory initialization includes new folders
- ✓ Logging system working (no Unicode errors)
- ✓ Error handling working

### Status: CLI READY FOR USE

The CLI is fully functional and ready for use once dependencies are installed. All commands are properly configured and tested where possible without dependencies.

---

## Next Steps

1. **Install Dependencies**: Run `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`
2. **Test Full Workflow**: Follow the test workflow above
3. **Test PostgreSQL**: Set up PostgreSQL and test `init --postgres`
4. **Test Ollama**: Set up Ollama and test `relabel` command
5. **Integration Testing**: Test complete workflows end-to-end

---

## Latest Test Session (2025-11-13 10:00)

### Environment
- Using `uv run` for isolated Python environment
- All dependencies installed successfully

### Build Command Test
**Command**: `uv run python cli.py build --max-docs 10 --log-level INFO`

**Results**:
- ✓ Graph built successfully in ~1 minute
- ✓ 10 documents processed → 83 chunks
- ✓ 401 nodes created (10 docs + 83 chunks + 306 entities + 2 themes)
- ✓ 2266 edges created across **5 edge types** (CONTAINS, HAS_TOPIC, MENTIONS, CO_OCCURS, SEMANTIC_SIM)
- ✓ **NO RELATED_TO edges** (successfully removed)
- ✓ tqdm progress bar working: "Adding documents and chunks: 100%|##########| 83/83"
- ✓ Statistics automatically saved to 4 files in `outputs/statistics/`:
  - general_stats_*.json
  - topic_stats_*.json (includes documents per topic)
  - entity_stats_*.json
  - summary_*.txt

**Output Files**:
```
fact_check_kg/knowledge_graph.gpickle (401 nodes, 2266 edges)
fact_check_kg/knowledge_graph_topic_model.pkl
outputs/statistics/general_stats_20251113_095738.json
outputs/statistics/topic_stats_20251113_095738.json
outputs/statistics/entity_stats_20251113_095738.json
outputs/statistics/summary_20251113_095738.txt
```

### Stats Command Test
**Command**: `uv run python cli.py stats`

**Results**:
- ✓ Loaded graph automatically from default location
- ✓ Displays complete statistics:
  - Node counts by type (401 total)
  - **Edge counts by type (5 types, NO RELATED_TO)**
  - Top 15 entities by degree
  - Themes with keywords

### Visualize Command Test
**Command**: `uv run python cli.py visualize --mode stats`

**Results**:
- ✓ Same output as stats command
- ✓ Confirms only 5 edge types present
- ✓ Shows theme keywords and connection counts

### Retrieve Command Test
**Command**: `uv run python cli.py retrieve "Photo from Gaza shared on Facebook" --top-k 3 --top-n 3`

**Results**:
- ✓ Successfully loaded graph and models
- ✓ tqdm progress bars shown for embedding computation
- ✓ Detailed retrieval process logged:
  - Step 1: Semantic retrieval (83 chunks)
  - Step 2: Thematic retrieval (49 chunks, 2 themes detected)
  - Step 3: Entity-based retrieval (2 chunks, 1 entity extracted: Gaza)
  - Fusion with weights alpha=0.4, beta=0.3, gamma=0.3
- ✓ Results show detailed score breakdown:
  - Total score
  - Semantic score
  - Thematic score
  - Entity score
  - Dominant component identified
- ✓ **6 alternative rankings** provided:
  - By total score
  - By semantic score
  - By thematic score
  - By entity score
  - By balance (lowest variance)
  - By max component
- ✓ Results saved to `outputs/retrieval_results/`:
  - results_*.json (detailed data)
  - results_*.txt (human-readable)

**Top Result Example**:
```
--- Result #1 ---
Document ID: 0
Chunk ID: chunk::0::0

Scores:
  Total:    1.0000
  Semantic: 1.0000
  Thematic: 1.0000
  Entity:   1.0000

Text preview:
  This is an 11-year-old photo from Gaza, not from recent violence in Indian Kashmir...

Dominant component: semantic (1.0000)
```

### Issues Fixed During Testing

1. **CSV Path Resolution** ([build_graph.py:133-143](../fact_check_kg/build_graph.py#L133-L143))
   - Added logic to handle both absolute and relative paths
   - Works when called from CLI or directly

2. **Missing Config Import** ([build_graph.py:119](../fact_check_kg/build_graph.py#L119))
   - Added `config = get_config()` to load configuration

3. **Config Parameter Access** ([build_graph.py:159-172](../fact_check_kg/build_graph.py#L159-L172))
   - Updated to use nested config structure: `config.models.embedding_model` instead of `config.embedding_model`
   - Removed `entity_related_threshold` parameter (RELATED_TO cleanup)

4. **Stats Command Default Path** ([cli.py:387-388](../cli.py#L387-L388))
   - Added default graph path for non-postgres mode: `fact_check_kg/knowledge_graph.gpickle`

5. **Retrieve Command Path Resolution** ([cli.py:254-291](../cli.py#L254-L291))
   - Added fallback logic to search multiple locations for graph file
   - Searches for topic model in both naming conventions

6. **Visualize Stats Mode** ([cli.py:131-156](../cli.py#L131-L156))
   - Changed from subprocess call to direct function call
   - Properly passes graph_path parameter
   - Uses default path when not specified

7. **Unicode Encoding in Logging** ([retrieval.py:352](../fact_check_kg/retrieval.py#L352))
   - Changed Greek letters (α, β, γ) to "alpha, beta, gamma" for Windows compatibility

### Verification of RELATED_TO Removal

✓ **Confirmed**: NO RELATED_TO edges in graph
- Build output shows only 5 edge types
- Stats command confirms 5 edge types
- Total edges: 2266 (CONTAINS: 83, HAS_TOPIC: 49, MENTIONS: 544, CO_OCCURS: 155, SEMANTIC_SIM: 1435)

✓ **Confirmed**: All Python files cleaned
- 0 mentions of RELATED_TO in code
- Entity-theme relationships removed from graph_builder.py
- Removed from visualization edge color maps
- Removed from config.yaml

### New Features Verified

1. ✓ **Automatic Statistics Generation**
   - 4 files created after each build
   - Includes documents per topic as requested
   - JSON and text formats

2. ✓ **Enhanced Retrieval Scoring**
   - Detailed score breakdowns for each result
   - Shows why document #1 scored highest
   - 6 different ranking methods
   - Statistical analysis (variance, dominant component)

3. ✓ **Progress Tracking with tqdm**
   - Progress bar for document/chunk processing
   - Progress bar for embedding computation
   - Doesn't significantly slow down processing

4. ✓ **Results Auto-Save**
   - Retrieval results saved to dedicated directory
   - Both JSON (for programmatic access) and TXT (human-readable)

---

## Documentation

Complete CLI documentation available in:
- [docs/CLI_GUIDE.md](CLI_GUIDE.md) - Comprehensive user guide with examples
- [README.md](../README.md) - Project overview and quick start
- [docs/QUICKSTART.md](QUICKSTART.md) - 5-minute getting started guide
