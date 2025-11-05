# Fact-Checking Knowledge Graph

A multi-level knowledge graph system for fact-checking that combines semantic, thematic, and entity-based retrieval.

## Architecture

The system builds a 3-level knowledge graph:

```
Documents → Chunks → {Themes, Entities}
```

### Node Types
- **Documents**: Original fact-check articles
- **Chunks**: Text fragments (500 chars with 100 char overlap)
- **Themes**: Topics extracted via BERTopic and labeled with LLM
- **Entities**: Named entities extracted via spaCy NER (PERSON, ORG, GPE, etc.)

### Edge Types
- `CONTAINS`: Document → Chunk
- `HAS_TOPIC`: Chunk → Theme (weighted by probability)
- `MENTIONS`: Chunk → Entity
- `RELATED_TO`: Entity ↔ Theme (via chunks)
- `CO_OCCURS`: Entity ↔ Entity (co-appearance frequency)
- `SEMANTIC_SIM`: Entity ↔ Entity (embedding similarity)

## Modules

### `config.py`
Configuration dataclasses for graph construction and retrieval parameters.

### `data_loader.py`
Functions for loading documents from CSV and chunking text.

### `graph_builder.py`
Main class `KnowledgeGraphBuilder` for constructing the multi-level graph.

### `retrieval.py`
Hybrid retrieval system `HybridRetriever` combining 3 strategies:
1. **Semantic**: Direct embedding similarity
2. **Thematic**: Topic classification + graph traversal
3. **Entity**: Entity extraction + graph traversal (3 paths)

### `build_graph.py`
CLI script to build and save the knowledge graph.

### `test_retrieval.py`
CLI script to test retrieval with various claims.

## Installation

```bash
cd PRL
uv add bertopic keybert
uv run python -m spacy download en_core_web_sm
```

## Usage

### 1. Build the Knowledge Graph

```bash
cd fact_check_kg

# Build with default parameters (100 documents)
uv run python build_graph.py

# Build with custom parameters
uv run python build_graph.py \
    --csv ../data/english_fact_checks.csv \
    --output my_graph.gpickle \
    --max-docs 200 \
    --chunk-size 500 \
    --chunk-overlap 100 \
    --log-level INFO
```

**Output files:**
- `knowledge_graph.gpickle`: The NetworkX graph
- `knowledge_graph_topic_model.pkl`: The trained BERTopic model

### 2. Test Retrieval

```bash
# Test with default claims
uv run python test_retrieval.py

# Test with custom claim
uv run python test_retrieval.py \
    --claim "Photo from Gaza shared on Facebook" \
    --alpha 0.4 \
    --beta 0.3 \
    --gamma 0.3 \
    --top-k 10

# Test with different weights (emphasize entities)
uv run python test_retrieval.py \
    --claim "Trump met Macron" \
    --alpha 0.2 \
    --beta 0.2 \
    --gamma 0.6
```

## Retrieval Strategy

The hybrid retrieval combines 3 approaches:

### Step 1: Semantic Retrieval
- Embed the claim
- Compute cosine similarity with all chunks
- Normalize scores

### Step 2: Thematic Retrieval
- Classify claim into topics (BERTopic)
- Traverse graph via detected themes
- Score = claim_confidence × chunk_topic_probability

### Step 3: Entity Retrieval
- Extract entities from claim (spaCy NER)
- Traverse graph via 3 paths:
  - **Direct**: entity → chunk (score = 1.0)
  - **Co-occurrence**: entity → co_entity → chunk (score = weight × 0.8)
  - **Via theme**: entity → theme → chunk (score = weight × prob × 0.6)

### Final Fusion
```
final_score = α × semantic_score + β × thematic_score + γ × entity_score
```

Default weights: α=0.4, β=0.3, γ=0.3

## Example Output

```
=============================================================
CLAIM: Photo from Gaza 2008 shared on Facebook claiming Kashmir violence
=============================================================

Top 3 results:

--- Result #1 ---
Document ID: 0
Chunk ID: chunk::0::1

Scores:
  Total:  0.9234
  Semantic:  0.8521
  Thematic:  0.9501
  Entity:    1.0000

Text preview:
  A photo from Gaza was shared on Facebook with false claims about
  Kashmir. The photo is from 2008 and was taken after an Israeli
  attack on a refugee camp in Gaza...

--- Result #2 ---
Document ID: 15
Chunk ID: chunk::15::2

Scores:
  Total:  0.8765
  Semantic:  0.7834
  Thematic:  0.9012
  Entity:    0.9500

Text preview:
  Old photos from Gaza frequently misattributed to Kashmir conflict.
  Social media users often share images from Middle East conflicts...

--- Result #3 ---
Document ID: 42
Chunk ID: chunk::42::5

Scores:
  Total:  0.7823
  Semantic:  0.7201
  Thematic:  0.8123
  Entity:    0.8200

Text preview:
  Facebook posts showing Gaza violence often go viral with false
  claims about other regions. Fact-checkers warn against sharing...
```

## Configuration

### Graph Construction Parameters

```python
GraphConfig(
    chunk_size=500,              # Characters per chunk
    chunk_overlap=100,           # Overlap between chunks
    embedding_model="all-MiniLM-L6-v2",
    ner_model="en_core_web_sm",
    entity_types=['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'NORP'],
    min_topic_size=10,           # Min docs per topic
    nr_topics=None,              # Auto-determine
    co_occur_threshold=2,        # Min co-occurrences
    semantic_sim_threshold=0.6,  # Min entity similarity
    entity_related_threshold=0.5 # Min entity-theme weight
)
```

### Retrieval Parameters

```python
RetrievalConfig(
    alpha=0.4,                   # Semantic weight
    beta=0.3,                    # Thematic weight
    gamma=0.3,                   # Entity weight
    top_k=10,                    # Number of results
    theme_threshold=0.1          # Min theme confidence
)
```

## Performance Tips

1. **Start small**: Test with 50-100 documents first
2. **Adjust chunk size**: Smaller chunks = more granular but slower
3. **Tune weights**: For entity-heavy claims, increase γ
4. **Topic count**: Let BERTopic auto-determine or set based on domain
5. **Caching**: Chunk embeddings are pre-computed for fast retrieval

## Dependencies

- `networkx`: Graph structure
- `sentence-transformers`: Embeddings
- `bertopic`: Topic modeling
- `spacy`: Named Entity Recognition
- `numpy`: Numerical operations
- `scikit-learn`: Normalization

## File Structure

```
fact_check_kg/
├── __init__.py                 # Module init
├── config.py                   # Configuration classes
├── data_loader.py              # Document loading & chunking
├── graph_builder.py            # Graph construction
├── retrieval.py                # Hybrid retrieval system
├── build_graph.py              # CLI: Build graph
├── test_retrieval.py           # CLI: Test retrieval
├── README.md                   # This file
├── knowledge_graph.gpickle     # Generated graph (after build)
└── knowledge_graph_topic_model.pkl  # Generated topic model
```

## Future Improvements

- [ ] LLM integration for topic labeling (Llama via ollama)
- [ ] Temporal consistency checks (date entity validation)
- [ ] Multi-hop reasoning with explanation paths
- [ ] Contradiction detection between chunks
- [ ] Graph visualization for specific queries
- [ ] Batch retrieval for multiple claims
- [ ] Performance optimization with FAISS for embeddings

## License

MIT

## Contact

For questions or issues, please contact the development team.
