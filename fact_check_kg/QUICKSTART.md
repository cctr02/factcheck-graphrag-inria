# Quick Start Guide - Fact-Checking Knowledge Graph

## 🚀 Installation

```bash
cd PRL
uv sync  # Install all dependencies
uv run python -m spacy download en_core_web_sm
```

## 📊 Build the Knowledge Graph

```bash
cd fact_check_kg

# Quick test (50 documents, ~30 seconds)
uv run python build_graph.py --max-docs 50

# Production (500 documents, ~5 minutes)
uv run python build_graph.py --max-docs 500
```

**Output:**
- `knowledge_graph.gpickle` - The graph (NetworkX format)
- `knowledge_graph_topic_model.pkl` - Topic model (BERTopic)

## 🔍 Test Retrieval

```bash
# Test with default claims
uv run python test_retrieval.py

# Test with your own claim
uv run python test_retrieval.py --claim "Your fact to check here"

# Adjust weights (default: α=0.4, β=0.3, γ=0.3)
uv run python test_retrieval.py \
    --claim "Your claim" \
    --alpha 0.5 \
    --beta 0.25 \
    --gamma 0.25
```

## 📖 Understanding the Results

Example output:
```
--- Result #1 ---
Document ID: 0
Chunk ID: chunk::0::0

Scores:
  Total:  0.8763       ← Final weighted score
  Semantic:  1.0000    ← Embedding similarity
  Thematic:  0.5877    ← Topic match
  Entity:    1.0000    ← Entity match (Gaza, 2008 found)

Text preview:
  This is an 11-year-old photo from Gaza...
```

**Score interpretation:**
- **Total > 0.8**: Very likely relevant
- **Total 0.6-0.8**: Probably relevant
- **Total < 0.6**: May be related but less confident

**Component scores:**
- **Semantic = 1.0**: Perfect embedding match
- **Entity = 1.0**: All claim entities found in chunk
- **Thematic = 1.0**: Strong topic overlap

## ⚙️ Adjusting Weights

Depending on your use case, adjust α, β, γ:

### For claims with specific names/dates (e.g., "Trump met Macron in 2018")
```bash
--alpha 0.2 --beta 0.2 --gamma 0.6  # Emphasize entities
```

### For vague thematic queries (e.g., "vaccine misinformation")
```bash
--alpha 0.3 --beta 0.5 --gamma 0.2  # Emphasize themes
```

### For general similarity search
```bash
--alpha 0.6 --beta 0.2 --gamma 0.2  # Emphasize semantics
```

## 📈 Performance

**50 documents:**
- Build time: ~30 seconds
- Graph: 1,728 nodes, 31,000 edges
- Retrieval: ~4 seconds

**500 documents (estimated):**
- Build time: ~5 minutes
- Graph: ~17,000 nodes, ~300,000 edges
- Retrieval: ~10 seconds

## 🛠️ Troubleshooting

### Error: "No module named 'fact_check_kg'"
```bash
# Make sure you're in the fact_check_kg directory
cd fact_check_kg
```

### Error: "Can't find model 'en_core_web_sm'"
```bash
uv run python -m spacy download en_core_web_sm
```

### Error: "File knowledge_graph.gpickle does not exist"
```bash
# Build the graph first
uv run python build_graph.py --max-docs 50
```

### Slow performance
- Reduce `--max-docs` (start with 50)
- Use `--chunk-size 300` for smaller chunks
- Run on a machine with more RAM/CPU

## 📚 Next Steps

- Read [README.md](README.md) for detailed documentation
- Read [ARCHITECTURE.md](../ARCHITECTURE.md) for system design
- Explore the code in:
  - `graph_builder.py` - Graph construction
  - `retrieval.py` - Hybrid retrieval system
  - `config.py` - Adjustable parameters

## 🎯 Example Workflow

```bash
# 1. Build graph with test data
uv run python build_graph.py --max-docs 100

# 2. Test retrieval
uv run python test_retrieval.py --claim "Photo from Gaza"

# 3. Adjust weights if needed
uv run python test_retrieval.py \
    --claim "Photo from Gaza" \
    --alpha 0.3 --beta 0.3 --gamma 0.4

# 4. Scale up
uv run python build_graph.py --max-docs 1000
```

## ✅ Success Indicators

You'll know it's working when:
1. ✅ Build completes without errors
2. ✅ Files `knowledge_graph.gpickle` and `*_topic_model.pkl` are created
3. ✅ Retrieval returns chunks with high scores (> 0.7)
4. ✅ Top results contain relevant text about the claim

## 📞 Help

For detailed questions, check:
- [README.md](README.md) - Full documentation
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design
- Code comments - Extensive docstrings in all modules

Happy fact-checking! 🎉
