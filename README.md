# Knowledge Graph for Fact-Checking

Fact-checking system based on a multi-level knowledge graph combining topic modeling (BERTopic), entity extraction (spaCy), and hybrid retrieval.

---

## 1. Graph Construction

The knowledge graph is built in several steps, creating a multi-level network linking documents, chunks, themes, and entities.

### 1.1 Node Construction

#### **Documents and Chunks**

Documents are split into **sentence-based chunks** with overlap to preserve context:

```python
# Intelligent sentence-based chunking
chunk_size = 500  # characters (soft limit)
chunk_overlap = 100  # characters

chunks = chunk_text(document, chunk_size=500, overlap=100)
```

**Intelligent chunking features**:
- Chunks are built by accumulating complete sentences until `chunk_size` is reached
- Chunks end at sentence boundaries (detected via regex: `(?<=[.!?])\s+(?=[A-Z])`)
- Overlap is calculated at sentence level (not character level)
- 90%+ of chunks end at natural sentence boundaries
- Preserves semantic coherence better than naive character-based chunking

**Node format**:
- Document: `doc::{doc_id}`
- Chunk: `chunk::{doc_id}::{chunk_idx}`

Each chunk contains:
- `text`: textual content
- `doc_id`: reference to parent document
- `chunk_idx`: chunk index in the document

#### **Themes (Topics)**

Themes are automatically discovered via **BERTopic**:

```python
# Topic model training
topic_model = BERTopic(
    min_topic_size=10,
    nr_topics=None  # automatic detection
)

topics, probabilities = topic_model.fit_transform(chunk_texts)
```

**Label generation**: Themes are labeled using **intelligent keyword processing** by default (fast), or optionally via LLM (slower but more descriptive):

```python
# Intelligent keyword processing (default):
# 1. Retrieve top 15 keywords from BERTopic
keywords_raw = topic_model.get_topic(topic_id)[:15]

# 2. Lemmatize keywords using spaCy (merge variants)
# Example: "presidents" → "president", "running" → "run"

# 3. Remove redundant n-grams
# Example: "sri lanka" (0.9) eliminates "sri" (0.8) and "lanka" (0.7)
# Keeps only the most informative keyword form

# 4. Return top 10 deduplicated keywords
keywords = process_keywords(keywords_raw)  # Top 10 after deduplication

# 5. Create label from top 3 keywords
# Default: LLM-generated labels (more descriptive)
label = llm_generate_topic_label(keywords)  # e.g., "middle east conflict"

# Alternative: Keyword-based labels (use --no-llm-labels flag for speed)
# label = " ".join(keywords[:3])  # e.g., "gaza israel conflict"
```

**Keyword processing benefits**:
- Removes morphological variants (e.g., "president" vs "presidents")
- Eliminates redundant n-gram components (e.g., keeps "sri lanka", removes "sri" and "lanka")
- Results in cleaner, more informative topic representations
- Reduces noise in topic labels

**Node format**:
- Theme: `theme::{topic_id}`

Attributes:
- `topic_id`: BERTopic topic ID
- `label`: LLM-generated label
- `keywords`: list of main keywords

#### **Entities**

Entities are extracted via **spaCy NER** (Named Entity Recognition):

```python
# Entity extraction
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

entities = []
for ent in doc.ents:
    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'NORP']:
        # Special handling for DATEs: extract only years
        if ent.label_ == "DATE":
            year_match = re.search(r'\b(19|20)\d{2}\b', ent.text.lower())
            if year_match:
                year = year_match.group(0)
                entities.append((year, "DATE"))
        else:
            entities.append((ent.text.lower(), ent.label_))
```

**Extracted entity types**:
- `PERSON`: people (real or fictional)
- `ORG`: organizations, companies
- `GPE`: geopolitical entities (countries, cities)
- `LOC`: non-GPE locations (mountains, oceans)
- `DATE`: **years only** (4-digit years like "2019", "1995") extracted from date mentions
- `EVENT`: named events (wars, disasters)
- `NORP`: nationalities, religious or political groups

**Special DATE handling**:
- Only 4-digit years (19xx or 20xx) are extracted from DATE entities
- Detailed dates like "January 15, 2020" → year node "2020"
- Reduces date nodes by 88% (218 dates → 26 unique years)
- Enables temporal similarity connections between years

**Node format**:
- Entity: `entity::{entity_type}::{entity_text}`

---

### 1.2 Edge Construction

#### **CONTAINS**: Document → Chunk

Simple edge linking each document to its chunks.

```
Weight: 1.0 (unweighted)
```

---

#### **HAS_TOPIC**: Chunk → Theme

Links each chunk to themes detected by BERTopic, with a probability threshold.

**Weight calculation**:

```python
# BERTopic probability for each topic
topic_probs = topic_model.transform([chunk_text])[1][0]

# Create edges only if prob > threshold
for topic_id, prob in enumerate(topic_probs):
    if prob > topic_probability_threshold:  # default: 0.15
        graph.add_edge(
            chunk_id,
            f"theme::{topic_id}",
            edge_type="HAS_TOPIC",
            weight=prob  # BERTopic probability
        )
```

**Formula**:
```
weight_HAS_TOPIC = P(topic|chunk)
```

where `P(topic|chunk)` is the BERTopic probability that the chunk belongs to the topic.

---

#### **MENTIONS**: Chunk → Entity

Links chunks to entities they mention, weighted by **TF-IDF**.

**Weight calculation**:

```python
# TF (Term Frequency): number of entity occurrences in the chunk
TF(entity, chunk) = count(entity in chunk)

# IDF (Inverse Document Frequency): entity rarity
IDF(entity) = log(total_chunks / chunks_containing_entity)

# Final TF-IDF
weight_MENTIONS = TF(entity, chunk) × IDF(entity)
```

**Formula**:
```
weight_MENTIONS(chunk, entity) = TF(entity, chunk) × log(N / df(entity))
```

where:
- `N` = total number of chunks
- `df(entity)` = number of chunks containing the entity

**Advantage**: Rare entities (high informative value) receive higher weights.

---

#### **CO_OCCURS**: Entity ↔ Entity

Links entities that appear together in the same documents.

**Weight calculation**:

```python
# Co-occurrence at document level (not chunk)
co_occur_count = number of documents containing (entity1 AND entity2)

# Entity frequencies
freq1 = number of documents containing entity1
freq2 = number of documents containing entity2

# Weight inversely proportional to frequency
inv_freq_weight = 1 / sqrt(freq1 × freq2)

# Final weight
weight_CO_OCCURS = co_occur_count × inv_freq_weight
```

**Formula**:
```
weight_CO_OCCURS(e1, e2) = count(e1, e2) × (1 / √(freq(e1) × freq(e2)))
```

**Intuition**:
- Entities that co-appear frequently have high weight
- **But** omnipresent entities are penalized (divisor)
- Example: "Trump" and "France" rarely co-appear → high weight if co-occurrence detected

---

#### **CO_PRESENCE**: Theme ↔ Theme

Links themes that appear together in the same chunks (via HAS_TOPIC edges).

**Weight calculation**:

```python
# Co-presence at chunk level
co_presence_count = number of chunks containing (theme1 AND theme2)

# Theme frequencies
freq1 = number of chunks containing theme1
freq2 = number of chunks containing theme2

# Weight inversely proportional to frequency
inv_freq_weight = 1 / sqrt(freq1 × freq2)

# Final weight (same formula as CO_OCCURS)
weight_CO_PRESENCE = co_presence_count × inv_freq_weight
```

**Formula**:
```
weight_CO_PRESENCE(t1, t2) = count(t1, t2) × (1 / √(freq(t1) × freq(t2)))
```

---

#### **SEMANTIC_SIM**: Entity ↔ Entity / Theme ↔ Theme

Links semantically similar entities (or themes) via their embeddings.

**For entities** (excluding DATE entities):

```python
# Embeddings of entity names (DATE entities excluded)
embeddings = SentenceTransformer.encode([entity_text1, entity_text2, ...])

# Cosine similarity
sim = cosine_similarity(embedding1, embedding2) = embedding1 · embedding2

# Edge if sim > threshold (default: 0.8)
if sim > semantic_similarity_threshold:
    weight_SEMANTIC_SIM = sim

# NOTE: DATE entities use TEMPORAL_DIST instead (see below)
```

**For themes**:

```python
# Representation = concatenation of top 10 keywords
theme_repr = " ".join(topic_keywords[:10])

# Embeddings then cosine similarity
sim = cosine_similarity(theme_embedding1, theme_embedding2)
```

**Formula**:
```
weight_SEMANTIC_SIM(n1, n2) = cos(emb(n1), emb(n2)) = (emb(n1) · emb(n2)) / (||emb(n1)|| × ||emb(n2)||)
```

With normalized embeddings: `||emb|| = 1`, thus:
```
weight_SEMANTIC_SIM(n1, n2) = emb(n1) · emb(n2)
```

---

#### **TEMPORAL_DIST**: DATE ↔ DATE

Links DATE entities (years) using **exponential temporal decay** to capture temporal proximity.

**Weight calculation**:

```python
# Extract years from DATE entities
years = [int(year_text) for year_text in date_entities]

# Connect years within max_year_gap (default: 10 years)
for year1 in years:
    for year2 in years:
        year_diff = abs(year1 - year2)

        if 0 < year_diff <= max_year_gap:  # default: 10
            # Exponential decay formula
            weight_TEMPORAL_DIST = exp(-decay_rate × year_diff)
            # default decay_rate: 0.3
```

**Formula**:
```
weight_TEMPORAL_DIST(y1, y2) = exp(-λ × |y1 - y2|)
```

where:
- `λ` = decay_rate (default: 0.3)
- `|y1 - y2|` = absolute year difference

**Examples** (with decay_rate=0.3):
- Same year (diff=0): weight = 1.0
- 1 year difference: weight ≈ 0.74
- 2 years difference: weight ≈ 0.55
- 3 years difference: weight ≈ 0.41
- 5 years difference: weight ≈ 0.22
- 10 years difference: weight ≈ 0.05

**Rationale**:
- Years closer in time are more related than distant years
- Exponential decay creates smooth temporal neighborhood
- Configurable max_year_gap limits connections to relevant temporal window
- 26 year nodes create 134 temporal edges (vs. 1506 with previous approach)

---

## 2. Graph Retrieval

The system uses **hybrid retrieval** combining 4 complementary strategies, fused by weighted average.

**Strategy organization**:
- **Strategies 1-2**: Direct comparison methods (no graph traversal)
  - Strategy 1: Semantic similarity via embeddings
  - Strategy 2: Topic distribution similarity via Jensen-Shannon divergence
- **Strategies 3-4**: Graph-based methods (with BFS traversal)
  - Strategy 3: Thematic retrieval via theme graph traversal
  - Strategy 4: Entity-based retrieval via entity graph traversal

### 2.1 Strategy 1: Semantic Retrieval

Based on embedding similarity between the claim and chunks.

**FAISS Embedding Cache**:
- Chunk embeddings are pre-computed during graph construction
- Stored in FAISS (Facebook AI Similarity Search) index for fast retrieval
- Saves to `outputs/graphs/embeddings_{graph_id}.faiss` + metadata pickle
- Eliminates redundant embedding computation during retrieval
- Supports fast similarity search via inner product (cosine similarity with normalized vectors)

**Process**:

```python
# 1. Encode the claim
claim_embedding = SentenceTransformer.encode(claim)

# 2. Load chunk embeddings from FAISS cache (or recompute if missing)
if cache_exists:
    embedding_cache = EmbeddingCache()
    embedding_cache.load(cache_dir, graph_id)
    chunk_ids, chunk_embeddings = embedding_cache.get_all_embeddings()
else:
    chunk_embeddings = SentenceTransformer.encode(all_chunks)

# 3. Compute cosine similarities
similarities = chunk_embeddings @ claim_embedding

# 4. Normalize (min-max)
score_sem(chunk) = (sim(chunk) - min_sim) / (max_sim - min_sim)
```

**Formula**:
```
score_sem(chunk) = normalize(cos(emb(claim), emb(chunk)))
```

**Output**: Dictionary `{chunk_id: score_sem}` with normalized scores `[0, 1]`.

---

### 2.2 Strategy 2: Topic Distribution Retrieval

Based on similarity between topic probability distributions (Jensen-Shannon divergence).

**Process**:

```python
# 1. Get the topic distribution of the claim
claim_topic_dist = BERTopic.transform(claim)  # Probability vector

# 2. Get distributions for all chunks (pre-computed)
chunk_topic_dists = [BERTopic.transform(chunk) for chunk in chunks]

# 3. Compute Jensen-Shannon divergence
for chunk, chunk_dist in zip(chunks, chunk_topic_dists):
    js_divergence = jensen_shannon(claim_topic_dist, chunk_dist)

    # Convert to similarity (divergence ∈ [0, 1])
    similarity = 1 - js_divergence
    score_topic_dist[chunk] = similarity

# 4. Normalize
score_topic_dist(chunk) = normalize(similarity)
```

**Jensen-Shannon Divergence Formula**:

```
JS(P || Q) = (1/2) × KL(P || M) + (1/2) × KL(Q || M)
```

where:
- `M = (P + Q) / 2` (average distribution)
- `KL(P || Q) = ∑_i P(i) × log(P(i) / Q(i))` (Kullback-Leibler divergence)

**Conversion to similarity**:
```
score_topic_dist(chunk) = normalize(1 - JS(P_claim || P_chunk))
```

**Properties**:
- Symmetric: `JS(P || Q) = JS(Q || P)`
- Bounded: `JS ∈ [0, 1]`
- `JS = 0` if distributions are identical

**Output**: Dictionary `{chunk_id: score_topic_dist}`.

---

### 2.3 Strategy 3: Thematic Retrieval

Based on BERTopic topics and relevance propagation through the theme graph.

**Process**:

```python
# 1. Classify the claim into topics
topic_probs = BERTopic.transform(claim)

# 2. Select relevant themes (prob > threshold)
seed_themes = [(theme_id, prob) for theme_id, prob in topic_probs if prob > 0.1]

# 3. Propagate relevance through the theme graph (BFS)
for theme, relevance in BFS(seed_themes):
    for neighbor_theme in graph.neighbors(theme):
        edge_weight = graph[theme][neighbor_theme]['weight']  # CO_PRESENCE or SEMANTIC_SIM
        propagated_relevance = relevance × edge_weight
        if propagated_relevance > threshold:
            update_relevance(neighbor_theme, propagated_relevance)

# 4. Score chunks via HAS_TOPIC edges
for theme, relevance in theme_relevance.items():
    for chunk in chunks_with_HAS_TOPIC_to(theme):
        prob_weight = graph[chunk][theme]['weight']  # BERTopic probability
        score_theme[chunk] += relevance × prob_weight

# 5. Normalize
score_theme(chunk) = normalize(score_theme[chunk])
```

**Formulas**:

1. **Relevance propagation** (BFS through theme graph):
```
relevance(theme_neighbor) = max(
    current_relevance(theme_neighbor),
    relevance(theme_current) × weight_edge(theme_current, theme_neighbor)
)
```

2. **Final chunk score**:
```
score_theme(chunk) = normalize(∑_{theme} relevance(theme) × P(theme|chunk))
```

where `P(theme|chunk)` is the HAS_TOPIC edge weight (BERTopic probability).

**Output**: Dictionary `{chunk_id: score_theme}`.

---

### 2.4 Strategy 4: Entity-based Retrieval

Based on entity extraction from the claim and relevance propagation through the entity graph.

**Process**:

```python
# 1. Extract entities from the claim
entities_claim = spaCy.extract_entities(claim)

# 2. Find corresponding entities in the graph
seed_entities = [entity_node for entity_text in entities_claim if entity_node in graph]

# 3. Propagate relevance through the entity graph (BFS)
for entity, relevance in BFS(seed_entities):
    for neighbor_entity in graph.neighbors(entity):
        edge_weight = graph[entity][neighbor_entity]['weight']  # CO_OCCURS or SEMANTIC_SIM
        propagated_relevance = relevance × edge_weight
        if propagated_relevance > threshold:
            update_relevance(neighbor_entity, propagated_relevance)

# 4. Score chunks via MENTIONS edges
for entity, relevance in entity_relevance.items():
    for chunk in chunks_with_MENTIONS_to(entity):
        tf_idf_weight = graph[chunk][entity]['weight']  # TF-IDF
        score_entity[chunk] += relevance × tf_idf_weight

# 5. Normalize
score_entity(chunk) = normalize(score_entity[chunk])
```

**Formulas**:

1. **Relevance propagation** (BFS through entity graph):
```
relevance(entity_neighbor) = max(
    current_relevance(entity_neighbor),
    relevance(entity_current) × weight_edge(entity_current, entity_neighbor)
)
```

2. **Final chunk score**:
```
score_entity(chunk) = normalize(∑_{entity} relevance(entity) × TF-IDF(entity, chunk))
```

**Output**: Dictionary `{chunk_id: score_entity}`.

---

### 2.5 Score Fusion

The 4 scores are combined by **weighted average** with configurable weights.

**Fusion formula**:

```
score_final(chunk) = α × score_sem(chunk)
                   + β × score_topic_dist(chunk)
                   + γ × score_theme(chunk)
                   + δ × score_entity(chunk)
```

**Constraint**: `α + β + γ + δ = 1.0`

**Default weights**:
- `α = 0.3` (semantic)
- `β = 0.2` (topic distribution)
- `γ = 0.2` (thematic)
- `δ = 0.3` (entities)

**Fusion process**:

```python
# Union of all chunks found by the 4 strategies
all_chunks = set(S1) ∪ set(S2) ∪ set(S3) ∪ set(S4)

# For each chunk, compute final score
results = []
for chunk in all_chunks:
    score_sem = S1.get(chunk, 0)
    score_topic_dist = S2.get(chunk, 0)
    score_theme = S3.get(chunk, 0)
    score_entity = S4.get(chunk, 0)

    score_final = α×score_sem + β×score_topic_dist + γ×score_theme + δ×score_entity

    results.append({
        'chunk_id': chunk,
        'score_total': score_final,
        'score_sem': score_sem,
        'score_topic_dist': score_topic_dist,
        'score_theme': score_theme,
        'score_entity': score_entity
    })

# Sort by final score (descending)
results.sort(key=lambda x: x['score_total'], reverse=True)

# Return top-K
return results[:top_k]
```

**Final output**: List of `top_k` most relevant chunks with detailed scores.

---

## 3. Technical Architecture

### Models used

- **Embeddings**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **NER**: `en_core_web_sm` (spaCy)
- **Topic Modeling**: BERTopic
- **LLM**: `llama3.1:latest` (Ollama) for topic labels (default, use `--no-llm-labels` to disable)

### Storage

- **Graph**: NetworkX (pickle) + PostgreSQL (versioned tables)
- **Topic Model**: pickle (.pkl)
- **Metadata**: JSON

### PostgreSQL table format

**Table `graph_{timestamp}`** (edges):
- `source`: source node
- `predicate`: edge type (CONTAINS, HAS_TOPIC, etc.)
- `target`: target node
- `weight`: edge weight

**Table `nodes_{timestamp}`** (nodes):
- `node_id`: unique identifier
- `node_type`: type (document, chunk, theme, entity)
- `attributes`: JSONB with all attributes

**Table `graph_metadata`**:
- `graph_id`: creation timestamp
- `num_nodes`: number of nodes
- `num_edges`: number of edges
- `metadata`: JSONB with configuration

---

## 4. Usage

### Quick Start Workflow

**1. Build a graph:**
```bash
uv run python cli.py build --max-docs 100
```
This creates a graph with a unique ID like `20251119_185707` (timestamp-based).

**2. List available graphs:**
```bash
uv run python cli.py list-graphs
```

Example output:
```
================================================================================
AVAILABLE GRAPHS
================================================================================

1. Graph ID: 20251119_190959
   Created: 2025-11-19T19:09:59
   Documents: 20, Chunks: 161
   Nodes: 762, Edges: 7042

2. Graph ID: 20251119_185707
   Created: 2025-11-19T18:57:28
   Documents: 1000, Chunks: 9270
   Nodes: 26272, Edges: 1510527
================================================================================
```

**Copy the graph ID** you want to use (e.g., `20251119_185707`).

**3. Use the graph for retrieval or visualization:**
```bash
# Retrieval (replace with your graph ID)
uv run python cli.py retrieve "Your claim here" --graph-id 20251119_185707

# Visualization (replace with your graph ID and entity)
uv run python cli.py visualize --mode entity --entity "trump" --graph-id 20251119_185707
```

---

### Building a graph

```bash
# Default build (uses LLM for topic labels - more descriptive)
uv run python cli.py build --max-docs 100

# Fast build with keyword-based topic labels (faster, less descriptive)
uv run python cli.py build --max-docs 100 --no-llm-labels

# With PostgreSQL storage
uv run python cli.py build --max-docs 100 --use-postgres
```

**Graph ID**: Each graph is identified by its creation timestamp (e.g., `20251119_185707`). All related files (visualizations, retrievals) use this ID.

**Important**: Always use `list-graphs` to get the correct graph ID before running retrieval or visualization commands.

**Note**: Statistics are automatically generated during graph build and saved to `outputs/statistics/{graph_id}/`:
- `general_stats_{graph_id}.json` - Node/edge counts by type
- `topic_stats_{graph_id}.json` - Topic details with chunks/documents
- `entity_stats_{graph_id}.json` - Entity statistics with top entities by degree
- `summary_{graph_id}.txt` - Human-readable summary report

### Retrieval

**Use `list-graphs` to get the correct graph ID, then retrieve:**

```bash
# Basic retrieval (use your actual graph ID from list-graphs)
uv run python cli.py retrieve "Trump met Macron in France" --graph-id 20251119_185707

# Custom parameters with named retrieval
uv run python cli.py retrieve "claim text" \
  --graph-id 20251119_185707 \
  --top-k 10 \
  --alpha 0.3 --beta 0.2 --gamma 0.3 --delta 0.2 \
  --name "my_experiment"
```

**Retrieval results** are saved in `outputs/retrieval_results/` as `results_{graph_id}_{name}.json/txt`:
- Without `--name`: Auto-incremented (e.g., `results_20251119_185707_1.json`, `results_20251119_185707_2.json`)
- With `--name "test"`: Custom name (e.g., `results_20251119_185707_test.json`)

**Result files include**:
- Top-ranked chunks with scores (semantic, thematic, entity, topic distribution)
- Alternative rankings (by semantic score, by entity score, etc.)
- Statistics (score distributions, component dominance)
- **Full text of ALL retrieved chunks** (at the end of the file)

### Visualization

**First, list available graphs to get the correct graph ID:**

```bash
uv run python cli.py list-graphs
```

This will show all available graphs with their IDs, creation dates, and statistics. Example output:
```
Available graphs:
  - 20251119_185707 (created: 2025-11-19 18:57:07)
  - 20251117_213022 (created: 2025-11-17 21:30:22)
```

**Then visualize using the correct graph ID:**

```bash
# Display pre-computed statistics
uv run python cli.py visualize --mode stats --graph-id 20251119_185707

# Full graph visualization
uv run python cli.py visualize --mode full --graph-id 20251119_185707

# Entity-centered subgraph (replace "trump" with your entity of interest)
uv run python cli.py visualize --mode entity --entity "trump" --graph-id 20251119_185707
```

**Important**: The `--graph-id` must match an existing graph. Use `list-graphs` to see available IDs.

**Visualization files** are saved as `graph_{mode}_{graph_id}.html` in `outputs/visualizations/`

**Visualization features**:
- **Chunk labels**: Display chunk IDs
- Hover to see full chunk text
- Interactive legend: Click node types to hide/show
- Normalized edge weights: All weights in [0, 1] range
- Larger nodes and fonts for readability


---


