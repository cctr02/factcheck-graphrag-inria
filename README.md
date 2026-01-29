# Knowledge Graph for Fact-Checking

Fact-checking system based on a multi-level knowledge graph combining topic modeling (BERTopic), entity extraction (spaCy), and hybrid retrieval.

---

## Quick Start

```bash
# 1. Initialize system (create directories)
uv run python cli.py init --dirs

# 2. Build a knowledge graph from your data
uv run python cli.py build --max-docs 100

# 3. List available graphs
uv run python cli.py list-graphs

# 4. Retrieve relevant chunks for a claim
uv run python cli.py retrieve "Your claim here" --graph-id GRAPH_ID

# 5. Visualize graph statistics
uv run python cli.py stats --graph-id GRAPH_ID
```

## Table of Contents

1. [Graph Construction](#1-graph-construction)
2. [Graph Retrieval](#2-graph-retrieval)
3. [Technical Architecture](#3-technical-architecture)
4. [Usage](#4-usage)
   - [Initialize system](#initialize-system)
   - [Building a graph](#building-a-graph)
   - [List available graphs](#list-available-graphs)
   - [Display graph statistics](#display-graph-statistics)
   - [Retrieval](#retrieval)
   - [Visualization](#visualization)
   - [Relabel topics with LLM](#relabel-topics-with-llm)
   - [Configuration management](#configuration-management)

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
- Chunks end at sentence boundaries (detected via regex: `(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$`)
- Overlap is calculated at sentence level (sentences from end of previous chunk start the next chunk)
- Preserves semantic coherence by never splitting mid-sentence
- Context is maintained across chunks through sentence-level overlap

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

**Keyword processing**: Themes use **intelligent keyword processing** to create clean, informative representations:

```python
# 1. Retrieve top 15 keywords from BERTopic
keywords_raw = topic_model.get_topic(topic_id)[:15]

# 2. Stem keywords using NLTK SnowballStemmer (merge morphological variants)
# Example: "presidents" → "presid", "running" → "run"
for word, score in keywords_raw:
    stemmed = stemmer.stem(word.lower())
    stem_scores[stemmed] += score  # Accumulate scores for same stem

# 3. Remove redundant n-grams
# Example: if "climate change" exists, remove "climate" and "change"
# Keeps only the most informative keyword form
sorted_stems = sorted(stem_scores.items(),
                     key=lambda x: (len(x[0].split()), x[1]),
                     reverse=True)

for stem, score in sorted_stems:
    is_subset = any(set(stem.split()).issubset(set(kept.split()))
                    for kept, _ in filtered)
    if not is_subset:
        filtered.append((stem, score))

# 4. Return top 10 deduplicated keywords
keywords = filtered[:10]
```

**Keyword processing benefits**:
- Removes morphological variants (e.g., "president" vs "presidents" both become "presid")
- Eliminates redundant n-gram components (e.g., keeps "climate change", removes "climate" and "change")
- Results in cleaner, more informative topic representations
- Reduces noise in topic labels

**Label generation**: Themes are labeled using **LLM-generated labels** (default, more descriptive) or **keyword-based labels** (faster):

```python
# Default: LLM-generated labels via Ollama
label = llm_generate_topic_label(keywords)  # e.g., "middle east conflict"

# Alternative: Keyword-based labels (use --no-llm-labels flag)
label = " ".join(keywords[:3])  # e.g., "gaza israel conflict"
```

**Node format**:
- Theme: `theme::{topic_id}`

Attributes:
- `topic_id`: BERTopic topic ID
- `label`: LLM-generated or keyword-based label
- `keywords`: list of processed keywords (top 10)

#### **Entities**

Entities are extracted via **spaCy NER** (Named Entity Recognition):

```python
# Entity extraction
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

entities = []
for ent in doc.ents:
    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'NORP']:
        entity_text = ent.text.strip().lower()

        # Special handling for DATEs: extract only years
        if ent.label_ == "DATE":
            year_match = re.search(r'\b(19|20)\d{2}\b', entity_text)
            if year_match:
                year = year_match.group(0)
                entities.append((year, "DATE"))
        else:
            if len(entity_text) > 2:  # Filter very short entities
                entities.append((entity_text, ent.label_))
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
- Reduces date nodes significantly while preserving temporal information
- Enables temporal similarity connections between years

**Node format**:
- Entity: `entity::{entity_type}::{entity_text}`

All entity text is **lowercased** for normalization.

---

### 1.2 Edge Construction

#### **CONTAINS**: Document → Chunk

Simple structural edge linking each document to its chunks.

```
Weight: 1.0 (unweighted)
```

---

#### **HAS_TOPIC**: Chunk → Theme

Links each chunk to its **top-3 most probable themes** detected by BERTopic.

**Weight calculation**:

```python
# BERTopic probability distribution for the chunk
topic_probs = topic_model.transform([chunk_text])[1][0]

# Get top-3 topic indices sorted by probability
top_k_topics = 3
top_k_indices = np.argsort(topic_probs)[-top_k_topics:][::-1]

# Create edges for top-K topics (skip if prob < 1%)
for topic_id in top_k_indices:
    prob = topic_probs[topic_id]
    if prob >= 0.01:  # Minimum threshold to filter noise
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

**Key features**:
- **Multi-label assignment**: Each chunk is assigned to its top-3 topics (not just the dominant one)
- **Weighted by probability**: Probabilities are preserved as edge weights
- **Enables CO_PRESENCE**: Multiple topics per chunk allow theme-theme co-occurrence edges
- **Noise filtering**: Topics with probability < 1% are excluded

---

#### **MENTIONS**: Chunk → Entity

Links chunks to entities they mention, weighted by **local normalized TF-IDF with sigmoid transformation**.

**Weight calculation**:

```python
# Step 1: Calculate TF for each entity-chunk pair
for chunk in chunks:
    for entity in chunk.entities:
        TF(entity, chunk) = count(entity in chunk)

# Step 2: Calculate IDF for each entity
for entity in all_entities:
    IDF(entity) = log(total_chunks / chunks_containing_entity)

# Step 3: Normalize TF locally within each chunk
for chunk in chunks:
    max_TF_in_chunk = max(TF values for all entities in this chunk)
    for entity in chunk.entities:
        normalized_TF = TF(entity, chunk) / max_TF_in_chunk

# Step 4: Calculate raw weight and apply sigmoid
scale = 3.0  # Sigmoid steepness parameter
for entity, chunk in entity_chunk_pairs:
    # Combine local importance (normalized TF) with global rarity (IDF)
    raw_weight = normalized_TF × IDF(entity)

    # Map to [0, 1] with sigmoid (soft boundaries)
    weight = 1.0 / (1.0 + exp(-raw_weight / scale))

    graph.add_edge(
        chunk_id,
        entity_node,
        edge_type="MENTIONS",
        weight=float(weight)  # Range: ≈[0, 1]
    )
```

**Formula**:
```
normalized_TF(entity, chunk) = TF(entity, chunk) / max_TF(chunk)
raw_weight = normalized_TF × log(N / df(entity))
weight_MENTIONS = σ(raw_weight / scale) = 1 / (1 + exp(-raw_weight / scale))
```

where:
- `N` = total number of chunks
- `df(entity)` = number of chunks containing the entity
- `max_TF(chunk)` = maximum TF among all entities in this specific chunk
- `scale = 3.0` = sigmoid steepness parameter

**Key features**:
- **Local TF normalization**: Each chunk's TF values are normalized relative to the most frequent entity in that chunk
- **Sigmoid transformation**: Soft mapping to [0, 1] (avoids hard clipping, smoother weight distribution)
- **Hybrid importance**: Combines local entity importance (within chunk) with global rarity (across corpus)

---

#### **CO_OCCURS**: Entity ↔ Entity

Links entities that appear together in the same documents, weighted by **PMI (Pointwise Mutual Information) with sigmoid transformation**.

**Weight calculation**:

```python
# Step 1: Count co-occurrences at document level (not chunk)
co_occur_count = number of documents containing (entity1 AND entity2)

# Step 2: Count entity document frequencies
freq1 = number of documents containing entity1
freq2 = number of documents containing entity2
total_docs = total number of documents

# Step 3: Calculate PMI (Pointwise Mutual Information)
# Measures how much more often entities co-occur than expected by chance
PMI = log((co_occur_count × total_docs) / (freq1 × freq2))

# Step 4: Apply sigmoid to map PMI to [0, 1]
scale = 2.0  # Sigmoid steepness parameter
weight_CO_OCCURS = 1.0 / (1.0 + exp(-PMI / scale))

# Only create edge if co_occur_count >= threshold (default: 2)
if co_occur_count >= threshold:
    graph.add_edge(
        entity1,
        entity2,
        edge_type="CO_OCCURS",
        weight=float(weight),
        co_occur_count=int(co_occur_count),
        pmi=float(PMI)
    )
```

**Formula**:
```
PMI(e1, e2) = log((count(e1, e2) × N) / (freq(e1) × freq(e2)))
weight_CO_OCCURS = σ(PMI / scale) = 1 / (1 + exp(-PMI / scale))
```

where:
- `N` = total number of documents
- `count(e1, e2)` = number of documents containing both entities
- `freq(e1)`, `freq(e2)` = number of documents containing each entity
- `scale = 2.0` = sigmoid steepness parameter

**Intuition**:
- **High PMI (positive)**: Entities co-occur much more than expected by chance → high weight
  - Example: "Trump" and "White House" appear together often relative to their individual frequencies
- **Low/negative PMI**: Entities co-occur randomly or less than expected → low weight
  - Example: Two common entities that just happen to appear in same documents by chance
- **Sigmoid transformation**: Smoothly maps PMI to [0, 1], handling both positive and negative PMI values
- **Advantages over count-based**: PMI is independent of corpus size and accounts for entity popularity

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

Links semantically similar entities or themes via their embeddings.

**Important**: SEMANTIC_SIM edges exist **only between entities** and **between themes**, **NOT between chunks**. Chunk similarity is computed on-the-fly during retrieval, not stored in the graph.

**For entities** (excluding DATE entities):

```python
# Embeddings of entity names (DATE entities excluded)
entity_texts = [entity_text for entities excluding DATE type]
embeddings = SentenceTransformer.encode(entity_texts, normalize_embeddings=True)

# Cosine similarity (dot product for normalized embeddings)
similarity_matrix = embeddings @ embeddings.T

# Edge if sim > threshold (default from config: 0.8)
for i, j in pairs:
    sim = similarity_matrix[i][j]
    if sim > semantic_similarity_threshold:
        graph.add_edge(
            entity_i,
            entity_j,
            edge_type="SEMANTIC_SIM",
            weight=float(sim)  # Range: [0.8, 1.0]
        )

# NOTE: DATE entities use TEMPORAL_DIST instead (see below)
```

**For themes**:

```python
# Representation = concatenation of top 5 processed keywords
theme_repr = " ".join(topic_keywords[:5])

# Embeddings then cosine similarity
theme_embeddings = SentenceTransformer.encode(theme_reprs, normalize_embeddings=True)
similarity_matrix = theme_embeddings @ theme_embeddings.T
```

**Formula**:
```
weight_SEMANTIC_SIM(n1, n2) = cos(emb(n1), emb(n2)) = emb(n1) · emb(n2)
```

(Since embeddings are normalized: `||emb|| = 1`)

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

            graph.add_edge(
                year_node1,
                year_node2,
                edge_type="TEMPORAL_DIST",
                weight=float(weight),
                year_diff=int(year_diff)
            )
```

**Formula**:
```
weight_TEMPORAL_DIST(y1, y2) = exp(-λ × |y1 - y2|)
```

where:
- `λ` = decay_rate (default from config: 0.15, slower decay)
- `|y1 - y2|` = absolute year difference

**Examples** (with decay_rate=0.15 from config):
- Same year (diff=0): weight = 1.0
- 1 year difference: weight ≈ 0.86
- 2 years difference: weight ≈ 0.74
- 3 years difference: weight ≈ 0.64
- 5 years difference: weight ≈ 0.47
- 10 years difference: weight ≈ 0.22

**Rationale**:
- Years closer in time are more related than distant years
- Exponential decay creates smooth temporal neighborhood
- Configurable max_year_gap limits connections to relevant temporal window

---

## 2. Graph Retrieval

The system uses **hybrid retrieval** combining 4 complementary strategies, fused by weighted average.

**Strategy organization**:
- **Strategies 1 & 4**: Direct comparison methods (no graph traversal)
  - Strategy 1: Semantic similarity via embeddings
  - Strategy 4: Topic distribution similarity via Jensen-Shannon divergence
- **Strategies 2 & 3**: Graph-based methods (with BFS traversal)
  - Strategy 2: Thematic retrieval via theme graph traversal
  - Strategy 3: Entity-based retrieval via entity graph traversal

### 2.1 Strategy 1: Semantic Retrieval

Based on embedding similarity between the claim and chunks.

**FAISS Embedding Cache**:
- Chunk embeddings are pre-computed during graph construction
- Stored in FAISS (Facebook AI Similarity Search) index for fast retrieval
- Saves to `outputs/graphs/embeddings_{graph_id}.faiss` + metadata pickle
- Eliminates redundant embedding computation during retrieval
- Supports fast similarity search via inner product (equivalent to cosine similarity with normalized vectors)

**Process**:

```python
# 1. Encode the claim
claim_embedding = SentenceTransformer.encode([claim], normalize_embeddings=True)[0]

# 2. Load chunk embeddings from FAISS cache (or recompute if missing)
if cache_exists:
    embedding_cache = EmbeddingCache()
    embedding_cache.load(cache_dir, graph_id)
    chunk_ids, chunk_embeddings = embedding_cache.get_all_embeddings()
else:
    # Fallback: compute on-the-fly
    chunk_embeddings = SentenceTransformer.encode(all_chunks, normalize_embeddings=True)

# 3. Compute cosine similarities (dot product for normalized embeddings)
similarities = chunk_embeddings @ claim_embedding

# 4. Min-max normalize to [0, 1]
score_sem(chunk) = (sim(chunk) - min_sim) / (max_sim - min_sim)
```

**Formula**:
```
score_sem(chunk) = normalize(cos(emb(claim), emb(chunk)))
```

**Output**: Dictionary `{chunk_id: score_sem}` with normalized scores in [0, 1].

---

### 2.2 Strategy 2: Thematic Retrieval

Based on BERTopic topics and relevance propagation through the theme graph.

**Process**:

```python
# 1. Classify the claim into topics
topic_probs = BERTopic.transform([claim])[1][0]

# 2. Select relevant themes (prob > threshold, default: 0.1)
seed_themes = [(f"theme::{topic_id}", prob)
               for topic_id, prob in enumerate(topic_probs)
               if prob > theme_threshold]

# 3. Propagate relevance through the theme graph (BFS)
theme_relevance = {theme: prob for theme, prob in seed_themes}
queue = deque(seed_themes)

while queue:
    current_theme, current_relevance = queue.popleft()

    # Explore neighbors via CO_PRESENCE and SEMANTIC_SIM edges
    for neighbor_theme in graph.neighbors(current_theme):
        if graph.nodes[neighbor_theme]['type'] != 'theme':
            continue

        edge_type = graph[current_theme][neighbor_theme]['edge_type']
        if edge_type not in ['CO_PRESENCE', 'SEMANTIC_SIM']:
            continue

        edge_weight = graph[current_theme][neighbor_theme]['weight']
        propagated_relevance = current_relevance × edge_weight

        # Only continue if above threshold (default: 0.1)
        if propagated_relevance >= relevance_threshold:
            # Update with max if already visited
            theme_relevance[neighbor_theme] = max(
                theme_relevance.get(neighbor_theme, 0),
                propagated_relevance
            )
            queue.append((neighbor_theme, propagated_relevance))

# 4. Score chunks via HAS_TOPIC edges
chunk_scores = defaultdict(float)
for theme, relevance in theme_relevance.items():
    for chunk in graph.neighbors(theme):
        if graph[chunk][theme]['edge_type'] == 'HAS_TOPIC':
            prob_weight = graph[chunk][theme]['weight']  # BERTopic probability
            chunk_scores[chunk] += relevance × prob_weight

# 5. Min-max normalize to [0, 1]
score_theme(chunk) = normalize(chunk_scores[chunk])
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

**Output**: Dictionary `{chunk_id: score_theme}` with normalized scores in [0, 1].

---

### 2.3 Strategy 3: Entity-based Retrieval

Based on entity extraction from the claim and relevance propagation through the entity graph.

**Process**:

```python
# 1. Extract entities from the claim (same NER as graph builder)
entities_claim = extract_entities(claim)

# 2. Find corresponding entities in the graph
seed_entities = [f"entity::{ent_type}::{ent_text}"
                 for ent_text, ent_type in entities_claim
                 if graph.has_node(entity_node)]

# 3. Propagate relevance through the entity graph (BFS)
entity_relevance = {entity: 1.0 for entity in seed_entities}
queue = deque([(entity, 1.0) for entity in seed_entities])

while queue:
    current_entity, current_relevance = queue.popleft()

    # Explore neighbors via CO_OCCURS and SEMANTIC_SIM edges
    for neighbor_entity in graph.neighbors(current_entity):
        if graph.nodes[neighbor_entity]['type'] != 'entity':
            continue

        edge_type = graph[current_entity][neighbor_entity]['edge_type']
        if edge_type not in ['CO_OCCURS', 'SEMANTIC_SIM']:
            continue

        edge_weight = graph[current_entity][neighbor_entity]['weight']
        propagated_relevance = current_relevance × edge_weight

        # Only continue if above threshold (default: 0.1)
        if propagated_relevance >= relevance_threshold:
            entity_relevance[neighbor_entity] = max(
                entity_relevance.get(neighbor_entity, 0),
                propagated_relevance
            )
            queue.append((neighbor_entity, propagated_relevance))

# 4. Score chunks via MENTIONS edges
chunk_scores = defaultdict(float)
for entity, relevance in entity_relevance.items():
    for chunk in graph.neighbors(entity):
        if graph[chunk][entity]['edge_type'] == 'MENTIONS':
            tf_idf_weight = graph[chunk][entity]['weight']  # Normalized TF-IDF
            chunk_scores[chunk] += relevance × tf_idf_weight

# 5. Min-max normalize to [0, 1]
score_entity(chunk) = normalize(chunk_scores[chunk])
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
score_entity(chunk) = normalize(∑_{entity} relevance(entity) × weight_MENTIONS(entity, chunk))
```

where `weight_MENTIONS` is the normalized TF-IDF from the MENTIONS edge.

**Output**: Dictionary `{chunk_id: score_entity}` with normalized scores in [0, 1].

---

### 2.4 Strategy 4: Topic Distribution Retrieval

Based on similarity between topic probability distributions (Jensen-Shannon divergence).

**Process**:

```python
# 1. Get the topic distribution of the claim
_, claim_topic_probs = BERTopic.transform([claim])
claim_distribution = claim_topic_probs[0]  # Shape: (n_topics,)

# 2. Get distributions for all chunks (pre-computed during initialization)
_, chunk_topic_probs = BERTopic.transform(chunk_texts)
chunk_distributions = chunk_topic_probs  # Shape: (n_chunks, n_topics)

# 3. Compute Jensen-Shannon divergence for each chunk
from scipy.spatial.distance import jensenshannon

chunk_scores = {}
for chunk_id, dist_idx in chunk_to_dist_idx.items():
    chunk_distribution = chunk_distributions[dist_idx]

    # JS divergence: symmetric, bounded in [0, 1]
    js_divergence = jensenshannon(claim_distribution, chunk_distribution)

    # Convert to similarity (1 - divergence)
    similarity = 1.0 - js_divergence
    chunk_scores[chunk_id] = float(similarity)

# 4. Min-max normalize to [0, 1]
score_topic_dist(chunk) = normalize(chunk_scores[chunk])
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
- `JS = 1` if distributions are completely different

**Output**: Dictionary `{chunk_id: score_topic_dist}` with normalized scores in [0, 1].

---

### 2.5 Score Fusion

The 4 scores are combined by **weighted average** with configurable weights.

**Fusion formula**:

```
score_final(chunk) = α × score_sem(chunk)
                   + β × score_theme(chunk)
                   + γ × score_entity(chunk)
                   + δ × score_topic_dist(chunk)
```

**Constraint**: `α + β + γ + δ = 1.0`

**Default weights**:
- `α = 0.3` (semantic)
- `β = 0.2` (thematic)
- `γ = 0.3` (entity)
- `δ = 0.2` (topic distribution)

**Fusion process**:

```python
# Union of all chunks found by the 4 strategies
all_chunks = set(S1) ∪ set(S2) ∪ set(S3) ∪ set(S4)

# For each chunk, compute final score
results = []
for chunk in all_chunks:
    score_sem = S1.get(chunk, 0)
    score_theme = S2.get(chunk, 0)
    score_entity = S3.get(chunk, 0)
    score_topic_dist = S4.get(chunk, 0)

    score_final = α×score_sem + β×score_theme + γ×score_entity + δ×score_topic_dist

    results.append({
        'chunk_id': chunk,
        'score_total': score_final,
        'score_sem': score_sem,
        'score_theme': score_theme,
        'score_entity': score_entity,
        'score_topic_dist': score_topic_dist
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

- **Embeddings**: `all-MiniLM-L6-v2` (SentenceTransformers, 384 dimensions)
- **NER**: `en_core_web_sm` (spaCy)
- **Topic Modeling**: BERTopic
- **Stemming**: NLTK SnowballStemmer (English)
- **LLM**: `llama3.1:latest` (Ollama, optional for topic labels)

### Storage

- **Graph**: NetworkX (pickle) + PostgreSQL (versioned tables)
- **Topic Model**: pickle (.pkl)
- **Embeddings**: FAISS index (.faiss) + metadata (.pkl)
- **Metadata**: JSON

### PostgreSQL table format

**Table `graph_{timestamp}`** (edges):
- `source`: source node
- `predicate`: edge type (CONTAINS, HAS_TOPIC, MENTIONS, etc.)
- `target`: target node
- `weight`: edge weight (float)
- `attributes`: JSONB with additional edge attributes

**Table `nodes_{timestamp}`** (nodes):
- `node_id`: unique identifier
- `node_type`: type (document, chunk, theme, entity)
- `attributes`: JSONB with all node attributes

**Table `graph_metadata`**:
- `graph_id`: creation timestamp
- `num_nodes`: number of nodes
- `num_edges`: number of edges
- `metadata`: JSONB with configuration and statistics

**Indexes**:
- `idx_graph_{timestamp}_source` on `source`
- `idx_graph_{timestamp}_target` on `target`
- `idx_graph_{timestamp}_predicate` on `predicate`
- `idx_graph_{timestamp}_source_target` on `(source, target)`
- `idx_nodes_{timestamp}_type` on `node_type`

### Configuration

**Three-layer configuration system**:

1. **config.yaml** (common settings):
   - Paths (data, outputs, logs)
   - Models (embedding, NER, LLM)
   - Database (PostgreSQL connection)
   - Logging

2. **config_build.yaml** (build-specific):
   - Graph construction parameters (chunk size, thresholds, etc.)
   - LLM settings for topic labeling
   - Export patterns

3. **config_retrieval.yaml** (retrieval-specific):
   - Retrieval weights (alpha, beta, gamma, delta)
   - Active graph selection
   - Thresholds for BFS propagation
   - Visualization settings

**Deep merge**: Specific configs override common config values.

---

## 4. Usage

### Building a graph

```bash
# Default build (uses LLM for topic labels - more descriptive)
uv run python cli.py build --max-docs 100

# Fast build with keyword-based topic labels (faster, less descriptive)
uv run python cli.py build --max-docs 100 --no-llm-labels

# With PostgreSQL storage
uv run python cli.py build --max-docs 100 --use-postgres

# Custom chunking parameters
uv run python cli.py build --max-docs 100 --chunk-size 500 --chunk-overlap 100
```

**Graph ID**: Each graph is identified by its creation timestamp (e.g., `20251118_155115`). All related files (visualizations, retrievals) use this ID.

**Outputs**:
- `outputs/graphs/graph_{graph_id}.gpickle` - NetworkX graph
- `outputs/graphs/topic_model_{graph_id}.pkl` - BERTopic model
- `outputs/graphs/embeddings_{graph_id}.faiss` - FAISS index
- `outputs/graphs/embeddings_{graph_id}.pkl` - FAISS metadata
- `outputs/graphs/metadata_{graph_id}.json` - Build metadata
- `outputs/statistics/general_stats_{graph_id}.json` - Node/edge counts
- `outputs/statistics/topic_stats_{graph_id}.json` - Topic details
- `outputs/statistics/entity_stats_{graph_id}.json` - Entity statistics
- `outputs/statistics/summary_{graph_id}.txt` - Human-readable summary

### List available graphs

```bash
uv run python cli.py list-graphs
```

Example output:
```
================================================================================
AVAILABLE GRAPHS
================================================================================

1. Graph ID: 20251118_190959
   Created: 2025-11-18T19:09:59
   Documents: 20, Chunks: 161
   Nodes: 762, Edges: 7042

2. Graph ID: 20251118_155115
   Created: 2025-11-18T15:51:15
   Documents: 1000, Chunks: 9270
   Nodes: 26272, Edges: 1510527
================================================================================
```

### Display graph statistics

```bash
# Show statistics for latest graph
uv run python cli.py stats

# Show statistics for specific graph
uv run python cli.py stats --graph-id 20251118_155115

# Show detailed statistics
uv run python cli.py stats --graph-id 20251118_155115 --detailed
```

**Output includes**:
- Total number of nodes and edges
- Breakdown by node type (document, chunk, theme, entity)
- Breakdown by edge type (CONTAINS, HAS_TOPIC, MENTIONS, etc.)
- Graph metadata (creation time, storage type)

### Retrieval

```bash
# Basic retrieval (use graph ID from list-graphs)
uv run python cli.py retrieve "Trump met Macron in France" --graph-id 20251118_155115

# Custom parameters with named retrieval
uv run python cli.py retrieve "claim text" \
  --graph-id 20251118_155115 \
  --top-k 10 \
  --alpha 0.3 --beta 0.2 --gamma 0.3 --delta 0.2 \
  --name "my_experiment"
```

**Retrieval results** are saved in `outputs/retrieval_results/`:
- Without `--name`: Auto-incremented (e.g., `results_{graph_id}_1.json`)
- With `--name "test"`: Custom name (e.g., `results_{graph_id}_test.json`)

**Result files include**:
- Top-ranked chunks with scores (semantic, thematic, entity, topic distribution)
- Alternative rankings (by semantic score, by entity score, etc.)
- Statistics (score distributions, component dominance)
- Full text of all retrieved chunks

### Visualization

```bash
# Display pre-computed statistics
uv run python cli.py visualize --mode stats --graph-id 20251118_155115

# Full graph visualization (interactive HTML)
uv run python cli.py visualize --mode full --graph-id 20251118_155115

# Node-centered subgraph (any node type: entity, chunk, theme, document)
uv run python cli.py visualize --mode node --node "entity::PERSON::trump" --graph-id 20251118_155115
uv run python cli.py visualize --mode node --node "chunk::0::5" --graph-id 20251118_155115
uv run python cli.py visualize --mode node --node "theme::2" --graph-id 20251118_155115
uv run python cli.py visualize --mode node --node "doc::0" --graph-id 20251118_155115
```

**Visualization files** are saved as `graph_{mode}_{node_id}_{graph_id}.html` in `outputs/visualizations/`

**Visualization features**:
- Interactive legend: Click node/edge types to hide/show
- Hover nodes to see details
- Click chunks to see full text in popup
- Click documents to see all chunks with separators
- Normalized edge weights displayed
- Physics simulation for layout
- Color-coded by node/edge type

### Relabel topics with LLM

If you initially built the graph with `--no-llm-labels` (keyword-based labels), you can retroactively generate more descriptive LLM labels:

```bash
# Relabel topics with LLM (default model from config)
uv run python cli.py relabel

# Use specific LLM model
uv run python cli.py relabel --llm-model llama3.1:latest

# With PostgreSQL storage
uv run python cli.py relabel --use-postgres
```

**Note**: This updates the topic labels in the graph and topic model files, replacing keyword-based labels with LLM-generated descriptions.

### Configuration management

```bash
# Show current configuration
uv run python cli.py config --show

# Validate configuration files
uv run python cli.py config --validate
```

The system uses a three-layer configuration system (see section 3 for details).

### Initialize system

Before first use, initialize the system to create necessary directories and optionally set up PostgreSQL:

```bash
# Create directory structure
uv run python cli.py init --dirs

# Initialize PostgreSQL database
uv run python cli.py init --postgres

# Do both
uv run python cli.py init --dirs --postgres
```

**Created directories**:
- `logs/` - Log files
- `outputs/` - All output files
- `outputs/graphs/` - Graph files (.gpickle, .pkl, .faiss)
- `outputs/visualizations/` - HTML visualizations
- `outputs/statistics/` - Statistics JSON files
- `outputs/retrieval_results/` - Retrieval result JSON files
- `outputs/exports/` - Exported graph files
- `models/` - Downloaded models (optional)

---

## 5. Installation

### Prerequisites

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

### Install uv

If you don't have `uv` installed yet:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Clone and Setup

1. **Clone the repository**:

```bash
git clone <repository-url>
cd PRL
```

2. **Initialize uv project** (creates virtual environment automatically):

```bash
uv init
```

3. **Install all dependencies**:

```bash
uv pip install -r requirements.txt
```

4. **Download spaCy language model**:

```bash
uv run python -m spacy download en_core_web_sm
```

5. **Install Ollama (for LLM-generated topic labels)**:

Ollama is required only if you want to use LLM-generated topic labels (more descriptive). If you use `--no-llm-labels` flag, you can skip this step.

```bash
# On macOS and Linux
curl -fsSL https://ollama.com/install.sh | sh

# On Windows
# Download and install from: https://ollama.com/download/windows
```

6. **Download the LLM model**:

```bash
# Pull the llama3.1 model (default model used by the system)
ollama pull llama3.1:latest

# Start Ollama service (if not already running)
ollama serve
```

**Verify Ollama installation**:
```bash
# Check if Ollama is running
ollama list

# Test the model
ollama run llama3.1:latest "Hello, world!"
```

**Note**: If you want to use a different model, update the `llm_model` setting in `config/config_build.yaml`:
```yaml
models:
  llm_model: "llama3.1:latest"  # Change to your preferred model
  llm_endpoint: "http://localhost:11434"
```

### Configuration

The system uses YAML configuration files in the `config/` directory:

- `config/config.yaml` - Common settings (paths, database, models)
- `config/config_build.yaml` - Build-specific settings (topic modeling, entity types)
- `config/config_retrieval.yaml` - Retrieval-specific settings (weights, thresholds)

**Optional: Setup PostgreSQL**

If you want to use PostgreSQL storage:

```bash
# Initialize database tables
uv run python cli.py init --postgres

# Or manually configure connection in config/config.yaml
database:
  host: localhost
  port: 5432
  database: knowledge_graph
  user: postgres
  password: your_password
```

### Quick Start

Once installed, build your first graph:

```bash
# Build a small test graph
uv run python cli.py build --max-docs 20 --no-llm-labels

# List available graphs
uv run python cli.py list-graphs

# Try a retrieval query
uv run python cli.py retrieve "your query here" --graph-id <graph_id_from_list>
```

---
