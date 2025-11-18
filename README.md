# Knowledge Graph for Fact-Checking

Fact-checking system based on a multi-level knowledge graph combining topic modeling (BERTopic), entity extraction (spaCy), and hybrid retrieval.

---

## 1. Graph Construction

The knowledge graph is built in several steps, creating a multi-level network linking documents, chunks, themes, and entities.

### 1.1 Node Construction

#### **Documents and Chunks**

Documents are split into chunks with overlap to preserve context:

```python
# Chunking with overlap
chunk_size = 500  # characters
chunk_overlap = 100  # characters

chunks = chunk_text(document, chunk_size=500, overlap=100)
```

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

**Label generation**: Themes are labeled by an LLM (Ollama) based on extracted keywords:

```python
# Top 5 keywords of the topic
keywords = topic_model.get_topic(topic_id)[:5]
# LLM prompt to generate concise label
label = llm_generate_topic_label(keywords)
```

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
        entities.append((ent.text.lower(), ent.label_))
```

**Extracted entity types**:
- `PERSON`: people (real or fictional)
- `ORG`: organizations, companies
- `GPE`: geopolitical entities (countries, cities)
- `LOC`: non-GPE locations (mountains, oceans)
- `DATE`: dates and time periods
- `EVENT`: named events (wars, disasters)
- `NORP`: nationalities, religious or political groups

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

#### **SEMANTIC_SIM**: Entity ↔ Entity / Theme ↔ Theme

Links semantically similar entities (or themes) via their embeddings.

**For entities**:

```python
# Embeddings of entity names
embeddings = SentenceTransformer.encode([entity_text1, entity_text2, ...])

# Cosine similarity
sim = cosine_similarity(embedding1, embedding2) = embedding1 · embedding2

# Edge if sim > threshold (default: 0.6)
if sim > semantic_similarity_threshold:
    weight_SEMANTIC_SIM = sim
```

**For themes**:

```python
# Representation = concatenation of top 5 keywords
theme_repr = " ".join(topic_keywords[:5])

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

## 2. Graph Retrieval

The system uses **hybrid retrieval** combining 4 complementary strategies, fused by weighted average.

### 2.1 Strategy 1: Semantic Retrieval

Based on embedding similarity between the claim and chunks.

**Process**:

```python
# 1. Encode the claim
claim_embedding = SentenceTransformer.encode(claim)

# 2. Encode all chunks (pre-computed)
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

### 2.2 Strategy 2: Thematic Retrieval

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

### 2.3 Strategy 3: Entity-based Retrieval

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

### 2.4 Strategy 4: Topic Distribution Retrieval

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
- `γ = 0.3` (entities)
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

- **Embeddings**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **NER**: `en_core_web_sm` (spaCy)
- **Topic Modeling**: BERTopic
- **LLM**: `llama3.1:latest` (Ollama) for topic labels

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

### Building a graph

```bash
uv run python cli.py build --max-docs 100
```

### Retrieval

```bash
uv run python cli.py retrieve "Trump met Macron in France" --top-k 10 --graph-id 20251118_150602
```

### Visualization

```bash
# Full visualization
uv run python cli.py visualize --mode full --graph-id 20251118_150602

# Entity-centered subgraph
uv run python cli.py visualize --mode entity --entity "trump" --graph-id 20251118_150602
```

---

## 5. Configuration

See [config.yaml](config.yaml) for all parameters:

```yaml
graph_construction:
  topic_probability_threshold: 0.15
  co_occurrence_threshold: 2
  semantic_similarity_threshold: 0.6

retrieval:
  alpha: 0.3  # semantic weight
  beta: 0.2   # thematic weight
  gamma: 0.3  # entity weight
  delta: 0.2  # topic distribution weight
  entity_relevance_threshold: 0.1
  theme_relevance_threshold: 0.1
```

---

## 6. References

- **BERTopic**: [Grootendorst, 2022](https://github.com/MaartenGr/BERTopic)
- **SentenceTransformers**: [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)
- **spaCy**: [Honnibal & Montani, 2017](https://spacy.io)
- **Jensen-Shannon Divergence**: [Lin, 1991](https://ieeexplore.ieee.org/document/61115)
