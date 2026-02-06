# Knowledge Graph Build and Retrieval Documentation

This document provides a detailed technical description of the knowledge graph construction pipeline and the multi-strategy retrieval system.

## Table of Contents

1. [Graph Build Pipeline](#graph-build-pipeline)
2. [Node Types](#node-types)
3. [Edge Types](#edge-types)
4. [Retrieval Strategies](#retrieval-strategies)
5. [Mathematical Formulas](#mathematical-formulas)

---

## Graph Build Pipeline

The knowledge graph is constructed through a 10-step pipeline:

### Step 1: Documents and Chunks

**Module:** `graph_nodes_chunk_docs.py`

Creates the base hierarchical structure:
- **Document nodes** (`doc::{doc_id}`): Source fact-check articles
- **Chunk nodes** (`chunk::{doc_id}::{chunk_idx}`): Text segments using sliding window

**CONTAINS edges**: Document → Chunk (structural, no weight)

### Step 2: Topic Modeling

**Module:** `graph_builder.py` → `build_topic_model()`

Uses BERTopic with the following pipeline:
1. **Embedding**: SentenceTransformer encodes all chunks
2. **Dimensionality Reduction**: UMAP reduces to 5 dimensions
3. **Clustering**: HDBSCAN finds topic clusters
4. **Representation**: c-TF-IDF creates topic word distributions

```
UMAP: n_neighbors=15, n_components=5, metric='cosine'
HDBSCAN: min_cluster_size=10, metric='euclidean', method='eom'
```

### Step 3: Themes and HAS_TOPIC Edges

**Module:** `graph_nodes_theme.py`

Creates theme nodes with LLM-generated labels:
1. Extract top keywords per topic (c-TF-IDF)
2. Process keywords: dedup, lemmatize, filter
3. Generate label via LLM (Ollama/Llama)
4. Create top-3 HAS_TOPIC edges per chunk

**HAS_TOPIC edge weight** = BERTopic probability ∈ [0, 1]

### Step 3b: Hierarchical Topics (Supertopics)

**Module:** `graph_nodes_theme.py` → `build_hierarchical_topics()`

Creates supertopics by merging semantically similar topics:

```
merge_distance(t_i, t_j) = 1 - cos(c-TF-IDF(t_i), c-TF-IDF(t_j))
```

**CONTAINS_TOPIC edge weight** = 1 - merge_distance = c-TF-IDF similarity

### Step 4: Entities and MENTIONS Edges

**Module:** `graph_nodes_entity.py`

Extracts named entities using spaCy NER:
- Types: PERSON, ORG, GPE, LOC, DATE, EVENT, NORP
- DATE normalization: Extract 4-digit years only

**MENTIONS edge weight (TF-IDF with Sigmoid):**

```
TF_normalized = tf / max_tf_in_chunk
IDF = log(N / df)
raw_weight = TF_normalized × IDF
weight = σ(raw_weight / scale)
```

Where:
- `tf` = term frequency in chunk
- `df` = document frequency (chunks containing entity)
- `N` = total number of chunks
- `σ(x) = 1 / (1 + exp(-x))` (sigmoid)
- `scale = 3.0`

### Step 4b: Keyphrases

**Module:** `graph_nodes_chunk_docs.py` → `add_keyphrases()`

Uses KeyBERT with Maximal Marginal Relevance (MMR):

```
MMR = λ × sim(d, q) - (1-λ) × max(sim(d, d_j))
```

Where:
- `sim(d, q)` = cosine similarity between candidate and document
- `sim(d, d_j)` = similarity to already selected keyphrases
- `λ = 0.5` (diversity parameter)

**HAS_KEYPHRASE edge weight** = KeyBERT relevance score ∈ [0, 1]

### Step 5: Entity CO_OCCURS Edges

**Module:** `graph_nodes_entity.py` → `add_entity_co_occurrences()`

Computes PMI-based co-occurrence weights at document level:

```
PMI = log((co_occur × N) / (freq_1 × freq_2))
weight = σ(PMI / scale)
```

Where:
- `co_occur` = number of documents where both entities appear
- `freq_1, freq_2` = document frequency of each entity
- `N` = total documents
- `scale = 2.0`

**Note:** DATE-DATE pairs are skipped (use TEMPORAL_DIST instead)

### Step 6: Entity SEMANTIC_SIM Edges

**Module:** `graph_nodes_entity.py` → `add_entity_semantic_similarity()`

Computes embedding similarity between non-DATE entities:

```
weight = cos(embed(e_1), embed(e_2))
```

Only created if `weight >= semantic_sim_threshold` (default 0.6)

### Step 6b: DATE TEMPORAL_DIST Edges

**Module:** `graph_nodes_entity.py` → `add_date_temporal_similarity()`

Exponential decay based on year difference:

```
weight = exp(-λ × |year_1 - year_2|)
```

Where:
- `λ = 0.3` (decay_rate)
- Only created if `|year_1 - year_2| <= max_year_gap` (default 10)

Examples with λ=0.3:
| Year Difference | Weight |
|-----------------|--------|
| 0 | 1.000 |
| 1 | 0.741 |
| 2 | 0.549 |
| 3 | 0.407 |
| 5 | 0.223 |
| 10 | 0.050 |

### Step 7: Theme SEMANTIC_SIM Edges

**Module:** `graph_nodes_theme.py` → `add_theme_semantic_similarity()`

Computes embedding similarity between themes using keyword representations:

```
representation(theme) = concat(top_5_keywords)
weight = cos(embed(rep_1), embed(rep_2))
```

Only created if `weight >= semantic_sim_threshold`

### Step 8: Theme CO_PRESENCE Edges

**Module:** `graph_nodes_theme.py` → `add_theme_co_presence()`

Counts themes co-occurring in the same chunks:

```
weight = co_presence_count × (1 / √(freq_1 × freq_2))
```

Where:
- `co_presence_count` = number of chunks with both themes
- `freq_1, freq_2` = number of chunks each theme appears in

### Step 9: Claims and REVIEWS Edges

**Module:** `graph_nodes_claims.py`

Creates claim nodes and links them to fact-check documents:
- **Claim nodes** (`claim::{claim_id}`): Fact-checked claims with ratings
- **REVIEWS edges**: Claim → Document (links claim to its fact-check)

### Step 10: Pre-compute Claim Retrieval

**Module:** `graph_builder.py` → `_precompute_claim_retrieval()`

For each claim, pre-computes retrieval scores using strategies S1-S4:

```
precomputed_score = 0.25×S1 + 0.25×S2 + 0.25×S3 + 0.25×S4
```

Results are stored for efficient claim-based retrieval at query time.

---

## Node Types

| Type | ID Format | Key Attributes |
|------|-----------|----------------|
| document | `doc::{doc_id}` | text, review_date, language |
| chunk | `chunk::{doc_id}::{idx}` | text, chunk_idx |
| theme | `theme::{topic_id}` | label, keywords, supertopic_id |
| supertopic | `supertopic::{id}` | label, keywords, child_topics |
| entity | `entity::{type}::{text}` | entity_type, entity_text |
| keyphrase | `keyphrase::{phrase}` | text, phrase |
| claim | `claim::{claim_id}` | text, rating, claimant |

---

## Edge Types

| Edge Type | Source → Target | Weight Formula |
|-----------|-----------------|----------------|
| CONTAINS | Document → Chunk | 1.0 (structural) |
| HAS_TOPIC | Chunk → Theme | BERTopic probability |
| CONTAINS_TOPIC | Supertopic → Theme | 1 - merge_distance |
| MENTIONS | Chunk → Entity | σ(TF×IDF / scale) |
| HAS_KEYPHRASE | Chunk → Keyphrase | KeyBERT score |
| CO_OCCURS | Entity ↔ Entity | σ(PMI / scale) |
| SEMANTIC_SIM | Entity ↔ Entity | cos(embed_1, embed_2) |
| SEMANTIC_SIM | Theme ↔ Theme | cos(embed_1, embed_2) |
| TEMPORAL_DIST | DATE ↔ DATE | exp(-λ × year_diff) |
| CO_PRESENCE | Theme ↔ Theme | count × inv_freq |
| REVIEWS | Claim → Document | 1.0 (structural) |

---

## Retrieval Strategies

### Strategy 1: Semantic Similarity (S1)

**Module:** `retrieval_semantic.py`

Direct embedding similarity between query and chunk embeddings:

```
S1(chunk) = cos(embed(query), embed(chunk))
```

**Normalization:** Min-max scaling to [0, 1]

### Strategy 2: Topic Distribution Similarity (S2)

**Module:** `retrieval_topic.py`

Jensen-Shannon divergence between topic distributions:

```
claim_dist = BERTopic.transform(query)
JS_divergence = JSD(claim_dist, chunk_dist)
S2(chunk) = 1 - JS_divergence
```

Jensen-Shannon divergence is symmetric and bounded [0, 1]:

```
JSD(P, Q) = 0.5 × KL(P || M) + 0.5 × KL(Q || M)
M = 0.5 × (P + Q)
```

**Normalization:** Min-max scaling to [0, 1]

### Strategy 3: Thematic Graph Traversal (S3)

**Module:** `retrieval_thematic.py`

Uses BFS propagation through theme graph:

1. **Detect themes** in query using BERTopic (probability > threshold)
2. **Propagate relevance** through CO_PRESENCE and SEMANTIC_SIM edges:
   ```
   propagated_relevance = current_relevance × edge_weight
   ```
3. **Score chunks** based on theme relevance:
   ```
   S3(chunk) = Σ(theme_relevance × HAS_TOPIC_weight)
   ```

### Strategy 4: Entity Graph Traversal (S4)

**Module:** `retrieval_entity.py`

Uses BFS propagation through entity graph:

1. **Extract entities** from query using spaCy NER
2. **Propagate relevance** through CO_OCCURS and SEMANTIC_SIM edges
3. **Score chunks** based on entity relevance:
   ```
   S4(chunk) = Σ(entity_relevance × MENTIONS_weight)
   ```

### Strategy 5: Claim-Based Retrieval (S5)

**Module:** `retrieval_claims.py`

Leverages pre-existing fact-checked claims:

1. **Find similar claims**: `similarity = cos(embed(query), embed(claim))`
2. **Get pre-computed results** for matched claims
3. **Combine scores**:
   ```
   S5(chunk) = max(claim_similarity × precomputed_score(claim, chunk))
   ```

### Strategy 6: Claim-Based Topic Distribution Traversal (S6)

**Module:** `retrieval_claim_topic.py`

Combines claim-based retrieval with topic distribution analysis. Uses matched claims
to traverse the graph and scores chunks based on topic distribution similarity
weighted by traversal relevance.

**Algorithm:**
1. **Find similar claims** using embedding cosine similarity (top K)
2. **Get chunks** connected to claims via pre-computed graph traversal
3. **Compute topic distribution similarity** between query and each chunk using Jensen-Shannon divergence
4. **Weight by relevance propagation** from claim to chunk (from pre-computed S5)
5. **Boost chunks with high HAS_TOPIC confidence** (TF-IDF-like weighting)

**Score Formula:**
```
S6(chunk) = claim_sim × traversal_rel × topic_sim × topic_conf
```

Where:
- `claim_sim` = cos(embed(query), embed(claim))
- `traversal_rel` = pre-computed relevance score from claim to chunk (from S5)
- `topic_sim` = 1 - JSD(query_distribution, chunk_distribution)
- `topic_conf` = max(HAS_TOPIC_weight) for chunk's topics

### Strategy 7: Claim-Based KeyBERT Phrase Similarity (S7)

**Module:** `retrieval_claim_keyphrase.py`

Extracts keyphrases from the query using KeyBERT and computes semantic similarity
with chunk keyphrases found through claim-based retrieval.

**Algorithm:**
1. **Find similar claims** using embedding cosine similarity (top K)
2. **Extract query keyphrases** using KeyBERT with MMR diversity
3. **Get chunks** connected to claims via pre-computed graph traversal
4. **Compute keyphrase similarity** between query keyphrases and chunk keyphrases
5. **Weight by keyphrase importance** (HAS_KEYPHRASE edge weight)
6. **Weight by traversal relevance** from claim to chunk

**Score Formula:**
```
S7(chunk) = claim_sim × traversal_rel × keyphrase_sim × keyphrase_importance
```

Where:
- `claim_sim` = cos(embed(query), embed(claim))
- `traversal_rel` = pre-computed relevance score from claim to chunk (from S5)
- `keyphrase_sim` = max(cos(embed(query_kp), embed(chunk_kp))) for all keyphrase pairs
- `keyphrase_importance` = max(HAS_KEYPHRASE_weight) for matched keyphrases

---

## Mathematical Formulas

### TF-IDF with Sigmoid Normalization

```
TF = count(entity, chunk)
TF_norm = TF / max(TF in chunk)
IDF = log(N / DF)
raw = TF_norm × IDF
weight = 1 / (1 + exp(-raw / scale))
```

### Pointwise Mutual Information (PMI)

```
PMI(e1, e2) = log(P(e1, e2) / (P(e1) × P(e2)))
            = log((co_occur × N) / (freq1 × freq2))
```

### Jensen-Shannon Divergence

```
JSD(P, Q) = 0.5 × D_KL(P || M) + 0.5 × D_KL(Q || M)
M = (P + Q) / 2
D_KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

### Exponential Decay (Temporal)

```
weight = exp(-λ × Δt)
```

### BFS Relevance Propagation

```
relevance(n) = max over paths p: ∏(w_e for e in p)
```

Where `w_e` is the edge weight and the product is over all edges in path `p`.

### Combined Retrieval Score

```
final_score = α×S1 + β×S2 + γ×S3 + δ×S4 + ε×S5 + ζ×S6 + η×S7
```

Where:
- `α (alpha)` = weight for semantic similarity (S1)
- `β (beta)` = weight for topic distribution similarity (S2)
- `γ (gamma)` = weight for thematic graph traversal (S3)
- `δ (delta)` = weight for entity graph traversal (S4)
- `ε (epsilon)` = weight for claim-document retrieval (S5)
- `ζ (zeta)` = weight for claim-topic distribution traversal (S6)
- `η (eta)` = weight for claim-keyphrase similarity (S7)

All weights should sum to 1.0.

Default weights (5-strategy): `α = β = γ = δ = ε = 0.20`, `ζ = η = 0.0`

Example 7-strategy weights: `α = β = γ = δ = ε = 0.15`, `ζ = η = 0.125`

---

## Configuration Parameters

### Build Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `semantic_sim_threshold` | 0.6 | Min similarity for SEMANTIC_SIM edges |
| `co_occur_threshold` | 2 | Min co-occurrences for CO_OCCURS edges |
| `temporal_decay_rate` | 0.3 | λ in exponential decay |
| `temporal_max_year_gap` | 10 | Max years for TEMPORAL_DIST |

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theme_threshold` | 0.01 | Min probability for theme detection |
| `entity_relevance_threshold` | 0.1 | Min relevance for entity BFS propagation |
| `theme_relevance_threshold` | 0.1 | Min relevance for theme BFS propagation |
| `claim_top_k` | 5 | Number of top claims for S5, S6, S7 |
| `keyphrase_top_n` | 5 | Number of keyphrases to extract for S7 |
| `top_k` | 10 | Number of chunks to return |

### Strategy Weights

| Weight | Parameter | Default | Strategy |
|--------|-----------|---------|----------|
| α | `alpha` | 0.20 | Semantic Similarity (S1) |
| β | `beta` | 0.20 | Topic Distribution (S2) |
| γ | `gamma` | 0.20 | Thematic Traversal (S3) |
| δ | `delta` | 0.20 | Entity Traversal (S4) |
| ε | `epsilon` | 0.20 | Claim-Document (S5) |
| ζ | `zeta` | 0.00 | Claim-Topic (S6) |
| η | `eta` | 0.00 | Claim-KeyPhrase (S7) |
