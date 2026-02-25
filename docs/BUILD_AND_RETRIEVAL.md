# Knowledge Graph Build and Retrieval Documentation

This document provides a detailed technical description of the knowledge graph construction pipeline and the multi-strategy retrieval system.

## Table of Contents

1. [Graph Build Pipeline](#graph-build-pipeline)
2. [Node Types](#node-types)
3. [Edge Types](#edge-types)
4. [Retrieval Strategies](#retrieval-strategies)
5. [Document-Level Aggregation](#document-level-aggregation)
6. [Mathematical Formulas](#mathematical-formulas)
7. [Configuration Parameters](#configuration-parameters)
8. [Evaluation](#evaluation)

---

## Graph Build Pipeline

The knowledge graph is constructed through a **9-step pipeline**

### Step 1: Documents and Chunks

**Module:** `graph_nodes_chunk_docs.py`

Creates the base hierarchical structure:
- **Document nodes** (`doc::{doc_id}`): Source fact-check articles
- **Chunk nodes** (`chunk::{doc_id}::{chunk_idx}`): Text segments using sliding window

**CONTAINS edges**: Document → Chunk (structural, weight 1.0)

### Step 2: Topic Modeling

**Module:** `graph_builder.py`

Uses BERTopic with the following pipeline:
1. **Embedding**: SentenceTransformer encodes all chunks (GPU-accelerated, normalized)
2. **Dimensionality Reduction**: UMAP reduces to 5 dimensions
3. **Clustering**: HDBSCAN finds topic clusters
4. **Representation**: c-TF-IDF creates topic word distributions

```
UMAP:    n_neighbors=15, n_components=5, metric='cosine'
HDBSCAN: min_cluster_size=min_topic_size, metric='euclidean', method='eom'
```

### Step 3: Themes and HAS_TOPIC Edges

**Module:** `graph_nodes_theme.py`

Creates theme nodes with LLM-generated labels (Ollama/Llama):
1. Extract top keywords per topic (c-TF-IDF)
2. Process keywords: dedup, lemmatize, filter stopwords
3. Generate label via LLM
4. Create **top-3 HAS_TOPIC edges** per chunk (highest probability topics)

**HAS_TOPIC edge weight** = BERTopic topic probability $\in [0, 1]$
Only created if probability $\geq$ `topic_probability_threshold` (default 0.15).

### Step 3b: Hierarchical Topics (Supertopics)

**Module:** `graph_nodes_theme.py` → `build_hierarchical_topics()`

Creates supertopics by merging semantically similar topics via agglomerative clustering:

$$d_\text{merge}(t_i, t_j) = 1 - \cos\!\bigl(\text{c-TF-IDF}(t_i),\ \text{c-TF-IDF}(t_j)\bigr)$$

**CONTAINS_TOPIC edge weight** = $1 - d_\text{merge} = \cos(\text{c-TF-IDF}(t_i),\ \text{c-TF-IDF}(t_j))$

### Step 4: Entities and MENTIONS Edges

**Module:** `graph_nodes_entity.py`

Extracts named entities using spaCy NER:
- Types: PERSON, ORG, GPE, LOC, DATE, EVENT, NORP
- DATE normalization: extract 4-digit years only
- Filter: entities must be > 2 characters

**MENTIONS edge weight (TF-IDF with Sigmoid):**

$$\text{TF}_\text{norm} = \frac{tf}{\max\_tf_\text{chunk}}, \qquad \text{IDF} = \log\!\left(\frac{N}{df}\right), \qquad w = \sigma\!\left(\frac{\text{TF}_\text{norm} \times \text{IDF}}{3.0}\right)$$

where $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ maps to $(0, 1)$.

### Step 4b: Keyphrases and HAS_KEYPHRASE Edges

**Module:** `graph_nodes_chunk_docs.py` → `add_keyphrases()`

Uses KeyBERT with Maximal Marginal Relevance (MMR):

$$\text{MMR} = \lambda \cdot \text{sim}(\text{candidate}, \text{doc}) - (1-\lambda) \cdot \max_{\text{sel}}\,\text{sim}(\text{candidate}, \text{selected})$$

where $\lambda = 0.5$ (diversity parameter). Top-`keyphrase_top_n` (default 5) keyphrases per chunk.

**HAS_KEYPHRASE edge weight** = KeyBERT relevance score $\in [0, 1]$

### Step 5: Entity CO_OCCURS Edges

**Module:** `graph_nodes_entity.py` → `add_entity_co_occurrences()`

Computes PMI-based co-occurrence weights at **document** level:

$$\text{PMI}(e_1, e_2) = \log\!\left(\frac{c_{12} \times N}{f_1 \times f_2}\right), \qquad w = \sigma\!\left(\frac{\text{PMI}}{2.0}\right)$$

where:
- $c_{12}$ = documents where both entities appear together
- $f_1, f_2$ = document frequency of each entity
- $N$ = total documents
- Sigmoid scale = 2.0

**Note:** DATE–DATE pairs are skipped (use TEMPORAL_DIST instead).
Only created if $c_{12} \geq$ `co_occurrence_threshold` (default 2).

### Step 6: Entity SEMANTIC_SIM Edges

**Module:** `graph_nodes_entity.py` → `add_entity_semantic_similarity()`

Embedding similarity between non-DATE entities:

$$w = \cos\!\bigl(\text{embed}(e_1),\ \text{embed}(e_2)\bigr)$$

Only created if $w \geq$ `semantic_similarity_threshold` (default 0.8).

### Step 6b: DATE TEMPORAL_DIST Edges

**Module:** `graph_nodes_entity.py` → `add_date_temporal_similarity()`

Exponential decay based on year difference:

$$w = \exp\!\bigl(-\lambda \cdot |y_1 - y_2|\bigr)$$

where $\lambda = 0.15$ (default) and only created if $|y_1 - y_2| \leq$ `max_year_gap` (default 10).

| Year Difference | Weight ($\lambda=0.15$) |
|-----------------|------------------------|
| 0  | 1.000 |
| 1  | 0.861 |
| 2  | 0.741 |
| 3  | 0.638 |
| 5  | 0.472 |
| 10 | 0.223 |

### Step 7: Theme SEMANTIC_SIM Edges

**Module:** `graph_nodes_theme.py` → `add_theme_semantic_similarity()`

Embedding similarity between themes using keyword representations:

$$\text{rep}(t) = \text{concat}(\text{top\_5\_keywords}), \qquad w = \cos\!\bigl(\text{embed}(\text{rep}_1),\ \text{embed}(\text{rep}_2)\bigr)$$

Only created if $w \geq$ `semantic_similarity_threshold` (default 0.8).

### Step 8: Theme CO_PRESENCE Edges

**Module:** `graph_nodes_theme.py` → `add_theme_co_presence()`

Counts themes co-occurring in the same chunks with inverse frequency weighting:

$$w = \frac{c_{12}}{\sqrt{f_1 \times f_2}}$$

where:
- $c_{12}$ = chunks where both themes appear
- $f_1, f_2$ = number of chunks each theme appears in

**Proof of bounds**: $c_{12} \leq \min(f_1, f_2) \leq \sqrt{f_1 \cdot f_2}$ (AM-GM), so $w \in [0, 1]$.

### Step 9: Claims, REVIEWS Edges, and CLAIM_SIM Edges

**Module:** `graph_nodes_claims.py`

Creates claim nodes and links them to fact-check documents:
- **Claim nodes** (`claim::{claim_id}`): Fact-checked claims with ratings and claimant info
- **REVIEWS edges**: Claim → Document (structural, links claim to its reviewed document)
- **CLAIM_SIM edges**: Claim ↔ Claim (semantic similarity between claims)

**CLAIM_SIM edge weight:**

$$w = \cos\!\bigl(\text{embed}(c_1),\ \text{embed}(c_2)\bigr) \in [0, 1]$$

Only created if $w \geq$ `claim_sim_threshold` (default 0.5).
Only the top-k most similar neighbors per claim are kept to bound edge count.

---

## Node Types

| Type | ID Format | Key Attributes |
|------|-----------|----------------|
| document | `doc::{doc_id}` | doc_id, text, review_date, language |
| chunk | `chunk::{doc_id}::{idx}` | doc_id, text, chunk_idx, review_date, language |
| theme | `theme::{topic_id}` | label, keywords, supertopic_id |
| supertopic | `supertopic::{id}` | label, keywords, child_topics |
| entity | `entity::{type}::{text}` | entity_type, entity_text |
| keyphrase | `keyphrase::{phrase}` | text, phrase |
| claim | `claim::{claim_id}` | text, rating, claimant |

---

## Edge Types

| Edge Type | Source → Target | Weight Formula | Bounds |
|-----------|-----------------|----------------|--------|
| CONTAINS | Document → Chunk | $1.0$ (structural) | $\{1\}$ |
| HAS_TOPIC | Chunk → Theme | BERTopic probability | $[0, 1]$ |
| CONTAINS_TOPIC | Supertopic → Theme | $1 - d_\text{merge}$ | $[0, 1]$ |
| MENTIONS | Chunk ← Entity | $\sigma\!\left(\frac{\text{TF}_\text{norm} \cdot \text{IDF}}{3.0}\right)$ | $(0, 1)$ |
| HAS_KEYPHRASE | Chunk → Keyphrase | KeyBERT MMR score | $[0, 1]$ |
| CO_OCCURS | Entity ↔ Entity | $\sigma\!\left(\frac{\text{PMI}}{2.0}\right)$ | $(0, 1)$ |
| SEMANTIC_SIM | Entity ↔ Entity | $\cos(\mathbf{e}_1, \mathbf{e}_2)$ | $[\theta, 1]$ |
| SEMANTIC_SIM | Theme ↔ Theme | $\cos(\mathbf{e}_1, \mathbf{e}_2)$ | $[\theta, 1]$ |
| TEMPORAL_DIST | DATE ↔ DATE | $e^{-\lambda \lvert\Delta t\rvert}$ | $(0, 1]$ |
| CO_PRESENCE | Theme ↔ Theme | $c_{12} / \sqrt{f_1 f_2}$ | $[0, 1]$ |
| REVIEWS | Claim → Document | $1.0$ (structural) | $\{1\}$ |
| CLAIM_SIM | Claim ↔ Claim | $\cos(\mathbf{c}_1, \mathbf{c}_2)$ | $[0, 1]$ |

---

## Retrieval Strategies

All 7 strategies return scores in $[0, 1]$ without min-max normalization (each score has an absolute, interpretable meaning).

### Strategy 1: Semantic Similarity (S1)

**Module:** `retrieval_semantic.py`

Direct embedding cosine similarity between query and chunk:

$$S_1(\text{chunk}) = \max\!\bigl(0,\ \cos(\text{embed}(\text{query}),\ \text{embed}(\text{chunk}))\bigr)$$

**Normalization:** Raw cosine of normalized embeddings is in $[-1, 1]$. Negative values (unrelated content) are clipped to 0. No min-max scaling — the score is an absolute cosine similarity.

### Strategy 2: Topic Distribution Similarity (S2)

**Module:** `retrieval_topic.py`

Jensen-Shannon divergence between topic distributions:

$$S_2(\text{chunk}) = 1 - \text{JSD}(P_\text{query},\ P_\text{chunk})$$

where $P_\text{query} = \text{BERTopic.transform(query)}$ is a probability vector over topics.

**Normalization:** JSD is symmetric and naturally bounded $[0, 1]$, so $1 - \text{JSD} \in [0, 1]$ directly. No additional scaling.

$$\text{JSD}(P, Q) = \frac{1}{2}\,D_{KL}(P \,\|\, M) + \frac{1}{2}\,D_{KL}(Q \,\|\, M), \qquad M = \frac{P + Q}{2}$$

### Strategy 3: Thematic Graph Traversal (S3)

**Module:** `retrieval_thematic.py`

Detects themes in the query and propagates relevance through the theme graph.

**Algorithm:**
1. **Detect seed themes**: BERTopic topics in query with probability $\geq$ `theme_threshold` (default 0.01).
   Fallback: if no theme clears the threshold, use the top-1 theme (if probability > 0).
2. **Propagate relevance** via BFS through CO_PRESENCE and SEMANTIC_SIM edges:

$$\text{rel}_\text{neighbor} = \text{rel}_\text{parent} \times w_\text{edge}$$

BFS stops when $\text{rel}_\text{neighbor} <$ `theme_relevance_threshold` (default 0.1).

3. **Score chunks** using all reachable themes:

$$S_3(\text{chunk}) = \frac{\displaystyle\sum_{t \in \mathcal{T}} \text{rel}_t \cdot w_{\text{HAS\_TOPIC}}(t \to \text{chunk})}{\displaystyle\sum_{t \in \mathcal{T}} \text{rel}_t}$$

**Proof of bounds**: Each term $\text{rel}_t \cdot w_{\text{HAS\_TOPIC}} \leq \text{rel}_t$ since $w_{\text{HAS\_TOPIC}} \in [0,1]$, so numerator $\leq$ denominator $\Rightarrow S_3 \in [0, 1]$.

### Strategy 4: Entity Mention Retrieval (S4)

**Module:** `retrieval_entity.py`

Extracts entities from the query and scores chunks by their direct MENTIONS weights — **no BFS propagation**.

**Algorithm:**
1. **Extract entities** from query using spaCy NER. DATE → 4-digit year only. Filter entities $\leq$ 2 characters.
2. **Build seed list**: only entities present in the graph as nodes.
3. **Score chunks**: sum of MENTIONS edge weights from each seed entity, averaged by seed count.

$$S_4(\text{chunk}) = \frac{\displaystyle\sum_{s \in \text{seeds}} w_{\text{MENTIONS}}(s \to \text{chunk})}{|\text{seeds}|}$$

**Proof of bounds**: $w_{\text{MENTIONS}} \in (0, 1)$ (sigmoid), so each term $< 1$, and averaging keeps $S_4 \in (0, 1)$.

### Strategy 5: Claim-Based Document Retrieval (S5)

**Module:** `retrieval_claims.py`

Finds similar claims, then traverses $\text{REVIEWS} \to \text{CONTAINS}$ to reach chunks. No pre-computed data needed.

**Algorithm:**
1. **Find direct claims**: $\text{direct\_sim}(c_i) = \max(0,\ \cos(\text{embed}(\text{query}),\ \text{embed}(c_i)))$, keep top-k.
2. **Expand claims** (embedding-based, no graph edges required):

$$\text{eff\_sim}(c_j) = \text{direct\_sim}(c_i) \times \max\!\bigl(0,\ \cos(\text{embed}(c_i),\ \text{embed}(c_j))\bigr)$$

Keep expansions above `expansion_threshold`.

3. **Traverse graph**: Claim $\xrightarrow{\text{REVIEWS}}$ Document $\xrightarrow{\text{CONTAINS}}$ Chunk.
4. **Score chunks**:

$$S_5(\text{chunk}) = \max_{c \in \text{matched claims}} \text{direct\_sim}(c)$$

**Normalization:** Score = max cosine similarity, clipped at 0. $S_5 \in [0, 1]$.

### Strategy 6: Claim-Based Topic Distribution Traversal (S6)

**Module:** `retrieval_claim_topic.py`

Combines claim matching with topic distribution similarity for retrieved chunks.

**Algorithm:**
1. Find and expand similar claims (same as S5).
2. Traverse $\text{REVIEWS} \to \text{CONTAINS}$ to reach chunks.
3. For each (claim, chunk) pair:

$$S_6(\text{chunk}) = \max_{c \in \text{claims}} \Bigl[\text{sim}(c) \times \bigl(1 - \text{JSD}(P_\text{query}, P_\text{chunk})\bigr) \times \text{conf}(\text{chunk})\Bigr]$$

where $\text{conf}(\text{chunk}) = \max w_{\text{HAS\_TOPIC}}$ over the chunk's connected themes (default 0.5 if no topics).

**Bounds:** Product of three values each $\in [0, 1]$ $\Rightarrow$ $S_6 \in [0, 1]$.

### Strategy 7: Claim-Based Keyphrase Similarity (S7)

**Module:** `retrieval_claim_keyphrase.py`

Combines claim matching with keyphrase embedding similarity.

**Algorithm:**
1. **Extract keyphrases** from query using KeyBERT (MMR, diversity=0.5, ngram_range=(1,3)).
2. Find and expand similar claims (same as S5).
3. Traverse $\text{REVIEWS} \to \text{CONTAINS}$ to reach chunks.
4. For each (claim, chunk) pair:

$$S_7(\text{chunk}) = \max_{c \in \text{claims}} \Bigl[\text{sim}(c) \times \max_{kp \in \text{keyphrases}} \cos(\text{embed}(kp),\ \text{embed}(\text{chunk})) \times \text{imp}(kp)\Bigr]$$

where $\text{imp}(kp)$ is the KeyBERT score of the best-matching keyphrase.

**Bounds:** Product of three values each $\in [0, 1]$ $\Rightarrow$ $S_7 \in [0, 1]$.

---

## Document-Level Aggregation

**Module:** `retrieval.py` → `_aggregate_to_documents()`

After chunk-level scoring, scores are aggregated to document level using **softmax pooling** (attention-weighted average). Each chunk's weight is proportional to $e^{\beta \cdot s_c}$, so strong chunks attract exponentially more attention while multiple moderate chunks still collectively contribute.

**Softmax weights and pooled score:**

$$w_c = \frac{e^{\beta \cdot S_X(c)}}{\displaystyle\sum_{c' \in d} e^{\beta \cdot S_X(c')}}, \qquad \text{doc\_score}_X(d) = \sum_{c \in d} w_c \cdot S_X(c)$$

**Properties:**

| $\beta$ value | Behaviour |
|---------------|-----------|
| $\beta \to 0$ | Uniform weights → simple mean (many chunks contribute equally) |
| $\beta = 5$ *(default)* | Balanced: strong chunk leads, several moderate ones also raise the score |
| $\beta \to \infty$ | All weight on the max chunk → approaches max pooling |

- **Bounded in $[0, 1]$** — it is a weighted average of values already in $[0, 1]$.
- **Length-fair** — the weight normalisation prevents longer documents from dominating just because they have more chunks.
- **Evidence-accumulating** — unlike max pooling, a document with several chunks at 0.7 outscores one with a single chunk at 0.7, while a document with one chunk at 0.9 still dominates both.

**Per-strategy document rankings:** For each of the 7 strategies, documents are ranked by their strategy-specific softmax-pooled score. Only documents with a non-zero strategy score are included.

**Output per document:**
- `doc_score_total`, `doc_score_sem`, `doc_score_topic_dist`, `doc_score_theme`, `doc_score_entity`, `doc_score_claim_doc`, `doc_score_claim_topic`, `doc_score_claim_keyphrase`
- `n_chunks_scored`: number of chunks scored for this document
- `best_chunk_id`: chunk with the highest `score_total` (for display purposes)

The top `top_docs` (default 50) documents are returned per retrieval call.

---

## Mathematical Formulas

### Final Hybrid Score (Chunk Level)

$$\text{score} = \alpha\,S_1 + \beta\,S_2 + \gamma\,S_3 + \delta\,S_4 + \varepsilon\,S_5 + \zeta\,S_6 + \eta\,S_7$$

| Symbol | Weight | Default | Strategy |
|--------|--------|---------|----------|
| $\alpha$ | alpha | 0.20 | Semantic Similarity (S1) |
| $\beta$ | beta | 0.20 | Topic Distribution (S2) |
| $\gamma$ | gamma | 0.20 | Thematic Traversal (S3) |
| $\delta$ | delta | 0.20 | Entity Mention (S4) |
| $\varepsilon$ | epsilon | 0.20 | Claim-Document (S5) |
| $\zeta$ | zeta | 0.00 | Claim-Topic (S6) |
| $\eta$ | eta | 0.00 | Claim-Keyphrase (S7) |

All weights should sum to 1.0.

### TF-IDF with Sigmoid (MENTIONS edges)

$$\text{TF}_\text{norm} = \frac{tf}{\max\_tf}, \qquad \text{IDF} = \log\!\left(\frac{N}{df}\right), \qquad w = \sigma\!\left(\frac{\text{TF}_\text{norm} \times \text{IDF}}{3.0}\right), \qquad \sigma(x) = \frac{1}{1+e^{-x}}$$

### PMI with Sigmoid (CO_OCCURS edges)

$$\text{PMI}(e_1, e_2) = \log\!\left(\frac{c_{12} \times N}{f_1 \times f_2}\right), \qquad w = \sigma\!\left(\frac{\text{PMI}}{2.0}\right)$$

### Jensen-Shannon Divergence (S2, S6)

$$\text{JSD}(P, Q) = \frac{1}{2}\,D_{KL}(P \,\|\, M) + \frac{1}{2}\,D_{KL}(Q \,\|\, M), \qquad M = \frac{P + Q}{2}$$

$$D_{KL}(P \,\|\, Q) = \sum_x P(x)\,\log\frac{P(x)}{Q(x)}, \qquad \text{similarity} = 1 - \text{JSD} \in [0, 1]$$

### Exponential Decay (TEMPORAL_DIST edges)

$$w = e^{-\lambda\,\Delta t}, \qquad \lambda = \text{decay\_rate} = 0.15$$

### Thematic BFS Propagation (S3)

$$\text{rel}_\text{seed} = P_\text{BERTopic}(\text{query},\,t), \qquad \text{rel}_\text{child} = \text{rel}_\text{parent} \times w_\text{edge}$$

$$S_3(\text{chunk}) = \frac{\displaystyle\sum_{t} \text{rel}_t \cdot w_{\text{HAS\_TOPIC}}(t,\,\text{chunk})}{\displaystyle\sum_{t} \text{rel}_t} \in [0, 1]$$

### Claim Embedding Expansion (S5, S6, S7)

$$\text{direct\_sim}(c_i) = \max\!\bigl(0,\ \cos(\text{embed}(\text{query}),\ \text{embed}(c_i))\bigr)$$

$$\text{eff\_sim}(c_j \mid c_i) = \text{direct\_sim}(c_i) \times \max\!\bigl(0,\ \cos(\text{embed}(c_i),\ \text{embed}(c_j))\bigr)$$

The effective similarity of an expanded claim $c_j$ via anchor $c_i$ is bounded by $\text{direct\_sim}(c_i)$.

### Softmax Document Pooling

$$w_c = \frac{e^{\beta \cdot s_c}}{\displaystyle\sum_{c' \in d} e^{\beta \cdot s_{c'}}}, \qquad \text{doc\_score}_X(d) = \sum_{c \in d} w_c \cdot S_X(c), \qquad \beta = \texttt{softmax\_beta}$$

**Bounds proof**: $\text{doc\_score}_X$ is a convex combination of values $S_X(c) \in [0,1]$, so $\text{doc\_score}_X \in [0,1]$.

**Limiting cases**: $\beta \to 0 \Rightarrow$ uniform weights $\Rightarrow$ mean; $\beta \to \infty \Rightarrow$ all weight on $\arg\max S_X \Rightarrow$ max pooling.

### CO_PRESENCE Bounds Proof

$$w = \frac{c_{12}}{\sqrt{f_1 \times f_2}}$$

$$c_{12} \leq \min(f_1, f_2) \leq \sqrt{f_1 \times f_2} \quad(\text{AM-GM}) \qquad \Rightarrow \qquad w \in [0, 1]$$

---

## Configuration Parameters

### Build Parameters (`config_build.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `semantic_similarity_threshold` | 0.8 | Min similarity for SEMANTIC_SIM edges (entity & theme) |
| `co_occurrence_threshold` | 2 | Min co-occurrences for CO_OCCURS edges |
| `claim_sim_threshold` | 0.5 | Min similarity for CLAIM_SIM edges |
| `temporal_decay_rate` | 0.15 | $\lambda$ in exponential decay |
| `temporal_max_year_gap` | 10 | Max year gap for TEMPORAL_DIST edges |
| `topic_probability_threshold` | 0.15 | Min probability to create HAS_TOPIC edge |
| `keyphrase_top_n` | 5 | Number of keyphrases per chunk (S7) |
| `max_documents` | −1 (all) | Maximum documents to process |

### Retrieval Parameters (`config_retrieval.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theme_threshold` | 0.01 | Min BERTopic probability to detect seed themes (S3) |
| `theme_relevance_threshold` | 0.1 | Min propagated relevance to continue BFS (S3) |
| `entity_relevance_threshold` | 0.1 | Reserved (S4 no longer uses BFS) |
| `claim_top_k` | 5 | Number of top claims for S5/S6/S7 |
| `keyphrase_top_n` | 5 | Number of keyphrases to extract for S7 |
| `top_k` | 10 | Number of top chunks returned |
| `top_docs` | 50 | Number of top documents returned (per strategy and overall) |
| `softmax_beta` | 5.0 | Sharpness $\beta$ of softmax document pooling ($\beta \to 0$: mean; $\beta \to \infty$: max) |

### Strategy Weights

| Weight | Symbol | Default | Strategy |
|--------|--------|---------|----------|
| alpha  | $\alpha$ | 0.20 | Semantic Similarity (S1) |
| beta   | $\beta$ | 0.20 | Topic Distribution (S2) |
| gamma  | $\gamma$ | 0.20 | Thematic Traversal (S3) |
| delta  | $\delta$ | 0.20 | Entity Mention (S4) |
| epsilon | $\varepsilon$ | 0.20 | Claim-Document (S5) |
| zeta   | $\zeta$ | 0.00 | Claim-Topic (S6) |
| eta    | $\eta$ | 0.00 | Claim-Keyphrase (S7) |

---

## Evaluation

**Module:** `fact_check_kg/evaluation/evaluate.py`

Benchmarks retrieval performance against ground-truth (query, document) pairs.

**Input CSVs** (columns named `0` and `1`):
- `queries.csv`: `0`=query_id, `1`=query_text (social-media posts)
- `documents.csv`: `0`=doc_id, `1`=doc_text (fact-check articles)
- `relations.csv`: `0`=query_id, `1`=doc_id (ground-truth alignments)

**Filtering**: Only queries whose ground-truth document is present in the graph are evaluated. The graph contains the first `max_documents` documents from the source CSV.

**Metrics:**
- **Hit@K** ($K=1,5,10,50$): fraction of queries where the target document is in the top-$K$ retrieved documents
- **MRR**: Mean Reciprocal Rank — $\displaystyle\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q}$ (contribution = 0 if not found)
- **Mean Rank / Median Rank**: average position of the relevant document (found cases only)
- **Per-strategy Hit@10**: individual strategy effectiveness using max-pooled document scores

**CLI:**
```bash
python cli.py evaluate --graph-id <ID> \
  --queries data/queries.csv \
  --documents data/documents.csv \
  --relations data/relations.csv \
  --max-rank 500
```

Reports are saved to `outputs/evaluation/eval_{graph_id}_{timestamp}.{json,txt}`.
