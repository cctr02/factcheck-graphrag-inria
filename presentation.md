# Oral Presentation - GraphRAG for Fact-Checking
## Duration: 15 minutes | Presenters: Cédric TRINH & Tom LEON

---

## SLIDE 1 - Title Page
**[VISUAL: Title with clean background, university logo]**

Good morning/afternoon everyone. Today we will present our research project titled "GraphRAG for Fact-Checking: Knowledge Graph and Hybrid Retrieval". This work was carried out in collaboration with Tom Leon, under the supervision of the Inria Saclay research team.

The objective of this project is to develop an information retrieval system capable of efficiently finding relevant fact-checking articles to verify dubious claims circulating on the internet.

---

## SLIDE 2 - Context and Problem Statement
**[VISUAL: Triptych with 3 panels - (1) Social media icon with fake news, (2) Fact-checker logos (AFP, Snopes), (3) Question mark "How to retrieve?"]**

Let's start with the context. Misinformation has become a major challenge of our time. Professional organizations like AFP Fact Check, Snopes, or PolitiFact publish detailed articles daily to verify the veracity of viral claims. Our dataset contains approximately 38,000 English fact-checking articles, provided by Inria Saclay.

**The problem** we aim to solve is the following: given a short claim - for example "Trump said that Egypt might blow up the Ethiopian dam" - how can we quickly retrieve relevant fact-checking articles from this massive collection?

Classical approaches like BM25 or simple embedding similarity have their limitations. BM25 fails when claims use paraphrasing. Embeddings alone ignore the relational structure between entities and themes.

**Mini-conclusion:** We need an approach that combines semantic similarity AND relational structure. This is where GraphRAG comes in.

---

## SLIDE 3 - What is GraphRAG?
**[VISUAL: Triptych - (1) Classic RAG (documents → embeddings → retrieval), (2) Transition arrow, (3) GraphRAG (documents → graph → multi-hop retrieval)]**

So, what exactly is GraphRAG? For those familiar with RAG - Retrieval-Augmented Generation - it's an extension that adds a knowledge graph layer.

In classic RAG, we encode documents as vectors and perform similarity search. It's effective, but treats each document in isolation.

GraphRAG explicitly structures information as a graph: entities, themes, temporal and semantic relations become nodes and edges. This enables "multi-hop" retrieval: we can start from an entity mentioned in the claim, traverse the graph to related entities, and retrieve relevant documents even if they don't share the exact same keywords.

Recent work like Microsoft GraphRAG has demonstrated the value of this approach for improving retrieval robustness, particularly for complex questions requiring aggregation of dispersed information.

**Mini-conclusion:** GraphRAG = structure + relevance propagation through the graph.

---

## SLIDE 4 - Knowledge Graph Architecture: 4 Node Types
**[VISUAL: Hierarchical diagram showing Document → Chunks → (Themes + Entities)]**

Let's move to the concrete architecture of our knowledge graph. We designed a multi-level graph with 4 node types.

**First level: Documents.** Each fact-checking article is represented as a document node with its metadata: URL, title, publisher, publication date.

**Second level: Chunks.** Each document is split into overlapping text segments of approximately 500 characters, with 100 characters of overlap. This chunking respects sentence boundaries to preserve semantic coherence. Why chunks? Because it enables more granular retrieval: we can find the exact relevant passage rather than the whole document.

**Third level: Themes.** We use BERTopic - a state-of-the-art topic modeling framework - to automatically discover latent topics in the corpus. BERTopic combines transformer embeddings, UMAP dimensionality reduction, HDBSCAN clustering, and TF-IDF keyword extraction. Each theme is then labeled by a local LLM to obtain understandable labels.

**Fourth level: Entities.** We extract named entities with spaCy NER: persons, organizations, locations, dates, events. Dates are normalized to years to avoid fragmentation.

**Mini-conclusion:** This multi-level graph captures document hierarchy, thematic structure, and factual entities.

---

## SLIDE 5 - Knowledge Graph Architecture: 7 Edge Types
**[VISUAL: Complete graph schema showing all edge types with different colors]**

Now, let's see how we connect these nodes. We designed 7 edge types, each capturing a specific relationship.

**CONTAINS** (Document → Chunk): Basic hierarchical relation, uniform weight of 1.

**HAS_TOPIC** (Chunk → Theme): Edge weighted by probability P(theme|chunk) provided by BERTopic. If a chunk has 60% probability on theme "US elections", the weight will be 0.6.

**MENTIONS** (Chunk → Entity): Weighted by normalized TF-IDF + sigmoid. This gives more weight to entities frequent in a chunk but rare in the global corpus.

**CO_OCCURS** (Entity ↔ Entity): Document-level co-occurrence, weighted by PMI - Pointwise Mutual Information - with sigmoid normalization. Important: DATE-DATE relations are excluded as they have their own edge type.

**CO_PRESENCE** (Theme ↔ Theme): Chunk-level co-presence, weighted by normalized frequency. This captures themes that often appear together.

**SEMANTIC_SIM** (Entity ↔ Entity, Theme ↔ Theme): Cosine similarity between embeddings. For non-DATE entities, we use SentenceTransformer embeddings. For themes, we concatenate the top-10 keywords. Threshold at 0.8.

**TEMPORAL_DIST** (DATE ↔ DATE): Exponential decay based on temporal gap between years. Formula: exp(-0.5 × |year1 - year2|). Adjacent years have high weight (~0.61), years 10 years apart have near-zero weight.

**Mini-conclusion:** These 7 edge types encode hierarchy, semantics, co-occurrence, and temporal proximity.

---

## SLIDE 6 - Retrieval Pipeline: Overview
**[VISUAL: Your retrieval_diagram.png]**

Let's move to the core of the system: the hybrid retrieval pipeline. When a user submits a claim to verify, we transform it into three complementary representations.

**First, embeddings**: we encode the claim with SentenceTransformer (all-MiniLM-L6-v2, 384 dimensions).

**Second, topics**: we apply the trained BERTopic model to obtain a probability distribution over all themes.

**Finally, entities**: we extract entities with spaCy NER, same normalization as for graph construction.

These three representations then feed our four retrieval strategies, which execute in parallel. Each strategy produces a normalized score [0,1] per chunk, and we fuse these scores by weighted average to obtain the final score.

**Mini-conclusion:** Claim → 3 representations → 4 parallel strategies → score fusion.

---

## SLIDE 7 - Strategy 1: Semantic Similarity
**[VISUAL: Triptych - (1) Claim embedding, (2) FAISS index, (3) Top-K chunks by cosine similarity]**

Let's start with the first strategy: **Semantic Similarity**. This is our non-graph baseline.

**Offline construction:** During graph construction, all chunks are encoded with SentenceTransformer and indexed in FAISS - Facebook AI Similarity Search. FAISS uses approximate nearest neighbor algorithms (like IVF with PQ compression) to enable sub-linear search over large collections.

**Online retrieval:** When a claim arrives, we encode it with the same model, query the FAISS index to retrieve the top-K chunks by cosine similarity, and normalize scores to [0,1] via min-max normalization.

**Strength of this strategy:** It captures paraphrase and semantic similarity at the text level, even without keyword overlap. For example, "COVID vaccines contain microchips" and "coronavirus vaccine microchip" will have close embeddings.

**Limitation:** It treats chunks independently and ignores structural relationships.

**Mini-conclusion:** Strategy 1 = dense similarity through embeddings, effective but isolated baseline.

---

## SLIDE 8 - Strategy 2: Topic Distribution Similarity
**[VISUAL: Triptych - (1) Claim topic distribution, (2) Chunk topic distributions, (3) Jensen-Shannon divergence computation]**

Second strategy: **Topic Distribution Similarity**. This is also a non-graph approach, but it compares thematic probability distributions.

**Principle:** Instead of comparing texts directly, we compare their "thematic fingerprints". The claim and each chunk are represented as probability vectors over all themes discovered by BERTopic.

**Metric:** We use Jensen-Shannon (JS) divergence - a symmetric and bounded measure of distance between distributions. The formula is JS(P, Q) = ½ KL(P||M) + ½ KL(Q||M), where M is the average of the two distributions. JS is in [0, log 2].

**Conversion to similarity:** We invert with the formula: score = 1 - JS/log(2), which gives a score in [0,1].

**Intuition:** Two chunks treating the same general thematic context (for example "US elections 2016") will have similar distributions, even if exact words differ. This is a form of coarse-grained matching.

**Empirical result:** This strategy proved very effective in our tests - relevant chunks have scores from 0.91 to 0.99.

**Mini-conclusion:** Strategy 2 = thematic distributional similarity, captures global alignment.

---

## SLIDE 9 - Strategy 3: Thematic Graph Traversal
**[VISUAL: Triptych - (1) Seed themes from claim, (2) BFS propagation in theme graph, (3) Score accumulation to chunks]**

Third strategy: **Thematic Graph Traversal**. Here we enter the world of graph-based retrieval.

**Step 1 - Initialization:** We take the claim's topic distribution and select themes with probability above a threshold (default: 0.1). These themes become our "seed nodes" with initial relevance equal to their probability.

**Step 2 - BFS Propagation:** We launch a Breadth-First Search traversal in the theme subgraph, traversing CO_PRESENCE and SEMANTIC_SIM edges. At each hop, relevance is propagated by multiplying it by the edge weight. We filter by a minimum relevance threshold to avoid combinatorial explosion.

**Step 3 - Chunk Scoring:** Once propagation is complete, each theme has a final relevance. We then traverse HAS_TOPIC edges in reverse (from themes to chunks) and accumulate scores: score_chunk = Σ (theme_relevance × HAS_TOPIC_weight).

**Final normalization:** Min-max normalization to obtain scores in [0,1].

**Advantage:** This strategy can retrieve thematically related chunks even if they don't contain exactly the same topics as the claim - we exploit multi-hop propagation.

**Observed limitation:** In our preliminary tests, this strategy didn't always activate (zero scores), probably due to overly fragmented themes or too strict thresholds.

**Mini-conclusion:** Strategy 3 = thematic relevance propagation via graph, multi-hop expansion.

---

## SLIDE 10 - Strategy 4: Entity Graph Traversal
**[VISUAL: Triptych - (1) Entity extraction from claim, (2) BFS propagation in entity graph, (3) Score accumulation to chunks]**

Fourth strategy: **Entity Graph Traversal**. Same logic as the previous strategy, but applied to entities.

**Step 1 - Extraction:** We apply spaCy NER on the claim to extract entities (PERSON, ORG, GPE, DATE, etc.) with the same normalization as during graph construction. Each entity found in the graph becomes a seed node with initial relevance = 1.0.

**Step 2 - BFS Propagation:** Breadth-first traversal in the entity subgraph, traversing CO_OCCURS and SEMANTIC_SIM edges. Relevance propagates while decreasing at each hop (multiplication by edge weight). Relevance threshold for filtering.

**Step 3 - Chunk Scoring:** We traverse MENTIONS edges in reverse (from entities to chunks) and accumulate: score_chunk = Σ (entity_relevance × TF-IDF_weight).

**Concrete example:** If the claim mentions "Trump", we can traverse to "Republican Party", "2016", "White House", then retrieve all chunks mentioning these related entities, even if the claim didn't explicitly mention them.

**Strength:** Excellent factual expansion, captures entity associations.

**Observed weakness:** Can over-amplify highly connected "hub" entities (like "Trump"). A non-relevant chunk talking about Trump in another context can get a high score simply because Trump is highly connected.

**Mini-conclusion:** Strategy 4 = factual relevance propagation via graph, entity expansion.

---

## SLIDE 11 - Score Fusion
**[VISUAL: Fusion formula with the 4 coefficients + weight explanation]**

Now, how do we combine these four strategies? Through a simple weighted average.

**Formula:** score_final(chunk) = α × score_semantic + β × score_topic_dist + γ × score_thematic + δ × score_entity.

**Constraint:** α + β + γ + δ = 1.

**Default weights:** α = 0.3, β = 0.2, γ = 0.3, δ = 0.2.

**Justification for these weights:**
- We give strong weight (0.3) to semantic similarity as it's our robust baseline.
- We also give 0.3 to the thematic graph to maximize structure exploitation.
- The entity graph receives 0.2 as it's powerful but risks false positives.
- Thematic distribution similarity receives 0.2 as a distributional complement.

**Process:** We collect all chunks with non-zero score in at least one strategy, compute the final score for each, sort in descending order, and return the top-K (default: 10).

**Fusion strength:** Each strategy compensates for others' weaknesses. Semantic similarity captures direct paraphrases, distributions capture global thematic alignment, graphs capture indirect relationships.

**Mini-conclusion:** Weighted fusion = multi-signal robustness, strategy complementarity.

---

## SLIDE 12 - Preliminary Results: Case Study
**[VISUAL: Triptych - (1) Example claim, (2) Score table (like in the report), (3) Key observations]**

Let's move to results. We performed a preliminary qualitative evaluation on real claims. I'll present an emblematic case study.

**Tested claim:** "Donald Trump tweet that Egypt might 'blow up' Ethiopia's dam."

**Results for relevant chunks:**
- Chunk #1: Semantic = 1.00, TopicDist = 0.99, Thematic = 0.00, Entity = 0.29
- Chunk #2: Semantic = 0.95, TopicDist = 0.98, Thematic = 0.00, Entity = 0.33
- Chunk #3: Semantic = 0.84, TopicDist = 0.91, Thematic = 0.00, Entity = 0.45

**Observed pattern:** Semantic similarity and topic distribution similarity are excellent for relevant chunks. Entity contributes moderately. Thematic graph didn't activate (zero scores).

**Identified false positive:** A non-relevant chunk about Bill Barr and Trump (different context) obtained: Semantic = 0.55, TopicDist = 0.82, Thematic = 0.00, Entity = 0.98.

**Why this false positive?** The entity score is extremely high (0.98) because "Trump" is a highly connected hub. This compensates for average semantic similarity and pulls the chunk up.

**Mini-conclusion:** Semantic and distributional signals are reliable, but entity graph requires regularization to avoid false positives.

---

## SLIDE 13 - Observations and Insights
**[VISUAL: Triptych - (1) ✓ icon "Strengths", (2) ✗ icon "Limitations", (3) → icon "Improvement directions"]**

What should we retain from these preliminary results?

**Observation 1 - Strength:** Topic distribution similarity (Jensen-Shannon) is a very strong signal for identifying relevant chunks. Consistent scores between 0.91 and 0.99 for true positives. BERTopic captures coarse-grained thematic alignment well.

**Observation 2 - Double-edged sword:** The entity graph is powerful for factual expansion, but over-amplifies "hub" entities like Trump. A chunk mentioning Trump in a completely different context can get a high score simply through propagation via Trump.

**Observation 3 - Limitation:** The thematic graph didn't activate in this case study. Hypotheses: theme fragmentation, too strict seeding threshold, lack of density in CO_PRESENCE or SEMANTIC_SIM edges.

**Identified improvement directions:**

1. **Regularize the entity graph:** Down-weighting hub entities via PageRank, limiting propagation depth, or combining with strict semantic threshold.

2. **Enrich the thematic graph:** Detect finer topics, cluster fragmented themes, increase CO_PRESENCE edge density.

3. **Quantitative evaluation:** For now it's qualitative. We need a benchmark with ground-truth labels to optimize fusion weights and thresholds.

**Mini-conclusion:** Promising but perfectible results - the direction is clear.

---

## SLIDE 14 - Contributions and Generalization
**[VISUAL: Triptych - (1) "Scientific contributions", (2) "Technical architecture", (3) "Application domains"]**

Let's recap the contributions of this work.

**Contribution 1 - Multi-level graph design:** We designed a knowledge graph specifically adapted to fact-checking, with 4 node types and 7 edge types encoding hierarchy, semantics, co-occurrence, and temporality. This is an original architecture that goes beyond classic RDF graphs.

**Contribution 2 - Complementary hybrid strategies:** We demonstrated that we can effectively combine non-graph strategies (semantic similarity, topic distribution) and graph strategies (BFS on themes and entities) to obtain robust retrieval. Complementarity is key.

**Contribution 3 - Signal insights:** We empirically showed that topic distribution similarity is an extremely strong signal for fact-checking, and that entity graphs require regularization to avoid popularity biases.

**Technical architecture:** Complete end-to-end system with graph construction, FAISS indexing, BFS propagation, and score fusion. Modular and extensible code.

**Generalization:** Although designed for fact-checking, this system generalizes to other domains requiring claim-based retrieval: case law, scientific literature, public policy archives, complex technical documentation.

**Mini-conclusion:** Scientific, technical contributions, generalizable beyond fact-checking.

---

## SLIDE 15 - Future Work and Perspectives
**[VISUAL: Roadmap with 4 axes]**

What are the next steps to improve this system?

**Axis 1 - Thematic graph enrichment:**
- Detect more fine-grained topics by adjusting BERTopic hyperparameters.
- Hierarchical clustering of themes to reduce fragmentation.
- Increase CO_PRESENCE and SEMANTIC_SIM edge density.

**Axis 2 - Entity graph regularization:**
- Down-weighting hub entities via PageRank or degree centrality.
- BFS depth limitation (max 2-3 hops).
- Strict semantic threshold filtering: only accept propagation if entity also has minimal semantic similarity with claim.

**Axis 3 - Rigorous quantitative evaluation:**
- Build or obtain a benchmark with ground-truth labels.
- Measure Precision@K, Recall@K, NDCG, MRR.
- Fusion weight optimization (α, β, γ, δ) and propagation thresholds via grid search or Bayesian optimization.

**Axis 4 - Alternative traversal algorithms:**
- Replace BFS with Personalized PageRank for more global scoring.
- Explore Graph Neural Networks (GNN) to learn relevance propagation end-to-end.
- Test Random Walk with Restart for stochastic exploration.

**Mini-conclusion:** Clear roadmap with technical and scientific axes to move from prototype to production-ready system.

---

## SLIDE 16 - Conclusion
**[VISUAL: Visual summary - graph + 4 strategies + results]**

To conclude, let's recap.

**Problem:** Retrieve relevant fact-checking articles from a large collection, given a short and potentially paraphrased claim.

**Proposed solution:** Multi-level GraphRAG combining:
- A knowledge graph with 4 node types (documents, chunks, themes, entities) and 7 edge types (hierarchical, thematic, factual, temporal).
- A hybrid retrieval pipeline with 4 complementary strategies: semantic similarity, topic distribution similarity, thematic graph traversal, entity graph traversal.
- Weighted fusion to obtain a robust final score.

**Preliminary results:** Topic distribution similarity is a very strong signal. Graphs enable effective factual expansion but require regularization. The system works but can be improved.

**Potential impact:** Beyond fact-checking, this architecture can apply to any domain requiring structural retrieval on corpora with recurring entities and complex themes.

**Final message:** GraphRAG is not just a trend - it's a natural evolution of RAG that exploits the latent structure of data. Our results show that combining dense similarity and graph propagation is a promising path for improving retrieval.

Thank you for your attention. We are now available for your questions.

---

## SLIDE 17 - Questions / Discussion
**[VISUAL: Simple slide "Questions?" with your contacts]**

[Answer jury questions]

---

## SUPPLEMENTARY NOTES FOR PREPARATION

### Recommended timing (15 min total):
- Slides 1-2 (Intro + Context): 1min30
- Slide 3 (GraphRAG concept): 1min
- Slides 4-5 (Graph architecture): 2min30
- Slide 6 (Pipeline overview): 1min
- Slides 7-10 (4 strategies): 4min (1min per strategy)
- Slide 11 (Fusion): 1min
- Slides 12-13 (Results): 2min
- Slides 14-15 (Contributions + Future): 1min30
- Slide 16 (Conclusion): 30sec

### Distribution suggestions between Cédric and Tom:
**Cédric can handle:**
- Technical implementation aspects (FAISS, BFS algorithms)
- Strategy 1 (Semantic Similarity)
- Strategy 3 (Thematic Graph Traversal)
- Score fusion mechanics
- Technical architecture details

**Tom can handle:**
- Contextualization and motivation
- Graph design (nodes and edges)
- Strategy 2 (Topic Distribution Similarity)
- Strategy 4 (Entity Graph Traversal)
- Results presentation
- Generalization aspects

### Probable questions to anticipate:

1. **"Why not Graph Neural Networks instead of BFS?"**
   → Good question. GNN would be the natural next step. We started with BFS for its simplicity and interpretability. BFS allows easy debugging of propagation. GNN would require ground-truth training labels which we don't have yet. It's in the roadmap (Axis 4).

2. **"How do you handle scalability?"**
   → Currently 38k documents, the graph fits in memory. FAISS enables sub-linear retrieval. To scale to millions of documents, we could partition the graph, use Neo4j or distributed GraphDB, and do two-phase retrieval: first fast FAISS to filter, then BFS on subgraph.

3. **"Have you compared with baselines like BM25 or DPR?"**
   → Not yet formally. That's precisely Axis 3 (quantitative evaluation). We implemented pure semantic similarity (FAISS) as internal baseline. Qualitative results show our fusion improves precision, but we need quantitative metrics to confirm rigorously.

4. **"Are the weights α, β, γ, δ fixed empirically?"**
   → Yes, for now it's empirical based on our qualitative observations. The next step would be to optimize them on a dev set with objective metrics (NDCG, MRR) via grid search or Bayesian optimization.

5. **"Why BERTopic and not LDA or other topic models?"**
   → BERTopic combines the best of both worlds: transformer embeddings to capture deep semantics + density-based clustering for coherent topics without fixing the number a priori. It's more robust than LDA on short and heterogeneous texts. Plus, BERTopic provides topic-chunk probabilities exploitable for our HAS_TOPIC edges.

6. **"How do you handle multilingual fact-checking?"**
   → Currently we work on English only. For multilingual, we'd need multilingual embeddings (e.g., mBERT, XLM-R), multilingual NER models, and language-specific or cross-lingual topic models. The graph architecture itself would remain the same.

7. **"What about the computational cost of graph construction?"**
   → Graph construction is a one-time offline cost. For 38k documents: BERTopic training ~30 min, embedding computation ~20 min (cached via FAISS), edge construction ~15 min, total ~1-2 hours. For retrieval, it's real-time: FAISS is sub-linear, BFS is fast on local subgraphs.

### Visual recommendations:

**Slide 4 (4 node types):**
- Show a simple hierarchical diagram: one "Document" rectangle at top, several "Chunk" rectangles below, then two columns at bottom for "Themes" and "Entities".
- Use colors: blue for documents, yellow for chunks, purple for themes, green for entities.

**Slide 5 (7 edge types):**
- Use your report figure (graph_overview) or recreate it cleanly.
- Color-code edges: gray for CONTAINS, purple for HAS_TOPIC, green for MENTIONS, red for CO_OCCURS, blue for SEMANTIC_SIM, orange for TEMPORAL_DIST, etc.

**Slide 6 (Pipeline):**
- Absolutely use your retrieval_diagram.png from the report. It's perfect for this slide.

**Slides 7-10 (Strategies):**
- For each strategy, a mini flow diagram: Input (claim) → Transformation → Graph/Index → Output (scores).
- Strategy 1: Diagram with embedding + FAISS index.
- Strategy 2: Two probability distributions + divergence computation.
- Strategies 3 and 4: Mini-graph with colored nodes and BFS propagation arrows.

**Slide 12 (Results):**
- Reproduce the report table (Chunk #1, #2, #3 + False positive).
- Add a small horizontal bar for each score, more visual than just numbers.

Good luck with your presentation!

```latex
\overline{\mathrm{TF}}(e, c) = \frac{\mathrm{TF}(e, c)}{\max_{e'} \mathrm{TF}(e', c)}
```