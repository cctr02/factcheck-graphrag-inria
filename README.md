# Knowledge Graph for Fact-Checking

Système de fact-checking basé sur un graphe de connaissances multi-niveaux combinant modélisation de topics (BERTopic), extraction d'entités (spaCy) et retrieval hybride.

---

## 1. Construction du Graph

Le graphe de connaissances est construit en plusieurs étapes, créant un réseau multi-niveaux reliant documents, chunks, thèmes et entités.

### 1.1 Construction des Nœuds

#### **Documents et Chunks**

Les documents sont découpés en chunks avec chevauchement pour préserver le contexte :

```python
# Chunking avec overlap
chunk_size = 500  # caractères
chunk_overlap = 100  # caractères

chunks = chunk_text(document, chunk_size=500, overlap=100)
```

**Format des nœuds** :
- Document : `doc::{doc_id}`
- Chunk : `chunk::{doc_id}::{chunk_idx}`

Chaque chunk contient :
- `text` : le contenu textuel
- `doc_id` : référence au document parent
- `chunk_idx` : index du chunk dans le document

#### **Thèmes (Topics)**

Les thèmes sont découverts automatiquement via **BERTopic** :

```python
# Entraînement du modèle de topics
topic_model = BERTopic(
    min_topic_size=10,
    nr_topics=None  # détection automatique
)

topics, probabilities = topic_model.fit_transform(chunk_texts)
```

**Génération des labels** : Les thèmes sont labellisés par un LLM (Ollama) basé sur les mots-clés extraits :

```python
# Top 5 keywords du topic
keywords = topic_model.get_topic(topic_id)[:5]
# Prompt LLM pour générer un label concis
label = llm_generate_topic_label(keywords)
```

**Format des nœuds** :
- Thème : `theme::{topic_id}`

Attributs :
- `topic_id` : ID du topic BERTopic
- `label` : label généré par LLM
- `keywords` : liste des mots-clés principaux

#### **Entités**

Les entités sont extraites via **spaCy NER** (Named Entity Recognition) :

```python
# Extraction des entités
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

entities = []
for ent in doc.ents:
    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'NORP']:
        entities.append((ent.text.lower(), ent.label_))
```

**Types d'entités extraites** :
- `PERSON` : personnes (réelles ou fictives)
- `ORG` : organisations, entreprises
- `GPE` : entités géopolitiques (pays, villes)
- `LOC` : lieux non-GPE (montagnes, océans)
- `DATE` : dates et périodes temporelles
- `EVENT` : événements nommés (guerres, catastrophes)
- `NORP` : nationalités, groupes religieux ou politiques

**Format des nœuds** :
- Entité : `entity::{entity_type}::{entity_text}`

---

### 1.2 Construction des Arêtes

#### **CONTAINS** : Document → Chunk

Arête simple reliant chaque document à ses chunks.

```
Poids : 1.0 (non pondéré)
```

---

#### **HAS_TOPIC** : Chunk → Thème

Relie chaque chunk aux thèmes détectés par BERTopic, avec un seuil de probabilité.

**Calcul du poids** :

```python
# Probabilité BERTopic pour chaque topic
topic_probs = topic_model.transform([chunk_text])[1][0]

# Création d'arêtes uniquement si prob > threshold
for topic_id, prob in enumerate(topic_probs):
    if prob > topic_probability_threshold:  # défaut: 0.15
        graph.add_edge(
            chunk_id,
            f"theme::{topic_id}",
            edge_type="HAS_TOPIC",
            weight=prob  # Probabilité BERTopic
        )
```

**Formule** :
```
weight_HAS_TOPIC = P(topic|chunk)
```

où `P(topic|chunk)` est la probabilité BERTopic que le chunk appartienne au topic.

---

#### **MENTIONS** : Chunk → Entité

Relie les chunks aux entités qu'ils mentionnent, pondéré par **TF-IDF**.

**Calcul du poids** :

```python
# TF (Term Frequency) : nombre d'occurrences de l'entité dans le chunk
TF(entity, chunk) = count(entity in chunk)

# IDF (Inverse Document Frequency) : rareté de l'entité
IDF(entity) = log(total_chunks / chunks_containing_entity)

# TF-IDF final
weight_MENTIONS = TF(entity, chunk) × IDF(entity)
```

**Formule** :
```
weight_MENTIONS(chunk, entity) = TF(entity, chunk) × log(N / df(entity))
```

où :
- `N` = nombre total de chunks
- `df(entity)` = nombre de chunks contenant l'entité

**Avantage** : Les entités rares (haute valeur informative) reçoivent des poids plus élevés.

---

#### **CO_OCCURS** : Entité ↔ Entité

Relie les entités qui apparaissent ensemble dans les mêmes documents.

**Calcul du poids** :

```python
# Co-occurrence au niveau document (pas chunk)
co_occur_count = nombre de documents contenant (entity1 ET entity2)

# Fréquences des entités
freq1 = nombre de documents contenant entity1
freq2 = nombre de documents contenant entity2

# Poids inversement proportionnel à la fréquence
inv_freq_weight = 1 / sqrt(freq1 × freq2)

# Poids final
weight_CO_OCCURS = co_occur_count × inv_freq_weight
```

**Formule** :
```
weight_CO_OCCURS(e1, e2) = count(e1, e2) × (1 / √(freq(e1) × freq(e2)))
```

**Intuition** :
- Les entités qui co-apparaissent fréquemment ont un poids élevé
- **Mais** les entités omniprésentes sont pénalisées (diviseur)
- Exemple : "Trump" et "France" co-apparaissent rarement → poids élevé si co-occurrence détectée

---

#### **SEMANTIC_SIM** : Entité ↔ Entité / Thème ↔ Thème

Relie les entités (ou thèmes) sémantiquement similaires via leurs embeddings.

**Pour les entités** :

```python
# Embeddings des noms d'entités
embeddings = SentenceTransformer.encode([entity_text1, entity_text2, ...])

# Similarité cosinus
sim = cosine_similarity(embedding1, embedding2) = embedding1 · embedding2

# Arête si sim > threshold (défaut: 0.6)
if sim > semantic_similarity_threshold:
    weight_SEMANTIC_SIM = sim
```

**Pour les thèmes** :

```python
# Représentation = concaténation des top 5 keywords
theme_repr = " ".join(topic_keywords[:5])

# Embeddings puis similarité cosinus
sim = cosine_similarity(theme_embedding1, theme_embedding2)
```

**Formule** :
```
weight_SEMANTIC_SIM(n1, n2) = cos(emb(n1), emb(n2)) = (emb(n1) · emb(n2)) / (||emb(n1)|| × ||emb(n2)||)
```

Avec normalisation des embeddings : `||emb|| = 1`, donc :
```
weight_SEMANTIC_SIM(n1, n2) = emb(n1) · emb(n2)
```

---

#### **CO_PRESENCE** : Thème ↔ Thème

Relie les thèmes qui apparaissent ensemble dans les mêmes chunks (via arêtes HAS_TOPIC).

**Calcul du poids** :

```python
# Co-présence au niveau chunk
co_presence_count = nombre de chunks contenant (theme1 ET theme2)

# Fréquences des thèmes
freq1 = nombre de chunks contenant theme1
freq2 = nombre de chunks contenant theme2

# Poids inversement proportionnel à la fréquence
inv_freq_weight = 1 / sqrt(freq1 × freq2)

# Poids final (même formule que CO_OCCURS)
weight_CO_PRESENCE = co_presence_count × inv_freq_weight
```

**Formule** :
```
weight_CO_PRESENCE(t1, t2) = count(t1, t2) × (1 / √(freq(t1) × freq(t2)))
```

---

## 2. Retrieval du Graph

Le système utilise un **retrieval hybride** combinant 4 stratégies complémentaires, fusionnées par moyenne pondérée.

### 2.1 Stratégie 1 : Retrieval Sémantique

Basé sur la similarité d'embeddings entre le claim et les chunks.

**Processus** :

```python
# 1. Encoder le claim
claim_embedding = SentenceTransformer.encode(claim)

# 2. Encoder tous les chunks (pré-calculé)
chunk_embeddings = SentenceTransformer.encode(all_chunks)

# 3. Calculer similarités cosinus
similarities = chunk_embeddings @ claim_embedding

# 4. Normaliser (min-max)
score_sem(chunk) = (sim(chunk) - min_sim) / (max_sim - min_sim)
```

**Formule** :
```
score_sem(chunk) = normalize(cos(emb(claim), emb(chunk)))
```

**Sortie** : Dictionnaire `{chunk_id: score_sem}` avec scores normalisés `[0, 1]`.

---

### 2.2 Stratégie 2 : Retrieval Thématique

Basé sur les topics BERTopic et la propagation de pertinence dans le graphe de thèmes.

**Processus** :

```python
# 1. Classifier le claim en topics
topic_probs = BERTopic.transform(claim)

# 2. Sélectionner les thèmes pertinents (prob > threshold)
seed_themes = [(theme_id, prob) for theme_id, prob in topic_probs if prob > 0.1]

# 3. Propager la pertinence dans le graphe de thèmes (BFS)
for theme, relevance in BFS(seed_themes):
    for neighbor_theme in graph.neighbors(theme):
        edge_weight = graph[theme][neighbor_theme]['weight']  # CO_PRESENCE ou SEMANTIC_SIM
        propagated_relevance = relevance × edge_weight
        if propagated_relevance > threshold:
            update_relevance(neighbor_theme, propagated_relevance)

# 4. Scorer les chunks via arêtes HAS_TOPIC
for theme, relevance in theme_relevance.items():
    for chunk in chunks_with_HAS_TOPIC_to(theme):
        prob_weight = graph[chunk][theme]['weight']  # Probabilité BERTopic
        score_theme[chunk] += relevance × prob_weight

# 5. Normaliser
score_theme(chunk) = normalize(score_theme[chunk])
```

**Formules** :

1. **Propagation de pertinence** (BFS dans le graphe de thèmes) :
```
relevance(theme_neighbor) = max(
    current_relevance(theme_neighbor),
    relevance(theme_current) × weight_edge(theme_current, theme_neighbor)
)
```

2. **Score final du chunk** :
```
score_theme(chunk) = normalize(∑_{theme} relevance(theme) × P(theme|chunk))
```

où `P(theme|chunk)` est le poids de l'arête HAS_TOPIC (probabilité BERTopic).

**Sortie** : Dictionnaire `{chunk_id: score_theme}`.

---

### 2.3 Stratégie 3 : Retrieval par Entités

Basé sur l'extraction d'entités du claim et la propagation de pertinence dans le graphe d'entités.

**Processus** :

```python
# 1. Extraire les entités du claim
entities_claim = spaCy.extract_entities(claim)

# 2. Trouver les entités correspondantes dans le graphe
seed_entities = [entity_node for entity_text in entities_claim if entity_node in graph]

# 3. Propager la pertinence dans le graphe d'entités (BFS)
for entity, relevance in BFS(seed_entities):
    for neighbor_entity in graph.neighbors(entity):
        edge_weight = graph[entity][neighbor_entity]['weight']  # CO_OCCURS ou SEMANTIC_SIM
        propagated_relevance = relevance × edge_weight
        if propagated_relevance > threshold:
            update_relevance(neighbor_entity, propagated_relevance)

# 4. Scorer les chunks via arêtes MENTIONS
for entity, relevance in entity_relevance.items():
    for chunk in chunks_with_MENTIONS_to(entity):
        tf_idf_weight = graph[chunk][entity]['weight']  # TF-IDF
        score_entity[chunk] += relevance × tf_idf_weight

# 5. Normaliser
score_entity(chunk) = normalize(score_entity[chunk])
```

**Formules** :

1. **Propagation de pertinence** (BFS dans le graphe d'entités) :
```
relevance(entity_neighbor) = max(
    current_relevance(entity_neighbor),
    relevance(entity_current) × weight_edge(entity_current, entity_neighbor)
)
```

2. **Score final du chunk** :
```
score_entity(chunk) = normalize(∑_{entity} relevance(entity) × TF-IDF(entity, chunk))
```

**Sortie** : Dictionnaire `{chunk_id: score_entity}`.

---

### 2.4 Stratégie 4 : Retrieval par Distribution de Topics

Basé sur la similarité entre les distributions de probabilités de topics (Jensen-Shannon divergence).

**Processus** :

```python
# 1. Obtenir la distribution de topics du claim
claim_topic_dist = BERTopic.transform(claim)  # Vecteur de probabilités

# 2. Obtenir les distributions de tous les chunks (pré-calculées)
chunk_topic_dists = [BERTopic.transform(chunk) for chunk in chunks]

# 3. Calculer la divergence de Jensen-Shannon
for chunk, chunk_dist in zip(chunks, chunk_topic_dists):
    js_divergence = jensen_shannon(claim_topic_dist, chunk_dist)

    # Convertir en similarité (divergence ∈ [0, 1])
    similarity = 1 - js_divergence
    score_topic_dist[chunk] = similarity

# 4. Normaliser
score_topic_dist(chunk) = normalize(similarity)
```

**Formule de Jensen-Shannon Divergence** :

```
JS(P || Q) = (1/2) × KL(P || M) + (1/2) × KL(Q || M)
```

où :
- `M = (P + Q) / 2` (distribution moyenne)
- `KL(P || Q) = ∑_i P(i) × log(P(i) / Q(i))` (divergence de Kullback-Leibler)

**Conversion en similarité** :
```
score_topic_dist(chunk) = normalize(1 - JS(P_claim || P_chunk))
```

**Propriétés** :
- Symétrique : `JS(P || Q) = JS(Q || P)`
- Bornée : `JS ∈ [0, 1]`
- `JS = 0` si distributions identiques

**Sortie** : Dictionnaire `{chunk_id: score_topic_dist}`.

---

### 2.5 Fusion des Scores

Les 4 scores sont combinés par **moyenne pondérée** avec poids configurables.

**Formule de fusion** :

```
score_final(chunk) = α × score_sem(chunk)
                   + β × score_theme(chunk)
                   + γ × score_entity(chunk)
                   + δ × score_topic_dist(chunk)
```

**Contrainte** : `α + β + γ + δ = 1.0`

**Poids par défaut** :
- `α = 0.3` (sémantique)
- `β = 0.2` (thématique)
- `γ = 0.3` (entités)
- `δ = 0.2` (distribution de topics)

**Processus de fusion** :

```python
# Union de tous les chunks trouvés par les 4 stratégies
all_chunks = set(S1) ∪ set(S2) ∪ set(S3) ∪ set(S4)

# Pour chaque chunk, calculer le score final
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

# Trier par score final décroissant
results.sort(key=lambda x: x['score_total'], reverse=True)

# Retourner top-K
return results[:top_k]
```

**Sortie finale** : Liste des `top_k` chunks les plus pertinents avec leurs scores détaillés.

---

## 3. Architecture Technique

### Modèles utilisés

- **Embeddings** : `all-MiniLM-L6-v2` (SentenceTransformers)
- **NER** : `en_core_web_sm` (spaCy)
- **Topic Modeling** : BERTopic
- **LLM** : `llama3.1:latest` (Ollama) pour les labels de topics

### Stockage

- **Graph** : NetworkX (pickle) + PostgreSQL (tables versionnées)
- **Topic Model** : pickle (.pkl)
- **Metadata** : JSON

### Format des tables PostgreSQL

**Table `graph_{timestamp}`** (edges) :
- `source` : nœud source
- `predicate` : type d'arête (CONTAINS, HAS_TOPIC, etc.)
- `target` : nœud cible
- `weight` : poids de l'arête

**Table `nodes_{timestamp}`** (nodes) :
- `node_id` : identifiant unique
- `node_type` : type (document, chunk, theme, entity)
- `attributes` : JSONB avec tous les attributs

**Table `graph_metadata`** :
- `graph_id` : timestamp de création
- `num_nodes` : nombre de nœuds
- `num_edges` : nombre d'arêtes
- `metadata` : JSONB avec configuration

---

## 4. Utilisation

### Construction d'un graph

```bash
uv run python cli.py build --max-docs 100
```

### Retrieval

```bash
uv run python cli.py retrieve "Trump met Macron in France" --top-k 10 --graph-id 20251118_150602
```

### Visualisation

```bash
# Visualisation complète
uv run python cli.py visualize --mode full --graph-id 20251118_150602

# Sous-graphe centré sur une entité
uv run python cli.py visualize --mode entity --entity "trump" --graph-id 20251118_150602
```

---

## 5. Configuration

Voir [config.yaml](config.yaml) pour tous les paramètres :

```yaml
graph_construction:
  topic_probability_threshold: 0.15
  co_occurrence_threshold: 2
  semantic_similarity_threshold: 0.6

retrieval:
  alpha: 0.3  # poids sémantique
  beta: 0.2   # poids thématique
  gamma: 0.3  # poids entités
  delta: 0.2  # poids distribution topics
  entity_relevance_threshold: 0.1
  theme_relevance_threshold: 0.1
```

---

## 6. Références

- **BERTopic** : [Grootendorst, 2022](https://github.com/MaartenGr/BERTopic)
- **SentenceTransformers** : [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)
- **spaCy** : [Honnibal & Montani, 2017](https://spacy.io)
- **Jensen-Shannon Divergence** : [Lin, 1991](https://ieeexplore.ieee.org/document/61115)
