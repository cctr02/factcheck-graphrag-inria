# Migration vers PostgreSQL

Ce document explique les modifications apportées au système de Knowledge Graph pour supporter le stockage dans PostgreSQL.

## Vue d'ensemble des changements

Le graphe peut maintenant être sauvegardé dans PostgreSQL au lieu d'un fichier pickle. Le format de stockage utilise deux tables :

### Table `graph`
```sql
CREATE TABLE graph (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    predicate TEXT NOT NULL,
    target TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table `nodes`
```sql
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT,
    attributes JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Fichiers modifiés

### 1. Nouveaux fichiers

#### `db_connector.py`
Nouveau module qui gère toutes les interactions avec PostgreSQL :
- `GraphDatabaseConnector`: Classe principale pour la connexion
- `create_tables()`: Crée les tables et index
- `save_graph()`: Sauvegarde un graphe NetworkX vers PostgreSQL
- `load_graph()`: Charge un graphe depuis PostgreSQL vers NetworkX
- `get_graph_statistics()`: Obtient des statistiques sur le graphe stocké
- `query_neighbors()`: Requête optimisée pour les voisins d'un nœud

### 2. Fichiers modifiés

#### `config.py`
- Ajout de la classe `DatabaseConfig` avec les paramètres de connexion PostgreSQL :
  - `host`: hostname (défaut: "localhost")
  - `port`: port (défaut: 5432)
  - `database`: nom de la base (défaut: "knowledge_graph")
  - `user`: utilisateur (défaut: "postgres")
  - `password`: mot de passe (défaut: "postgres")

#### `graph_builder.py`
- Ajout de la méthode `save_to_postgres()` à la classe `KnowledgeGraphBuilder`
- Cette méthode utilise `GraphDatabaseConnector` pour sauvegarder le graphe

#### `build_graph.py`
- Ajout de l'option `--use-postgres` pour sauvegarder dans PostgreSQL
- Ajout des options de configuration de base de données :
  - `--db-host`
  - `--db-port`
  - `--db-name`
  - `--db-user`
  - `--db-password`
- Support du mode legacy (pickle) si `--use-postgres` n'est pas spécifié

#### `visualize_graph.py`
- Ajout de la fonction `load_graph_from_source()` qui charge depuis PostgreSQL ou pickle
- Modification de toutes les fonctions de visualisation pour supporter PostgreSQL :
  - `visualize_full_graph()`
  - `visualize_entity_subgraph()`
  - `analyze_graph_stats()`
- Ajout des mêmes options CLI que `build_graph.py`

#### `test_retrieval.py`
- Ajout du support PostgreSQL pour le chargement du graphe
- Ajout des options CLI pour PostgreSQL
- Le topic model reste chargé depuis pickle

## Utilisation

### Configuration de PostgreSQL

1. Installer PostgreSQL et psycopg2 :
```bash
pip install psycopg2-binary
```

2. Créer la base de données :
```sql
CREATE DATABASE knowledge_graph;
```

### Construction du graphe

#### Mode PostgreSQL (nouveau)
```bash
python fact_check_kg/build_graph.py \
    --use-postgres \
    --db-host localhost \
    --db-port 5432 \
    --db-name knowledge_graph \
    --db-user postgres \
    --db-password your_password \
    --max-docs 100
```

#### Mode pickle (legacy)
```bash
python fact_check_kg/build_graph.py \
    --output knowledge_graph.gpickle \
    --max-docs 100
```

### Visualisation du graphe

#### Depuis PostgreSQL
```bash
# Statistiques
python fact_check_kg/visualize_graph.py \
    --use-postgres \
    --db-name knowledge_graph \
    --mode stats

# Graphe complet
python fact_check_kg/visualize_graph.py \
    --use-postgres \
    --db-name knowledge_graph \
    --mode full \
    --output graph_full.html

# Sous-graphe d'entité
python fact_check_kg/visualize_graph.py \
    --use-postgres \
    --db-name knowledge_graph \
    --mode entity \
    --entity "gaza"
```

#### Depuis pickle
```bash
python fact_check_kg/visualize_graph.py \
    --graph knowledge_graph.gpickle \
    --mode stats
```

### Test de retrieval

#### Avec PostgreSQL
```bash
python fact_check_kg/test_retrieval.py \
    --use-postgres \
    --db-name knowledge_graph \
    --claim "Your claim here"
```

#### Avec pickle
```bash
python fact_check_kg/test_retrieval.py \
    --graph knowledge_graph.gpickle \
    --claim "Your claim here"
```

## Avantages de PostgreSQL

1. **Scalabilité** : Peut gérer des graphes très larges sans charger tout en mémoire
2. **Persistance** : Données durables et fiables
3. **Requêtes SQL** : Possibilité d'effectuer des requêtes SQL complexes
4. **Index** : Recherches optimisées sur source, target, predicate
5. **Concurrence** : Plusieurs processus peuvent accéder au graphe simultanément
6. **Backup** : Outils de backup PostgreSQL standard

## Structure des données

### Nœuds
Les attributs des nœuds sont stockés en JSONB dans la table `nodes` :
- `type`: "document", "chunk", "theme", ou "entity"
- Attributs spécifiques selon le type :
  - **document**: `doc_id`
  - **chunk**: `doc_id`, `text`, `chunk_idx`
  - **theme**: `topic_id`, `label`, `keywords`
  - **entity**: `entity_type`, `entity_text`

### Arêtes
Les arêtes sont stockées dans la table `graph` avec :
- `source`: ID du nœud source
- `predicate`: Type d'arête ("CONTAINS", "HAS_TOPIC", "MENTIONS", "RELATED_TO", "CO_OCCURS", "SEMANTIC_SIM")
- `target`: ID du nœud cible
- `weight`: Poids de l'arête (importance/probabilité)

## Compatibilité

- Le mode pickle reste entièrement supporté
- Les deux formats peuvent coexister
- Le topic model BERTopic reste sauvegardé en pickle car il contient des objets Python complexes
- Tous les scripts peuvent fonctionner avec l'un ou l'autre format

## Dépendances ajoutées

```txt
psycopg2-binary>=2.9.0
```

## Notes techniques

- Les index sont créés automatiquement pour optimiser les performances
- Les transactions sont utilisées pour garantir la cohérence des données
- Les batch inserts sont utilisés pour améliorer les performances d'insertion
- Le graphe NetworkX est reconstruit depuis PostgreSQL lors du chargement
