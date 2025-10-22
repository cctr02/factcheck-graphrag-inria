# ============================================
#  Minimal GraphRAG pipeline:
#  - Nodes: documents
#  - Edge: shared named entities between docs
# ============================================

from typing import List, Dict, Tuple, Optional
import itertools
import logging
import pandas as pd
import networkx as nx
import spacy
from tqdm import tqdm


# --- 1) Load the corpus ---
def load_corpus_from_csv(
    path: str,
    id_col: str = "id",
    title_col: str = "title",
    body_col: str = "body",
    lang_col: str = "language",
    lang_value: str = "english",
    encoding: str = "utf-8",
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str]]:
    """Load and combine title + body from a CSV file; filter by language."""
    if logger:
        logger.info(f"Loading CSV: {path}")

    df = pd.read_csv(path, encoding=encoding)
    for c in [id_col, title_col, body_col, lang_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    # Filter by language
    df = df[df[lang_col].astype(str).str.lower() == str(lang_value).lower()].copy()

    # Combine title and body
    def _combine(row) -> str:
        t = str(row.get(title_col, "") or "").strip()
        b = str(row.get(body_col, "") or "").strip()
        return f"{t}. {b}" if t and b else t or b

    texts = df.apply(_combine, axis=1).astype(str).tolist()
    doc_names = df[id_col].astype(str).tolist()
    return texts, doc_names


# --- 2) Named Entity Recognition (NER) ---
def extract_named_entities(texts: List[str], model: str = "en_core_web_sm") -> List[List[str]]:
    """
    Extract named entities (ORG, PERSON, GPE, PRODUCT...) for each document.
    Returns: list of entity lists aligned with texts.
    """
    nlp = spacy.load(model, disable=["parser", "tagger"])
    all_entities = []

    for doc in tqdm(nlp.pipe(texts, batch_size=16), total=len(texts), desc="Extracting NER"):
        entities = [
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT"}
        ]
        all_entities.append(list(set(entities)))  # deduplicate per document

    return all_entities


# --- 3) Build a document–document graph ---
def build_doc_graph(doc_names: List[str], doc_entities: List[List[str]]) -> nx.Graph:
    """
    Build a graph where:
      - each node = document
      - an edge connects two docs if they share ≥ 1 named entity
    Edge attributes:
      - 'weight': number of shared entities
      - 'entities': set of shared entity names
    """
    G = nx.Graph()
    for name in doc_names:
        G.add_node(name)

    # Build inverted index: entity -> [docs]
    entity_to_docs: Dict[str, List[str]] = {}
    for doc_id, entities in zip(doc_names, doc_entities):
        for e in entities:
            entity_to_docs.setdefault(e, []).append(doc_id)

    # Create edges for shared entities
    for entity, docs in tqdm(entity_to_docs.items(), desc="Building edges"):
        if len(docs) > 1:
            for d1, d2 in itertools.combinations(set(docs), 2):
                if G.has_edge(d1, d2):
                    G[d1][d2]["weight"] += 1
                    G[d1][d2]["entities"].add(entity)
                else:
                    G.add_edge(d1, d2, weight=1, entities={entity})
    return G


# --- 4) Example usage ---
if __name__ == "__main__":
    # Load the corpus
    texts, doc_names = load_corpus_from_csv("factchecks.csv")

    # Extract named entities
    doc_entities = extract_named_entities(texts, model="en_core_web_sm")

    # Build the document graph
    G = build_doc_graph(doc_names, doc_entities)

    # Display graph summary
    print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Example: show neighbors of one document
    example_doc = doc_names[0]
    print(f"\n🔹 Neighbors of {example_doc}:")
    for neighbor in list(G.neighbors(example_doc))[:5]:
        edge = G[example_doc][neighbor]
        print(f"  ↳ {neighbor} (shared entities: {list(edge['entities'])})")

    # Save for visualization (e.g., in Gephi)
    nx.write_gexf(G, "graph_rag_docdoc.gexf")
    print("💾 Graph saved to graph_rag_docdoc.gexf")
