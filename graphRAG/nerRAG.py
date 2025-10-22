# ============================================
#  Minimal GraphRAG pipeline:
#  - Nodes: documents
#  - Edge: shared named entities between docs
# ============================================

from typing import List, Dict
import itertools
import networkx as nx
import spacy
import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.io_utils import load_corpus_from_csv
from utils.logger import setup_logger
from utils.helpers import maybe_tqdm

logger = setup_logger("graphRAG/nerRAG.log", log_level="INFO")

def extract_named_entities(texts: List[str], model: str = "en_core_web_sm") -> List[List[str]]:
    """
    Extract named entities (ORG, PERSON, GPE, PRODUCT...) for each document.
    Returns: list of entity lists in a doc aligned with texts.
    """
    nlp = spacy.load(model, disable=["parser", "tagger"])
    all_entities = []

    for doc in maybe_tqdm(
        nlp.pipe(texts, batch_size=16),
        enable=True,
        total=len(texts),
        desc="Extracting NER"
    ):
        entities = [
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT"} 
        ]
        all_entities.append(list(set(entities)))  # deduplicate per document

    return all_entities

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
    for entity, docs in maybe_tqdm(entity_to_docs.items(),enable=True, desc="Building edges"):
        if len(docs) > 1:
            for d1, d2 in itertools.combinations(set(docs), 2):
                if G.has_edge(d1, d2):
                    G[d1][d2]["weight"] += 1
                    G[d1][d2]["entities"].add(entity)
                else:
                    G.add_edge(d1, d2, weight=1, entities={entity})
    return G

if __name__ == "__main__":
    texts, doc_names = load_corpus_from_csv(path="data/english_fact_checks.csv", n_rows=1000, logger= logger)
    doc_entities = extract_named_entities(texts, model="en_core_web_sm")

    # Build the document graph
    G = build_doc_graph(doc_names, doc_entities)

    # Display graph summary
    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Example: show neighbors of one document
    example_doc = doc_names[0]
    logger.info(f"\n Neighbors of {example_doc}:")
    for neighbor in list(G.neighbors(example_doc))[:5]:
        edge = G[example_doc][neighbor]
        logger.info(f"  {neighbor} (shared entities: {list(edge['entities'])})")

    # Convert non-serializable attribute types (set/list) to a serializable string before writing GEXF
    for u, v, data in G.edges(data=True):
        ents = data.get("entities")
        if isinstance(ents, (set, list)):
            # deterministic ordering + comma-separated string for GEXF compatibility
            data["entities"] = ", ".join(sorted(list(ents)))

    graph_output_path = "graphRAG/nerRAG.gpickle"
    with open(graph_output_path, 'wb') as f:
        pickle.dump(G, f)
    logger.info(f"Graph saved to {graph_output_path}")
