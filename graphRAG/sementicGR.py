import os
import sys
import numpy as np
import networkx as nx
import logging
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from utils.io_utils import load_corpus_from_csv

from sentence_transformers import SentenceTransformer

def build_doc_graph_embeddings(
    csv_path,
    lang_value="english",
    model_name="all-MiniLM-L6-v2",
    top_k=5,
    sim_threshold=0.5,
    logger=None,
    max_docs=1000
):
    # 1. Load texts and ids
    texts, doc_names = load_corpus_from_csv(
        csv_path,
        lang_value=lang_value,
        logger=logger
    )

    if max_docs:
        print(f"Limiting to first {max_docs} documents")
        texts = texts[:max_docs]
        doc_names = doc_names[:max_docs]

    # 2. Embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 3. Cosine similarity
    S = np.matmul(embeddings, embeddings.T)
    np.fill_diagonal(S, -1)  # Ignore self-similarity

    # 4. Graph
    G = nx.Graph()
    for i, doc_id in enumerate(doc_names):
        G.add_node(f"doc::{doc_id}", doc_id=doc_id)

    # 5. Add kNN edges above threshold
    for i in range(len(doc_names)):
        sims = [(j, S[i, j]) for j in range(len(doc_names)) if j != i]
        sims.sort(key=lambda x: -x[1])
        for j, s in sims[:top_k]:
            if s >= sim_threshold:
                di = f"doc::{doc_names[i]}"
                dj = f"doc::{doc_names[j]}"
                if not G.has_edge(di, dj):
                    G.add_edge(di, dj, weight=float(s), type="embedding_cosine")
    return G

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("build_doc_graph_embed")
    
    # Complete path to CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "english_fact_checks.csv")
    
    G = build_doc_graph_embeddings(csv_path, logger=logger, max_docs=1000)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    
    with open("doc_graph_embed.gpickle", 'wb') as f:
        pickle.dump(G, f)
    print("Graph saved in doc_graph_embed.gpickle")