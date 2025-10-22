import os
import sys
import numpy as np
import networkx as nx
import logging

# Import de la fonction utilitaire

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from io_utils import load_corpus_from_csv


from sentence_transformers import SentenceTransformer

def build_doc_graph_embeddings(
    csv_path,
    lang_value="english",
    model_name="all-MiniLM-L6-v2",
    top_k=5,
    sim_threshold=0.5,
    logger=None
):
    # 1. Charger les textes et ids
    texts, doc_names = load_corpus_from_csv(
        csv_path,
        lang_value=lang_value,
        logger=logger
    )

    # 2. Embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 3. Similarité cosinus
    S = np.matmul(embeddings, embeddings.T)
    np.fill_diagonal(S, -1)  # Ignore self-similarity

    # 4. Graphe
    G = nx.Graph()
    for i, doc_id in enumerate(doc_names):
        G.add_node(f"doc::{doc_id}", doc_id=doc_id)

    # 5. Ajout des arêtes kNN au-dessus du seuil
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
    
    # Chemin complet vers le fichier CSV
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "english_fact_checks.csv")
    
    G = build_doc_graph_embeddings(csv_path, logger=logger)
    print(f"Graphe construit: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    nx.write_gpickle(G, "doc_graph_embed.gpickle")