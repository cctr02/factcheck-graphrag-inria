# file: graph_search.py
import argparse
import pickle
import sys
from typing import List, Tuple, Dict, Any

import networkx as nx
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print("Missing dependency. Install with: uv pip install sentence-transformers")
    raise


# -----------------------------
# Utilities
# -----------------------------
def get_doc_text(attrs: Dict[str, Any]) -> str:
    """Compose the textual representation for embedding (title + body)."""
    title = attrs.get("title") or ""
    body = attrs.get("body") or ""
    return (title + " " + body).strip()


def coerce_embedding(x) -> np.ndarray:
    """Turn a stored embedding attribute (list/tuple) into a numpy array."""
    if x is None:
        return None
    arr = np.array(x, dtype=np.float32)
    # handle nested lists if needed
    if arr.ndim > 1:
        arr = arr.squeeze()
    return arr



# -----------------------------
# Index building
# -----------------------------
def build_doc_matrix(
    G: nx.Graph,
    model: SentenceTransformer,
    prefer_stored_embeddings: bool = True,
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Collect document nodes, their texts, and an embedding matrix.
    If 'embedding' attribute exists on nodes and prefer_stored_embeddings=True, reuse it.
    Otherwise, encode from title+body.
    """
    docs = []
    for n, d in G.nodes(data=True):
        t = d.get("type")
        if (t == "document") or (isinstance(n, str) and n.startswith("doc::")):
            docs.append((n, d))

    if not docs:
        raise ValueError("No document nodes found. Ensure nodes have type='document' or 'doc::' prefix.")

    ids = [n for n, _ in docs]
    texts = [get_doc_text(d) for _, d in docs]

    # Try to reuse stored embeddings (if present and consistent)
    X = None
    if prefer_stored_embeddings:
        stored = [coerce_embedding(d.get("embedding")) for _, d in docs]
        if all((emb is not None and emb.ndim == 1) for emb in stored):
            try:
                X = np.vstack(stored)
                # normalize
                norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
                X = X / norms
            except Exception:
                X = None

    if X is None:
        # Compute from text
        X = model.encode(texts, normalize_embeddings=True)

    return ids, texts, X


# -----------------------------
# Similarity & seeds
# -----------------------------
def cosine_sim_single_to_matrix(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity where B is already normalized. We'll normalize 'a' here."""
    a = a / (np.linalg.norm(a) + 1e-9)
    return B @ a


def top_seeds_from_query(
    query: str,
    model: SentenceTransformer,
    doc_ids: List[str],
    doc_emb: np.ndarray,
    k: int = 5,
) -> List[Tuple[str, float]]:
    q = model.encode([query], normalize_embeddings=True)[0]
    sims = cosine_sim_single_to_matrix(q, doc_emb)
    idx = np.argsort(-sims)[:k]
    return [(doc_ids[i], float(sims[i])) for i in idx]


# -----------------------------
# Graph expansion & reranking
# -----------------------------
def k_hop_subgraph(
    G: nx.Graph,
    seeds: List[str],
    hops: int = 1,
    max_nodes: int = 2000,
) -> nx.Graph:
    """Undirected k-hop neighborhood around seed nodes."""
    SG_nodes = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        newf = set()
        for u in frontier:
            # use undirected neighborhood; both Graph and MultiGraph are fine
            newf.update(G.neighbors(u))
        SG_nodes.update(newf)
        frontier = newf
        if len(SG_nodes) >= max_nodes:
            break
    return G.subgraph(SG_nodes).copy()


def personalized_rerank(
    SG: nx.Graph,
    seed_scores: Dict[str, float],
    alpha: float = 0.7,
) -> List[Tuple[str, float, float, float]]:
    """
    Combine query similarity (seed_scores) and structure via Personalized PageRank.
    Returns a list of (node_id, fused_score, sim_component, pr_component) for document nodes.
    """
    # Keep documents only for scoring
    docs = [n for n, d in SG.nodes(data=True) if (d.get("type") == "document") or (isinstance(n, str) and n.startswith("doc::"))]
    if not docs:
        return []

    # Personalization: only documents, mass from seed similarity (fallback to a tiny mass if empty)
    pers = {n: max(seed_scores.get(n, 0.0), 0.0) for n in docs}
    if sum(pers.values()) == 0:
        pers[docs[0]] = 1.0

    # Undirected view & weighted PR (uses edge attribute 'weight' if present)
    H = SG.to_undirected()
    pr = nx.pagerank(H, alpha=0.85, personalization=pers, weight="weight")

    out = []
    for n in docs:
        sim = seed_scores.get(n, 0.0)
        prc = pr.get(n, 0.0)
        fused = alpha * sim + (1.0 - alpha) * prc
        out.append((n, fused, sim, prc))
    out.sort(key=lambda x: -x[1])
    return out


def search_graph(
    G: nx.Graph,
    model: SentenceTransformer,
    query: str,
    seed_k: int = 5,
    hops: int = 1,
    top_k: int = 10,
    alpha: float = 0.7,
    prefer_stored_embeddings: bool = True,
) -> Tuple[List[Dict[str, Any]], nx.Graph]:
    """End-to-end: seeds → k-hop subgraph → personalized rerank → top-K results."""
    doc_ids, _, doc_emb = build_doc_matrix(G, model, prefer_stored_embeddings=prefer_stored_embeddings)
    id2idx = {d: i for i, d in enumerate(doc_ids)}

    seeds = top_seeds_from_query(query, model, doc_ids, doc_emb, k=seed_k)
    seed_ids = [sid for sid, _ in seeds]
    seed_scores = {sid: sc for sid, sc in seeds}

    SG = k_hop_subgraph(G, seed_ids, hops=hops)

    # If a seed doc is outside SG (possible when hops=0), add tiny mass
    for sid in seed_ids:
        if sid not in SG:
            SG.add_node(sid, **G.nodes[sid])

    ranked = personalized_rerank(SG, seed_scores, alpha=alpha)

    results = []
    for n, fused, sim, prc in ranked[:top_k]:
        attrs = G.nodes[n]
        results.append(
            dict(
                node=n,
                score=fused,
                sim=sim,
                pr=prc,
                title=attrs.get("title"),
                url=attrs.get("url"),
                publisher=attrs.get("publisher_name") or attrs.get("publisher"),
                review_date=attrs.get("review_date"),
            )
        )
    return results, SG


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="GraphRAG document-only search over a stored NetworkX graph.")
    parser.add_argument("-g", "--graph", default="../doc_graph_embed.gpickle", help="Path to the .gpickle graph file.")
    parser.add_argument("-q", "--query", required=True, help="User query text.")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformers model name.")
    parser.add_argument("--seed-k", type=int, default=5, help="Number of nearest neighbors from the query.")
    parser.add_argument("--hops", type=int, default=1, help="k-hop expansion depth (1 or 2 recommended).")
    parser.add_argument("--top-k", type=int, default=8, help="How many results to print.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Fusion weight: alpha*similarity + (1-alpha)*personalized PageRank.")
    parser.add_argument("--no-stored-emb", action="store_true", help="Ignore stored node embeddings and recompute from text.")
    args = parser.parse_args()

    # Load graph
    try:
        with open(args.graph, "rb") as f:
            G = pickle.load(f)
    except FileNotFoundError:
        print(f"Graph not found at {args.graph}. Make sure the file exists (expected '../doc_graph_embed.gpickle').")
        sys.exit(1)

    # Load encoder
    model = SentenceTransformer(args.model)

    # Search
    results, SG = search_graph(
        G,
        model,
        query=args.query,
        seed_k=args.seed_k,
        hops=args.hops,
        top_k=args.top_k,
        alpha=args.alpha,
        prefer_stored_embeddings=not args.no_stored_emb,
    )

    # Print results
    if not results:
        print("No results found.")
        return

    print(f"\nTop {len(results)} results for: {args.query!r}")
    for i, r in enumerate(results, 1):
        line = f"{i:>2}. score={r['score']:.3f} (sim={r['sim']:.3f}, pr={r['pr']:.3f}) | {r.get('title') or r['node']}"
        if r.get("url"):
            line += f" | {r['url']}"
        print(line)


if __name__ == "__main__":
    main()
