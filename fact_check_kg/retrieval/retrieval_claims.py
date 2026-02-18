"""
Strategy 5: Claim-Based Document Retrieval.

Leverages pre-existing fact-checked claims to find relevant evidence chunks.
When a query is similar to a previously fact-checked claim, we can use
the documents that were used to fact-check that claim as relevant evidence.

Algorithm:
    1. Embed the query using SentenceTransformer
    2. Find top-k claims most similar to the query:
       similarity(q, c) = cos(embed(q), embed(c))
    3. For each matched claim, traverse REVIEWS edges to find linked documents,
       then CONTAINS edges to find chunks in those documents
    4. Score each chunk: S5(c) = max over matched claims of claim_similarity

Score is naturally bounded to [0, 1] (cosine similarity, clipped at 0).
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx


def retrieve_claim_document(
    query: str,
    graph: nx.Graph,
    embedding_model,
    claim_embeddings: Optional[np.ndarray],
    claim_nodes: List[str],
    claim_texts: List[str],
    claim_top_k: int,
    logger: logging.Logger
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 5: Claim-Based Retrieval via graph traversal (REVIEWS → CONTAINS).

    Finds claims semantically similar to the query, then traverses REVIEWS edges
    to linked documents, and CONTAINS edges to retrieve their chunks.

    Score formula:
        S5(chunk) = max over matched claims of: max(0, cos(embed(query), embed(claim)))

    Scores are bounded to [0, 1] — no min-max normalization applied.

    Args:
        query: Query text to match against claims
        graph: Knowledge graph (contains REVIEWS and CONTAINS edges)
        embedding_model: SentenceTransformer model
        claim_embeddings: Pre-computed claim embeddings (n_claims, dim) or None
        claim_nodes: List of claim node IDs
        claim_texts: List of claim texts corresponding to nodes
        claim_top_k: Number of top claims to consider
        logger: Logger instance

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to score ∈ [0, 1]
        - matched_claims_details: List of matched claim info
    """
    logger.info("\n--- Strategy 5: Claim-Based Retrieval (REVIEWS traversal) ---")

    matched_claims_details = []

    if claim_embeddings is None or len(claim_nodes) == 0:
        logger.warning("No claims in graph, skipping Strategy 5")
        return {}, matched_claims_details

    # Embed query
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Compute similarities with all claims
    similarities = np.matmul(claim_embeddings, query_embedding)

    # Get top-k claims
    top_indices = np.argsort(similarities)[-claim_top_k:][::-1]

    logger.info(f"Top {claim_top_k} claims by similarity to query:")
    scores = {}

    for idx in top_indices:
        claim_node = claim_nodes[idx]
        claim_sim = max(0.0, float(similarities[idx]))  # Clip to [0, 1]
        claim_text = claim_texts[idx]

        claim_text_preview = claim_text[:100] + "..." if len(claim_text) > 100 else claim_text
        logger.info(f"  {claim_node}: similarity={claim_sim:.3f}")
        logger.info(f"    Claim: {claim_text_preview}")

        # Traverse REVIEWS edges to find linked documents
        chunks_found = 0
        node_data = graph.nodes.get(claim_node, {})

        for neighbor in graph.neighbors(claim_node):
            edge_data = graph[claim_node][neighbor]
            if edge_data.get('edge_type') != 'REVIEWS':
                continue

            doc_node = neighbor

            # Traverse CONTAINS edges to find chunks of this document
            for doc_neighbor in graph.neighbors(doc_node):
                doc_edge_data = graph[doc_node][doc_neighbor]
                if (doc_edge_data.get('edge_type') == 'CONTAINS' and
                        graph.nodes[doc_neighbor].get('type') == 'chunk'):
                    chunk_id = doc_neighbor
                    # Score = claim similarity (take max across multiple claims)
                    if chunk_id not in scores:
                        scores[chunk_id] = claim_sim
                    else:
                        scores[chunk_id] = max(scores[chunk_id], claim_sim)
                    chunks_found += 1

        matched_claims_details.append({
            'claim_node': claim_node,
            'claim_text': claim_text,
            'query_similarity': claim_sim,
            'rating': node_data.get('rating', ''),
            'chunks_found': chunks_found
        })

        logger.info(f"    Found {chunks_found} chunks via REVIEWS traversal")

    logger.info(f"Found {len(scores)} chunks via claim-based retrieval")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")

    return scores, matched_claims_details
