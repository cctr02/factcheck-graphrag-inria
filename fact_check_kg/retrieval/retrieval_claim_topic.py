"""
Strategy 6: Claim-Based Topic Distribution Traversal.

Combines claim-based retrieval with topic distribution analysis.
Uses matched claims to traverse the graph and scores chunks based on
topic distribution similarity weighted by claim similarity.

Algorithm:
1. Find claims similar to query (cosine similarity)
2. Traverse REVIEWS → CONTAINS to get chunks linked to matched claims
3. Compute topic distribution similarity between query and each chunk
4. Score = claim_sim × topic_sim × topic_confidence

Final Score:
    S6(chunk) = claim_sim × topic_sim × topic_confidence

Where:
- claim_sim = max(0, cos(embed(query), embed(claim)))
- topic_sim = 1 - JSD(query_distribution, chunk_distribution)
- topic_confidence = max(HAS_TOPIC_weight for chunk's topics), default 0.5

All components ∈ [0, 1], so the product is ∈ [0, 1]. No min-max normalization.
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import networkx as nx
from scipy.spatial.distance import jensenshannon
from bertopic import BERTopic

from fact_check_kg.retrieval.retrieval_claims import expand_claims_via_embeddings


def retrieve_claim_topic_traversal(
    query: str,
    graph: nx.Graph,
    topic_model: BERTopic,
    embedding_model,
    chunk_topic_distributions: np.ndarray,
    chunk_to_dist_idx: Dict[str, int],
    claim_embeddings: Optional[np.ndarray],
    claim_nodes: List[str],
    claim_texts: List[str],
    claim_top_k: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 6: Claim-Based Topic Distribution Traversal via REVIEWS graph edges.

    Finds similar claims, traverses REVIEWS → CONTAINS to reach chunks,
    and scores based on topic distribution similarity weighted by claim similarity
    and topic confidence (HAS_TOPIC edge weight).

    Score Formula:
        score = claim_sim × topic_sim × topic_conf

    Scores are bounded to [0, 1] — no min-max normalization applied.

    Args:
        query: Query text
        graph: Knowledge graph
        topic_model: Trained BERTopic model
        embedding_model: SentenceTransformer for embeddings
        chunk_topic_distributions: Pre-computed topic distributions (n_chunks, n_topics)
        chunk_to_dist_idx: Mapping chunk_id -> index in distributions array
        claim_embeddings: Pre-computed claim embeddings
        claim_nodes: List of claim node IDs
        claim_texts: List of claim texts
        claim_top_k: Number of top claims to consider
        logger: Logger instance

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to score ∈ [0, 1]
        - matched_claims_details: Details about matched claims
    """
    logger.info("\n--- Strategy 6: Claim-Based Topic Distribution Traversal (+ CLAIM_SIM expansion) ---")

    matched_claims_details = []

    if claim_embeddings is None or len(claim_nodes) == 0:
        logger.warning("No claims in graph, skipping Strategy 6")
        return {}, matched_claims_details

    # Step 1: Embed query and find similar claims
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    claim_similarities = np.matmul(claim_embeddings, query_embedding)
    top_claim_indices = np.argsort(claim_similarities)[-claim_top_k:][::-1]

    # Expand claims via embedding-based claim-to-claim similarity
    logger.info(f"Claim matching (direct top-{claim_top_k} + embedding expansion):")
    direct_claims, expanded_claims = expand_claims_via_embeddings(
        top_indices=list(top_claim_indices),
        claim_nodes=claim_nodes,
        similarities=claim_similarities,
        claim_embeddings=claim_embeddings,
        max_expanded=claim_top_k,
        expansion_threshold=0.0,
        graph=graph,
        logger=logger,
    )
    all_claims_to_process = direct_claims + expanded_claims

    # Step 2: Get query topic distribution
    _, query_topic_probs = topic_model.transform([query])
    query_distribution = query_topic_probs[0]

    logger.info(f"Query topic distribution computed (shape: {query_distribution.shape})")

    # Step 3: Process each matched claim (direct + expanded)
    scores = {}

    for claim_node, claim_sim in all_claims_to_process:
        node_data = graph.nodes.get(claim_node, {})
        claim_text = node_data.get('text', '')
        chunks_found = 0

        # Step 4: Traverse REVIEWS → CONTAINS to find chunks linked to this claim
        for neighbor in graph.neighbors(claim_node):
            edge_data = graph[claim_node][neighbor]
            if edge_data.get('edge_type') != 'REVIEWS':
                continue
            doc_node = neighbor

            for doc_neighbor in graph.neighbors(doc_node):
                doc_edge_data = graph[doc_node][doc_neighbor]
                if (doc_edge_data.get('edge_type') != 'CONTAINS' or
                        graph.nodes[doc_neighbor].get('type') != 'chunk'):
                    continue

                chunk_id = doc_neighbor

                if chunk_id not in chunk_to_dist_idx:
                    continue

                # Compute topic distribution similarity (1 - JSD)
                dist_idx = chunk_to_dist_idx[chunk_id]
                chunk_distribution = chunk_topic_distributions[dist_idx]
                js_divergence = jensenshannon(query_distribution, chunk_distribution)
                topic_similarity = 1.0 - float(js_divergence)

                # Get topic confidence (max HAS_TOPIC weight for this chunk)
                topic_confidence = 0.5  # default when no HAS_TOPIC found
                if graph.has_node(chunk_id):
                    for c_neighbor in graph.neighbors(chunk_id):
                        c_edge_data = graph[chunk_id][c_neighbor]
                        if c_edge_data.get('edge_type') == 'HAS_TOPIC':
                            topic_confidence = max(topic_confidence, c_edge_data.get('weight', 0.0))

                # Score = claim_sim × topic_sim × topic_conf ∈ [0, 1]
                final_score = claim_sim * topic_similarity * topic_confidence

                if chunk_id not in scores or final_score > scores[chunk_id]:
                    scores[chunk_id] = final_score
                chunks_found += 1

        matched_claims_details.append({
            'claim_node': claim_node,
            'claim_text': claim_text,
            'query_similarity': claim_sim,
            'rating': node_data.get('rating', ''),
            'chunks_found': chunks_found,
            'source': 'direct' if any(cn == claim_node for cn, _ in direct_claims) else 'graph'
        })

    logger.info(f"Total matched claims: {len(all_claims_to_process)} "
                f"({len(direct_claims)} direct + {len(expanded_claims)} via CLAIM_SIM)")
    logger.info(f"Found {len(scores)} chunks via claim-topic traversal")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")
        logger.info(f"Mean score: {np.mean(list(scores.values())):.3f}")

    return scores, matched_claims_details
