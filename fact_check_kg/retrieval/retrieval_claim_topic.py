"""
Strategy 6: Claim-Based Topic Distribution Traversal.

Combines claim-based retrieval with topic distribution analysis.
Uses matched claims to traverse the graph and scores chunks based on
topic distribution similarity weighted by traversal relevance.

Algorithm:
1. Find claims similar to query (via S5)
2. Get chunks connected to claims via graph traversal
3. Compute topic distribution similarity between query and each chunk
4. Weight by relevance propagation from claim to chunk
5. Boost chunks with high HAS_TOPIC confidence (TF-IDF-like weighting)

Final Score:
    S6(chunk) = claim_sim × traversal_relevance × topic_sim × topic_confidence

Where:
- claim_sim = cos(embed(query), embed(claim))
- traversal_relevance = propagated relevance from claim to chunk
- topic_sim = 1 - JSD(query_dist, chunk_dist)
- topic_confidence = max(HAS_TOPIC_weight for chunk's topics)
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import networkx as nx
from scipy.spatial.distance import jensenshannon
from bertopic import BERTopic


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
    claim_retrieval_results: Dict,
    claim_top_k: int,
    logger: logging.Logger
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 6: Claim-Based Topic Distribution Traversal.

    Finds similar claims, traverses to related chunks, and scores based on
    topic distribution similarity weighted by traversal relevance and
    topic confidence (HAS_TOPIC edge weight).

    Score Formula:
        score = claim_sim × traversal_rel × (1 - JSD) × topic_conf

    Where:
    - claim_sim: Cosine similarity between query and matched claim
    - traversal_rel: Relevance score from claim's pre-computed retrieval
    - (1 - JSD): Topic distribution similarity (Jensen-Shannon)
    - topic_conf: Maximum HAS_TOPIC edge weight for the chunk

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
        claim_retrieval_results: Pre-computed retrieval results per claim
        claim_top_k: Number of top claims to consider
        logger: Logger instance

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to score (normalized 0-1)
        - matched_claims_details: Details about matched claims

    Example:
        >>> scores, details = retrieve_claim_topic_traversal(
        ...     "COVID vaccines are dangerous",
        ...     graph, topic_model, embed_model, chunk_dists, idx_map,
        ...     claim_embeds, claim_nodes, claim_texts, claim_results, 5, logger
        ... )
    """
    logger.info("\n--- Strategy 6: Claim-Based Topic Distribution Traversal ---")

    matched_claims_details = []

    # Check if claims exist
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

    # Step 2: Get query topic distribution
    _, query_topic_probs = topic_model.transform([query])
    query_distribution = query_topic_probs[0]

    logger.info(f"Query topic distribution computed (shape: {query_distribution.shape})")

    # Step 3: Process each matched claim
    scores = defaultdict(float)

    for claim_idx in top_claim_indices:
        claim_node = claim_nodes[claim_idx]
        claim_sim = float(claim_similarities[claim_idx])
        claim_text = claim_texts[claim_idx]

        logger.info(f"Processing claim: {claim_node} (similarity: {claim_sim:.3f})")

        # Get pre-computed retrieval results for this claim
        if claim_node not in claim_retrieval_results:
            logger.debug(f"No pre-computed results for {claim_node}")
            continue

        claim_data = claim_retrieval_results[claim_node]
        retrieval_results = claim_data.get('retrieval_results', [])

        # Store claim details
        node_data = graph.nodes.get(claim_node, {})
        matched_claims_details.append({
            'claim_node': claim_node,
            'claim_text': claim_text,
            'query_similarity': claim_sim,
            'rating': node_data.get('rating', ''),
            'chunks_found': len(retrieval_results)
        })

        # Step 4: Score each chunk from claim traversal
        for result in retrieval_results:
            chunk_id = result['chunk_id']
            traversal_relevance = result['score_total']  # Pre-computed relevance

            # Skip if chunk not in distribution mapping
            if chunk_id not in chunk_to_dist_idx:
                continue

            # Get chunk topic distribution
            dist_idx = chunk_to_dist_idx[chunk_id]
            chunk_distribution = chunk_topic_distributions[dist_idx]

            # Compute topic distribution similarity (1 - JSD)
            js_divergence = jensenshannon(query_distribution, chunk_distribution)
            topic_similarity = 1.0 - js_divergence

            # Get topic confidence (max HAS_TOPIC weight for this chunk)
            topic_confidence = 0.0
            if graph.has_node(chunk_id):
                for neighbor in graph.neighbors(chunk_id):
                    edge_data = graph[chunk_id][neighbor]
                    if edge_data.get('edge_type') == 'HAS_TOPIC':
                        weight = edge_data.get('weight', 0.0)
                        topic_confidence = max(topic_confidence, weight)

            # If no topics found, use default
            if topic_confidence == 0.0:
                topic_confidence = 0.5

            # Compute final score
            # score = claim_sim × traversal_rel × topic_sim × topic_conf
            final_score = claim_sim * traversal_relevance * topic_similarity * topic_confidence

            # Keep maximum score across all claims
            if chunk_id not in scores or final_score > scores[chunk_id]:
                scores[chunk_id] = final_score

    # Normalize scores (min-max)
    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
        if max_score > min_score:
            scores = {
                k: (v - min_score) / (max_score - min_score)
                for k, v in scores.items()
            }

    logger.info(f"Found {len(scores)} chunks via claim-topic traversal")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")
        logger.info(f"Mean score: {np.mean(list(scores.values())):.3f}")

    return dict(scores), matched_claims_details
