"""
Strategy 2: Topic Distribution Similarity Retrieval.

Uses Jensen-Shannon divergence to compare topic probability distributions
between the claim and chunks.
"""

import logging
from typing import Dict
import numpy as np
from scipy.spatial.distance import jensenshannon
from bertopic import BERTopic
import networkx as nx


def retrieve_topic_distribution(
    claim: str,
    graph: nx.Graph,
    topic_model: BERTopic,
    chunk_topic_distributions: np.ndarray,
    chunk_to_dist_idx: Dict[str, int],
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Strategy 2: Topic Distribution Similarity using Jensen-Shannon divergence.

    Compares the topic probability distribution of the claim with each chunk's
    topic distribution. Uses Jensen-Shannon divergence (symmetric, bounded [0,1]).

    Args:
        claim: Claim text to verify
        graph: Knowledge graph (for theme labels)
        topic_model: Trained BERTopic model
        chunk_topic_distributions: Pre-computed chunk distributions (n_chunks, n_topics)
        chunk_to_dist_idx: Mapping chunk_id -> index in distributions array
        logger: Logger instance

    Returns:
        Dictionary mapping chunk_id to distribution similarity score (normalized 0-1)

    Example:
        >>> scores = retrieve_topic_distribution(
        ...     "COVID vaccines are dangerous",
        ...     graph, topic_model, distributions, idx_map, logger
        ... )
    """
    logger.info("\n--- Strategy 2: Topic Distribution Similarity ---")

    # Get topic distribution for claim
    _, claim_topic_probs = topic_model.transform([claim])
    claim_distribution = claim_topic_probs[0]

    logger.info(f"Claim topic distribution shape: {claim_distribution.shape}")

    # Show top topics in claim
    top_topics = np.argsort(claim_distribution)[-5:][::-1]
    logger.info("Top 5 topics in claim:")
    for topic_id in top_topics:
        if claim_distribution[topic_id] > 0.01:
            theme_label = graph.nodes.get(f"theme::{topic_id}", {}).get('label', f'topic_{topic_id}')
            logger.info(f"  Topic {topic_id} ({theme_label}): {claim_distribution[topic_id]:.3f}")

    # Compute Jensen-Shannon divergence for each chunk
    scores = {}
    for chunk_id, dist_idx in chunk_to_dist_idx.items():
        chunk_distribution = chunk_topic_distributions[dist_idx]

        # Jensen-Shannon divergence (returns value in [0, 1])
        # Lower divergence = more similar distributions
        js_divergence = jensenshannon(claim_distribution, chunk_distribution)

        # Convert to similarity score (1 - divergence)
        # So higher score = more similar
        similarity = 1.0 - js_divergence

        scores[chunk_id] = float(similarity)

    # Scores are already in [0, 1] (1 - JSD where JSD ∈ [0, 1])
    # No normalization needed

    logger.info(f"Found {len(scores)} chunks with topic distribution scores")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")
        logger.info(f"Mean score: {np.mean(list(scores.values())):.3f}")

    return scores
