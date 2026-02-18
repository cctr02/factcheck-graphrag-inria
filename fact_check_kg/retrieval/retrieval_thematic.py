"""
Strategy 3: Thematic Graph Traversal Retrieval.

Uses topic modeling and graph propagation to find relevant chunks
based on thematic similarity.
"""

import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import networkx as nx
from bertopic import BERTopic

from fact_check_kg.retrieval.retrieval_propagation import propagate_relevance_themes


def retrieve_thematic(
    claim: str,
    graph: nx.Graph,
    topic_model: BERTopic,
    theme_threshold: float,
    theme_relevance_threshold: float,
    logger: logging.Logger
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 3: Thematic Graph Traversal using topic model and graph traversal.

    Uses BFS to propagate relevance through theme graph (CO_PRESENCE, SEMANTIC_SIM edges),
    then scores chunks based on theme relevance weighted by probabilities from HAS_TOPIC edges.

    Args:
        claim: Claim text to verify
        graph: Knowledge graph
        topic_model: Trained BERTopic model
        theme_threshold: Minimum probability to consider a theme detected
        theme_relevance_threshold: Minimum relevance to continue traversal
        logger: Logger instance

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to thematic score (normalized 0-1)
        - detected_themes_info: List of dicts with theme details (label, keywords, probability)

    Example:
        >>> scores, themes = retrieve_thematic(
        ...     "Vaccines cause autism",
        ...     graph, topic_model, 0.1, 0.05, logger
        ... )
        >>> themes[0]['label']
        'Medical Misinformation'
    """
    logger.info("\n--- Strategy 3: Thematic Graph Traversal ---")

    # Classify claim into topics
    topic_probs = topic_model.transform([claim])[1][0]

    # Detect themes above threshold
    detected_themes = [
        (idx, prob)
        for idx, prob in enumerate(topic_probs)
        if prob >= theme_threshold
    ]

    # Fallback: if no theme passes the threshold, keep the top-1 theme
    if not detected_themes and len(topic_probs) > 0:
        best_idx = int(np.argmax(topic_probs))
        best_prob = float(topic_probs[best_idx])
        if best_prob > 0:
            detected_themes = [(best_idx, best_prob)]
            logger.info(f"No theme above threshold ({theme_threshold}), using top-1 fallback (prob={best_prob:.4f})")

    # Collect theme info for output (label, keywords, probability)
    detected_themes_info = []
    for idx, prob in sorted(detected_themes, key=lambda x: x[1], reverse=True):
        theme_node = f"theme::{idx}"
        node_data = graph.nodes.get(theme_node, {})
        detected_themes_info.append({
            'theme_node': theme_node,
            'topic_id': idx,
            'label': node_data.get('label', f'topic_{idx}'),
            'probability': round(float(prob), 4),
            'keywords': node_data.get('keywords', [])[:5]
        })

    logger.info(f"Detected {len(detected_themes)} themes in claim")
    for info in detected_themes_info:
        logger.info(f"  Theme {info['topic_id']} ({info['label']}): {info['probability']:.3f}")

    # Build seed theme list with initial probabilities
    seed_themes = []
    for theme_id, prob in detected_themes:
        theme_node = f"theme::{theme_id}"
        if graph.has_node(theme_node):
            seed_themes.append((theme_node, prob))
        else:
            logger.debug(f"Theme not in graph: {theme_node}")

    if not seed_themes:
        logger.warning("No themes from claim found in graph")
        return {}, detected_themes_info

    logger.info(f"Found {len(seed_themes)} seed themes in graph")

    # Propagate relevance through theme graph
    theme_relevance = propagate_relevance_themes(
        graph=graph,
        seed_themes=seed_themes,
        relevance_threshold=theme_relevance_threshold
    )

    logger.info(f"Relevance propagated to {len(theme_relevance)} themes")

    # Score chunks based on theme relevance and HAS_TOPIC edge weights (probabilities)
    scores = defaultdict(float)

    for theme_node, relevance in theme_relevance.items():
        # Find chunks connected via HAS_TOPIC edges
        for neighbor in graph.neighbors(theme_node):
            edge_data = graph[theme_node][neighbor]

            # Only consider HAS_TOPIC edges pointing to chunks
            if (edge_data.get('edge_type') == 'HAS_TOPIC' and
                graph.nodes[neighbor].get('type') == 'chunk'):

                # Get probability weight from HAS_TOPIC edge
                prob_weight = edge_data.get('weight', 0.0)

                # Chunk score = theme relevance × HAS_TOPIC probability
                chunk_score = relevance * prob_weight

                # Accumulate scores (chunks can be associated with multiple themes)
                scores[neighbor] += chunk_score

    # Normalize scores
    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
        if max_score > min_score:
            scores = {
                k: (v - min_score) / (max_score - min_score)
                for k, v in scores.items()
            }

    logger.info(f"Found {len(scores)} chunks via thematic graph traversal")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")

    return dict(scores), detected_themes_info
