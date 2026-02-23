"""
Strategy 3: Thematic Graph Traversal Retrieval.

Uses topic modeling and BFS graph propagation to find relevant chunks
based on thematic similarity.

Scoring formula:
    score(chunk) = Σ_{t ∈ reachable_themes} (rel_t × w_HAS_TOPIC_{t→chunk})
                  / Σ_{t ∈ reachable_themes} rel_t                           ∈ [0, 1]

Naturally bounded: w_HAS_TOPIC ∈ (0, 1], so numerator ≤ Σ(rel_t) = denominator.
Interpretation: weighted-average HAS_TOPIC probability across all reachable themes,
weighted by their propagated relevance.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx
from bertopic import BERTopic

from fact_check_kg.retrieval.retrieval_propagation import propagate_relevance_themes


def retrieve_thematic(
    claim: str,
    graph: nx.Graph,
    topic_model: BERTopic,
    theme_threshold: float,
    theme_relevance_threshold: float,   # kept for API compatibility, unused
    logger: logging.Logger
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 3: Thematic Graph Traversal.

    Propagates relevance through the theme graph (CO_PRESENCE, SEMANTIC_SIM edges)
    starting from seed themes detected in the claim, then scores chunks via
    HAS_TOPIC edges.

    Scoring:
        score(chunk) = Σ_{t ∈ reachable_themes} (rel_t × w_HAS_TOPIC_{t→chunk})
                       / Σ_{t ∈ reachable_themes} rel_t                  ∈ [0, 1]

    where rel_t is the BFS-propagated relevance of theme t and w_HAS_TOPIC ∈ (0, 1].
    The formula is naturally bounded: numerator ≤ Σ(rel_t) = denominator, no clipping.

    Args:
        claim: Claim text to verify
        graph: Knowledge graph
        topic_model: Trained BERTopic model
        theme_threshold: Minimum probability to consider a theme detected
        theme_relevance_threshold: Minimum relevance to continue BFS traversal
        logger: Logger instance

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to thematic score in [0, 1]
          (only chunks with score > 0 are included)
        - detected_themes_info: List of dicts with theme details
          (label, keywords, probability)

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

    # Propagate relevance through theme graph (CO_PRESENCE, SEMANTIC_SIM edges)
    theme_relevance = propagate_relevance_themes(
        graph=graph,
        seed_themes=seed_themes,
        relevance_threshold=theme_relevance_threshold
    )

    logger.info(f"Relevance propagated to {len(theme_relevance)} themes")

    # --- Score chunks via HAS_TOPIC edges ---
    # score(chunk) = Σ_{t} (rel_t × w_HAS_TOPIC_{t→chunk}) / Σ_{t} rel_t  ∈ [0, 1]
    # Denominator = total propagated relevance = theoretical max any chunk can achieve.
    # Naturally bounded: w_HAS_TOPIC ∈ (0,1] → numerator ≤ Σ(rel_t) = denominator.
    sum_rel = sum(theme_relevance.values())
    scores: Dict[str, float] = {}
    chunk_contributions: Dict[str, List] = {}

    for theme_node, relevance in theme_relevance.items():
        for neighbor in graph.neighbors(theme_node):
            edge_data = graph[theme_node][neighbor]
            if (edge_data.get('edge_type') == 'HAS_TOPIC' and
                    graph.nodes[neighbor].get('type') == 'chunk'):
                w = edge_data.get('weight', 0.0)
                contribution = relevance * w
                scores[neighbor] = scores.get(neighbor, 0.0) + contribution
                chunk_contributions.setdefault(neighbor, []).append(
                    (theme_node, relevance, w, contribution)
                )

    scores = {k: v / sum_rel for k, v in scores.items()}

    logger.info(f"Found {len(scores)} chunks via thematic graph traversal")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.4f} | "
                    f"Mean: {sum(scores.values())/len(scores):.4f}")

        top_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 chunks (theme breakdown):")
        for chunk_id, score in top_chunks:
            logger.info(f"  {chunk_id} → score={score:.5f}")
            for theme, rel, w, contrib in chunk_contributions.get(chunk_id, [])[:3]:
                logger.info(f"    [{theme}] rel={rel:.4f} × w={w:.4f} = {contrib:.4f}")

    return scores, detected_themes_info
