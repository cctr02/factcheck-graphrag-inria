"""
Strategy 4: Entity Graph Traversal Retrieval.

Uses named entity recognition and graph propagation to find relevant chunks
based on entity co-occurrence patterns.
"""

import re
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import networkx as nx
import spacy

from fact_check_kg.retrieval.retrieval_propagation import propagate_relevance_entities


def extract_entities(
    text: str,
    nlp: spacy.Language,
    entity_types: Set[str]
) -> List[Tuple[str, str]]:
    """
    Extract named entities from text using spaCy.

    For DATE entities, only extract the year (4-digit format).

    Args:
        text: Input text
        nlp: spaCy language model
        entity_types: Set of entity types to extract

    Returns:
        List of (entity_text, entity_type) tuples
    """
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in entity_types:
            entity_text = ent.text.strip().lower()

            if ent.label_ == "DATE":
                year_match = re.search(r'\b(19|20)\d{2}\b', entity_text)
                if year_match:
                    year = year_match.group(0)
                    entities.append((year, "DATE"))
            else:
                if len(entity_text) > 2:
                    entities.append((entity_text, ent.label_))

    return entities


def retrieve_entity(
    claim: str,
    graph: nx.Graph,
    nlp: spacy.Language,
    entity_types: Set[str],
    entity_relevance_threshold: float,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Strategy 4: Entity Graph Traversal using graph traversal with relevance propagation.

    Uses BFS to propagate relevance through entity graph (CO_OCCURS, SEMANTIC_SIM edges),
    then scores chunks based on entity relevance weighted by TF-IDF from MENTIONS edges.

    Args:
        claim: Claim text to verify
        graph: Knowledge graph
        nlp: spaCy language model
        entity_types: Set of entity types to extract
        entity_relevance_threshold: Minimum relevance to continue traversal
        logger: Logger instance

    Returns:
        Dictionary mapping chunk_id to entity score (normalized 0-1)

    Example:
        >>> scores = retrieve_entity(
        ...     "Trump visited France",
        ...     graph, nlp, {"PERSON", "GPE"}, 0.1, logger
        ... )
    """
    logger.info("\n--- Strategy 4: Entity Graph Traversal ---")

    # Extract entities from claim
    entities_claim = extract_entities(claim, nlp, entity_types)

    logger.info(f"Extracted {len(entities_claim)} entities from claim:")
    for ent_text, ent_type in entities_claim:
        logger.info(f"  {ent_type}: {ent_text}")

    # Build seed entity list (only entities present in graph)
    seed_entities = []
    for entity_text, entity_type in entities_claim:
        entity_node = f"entity::{entity_type}::{entity_text}"
        if graph.has_node(entity_node):
            seed_entities.append(entity_node)
        else:
            logger.debug(f"Entity not in graph: {entity_node}")

    if not seed_entities:
        logger.warning("No entities from claim found in graph")
        return {}

    logger.info(f"Found {len(seed_entities)} seed entities in graph")

    # Propagate relevance through entity graph
    entity_relevance = propagate_relevance_entities(
        graph=graph,
        seed_entities=seed_entities,
        relevance_threshold=entity_relevance_threshold
    )

    logger.info(f"Relevance propagated to {len(entity_relevance)} entities")

    # Score chunks based on entity relevance and MENTIONS edge weights (TF-IDF)
    scores = defaultdict(float)

    for entity_node, relevance in entity_relevance.items():
        # Find chunks connected via MENTIONS edges
        for neighbor in graph.neighbors(entity_node):
            edge_data = graph[entity_node][neighbor]

            if (edge_data.get('edge_type') == 'MENTIONS' and
                graph.nodes[neighbor].get('type') == 'chunk'):

                # Get TF-IDF weight from MENTIONS edge
                tf_idf_weight = edge_data.get('weight', 1.0)

                # Chunk score = entity relevance × TF-IDF weight
                chunk_score = relevance * tf_idf_weight

                # Accumulate scores (chunks can be mentioned by multiple entities)
                scores[neighbor] += chunk_score

    # Normalize by number of seed entities (theoretical max).
    # Each seed contributes at most 1.0 × MENTIONS_weight ≤ 1.0 per chunk,
    # so dividing by n_seeds guarantees scores ∈ [0, 1].
    n_seeds = len(seed_entities)
    if scores and n_seeds > 0:
        scores = {k: min(v / n_seeds, 1.0) for k, v in scores.items()}

    logger.info(f"Found {len(scores)} chunks via entity graph traversal")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")

    return dict(scores)
