"""
Strategy 4: Entity Mention Retrieval.

Uses named entity recognition to find chunks that directly mention the entities
extracted from the claim.

Scoring formula:
    score(chunk) = Σ(mentions_weight_{seed→chunk}) / n_seed_entities  ∈ [0, 1]

Only seed entities (entities extracted directly from the claim and found in the
graph) contribute to chunk scores.  A chunk that does not mention any seed entity
gets score = 0.  The MENTIONS edge weight (chunk-local TF-IDF, set at build time)
captures how characteristic the entity is to that specific chunk.
"""

import re
import logging
from typing import Dict, List, Tuple, Set
import networkx as nx
import spacy


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
    entity_relevance_threshold: float,   # kept for API compatibility, unused
    logger: logging.Logger
) -> Tuple[Dict[str, float], Dict]:
    """
    Strategy 4: Entity Mention Retrieval.

    Scores chunks based on how well they mention the named entities extracted
    from the claim.  Only seed entities (extracted from the claim and present
    in the graph) contribute to scores.

    Scoring:
        score(chunk) = Σ_{s in seeds} w_{s→chunk} / n_seeds   ∈ (0, 1)

    where w_{s→chunk} is the MENTIONS edge weight.  This weight is computed
    at build time via sigmoid(TF_normalized × IDF / scale), so it is always
    strictly in (0, 1).  Therefore the formula is naturally bounded without
    any clipping.
    Chunks that mention no seed entity receive score = 0 (absent from dict).

    Args:
        claim: Claim text to verify
        graph: Knowledge graph
        nlp: spaCy language model
        entity_types: Set of entity types to extract
        entity_relevance_threshold: Unused (kept for API compatibility)
        logger: Logger instance

    Returns:
        Tuple of:
        - Dictionary mapping chunk_id to entity score in [0, 1]
          (only chunks with score > 0 are included)
        - entity_info dict with keys:
            'extracted'      – list of (text, type) from the claim
            'found_in_graph' – seed entity node IDs present in graph
            'not_in_graph'   – entity node IDs missing from graph

    Example:
        >>> scores, info = retrieve_entity(
        ...     "Trump visited France",
        ...     graph, nlp, {"PERSON", "GPE"}, 0.1, logger
        ... )
    """
    logger.info("\n--- Strategy 4: Entity Mention Retrieval ---")

    # --- Entity extraction ---
    entities_claim = extract_entities(claim, nlp, entity_types)

    logger.info(f"Extracted {len(entities_claim)} entities from claim:")
    for ent_text, ent_type in entities_claim:
        logger.info(f"  {ent_type}: {ent_text}")

    # --- Build seed list (entities present in graph) ---
    seed_entities = []
    for entity_text, entity_type in entities_claim:
        entity_node = f"entity::{entity_type}::{entity_text}"
        if graph.has_node(entity_node):
            seed_entities.append(entity_node)

    not_found = [f"entity::{et}::{ev}" for ev, et in entities_claim
                 if f"entity::{et}::{ev}" not in seed_entities]

    entity_info = {
        'extracted': entities_claim,
        'found_in_graph': seed_entities,
        'not_in_graph': not_found,
    }

    if not seed_entities:
        logger.warning("No entities from claim found in graph")
        if not_found:
            logger.info(f"  Entities NOT in graph ({len(not_found)}):")
            for nf in not_found:
                logger.info(f"    (missing) {nf}")
        return {}, entity_info

    logger.info(f"Found {len(seed_entities)} seed entities in graph:")
    for node in seed_entities:
        n_ch = sum(1 for nbr in graph.neighbors(node)
                   if graph[node][nbr].get('edge_type') == 'MENTIONS'
                   and graph.nodes[nbr].get('type') == 'chunk')
        weights = [graph[node][nbr].get('weight', 1.0)
                   for nbr in graph.neighbors(node)
                   if graph[node][nbr].get('edge_type') == 'MENTIONS'
                   and graph.nodes[nbr].get('type') == 'chunk']
        avg_w = sum(weights) / len(weights) if weights else 0.0
        logger.info(f"  {node} | chunks_mentioning={n_ch} | avg_mentions_weight={avg_w:.4f}")

    if not_found:
        logger.info(f"  Entities NOT in graph ({len(not_found)}):")
        for nf in not_found:
            logger.info(f"    (missing) {nf}")

    # --- Score chunks from seed entity MENTIONS only ---
    # score(chunk) = Σ(w_{seed→chunk}) / n_seeds  ∈ [0, 1]
    # Chunks not mentioned by any seed entity get score = 0 (excluded from dict).
    n_seeds = len(seed_entities)
    scores: Dict[str, float] = {}
    chunk_contributions: Dict[str, List] = {}

    for seed_entity in seed_entities:
        for neighbor in graph.neighbors(seed_entity):
            edge_data = graph[seed_entity][neighbor]
            if (edge_data.get('edge_type') == 'MENTIONS' and
                    graph.nodes[neighbor].get('type') == 'chunk'):
                w = edge_data.get('weight', 1.0)
                scores[neighbor] = scores.get(neighbor, 0.0) + w
                chunk_contributions.setdefault(neighbor, []).append(
                    (seed_entity, w)
                )

    scores = {k: v / n_seeds for k, v in scores.items()}

    logger.info(f"Found {len(scores)} chunks mentioning at least one seed entity")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.4f} | "
                    f"Mean: {sum(scores.values())/len(scores):.4f}")

        top_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 chunks (seed entity breakdown):")
        for chunk_id, score in top_chunks:
            logger.info(f"  {chunk_id} → score={score:.5f}")
            for ent, w in chunk_contributions.get(chunk_id, []):
                logger.info(f"    [{ent}] w={w:.4f}")

    return scores, entity_info
