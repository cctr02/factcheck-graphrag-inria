"""
Entity node creation and edge module.

This module handles creation of:
- Entity nodes (NER-extracted named entities via spaCy)
- MENTIONS edges (Chunk -> Entity with TF-IDF weights)
- CO_OCCURS edges (Entity <-> Entity with PMI weights)
- SEMANTIC_SIM edges (Entity <-> Entity based on embeddings)
- TEMPORAL_DIST edges (DATE <-> DATE with exponential decay)

Entity Extraction:
    Uses spaCy NER to extract named entities of types:
    PERSON, ORG, GPE, LOC, DATE, EVENT, NORP

    DATE entities are normalized to 4-digit years (e.g., "January 2020" -> "2020")

MENTIONS Edge Weights (TF-IDF with Sigmoid):
    weight = σ((TF_normalized × IDF) / scale)

    Where:
    - TF_normalized = tf / max_tf_in_chunk (local normalization)
    - IDF = log(N / df) where N=total chunks, df=chunks containing entity
    - σ(x) = 1 / (1 + exp(-x)) (sigmoid normalization to [0,1])
    - scale = 3.0 (controls sigmoid steepness)

CO_OCCURS Edge Weights (PMI with Sigmoid):
    weight = σ(PMI / scale)

    Where:
    - PMI = log((co_occur × N) / (freq₁ × freq₂))
    - N = total documents
    - freq₁, freq₂ = document frequency of each entity
    - scale = 2.0

SEMANTIC_SIM Edge Weights:
    weight = cos(embed(e₁), embed(e₂))

    Where embed() uses SentenceTransformer on entity text.
    Only created if similarity >= threshold.

TEMPORAL_DIST Edge Weights (Exponential Decay):
    weight = exp(-λ × |year₁ - year₂|)

    Where λ = decay_rate (default 0.3).
    Only created if |year₁ - year₂| <= max_year_gap.
"""

import re
import time
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import networkx as nx


def extract_entities(
    text: str,
    nlp,
    entity_types: Set[str]
) -> List[Tuple[str, str]]:
    """
    Extract named entities from text using spaCy.

    For DATE entities, only extract the year (4-digit format).

    Args:
        text: Input text
        nlp: spaCy language model
        entity_types: Set of entity types to extract (e.g., {"PERSON", "ORG", "GPE", "DATE"})

    Returns:
        List of (entity_text, entity_type) tuples

    Example:
        >>> entities = extract_entities("Trump visited France in 2019", nlp, {"PERSON", "GPE", "DATE"})
        >>> entities
        [("trump", "PERSON"), ("france", "GPE"), ("2019", "DATE")]
    """
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        # Filter for relevant entity types
        if ent.label_ in entity_types:
            entity_text = ent.text.strip().lower()

            # Special handling for DATE: extract only the year
            if ent.label_ == "DATE":
                # Look for 4-digit year in the entity text
                year_match = re.search(r'\b(19|20)\d{2}\b', entity_text)
                if year_match:
                    year = year_match.group(0)
                    entities.append((year, "DATE"))
                # Skip dates without a clear year
            else:
                # For non-DATE entities, keep as is
                if len(entity_text) > 2:  # Filter very short entities
                    entities.append((entity_text, ent.label_))

    return entities


def add_entities_and_mentions(
    graph: nx.Graph,
    chunks: List[Dict],
    nlp,
    entity_types: Set[str],
    logger: logging.Logger,
    log_step_fn
) -> Dict[str, Set[str]]:
    """
    Extract entities and add MENTIONS edges with TF-IDF weights.

    Args:
        graph: NetworkX graph to add nodes/edges to
        chunks: List of chunk dictionaries with 'chunk_id' and 'text' keys
        nlp: spaCy language model
        entity_types: Set of entity types to extract
        logger: Logger instance
        log_step_fn: Function to log step completion (step_name, start_time, count)

    Returns:
        Dictionary mapping chunk_id to set of entity nodes
    """
    chunk_to_entities = defaultdict(set)
    # Track entity mentions: {entity_node: {chunk_id: count}}
    entity_mentions_in_chunks = defaultdict(lambda: defaultdict(int))

    # Step 1: Extract all entities and count term frequency (TF)
    # Use batch processing with nlp.pipe() for GPU acceleration
    start_time_extract = time.time()

    # Prepare texts for batch processing
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_ids_list = [chunk['chunk_id'] for chunk in chunks]

    # Determine batch size based on GPU availability
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = nlp.batch_size if hasattr(nlp, 'batch_size') else (1000 if device == 'cuda' else 50)

    logger.info(f"Extracting entities with {device.upper()} using batch_size={batch_size}...")

    # Batch process all texts with nlp.pipe() for better GPU utilization
    for chunk_id, doc in zip(
        chunk_ids_list,
        tqdm(
            nlp.pipe(chunk_texts, batch_size=batch_size),
            total=len(chunk_texts),
            desc="Extracting entities (batched)",
            disable=logger.level > logging.INFO
        )
    ):
        # Extract entities from spaCy doc
        entity_counts = defaultdict(int)
        for ent in doc.ents:
            # Filter for relevant entity types
            if ent.label_ in entity_types:
                entity_text = ent.text.strip().lower()

                # Special handling for DATE: extract only the year
                if ent.label_ == "DATE":
                    # Look for 4-digit year in the entity text
                    year_match = re.search(r'\b(19|20)\d{2}\b', entity_text)
                    if year_match:
                        year = year_match.group(0)
                        entity_node = f"entity::DATE::{year}"
                        entity_counts[entity_node] += 1
                        chunk_to_entities[chunk_id].add(entity_node)
                    # Skip dates without a clear year
                else:
                    # For non-DATE entities, keep as is
                    if len(entity_text) > 2:  # Filter very short entities
                        entity_node = f"entity::{ent.label_}::{entity_text}"
                        entity_counts[entity_node] += 1
                        chunk_to_entities[chunk_id].add(entity_node)

        # Store counts
        for entity_node, count in entity_counts.items():
            entity_mentions_in_chunks[entity_node][chunk_id] = count

    log_step_fn("Extracting entities (batched)", start_time_extract, len(chunks))

    # Step 2: Calculate IDF (Inverse Document Frequency)
    total_chunks = len(chunks)
    entity_idf = {}

    for entity_node, chunks_with_entity in entity_mentions_in_chunks.items():
        # IDF = log(total_chunks / number of chunks containing this entity)
        idf = np.log(total_chunks / len(chunks_with_entity))
        entity_idf[entity_node] = idf

    # Step 3: Calculate max_TF per chunk for local normalization
    chunk_max_tf = {}
    for chunk in chunks:
        chunk_id = chunk['chunk_id']
        # Get all TF values for entities in this chunk
        tf_values = []
        for entity_node, chunks_with_counts in entity_mentions_in_chunks.items():
            if chunk_id in chunks_with_counts:
                tf_values.append(chunks_with_counts[chunk_id])

        # Store max TF for this chunk
        chunk_max_tf[chunk_id] = max(tf_values) if tf_values else 1.0

    logger.info(f"Calculated max TF for {len(chunk_max_tf)} chunks")

    # Step 4: Add entity nodes and MENTIONS edges with hybrid sigmoid weights
    start_time_mentions = time.time()
    scale = 3.0  # Sigmoid scale parameter (controls curve steepness)

    for entity_node in tqdm(entity_mentions_in_chunks.keys(), desc="Adding entities and MENTIONS edges", disable=logger.level > logging.INFO):
        # Extract entity info from node name
        parts = entity_node.split("::")
        entity_type = parts[1]
        entity_text = "::".join(parts[2:])  # Handle entity text with "::"

        # Add entity node (if not exists)
        if not graph.has_node(entity_node):
            graph.add_node(
                entity_node,
                type="entity",
                entity_type=entity_type,
                entity_text=entity_text
            )

        # Add MENTIONS edges with hybrid sigmoid weight
        idf = entity_idf[entity_node]
        for chunk_id, tf in entity_mentions_in_chunks[entity_node].items():
            # Normalize TF locally within chunk
            max_tf_in_chunk = chunk_max_tf.get(chunk_id, 1.0)
            normalized_tf = tf / max_tf_in_chunk

            # Calculate raw weight (local importance × global rarity)
            raw_weight = normalized_tf * idf

            # Apply sigmoid transformation to [0, 1]
            weight = 1.0 / (1.0 + np.exp(-raw_weight / scale))

            graph.add_edge(
                chunk_id,
                entity_node,
                edge_type="MENTIONS",
                weight=float(weight)
            )

    log_step_fn("Adding entities and MENTIONS edges", start_time_mentions, len(entity_mentions_in_chunks))

    return chunk_to_entities


def add_entity_co_occurrences(
    graph: nx.Graph,
    chunk_to_entities: Dict[str, Set[str]],
    co_occur_threshold: int,
    logger: logging.Logger,
    log_step_fn
):
    """
    Add CO_OCCURS edges between entities with PMI-based weighting.

    Weight calculation uses Pointwise Mutual Information (PMI):
    - PMI = log((co_occur × total_docs) / (freq1 × freq2))
    - High PMI means entities co-occur more than expected by chance
    - Sigmoid applied to map PMI to [0, 1]

    Args:
        graph: NetworkX graph to add edges to
        chunk_to_entities: Mapping of chunk_id to set of entity nodes
        co_occur_threshold: Minimum co-occurrence count to create edge
        logger: Logger instance
        log_step_fn: Function to log step completion
    """
    co_occur_counter = Counter()
    entity_document_frequency = Counter()
    edges_skipped_date_date = 0

    # Step 1: Count co-occurrences and entity frequencies across documents
    doc_to_entities = defaultdict(set)

    # Group chunks by document
    for chunk_id, entities in chunk_to_entities.items():
        # Extract doc_id from chunk_id (format: "chunk::doc_id::chunk_idx")
        doc_id = chunk_id.split("::")[1]
        doc_to_entities[doc_id].update(entities)

    # Count entity frequencies at document level
    for doc_id, entities in doc_to_entities.items():
        for entity in entities:
            entity_document_frequency[entity] += 1

    # Count co-occurrences at document level (not chunk level)
    start_time_count = time.time()
    for doc_id, entities in tqdm(doc_to_entities.items(), desc="Counting entity co-occurrences by document", disable=logger.level > logging.INFO):
        entities_list = list(entities)
        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                pair = tuple(sorted([entities_list[i], entities_list[j]]))
                co_occur_counter[pair] += 1

    log_step_fn("Counting entity co-occurrences", start_time_count, len(doc_to_entities))

    # Step 2: Calculate weights
    if not co_occur_counter:
        logger.warning("No entity co-occurrences found")
        return

    # Add edges above threshold with PMI-based weighted score
    start_time_add = time.time()
    edges_added = 0
    total_docs = len(doc_to_entities)
    scale = 2.0  # Sigmoid scale parameter for PMI

    for (entity1, entity2), co_occur_count in tqdm(co_occur_counter.items(), desc="Adding CO_OCCURS edges", disable=logger.level > logging.INFO):
        if co_occur_count >= co_occur_threshold:
            # Skip DATE-DATE co-occurrences (they use TEMPORAL_DIST edges instead)
            entity1_type = graph.nodes[entity1].get('entity_type')
            entity2_type = graph.nodes[entity2].get('entity_type')

            if entity1_type == 'DATE' and entity2_type == 'DATE':
                edges_skipped_date_date += 1
                continue

            # Entity frequencies (number of documents they appear in)
            freq1 = entity_document_frequency[entity1]
            freq2 = entity_document_frequency[entity2]

            # Calculate PMI (Pointwise Mutual Information)
            # PMI = log((co_occur × total) / (freq1 × freq2))
            # Measures how much more often entities co-occur than expected by chance
            pmi = np.log((co_occur_count * total_docs) / (freq1 * freq2))

            # Apply sigmoid to map PMI to [0, 1]
            # High PMI (rare entities co-occurring) → high weight
            # Low/negative PMI (common entities, random co-occurrence) → low weight
            weight = 1.0 / (1.0 + np.exp(-pmi / scale))

            graph.add_edge(
                entity1,
                entity2,
                edge_type="CO_OCCURS",
                weight=float(weight),
                co_occur_count=int(co_occur_count),
                entity1_freq=int(freq1),
                entity2_freq=int(freq2),
                pmi=float(pmi)
            )
            edges_added += 1

    if edges_skipped_date_date > 0:
        logger.info(f"Skipped {edges_skipped_date_date} DATE-DATE co-occurrences (use TEMPORAL_DIST instead)")

    log_step_fn("Adding CO_OCCURS edges", start_time_add, edges_added)


def add_entity_semantic_similarity(
    graph: nx.Graph,
    embedding_model,
    semantic_sim_threshold: float,
    logger: logging.Logger,
    log_step_fn
):
    """
    Add SEMANTIC_SIM edges between non-DATE entities based on embeddings.

    Args:
        graph: NetworkX graph to add edges to
        embedding_model: Sentence transformer model for embeddings
        semantic_sim_threshold: Minimum similarity to create edge
        logger: Logger instance
        log_step_fn: Function to log step completion
    """
    start_time = time.time()

    # Get all entity nodes EXCEPT DATE entities
    entity_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get('type') == 'entity' and d.get('entity_type') != 'DATE'
    ]

    if len(entity_nodes) < 2:
        logger.info("Skipping entity semantic similarity (< 2 non-DATE entities)")
        return

    # Extract entity texts
    entity_texts = [
        graph.nodes[n]['entity_text']
        for n in entity_nodes
    ]

    # Embed entities (just the name, not context - Option B)
    logger.info(f"Embedding {len(entity_texts)} non-DATE entities...")

    # Use optimal batch size based on device
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128 if device == 'cuda' else 32

    embeddings = embedding_model.encode(
        entity_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device
    )

    # Compute similarity matrix
    similarity_matrix = np.matmul(embeddings, embeddings.T)
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity

    # Add edges above threshold
    edge_count = 0
    for i in tqdm(range(len(entity_nodes)), desc="Adding SEMANTIC_SIM edges", disable=logger.level > logging.INFO):
        for j in range(i + 1, len(entity_nodes)):
            sim = similarity_matrix[i, j]
            if sim >= semantic_sim_threshold:
                # Clamp to [0, 1] to fix floating-point rounding errors
                # If sim is slightly above 1.0 (e.g., 1.0000006), set to 1.0
                if sim > 1.0 and sim <= 1.001:
                    sim = 1.0
                elif sim > 1.001:
                    logger.warning(f"Similarity value {sim} > 1.001 detected, clamping to 1.0")
                    sim = 1.0

                graph.add_edge(
                    entity_nodes[i],
                    entity_nodes[j],
                    edge_type="SEMANTIC_SIM",
                    weight=float(sim),
                    semantic_type="entity"  # Mark as entity-entity similarity
                )
                edge_count += 1

    log_step_fn("Adding entity SEMANTIC_SIM edges", start_time, edge_count)


def add_date_temporal_similarity(
    graph: nx.Graph,
    max_year_gap: int,
    decay_rate: float,
    logger: logging.Logger,
    log_step_fn
):
    """
    Add TEMPORAL_DIST edges between year entities with exponential decay.

    Uses an exponential negative decay function:
    weight = exp(-decay_rate * year_diff)

    Where:
    - year_diff = |year2 - year1|
    - decay_rate controls how fast the weight decreases

    Examples (with decay_rate=0.3):
    - Same year (diff=0): weight = 1.0
    - 1 year diff: weight ≈ 0.74
    - 2 years diff: weight ≈ 0.55
    - 3 years diff: weight ≈ 0.41
    - 5 years diff: weight ≈ 0.22
    - 10 years diff: weight ≈ 0.05

    Args:
        graph: NetworkX graph to add edges to
        max_year_gap: Maximum year difference to create edges
        decay_rate: Decay rate for exponential function
        logger: Logger instance
        log_step_fn: Function to log step completion
    """
    start_time = time.time()

    # Get all DATE entity nodes (which are now years)
    date_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get('type') == 'entity' and d.get('entity_type') == 'DATE'
    ]

    if len(date_nodes) < 2:
        logger.info("Skipping date temporal similarity (< 2 years)")
        return

    logger.info(f"Processing temporal similarity for {len(date_nodes)} year entities...")
    logger.info(f"Using exponential decay: weight = exp(-{decay_rate} * year_diff), max_gap={max_year_gap}")

    # Extract years
    year_nodes = {}
    for node in date_nodes:
        year_text = graph.nodes[node]['entity_text']
        try:
            year = int(year_text)
            year_nodes[year] = node
        except ValueError:
            continue

    years_list = list(year_nodes.keys())

    # Connect all years within max_year_gap with exponential decay
    edge_count = 0
    for i in range(len(years_list)):
        for j in range(i + 1, len(years_list)):
            year1 = years_list[i]
            year2 = years_list[j]
            year_diff = abs(year2 - year1)

            # Only connect if within max_year_gap
            if year_diff <= max_year_gap:
                node1 = year_nodes[year1]
                node2 = year_nodes[year2]

                # Exponential decay: weight = exp(-decay_rate * year_diff)
                weight = float(np.exp(-decay_rate * year_diff))

                graph.add_edge(
                    node1,
                    node2,
                    edge_type="TEMPORAL_DIST",
                    weight=weight,
                    year_diff=year_diff
                )
                edge_count += 1

    log_step_fn("Adding TEMPORAL_DIST edges (exponential decay)", start_time, edge_count)
