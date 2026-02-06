"""
Theme node creation and edge module.

This module handles creation of:
- Theme nodes (BERTopic topics with LLM-generated labels)
- Supertopic nodes (hierarchical groupings of themes via c-TF-IDF merging)
- HAS_TOPIC edges (Chunk -> Theme with probability weights)
- CONTAINS_TOPIC edges (Supertopic -> Theme with c-TF-IDF similarity weights)
- SEMANTIC_SIM edges (Theme <-> Theme based on keyword embeddings)
- CO_PRESENCE edges (Theme <-> Theme based on chunk co-occurrence)

Hierarchical Structure:
    BERTopic uses hierarchical agglomerative clustering on topic c-TF-IDF
    representations to create supertopics. The merge distance is:

        distance(t_i, t_j) = 1 - cosine_sim(c-TF-IDF(t_i), c-TF-IDF(t_j))

    Where c-TF-IDF is the class-based TF-IDF representation of each topic.
    Supertopics group semantically similar topics together.
"""

import time
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import networkx as nx
from bertopic import BERTopic
import pandas as pd


def add_themes_and_edges(
    graph: nx.Graph,
    topic_model: BERTopic,
    topics: List[int],
    probabilities: np.ndarray,
    chunk_ids: List[str],
    topic_labeler,
    logger: logging.Logger,
    log_step_fn
):
    """
    Add theme nodes and HAS_TOPIC edges using batched LLM labeling.

    Args:
        graph: NetworkX graph to add nodes/edges to
        topic_model: Trained BERTopic model
        topics: List of topic assignments per chunk
        probabilities: Topic probability matrix (n_chunks, n_topics)
        chunk_ids: List of chunk IDs
        topic_labeler: TopicLabeler instance for LLM labeling
        logger: Logger instance
        log_step_fn: Function to log step completion
    """
    # Get unique topics (excluding -1 which is outlier)
    unique_topics = set(topics)
    if -1 in unique_topics:
        unique_topics.remove(-1)

    # Step 1: Collect all topics to label
    start_time_themes = time.time()
    topics_to_label = []

    for topic_id in sorted(unique_topics):
        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            continue
        topics_to_label.append((topic_id, topic_words))

    # Step 2: Label all topics in batches (with auto-tuning)
    # topic_labeler.label_topics_batch processes keywords once and returns both labels and processed keywords
    logger.info(f"Labeling {len(topics_to_label)} topics using batched LLM requests...")
    labels, processed_keywords_map = topic_labeler.label_topics_batch(topics_to_label)

    # Step 3: Create theme nodes with labels (reuse already-processed keywords)
    for topic_id in sorted(unique_topics):
        if topic_id not in labels:
            logger.warning(f"No label for topic {topic_id}, skipping node creation")
            continue

        # Reuse processed keywords from label_topics_batch (no re-processing needed)
        processed_keywords = processed_keywords_map.get(topic_id, [])
        theme_node = f"theme::{topic_id}"
        graph.add_node(
            theme_node,
            type="theme",
            topic_id=topic_id,
            label=labels[topic_id],
            keywords=[w for w, _ in processed_keywords]
        )

    log_step_fn("Creating theme nodes (batched)", start_time_themes, len(unique_topics))

    # Step 2: Add HAS_TOPIC edges for all topics above threshold OR top-K topics
    start_time_edges = time.time()
    edges_added = 0
    top_k_topics = 3  # Assign top-3 topics to each chunk for better multi-label coverage

    for idx, chunk_id in enumerate(tqdm(chunk_ids, desc="Adding HAS_TOPIC edges", disable=logger.level > logging.INFO)):
        # Skip outliers (topic -1)
        if topics[idx] == -1:
            continue

        # Multi-topic assignment with probabilities
        if probabilities is not None and probabilities.ndim > 1:
            # probabilities shape: (n_chunks, n_topics)
            chunk_probs = probabilities[idx]

            # Strategy: Use TOP-K topics (with their probabilities as weights)
            # This ensures every chunk has multiple topics for CO_PRESENCE edges

            # Get top-K topic indices sorted by probability
            top_k_indices = np.argsort(chunk_probs)[-top_k_topics:][::-1]

            for topic_id in top_k_indices:
                prob = chunk_probs[topic_id]

                # Skip if topic is -1 (outlier) or probability is too low (< 1%)
                if topic_id == -1 or prob < 0.01:
                    continue

                theme_node = f"theme::{topic_id}"

                # Add edge with probability weight
                graph.add_edge(
                    chunk_id,
                    theme_node,
                    edge_type="HAS_TOPIC",
                    weight=float(prob)
                )
                edges_added += 1
        else:
            # Fallback: Hard assignment if no probabilities available
            theme_node = f"theme::{topics[idx]}"
            graph.add_edge(
                chunk_id,
                theme_node,
                edge_type="HAS_TOPIC",
                weight=1.0
            )
            edges_added += 1

    log_step_fn("Adding HAS_TOPIC edges (top-3 per chunk)", start_time_edges, edges_added)


def build_hierarchical_topics(
    graph: nx.Graph,
    topic_model: BERTopic,
    chunk_texts: List[str],
    topic_labeler,
    nr_supertopics: Optional[int],
    logger: logging.Logger,
    log_step_fn
) -> Dict[int, int]:
    """
    Build hierarchical topic structure with supertopics.

    Uses BERTopic's hierarchical_topics() to create supertopics by merging
    semantically similar topics based on c-TF-IDF cosine similarity.

    The hierarchical merging process:
    1. Computes c-TF-IDF representations for each topic
    2. Uses agglomerative clustering with average linkage
    3. Merges topics with highest c-TF-IDF similarity first
    4. Creates supertopics by cutting the dendrogram at desired level

    Edge weight formula:
        weight = 1 - merge_distance = cosine_sim(c-TF-IDF(t_i), c-TF-IDF(t_j))

    Args:
        graph: NetworkX graph to add nodes/edges to
        topic_model: Trained BERTopic model
        chunk_texts: Original documents used for training (needed for hierarchical_topics)
        topic_labeler: TopicLabeler instance for LLM labeling of supertopics
        nr_supertopics: Target number of supertopics (None = auto ~40% of topics)
        logger: Logger instance
        log_step_fn: Function to log step completion

    Returns:
        Dictionary mapping topic_id -> supertopic_id

    Example:
        >>> mapping = build_hierarchical_topics(graph, model, texts, labeler, 10, logger, log_fn)
        >>> mapping[5]  # Topic 5 belongs to supertopic...
        2
    """
    start_time = time.time()

    # Get all valid topics (excluding outliers)
    all_topics = [t for t in topic_model.get_topics().keys() if t != -1]
    n_topics = len(all_topics)

    if n_topics < 2:
        logger.warning("Not enough topics for hierarchical structure")
        return {t: t for t in all_topics}

    # Compute target number of supertopics
    if nr_supertopics is None:
        nr_supertopics = max(3, int(n_topics * 0.4))

    logger.info(f"Building hierarchical structure: {n_topics} topics -> {nr_supertopics} supertopics")

    # Get hierarchical topic structure from BERTopic
    # This uses c-TF-IDF similarity for merging
    hierarchical_df = topic_model.hierarchical_topics(chunk_texts)

    # Calculate number of merges needed
    n_merges_needed = max(0, n_topics - nr_supertopics)

    # Initialize: each topic is its own supertopic
    topic_to_supertopic = {tid: tid for tid in all_topics}

    # Track merge distances for edge weights
    merge_distances = {}

    # Process merges from the hierarchical structure
    for idx in range(min(n_merges_needed, len(hierarchical_df))):
        row = hierarchical_df.iloc[idx]

        # Get topics in this merge
        topics_in_parent = row["Topics"]
        if isinstance(topics_in_parent, (int, np.integer)):
            topics_in_parent = [int(topics_in_parent)]
        else:
            topics_in_parent = [int(t) for t in topics_in_parent]

        # Get merge distance (1 - similarity)
        merge_distance = row.get("Distance", 0.5)

        # Use minimum topic ID as supertopic ID
        super_id = min(topics_in_parent)

        # Update mappings
        for tid in topics_in_parent:
            if tid in topic_to_supertopic:
                old_super = topic_to_supertopic[tid]
                topic_to_supertopic[tid] = super_id
                # Store merge distance for weight calculation
                if tid != super_id:
                    merge_distances[(super_id, tid)] = merge_distance

    # Get unique supertopics
    unique_supertopics = set(topic_to_supertopic.values())
    supertopic_groups = defaultdict(list)
    for topic_id, supertopic_id in topic_to_supertopic.items():
        supertopic_groups[supertopic_id].append(topic_id)

    logger.info(f"Created {len(unique_supertopics)} supertopics from {n_topics} topics")

    # Create supertopic nodes with aggregated labels
    supertopics_to_label = []
    for super_id in sorted(unique_supertopics):
        child_topics = supertopic_groups[super_id]

        # Aggregate keywords from child topics
        all_keywords = defaultdict(float)
        for topic_id in child_topics:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                for word, score in topic_words[:10]:
                    all_keywords[word] += score

        # Sort by aggregated score
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        supertopics_to_label.append((super_id, sorted_keywords[:15]))

    # Label supertopics using LLM
    logger.info(f"Labeling {len(supertopics_to_label)} supertopics...")
    super_labels, super_keywords_map = topic_labeler.label_topics_batch(supertopics_to_label)

    # Add supertopic nodes
    for super_id in sorted(unique_supertopics):
        child_topics = supertopic_groups[super_id]
        processed_keywords = super_keywords_map.get(super_id, [])

        supertopic_node = f"supertopic::{super_id}"
        graph.add_node(
            supertopic_node,
            type="supertopic",
            supertopic_id=super_id,
            label=super_labels.get(super_id, f"Supertopic {super_id}"),
            keywords=[w for w, _ in processed_keywords],
            child_topics=child_topics,
            n_topics=len(child_topics)
        )

    # Add CONTAINS_TOPIC edges from supertopics to their child topics
    edges_added = 0
    for super_id, child_topics in supertopic_groups.items():
        supertopic_node = f"supertopic::{super_id}"

        for topic_id in child_topics:
            theme_node = f"theme::{topic_id}"

            if not graph.has_node(theme_node):
                continue

            # Calculate edge weight: similarity = 1 - distance
            merge_dist = merge_distances.get((super_id, topic_id), 0.0)
            similarity_weight = 1.0 - merge_dist

            # If topic IS the supertopic (it's the representative), weight = 1.0
            if topic_id == super_id:
                similarity_weight = 1.0

            graph.add_edge(
                supertopic_node,
                theme_node,
                edge_type="CONTAINS_TOPIC",
                weight=float(similarity_weight),
                merge_distance=float(merge_dist)
            )
            edges_added += 1

            # Also update theme node with supertopic reference
            graph.nodes[theme_node]['supertopic_id'] = super_id

    log_step_fn("Building hierarchical topics (supertopics)", start_time, edges_added)

    return topic_to_supertopic


def add_theme_semantic_similarity(
    graph: nx.Graph,
    topic_model: BERTopic,
    embedding_model,
    semantic_sim_threshold: float,
    logger: logging.Logger,
    log_step_fn
):
    """
    Add SEMANTIC_SIM edges between themes based on topic word embeddings.

    Args:
        graph: NetworkX graph to add edges to
        topic_model: Trained BERTopic model (to get topic keywords)
        embedding_model: Sentence transformer model for embeddings
        semantic_sim_threshold: Minimum similarity to create edge
        logger: Logger instance
        log_step_fn: Function to log step completion
    """
    start_time = time.time()

    # Get all theme nodes
    theme_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get('type') == 'theme'
    ]

    if len(theme_nodes) < 2:
        logger.info("Skipping theme semantic similarity (< 2 themes)")
        return

    # For each theme, create a representation from top keywords
    theme_representations = []
    for theme_node in theme_nodes:
        topic_id = graph.nodes[theme_node]['topic_id']
        topic_words = topic_model.get_topic(topic_id)

        if not topic_words:
            # Fallback: use label
            theme_representations.append(graph.nodes[theme_node].get('label', ''))
        else:
            # Concatenate top 5 keywords
            keywords = [word for word, _ in topic_words[:5]]
            theme_representations.append(' '.join(keywords))

    # Embed theme representations
    logger.info(f"Embedding {len(theme_representations)} themes...")
    theme_embeddings = embedding_model.encode(
        theme_representations,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    # Compute similarity matrix
    similarity_matrix = np.matmul(theme_embeddings, theme_embeddings.T)
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity

    # Add edges above threshold (use same threshold as entities for consistency)
    edge_count = 0
    for i in tqdm(range(len(theme_nodes)), desc="Adding theme SEMANTIC_SIM edges", disable=logger.level > logging.INFO):
        for j in range(i + 1, len(theme_nodes)):
            sim = similarity_matrix[i, j]
            if sim >= semantic_sim_threshold:
                # Clamp to [0, 1] to fix floating-point rounding errors
                if sim > 1.0 and sim <= 1.001:
                    sim = 1.0
                elif sim > 1.001:
                    logger.warning(f"Theme similarity value {sim} > 1.001 detected, clamping to 1.0")
                    sim = 1.0

                graph.add_edge(
                    theme_nodes[i],
                    theme_nodes[j],
                    edge_type="SEMANTIC_SIM",
                    weight=float(sim),
                    semantic_type="theme"  # Mark as theme-theme similarity
                )
                edge_count += 1

    log_step_fn("Adding theme SEMANTIC_SIM edges", start_time, edge_count)


def add_theme_co_presence(
    graph: nx.Graph,
    chunk_ids: List[str],
    co_occur_threshold: int,
    logger: logging.Logger,
    log_step_fn
):
    """
    Add CO_PRESENCE edges between themes based on co-occurrence in chunks.

    Similar to entity CO_OCCURS but for themes that appear together in chunks.
    Uses weighting: co_presence_count * (1 / sqrt(freq1 * freq2))

    Args:
        graph: NetworkX graph to add edges to
        chunk_ids: List of chunk IDs
        co_occur_threshold: Minimum co-occurrence count to create edge
        logger: Logger instance
        log_step_fn: Function to log step completion
    """
    theme_co_presence_counter = Counter()
    theme_chunk_frequency = Counter()

    # Get all chunks and their connected themes
    chunk_to_themes = defaultdict(set)

    start_time_collect = time.time()
    for chunk_id in tqdm(chunk_ids, desc="Collecting chunk-theme pairs", disable=logger.level > logging.INFO):
        # Find all themes connected to this chunk via HAS_TOPIC edges
        if not graph.has_node(chunk_id):
            continue

        for neighbor in graph.neighbors(chunk_id):
            edge_data = graph[chunk_id][neighbor]
            if (edge_data.get('edge_type') == 'HAS_TOPIC' and
                graph.nodes[neighbor].get('type') == 'theme'):
                chunk_to_themes[chunk_id].add(neighbor)
                theme_chunk_frequency[neighbor] += 1

    log_step_fn("Collecting chunk-theme pairs", start_time_collect, len(chunk_ids))

    # Count co-presences
    start_time_count = time.time()
    for chunk_id, themes in tqdm(chunk_to_themes.items(), desc="Counting theme co-presences", disable=logger.level > logging.INFO):
        themes_list = list(themes)
        for i in range(len(themes_list)):
            for j in range(i + 1, len(themes_list)):
                pair = tuple(sorted([themes_list[i], themes_list[j]]))
                theme_co_presence_counter[pair] += 1

    log_step_fn("Counting theme co-presences", start_time_count, len(chunk_to_themes))

    # Calculate weights and add edges
    if not theme_co_presence_counter:
        logger.warning("No theme co-presences found")
        return

    edges_added = 0
    start_time_add = time.time()

    for (theme1, theme2), co_presence_count in tqdm(theme_co_presence_counter.items(), desc="Adding CO_PRESENCE edges", disable=logger.level > logging.INFO):
        if co_presence_count >= co_occur_threshold:  # Use same threshold
            # Theme frequencies (number of chunks they appear in)
            freq1 = theme_chunk_frequency[theme1]
            freq2 = theme_chunk_frequency[theme2]

            # Inverse frequency weight
            inv_freq_weight = 1.0 / np.sqrt(freq1 * freq2)

            # Final weight
            weight = co_presence_count * inv_freq_weight

            graph.add_edge(
                theme1,
                theme2,
                edge_type="CO_PRESENCE",
                weight=float(weight),
                co_presence_count=int(co_presence_count),
                theme1_freq=int(freq1),
                theme2_freq=int(freq2),
                inv_freq_weight=float(inv_freq_weight)
            )
            edges_added += 1

    log_step_fn("Adding CO_PRESENCE edges", start_time_add, edges_added)
