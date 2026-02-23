"""
Claim node creation and edge module.

This module handles creation of:
- Claim nodes (fact-checked claims with metadata)
- REVIEWS edges (Claim -> Document linking claims to their fact-check documents)
- Pre-computed claim retrieval results for Strategy 5

Claim Retrieval Pre-computation:
    For each claim, computes scores using 4 strategies:
    - S1: Semantic similarity (embedding cosine)
    - S2: Topic distribution similarity (1 - Jensen-Shannon divergence)
    - S3: Thematic graph traversal (BFS through theme edges)
    - S4: Entity graph traversal (BFS through entity edges)

    Combined score:
        score = 0.25×S1 + 0.25×S2 + 0.25×S3 + 0.25×S4

    Results are stored for efficient retrieval at query time.
"""

import time
import logging
from typing import List, Dict, Set
from collections import deque, defaultdict
from tqdm import tqdm
import numpy as np
import networkx as nx
from scipy.spatial.distance import jensenshannon
from bertopic import BERTopic
import torch


def add_claims_and_edges(
    graph: nx.Graph,
    claims: List[Dict],
    claim_to_docs: Dict[str, List[str]],
    logger: logging.Logger,
    log_step_fn
):
    """
    Add claim nodes and REVIEWS edges linking claims to documents.

    Args:
        graph: NetworkX graph to add nodes/edges to
        claims: List of claim dictionaries with keys:
            - claim_id: str
            - text: str (the claim text)
            - rating: str (fact-check verdict)
            - claimant: str (who made the claim)
            - claimant_norm: str (normalized claimant name)
            - review_date: str
            - language: str
        claim_to_docs: Mapping of claim_id to list of doc_ids
        logger: Logger instance
        log_step_fn: Function to log step completion (step_name, start_time, count)

    Example:
        >>> claims = [{"claim_id": "c1", "text": "The earth is flat", "rating": "False"}]
        >>> claim_to_docs = {"c1": ["doc1", "doc2"]}
        >>> add_claims_and_edges(graph, claims, claim_to_docs, logger, log_fn)
    """
    start_time = time.time()
    edges_added = 0

    for claim in tqdm(claims, desc="Adding claims and REVIEWS edges", disable=logger.level > logging.INFO):
        claim_id = claim['claim_id']
        claim_node = f"claim::{claim_id}"

        # Add claim node with all attributes
        graph.add_node(
            claim_node,
            type="claim",
            claim_id=claim_id,
            text=claim.get('text', ''),
            rating=claim.get('rating', ''),
            claimant=claim.get('claimant', ''),
            claimant_norm=claim.get('claimant_norm', ''),
            review_date=claim.get('review_date', ''),
            language=claim.get('language', '')
        )

        # Add REVIEWS edges to linked documents
        doc_ids = claim_to_docs.get(claim_id, [])
        for doc_id in doc_ids:
            doc_node = f"doc::{doc_id}"
            if graph.has_node(doc_node):
                graph.add_edge(
                    claim_node,
                    doc_node,
                    edge_type="REVIEWS"
                )
                edges_added += 1

    log_step_fn("Adding claims and REVIEWS edges", start_time, edges_added)


def add_claim_semantic_similarity(
    graph: nx.Graph,
    embedding_model,
    sim_threshold: float,
    top_k_per_claim: int = 10,
    logger: logging.Logger = None,
    log_step_fn = None
):
    """
    Add CLAIM_SIM edges between semantically similar claims.

    For each claim, finds its top-K most similar claims (by cosine similarity
    of their text embeddings) and adds a CLAIM_SIM edge when similarity
    exceeds the threshold.

    Edge weight = cosine similarity ∈ [sim_threshold, 1].

    Only the upper triangle of the similarity matrix is processed to avoid
    duplicate edges (undirected graph).

    Args:
        graph: NetworkX graph containing claim nodes
        embedding_model: SentenceTransformer model for text embeddings
        sim_threshold: Minimum cosine similarity to create a CLAIM_SIM edge
        top_k_per_claim: Maximum neighbours to consider per claim
        logger: Logger instance
        log_step_fn: Function to log step completion (step_name, start_time, count)
    """
    start_time = time.time()
    log = logger or logging.getLogger(__name__)

    claim_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'claim']
    n_claims = len(claim_nodes)

    if n_claims < 2:
        log.info(f"Skipping CLAIM_SIM edges: only {n_claims} claim(s) in graph")
        return

    log.info(f"Computing CLAIM_SIM edges for {n_claims} claims (threshold={sim_threshold}, top_k={top_k_per_claim})...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128 if device == 'cuda' else 32

    claim_texts = [graph.nodes[n].get('text', '') for n in claim_nodes]

    embeddings = embedding_model.encode(
        claim_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device
    )

    edge_count = 0
    chunk_size = 512  # Process 512 rows at a time to bound memory usage

    for i_start in tqdm(range(0, n_claims, chunk_size), desc="Adding CLAIM_SIM edges"):
        i_end = min(i_start + chunk_size, n_claims)
        batch = embeddings[i_start:i_end]  # (chunk_size, dim)

        # Similarity of this batch against ALL claims: (chunk_size, n_claims)
        sims = np.matmul(batch, embeddings.T)

        for k in range(i_end - i_start):
            global_i = i_start + k
            row = sims[k].copy()

            # Zero out lower triangle (including diagonal) to avoid duplicates
            row[:global_i + 1] = 0.0

            # Get top-K candidates among j > global_i
            remaining = n_claims - global_i - 1
            if remaining <= 0:
                continue
            top_k = min(top_k_per_claim, remaining)
            top_j = np.argpartition(row, -top_k)[-top_k:]

            for j in top_j:
                sim = float(row[j])
                if sim >= sim_threshold:
                    graph.add_edge(
                        claim_nodes[global_i],
                        claim_nodes[j],
                        edge_type="CLAIM_SIM",
                        weight=sim
                    )
                    edge_count += 1

    log.info(f"Created {edge_count} CLAIM_SIM edges between {n_claims} claims")
    if log_step_fn:
        log_step_fn("Adding CLAIM_SIM edges (Claim ↔ Claim)", start_time, edge_count)


def compute_thematic_scores_for_claim(
    graph: nx.Graph,
    claim_topic_probs: np.ndarray,
    relevance_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Compute thematic scores for a claim using BFS graph traversal.

    Uses pre-computed topic probabilities to propagate relevance through
    the theme graph and score chunks based on theme relevance.

    Algorithm:
    1. Identify seed themes where claim has probability > 0.01
    2. BFS through CO_PRESENCE and SEMANTIC_SIM edges
    3. Score chunks: chunk_score = theme_relevance × HAS_TOPIC_weight

    Args:
        graph: Knowledge graph containing themes and chunks
        claim_topic_probs: Topic probability distribution for the claim
        relevance_threshold: Minimum relevance to continue propagation

    Returns:
        Dictionary mapping chunk_id to thematic score
    """
    # Detect themes above threshold (0.01)
    seed_themes = []
    for idx, prob in enumerate(claim_topic_probs):
        if prob > 0.01:
            theme_node = f"theme::{idx}"
            if graph.has_node(theme_node):
                seed_themes.append((theme_node, float(prob)))

    if not seed_themes:
        return {}

    # BFS propagation through theme graph
    theme_relevance = {}
    queue = deque()
    visited = set()

    for theme_node, prob in seed_themes:
        theme_relevance[theme_node] = prob
        queue.append((theme_node, prob))
        visited.add(theme_node)

    while queue:
        current_theme, current_rel = queue.popleft()
        for neighbor in graph.neighbors(current_theme):
            if graph.nodes[neighbor].get('type') != 'theme':
                continue
            edge_data = graph[current_theme][neighbor]
            edge_type = edge_data.get('edge_type')
            if edge_type not in ['CO_PRESENCE', 'SEMANTIC_SIM']:
                continue
            edge_weight = edge_data.get('weight', 0)
            propagated_rel = current_rel * edge_weight
            if propagated_rel < relevance_threshold:
                continue
            if neighbor not in theme_relevance:
                theme_relevance[neighbor] = propagated_rel
            else:
                theme_relevance[neighbor] = max(theme_relevance[neighbor], propagated_rel)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, propagated_rel))

    # Score chunks based on theme relevance
    scores = defaultdict(float)
    for theme_node, relevance in theme_relevance.items():
        for neighbor in graph.neighbors(theme_node):
            if graph.nodes[neighbor].get('type') == 'chunk':
                edge_data = graph[theme_node][neighbor]
                if edge_data.get('edge_type') == 'HAS_TOPIC':
                    weight = edge_data.get('weight', 1.0)
                    scores[neighbor] = max(scores[neighbor], relevance * weight)

    return dict(scores)


def compute_entity_scores_for_claim(
    graph: nx.Graph,
    claim_text: str,
    nlp,
    entity_types: Set[str],
    relevance_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Compute entity scores for a claim using BFS graph traversal.

    Extracts entities from the claim text and propagates relevance through
    the entity graph to score chunks.

    Algorithm:
    1. Extract named entities from claim using spaCy
    2. Start BFS from entities found in graph (relevance = 1.0)
    3. Propagate through CO_OCCURS and SEMANTIC_SIM edges
    4. Score chunks: chunk_score = entity_relevance × MENTIONS_weight

    Args:
        graph: Knowledge graph containing entities and chunks
        claim_text: Text of the claim to analyze
        nlp: spaCy language model for NER
        entity_types: Set of entity types to extract
        relevance_threshold: Minimum relevance to continue propagation

    Returns:
        Dictionary mapping chunk_id to entity score
    """
    # Extract entities from claim
    doc = nlp(claim_text)
    claim_entities = []
    for ent in doc.ents:
        if ent.label_ in entity_types:
            entity_text = ent.text.strip().lower()
            if len(entity_text) > 2:
                entity_node = f"entity::{ent.label_}::{entity_text}"
                if graph.has_node(entity_node):
                    claim_entities.append(entity_node)

    if not claim_entities:
        return {}

    # BFS propagation through entity graph
    entity_relevance = {e: 1.0 for e in claim_entities}
    queue = deque([(e, 1.0) for e in claim_entities])
    visited = set(claim_entities)

    while queue:
        current_entity, current_rel = queue.popleft()
        for neighbor in graph.neighbors(current_entity):
            if graph.nodes[neighbor].get('type') != 'entity':
                continue
            edge_data = graph[current_entity][neighbor]
            edge_type = edge_data.get('edge_type')
            if edge_type not in ['CO_OCCURS', 'SEMANTIC_SIM']:
                continue
            edge_weight = edge_data.get('weight', 0)
            propagated_rel = current_rel * edge_weight
            if propagated_rel < relevance_threshold:
                continue
            if neighbor not in entity_relevance:
                entity_relevance[neighbor] = propagated_rel
            else:
                entity_relevance[neighbor] = max(entity_relevance[neighbor], propagated_rel)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, propagated_rel))

    # Score chunks based on entity relevance
    scores = defaultdict(float)
    for entity_node, relevance in entity_relevance.items():
        for neighbor in graph.neighbors(entity_node):
            edge_data = graph[entity_node][neighbor]
            if (edge_data.get('edge_type') == 'MENTIONS' and
                graph.nodes[neighbor].get('type') == 'chunk'):
                tfidf_weight = edge_data.get('weight', 1.0)
                scores[neighbor] += relevance * tfidf_weight

    return dict(scores)


def precompute_claim_retrieval(
    graph: nx.Graph,
    claims: List[Dict],
    topic_model: BERTopic,
    probabilities: np.ndarray,
    chunk_ids: List[str],
    embedding_model,
    nlp,
    entity_types: Set[str],
    logger: logging.Logger
) -> Dict[str, Dict]:
    """
    Pre-compute retrieval results for each claim using 4 strategies.

    Uses batch processing for embeddings and topic transforms to optimize
    computation time. Results are stored for efficient claim-based retrieval.

    Strategies:
    - S1: Semantic similarity = cos(embed(claim), embed(chunk))
    - S2: Topic distribution = 1 - JSD(claim_dist, chunk_dist)
    - S3: Thematic traversal = BFS through theme graph
    - S4: Entity traversal = BFS through entity graph

    Combined: score = 0.25×S1 + 0.25×S2 + 0.25×S3 + 0.25×S4

    Args:
        graph: Knowledge graph
        claims: List of claim dictionaries with 'claim_id' and 'text'
        topic_model: Trained BERTopic model
        probabilities: Topic probability matrix (n_chunks, n_topics)
        chunk_ids: List of chunk IDs
        embedding_model: SentenceTransformer for embeddings
        nlp: spaCy model for entity extraction
        entity_types: Entity types to extract
        logger: Logger instance

    Returns:
        Dictionary mapping claim_node to retrieval results:
        {
            'claim::{id}': {
                'claim_id': str,
                'claim_text': str,
                'retrieval_results': [
                    {'chunk_id': str, 'score_total': float, ...}
                ]
            }
        }
    """
    logger.info(f"Pre-computing retrieval for {len(claims)} claims...")

    # Filter valid claims
    valid_claims = [(c['claim_id'], c.get('text', '')) for c in claims if c.get('text', '')]
    if not valid_claims:
        logger.warning("No valid claims to process")
        return {}

    claim_ids, claim_texts = zip(*valid_claims)
    logger.info(f"Processing {len(claim_texts)} valid claims")

    # Determine optimal batch size based on device
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128 if device == 'cuda' else 32

    # BATCH 1: Pre-compute chunk embeddings
    chunk_texts = [graph.nodes[cid].get('text', '') for cid in chunk_ids]
    logger.info(f"Computing chunk embeddings (batch) on {device.upper()} with batch_size={batch_size}...")
    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device
    )

    # BATCH 2: Pre-compute claim embeddings
    logger.info(f"Computing claim embeddings (batch) on {device.upper()} with batch_size={batch_size}...")
    claim_embeddings = embedding_model.encode(
        list(claim_texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device
    )

    # BATCH 3: Compute semantic similarities (claims × chunks)
    logger.info("Computing semantic similarity matrix...")
    all_S1_scores = np.matmul(claim_embeddings, chunk_embeddings.T)
    # Min-max normalize per claim
    S1_min = all_S1_scores.min(axis=1, keepdims=True)
    S1_max = all_S1_scores.max(axis=1, keepdims=True)
    S1_range = S1_max - S1_min
    S1_range[S1_range == 0] = 1
    all_S1_normalized = (all_S1_scores - S1_min) / S1_range

    # BATCH 4: Pre-compute claim topic distributions
    logger.info("Computing claim topic distributions (batch)...")
    _, all_claim_topic_probs = topic_model.transform(list(claim_texts))

    # BATCH 5: Compute JS divergence scores
    logger.info("Computing topic distribution similarities...")
    all_S2_scores = np.zeros((len(claim_texts), len(chunk_ids)))
    for claim_idx, claim_dist in enumerate(all_claim_topic_probs):
        for chunk_idx in range(len(chunk_ids)):
            chunk_dist = probabilities[chunk_idx]
            js_div = jensenshannon(claim_dist, chunk_dist)
            all_S2_scores[claim_idx, chunk_idx] = 1.0 - js_div
    # Min-max normalize per claim
    S2_min = all_S2_scores.min(axis=1, keepdims=True)
    S2_max = all_S2_scores.max(axis=1, keepdims=True)
    S2_range = S2_max - S2_min
    S2_range[S2_range == 0] = 1
    all_S2_normalized = (all_S2_scores - S2_min) / S2_range

    # Process S3 and S4 per claim
    logger.info("Computing thematic and entity scores per claim...")
    claim_retrieval_results = {}

    for idx, (claim_id, claim_text) in enumerate(tqdm(valid_claims, desc="Finalizing claim retrieval")):
        claim_node = f"claim::{claim_id}"

        # Get pre-computed S1 and S2 scores
        S1_scores = {cid: float(all_S1_normalized[idx, i]) for i, cid in enumerate(chunk_ids)}
        S2_scores = {cid: float(all_S2_normalized[idx, i]) for i, cid in enumerate(chunk_ids)}

        # Strategy 3: Thematic Graph Traversal
        S3_scores = compute_thematic_scores_for_claim(graph, all_claim_topic_probs[idx])
        if S3_scores:
            min_s, max_s = min(S3_scores.values()), max(S3_scores.values())
            if max_s > min_s:
                S3_scores = {k: (v - min_s) / (max_s - min_s) for k, v in S3_scores.items()}

        # Strategy 4: Entity Graph Traversal
        S4_scores = compute_entity_scores_for_claim(graph, claim_text, nlp, entity_types)
        if S4_scores:
            min_s, max_s = min(S4_scores.values()), max(S4_scores.values())
            if max_s > min_s:
                S4_scores = {k: (v - min_s) / (max_s - min_s) for k, v in S4_scores.items()}

        # Combine scores
        all_chunks = set(S1_scores.keys()) | set(S2_scores.keys()) | set(S3_scores.keys()) | set(S4_scores.keys())
        combined_results = []

        for cid in all_chunks:
            s1 = S1_scores.get(cid, 0)
            s2 = S2_scores.get(cid, 0)
            s3 = S3_scores.get(cid, 0)
            s4 = S4_scores.get(cid, 0)
            total_score = 0.25 * s1 + 0.25 * s2 + 0.25 * s3 + 0.25 * s4

            combined_results.append({
                "chunk_id": cid,
                "score_total": round(float(total_score), 4),
                "score_semantic": round(float(s1), 4),
                "score_topic_dist": round(float(s2), 4),
                "score_thematic": round(float(s3), 4),
                "score_entity": round(float(s4), 4)
            })

        # Sort and keep top 20
        combined_results.sort(key=lambda x: x['score_total'], reverse=True)
        top_results = combined_results[:20]

        claim_retrieval_results[claim_node] = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "retrieval_results": top_results
        }

    logger.info(f"Pre-computed retrieval for {len(claim_retrieval_results)} claims")
    return claim_retrieval_results
