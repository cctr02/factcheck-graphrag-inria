"""
Strategy 7: Claim-Based KeyBERT Phrase Similarity.

Combines claim-based retrieval with keyphrase analysis.
Extracts keyphrases from the query using KeyBERT and computes
semantic similarity with chunks found through claim traversal.

Algorithm:
1. Extract keyphrases from query using KeyBERT
2. Find claims similar to query (cosine similarity)
3. Traverse REVIEWS → CONTAINS to get chunks linked to matched claims
4. Compute semantic similarity between query keyphrases and each chunk
5. Score = claim_sim × keyphrase_sim × keyphrase_importance

Final Score:
    S7(chunk) = claim_sim × keyphrase_sim × keyphrase_importance

Where:
- claim_sim = max(0, cos(embed(query), embed(claim)))
- keyphrase_sim = max cosine similarity between query keyphrases and chunk
- keyphrase_importance = KeyBERT relevance score for the best-matching keyphrase

All components ∈ [0, 1], so the product is ∈ [0, 1]. No min-max normalization.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from keybert import KeyBERT

from fact_check_kg.retrieval.retrieval_claims import expand_claims_via_embeddings


def retrieve_claim_keyphrase_similarity(
    query: str,
    graph: nx.Graph,
    embedding_model,
    keybert_model: KeyBERT,
    claim_embeddings: Optional[np.ndarray],
    claim_nodes: List[str],
    claim_texts: List[str],
    claim_top_k: int,
    keyphrase_top_n: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 7: Claim-Based KeyBERT Phrase Similarity via REVIEWS graph edges.

    Extracts keyphrases from query, finds similar claims, traverses REVIEWS → CONTAINS
    to reach chunks, and scores based on keyphrase-chunk semantic similarity
    weighted by claim similarity and keyphrase importance.

    Score Formula:
        score = claim_sim × keyphrase_sim × keyphrase_importance

    Scores are bounded to [0, 1] — no min-max normalization applied.

    Args:
        query: Query text
        graph: Knowledge graph
        embedding_model: SentenceTransformer for embeddings
        keybert_model: Initialized KeyBERT model
        claim_embeddings: Pre-computed claim embeddings
        claim_nodes: List of claim node IDs
        claim_texts: List of claim texts
        claim_top_k: Number of top claims to consider
        keyphrase_top_n: Number of keyphrases to extract from query
        logger: Logger instance

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to score ∈ [0, 1]
        - details: Details including extracted keyphrases and matched claims
    """
    logger.info("\n--- Strategy 7: Claim-Based KeyBERT Phrase Similarity (+ CLAIM_SIM expansion) ---")

    details = {
        'query_keyphrases': [],
        'matched_claims': []
    }

    if claim_embeddings is None or len(claim_nodes) == 0:
        logger.warning("No claims in graph, skipping Strategy 7")
        return {}, details

    # Step 1: Extract keyphrases from query using KeyBERT
    try:
        keyphrases = keybert_model.extract_keywords(
            query,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=keyphrase_top_n,
            use_mmr=True,
            diversity=0.5
        )
        query_keyphrases = [phrase for phrase, score in keyphrases]
        keyphrase_scores = {phrase: score for phrase, score in keyphrases}

        logger.info(f"Extracted {len(query_keyphrases)} keyphrases from query:")
        for phrase, score in keyphrases:
            logger.info(f"  '{phrase}': {score:.3f}")

        details['query_keyphrases'] = keyphrases

    except Exception as e:
        logger.warning(f"KeyBERT extraction failed: {e}")
        return {}, details

    if not query_keyphrases:
        logger.warning("No keyphrases extracted from query")
        return {}, details

    # Step 2: Embed query keyphrases
    keyphrase_embeddings = embedding_model.encode(
        query_keyphrases,
        normalize_embeddings=True
    )

    # Step 3: Find similar claims + expand via CLAIM_SIM edges
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    claim_similarities = np.matmul(claim_embeddings, query_embedding)
    top_claim_indices = np.argsort(claim_similarities)[-claim_top_k:][::-1]

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

    # Step 4: Process each matched claim (direct + expanded)
    scores = {}
    chunk_embed_cache = {}  # Cache chunk embeddings to avoid recomputing

    for claim_node, claim_sim in all_claims_to_process:
        node_data = graph.nodes.get(claim_node, {})
        claim_text = node_data.get('text', '')
        chunks_to_embed = []
        chunk_ids_batch = []

        # Step 5: Traverse REVIEWS → CONTAINS to collect chunk IDs
        for neighbor in graph.neighbors(claim_node):
            edge_data = graph[claim_node][neighbor]
            if edge_data.get('edge_type') != 'REVIEWS':
                continue
            doc_node = neighbor
            for doc_neighbor in graph.neighbors(doc_node):
                doc_edge_data = graph[doc_node][doc_neighbor]
                if (doc_edge_data.get('edge_type') == 'CONTAINS' and
                        graph.nodes[doc_neighbor].get('type') == 'chunk'):
                    chunk_id = doc_neighbor
                    if chunk_id not in chunk_embed_cache:
                        chunk_text = graph.nodes[chunk_id].get('text', '')
                        if chunk_text:
                            chunks_to_embed.append(chunk_text)
                            chunk_ids_batch.append(chunk_id)

        # Step 6: Batch-embed chunks not yet cached
        if chunks_to_embed:
            batch_embeddings = embedding_model.encode(
                chunks_to_embed,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            for cid, emb in zip(chunk_ids_batch, batch_embeddings):
                chunk_embed_cache[cid] = emb

        # Step 7: Score each chunk within the claim-linked preselection
        chunks_scored = 0
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
                if chunk_id not in chunk_embed_cache:
                    continue

                chunk_embedding = chunk_embed_cache[chunk_id]

                # Max cosine similarity over all query keyphrases
                keyphrase_chunk_sims = np.matmul(keyphrase_embeddings, chunk_embedding)
                max_keyphrase_sim = max(0.0, float(np.max(keyphrase_chunk_sims)))

                best_keyphrase_idx = int(np.argmax(keyphrase_chunk_sims))
                best_keyphrase = query_keyphrases[best_keyphrase_idx]
                keyphrase_importance = keyphrase_scores.get(best_keyphrase, 0.5)

                # Score = claim_sim × keyphrase_sim × keyphrase_importance ∈ [0, 1]
                final_score = claim_sim * max_keyphrase_sim * keyphrase_importance

                if chunk_id not in scores or final_score > scores[chunk_id]:
                    scores[chunk_id] = final_score
                chunks_scored += 1

        details['matched_claims'].append({
            'claim_node': claim_node,
            'claim_text': claim_text,
            'query_similarity': claim_sim,
            'rating': node_data.get('rating', ''),
            'chunks_found': chunks_scored,
            'source': 'direct' if any(cn == claim_node for cn, _ in direct_claims) else 'graph'
        })

    logger.info(f"Total matched claims: {len(all_claims_to_process)} "
                f"({len(direct_claims)} direct + {len(expanded_claims)} via CLAIM_SIM)")
    logger.info(f"Found {len(scores)} chunks via claim-keyphrase similarity")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")
        logger.info(f"Mean score: {np.mean(list(scores.values())):.3f}")

    return scores, details
