"""
Strategy 7: Claim-Based KeyBERT Phrase Similarity.

Combines claim-based retrieval with keyphrase analysis.
Extracts keyphrases from the query using KeyBERT and computes
semantic similarity with chunks found through claim traversal.

Algorithm:
1. Extract keyphrases from query using KeyBERT
2. Find claims similar to query (via S5)
3. Get chunks connected to claims via graph traversal
4. Compute semantic similarity between query keyphrases and chunk text
5. Weight by relevance propagation from claim to chunk

Final Score:
    S7(chunk) = claim_sim × traversal_relevance × keyphrase_similarity

Where:
- claim_sim = cos(embed(query), embed(claim))
- traversal_relevance = propagated relevance from claim to chunk
- keyphrase_similarity = max cosine similarity between query keyphrases and chunk
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import networkx as nx
from keybert import KeyBERT


def retrieve_claim_keyphrase_similarity(
    query: str,
    graph: nx.Graph,
    embedding_model,
    keybert_model: KeyBERT,
    claim_embeddings: Optional[np.ndarray],
    claim_nodes: List[str],
    claim_texts: List[str],
    claim_retrieval_results: Dict,
    claim_top_k: int,
    keyphrase_top_n: int,
    logger: logging.Logger
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 7: Claim-Based KeyBERT Phrase Similarity.

    Extracts keyphrases from query, finds similar claims, traverses to
    chunks, and scores based on keyphrase-chunk semantic similarity
    weighted by claim traversal relevance.

    Score Formula:
        score = claim_sim × traversal_rel × keyphrase_sim

    Where:
    - claim_sim: Cosine similarity between query and matched claim
    - traversal_rel: Relevance score from claim's pre-computed retrieval
    - keyphrase_sim: Max cosine similarity between query keyphrases and chunk

    Args:
        query: Query text
        graph: Knowledge graph
        embedding_model: SentenceTransformer for embeddings
        keybert_model: Initialized KeyBERT model
        claim_embeddings: Pre-computed claim embeddings
        claim_nodes: List of claim node IDs
        claim_texts: List of claim texts
        claim_retrieval_results: Pre-computed retrieval results per claim
        claim_top_k: Number of top claims to consider
        keyphrase_top_n: Number of keyphrases to extract from query
        logger: Logger instance

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to score (normalized 0-1)
        - details: Details including extracted keyphrases and matched claims

    Example:
        >>> scores, details = retrieve_claim_keyphrase_similarity(
        ...     "COVID vaccines cause autism",
        ...     graph, embed_model, keybert_model,
        ...     claim_embeds, claim_nodes, claim_texts, claim_results, 5, 5, logger
        ... )
    """
    logger.info("\n--- Strategy 7: Claim-Based KeyBERT Phrase Similarity ---")

    details = {
        'query_keyphrases': [],
        'matched_claims': []
    }

    # Check if claims exist
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

    # Step 3: Find similar claims
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    claim_similarities = np.matmul(claim_embeddings, query_embedding)
    top_claim_indices = np.argsort(claim_similarities)[-claim_top_k:][::-1]

    # Step 4: Process each matched claim
    scores = defaultdict(float)
    chunk_texts_cache = {}  # Cache chunk texts to avoid repeated lookups

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
        details['matched_claims'].append({
            'claim_node': claim_node,
            'claim_text': claim_text,
            'query_similarity': claim_sim,
            'rating': node_data.get('rating', ''),
            'chunks_found': len(retrieval_results)
        })

        # Collect all chunk texts for batch embedding
        chunk_ids_to_process = []
        chunk_relevances = {}

        for result in retrieval_results:
            chunk_id = result['chunk_id']
            traversal_relevance = result['score_total']

            if chunk_id not in chunk_texts_cache:
                if graph.has_node(chunk_id):
                    chunk_text = graph.nodes[chunk_id].get('text', '')
                    if chunk_text:
                        chunk_texts_cache[chunk_id] = chunk_text
                        chunk_ids_to_process.append(chunk_id)

            chunk_relevances[chunk_id] = traversal_relevance

        # Step 5: Embed chunk texts (batch)
        if chunk_ids_to_process:
            chunk_texts_list = [chunk_texts_cache[cid] for cid in chunk_ids_to_process]
            chunk_embeddings = embedding_model.encode(
                chunk_texts_list,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Create embedding lookup
            chunk_embed_map = {
                cid: emb for cid, emb in zip(chunk_ids_to_process, chunk_embeddings)
            }
        else:
            chunk_embed_map = {}

        # Step 6: Score each chunk
        for result in retrieval_results:
            chunk_id = result['chunk_id']
            traversal_relevance = chunk_relevances.get(chunk_id, 0)

            if chunk_id not in chunk_embed_map:
                continue

            chunk_embedding = chunk_embed_map[chunk_id]

            # Compute similarity between each keyphrase and chunk
            # Take maximum similarity across all keyphrases
            keyphrase_chunk_sims = np.matmul(keyphrase_embeddings, chunk_embedding)
            max_keyphrase_sim = float(np.max(keyphrase_chunk_sims))

            # Weight by keyphrase importance (optional enhancement)
            # Use the score of the best-matching keyphrase
            best_keyphrase_idx = np.argmax(keyphrase_chunk_sims)
            best_keyphrase = query_keyphrases[best_keyphrase_idx]
            keyphrase_importance = keyphrase_scores.get(best_keyphrase, 0.5)

            # Compute final score
            # score = claim_sim × traversal_rel × keyphrase_sim × keyphrase_importance
            final_score = (
                claim_sim *
                traversal_relevance *
                max_keyphrase_sim *
                keyphrase_importance
            )

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

    logger.info(f"Found {len(scores)} chunks via claim-keyphrase similarity")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")
        logger.info(f"Mean score: {np.mean(list(scores.values())):.3f}")

    return dict(scores), details
