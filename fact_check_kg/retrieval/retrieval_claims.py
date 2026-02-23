"""
Strategy 5: Claim-Based Document Retrieval with Semantic Claim Expansion.

Leverages pre-existing fact-checked claims to find relevant evidence chunks.
When a query is similar to a previously fact-checked claim, we can use
the documents that were used to fact-check that claim as relevant evidence.

Algorithm:
    1. Embed the query using SentenceTransformer
    2. Find top-k claims most similar to the query:
       similarity(q, c) = cos(embed(q), embed(c))
    3. For each matched claim, explore connected claims via SIMILAR_CLAIM edges
    4. Retrieve pre-computed chunk scores for all claims (direct + expanded)
    5. Combine scores: final_score = claim_similarity × chunk_score × expansion_decay

Score Calculation:
    For each chunk c reached via claim clm:
        S5(c) = max over matched claims of:
            cos(query, claim) × precomputed_score(claim, chunk)

    For expanded claims (reached via SIMILAR_CLAIM edges):
        expansion_score = query_similarity × edge_weight × precomputed_score
        (where edge_weight is the SIMILAR_CLAIM cosine similarity)

This strategy is particularly useful when:
- The query is semantically similar to a known false claim
- The fact-check documents contain authoritative refutations
- Cross-referencing multiple fact-checks strengthens evidence
- Exploring semantically related claims provides broader context
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx


def retrieve_claim_document(
    query: str,
    graph: nx.Graph,
    embedding_model,
    claim_embeddings: Optional[np.ndarray],
    claim_nodes: List[str],
    claim_texts: List[str],
    claim_retrieval_results: Dict,
    claim_top_k: int,
    logger: logging.Logger,
    explore_similar_claims: bool = True,
    max_expansion_depth: int = 1
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 5: Claim-Based Retrieval using pre-computed claim retrieval results.

    Finds claims semantically similar to the query, then returns:
    1. Chunk scores from pre-computed retrieval results for those claims
    2. If explore_similar_claims=True, expands to claims connected via SIMILAR_CLAIM edges
    3. Details about which claims were matched and their similarity to the query

    Args:
        query: Query text to match against claims
        graph: Knowledge graph (for claim node data and SIMILAR_CLAIM edges)
        embedding_model: SentenceTransformer model
        claim_embeddings: Pre-computed claim embeddings (n_claims, dim) or None
        claim_nodes: List of claim node IDs
        claim_texts: List of claim texts corresponding to nodes
        claim_retrieval_results: Dict mapping claim_node to pre-computed results
        claim_top_k: Number of top claims to consider
        logger: Logger instance
        explore_similar_claims: If True, explores claims connected via SIMILAR_CLAIM edges (default: True)
        max_expansion_depth: Maximum depth for exploring SIMILAR_CLAIM edges (default: 1)

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to score (normalized 0-1)
        - matched_claims_details: List of matched claim info with retrieval results

    Example:
        >>> scores, claims = retrieve_claim_document(
        ...     "Vaccines cause autism",
        ...     graph, model, embeddings, nodes, texts, results, 5, logger,
        ...     explore_similar_claims=True
        ... )
        >>> claims[0]['rating']
        'False'
    """
    logger.info("\n--- Strategy 5: Claim-Based Retrieval (Pre-computed) ---")

    matched_claims_details = []

    # Check if claims exist
    if claim_embeddings is None or len(claim_nodes) == 0:
        logger.warning("No claims in graph, skipping Strategy 5")
        return {}, matched_claims_details

    # Embed query
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Compute similarities with all claims
    similarities = np.matmul(claim_embeddings, query_embedding)

    # Get top-k claims
    top_indices = np.argsort(similarities)[-claim_top_k:][::-1]

    logger.info(f"Top {claim_top_k} claims by similarity to query:")
    scores = {}

    # Track all claims to explore (initial + expanded)
    claims_to_explore = []  # List of (claim_node, query_similarity, expansion_depth, source_claim)

    # Add initial top-k claims
    for idx in top_indices:
        claim_node = claim_nodes[idx]
        claim_similarity = float(similarities[idx])
        claims_to_explore.append((claim_node, claim_similarity, 0, None))  # depth 0, no source

    # Expand to similar claims if enabled
    if explore_similar_claims and max_expansion_depth > 0:
        logger.info(f"\n--- Claim Expansion: Exploring SIMILAR_CLAIM edges (max depth: {max_expansion_depth}) ---")
        explored_claims = set()

        for claim_node, query_sim, depth, source in list(claims_to_explore):
            if depth >= max_expansion_depth:
                continue

            explored_claims.add(claim_node)

            # Find neighbors via SIMILAR_CLAIM edges
            if graph.has_node(claim_node):
                for neighbor in graph.neighbors(claim_node):
                    edge_data = graph[claim_node][neighbor]
                    edge_type = edge_data.get('edge_type')

                    if edge_type == 'SIMILAR_CLAIM' and neighbor not in explored_claims:
                        edge_weight = edge_data.get('weight', 0)
                        # Propagated similarity = query_sim × edge_weight
                        propagated_sim = query_sim * edge_weight

                        claims_to_explore.append((neighbor, propagated_sim, depth + 1, claim_node))
                        explored_claims.add(neighbor)

                        logger.info(f"  Expanded: {neighbor} (via {claim_node}, similarity={propagated_sim:.3f}, edge_weight={edge_weight:.3f})")

        logger.info(f"Total claims after expansion: {len(claims_to_explore)} (initial={len(top_indices)}, expanded={len(claims_to_explore)-len(top_indices)})")

    # Process all claims (initial + expanded)
    for claim_node, claim_similarity, expansion_depth, source_claim in claims_to_explore:
        # Get claim text from graph
        claim_text = graph.nodes.get(claim_node, {}).get('text', '')

        claim_text_preview = claim_text[:100] + "..." if len(claim_text) > 100 else claim_text
        depth_str = f" [depth={expansion_depth}]" if expansion_depth > 0 else ""
        source_str = f" (via {source_claim})" if source_claim else ""
        logger.info(f"  {claim_node}{depth_str}: similarity={claim_similarity:.3f}{source_str}")
        logger.info(f"    Claim: {claim_text_preview}")

        # Get pre-computed retrieval results for this claim
        if claim_node in claim_retrieval_results:
            claim_data = claim_retrieval_results[claim_node]
            retrieval_results = claim_data.get('retrieval_results', [])

            logger.info(f"    Pre-computed results: {len(retrieval_results)} chunks")

            # Store matched claim details (pull rating from graph node)
            node_data = graph.nodes.get(claim_node, {})
            matched_claims_details.append({
                'claim_node': claim_node,
                'claim_text': claim_text,
                'query_similarity': claim_similarity,
                'rating': node_data.get('rating', ''),
                'expansion_depth': expansion_depth,
                'source_claim': source_claim if source_claim else 'direct',
                'retrieval_results': retrieval_results[:10]  # Top 10 chunks per claim
            })

            # Add chunk scores weighted by claim similarity
            for result in retrieval_results:
                chunk_id = result['chunk_id']
                chunk_score = result['score_total']
                # Final score = claim similarity × chunk retrieval score
                combined_score = claim_similarity * chunk_score

                if chunk_id not in scores:
                    scores[chunk_id] = combined_score
                else:
                    scores[chunk_id] = max(scores[chunk_id], combined_score)
        else:
            logger.warning(f"    No pre-computed results for {claim_node}")

    # Normalize scores (min-max)
    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
        if max_score > min_score:
            scores = {
                k: (v - min_score) / (max_score - min_score)
                for k, v in scores.items()
            }

    logger.info(f"Found {len(scores)} chunks via claim-based retrieval")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")

    return scores, matched_claims_details
