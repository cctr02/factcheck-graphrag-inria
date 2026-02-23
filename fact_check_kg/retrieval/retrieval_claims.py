"""
Strategy 5: Claim-Based Document Retrieval.

Leverages pre-existing fact-checked claims to find relevant evidence chunks.
When a query is similar to a previously fact-checked claim, we can use
the documents that were used to fact-check that claim as relevant evidence.

Algorithm:
    1. Embed the query using SentenceTransformer
    2. Find top-k claims most similar to the query:
       similarity(q, c) = cos(embed(q), embed(c))
    3. Expand the claim set by computing claim-to-claim cosine similarity
       from pre-computed embeddings (no graph edges needed):
         eff_sim = direct_query_sim × cos(direct_claim_embed, other_claim_embed)
       Keep top-k expansions above expansion_threshold.
    4. For each matched claim, traverse REVIEWS edges to find linked documents,
       then CONTAINS edges to find chunks in those documents
    5. Score each chunk: S5(c) = max over matched claims of claim_similarity

Score is naturally bounded to [0, 1] (cosine similarity, clipped at 0).
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx


def expand_claims_via_embeddings(
    top_indices: List[int],
    claim_nodes: List[str],
    similarities: np.ndarray,
    claim_embeddings: np.ndarray,
    max_expanded: int,
    expansion_threshold: float,
    graph: nx.Graph,
    logger: logging.Logger,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Expand an initial set of matched claims by computing claim-to-claim similarity
    from pre-computed embeddings (no CLAIM_SIM graph edges required).

    For each direct claim c_i matched to the query, finds other claims c_j
    whose effective similarity exceeds the threshold:

        eff_sim(c_j) = query_sim(c_i) × cos(embed(c_i), embed(c_j))

    The best eff_sim across all direct claims is kept for each candidate.
    Up to `max_expanded` claims are added, selected by descending eff_sim.

    Args:
        top_indices: Indices into claim_nodes of the directly-matched claims
        claim_nodes: Full list of claim node IDs
        similarities: 1-D query-to-claim cosine similarity array (len = n_claims)
        claim_embeddings: Normalised claim embeddings (n_claims, dim)
        max_expanded: Maximum number of expansion claims to add
        expansion_threshold: Minimum eff_sim to keep an expanded claim
        graph: Knowledge graph (used only for logging claim text)
        logger: Logger instance

    Returns:
        Tuple of:
        - direct_claims: List of (claim_node, query_sim) for direct matches
        - expanded_claims: List of (claim_node, eff_sim) for expanded matches
    """
    # Build direct claim set
    direct_claims = []
    direct_set = set()
    for idx in top_indices:
        sim = max(0.0, float(similarities[idx]))
        direct_claims.append((claim_nodes[idx], sim))
        direct_set.add(claim_nodes[idx])

    logger.info(f"  Direct claims ({len(direct_claims)}, by query-claim similarity):")
    for cn, sim in direct_claims:
        text = graph.nodes.get(cn, {}).get('text', '')
        preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"    [{sim:.3f}] {preview}")

    # Expand via embedding-based claim-to-claim similarity
    # For each direct claim, compute similarity with all other claims
    expansion_candidates: Dict[str, float] = {}
    for idx in top_indices:
        direct_sim = max(0.0, float(similarities[idx]))
        if direct_sim == 0.0:
            continue
        # Cosine similarity between this direct claim and all other claims
        # (embeddings are already normalized)
        claim_sims = np.matmul(claim_embeddings, claim_embeddings[idx])  # (n_claims,)

        for j, csim in enumerate(claim_sims):
            cn = claim_nodes[j]
            if cn in direct_set:
                continue
            eff_sim = direct_sim * max(0.0, float(csim))
            if eff_sim < expansion_threshold:
                continue
            if cn not in expansion_candidates or eff_sim > expansion_candidates[cn]:
                expansion_candidates[cn] = eff_sim

    sorted_expansions = sorted(expansion_candidates.items(), key=lambda x: x[1], reverse=True)[:max_expanded]
    expanded_claims = list(sorted_expansions)

    if expanded_claims:
        logger.info(f"  Embedding-expanded claims ({len(expanded_claims)}, via claim-to-claim similarity):")
        for cn, sim in expanded_claims:
            text = graph.nodes.get(cn, {}).get('text', '')
            preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"    [{sim:.3f}] {preview}")
    else:
        logger.info("  Embedding-expanded claims: none above threshold")

    return direct_claims, expanded_claims


def retrieve_claim_document(
    query: str,
    graph: nx.Graph,
    embedding_model,
    claim_embeddings: Optional[np.ndarray],
    claim_nodes: List[str],
    claim_texts: List[str],
    claim_top_k: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Strategy 5: Claim-Based Retrieval via REVIEWS → CONTAINS graph traversal.

    Finds claims semantically similar to the query, expands the set using
    claim-to-claim embedding similarity (no CLAIM_SIM graph edges needed),
    then traverses REVIEWS edges to linked documents and CONTAINS edges to chunks.

    Score formula:
        S5(chunk) = max over matched claims of: max(0, cos(embed(query), embed(claim)))

    Scores are bounded to [0, 1] — no normalization applied.

    Args:
        query: Query text to match against claims
        graph: Knowledge graph (contains REVIEWS and CONTAINS edges)
        embedding_model: SentenceTransformer model
        claim_embeddings: Pre-computed normalised claim embeddings (n_claims, dim)
        claim_nodes: List of claim node IDs
        claim_texts: List of claim texts corresponding to nodes
        claim_top_k: Number of top direct claims + expansion claims to consider
        logger: Logger instance
        claim_expansion_threshold: Minimum eff_sim to keep an expanded claim

    Returns:
        Tuple of:
        - scores: Dictionary mapping chunk_id to score ∈ [0, 1]
        - matched_claims_details: List of matched claim info dicts
    """
    logger.info("\n--- Strategy 5: Claim-Based Retrieval (REVIEWS traversal + embedding expansion) ---")

    matched_claims_details = []

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

    # Get top-k claims by direct similarity
    top_indices = np.argsort(similarities)[-claim_top_k:][::-1]

    logger.info(f"Claim matching (direct top-{claim_top_k} + embedding expansion):")
    direct_claims, expanded_claims = expand_claims_via_embeddings(
        top_indices=list(top_indices),
        claim_nodes=claim_nodes,
        similarities=similarities,
        claim_embeddings=claim_embeddings,
        max_expanded=claim_top_k,
        expansion_threshold=0.0,
        graph=graph,
        logger=logger,
    )

    all_claims_to_process = direct_claims + expanded_claims

    scores = {}

    for claim_node, claim_sim in all_claims_to_process:
        node_data = graph.nodes.get(claim_node, {})
        claim_text = node_data.get('text', '')
        chunks_found = 0

        # Traverse REVIEWS → CONTAINS to find chunks linked to this claim
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
                    if chunk_id not in scores:
                        scores[chunk_id] = claim_sim
                    else:
                        scores[chunk_id] = max(scores[chunk_id], claim_sim)
                    chunks_found += 1

        matched_claims_details.append({
            'claim_node': claim_node,
            'claim_text': claim_text,
            'query_similarity': claim_sim,
            'rating': node_data.get('rating', ''),
            'chunks_found': chunks_found,
            'source': 'direct' if any(cn == claim_node for cn, _ in direct_claims) else 'embedding'
        })

    logger.info(f"Total matched claims: {len(all_claims_to_process)} "
                f"({len(direct_claims)} direct + {len(expanded_claims)} embedding-expanded)")
    logger.info(f"Found {len(scores)} unique chunks via claim-based retrieval")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")

    return scores, matched_claims_details
