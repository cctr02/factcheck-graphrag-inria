"""
Strategy 1: Semantic Similarity Retrieval.

Direct embedding similarity between claim and chunk embeddings.
"""

import logging
from typing import Dict
import numpy as np


def retrieve_semantic(
    claim: str,
    embedding_model,
    chunk_embeddings: np.ndarray,
    chunk_nodes: list,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Strategy 1: Semantic Similarity using embedding similarity.

    Computes cosine similarity between the claim embedding and all chunk embeddings.
    Returns normalized scores (min-max scaling to [0, 1]).

    Args:
        claim: Claim text to verify
        embedding_model: SentenceTransformer model
        chunk_embeddings: Pre-computed chunk embeddings (n_chunks, dim)
        chunk_nodes: List of chunk node IDs corresponding to embeddings
        logger: Logger instance

    Returns:
        Dictionary mapping chunk_id to semantic score (normalized 0-1)

    Example:
        >>> scores = retrieve_semantic(
        ...     "Climate change is a hoax",
        ...     embedding_model, chunk_embeddings, chunk_nodes, logger
        ... )
        >>> scores["chunk::doc1::0"]
        0.85
    """
    logger.info("\n--- Strategy 1: Semantic Similarity ---")

    # Embed claim
    claim_embedding = embedding_model.encode(
        [claim],
        normalize_embeddings=True
    )[0]

    # Compute similarities
    similarities = np.matmul(chunk_embeddings, claim_embedding)

    # Create score dictionary
    scores = {}
    for chunk_node, sim in zip(chunk_nodes, similarities):
        scores[chunk_node] = float(sim)

    # Clip to [0, 1]: cosine of normalized embeddings is in [-1, 1], negative = unrelated
    scores = {k: max(0.0, v) for k, v in scores.items()}

    logger.info(f"Found {len(scores)} chunks with semantic scores")
    if scores:
        logger.info(f"Top score: {max(scores.values()):.3f}")

    return scores
