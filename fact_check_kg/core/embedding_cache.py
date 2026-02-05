"""
Embedding cache module using FAISS for fast vector similarity search.

This module provides functionality to:
- Save chunk embeddings to FAISS index
- Load embeddings from FAISS index
- Perform fast similarity search
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging


class EmbeddingCache:
    """
    FAISS-based cache for chunk embeddings.

    Stores embeddings with their corresponding chunk IDs for fast retrieval.
    """

    def __init__(
        self,
        dimension: int,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the embedding cache.

        Args:
            dimension: Dimension of embeddings (from config.models.embedding_dimension)
            logger: Optional logger instance
        """
        self.dimension = dimension
        self.logger = logger or logging.getLogger(__name__)

        # FAISS index (Inner Product for normalized embeddings = cosine similarity)
        self.index = None
        self.chunk_ids = []  # Maps index position to chunk ID

    def build_index(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray
    ):
        """
        Build FAISS index from chunk embeddings.

        Args:
            chunk_ids: List of chunk IDs
            embeddings: Numpy array of shape (n_chunks, dimension)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")

        # Create FAISS index (IndexFlatIP for Inner Product = cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)

        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))

        # Store chunk IDs
        self.chunk_ids = chunk_ids

        self.logger.info(f"Built FAISS index with {len(chunk_ids)} chunks")

    def save(self, output_dir: Path, graph_id: str):
        """
        Save FAISS index and metadata to disk.

        Args:
            output_dir: Directory to save files
            graph_id: Graph ID for naming files
        """
        if self.index is None:
            raise ValueError("No index to save. Call build_index() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = output_dir / f"embeddings_{graph_id}.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save chunk IDs mapping
        metadata_path = output_dir / f"embeddings_{graph_id}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunk_ids': self.chunk_ids,
                'dimension': self.dimension
            }, f)

        self.logger.info(f"Saved FAISS index to {index_path}")
        self.logger.info(f"Saved metadata to {metadata_path}")

    def load(self, output_dir: Path, graph_id: str):
        """
        Load FAISS index and metadata from disk.

        Args:
            output_dir: Directory containing saved files
            graph_id: Graph ID for naming files
        """
        output_dir = Path(output_dir)

        # Load FAISS index
        index_path = output_dir / f"embeddings_{graph_id}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(index_path))

        # Load chunk IDs mapping
        metadata_path = output_dir / f"embeddings_{graph_id}.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.chunk_ids = metadata['chunk_ids']
        self.dimension = metadata['dimension']

        self.logger.info(f"Loaded FAISS index with {len(self.chunk_ids)} chunks from {index_path}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar chunks using FAISS.

        Args:
            query_embedding: Query embedding of shape (1, dimension) or (dimension,)
            k: Number of results to return

        Returns:
            Tuple of (chunk_ids, similarities)
        """
        if self.index is None:
            raise ValueError("No index loaded. Call load() or build_index() first.")

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        similarities, indices = self.index.search(query_embedding.astype(np.float32), k)

        # Map indices to chunk IDs
        result_chunk_ids = [self.chunk_ids[i] for i in indices[0]]
        result_similarities = similarities[0].tolist()

        return result_chunk_ids, result_similarities

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Get all embeddings from the index.

        Returns:
            Tuple of (chunk_ids, embeddings)
        """
        if self.index is None:
            raise ValueError("No index loaded. Call load() or build_index() first.")

        # Reconstruct all embeddings from FAISS index
        n = self.index.ntotal
        embeddings = np.zeros((n, self.dimension), dtype=np.float32)

        for i in range(n):
            embeddings[i] = self.index.reconstruct(i)

        return self.chunk_ids, embeddings
