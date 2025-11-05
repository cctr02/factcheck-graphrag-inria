"""
Data loading module for Knowledge Graph construction.

This module handles loading documents from CSV and chunking them.
"""

import os
import sys
from typing import List, Tuple, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.io_utils import load_corpus_from_csv


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks by characters.

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks

    Example:
        >>> text = "This is a long text..." * 100
        >>> chunks = chunk_text(text, chunk_size=200, overlap=50)
        >>> len(chunks[0]), len(chunks[1])
        (200, 200)
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # Avoid empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())

        # Move start position with overlap
        start = end - overlap

        # Break if we're at the end
        if end >= text_len:
            break

    return chunks


def load_and_chunk_documents(
    csv_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    lang_value: str = "english",
    max_docs: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[dict], int]:
    """
    Load documents from CSV and chunk them.

    Args:
        csv_path: Path to CSV file
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        lang_value: Language filter
        max_docs: Maximum number of documents to load (None = all)
        logger: Optional logger instance

    Returns:
        Tuple of (list of chunk dicts, total number of documents)
        Each chunk dict contains:
            - chunk_id: str (format: "chunk::doc_id::chunk_idx")
            - doc_id: str
            - text: str
            - chunk_idx: int

    Example:
        >>> chunks, n_docs = load_and_chunk_documents(
        ...     "data.csv", chunk_size=500, max_docs=10
        ... )
        >>> len(chunks) >= n_docs  # Multiple chunks per document
        True
    """
    if logger:
        logger.info(f"Loading documents from {csv_path}")

    # Load documents
    texts, doc_names = load_corpus_from_csv(
        csv_path,
        lang_value=lang_value,
        logger=logger
    )

    # Limit documents if specified
    if max_docs:
        if logger:
            logger.info(f"Limiting to first {max_docs} documents")
        texts = texts[:max_docs]
        doc_names = doc_names[:max_docs]

    # Chunk documents
    all_chunks = []
    chunk_count = 0

    for doc_id, text in zip(doc_names, texts):
        doc_chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)

        for chunk_idx, chunk_content in enumerate(doc_chunks):
            chunk_dict = {
                'chunk_id': f"chunk::{doc_id}::{chunk_idx}",
                'doc_id': doc_id,
                'text': chunk_content,
                'chunk_idx': chunk_idx
            }
            all_chunks.append(chunk_dict)
            chunk_count += 1

    if logger:
        logger.info(f"Created {chunk_count} chunks from {len(texts)} documents")
        logger.info(f"Average chunks per document: {chunk_count / len(texts):.1f}")

    return all_chunks, len(texts)
