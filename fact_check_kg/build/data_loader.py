"""
Data loading module for Knowledge Graph construction.

This module handles loading documents from CSV and chunking them.
"""

import os
import sys
from typing import List, Tuple, Optional, Dict, Any
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.io_utils import load_corpus_from_csv, load_claims_from_csv


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks respecting sentence boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters (soft limit)
        overlap: Approximate number of overlapping characters between chunks

    Returns:
        List of text chunks (each chunk ends at a sentence boundary)

    Example:
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> chunks = chunk_text(text, chunk_size=30, overlap=10)
        >>> chunks[0]
        'First sentence. Second sentence.'
    """
    import re

    # Split text into sentences using regex
    # Match sentence endings: . ! ? followed by space or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(sentence_pattern, text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(sentence)

        # If adding this sentence would exceed chunk_size
        if current_length > 0 and current_length + sentence_length + 1 > chunk_size:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            if chunk_text:
                chunks.append(chunk_text)

            # Start new chunk with overlap
            # Calculate how many sentences to keep for overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) + 1 <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break

            # Start new chunk with overlap sentences
            current_chunk = overlap_sentences
            current_length = overlap_length

        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length + (1 if current_length > 0 else 0)  # +1 for space
        i += 1

    # Add last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def load_and_chunk_documents(
    csv_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    lang_value: str = "english",
    max_docs: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[dict], int, Dict[str, Dict[str, Any]]]:
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
        Tuple of (list of chunk dicts, total number of documents, doc_metadata_dict)
        Each chunk dict contains:
            - chunk_id: str (format: "chunk::doc_id::chunk_idx")
            - doc_id: str
            - text: str
            - chunk_idx: int
            - review_date: str
            - language: str
        doc_metadata_dict maps doc_id to {text, review_date, language}

    Example:
        >>> chunks, n_docs, doc_meta = load_and_chunk_documents(
        ...     "data.csv", chunk_size=500, max_docs=10
        ... )
        >>> len(chunks) >= n_docs  # Multiple chunks per document
        True
    """
    if logger:
        logger.info(f"Loading documents from {csv_path}")

    # Load documents with metadata
    # Try to detect the correct language column (fc_lang or language)
    import pandas as pd
    df_peek = pd.read_csv(csv_path, nrows=1)
    if "fc_lang" in df_peek.columns:
        lang_col = "fc_lang"
    elif "language" in df_peek.columns:
        lang_col = "language"
    else:
        lang_col = "fc_lang"  # Default, will raise error if missing

    texts, doc_names, metadata_list = load_corpus_from_csv(
        csv_path,
        lang_value=lang_value,
        logger=logger,
        return_metadata=True,
        lang_col=lang_col,
        review_date_col="review_date"
    )

    # Limit documents if specified
    if max_docs:
        if logger:
            logger.info(f"Limiting to first {max_docs} documents")
        texts = texts[:max_docs]
        doc_names = doc_names[:max_docs]
        if metadata_list:
            metadata_list = metadata_list[:max_docs]

    # Build doc_id -> metadata mapping
    doc_metadata: Dict[str, Dict[str, Any]] = {}
    if metadata_list:
        for doc_id, meta in zip(doc_names, metadata_list):
            doc_metadata[doc_id] = {
                "text": meta["text"],  # title\nbody format
                "review_date": meta["review_date"],
                "language": meta["language"]
            }

    # Chunk documents
    all_chunks = []
    chunk_count = 0

    for idx, (doc_id, text) in enumerate(zip(doc_names, texts)):
        doc_chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)

        # Get metadata for this document
        meta = doc_metadata.get(doc_id, {})
        review_date = meta.get("review_date", "")
        language = meta.get("language", "")

        for chunk_idx, chunk_content in enumerate(doc_chunks):
            chunk_dict = {
                'chunk_id': f"chunk::{doc_id}::{chunk_idx}",
                'doc_id': doc_id,
                'text': chunk_content,
                'chunk_idx': chunk_idx,
                'review_date': review_date,
                'language': language
            }
            all_chunks.append(chunk_dict)
            chunk_count += 1

    if logger:
        logger.info(f"Created {chunk_count} chunks from {len(texts)} documents")
        logger.info(f"Average chunks per document: {chunk_count / len(texts):.1f}")

    return all_chunks, len(texts), doc_metadata


def load_claims(
    csv_path: str,
    lang_value: str = "english",
    max_claims: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Load claims from CSV.

    Args:
        csv_path: Path to claim_reviews CSV file
        lang_value: Language filter (on fc_lang column)
        max_claims: Maximum number of claims to load (None = all)
        logger: Optional logger instance

    Returns:
        Tuple of (claims_list, claim_to_docs_mapping)
        - claims_list: List of claim dicts with keys:
            - claim_id: str
            - text: str (the claim)
            - rating: str
            - claimant: str
            - claimant_norm: str
            - review_date: str
            - language: str (from claim_lang)
        - claim_to_docs_mapping: Dict mapping claim_id to list of doc_ids

    Example:
        >>> claims, claim_to_docs = load_claims("claim_reviews.csv")
        >>> claims[0]["text"]
        'Some claim text...'
    """
    if logger:
        logger.info(f"Loading claims from {csv_path}")

    claims_list, claim_to_docs = load_claims_from_csv(
        csv_path,
        lang_value=lang_value,
        logger=logger
    )

    # Limit claims if specified
    if max_claims and len(claims_list) > max_claims:
        if logger:
            logger.info(f"Limiting to first {max_claims} claims")
        # Keep only the first max_claims
        claims_to_keep = set(c["claim_id"] for c in claims_list[:max_claims])
        claims_list = claims_list[:max_claims]
        # Filter claim_to_docs to only include kept claims
        claim_to_docs = {k: v for k, v in claim_to_docs.items() if k in claims_to_keep}

    if logger:
        logger.info(f"Loaded {len(claims_list)} claims")

    return claims_list, claim_to_docs
