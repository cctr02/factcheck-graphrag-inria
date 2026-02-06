"""
Document, Chunk, and Keyphrase node creation module.

This module handles creation of the base hierarchical structure:
- Document nodes: Source documents with metadata (title, body, review_date)
- Chunk nodes: Text segments from documents (sliding window with overlap)
- CONTAINS edges: Document -> Chunk (hierarchical structure, weight=1.0)
- Keyphrase nodes: Key phrases extracted via KeyBERT with MMR diversity
- HAS_KEYPHRASE edges: Chunk -> Keyphrase (weighted by KeyBERT relevance score)

Keyphrase Extraction:
    Uses KeyBERT with Maximal Marginal Relevance (MMR) for diverse keyphrases.

    MMR formula:
        MMR = argmax[λ * sim(d, q) - (1-λ) * max(sim(d, d_j))]

    Where:
    - sim(d, q) = cosine similarity between candidate and document
    - sim(d, d_j) = similarity to already selected keyphrases
    - λ = diversity parameter (default 0.5)

    Edge weight = KeyBERT relevance score ∈ [0, 1]
"""

import time
import logging
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import networkx as nx
from keybert import KeyBERT


def add_documents_and_chunks(
    graph: nx.Graph,
    chunks: List[Dict],
    doc_metadata: Optional[Dict[str, Dict]],
    logger: logging.Logger,
    log_step_fn
):
    """
    Add document and chunk nodes with CONTAINS edges.

    Creates the base hierarchical structure of the knowledge graph:
    - Document nodes (type="document") represent source fact-check articles
    - Chunk nodes (type="chunk") represent text segments for fine-grained retrieval
    - CONTAINS edges link documents to their constituent chunks

    Node attributes:
        Document: doc_id, text (full document), review_date, language
        Chunk: doc_id, text, chunk_idx, review_date, language

    Edge attributes:
        CONTAINS: edge_type="CONTAINS" (no weight, structural edge)

    Args:
        graph: NetworkX graph to add nodes/edges to
        chunks: List of chunk dictionaries with keys:
            - chunk_id: str (format: "chunk::{doc_id}::{chunk_idx}")
            - doc_id: str (source document identifier)
            - text: str (chunk text content)
            - chunk_idx: int (position in document, 0-indexed)
            - review_date: str (optional, ISO date)
            - language: str (optional, e.g., "en")
        doc_metadata: Optional mapping of doc_id to metadata dict with keys:
            text (title + "\\n" + body), review_date, language
        logger: Logger instance
        log_step_fn: Callback function(step_name, start_time, count)

    Example:
        >>> add_documents_and_chunks(graph, chunks, doc_meta, logger, log_fn)
        >>> graph.nodes["doc::123"]["type"]
        'document'
        >>> graph.nodes["chunk::123::0"]["chunk_idx"]
        0
    """
    start_time = time.time()
    doc_ids_seen: Set[str] = set()

    for chunk in tqdm(chunks, desc="Adding documents and chunks", disable=logger.level > logging.INFO):
        chunk_id = chunk['chunk_id']
        doc_id = chunk['doc_id']
        text = chunk['text']

        # Add document node (once per document)
        doc_node = f"doc::{doc_id}"
        if doc_node not in doc_ids_seen:
            # Get document metadata if available
            meta = doc_metadata.get(doc_id, {}) if doc_metadata else {}
            doc_text = meta.get("text", "")
            review_date = meta.get("review_date", "")
            language = meta.get("language", "")

            graph.add_node(
                doc_node,
                type="document",
                doc_id=doc_id,
                text=doc_text,
                review_date=review_date,
                language=language
            )
            doc_ids_seen.add(doc_node)

        # Get chunk metadata
        chunk_review_date = chunk.get('review_date', '')
        chunk_language = chunk.get('language', '')

        # Add chunk node
        graph.add_node(
            chunk_id,
            type="chunk",
            doc_id=doc_id,
            text=text,
            chunk_idx=chunk['chunk_idx'],
            review_date=chunk_review_date,
            language=chunk_language
        )

        # Add CONTAINS edge
        graph.add_edge(doc_node, chunk_id, edge_type="CONTAINS")

    log_step_fn("Adding documents and chunks", start_time, len(chunks))


def add_keyphrases(
    graph: nx.Graph,
    chunks: List[Dict],
    keybert_model: KeyBERT,
    keyphrase_top_n: int,
    keyphrase_ngram_range: tuple,
    logger: logging.Logger,
    log_step_fn
):
    """
    Extract keyphrases from chunks using KeyBERT with MMR diversity.

    Uses KeyBERT to extract semantically relevant keyphrases that capture
    the main concepts in each chunk. MMR (Maximal Marginal Relevance)
    ensures diversity among selected keyphrases.

    Algorithm:
    1. Encode document and candidate phrases using SentenceTransformer
    2. Compute cosine similarity between candidates and document
    3. Apply MMR to select diverse, relevant keyphrases:
       MMR = λ * relevance - (1-λ) * max_similarity_to_selected

    Node attributes:
        Keyphrase: type="keyphrase", text (normalized), phrase (original)

    Edge attributes:
        HAS_KEYPHRASE: edge_type="HAS_KEYPHRASE", weight=relevance_score

    Args:
        graph: NetworkX graph to add nodes/edges to
        chunks: List of chunk dictionaries with 'chunk_id' and 'text' keys
        keybert_model: Initialized KeyBERT model (uses SentenceTransformer backend)
        keyphrase_top_n: Number of keyphrases to extract per chunk (e.g., 5)
        keyphrase_ngram_range: Tuple of (min_n, max_n) for n-gram range (e.g., (1, 3))
        logger: Logger instance
        log_step_fn: Callback function(step_name, start_time, count)

    Example:
        >>> add_keyphrases(graph, chunks, keybert, top_n=5, ngram=(1,3), logger, log_fn)
        >>> graph.nodes["keyphrase::climate change"]["type"]
        'keyphrase'
        >>> graph["chunk::1::0"]["keyphrase::climate change"]["weight"]
        0.85
    """
    start_time = time.time()
    keyphrase_count = 0
    edge_count = 0

    # Track unique keyphrases to avoid duplicates
    keyphrase_to_node: Dict[str, bool] = {}

    for chunk in tqdm(chunks, desc="Extracting keyphrases", disable=logger.level > logging.INFO):
        chunk_id = chunk['chunk_id']
        text = chunk['text']

        # Skip empty or very short texts
        if not text or len(text.strip()) < 20:
            continue

        try:
            # Extract keyphrases using KeyBERT
            keyphrases = keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words='english',
                top_n=keyphrase_top_n,
                use_mmr=True,  # Use Maximal Marginal Relevance for diversity
                diversity=0.5
            )

            for phrase, score in keyphrases:
                # Normalize keyphrase (lowercase, strip)
                phrase_normalized = phrase.lower().strip()

                # Skip very short phrases
                if len(phrase_normalized) < 3:
                    continue

                # Create or get keyphrase node
                keyphrase_node = f"keyphrase::{phrase_normalized}"

                if keyphrase_node not in keyphrase_to_node:
                    # Create new keyphrase node
                    graph.add_node(
                        keyphrase_node,
                        type="keyphrase",
                        text=phrase_normalized,
                        phrase=phrase  # Original casing
                    )
                    keyphrase_to_node[keyphrase_node] = True
                    keyphrase_count += 1

                # Add HAS_KEYPHRASE edge with relevance score
                graph.add_edge(
                    chunk_id,
                    keyphrase_node,
                    edge_type="HAS_KEYPHRASE",
                    weight=float(score)
                )
                edge_count += 1

        except Exception as e:
            # Log but don't fail on individual chunk errors
            logger.debug(f"KeyBERT extraction failed for {chunk_id}: {e}")
            continue

    logger.info(f"Created {keyphrase_count} unique keyphrase nodes")
    log_step_fn("Extracting keyphrases", start_time, edge_count)
