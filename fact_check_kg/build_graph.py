"""
Main script to build the knowledge graph from CSV data.

This script loads documents, chunks them, and builds the complete
multi-level knowledge graph with documents, chunks, themes, and entities.
"""

import os
import sys
import logging
import pickle
import argparse

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fact_check_kg.config import GraphConfig
from fact_check_kg.data_loader import load_and_chunk_documents
from fact_check_kg.graph_builder import KnowledgeGraphBuilder


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("build_graph")


def main():
    """Main function to build and save the knowledge graph."""
    parser = argparse.ArgumentParser(description="Build Knowledge Graph for Fact-Checking")
    parser.add_argument(
        "--csv",
        type=str,
        default="../data/english_fact_checks.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="knowledge_graph.gpickle",
        help="Output path for graph file"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=100,
        help="Maximum number of documents to process (for testing)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Size of text chunks in characters"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between chunks in characters"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 60)
    logger.info(f"CSV file: {args.csv}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max documents: {args.max_docs}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Chunk overlap: {args.chunk_overlap}")
    logger.info("=" * 60)

    # Load configuration
    config = GraphConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Step 1: Load and chunk documents
    logger.info("\n### STEP 1: Loading and chunking documents ###")
    csv_path = os.path.join(os.path.dirname(__file__), args.csv)

    chunks, n_docs = load_and_chunk_documents(
        csv_path=csv_path,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        max_docs=args.max_docs,
        logger=logger
    )

    # Step 2: Initialize graph builder
    logger.info("\n### STEP 2: Initializing graph builder ###")
    builder = KnowledgeGraphBuilder(
        embedding_model=config.embedding_model,
        ner_model=config.ner_model,
        entity_types=config.entity_types,
        co_occur_threshold=config.co_occur_threshold,
        semantic_sim_threshold=config.semantic_sim_threshold,
        entity_related_threshold=config.entity_related_threshold,
        logger=logger
    )

    # Step 3: Build graph
    logger.info("\n### STEP 3: Building knowledge graph ###")
    graph = builder.build_graph(
        chunks=chunks,
        min_topic_size=config.min_topic_size,
        nr_topics=config.nr_topics
    )

    # Step 4: Save graph and models
    logger.info("\n### STEP 4: Saving graph and models ###")

    # Save graph
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    logger.info(f"Graph saved to: {output_path}")

    # Save topic model (for retrieval)
    topic_model_path = output_path.replace('.gpickle', '_topic_model.pkl')
    with open(topic_model_path, 'wb') as f:
        pickle.dump(builder.topic_model, f)
    logger.info(f"Topic model saved to: {topic_model_path}")

    logger.info("\n" + "=" * 60)
    logger.info("GRAPH BUILDING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
