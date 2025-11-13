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
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fact_check_kg.config_loader import get_config
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
    parser.add_argument(
        "--db-host",
        type=str,
        default="localhost",
        help="PostgreSQL host"
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="PostgreSQL port"
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="knowledge_graph",
        help="PostgreSQL database name"
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default="postgres",
        help="PostgreSQL username"
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default="postgres",
        help="PostgreSQL password"
    )
    parser.add_argument(
        "--use-postgres",
        action="store_true",
        help="Save to PostgreSQL instead of pickle file"
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config()

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

    # Step 1: Load and chunk documents
    logger.info("\n### STEP 1: Loading and chunking documents ###")

    # Handle both absolute and relative paths
    if os.path.isabs(args.csv):
        csv_path = args.csv
    else:
        # If relative path starts with "..", resolve from script directory
        # Otherwise, resolve from current working directory (for CLI usage)
        if args.csv.startswith(".."):
            csv_path = os.path.join(os.path.dirname(__file__), args.csv)
        else:
            csv_path = args.csv

    chunks, n_docs = load_and_chunk_documents(
        csv_path=csv_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_docs=args.max_docs,
        logger=logger
    )

    # Step 2: Initialize graph builder
    logger.info("\n### STEP 2: Initializing graph builder ###")
    builder = KnowledgeGraphBuilder(
        embedding_model=config.models.embedding_model,
        ner_model=config.models.ner_model,
        entity_types=config.graph_construction.entity_types,
        co_occur_threshold=config.graph_construction.co_occurrence_threshold,
        semantic_sim_threshold=config.graph_construction.semantic_similarity_threshold,
        logger=logger
    )

    # Step 3: Build graph
    logger.info("\n### STEP 3: Building knowledge graph ###")
    graph = builder.build_graph(
        chunks=chunks,
        min_topic_size=config.graph_construction.min_topic_size,
        nr_topics=config.graph_construction.nr_topics
    )

    # Step 4: Save graph and models
    logger.info("\n### STEP 4: Saving graph and models ###")

    # Generate timestamp for unique graph identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure output directories exist
    graphs_dir = Path(config.paths.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Get edge counts from the graph
    edge_type_counts = {}
    for _, _, data in graph.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

    # Create metadata
    metadata = {
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(),
        "max_documents": args.max_docs,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_documents": n_docs,
        "num_chunks": len(chunks),
        "edge_types": edge_type_counts
    }

    if args.use_postgres:
        # Create database with timestamp
        db_name = f"{args.db_name}_{timestamp}"
        logger.info(f"Creating PostgreSQL database: {db_name}")

        # Save to PostgreSQL
        builder.save_to_postgres(
            host=args.db_host,
            port=args.db_port,
            database=db_name,
            user=args.db_user,
            password=args.db_password,
            clear_existing=True
        )

        metadata["storage_type"] = "postgresql"
        metadata["database_name"] = db_name
        metadata["database_host"] = args.db_host

        # Save topic model in models directory
        models_dir = Path(config.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        topic_model_path = models_dir / f"topic_model_{timestamp}.pkl"
        with open(topic_model_path, 'wb') as f:
            pickle.dump(builder.topic_model, f)
        logger.info(f"Topic model saved to: {topic_model_path}")
        metadata["topic_model_path"] = str(topic_model_path)

    else:
        # Save graph as pickle in outputs/graphs
        graph_filename = f"graph_{timestamp}.gpickle"
        output_path = graphs_dir / graph_filename

        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
        logger.info(f"Graph saved to: {output_path}")

        metadata["storage_type"] = "pickle"
        metadata["graph_path"] = str(output_path)

        # Save topic model with same timestamp
        topic_model_filename = f"topic_model_{timestamp}.pkl"
        topic_model_path = graphs_dir / topic_model_filename
        with open(topic_model_path, 'wb') as f:
            pickle.dump(builder.topic_model, f)
        logger.info(f"Topic model saved to: {topic_model_path}")
        metadata["topic_model_path"] = str(topic_model_path)

    # Save metadata
    metadata_path = graphs_dir / f"metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, indent=2, fp=f)
    logger.info(f"Metadata saved to: {metadata_path}")

    # Save graph ID for easy reference
    logger.info(f"\nGraph ID: {timestamp}")
    logger.info(f"To use this graph later, specify: --graph-id {timestamp}")

    logger.info("\n" + "=" * 60)
    logger.info("GRAPH BUILDING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
