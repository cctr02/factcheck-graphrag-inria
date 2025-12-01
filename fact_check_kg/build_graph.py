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

from fact_check_kg.config_loader import get_build_config, load_raw_config
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
        "--use-llm-labels",
        action="store_true",
        help="Use LLM to generate topic labels (slower but more descriptive)"
    )
    args = parser.parse_args()

    # Load build configuration
    config = get_build_config()

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
        topic_prob_threshold=config.graph_construction.topic_probability_threshold,
        temporal_max_year_gap=config.graph_construction.temporal_max_year_gap,
        temporal_decay_rate=config.graph_construction.temporal_decay_rate,
        topic_keywords_initial=config.graph_construction.topic_keywords_initial,
        topic_keywords_final=config.graph_construction.topic_keywords_final,
        llm_endpoint=config.models.llm_endpoint,
        llm_model=config.models.llm_model,
        llm_temperature=config.models.llm_temperature,
        llm_max_tokens=config.models.llm_max_tokens,
        llm_timeout=config.models.llm_timeout,
        llm_label_max_length=config.models.llm_label_max_length,
        llm_label_max_words=config.models.llm_label_max_words,
        logger=logger
    )

    # Step 3: Build graph
    logger.info("\n### STEP 3: Building knowledge graph ###")
    if args.use_llm_labels:
        logger.info("Using LLM for topic labeling (this may take a while)...")
    else:
        logger.info("Using keyword-based topic labels (top 3 words)")

    graph = builder.build_graph(
        chunks=chunks,
        min_topic_size=config.graph_construction.min_topic_size,
        nr_topics=config.graph_construction.nr_topics,
        use_llm_labels=args.use_llm_labels
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

    # DUAL STORAGE: Save to BOTH pickle and PostgreSQL
    logger.info("Saving graph to BOTH pickle and PostgreSQL...")

    # 1. Save graph as pickle in outputs/graphs
    graph_filename = f"graph_{timestamp}.gpickle"
    pickle_path = graphs_dir / graph_filename

    with open(pickle_path, 'wb') as f:
        pickle.dump(graph, f)
    logger.info(f"Graph (pickle) saved to: {pickle_path}")

    # 2. Save topic model in outputs/graphs
    topic_model_filename = f"topic_model_{timestamp}.pkl"
    topic_model_path = graphs_dir / topic_model_filename
    with open(topic_model_path, 'wb') as f:
        pickle.dump(builder.topic_model, f)
    logger.info(f"Topic model saved to: {topic_model_path}")

    # 3. Save chunk embeddings to FAISS index
    logger.info("Computing and saving chunk embeddings to FAISS...")
    from fact_check_kg.embedding_cache import EmbeddingCache

    # Extract chunk IDs and texts
    chunk_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chunk']
    chunk_texts = [graph.nodes[n]['text'] for n in chunk_nodes]

    # Compute embeddings
    embeddings = builder.embedding_model.encode(
        chunk_texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Build and save FAISS index
    embedding_cache = EmbeddingCache(
        dimension=config.models.embedding_dimension,
        logger=logger
    )
    embedding_cache.build_index(chunk_nodes, embeddings)
    embedding_cache.save(graphs_dir, timestamp)

    logger.info(f"Chunk embeddings saved for {len(chunk_nodes)} chunks")

    # 4. Create metadata
    # Load raw configs separately
    base_config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    build_config_path = Path(__file__).parent.parent / "config" / "config_build.yaml"

    metadata = {
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(),
        "pickle_path": str(pickle_path),
        "topic_model_path": str(topic_model_path),
        "max_documents": args.max_docs,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_documents": n_docs,
        "num_chunks": len(chunks),
        "edge_types": edge_type_counts,
        "config": load_raw_config(base_config_path),  # Save common configuration
        "config_build": load_raw_config(build_config_path)  # Save build-specific configuration
    }

    # 4. Save to PostgreSQL with timestamped tables (optional - continues if fails)
    try:
        logger.info(f"Saving to PostgreSQL with graph_id: {timestamp}")
        builder.save_to_postgres(
            graph_id=timestamp,
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password,
            metadata=metadata
        )
        logger.info(f"Graph (PostgreSQL) saved successfully with graph_id: {timestamp}")
        metadata["storage_type"] = "dual"
        metadata["postgres_database"] = args.db_name
        metadata["postgres_host"] = args.db_host
        metadata["postgres_graph_id"] = timestamp
    except Exception as e:
        logger.warning(f"Failed to save to PostgreSQL: {e}")
        logger.warning("Continuing with pickle-only storage...")
        metadata["storage_type"] = "pickle"

    # 5. Save metadata to JSON file
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
