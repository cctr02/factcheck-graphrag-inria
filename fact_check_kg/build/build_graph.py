"""
Main script to build the knowledge graph from CSV data.

This script loads documents, chunks them, and builds the complete
multi-level knowledge graph with documents, chunks, themes, and entities.
"""

import os
import sys
import pickle
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fact_check_kg.core.config_loader import get_build_config, load_raw_config
from fact_check_kg.core.logger import get_logger
from fact_check_kg.build.data_loader import load_and_chunk_documents, load_claims
from fact_check_kg.build.graph_builder import KnowledgeGraphBuilder
from fact_check_kg.build.checkpoint_manager import CheckpointManager


def main():
    """Main function to build and save the knowledge graph."""
    parser = argparse.ArgumentParser(description="Build Knowledge Graph for Fact-Checking")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: from config)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint instead of starting fresh"
    )
    args = parser.parse_args()

    # Load build configuration - ALL parameters come from config
    config = get_build_config()

    # Setup logging first
    logger = get_logger("build_graph", args.log_level)

    # Check for CLI overrides and log them
    max_docs = config.graph_construction.max_documents
    if args.max_docs is not None:
        logger.info(f"[CLI OVERRIDE] max_docs: {config.graph_construction.max_documents} -> {args.max_docs}")
        max_docs = args.max_docs

    # Initialize checkpoint manager
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir, logger=logger)

    if args.resume:
        last = checkpoint_manager.get_last_completed()
        if last:
            logger.info(f"[CHECKPOINT] Resuming from: {last}")
            summary = checkpoint_manager.get_summary()
            logger.info(f"[CHECKPOINT] Completed steps: {summary.get('completed_steps', [])}")
            logger.info(f"[CHECKPOINT] Total checkpoint size: {summary.get('total_size_mb', 0):.1f} MB")

            # Validate config hasn't changed
            if not checkpoint_manager.validate_config(config, max_docs):
                logger.error("[CHECKPOINT] Config has changed since checkpoint was created!")
                logger.error("Run without --resume to start fresh, or delete outputs/checkpoints/")
                sys.exit(1)
        else:
            logger.info("[CHECKPOINT] No checkpoint found, starting fresh build")
    else:
        # Fresh build - clear old checkpoints
        checkpoint_manager.clear()
        checkpoint_manager.init_run(config, max_docs)

    # All parameters from config
    csv_path = config.paths.csv_file
    claims_csv_path = config.paths.claims_csv_file
    chunk_size = config.graph_construction.chunk_size
    chunk_overlap = config.graph_construction.chunk_overlap
    db_host = config.database.host
    db_port = config.database.port
    db_name = config.database.database
    db_user = config.database.user
    db_password = config.database.password

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 60)
    logger.info(f"CSV file: {csv_path}")
    logger.info(f"Claims CSV: {claims_csv_path}")
    logger.info(f"Max documents: {max_docs}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Chunk overlap: {chunk_overlap}")
    logger.info("=" * 60)

    # Step 1: Load and chunk documents
    logger.info("\n### STEP 1: Loading and chunking documents ###")

    chunks, n_docs, doc_metadata = load_and_chunk_documents(
        csv_path=csv_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_docs=max_docs,
        logger=logger
    )

    # Step 1b: Load claims from separate CSV
    claims = None
    claim_to_docs = None
    logger.info("\n### STEP 1b: Loading claims ###")

    if os.path.exists(claims_csv_path):
        claims, claim_to_docs = load_claims(
            csv_path=claims_csv_path,
            max_claims=None,  # Load all claims first, then filter
            logger=logger
        )

        # Filter claims to only those that link to loaded documents
        loaded_doc_ids = set(c['doc_id'] for c in chunks)
        logger.info(f"Loaded {len(loaded_doc_ids)} documents, filtering claims...")

        # Filter claim_to_docs to only include loaded documents
        filtered_claim_to_docs = {}
        for claim_id, doc_ids in claim_to_docs.items():
            matching_docs = [d for d in doc_ids if d in loaded_doc_ids]
            if matching_docs:
                filtered_claim_to_docs[claim_id] = matching_docs

        # Filter claims to only those with at least one matching document
        matching_claim_ids = set(filtered_claim_to_docs.keys())
        claims = [c for c in claims if c['claim_id'] in matching_claim_ids]
        claim_to_docs = filtered_claim_to_docs

        logger.info(f"After filtering: {len(claims)} claims linked to loaded documents")
    else:
        logger.warning(f"Claims CSV not found: {claims_csv_path}")
        logger.warning("Skipping claim loading (Steps 9-10 will be skipped)")

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
        # LLM configuration
        llm_endpoint=config.models.llm_endpoint,
        llm_model=config.models.llm_model,
        # LLM batch labeling configuration
        llm_batch_prompt_template=config.llm_batch_labeling.prompt_template,
        llm_batch_size_default=config.llm_batch_labeling.batch_size_default,
        llm_batch_size_min=config.llm_batch_labeling.batch_size_min,
        llm_batch_size_max=config.llm_batch_labeling.batch_size_max,
        llm_batch_size_step=config.llm_batch_labeling.batch_size_step,
        llm_label_min_words=config.llm_batch_labeling.label_min_words,
        llm_label_max_words=config.llm_batch_labeling.label_max_words,
        llm_label_max_length=config.llm_batch_labeling.label_max_length,
        llm_max_keywords_per_topic=config.llm_batch_labeling.max_keywords_per_topic,
        llm_retry_count=config.llm_batch_labeling.retry_count,
        llm_retry_temperature=config.llm_batch_labeling.retry_temperature,
        llm_batch_timeout=config.llm_batch_labeling.batch_timeout,
        llm_success_streak_threshold=config.llm_batch_labeling.success_streak_threshold,
        llm_hysteresis_decrease_factor=config.llm_batch_labeling.hysteresis_decrease_factor,
        llm_failure_strategy=config.llm_batch_labeling.failure_strategy,
        # Keyphrase configuration
        keyphrase_top_n=config.graph_construction.keyphrase_top_n,
        keyphrase_ngram_range=(
            config.graph_construction.keyphrase_ngram_min,
            config.graph_construction.keyphrase_ngram_max
        ),
        logger=logger
    )

    # Step 3: Build graph
    logger.info("\n### STEP 3: Building knowledge graph ###")
    logger.info("Using LLM for topic labeling (requires Ollama running)...")

    graph, topic_probabilities, chunk_ids, claim_retrieval_results = builder.build_graph(
        chunks=chunks,
        min_topic_size=config.graph_construction.min_topic_size,
        nr_topics=config.graph_construction.nr_topics,
        claims=claims,
        claim_to_docs=claim_to_docs,
        doc_metadata=doc_metadata,
        checkpoint_manager=checkpoint_manager
    )

    # Step 4: Save graph and models
    logger.info("\n### STEP 4: Saving graph and models ###")

    # Generate timestamp for unique graph identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure output directories exist
    graphs_dir = Path(config.paths.graphs_dir)
    models_dir = Path(config.paths.models_dir)
    embeddings_dir = Path(config.paths.embeddings_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Get edge counts from the graph
    edge_type_counts = {}
    for _, _, data in graph.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

    # 1. Save graph as pickle in outputs/graphs
    graph_filename = f"graph_{timestamp}.gpickle"
    pickle_path = graphs_dir / graph_filename

    with open(pickle_path, 'wb') as f:
        pickle.dump(graph, f)
    logger.info(f"Graph (pickle) saved to: {pickle_path}")

    # 2. Save topic model in outputs/models
    topic_model_filename = f"topic_model_{timestamp}.pkl"
    topic_model_path = models_dir / topic_model_filename
    with open(topic_model_path, 'wb') as f:
        pickle.dump(builder.topic_model, f)
    logger.info(f"Topic model saved to: {topic_model_path}")

    # 2b. Save topic probabilities for efficient retrieval in outputs/models
    topic_probs_filename = f"topic_probs_{timestamp}.npz"
    topic_probs_path = models_dir / topic_probs_filename
    np.savez_compressed(
        topic_probs_path,
        probabilities=topic_probabilities,
        chunk_ids=chunk_ids
    )
    logger.info(f"Topic probabilities saved to: {topic_probs_path} (shape: {topic_probabilities.shape})")

    # 2c. Save claim pre-retrieval results for Strategy 5
    if claim_retrieval_results:
        claim_retrieval_filename = f"claim_retrieval_{timestamp}.json"
        claim_retrieval_path = models_dir / claim_retrieval_filename
        with open(claim_retrieval_path, 'w', encoding='utf-8') as f:
            json.dump(claim_retrieval_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Claim pre-retrieval results saved to: {claim_retrieval_path} ({len(claim_retrieval_results)} claims)")

    # 3. Save chunk embeddings to FAISS index in outputs/embeddings
    logger.info("Computing and saving chunk embeddings to FAISS...")
    from fact_check_kg.core.embedding_cache import EmbeddingCache

    # Extract chunk IDs and texts
    chunk_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chunk']
    chunk_texts = [graph.nodes[n]['text'] for n in chunk_nodes]

    # Compute embeddings with optimal batch size for GPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128 if device == 'cuda' else 32

    logger.info(f"Computing embeddings on {device.upper()} with batch_size={batch_size}...")
    embeddings = builder.embedding_model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device
    )

    # Build and save FAISS index
    embedding_cache = EmbeddingCache(
        dimension=config.models.embedding_dimension,
        logger=logger
    )
    embedding_cache.build_index(chunk_nodes, embeddings)
    embedding_cache.save(embeddings_dir, timestamp)

    logger.info(f"Chunk embeddings saved for {len(chunk_nodes)} chunks")

    # 4. Create metadata
    # Load raw configs separately
    base_config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    build_config_path = Path(__file__).parent.parent.parent / "config" / "config_build.yaml"

    metadata = {
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(),
        "paths": {
            "graph": str(pickle_path),
            "topic_model": str(topic_model_path),
            "topic_probs": str(topic_probs_path),
            "claim_retrieval": str(models_dir / f"claim_retrieval_{timestamp}.json") if claim_retrieval_results else None,
            "embeddings_faiss": str(embeddings_dir / f"embeddings_{timestamp}.faiss"),
            "embeddings_meta": str(embeddings_dir / f"embeddings_{timestamp}.pkl"),
        },
        "max_documents": max_docs,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_documents": n_docs,
        "num_chunks": len(chunks),
        "num_claims": len(claims) if claims else 0,
        "edge_types": edge_type_counts,
        "config": load_raw_config(base_config_path),  # Save common configuration
        "config_build": load_raw_config(build_config_path)  # Save build-specific configuration
    }

    # 4. Save to PostgreSQL with timestamped tables (optional - continues if fails)
    try:
        logger.info(f"Saving to PostgreSQL with graph_id: {timestamp}")
        builder.save_to_postgres(
            graph_id=timestamp,
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            metadata=metadata,
            graph=graph  # Explicitly pass the same graph saved in pickle
        )
        logger.info(f"Graph (PostgreSQL) saved successfully with graph_id: {timestamp}")
        metadata["storage_type"] = "dual"
        metadata["postgres_database"] = db_name
        metadata["postgres_host"] = db_host
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
