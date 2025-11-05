"""
Test script for retrieval system.

This script loads a pre-built knowledge graph and tests the hybrid
retrieval system with various fact-checking claims.
"""

import os
import sys
import logging
import pickle
import argparse
import spacy
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fact_check_kg.config import RetrievalConfig, GraphConfig
from fact_check_kg.retrieval import HybridRetriever


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
    return logging.getLogger("test_retrieval")


def display_results(claim: str, results: list, top_n: int = 3):
    """
    Display retrieval results in a readable format.

    Args:
        claim: The claim that was searched
        results: List of result dictionaries
        top_n: Number of top results to display in detail
    """
    print("\n" + "=" * 80)
    print(f"CLAIM: {claim}")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    print(f"\nTop {min(top_n, len(results))} results:\n")

    for i, result in enumerate(results[:top_n], 1):
        print(f"\n--- Result #{i} ---")
        print(f"Document ID: {result['doc_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"\nScores:")
        print(f"  Total:  {result['score_total']:.4f}")
        print(f"  Semantic:  {result['score_sem']:.4f}")
        print(f"  Thematic:  {result['score_theme']:.4f}")
        print(f"  Entity:    {result['score_entity']:.4f}")
        print(f"\nText preview:")
        text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
        print(f"  {text_preview}")

    print("\n" + "=" * 80)


def main():
    """Main function to test retrieval."""
    parser = argparse.ArgumentParser(description="Test Knowledge Graph Retrieval")
    parser.add_argument(
        "--graph",
        type=str,
        default="knowledge_graph.gpickle",
        help="Path to graph file"
    )
    parser.add_argument(
        "--claim",
        type=str, 
        help="Claim to verify (optional, uses test claims if not provided)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Weight for semantic score"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Weight for thematic score"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.3,
        help="Weight for entity score"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve"
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
    logger.info("KNOWLEDGE GRAPH RETRIEVAL TEST")
    logger.info("=" * 60)
    logger.info(f"Graph file: {args.graph}")
    logger.info(f"Weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info("=" * 60)

    # Load graph
    logger.info("\nLoading knowledge graph...")
    graph_path = os.path.join(os.path.dirname(__file__), args.graph)
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    logger.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Load topic model
    logger.info("\nLoading topic model...")
    topic_model_path = graph_path.replace('.gpickle', '_topic_model.pkl')
    with open(topic_model_path, 'rb') as f:
        topic_model = pickle.load(f)
    logger.info("Topic model loaded")

    # Load models
    logger.info("\nLoading embedding and NER models...")
    graph_config = GraphConfig()
    embedding_model = SentenceTransformer(graph_config.embedding_model)
    nlp = spacy.load(graph_config.ner_model)
    logger.info("Models loaded")

    # Initialize retriever
    logger.info("\nInitializing retriever...")
    retrieval_config = RetrievalConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        top_k=args.top_k
    )

    retriever = HybridRetriever(
        graph=graph,
        topic_model=topic_model,
        embedding_model=embedding_model,
        nlp=nlp,
        entity_types=graph_config.entity_types,
        alpha=retrieval_config.alpha,
        beta=retrieval_config.beta,
        gamma=retrieval_config.gamma,
        top_k=retrieval_config.top_k,
        theme_threshold=retrieval_config.theme_threshold,
        logger=logger
    )

    # Test claims
    if args.claim:
        test_claims = [args.claim]
    else:
        # Default test claims
        test_claims = [
            "Photo from Gaza 2008 shared on Facebook claiming Kashmir violence",
            "Vaccine causes autism",
            "Trump met Macron in France to discuss NATO"
        ]

    # Run retrieval for each claim
    for claim in test_claims:
        results = retriever.retrieve(claim)
        display_results(claim, results, top_n=3)

    logger.info("\n" + "=" * 60)
    logger.info("RETRIEVAL TEST COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
