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

from fact_check_kg.config import RetrievalConfig, GraphConfig, DatabaseConfig
from fact_check_kg.retrieval import HybridRetriever
from fact_check_kg.db_connector import GraphDatabaseConnector
from fact_check_kg.retrieval_results import RetrievalResultsManager


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


def display_results(
    claim: str,
    results: list,
    alpha: float,
    beta: float,
    gamma: float,
    results_manager: RetrievalResultsManager,
    top_n: int = 5,
    save_results: bool = True
):
    """
    Display retrieval results with detailed scoring and multiple rankings.

    Args:
        claim: The claim that was searched
        results: List of result dictionaries
        alpha: Semantic weight
        beta: Thematic weight
        gamma: Entity weight
        results_manager: RetrievalResultsManager instance
        top_n: Number of top results to display in detail
        save_results: Whether to save results to files
    """
    # Display results in console
    results_manager.display_results_console(claim, results, top_n=top_n)

    # Generate multiple rankings
    rankings = results_manager.create_multiple_rankings(results)

    # Display alternative rankings summary
    print("\n" + "=" * 80)
    print("ALTERNATIVE RANKINGS (Top 3)")
    print("=" * 80)

    for ranking_name, ranking_results in rankings.items():
        if ranking_name == "by_total_score":
            continue  # Already shown above

        print(f"\n{ranking_name.replace('_', ' ').title()}:")
        for i, result in enumerate(ranking_results[:3], 1):
            print(f"  {i}. {result['chunk_id']} - Total: {result['score_total']:.4f}")

    # Generate statistics
    stats = results_manager.generate_statistics(results)
    if stats:
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"\nScore Averages:")
        for score_type, values in stats["score_statistics"].items():
            print(f"  {score_type.capitalize()}: {values['mean']:.4f} (min: {values['min']:.4f}, max: {values['max']:.4f})")

        print(f"\nComponent Dominance:")
        for component, count in stats["component_analysis"].items():
            percentage = (count / stats["total_results"]) * 100
            print(f"  {component.capitalize()}: {count} ({percentage:.1f}%)")

    # Save results to files
    if save_results and results:
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        json_path, text_path = results_manager.save_results(
            claim, results, alpha, beta, gamma, rankings
        )
        print(f"Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {text_path}")

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
    parser.add_argument(
        "--use-postgres",
        action="store_true",
        help="Load from PostgreSQL instead of pickle file"
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
    if args.use_postgres:
        db_config = DatabaseConfig(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password
        )
        db = GraphDatabaseConnector(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password,
            logger=logger
        )
        try:
            db.connect()
            graph = db.load_graph()
        finally:
            db.disconnect()
        logger.info(f"Graph loaded from PostgreSQL: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Load topic model (still from pickle)
        logger.info("\nLoading topic model...")
        topic_model_path = os.path.join(os.path.dirname(__file__), 'topic_model.pkl')
        with open(topic_model_path, 'rb') as f:
            topic_model = pickle.load(f)
        logger.info("Topic model loaded")
    else:
        graph_path = os.path.join(os.path.dirname(__file__), args.graph)
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        logger.info(f"Graph loaded from pickle: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

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

    # Initialize results manager
    logger.info("Initializing results manager...")
    results_manager = RetrievalResultsManager(logger=logger)

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
        display_results(
            claim=claim,
            results=results,
            alpha=retrieval_config.alpha,
            beta=retrieval_config.beta,
            gamma=retrieval_config.gamma,
            results_manager=results_manager,
            top_n=5,
            save_results=True
        )

    logger.info("\n" + "=" * 60)
    logger.info("RETRIEVAL TEST COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
