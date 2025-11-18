#!/usr/bin/env python3
"""
Knowledge Graph Fact-Checking CLI

Centralized command-line interface for all Knowledge Graph operations.

Usage:
    python cli.py build --help
    python cli.py visualize --help
    python cli.py retrieve --help
    python cli.py export --help
"""

import click
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fact_check_kg.config_loader import get_config
from fact_check_kg.logger import get_logger


# Initialize configuration
config = get_config()
logger = get_logger("cli", config.logging.level)


@click.group()
@click.version_option(version="1.0.0", prog_name="Knowledge Graph Fact-Checking CLI")
def cli():
    """
    Knowledge Graph Fact-Checking System CLI

    A comprehensive tool for building, visualizing, and querying knowledge graphs
    for fact-checking purposes.
    """
    pass


# ============================================================================
# BUILD COMMAND
# ============================================================================

@cli.command()
@click.option('--max-docs', type=int, default=None,
              help=f'Maximum number of documents to process (default: {config.graph_construction.max_documents})')
@click.option('--chunk-size', type=int, default=None,
              help=f'Size of text chunks (default: {config.graph_construction.chunk_size})')
@click.option('--chunk-overlap', type=int, default=None,
              help=f'Overlap between chunks (default: {config.graph_construction.chunk_overlap})')
@click.option('--use-postgres', is_flag=True,
              help='Save graph to PostgreSQL instead of pickle')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
def build(max_docs, chunk_size, chunk_overlap, use_postgres, log_level):
    """Build the knowledge graph from CSV data."""
    import subprocess

    logger.info("=" * 60)
    logger.info("BUILDING KNOWLEDGE GRAPH")
    logger.info("=" * 60)

    # Use config defaults if not specified
    max_docs = max_docs or config.graph_construction.max_documents
    chunk_size = chunk_size or config.graph_construction.chunk_size
    chunk_overlap = chunk_overlap or config.graph_construction.chunk_overlap

    # Build command
    cmd = [
        sys.executable,
        "fact_check_kg/build_graph.py",
        "--csv", config.paths.csv_file,
        "--max-docs", str(max_docs),
        "--chunk-size", str(chunk_size),
        "--chunk-overlap", str(chunk_overlap),
        "--log-level", log_level
    ]

    if use_postgres:
        cmd.extend([
            "--use-postgres",
            "--db-host", config.database.host,
            "--db-port", str(config.database.port),
            "--db-name", config.database.database,
            "--db-user", config.database.user,
            "--db-password", config.database.password
        ])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("Graph building completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building graph: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error building graph: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# VISUALIZE COMMAND
# ============================================================================

@cli.command()
@click.option('--mode', type=click.Choice(['full', 'entity', 'hierarchical', 'stats']),
              default='stats', help='Visualization mode')
@click.option('--entity', type=str, default=None,
              help='Entity name for entity/hierarchical mode')
@click.option('--max-nodes', type=int, default=None,
              help=f'Maximum nodes to display (default: {config.visualization.max_nodes_full_view})')
@click.option('--output', type=str, default=None,
              help='Output HTML file (auto-generated if not specified)')
@click.option('--use-postgres', is_flag=True,
              help='Load graph from PostgreSQL')
@click.option('--graph-file', type=str, default=None,
              help='Path to graph pickle file (if not using postgres)')
def visualize(mode, entity, max_nodes, output, use_postgres, graph_file):
    """Visualize the knowledge graph."""
    import subprocess
    from fact_check_kg.visualize_graph import analyze_graph_stats
    from fact_check_kg.config import DatabaseConfig

    logger.info("=" * 60)
    logger.info(f"VISUALIZING GRAPH - MODE: {mode.upper()}")
    logger.info("=" * 60)

    # Build command based on mode
    if mode == 'stats':
        # Call analyze_graph_stats directly with proper parameters
        try:
            db_config = None
            if use_postgres:
                db_config = DatabaseConfig(
                    host=config.database.host,
                    port=config.database.port,
                    database=config.database.database,
                    user=config.database.user,
                    password=config.database.password
                )

            # Use provided graph_file or default
            if not graph_file and not use_postgres:
                graph_file = "fact_check_kg/knowledge_graph.gpickle"

            analyze_graph_stats(
                graph_path=graph_file,
                use_postgres=use_postgres,
                db_config=db_config
            )
            return
        except Exception as e:
            logger.error(f"Error showing stats: {e}", exc_info=True)
            sys.exit(1)

    # For other modes, call the main() function with arguments
    cmd = [sys.executable, "fact_check_kg/visualize_graph.py"]

    if graph_file:
        cmd.extend(["--graph", graph_file])

    if mode == 'entity' and entity:
        cmd.extend(["--entity", entity])
        if not output:
            timestamp = datetime.now().strftime(config.export.timestamp_format)
            output = f"{config.paths.visualizations_dir}/graph_entity_{entity}_{timestamp}.html"
    elif mode == 'hierarchical' and entity:
        # Call hierarchical visualization
        cmd = [sys.executable, "fact_check_kg/visualize_hierarchical.py", "--entity", entity]
        if not output:
            timestamp = datetime.now().strftime(config.export.timestamp_format)
            output = f"{config.paths.visualizations_dir}/graph_hierarchical_{entity}_{timestamp}.html"
    elif mode == 'full':
        if not output:
            timestamp = datetime.now().strftime(config.export.timestamp_format)
            output = f"{config.paths.visualizations_dir}/graph_full_{timestamp}.html"
        if max_nodes:
            cmd.extend(["--max-nodes", str(max_nodes)])

    if output:
        cmd.extend(["--output", output])

    if use_postgres:
        cmd.extend(["--use-postgres"])

    try:
        subprocess.run(cmd, check=True)
        if output:
            logger.info(f"Visualization saved to: {output}")
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# RETRIEVE COMMAND
# ============================================================================

@cli.command()
@click.argument('claim', type=str, required=False)
@click.option('--alpha', type=float, default=None,
              help=f'Weight for semantic score (default: {config.retrieval.alpha})')
@click.option('--beta', type=float, default=None,
              help=f'Weight for thematic score (default: {config.retrieval.beta})')
@click.option('--gamma', type=float, default=None,
              help=f'Weight for entity score (default: {config.retrieval.gamma})')
@click.option('--delta', type=float, default=None,
              help=f'Weight for topic distribution score (default: {config.retrieval.delta})')
@click.option('--top-k', type=int, default=None,
              help=f'Number of results (default: {config.retrieval.top_k})')
@click.option('--graph-id', type=str, default=None,
              help='Graph ID (timestamp) to use (e.g., 20251117_213022). If not specified, uses latest or default.')
@click.option('--use-postgres', is_flag=True,
              help='Load graph from PostgreSQL')
@click.option('--top-n', type=int, default=5,
              help='Number of top results to display in detail (default: 5)')
@click.option('--save-results/--no-save-results', default=True,
              help='Save results to files (default: True)')
def retrieve(claim, alpha, beta, gamma, delta, top_k, graph_id, use_postgres, top_n, save_results):
    """Retrieve relevant chunks for a fact-checking claim with detailed scoring."""
    import pickle
    import spacy
    from sentence_transformers import SentenceTransformer
    from fact_check_kg.retrieval import HybridRetriever
    from fact_check_kg.retrieval_results import RetrievalResultsManager
    from fact_check_kg.db_connector import GraphDatabaseConnector

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH RETRIEVAL")
    logger.info("=" * 60)

    # Use config defaults
    alpha = alpha or config.retrieval.alpha
    beta = beta or config.retrieval.beta
    gamma = gamma or config.retrieval.gamma
    delta = delta or config.retrieval.delta
    top_k = top_k or config.retrieval.top_k

    # Get claim
    if not claim:
        logger.error("Please provide a claim")
        sys.exit(1)

    claims = [claim]

    try:
        # Load graph
        logger.info("Loading knowledge graph...")
        if use_postgres:
            db = GraphDatabaseConnector(
                host=config.database.host,
                port=config.database.port,
                database=config.database.database,
                user=config.database.user,
                password=config.database.password,
                graph_id=graph_id,  # Use specified graph_id
                logger=logger
            )
            try:
                db.connect()
                graph = db.load_graph()
            finally:
                db.disconnect()

            graph_id_str = f" (graph_id: {graph_id})" if graph_id else ""
            logger.info(f"Graph loaded from PostgreSQL{graph_id_str}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # Load topic model with graph_id if specified
            if graph_id:
                topic_model_path = Path(f"{config.paths.graphs_dir}/topic_model_{graph_id}.pkl")
            else:
                topic_model_path = Path(f"{config.paths.graphs_dir}/topic_model.pkl")

            with open(topic_model_path, 'rb') as f:
                topic_model = pickle.load(f)
        else:
            # Build list of possible paths based on graph_id
            if graph_id:
                # If graph_id is specified, look for that specific graph
                possible_paths = [
                    Path(f"{config.paths.graphs_dir}/graph_{graph_id}.gpickle"),
                ]
            else:
                # Try default names
                possible_paths = [
                    Path(f"{config.paths.graphs_dir}/knowledge_graph.gpickle"),  # Default name
                    Path("fact_check_kg/knowledge_graph.gpickle"),  # Legacy location
                ]

            graph_path = None
            for path in possible_paths:
                if path.exists():
                    graph_path = path
                    break

            if not graph_path:
                if graph_id:
                    logger.error(f"Graph file not found for graph_id: {graph_id}")
                else:
                    logger.error("No graph file found. Please run 'build' command first or specify --graph-id.")
                sys.exit(1)

            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
            logger.info(f"Graph loaded from {graph_path}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # Load topic model with matching graph_id if available
            if graph_id:
                topic_model_paths = [
                    Path(f"{config.paths.graphs_dir}/topic_model_{graph_id}.pkl"),
                ]
            else:
                topic_model_paths = [
                    graph_path.parent / "topic_model.pkl",  # Default naming
                    graph_path.parent / "knowledge_graph_topic_model.pkl",  # Legacy naming
                ]

            topic_model_path = None
            for path in topic_model_paths:
                if path.exists():
                    topic_model_path = path
                    break

            if not topic_model_path:
                logger.error(f"No topic model found near {graph_path}")
                sys.exit(1)

            with open(topic_model_path, 'rb') as f:
                topic_model = pickle.load(f)

        # Load models
        logger.info("Loading embedding and NER models...")
        embedding_model = SentenceTransformer(config.models.embedding_model)
        nlp = spacy.load(config.models.ner_model)
        logger.info("Models loaded")

        # Initialize retriever
        logger.info("Initializing retriever...")
        retriever = HybridRetriever(
            graph=graph,
            topic_model=topic_model,
            embedding_model=embedding_model,
            nlp=nlp,
            entity_types=config.graph_construction.entity_types,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            top_k=top_k,
            theme_threshold=config.retrieval.theme_threshold,
            entity_relevance_threshold=config.retrieval.entity_relevance_threshold,
            theme_relevance_threshold=config.retrieval.theme_relevance_threshold,
            logger=logger
        )

        # Initialize results manager
        logger.info("Initializing results manager...")
        results_manager = RetrievalResultsManager(logger=logger)

        # Process each claim
        for claim_text in claims:
            results = retriever.retrieve(claim_text)

            # Display results
            results_manager.display_results_console(claim_text, results, top_n=top_n)

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
                    claim_text, results, alpha, beta, gamma, rankings
                )
                print(f"Results saved to:")
                print(f"  JSON: {json_path}")
                print(f"  Text: {text_path}")

            print("\n" + "=" * 80)

        logger.info("Retrieval completed successfully!")

    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# EXPORT COMMAND
# ============================================================================
# Note: Export functionality is currently not implemented.
# The graph can be exported through NetworkX methods directly in Python code.


# ============================================================================
# LIST-GRAPHS COMMAND
# ============================================================================

@cli.command('list-graphs')
def list_graphs():
    """List all available knowledge graphs in the system."""
    from fact_check_kg.graph_manager import display_available_graphs

    display_available_graphs()


# ============================================================================
# STATS COMMAND
# ============================================================================

@cli.command()
@click.option('--graph-id', type=str, default=None,
              help='Graph ID (timestamp) to analyze. Use "latest" for most recent graph or omit to use latest.')
@click.option('--use-postgres', is_flag=True,
              help='Load graph from PostgreSQL (deprecated - use --graph-id)')
@click.option('--detailed', is_flag=True,
              help='Show detailed statistics')
def stats(graph_id, use_postgres, detailed):
    """Display statistics about the knowledge graph."""
    from fact_check_kg.visualize_graph import analyze_graph_stats
    from fact_check_kg.config import DatabaseConfig
    from fact_check_kg.graph_manager import get_latest_graph_id, load_graph_by_id
    import pickle

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH STATISTICS")
    logger.info("=" * 60)

    try:
        # Determine which graph to use
        if not graph_id or graph_id == "latest":
            graph_id = get_latest_graph_id()
            if not graph_id:
                logger.error("No graphs found in the system. Please build a graph first with 'build' command.")
                sys.exit(1)
            logger.info(f"Using latest graph: {graph_id}")

        # Load graph by ID
        graph, topic_model, metadata = load_graph_by_id(graph_id)

        logger.info(f"Loaded graph: {metadata['timestamp']}")
        logger.info(f"Created: {metadata['created_at']}")
        logger.info(f"Storage: {metadata['storage_type']}")

        # Display stats (reuse visualize_graph analyze function logic inline)
        print("\n" + "=" * 60)
        print("GRAPH STATISTICS")
        print("=" * 60)

        print(f"\nTotal nodes: {graph.number_of_nodes()}")
        print(f"Total edges: {graph.number_of_edges()}")

        # Nodes by type
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print("\nNodes by type:")
        for node_type, count in sorted(node_types.items()):
            print(f"  {node_type}: {count}")

        # Edges by type
        edge_types = {}
        for _, _, data in graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        print("\nEdges by type:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"  {edge_type}: {count}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# RELABEL COMMAND
# ============================================================================

@cli.command()
@click.option('--use-postgres', is_flag=True,
              help='Use PostgreSQL storage')
@click.option('--llm-model', type=str, default=None,
              help=f'LLM model to use (default: {config.models.llm_model})')
def relabel(use_postgres, llm_model):
    """Relabel topics using LLM."""
    from fact_check_kg.relabel_topics import relabel_graph

    logger.info("=" * 60)
    logger.info("RELABELING TOPICS WITH LLM")
    logger.info("=" * 60)

    llm_model = llm_model or config.models.llm_model

    try:
        relabel_graph(
            use_postgres=use_postgres,
            llm_model=llm_model
        )
        logger.info("Topics relabeled successfully!")
    except Exception as e:
        logger.error(f"Error relabeling topics: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# CONFIG COMMAND
# ============================================================================

@cli.command()
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--validate', is_flag=True, help='Validate configuration file')
def config_cmd(show, validate):
    """Manage configuration."""
    if show:
        import yaml
        logger.info("=" * 60)
        logger.info("CURRENT CONFIGURATION")
        logger.info("=" * 60)
        print(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))

    if validate:
        logger.info("=" * 60)
        logger.info("VALIDATING CONFIGURATION")
        logger.info("=" * 60)
        try:
            # Attempt to load config
            test_config = get_config()
            logger.info("[OK] Configuration is valid!")
            logger.info(f"  - Paths: OK")
            logger.info(f"  - Graph construction: OK")
            logger.info(f"  - Models: OK")
            logger.info(f"  - Database: OK")
            logger.info(f"  - Retrieval: OK")
            logger.info(f"  - Visualization: OK")
        except Exception as e:
            logger.error(f"[FAIL] Configuration is invalid: {e}")
            sys.exit(1)


# ============================================================================
# INIT COMMAND
# ============================================================================

@cli.command()
@click.option('--postgres', is_flag=True, help='Initialize PostgreSQL database')
@click.option('--dirs', is_flag=True, help='Create directory structure')
def init(postgres, dirs):
    """Initialize the system (create directories, database, etc.)."""
    logger.info("=" * 60)
    logger.info("INITIALIZING SYSTEM")
    logger.info("=" * 60)

    if dirs:
        # Create directories
        from pathlib import Path
        directories = [
            config.paths.logs_dir,
            config.paths.outputs_dir,
            config.paths.graphs_dir,
            config.paths.visualizations_dir,
            config.paths.models_dir,
            config.paths.exports_dir,
            "outputs/statistics",
            "outputs/retrieval_results"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"[OK] Created directory: {directory}")

    if postgres:
        from fact_check_kg.db_connector import GraphDatabaseConnector
        db = GraphDatabaseConnector(
            host=config.database.host,
            port=config.database.port,
            database=config.database.database,
            user=config.database.user,
            password=config.database.password
        )
        try:
            db.connect()
            db.create_tables()
            logger.info("[OK] PostgreSQL database initialized")
            db.disconnect()
        except Exception as e:
            logger.error(f"[FAIL] Failed to initialize database: {e}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("INITIALIZATION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    cli()
