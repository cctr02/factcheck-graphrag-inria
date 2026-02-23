"""
Knowledge Graph Fact-Checking CLI

Centralized command-line interface for all Knowledge Graph operations.

Usage:
    python cli.py build --help
    python cli.py list-graphs
    python cli.py stats --help
    python cli.py visualize --help
    python cli.py retrieve --help
    python cli.py relabel --help
    python cli.py config --help
    python cli.py init --help
"""

import click
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fact_check_kg.core.config_loader import get_build_config, get_retrieval_config, get_config
from fact_check_kg.core.logger import get_logger

logger = None

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
              help='Maximum number of documents to process (default: from config_build.yaml)')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--resume', is_flag=True,
              help='Resume from last checkpoint instead of starting fresh')
def build(max_docs, log_level, resume):
    """
    Build the knowledge graph from CSV data.

    Requires Ollama running for LLM-based topic labeling.

    Graph is saved to BOTH:
    - Pickle file: outputs/graphs/graph_{timestamp}.gpickle
    - PostgreSQL: tables graph_{timestamp}, nodes_{timestamp}

    All other parameters (chunk_size, chunk_overlap, etc.) are read from config_build.yaml.

    Use --resume to continue from the last checkpoint after an interruption.
    """
    import subprocess

    build_config = get_build_config()
    logger = get_logger("cli", build_config.logging.level)

    logger.info("=" * 60)
    logger.info("BUILDING KNOWLEDGE GRAPH")
    if resume:
        logger.info("  MODE: RESUME from checkpoint")
    logger.info("=" * 60)

    # Build command - only pass CLI overrides if explicitly provided
    cmd = [
        sys.executable,
        "fact_check_kg/build/build_graph.py",
        "--log-level", log_level,
    ]

    # Only pass --max-docs if explicitly provided via CLI (override)
    if max_docs is not None:
        cmd.extend(["--max-docs", str(max_docs)])

    if resume:
        cmd.append("--resume")

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
@click.option('--mode', type=click.Choice(['full', 'node', 'stats']),
              default='stats', help='Visualization mode')
@click.option('--node', type=str, default=None,
              help='Node ID for node mode (e.g., "entity::PERSON::trump", "chunk::0::5", "theme::2")')
@click.option('--max-nodes', type=int, default=None,
              help='Maximum nodes to display (default: from config_retrieval.yaml)')
@click.option('--output', type=str, default=None,
              help='Output HTML file (auto-generated if not specified)')
@click.option('--graph-id', type=str, default=None,
              help='Graph ID (timestamp) to use (e.g., 20251117_213022)')
@click.option('--use-postgres', is_flag=True,
              help='Load graph from PostgreSQL')
@click.option('--graph-file', type=str, default=None,
              help='Path to graph pickle file (if not using postgres)')
def visualize(mode, node, max_nodes, output, graph_id, use_postgres, graph_file):
    """
    Visualize the knowledge graph.

    Modes:
    - stats: Display pre-computed statistics (fast)
    - full: Full graph visualization (interactive HTML)
    - node: Node-centered subgraph visualization

    Examples:
    - uv run python cli.py visualize --mode node --node "entity::PERSON::trump" --graph-id 20251201_185321
    - uv run python cli.py visualize --mode node --node "chunk::0::5" --graph-id 20251201_185321
    - uv run python cli.py visualize --mode node --node "theme::2" --graph-id 20251201_185321

    Visualization files are saved as graph_{mode}_{graph_id}.html
    """
    import subprocess
    from fact_check_kg.analysis.visualize_graph import analyze_graph_stats
    from fact_check_kg.core.config_loader import DatabaseConfig

    # Load retrieval configuration and initialize logger
    config = get_retrieval_config()
    logger = get_logger("cli", config.logging.level)

    logger.info("=" * 60)
    logger.info(f"VISUALIZING GRAPH - MODE: {mode.upper()}")
    logger.info("=" * 60)

    # Determine graph file path based on graph_id
    if graph_id and not graph_file:
        from pathlib import Path
        graph_file = str(Path(config.paths.graphs_dir) / f"graph_{graph_id}.gpickle")
        logger.info(f"Using graph with ID: {graph_id}")
        logger.info(f"Graph file: {graph_file}")

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
    cmd = [sys.executable, "fact_check_kg/analysis/visualize_graph.py"]

    if graph_file:
        cmd.extend(["--graph", graph_file])

    # Add mode to command
    cmd.extend(["--mode", mode])

    if mode == 'node' and node:
        cmd.extend(["--entity", node])  # visualize_graph.py still uses --entity parameter
        if not output:
            # Use graph_id as timestamp for visualization
            vis_timestamp = graph_id if graph_id else datetime.now().strftime(config.export.timestamp_format)
            # Clean node ID for filename
            node_clean = node.replace('::', '_').replace(' ', '_')
            output = f"{config.paths.visualizations_dir}/graph_node_{node_clean}_{vis_timestamp}.html"
    elif mode == 'full':
        if not output:
            # Use graph_id as timestamp for visualization
            vis_timestamp = graph_id if graph_id else datetime.now().strftime(config.export.timestamp_format)
            output = f"{config.paths.visualizations_dir}/graph_full_{vis_timestamp}.html"
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
              help='Weight for semantic score (default: from config_retrieval.yaml)')
@click.option('--beta', type=float, default=None,
              help='Weight for thematic score (default: from config_retrieval.yaml)')
@click.option('--gamma', type=float, default=None,
              help='Weight for entity score (default: from config_retrieval.yaml)')
@click.option('--delta', type=float, default=None,
              help='Weight for topic distribution score (default: from config_retrieval.yaml)')
@click.option('--epsilon', type=float, default=None,
              help='Weight for claim-document score (default: from config_retrieval.yaml)')
@click.option('--zeta', type=float, default=None,
              help='Weight for claim-topic traversal score (default: from config_retrieval.yaml)')
@click.option('--eta', type=float, default=None,
              help='Weight for claim-keyphrase score (default: from config_retrieval.yaml)')
@click.option('--top-k', type=int, default=None,
              help='Number of results (default: from config_retrieval.yaml)')
@click.option('--graph-id', type=str, default=None,
              help='Graph ID (timestamp) to use (e.g., 20251117_213022). If not specified, uses latest or default.')
@click.option('--use-postgres', is_flag=True,
              help='Load graph from PostgreSQL')
@click.option('--top-n', type=int, default=5,
              help='Number of top results to display in detail (default: 5)')
@click.option('--save-results/--no-save-results', default=True,
              help='Save results to files (default: True)')
@click.option('--name', type=str, default=None,
              help='Custom name for this retrieval (e.g., "test1"). If not specified, uses _1, _2, etc.')
def retrieve(claim, alpha, beta, gamma, delta, epsilon, zeta, eta, top_k, graph_id, use_postgres, top_n, save_results, name):
    """
    Retrieve relevant chunks for a fact-checking claim with detailed scoring.

    Uses 7 retrieval strategies:
    - Semantic (alpha): Embedding similarity
    - Thematic (beta): Topic-based graph traversal
    - Entity (gamma): Entity-based graph traversal
    - Topic Distribution (delta): Jensen-Shannon divergence
    - Claim-Document (epsilon): Chunks from documents linked to similar claims
    - Claim-Topic (zeta): Claim-based topic distribution traversal
    - Claim-KeyPhrase (eta): Claim-based keyphrase similarity

    Weights must sum to 1.0.

    Results are saved as:
    - results_{graph_id}_1.json/txt (auto-incremented)
    - results_{graph_id}_{name}.json/txt (with --name flag)

    Result files include full text of all retrieved chunks at the end.
    """
    import pickle
    import spacy
    from sentence_transformers import SentenceTransformer
    from fact_check_kg.retrieval.retrieval import HybridRetriever
    from fact_check_kg.retrieval.retrieval_results import RetrievalResultsManager
    from fact_check_kg.core.db_connector import GraphDatabaseConnector

    # Load retrieval configuration
    retrieval_config = get_retrieval_config()

    # Load raw configs for saving to metadata
    from pathlib import Path
    from fact_check_kg.core.config_loader import load_raw_config
    base_config_path = Path(__file__).parent / "config" / "config.yaml"
    retrieval_config_path = Path(__file__).parent / "config" / "config_retrieval.yaml"
    raw_config = load_raw_config(base_config_path)
    raw_config_retrieval = load_raw_config(retrieval_config_path)

    # Initialize logger
    logger = get_logger("cli", retrieval_config.logging.level)

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH RETRIEVAL")
    logger.info("=" * 60)

    # Use config defaults
    alpha = alpha or retrieval_config.retrieval.alpha
    beta = beta or retrieval_config.retrieval.beta
    gamma = gamma or retrieval_config.retrieval.gamma
    delta = delta or retrieval_config.retrieval.delta
    epsilon = epsilon or retrieval_config.retrieval.epsilon
    zeta = zeta if zeta is not None else getattr(retrieval_config.retrieval, 'zeta', 0.0)
    eta = eta if eta is not None else getattr(retrieval_config.retrieval, 'eta', 0.0)
    top_k = top_k or retrieval_config.retrieval.top_k
    claim_top_k = retrieval_config.retrieval.claim_top_k
    keyphrase_top_n = getattr(retrieval_config.retrieval, 'keyphrase_top_n', 5)

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
                host=retrieval_config.database.host,
                port=retrieval_config.database.port,
                database=retrieval_config.database.database,
                user=retrieval_config.database.user,
                password=retrieval_config.database.password,
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

            # Load topic model with graph_id if specified (from models_dir)
            if graph_id:
                topic_model_path = Path(f"{retrieval_config.paths.models_dir}/topic_model_{graph_id}.pkl")
            else:
                topic_model_path = Path(f"{retrieval_config.paths.models_dir}/topic_model.pkl")

            with open(topic_model_path, 'rb') as f:
                topic_model = pickle.load(f)
        else:
            # Build list of possible paths based on graph_id
            if graph_id:
                # If graph_id is specified, look for that specific graph
                possible_paths = [
                    Path(f"{retrieval_config.paths.graphs_dir}/graph_{graph_id}.gpickle"),
                ]
            else:
                # Try default names
                possible_paths = [
                    Path(f"{retrieval_config.paths.graphs_dir}/knowledge_graph.gpickle"),  # Default name
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

            # Load topic model with matching graph_id if available (from models_dir)
            if graph_id:
                topic_model_paths = [
                    Path(f"{retrieval_config.paths.models_dir}/topic_model_{graph_id}.pkl"),
                ]
            else:
                topic_model_paths = [
                    Path(f"{retrieval_config.paths.models_dir}/topic_model.pkl"),  # Default naming
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
        embedding_model = SentenceTransformer(retrieval_config.models.embedding_model)
        nlp = spacy.load(retrieval_config.models.ner_model)
        logger.info("Models loaded")

        # Initialize retriever
        logger.info("Initializing retriever...")
        # Initialize retriever with entity_types from retrieval config
        retriever = HybridRetriever(
            graph=graph,
            topic_model=topic_model,
            embedding_model=embedding_model,
            nlp=nlp,
            entity_types=retrieval_config.retrieval.entity_types,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            eta=eta,
            top_k=top_k,
            theme_threshold=retrieval_config.retrieval.theme_threshold,
            entity_relevance_threshold=retrieval_config.retrieval.entity_relevance_threshold,
            theme_relevance_threshold=retrieval_config.retrieval.theme_relevance_threshold,
            claim_top_k=claim_top_k,
            keyphrase_top_n=keyphrase_top_n,
            graph_id=graph_id,
            cache_dir=retrieval_config.paths.embeddings_dir,
            models_dir=retrieval_config.paths.models_dir,
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

                # Extract matched_claims, detected_themes, entity_info, random_samples from first result
                matched_claims_s5 = results[0].get('matched_claims_s5') if results else None
                matched_claims_s6 = results[0].get('matched_claims_s6') if results else None
                matched_claims_s7 = results[0].get('matched_claims_s7') if results else None
                detected_themes = results[0].get('detected_themes') if results else None
                entity_info_s4 = results[0].get('entity_info_s4') if results else None
                random_samples = results[0].get('random_samples') if results else None

                json_path, text_path = results_manager.save_results(
                    claim_text, results, alpha, beta, gamma, delta, rankings,
                    graph_id=graph_id, custom_name=name,
                    config=raw_config, config_retrieval=raw_config_retrieval,
                    matched_claims_s5=matched_claims_s5,
                    matched_claims_s6=matched_claims_s6,
                    matched_claims_s7=matched_claims_s7,
                    epsilon=epsilon, zeta=zeta, eta=eta,
                    detected_themes=detected_themes,
                    extracted_entities=entity_info_s4,
                    random_samples=random_samples
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
    from fact_check_kg.analysis.graph_manager import display_available_graphs

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
    from fact_check_kg.analysis.visualize_graph import analyze_graph_stats
    from fact_check_kg.core.config_loader import DatabaseConfig
    from fact_check_kg.analysis.graph_manager import get_latest_graph_id, load_graph_by_id
    import pickle

    config = get_retrieval_config()
    logger = get_logger("cli", config.logging.level)

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
              help='LLM model to use (default: from config_build.yaml)')
def relabel(use_postgres, llm_model):
    """Relabel topics using LLM."""
    from fact_check_kg.analysis.relabel_topics import relabel_graph

    config = get_build_config()
    logger = get_logger("cli", config.logging.level)

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
    config = get_config()
    logger = get_logger("cli_config", "INFO")

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
    config = get_config()
    logger = get_logger("cli_init", "INFO")

    logger.info("=" * 60)
    logger.info("INITIALIZING SYSTEM")
    logger.info("=" * 60)

    if dirs:
        # Create directories
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
        from fact_check_kg.core.db_connector import GraphDatabaseConnector
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
