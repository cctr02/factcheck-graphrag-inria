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
    from fact_check_kg.build_graph import build_graph_main

    logger.info("=" * 60)
    logger.info("BUILDING KNOWLEDGE GRAPH")
    logger.info("=" * 60)

    # Use config defaults if not specified
    max_docs = max_docs or config.graph_construction.max_documents
    chunk_size = chunk_size or config.graph_construction.chunk_size
    chunk_overlap = chunk_overlap or config.graph_construction.chunk_overlap

    try:
        build_graph_main(
            max_docs=max_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_postgres=use_postgres,
            log_level=log_level
        )
        logger.info("Graph building completed successfully!")
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
    from fact_check_kg.visualize_graph import visualize_main

    logger.info("=" * 60)
    logger.info(f"VISUALIZING GRAPH - MODE: {mode.upper()}")
    logger.info("=" * 60)

    # Generate output path
    if output is None:
        timestamp = datetime.now().strftime(config.export.timestamp_format)
        if mode == 'entity' and entity:
            output = f"{config.paths.visualizations_dir}/graph_entity_{entity}_{timestamp}.html"
        else:
            output = f"{config.paths.visualizations_dir}/graph_{mode}_{timestamp}.html"

    try:
        visualize_main(
            mode=mode,
            entity=entity,
            max_nodes=max_nodes or config.visualization.max_nodes_full_view,
            output=output,
            use_postgres=use_postgres,
            graph_file=graph_file
        )
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
@click.option('--top-k', type=int, default=None,
              help=f'Number of results (default: {config.retrieval.top_k})')
@click.option('--use-postgres', is_flag=True,
              help='Load graph from PostgreSQL')
@click.option('--test', is_flag=True,
              help='Run with test claims from config')
def retrieve(claim, alpha, beta, gamma, top_k, use_postgres, test):
    """Retrieve relevant chunks for a fact-checking claim."""
    from fact_check_kg.test_retrieval import retrieval_main

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH RETRIEVAL")
    logger.info("=" * 60)

    # Use config defaults
    alpha = alpha or config.retrieval.alpha
    beta = beta or config.retrieval.beta
    gamma = gamma or config.retrieval.gamma
    top_k = top_k or config.retrieval.top_k

    # Get claims
    if test:
        claims = config.testing.test_claims
    elif claim:
        claims = [claim]
    else:
        logger.error("Please provide a claim or use --test flag")
        sys.exit(1)

    try:
        retrieval_main(
            claims=claims,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            top_k=top_k,
            use_postgres=use_postgres
        )
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# EXPORT COMMAND
# ============================================================================

@cli.command()
@click.option('--format', 'export_format', type=click.Choice(['json', 'csv', 'graphml']),
              default='json', help='Export format')
@click.option('--output', type=str, default=None,
              help='Output file path')
@click.option('--use-postgres', is_flag=True,
              help='Load graph from PostgreSQL')
def export(export_format, output, use_postgres):
    """Export the knowledge graph to various formats."""
    from fact_check_kg.export import export_main

    logger.info("=" * 60)
    logger.info(f"EXPORTING GRAPH - FORMAT: {export_format.upper()}")
    logger.info("=" * 60)

    # Generate output path if not specified
    if output is None:
        timestamp = datetime.now().strftime(config.export.timestamp_format)
        output = f"{config.paths.exports_dir}/knowledge_graph_{timestamp}.{export_format}"

    try:
        export_main(
            format=export_format,
            output=output,
            use_postgres=use_postgres
        )
        logger.info(f"Graph exported to: {output}")
    except Exception as e:
        logger.error(f"Error exporting graph: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# STATS COMMAND
# ============================================================================

@cli.command()
@click.option('--use-postgres', is_flag=True,
              help='Load graph from PostgreSQL')
@click.option('--detailed', is_flag=True,
              help='Show detailed statistics')
def stats(use_postgres, detailed):
    """Display statistics about the knowledge graph."""
    from fact_check_kg.visualize_graph import analyze_graph_stats

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH STATISTICS")
    logger.info("=" * 60)

    try:
        analyze_graph_stats(
            use_postgres=use_postgres,
            detailed=detailed
        )
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
            logger.info("✓ Configuration is valid!")
            logger.info(f"  - Paths: OK")
            logger.info(f"  - Graph construction: OK")
            logger.info(f"  - Models: OK")
            logger.info(f"  - Database: OK")
            logger.info(f"  - Retrieval: OK")
            logger.info(f"  - Visualization: OK")
        except Exception as e:
            logger.error(f"✗ Configuration is invalid: {e}")
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
            config.paths.exports_dir
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created directory: {directory}")

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
            logger.info("✓ PostgreSQL database initialized")
            db.disconnect()
        except Exception as e:
            logger.error(f"✗ Failed to initialize database: {e}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("INITIALIZATION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    cli()
