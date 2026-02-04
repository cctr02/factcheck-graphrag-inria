"""
Fact-Checking Knowledge Graph Module

A multi-level knowledge graph for fact-checking that combines:
- Strategy 1: Semantic Similarity (embedding-based)
- Strategy 2: Topic Distribution Similarity (Jensen-Shannon divergence)
- Strategy 3: Thematic Graph Traversal (BFS over theme graph)
- Strategy 4: Entity Graph Traversal (BFS over entity graph)

Module structure:
- core: Configuration, database connector, embedding cache, logging
- build: Graph building components (data loading, graph construction)
- analysis: Graph analysis and visualization
- retrieval: Hybrid retrieval system
"""

__version__ = "0.1.0"

# Core components
from fact_check_kg.core.config_loader import (
    get_config,
    get_build_config,
    get_retrieval_config,
    reload_config,
)
from fact_check_kg.core.db_connector import GraphDatabaseConnector
from fact_check_kg.core.embedding_cache import EmbeddingCache
from fact_check_kg.core.logger import get_logger

# Build components
from fact_check_kg.build.graph_builder import KnowledgeGraphBuilder
from fact_check_kg.build.data_loader import load_and_chunk_documents, load_claims

# Analysis components
from fact_check_kg.analysis.graph_manager import (
    list_available_graphs,
    get_latest_graph_id,
    load_graph_by_id,
    display_available_graphs,
)
from fact_check_kg.analysis.visualize_graph import (
    visualize_full_graph,
    visualize_entity_subgraph,
    analyze_graph_stats,
)

# Retrieval components
from fact_check_kg.retrieval.retrieval import HybridRetriever
from fact_check_kg.retrieval.retrieval_results import RetrievalResultsManager

__all__ = [
    # Version
    "__version__",
    # Core
    "get_config",
    "get_build_config",
    "get_retrieval_config",
    "reload_config",
    "GraphDatabaseConnector",
    "EmbeddingCache",
    "get_logger",
    # Build
    "KnowledgeGraphBuilder",
    "load_and_chunk_documents",
    "load_claims",
    # Analysis
    "list_available_graphs",
    "get_latest_graph_id",
    "load_graph_by_id",
    "display_available_graphs",
    "visualize_full_graph",
    "visualize_entity_subgraph",
    "analyze_graph_stats",
    # Retrieval
    "HybridRetriever",
    "RetrievalResultsManager",
]
