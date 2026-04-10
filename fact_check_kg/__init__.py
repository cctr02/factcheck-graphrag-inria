"""
Fact-Checking Knowledge Graph Package

A multi-level knowledge graph for fact-checking that retrieves relevant
documents given a textual claim. The graph combines documents, text chunks,
BERTopic themes, spaCy named entities, KeyBERT keyphrases, and claim nodes.

Retrieval uses seven complementary strategies fused by weighted sum:
    S1 (alpha)   - Semantic Similarity: FAISS embedding cosine similarity
    S2 (beta)    - Topic Distribution: Jensen-Shannon divergence
    S3 (gamma)   - Thematic Graph Traversal: BFS over theme graph
    S4 (delta)   - Entity Graph Traversal: BFS over entity graph
    S5 (epsilon) - Claim-Document: chunks from documents linked to similar claims
    S6 (zeta)    - Claim-Topic Traversal: claim-based topic distribution BFS
    S7 (eta)     - Claim-KeyPhrase: claim-based keyphrase matching

Package structure:
    core/       - Configuration (config_loader), database connector, FAISS
                  embedding cache, and logging utilities
    build/      - Graph construction: data loading, chunking, BERTopic,
                  NER, KeyBERT, LLM topic labeling, checkpoint management
    retrieval/  - HybridRetriever (7 strategies) and result formatting
    analysis/   - Graph management, Pyvis visualization, topic relabeling

Entry point: all operations are exposed through ``cli.py``.
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
