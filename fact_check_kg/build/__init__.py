"""
Graph building module.

Contains components for building the knowledge graph from CSV data:
- data_loader: Load and chunk documents
- graph_builder: Build the multi-level knowledge graph
- build_graph: Main entry point for graph construction
"""

from fact_check_kg.build.data_loader import load_and_chunk_documents, load_claims, chunk_text
from fact_check_kg.build.graph_builder import KnowledgeGraphBuilder

__all__ = [
    "load_and_chunk_documents",
    "load_claims",
    "chunk_text",
    "KnowledgeGraphBuilder",
]
