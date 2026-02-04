"""
Graph analysis and visualization module.

Contains components for analyzing and visualizing the knowledge graph:
- visualize_graph: Interactive graph visualization
- inspect_topics: Topic inspection utilities
- relabel_topics: LLM-based topic relabeling
- graph_manager: Graph management utilities
"""

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
    load_graph_from_source,
)
from fact_check_kg.analysis.relabel_topics import relabel_graph
from fact_check_kg.analysis.inspect_topics import inspect_topics

__all__ = [
    "list_available_graphs",
    "get_latest_graph_id",
    "load_graph_by_id",
    "display_available_graphs",
    "visualize_full_graph",
    "visualize_entity_subgraph",
    "analyze_graph_stats",
    "load_graph_from_source",
    "relabel_graph",
    "inspect_topics",
]
