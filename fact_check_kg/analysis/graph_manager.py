"""
Graph Manager - Utilities for managing multiple knowledge graphs.

This module provides functions to list, load, and manage multiple
knowledge graphs stored in the system.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from fact_check_kg.core.config_loader import get_config


def list_available_graphs() -> List[Dict]:
    """
    List all available graphs in the system.

    Returns:
        List of dictionaries containing graph metadata
    """
    config = get_config()
    graphs_dir = Path(config.paths.graphs_dir)

    if not graphs_dir.exists():
        return []

    graphs = []

    # Find all metadata files
    for metadata_file in sorted(graphs_dir.glob("metadata_*.json"), reverse=True):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            graphs.append(metadata)
        except Exception as e:
            logging.warning(f"Could not read metadata file {metadata_file}: {e}")

    return graphs


def get_latest_graph_id() -> Optional[str]:
    """
    Get the ID (timestamp) of the most recently created graph.

    Returns:
        Timestamp string of the latest graph, or None if no graphs exist
    """
    graphs = list_available_graphs()
    if not graphs:
        return None
    return graphs[0]["timestamp"]


def load_graph_by_id(graph_id: str) -> Tuple[object, object, Dict]:
    """
    Load a specific graph by its ID (timestamp).

    Args:
        graph_id: Timestamp string identifying the graph

    Returns:
        Tuple of (graph, topic_model, metadata)

    Raises:
        FileNotFoundError: If graph with given ID doesn't exist
    """
    config = get_config()
    graphs_dir = Path(config.paths.graphs_dir)

    # Load metadata
    metadata_path = graphs_dir / f"metadata_{graph_id}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No graph found with ID: {graph_id}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Resolve graph file path: metadata stores paths under metadata["paths"]["graph"]
    # (top-level "graph_path" / "pickle_path" keys no longer exist as of current build)
    def _resolve_graph_path(meta: dict) -> Optional[str]:
        """Return the pickle file path from metadata, regardless of schema version."""
        # Current schema: paths.graph
        if isinstance(meta.get("paths"), dict):
            p = meta["paths"].get("graph")
            if p:
                return p
        # Legacy top-level keys
        return meta.get("graph_path") or meta.get("pickle_path")

    def _resolve_topic_model_path(meta: dict) -> Optional[str]:
        """Return the topic model path from metadata."""
        if isinstance(meta.get("paths"), dict):
            p = meta["paths"].get("topic_model")
            if p:
                return p
        return meta.get("topic_model_path")

    storage_type = metadata["storage_type"]

    if storage_type == "dual":
        # Dual storage: prefer pickle for faster loading, fallback to PostgreSQL
        try:
            graph_file = _resolve_graph_path(metadata)
            if not graph_file:
                raise FileNotFoundError("Graph path not found in metadata")
            with open(Path(graph_file), 'rb') as f:
                graph = pickle.load(f)
        except Exception as e:
            logging.warning(f"Could not load from pickle, trying PostgreSQL: {e}")
            from fact_check_kg.core.db_connector import GraphDatabaseConnector
            db = GraphDatabaseConnector(
                host=metadata["postgres_host"],
                port=config.database.port,
                database=metadata["postgres_database"],
                user=config.database.user,
                password=config.database.password
            )
            try:
                db.connect()
                graph = db.load_graph()
            finally:
                db.disconnect()

    elif storage_type == "postgresql":
        from fact_check_kg.core.db_connector import GraphDatabaseConnector
        db = GraphDatabaseConnector(
            host=metadata.get("database_host", metadata.get("postgres_host")),
            port=config.database.port,
            database=metadata.get("database_name", metadata.get("postgres_database")),
            user=config.database.user,
            password=config.database.password
        )
        try:
            db.connect()
            graph = db.load_graph()
        finally:
            db.disconnect()

    else:  # pickle
        graph_file = _resolve_graph_path(metadata)
        if not graph_file:
            raise FileNotFoundError(f"Cannot resolve graph file path from metadata for ID: {graph_id}")
        with open(Path(graph_file), 'rb') as f:
            graph = pickle.load(f)

    # Load topic model (same for all storage types)
    topic_model_file = _resolve_topic_model_path(metadata)
    if not topic_model_file:
        raise FileNotFoundError(f"Cannot resolve topic model path from metadata for ID: {graph_id}")
    with open(Path(topic_model_file), 'rb') as f:
        topic_model = pickle.load(f)

    return graph, topic_model, metadata


def display_available_graphs():
    """Display a formatted list of available graphs."""
    graphs = list_available_graphs()

    if not graphs:
        print("No graphs found in the system.")
        return

    print("\n" + "=" * 80)
    print("AVAILABLE GRAPHS")
    print("=" * 80)

    for i, graph in enumerate(graphs, 1):
        print(f"\n{i}. Graph ID: {graph['timestamp']}")
        print(f"   Created: {graph['created_at']}")
        print(f"   Storage: {graph['storage_type']}")
        print(f"   Documents: {graph['num_documents']}, Chunks: {graph['num_chunks']}")
        print(f"   Nodes: {graph['num_nodes']}, Edges: {graph['num_edges']}")

        def _graph_filename(meta: dict) -> str:
            """Extract the pickle filename from metadata."""
            paths = meta.get("paths", {})
            p = paths.get("graph") if isinstance(paths, dict) else None
            if not p:
                p = meta.get("graph_path") or meta.get("pickle_path")
            return Path(p).name if p else "unknown"

        if graph['storage_type'] == 'dual':
            print(f"   Pickle: {_graph_filename(graph)}")
            print(f"   PostgreSQL: {graph.get('postgres_database', 'N/A')}")
        elif graph['storage_type'] == 'postgresql':
            print(f"   Database: {graph.get('database_name', graph.get('postgres_database', 'N/A'))}")
        else:
            print(f"   File: {_graph_filename(graph)}")

    print("\n" + "=" * 80)
    print(f"\nTotal: {len(graphs)} graph(s)")
    print("=" * 80)
