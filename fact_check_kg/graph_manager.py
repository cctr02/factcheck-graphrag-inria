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

from fact_check_kg.config_loader import get_config


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

    # Load based on storage type
    if metadata["storage_type"] == "postgresql":
        # Load from PostgreSQL
        from fact_check_kg.db_connector import GraphDatabaseConnector

        db = GraphDatabaseConnector(
            host=metadata["database_host"],
            port=config.database.port,
            database=metadata["database_name"],
            user=config.database.user,
            password=config.database.password
        )

        try:
            db.connect()
            graph = db.load_graph()
        finally:
            db.disconnect()

        # Load topic model
        topic_model_path = Path(metadata["topic_model_path"])
        with open(topic_model_path, 'rb') as f:
            topic_model = pickle.load(f)

    else:  # pickle
        # Load graph from pickle
        graph_path = Path(metadata["graph_path"])
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)

        # Load topic model
        topic_model_path = Path(metadata["topic_model_path"])
        with open(topic_model_path, 'rb') as f:
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

        if graph['storage_type'] == 'postgresql':
            print(f"   Database: {graph['database_name']}")
        else:
            print(f"   File: {Path(graph['graph_path']).name}")

    print("\n" + "=" * 80)
    print(f"\nTotal: {len(graphs)} graph(s)")
    print("=" * 80)
