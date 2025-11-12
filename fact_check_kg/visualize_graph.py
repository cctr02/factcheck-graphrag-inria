"""
Graph visualization script.

Provides multiple visualization options for the knowledge graph:
1. Full graph overview (filtered for readability)
2. Subgraph around a specific entity
3. Retrieval path visualization
"""

import pickle
import networkx as nx
from pyvis.network import Network
import argparse
from collections import defaultdict
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fact_check_kg.db_connector import GraphDatabaseConnector
from fact_check_kg.config import DatabaseConfig


def load_graph_from_source(graph_path: str = None, use_postgres: bool = False, db_config: DatabaseConfig = None) -> nx.Graph:
    """
    Load graph either from PostgreSQL or pickle file.

    Args:
        graph_path: Path to .gpickle file (used if use_postgres=False)
        use_postgres: Whether to load from PostgreSQL
        db_config: Database configuration (used if use_postgres=True)

    Returns:
        NetworkX graph
    """
    if use_postgres:
        if db_config is None:
            db_config = DatabaseConfig()

        print(f"Loading graph from PostgreSQL ({db_config.database})...")
        db = GraphDatabaseConnector(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password
        )

        try:
            db.connect()
            G = db.load_graph()
            return G
        finally:
            db.disconnect()
    else:
        print(f"Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        return G


def visualize_full_graph(
    graph_path: str = None,
    output_path: str = "graph_full.html",
    max_nodes: int = 200,
    use_postgres: bool = False,
    db_config: DatabaseConfig = None
):
    """
    Visualize a filtered version of the full graph.

    Args:
        graph_path: Path to .gpickle file (if not using postgres)
        output_path: Output HTML file
        max_nodes: Maximum nodes to display (for readability)
        use_postgres: Whether to load from PostgreSQL
        db_config: Database configuration
    """
    G = load_graph_from_source(graph_path, use_postgres, db_config)

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Filter: only keep high-degree nodes for overview
    if G.number_of_nodes() > max_nodes:
        print(f"Filtering to top {max_nodes} nodes by degree...")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        subgraph_nodes = [node for node, deg in top_nodes]
        G_viz = G.subgraph(subgraph_nodes)
    else:
        G_viz = G

    print(f"Visualizing {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges...")

    # Create pyvis network
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)

    # Color mapping by type
    color_map = {
        'document': '#0277bd',  # Blue
        'chunk': '#fbc02d',     # Yellow
        'theme': '#7b1fa2',     # Purple
        'entity': '#388e3c'     # Green
    }

    # Add nodes
    for node in G_viz.nodes():
        node_data = G_viz.nodes[node]
        node_type = node_data.get('type', 'unknown')

        # Create label
        if node_type == 'entity':
            label = node_data.get('entity_text', node.split('::')[-1])
        elif node_type == 'theme':
            label = node_data.get('label', f"theme_{node.split('::')[-1]}")
        else:
            label = node.split('::')[-1][:20]

        # Size by degree
        degree = G_viz.degree(node)
        size = 10 + min(degree * 2, 40)

        net.add_node(
            node,
            label=label,
            color=color_map.get(node_type, 'gray'),
            size=size,
            title=f"{node_type}: {node}<br>Degree: {degree}"
        )

    # Add edges with colors by type
    edge_color_map = {
        'CONTAINS': '#666666',
        'HAS_TOPIC': '#ab47bc',
        'MENTIONS': '#66bb6a',
        'RELATED_TO': '#ffa726',
        'CO_OCCURS': '#ef5350',
        'SEMANTIC_SIM': '#42a5f5'
    }

    for edge in G_viz.edges(data=True):
        edge_type = edge[2].get('edge_type', 'unknown')
        weight = edge[2].get('weight', 1)

        net.add_edge(
            edge[0],
            edge[1],
            title=f"{edge_type} (weight: {weight:.2f})",
            color=edge_color_map.get(edge_type, 'gray'),
            width=min(weight * 2, 5)
        )

    # Show controls
    net.show_buttons(filter_=['physics'])

    # Save
    net.save_graph(output_path)
    print(f"\nVisualization saved to: {output_path}")
    print(f"Open in browser to explore!")


def visualize_entity_subgraph(
    entity_name: str,
    output_path: str = "graph_entity.html",
    max_neighbors: int = 50,
    graph_path: str = None,
    use_postgres: bool = False,
    db_config: DatabaseConfig = None
):
    """
    Visualize subgraph around a specific entity.

    Args:
        entity_name: Entity to center on (e.g., "gaza", "facebook")
        output_path: Output HTML file
        max_neighbors: Max neighbors to include
        graph_path: Path to .gpickle file (if not using postgres)
        use_postgres: Whether to load from PostgreSQL
        db_config: Database configuration
    """
    G = load_graph_from_source(graph_path, use_postgres, db_config)

    # Find entity node
    entity_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get('type') == 'entity' and entity_name.lower() in d.get('entity_text', '').lower()
    ]

    if not entity_nodes:
        print(f"Error: Entity '{entity_name}' not found in graph")
        return

    entity_node = entity_nodes[0]
    print(f"Found entity: {entity_node}")

    # Get neighbors (2-hop)
    neighbors_1hop = list(G.neighbors(entity_node))[:max_neighbors]
    neighbors_2hop = []
    for n in neighbors_1hop[:10]:  # Only expand first 10
        neighbors_2hop.extend(list(G.neighbors(n))[:5])

    subgraph_nodes = [entity_node] + neighbors_1hop + neighbors_2hop
    subgraph_nodes = list(set(subgraph_nodes))  # Unique

    G_viz = G.subgraph(subgraph_nodes)
    print(f"Subgraph: {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges")

    # Create network
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()

    # Color mapping
    color_map = {
        'document': '#0277bd',
        'chunk': '#fbc02d',
        'theme': '#7b1fa2',
        'entity': '#388e3c'
    }

    # Add nodes
    for node in G_viz.nodes():
        node_data = G_viz.nodes[node]
        node_type = node_data.get('type', 'unknown')

        if node_type == 'entity':
            label = node_data.get('entity_text', node.split('::')[-1])
        elif node_type == 'theme':
            label = node_data.get('label', f"theme_{node.split('::')[-1]}")
        else:
            label = node.split('::')[-1][:20]

        # Highlight central entity
        if node == entity_node:
            size = 40
            color = '#ff0000'  # Red for central node
        else:
            size = 20
            color = color_map.get(node_type, 'gray')

        net.add_node(
            node,
            label=label,
            color=color,
            size=size,
            title=f"{node_type}: {node}"
        )

    # Add edges
    edge_color_map = {
        'CONTAINS': '#666666',
        'HAS_TOPIC': '#ab47bc',
        'MENTIONS': '#66bb6a',
        'RELATED_TO': '#ffa726',
        'CO_OCCURS': '#ef5350',
        'SEMANTIC_SIM': '#42a5f5'
    }

    for edge in G_viz.edges(data=True):
        edge_type = edge[2].get('edge_type', 'unknown')
        weight = edge[2].get('weight', 1)

        net.add_edge(
            edge[0],
            edge[1],
            title=f"{edge_type} (weight: {weight:.2f})",
            color=edge_color_map.get(edge_type, 'gray')
        )

    net.show_buttons(filter_=['physics'])
    net.save_graph(output_path)
    print(f"\nVisualization saved to: {output_path}")


def analyze_graph_stats(graph_path: str = None, use_postgres: bool = False, db_config: DatabaseConfig = None):
    """
    Print detailed graph statistics.

    Args:
        graph_path: Path to .gpickle file (if not using postgres)
        use_postgres: Whether to load from PostgreSQL
        db_config: Database configuration
    """
    G = load_graph_from_source(graph_path, use_postgres, db_config)

    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)

    print(f"\nTotal nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    # Nodes by type
    node_types = defaultdict(int)
    for node, data in G.nodes(data=True):
        node_types[data.get('type', 'unknown')] += 1

    print("\nNodes by type:")
    for ntype, count in sorted(node_types.items()):
        print(f"  {ntype}: {count}")

    # Edges by type
    edge_types = defaultdict(int)
    for u, v, data in G.edges(data=True):
        edge_types[data.get('edge_type', 'unknown')] += 1

    print("\nEdges by type:")
    for etype, count in sorted(edge_types.items()):
        print(f"  {etype}: {count}")

    # Top entities by degree
    print("\nTop 15 entities by degree:")
    entities = [
        (n, G.degree(n), G.nodes[n].get('entity_text', ''))
        for n, d in G.nodes(data=True)
        if d.get('type') == 'entity'
    ]
    entities.sort(key=lambda x: x[1], reverse=True)

    for i, (node, degree, text) in enumerate(entities[:15], 1):
        entity_type = node.split('::')[1]
        print(f"  {i}. {entity_type}::{text} - {degree} connections")

    # Themes info
    print("\nThemes:")
    themes = [
        (n, G.nodes[n].get('label', ''), G.degree(n))
        for n, d in G.nodes(data=True)
        if d.get('type') == 'theme'
    ]
    for node, label, degree in themes:
        print(f"  {node}: '{label}' - {degree} connections")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize Knowledge Graph")
    parser.add_argument(
        "--graph",
        type=str,
        default="knowledge_graph.gpickle",
        help="Path to graph file (ignored if --use-postgres is set)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "entity", "stats"],
        default="stats",
        help="Visualization mode"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="gaza",
        help="Entity name for entity mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output HTML file (auto-generated if not specified)"
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200,
        help="Max nodes for full mode"
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

    # Create database config if using postgres
    db_config = None
    if args.use_postgres:
        db_config = DatabaseConfig(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password
        )

    if args.mode == "stats":
        analyze_graph_stats(args.graph, args.use_postgres, db_config)

    elif args.mode == "full":
        output = args.output or "graph_full.html"
        visualize_full_graph(args.graph, output, args.max_nodes, args.use_postgres, db_config)

    elif args.mode == "entity":
        output = args.output or f"graph_entity_{args.entity}.html"
        visualize_entity_subgraph(args.entity, output, 50, args.graph, args.use_postgres, db_config)


if __name__ == "__main__":
    main()
