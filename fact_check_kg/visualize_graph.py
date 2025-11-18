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
import re
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fact_check_kg.db_connector import GraphDatabaseConnector
from fact_check_kg.config import DatabaseConfig


def extract_timestamp_from_path(file_path: str) -> str:
    """
    Extract timestamp from file path.

    Args:
        file_path: Path to graph file

    Returns:
        Timestamp string (e.g., "20251117_213022") or empty string if not found
    """
    if not file_path:
        return ""

    # Look for timestamp pattern: YYYYMMDD_HHMMSS
    match = re.search(r'(\d{8}_\d{6})', file_path)
    if match:
        return match.group(1)
    return ""


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
    output_path: str = None,
    max_nodes: int = 200,
    use_postgres: bool = False,
    db_config: DatabaseConfig = None
):
    """
    Visualize a filtered version of the full graph.

    Args:
        graph_path: Path to .gpickle file (if not using postgres)
        output_path: Output HTML file (auto-generated with timestamp if not provided)
        max_nodes: Maximum nodes to display (for readability)
        use_postgres: Whether to load from PostgreSQL
        db_config: Database configuration
    """
    G = load_graph_from_source(graph_path, use_postgres, db_config)

    # Auto-generate output path with timestamp if not provided
    if output_path is None:
        timestamp = extract_timestamp_from_path(graph_path) if graph_path else ""
        if timestamp:
            output_path = f"outputs/visualizations/graph_full_{timestamp}.html"
        else:
            output_path = "outputs/visualizations/graph_full.html"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    # Create pyvis network with better font settings
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)

    # Show controls first before setting custom options
    net.show_buttons(filter_=['physics'])

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
            title=f"{node_type}: {node}<br>Degree: {degree}",
            font={'size': 16, 'face': 'arial'}
        )

    # Add edges with colors by type
    edge_color_map = {
        'CONTAINS': '#666666',
        'HAS_TOPIC': '#ab47bc',
        'MENTIONS': '#66bb6a',
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
            label=f"{weight:.2f}",  # Display weight on edge
            color=edge_color_map.get(edge_type, 'gray'),
            width=min(weight * 2, 5)
        )

    # Add legend
    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; background-color: rgba(34, 34, 34, 0.9);
                padding: 15px; border-radius: 8px; border: 2px solid #666; font-family: Arial;
                color: white; z-index: 1000; max-width: 250px;">
        <h3 style="margin-top: 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 5px;">Legend</h3>

        <h4 style="margin: 10px 0 5px 0; color: #aaa;">Node Types:</h4>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #0277bd;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Document</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #fbc02d;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Chunk</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #7b1fa2;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Theme</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #388e3c;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Entity</span>
        </div>

        <h4 style="margin: 15px 0 5px 0; color: #aaa;">Edge Types:</h4>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #666666;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>CONTAINS</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #ab47bc;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>HAS_TOPIC</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #66bb6a;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>MENTIONS</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #ef5350;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>CO_OCCURS</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #42a5f5;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>SEMANTIC_SIM</span>
        </div>
    </div>
    """

    # Save
    net.save_graph(output_path)

    # Inject legend into HTML
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Insert legend just before </body>
    html_content = html_content.replace('</body>', f'{legend_html}</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\nVisualization saved to: {output_path}")
    print(f"Open in browser to explore!")


def visualize_entity_subgraph(
    entity_name: str,
    output_path: str = None,
    max_neighbors: int = 50,
    graph_path: str = None,
    use_postgres: bool = False,
    db_config: DatabaseConfig = None
):
    """
    Visualize subgraph around a specific entity.

    Args:
        entity_name: Entity to center on (e.g., "gaza", "facebook")
        output_path: Output HTML file (auto-generated with timestamp if not provided)
        max_neighbors: Max neighbors to include
        graph_path: Path to .gpickle file (if not using postgres)
        use_postgres: Whether to load from PostgreSQL
        db_config: Database configuration
    """
    G = load_graph_from_source(graph_path, use_postgres, db_config)

    # Auto-generate output path with timestamp if not provided
    if output_path is None:
        timestamp = extract_timestamp_from_path(graph_path) if graph_path else ""
        if timestamp:
            output_path = f"outputs/visualizations/graph_entity_{entity_name}_{timestamp}.html"
        else:
            output_path = f"outputs/visualizations/graph_entity_{entity_name}.html"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    # Create network with better font settings
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()

    # Show controls first
    net.show_buttons(filter_=['physics'])

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

        # Don't truncate labels for subgraphs - show full text
        if node_type == 'entity':
            label = node_data.get('entity_text', node.split('::')[-1])
        elif node_type == 'theme':
            label = node_data.get('label', f"theme_{node.split('::')[-1]}")
        elif node_type == 'chunk':
            # For chunks, show a slightly shorter preview but not too truncated
            chunk_id = node.split('::')[-1]
            label = f"Chunk {chunk_id[:10]}"
        else:
            label = node.split('::')[-1]

        # Highlight central entity
        if node == entity_node:
            size = 50
            color = '#ff0000'  # Red for central node
        else:
            size = 25
            color = color_map.get(node_type, 'gray')

        net.add_node(
            node,
            label=label,
            color=color,
            size=size,
            title=f"{node_type}: {node}",
            font={'size': 20, 'face': 'arial'}  # Larger font for subgraph
        )

    # Add edges
    edge_color_map = {
        'CONTAINS': '#666666',
        'HAS_TOPIC': '#ab47bc',
        'MENTIONS': '#66bb6a',
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
            label=f"{weight:.2f}",  # Display weight on edge
            color=edge_color_map.get(edge_type, 'gray'),
            width=min(weight * 2, 5)
        )

    # Add legend
    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; background-color: rgba(34, 34, 34, 0.9);
                padding: 15px; border-radius: 8px; border: 2px solid #666; font-family: Arial;
                color: white; z-index: 1000; max-width: 250px;">
        <h3 style="margin-top: 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 5px;">Legend</h3>

        <h4 style="margin: 10px 0 5px 0; color: #aaa;">Node Types:</h4>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #0277bd;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Document</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #fbc02d;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Chunk</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #7b1fa2;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Theme</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #388e3c;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Entity</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #ff0000;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Central Entity</span>
        </div>

        <h4 style="margin: 15px 0 5px 0; color: #aaa;">Edge Types:</h4>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #666666;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>CONTAINS</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #ab47bc;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>HAS_TOPIC</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #66bb6a;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>MENTIONS</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #ef5350;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>CO_OCCURS</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #42a5f5;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>SEMANTIC_SIM</span>
        </div>
    </div>
    """

    # Save
    net.save_graph(output_path)

    # Inject legend into HTML
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Insert legend just before </body>
    html_content = html_content.replace('</body>', f'{legend_html}</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

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
        # Pass None if output not specified - will auto-generate with timestamp
        visualize_full_graph(args.graph, args.output, args.max_nodes, args.use_postgres, db_config)

    elif args.mode == "entity":
        # Pass None if output not specified - will auto-generate with timestamp
        visualize_entity_subgraph(args.entity, args.output, 50, args.graph, args.use_postgres, db_config)


if __name__ == "__main__":
    main()
