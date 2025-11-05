"""
Hierarchical graph visualization.

Creates a clearer, hierarchical visualization of the knowledge graph
with proper layout and grouping by node types.
"""

import pickle
import networkx as nx
from pyvis.network import Network
import argparse


def visualize_hierarchical_entity(
    graph_path: str,
    entity_name: str,
    output_path: str = "graph_hierarchical.html"
):
    """
    Create a hierarchical visualization centered on an entity.

    Layout:
    - Central entity at the center
    - Chunks in one circle around it
    - Themes and related entities in outer circles
    - Documents at the periphery

    Args:
        graph_path: Path to .gpickle file
        entity_name: Entity to center on
        output_path: Output HTML file
    """
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    # Find entity
    entity_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get('type') == 'entity' and entity_name.lower() in d.get('entity_text', '').lower()
    ]

    if not entity_nodes:
        print(f"Error: Entity '{entity_name}' not found")
        return

    entity_node = entity_nodes[0]
    print(f"Found entity: {entity_node}")

    # Build hierarchical subgraph
    nodes_by_level = {
        'center': [entity_node],
        'chunks': [],
        'themes': [],
        'related_entities': [],
        'documents': []
    }

    # Level 1: Direct neighbors (chunks, themes, co-occurring entities)
    for neighbor in G.neighbors(entity_node):
        neighbor_type = G.nodes[neighbor].get('type')
        edge_type = G[entity_node][neighbor].get('edge_type')

        if neighbor_type == 'chunk':
            nodes_by_level['chunks'].append(neighbor)
        elif neighbor_type == 'theme':
            nodes_by_level['themes'].append(neighbor)
        elif neighbor_type == 'entity' and edge_type in ['CO_OCCURS', 'SEMANTIC_SIM']:
            nodes_by_level['related_entities'].append(neighbor)

    # Level 2: Documents from chunks
    for chunk in nodes_by_level['chunks'][:20]:  # Limit for clarity
        for doc in G.neighbors(chunk):
            if G.nodes[doc].get('type') == 'document':
                if doc not in nodes_by_level['documents']:
                    nodes_by_level['documents'].append(doc)

    # Limit for readability
    nodes_by_level['chunks'] = nodes_by_level['chunks'][:15]
    nodes_by_level['related_entities'] = nodes_by_level['related_entities'][:10]
    nodes_by_level['documents'] = nodes_by_level['documents'][:5]

    # Build subgraph
    all_nodes = (
        nodes_by_level['center'] +
        nodes_by_level['chunks'] +
        nodes_by_level['themes'] +
        nodes_by_level['related_entities'] +
        nodes_by_level['documents']
    )
    G_viz = G.subgraph(all_nodes)

    print(f"\nHierarchical structure:")
    print(f"  Center: 1 entity")
    print(f"  Chunks: {len(nodes_by_level['chunks'])}")
    print(f"  Themes: {len(nodes_by_level['themes'])}")
    print(f"  Related entities: {len(nodes_by_level['related_entities'])}")
    print(f"  Documents: {len(nodes_by_level['documents'])}")
    print(f"  Total: {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges")

    # Create network with hierarchical layout
    net = Network(
        height="1000px",
        width="100%",
        bgcolor="#1a1a1a",
        font_color="white",
        directed=False
    )

    # Use hierarchical layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "hierarchicalRepulsion": {
          "centralGravity": 0.2,
          "springLength": 150,
          "springConstant": 0.01,
          "nodeDistance": 200,
          "damping": 0.09
        },
        "solver": "hierarchicalRepulsion"
      },
      "layout": {
        "hierarchical": {
          "enabled": false,
          "sortMethod": "directed"
        }
      }
    }
    """)

    # Color scheme by type
    colors = {
        'center': '#ff1744',      # Red - Central entity
        'chunk': '#ffd54f',        # Yellow - Chunks
        'theme': '#ab47bc',        # Purple - Themes
        'entity': '#66bb6a',       # Green - Related entities
        'document': '#42a5f5'      # Blue - Documents
    }

    # Add nodes with hierarchical positioning
    for node in G_viz.nodes():
        node_data = G_viz.nodes[node]
        node_type = node_data.get('type', 'unknown')

        # Determine level and color
        if node == entity_node:
            level = 'center'
            size = 50
            label = node_data.get('entity_text', entity_name).upper()
            shape = 'star'
        elif node in nodes_by_level['chunks']:
            level = 'chunk'
            size = 25
            text = node_data.get('text', '')[:50]
            label = f"Chunk {node.split('::')[-1]}"
            shape = 'box'
        elif node in nodes_by_level['themes']:
            level = 'theme'
            size = 35
            label = node_data.get('label', f"Theme {node.split('::')[-1]}")
            shape = 'diamond'
        elif node in nodes_by_level['related_entities']:
            level = 'entity'
            size = 20
            label = node_data.get('entity_text', node.split('::')[-1])
            shape = 'dot'
        else:  # documents
            level = 'document'
            size = 15
            label = f"Doc {node.split('::')[-1]}"
            shape = 'square'

        # Create tooltip
        if node_type == 'chunk':
            text_preview = node_data.get('text', '')[:200]
            title = f"<b>Chunk {node.split('::')[-1]}</b><br>{text_preview}..."
        elif node_type == 'theme':
            keywords = node_data.get('keywords', [])[:5]
            title = f"<b>{label}</b><br>Keywords: {', '.join(keywords)}"
        elif node_type == 'entity':
            ent_type = node.split('::')[1]
            title = f"<b>{ent_type}</b>: {label}<br>Degree: {G.degree(node)}"
        else:
            title = f"<b>{node_type}</b>: {label}"

        net.add_node(
            node,
            label=label,
            color=colors.get(level, 'gray'),
            size=size,
            title=title,
            shape=shape,
            font={'size': 14 if level == 'center' else 10}
        )

    # Add edges with colors and labels
    edge_colors = {
        'CONTAINS': '#616161',
        'HAS_TOPIC': '#ab47bc',
        'MENTIONS': '#66bb6a',
        'RELATED_TO': '#ff9800',
        'CO_OCCURS': '#ef5350',
        'SEMANTIC_SIM': '#42a5f5'
    }

    for u, v, data in G_viz.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        weight = data.get('weight', 1)

        # Edge width based on weight
        if edge_type == 'SEMANTIC_SIM':
            width = max(0.5, min(weight * 3, 3))
        elif edge_type == 'CO_OCCURS':
            width = max(1, min(weight * 0.5, 4))
        else:
            width = 2

        # Tooltip
        title = f"<b>{edge_type}</b>"
        if weight != 1:
            title += f"<br>Weight: {weight:.2f}"

        net.add_edge(
            u, v,
            color=edge_colors.get(edge_type, '#666666'),
            width=width,
            title=title,
            smooth={'enabled': True, 'type': 'continuous'}
        )

    # Add legend
    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.8);
                padding: 15px; border-radius: 8px; color: white; font-family: Arial; z-index: 1000;">
        <h3 style="margin-top: 0;">Legend</h3>
        <div><span style="color: #ff1744;">★</span> Central Entity</div>
        <div><span style="color: #ffd54f;">■</span> Chunks</div>
        <div><span style="color: #ab47bc;">◆</span> Themes</div>
        <div><span style="color: #66bb6a;">●</span> Related Entities</div>
        <div><span style="color: #42a5f5;">□</span> Documents</div>
        <hr style="border-color: #444;">
        <div style="font-size: 11px;">
            <div><span style="color: #66bb6a;">━</span> MENTIONS</div>
            <div><span style="color: #ab47bc;">━</span> HAS_TOPIC</div>
            <div><span style="color: #ef5350;">━</span> CO_OCCURS</div>
            <div><span style="color: #42a5f5;">━</span> SEMANTIC_SIM</div>
            <div><span style="color: #ff9800;">━</span> RELATED_TO</div>
        </div>
    </div>
    """

    # Save with legend
    net.save_graph(output_path)

    # Add legend to HTML
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Insert legend before </body>
    html_content = html_content.replace('</body>', f'{legend_html}</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nHierarchical visualization saved to: {output_path}")
    print(f"Open in browser to explore!")


def visualize_full_hierarchical(
    graph_path: str,
    output_path: str = "graph_full_hierarchical.html"
):
    """
    Create hierarchical view of entire graph structure.

    Layout by levels:
    - Documents (top)
    - Chunks (level 2)
    - Themes and Entities (level 3)

    Args:
        graph_path: Path to .gpickle file
        output_path: Output HTML file
    """
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # For full graph, sample for readability
    # Take top documents, their chunks, and connected entities/themes
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'document'][:10]

    selected_nodes = set(doc_nodes)
    for doc in doc_nodes:
        # Get chunks
        chunks = [n for n in G.neighbors(doc) if G.nodes[n].get('type') == 'chunk'][:5]
        selected_nodes.update(chunks)

        # Get entities and themes from chunks
        for chunk in chunks:
            entities = [n for n in G.neighbors(chunk) if G.nodes[n].get('type') == 'entity'][:3]
            themes = [n for n in G.neighbors(chunk) if G.nodes[n].get('type') == 'theme']
            selected_nodes.update(entities)
            selected_nodes.update(themes)

    G_viz = G.subgraph(list(selected_nodes))
    print(f"Visualization: {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges")

    # Create network
    net = Network(height="1000px", width="100%", bgcolor="#1a1a1a", font_color="white")

    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "levelSeparation": 200,
          "nodeSpacing": 150
        }
      },
      "physics": {
        "enabled": false
      }
    }
    """)

    # Colors
    colors = {
        'document': '#42a5f5',
        'chunk': '#ffd54f',
        'theme': '#ab47bc',
        'entity': '#66bb6a'
    }

    # Add nodes
    for node in G_viz.nodes():
        node_data = G_viz.nodes[node]
        node_type = node_data.get('type', 'unknown')

        if node_type == 'document':
            label = f"Doc {node.split('::')[-1]}"
            size = 20
            level = 0
        elif node_type == 'chunk':
            label = f"Chunk {node.split('::')[-1]}"
            size = 15
            level = 1
        elif node_type == 'theme':
            label = node_data.get('label', '')[:20]
            size = 25
            level = 2
        else:  # entity
            label = node_data.get('entity_text', '')[:15]
            size = 12
            level = 2

        net.add_node(
            node,
            label=label,
            color=colors.get(node_type, 'gray'),
            size=size,
            level=level,
            title=f"{node_type}: {node}"
        )

    # Add edges
    for u, v, data in G_viz.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        net.add_edge(u, v, title=edge_type)

    net.save_graph(output_path)
    print(f"\nHierarchical visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Graph Visualization")
    parser.add_argument(
        "--graph",
        type=str,
        default="knowledge_graph.gpickle",
        help="Path to graph file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["entity", "full"],
        default="entity",
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
        help="Output HTML file"
    )

    args = parser.parse_args()

    if args.mode == "entity":
        output = args.output or f"graph_hierarchical_{args.entity}.html"
        visualize_hierarchical_entity(args.graph, args.entity, output)

    elif args.mode == "full":
        output = args.output or "graph_full_hierarchical.html"
        visualize_full_hierarchical(args.graph, output)


if __name__ == "__main__":
    main()
