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
import requests
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fact_check_kg.db_connector import GraphDatabaseConnector
from fact_check_kg.config import DatabaseConfig


def summarize_chunk_with_ollama(chunk_text: str, model: str = "llama3.1:latest", max_length: int = 30) -> str:
    """
    Summarize chunk text using Ollama LLM.

    Args:
        chunk_text: The chunk text to summarize
        model: Ollama model to use
        max_length: Maximum words in summary

    Returns:
        Summary string (or truncated text if Ollama fails)
    """
    try:
        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"Summarize the following text in {max_length} words or less. Be concise and capture the main point:\n\n{chunk_text}\n\nSummary:",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": max_length * 2  # Allow some flexibility
                }
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            summary = result.get('response', '').strip()
            # Remove common prefixes
            for prefix in ["Summary:", "Here's a summary:", "Here is a summary:"]:
                if summary.startswith(prefix):
                    summary = summary[len(prefix):].strip()
            return summary if summary else chunk_text[:50] + "..."
        else:
            return chunk_text[:50] + "..."
    except Exception as e:
        # Fallback to simple truncation if Ollama is not available
        return chunk_text[:50] + "..."


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


def generate_legend_html(chunk_full_texts: dict = None) -> str:
    """
    Generate interactive legend HTML with optional chunk popup functionality.

    Args:
        chunk_full_texts: Dictionary mapping chunk node IDs to full text (None for entity view)

    Returns:
        HTML string with legend and optional chunk popup
    """
    # Chunk modal (only if chunk_full_texts provided)
    chunk_modal = ""
    chunk_texts_js = "{}"
    chunk_click_handler = ""

    if chunk_full_texts:
        chunk_modal = """
    <!-- Popup modal for chunk full text -->
    <div id="chunkModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                                 background-color: rgba(0,0,0,0.8); z-index: 2000; overflow: auto;">
        <div style="background-color: #2a2a2a; margin: 5% auto; padding: 30px; width: 70%; max-width: 800px;
                    border-radius: 10px; border: 2px solid #666; color: white; font-family: Arial; position: relative;">
            <span onclick="closeChunkModal()" style="position: absolute; top: 15px; right: 20px; font-size: 32px;
                                                     font-weight: bold; cursor: pointer; color: #aaa;">&times;</span>
            <h2 style="margin-top: 0; color: #fbc02d; border-bottom: 2px solid #666; padding-bottom: 10px;">Chunk Content</h2>
            <div id="chunkModalContent" style="max-height: 500px; overflow-y: auto; line-height: 1.6; white-space: pre-wrap;">
            </div>
        </div>
    </div>
        """
        chunk_texts_js = json.dumps(chunk_full_texts)
        chunk_click_handler = """
                // Add click event listener for chunks
                network.on("click", function(params) {
                    if (params.nodes.length > 0) {
                        var nodeId = params.nodes[0];
                        var nodeData = nodeMetadata[nodeId];

                        // If it's a chunk node, show the full text
                        if (nodeData && nodeData.type === 'chunk' && chunkFullTexts[nodeId]) {
                            showChunkModal(chunkFullTexts[nodeId]);
                        }
                    }
                });
        """

    chunk_functions = """
        function showChunkModal(fullText) {
            var modal = document.getElementById('chunkModal');
            var content = document.getElementById('chunkModalContent');
            content.textContent = fullText;
            modal.style.display = 'block';
        }

        function closeChunkModal() {
            document.getElementById('chunkModal').style.display = 'none';
        }

        // Close modal when clicking outside of it
        window.onclick = function(event) {
            var modal = document.getElementById('chunkModal');
            if (event.target === modal) {
                closeChunkModal();
            }
        }
    """ if chunk_full_texts else ""

    legend_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; background-color: rgba(34, 34, 34, 0.9);
                padding: 15px; border-radius: 8px; border: 2px solid #666; font-family: Arial;
                color: white; z-index: 1000; max-width: 250px;">
        <h3 style="margin-top: 0; color: #fff; border-bottom: 2px solid #666; padding-bottom: 5px;">Legend (Click to filter)</h3>

        <h4 style="margin: 10px 0 5px 0; color: #aaa;">Node Types:</h4>
        <div id="filter-document" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleNodeType('document')" data-active="true">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #0277bd;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Document</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-chunk" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleNodeType('chunk')" data-active="true">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #fbc02d;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Chunk{' (click to view)' if chunk_full_texts else ''}</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-theme" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleNodeType('theme')" data-active="true">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #7b1fa2;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Theme</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-entity" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleNodeType('entity')" data-active="true">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #388e3c;
                        border-radius: 50%; margin-right: 8px;"></span>
            <span>Entity</span>
            <span style="float: right;">✓</span>
        </div>

        <h4 style="margin: 15px 0 5px 0; color: #aaa;">Edge Types:</h4>
        <div id="filter-CONTAINS" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleEdgeType('CONTAINS')" data-active="true">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #666666;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>CONTAINS</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-HAS_TOPIC" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleEdgeType('HAS_TOPIC')" data-active="true">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #ab47bc;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>HAS_TOPIC</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-MENTIONS" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleEdgeType('MENTIONS')" data-active="true">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #66bb6a;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>MENTIONS</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-CO_OCCURS" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleEdgeType('CO_OCCURS')" data-active="true">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #ef5350;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>CO_OCCURS</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-SEMANTIC_SIM" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleEdgeType('SEMANTIC_SIM')" data-active="true">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #42a5f5;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>SEMANTIC_SIM</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-CO_PRESENCE" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleEdgeType('CO_PRESENCE')" data-active="true">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #9c27b0;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>CO_PRESENCE</span>
            <span style="float: right;">✓</span>
        </div>
        <div id="filter-TEMPORAL_DIST" style="margin: 5px 0; cursor: pointer; padding: 3px; border-radius: 3px;"
             onclick="toggleEdgeType('TEMPORAL_DIST')" data-active="true">
            <span style="display: inline-block; width: 25px; height: 3px; background-color: #ff9800;
                        margin-right: 8px; vertical-align: middle;"></span>
            <span>TEMPORAL_DIST</span>
            <span style="float: right;">✓</span>
        </div>
    </div>

    {chunk_modal}

    <script type="text/javascript">
        // Store node and edge metadata
        var nodeMetadata = {{}};
        var edgeMetadata = {{}};
        var chunkFullTexts = {chunk_texts_js};

        // Parse nodes and edges from network
        setTimeout(function() {{
            if (typeof network !== 'undefined') {{
                var nodes = network.body.data.nodes;
                var edges = network.body.data.edges;

                // Parse nodes
                nodes.forEach(function(node) {{
                    // Extract type from title (format: "type: node_id")
                    var title = node.title || '';
                    var typeMatch = title.match(/^(\\w+):/);
                    if (typeMatch) {{
                        nodeMetadata[node.id] = {{type: typeMatch[1], originalHidden: node.hidden || false}};
                    }}
                }});

                // Parse edges
                edges.forEach(function(edge) {{
                    // Extract edge type from title (format: "edge_type (weight: X)")
                    var title = edge.title || '';
                    var typeMatch = title.match(/^([A-Z_]+)/);
                    if (typeMatch) {{
                        edgeMetadata[edge.id] = {{
                            type: typeMatch[1],
                            from: edge.from,
                            to: edge.to
                        }};
                    }}
                }});

                {chunk_click_handler}
            }}
        }}, 500);

        {chunk_functions}

        function toggleNodeType(nodeType) {{
            if (typeof network === 'undefined') {{
                console.error('Network not loaded yet');
                return;
            }}

            var filterElement = document.getElementById('filter-' + nodeType);
            var isActive = filterElement.getAttribute('data-active') === 'true';

            // Toggle state
            var newState = !isActive;
            filterElement.setAttribute('data-active', newState);

            // Update visual feedback
            if (newState) {{
                filterElement.style.backgroundColor = '';
                filterElement.querySelector('span:last-child').textContent = '✓';
            }} else {{
                filterElement.style.backgroundColor = 'rgba(255, 0, 0, 0.2)';
                filterElement.querySelector('span:last-child').textContent = '✗';
            }}

            // Update network
            var nodes = network.body.data.nodes;
            var nodesToUpdate = [];

            for (var nodeId in nodeMetadata) {{
                if (nodeMetadata[nodeId].type === nodeType) {{
                    nodesToUpdate.push({{
                        id: nodeId,
                        hidden: !newState
                    }});
                }}
            }}

            if (nodesToUpdate.length > 0) {{
                nodes.update(nodesToUpdate);
            }}
        }}

        function toggleEdgeType(edgeType) {{
            if (typeof network === 'undefined') {{
                console.error('Network not loaded yet');
                return;
            }}

            var filterElement = document.getElementById('filter-' + edgeType);
            var isActive = filterElement.getAttribute('data-active') === 'true';

            // Toggle state
            var newState = !isActive;
            filterElement.setAttribute('data-active', newState);

            // Update visual feedback
            if (newState) {{
                filterElement.style.backgroundColor = '';
                filterElement.querySelector('span:last-child').textContent = '✓';
            }} else {{
                filterElement.style.backgroundColor = 'rgba(255, 0, 0, 0.2)';
                filterElement.querySelector('span:last-child').textContent = '✗';
            }}

            // Update network
            var edges = network.body.data.edges;
            var edgesToUpdate = [];

            for (var edgeId in edgeMetadata) {{
                if (edgeMetadata[edgeId].type === edgeType) {{
                    edgesToUpdate.push({{
                        id: edgeId,
                        hidden: !newState
                    }});
                }}
            }}

            if (edgesToUpdate.length > 0) {{
                edges.update(edgesToUpdate);
            }}
        }}
    </script>
    """
    return legend_html


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
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"Graph file not found: {graph_path}\n"
                f"Available graphs can be listed with: uv run python cli.py list-graphs"
            )
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

    # Store chunk full texts for hover display
    print("Collecting chunk texts...")
    chunk_full_texts = {}
    chunk_nodes = [n for n, d in G_viz.nodes(data=True) if d.get('type') == 'chunk']

    for node in chunk_nodes:
        node_data = G_viz.nodes[node]
        chunk_text = node_data.get('text', '')
        if chunk_text:
            chunk_full_texts[node] = chunk_text
    print(f"Collected texts for {len(chunk_full_texts)} chunks")

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
        elif node_type == 'chunk':
            # Use chunk ID as label
            label = node
        else:
            label = node.split('::')[-1][:20]

        # Size by degree (increased from 10+min(deg*2,40) to 30+min(deg*3,80))
        degree = G_viz.degree(node)
        size = 30 + min(degree * 3, 80)

        # Prepare title (tooltip) - include full text for chunks
        if node_type == 'chunk':
            full_text = chunk_full_texts.get(node, '')
            # Escape HTML special characters and limit length for tooltip
            full_text_preview = full_text[:200].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            title = f"{node_type}: {node}<br>Degree: {degree}<br><br>Preview: {full_text_preview}..."
        else:
            title = f"{node_type}: {node}<br>Degree: {degree}"

        net.add_node(
            node,
            label=label,
            color=color_map.get(node_type, 'gray'),
            size=size,
            title=title,
            font={'size': 24, 'face': 'arial'},  # Increased from 16 to 24
            # Store full text as custom attribute for chunks
            full_text=chunk_full_texts.get(node, '') if node_type == 'chunk' else ''
        )

    # Add edges with colors by type
    edge_color_map = {
        'CONTAINS': '#666666',
        'HAS_TOPIC': '#ab47bc',
        'MENTIONS': '#66bb6a',
        'CO_OCCURS': '#ef5350',
        'SEMANTIC_SIM': '#42a5f5',
        'CO_PRESENCE': '#9c27b0'
    }

    for edge in G_viz.edges(data=True):
        edge_type = edge[2].get('edge_type', 'unknown')
        weight = edge[2].get('weight', 1)

        # Normalize weight to [0, 1] if > 1 (for CO_OCCURS and CO_PRESENCE edges)
        if weight > 1:
            normalized_weight = min(weight / 10.0, 1.0)  # Divide by 10 and cap at 1
        else:
            normalized_weight = weight

        net.add_edge(
            edge[0],
            edge[1],
            title=f"{edge_type} (weight: {weight:.2f}, normalized: {normalized_weight:.2f})",
            label=f"{normalized_weight:.2f}",  # Display normalized weight on edge
            color=edge_color_map.get(edge_type, 'gray'),
            width=min(normalized_weight * 3, 5)  # Increased multiplier from 2 to 3
        )

    # Generate interactive legend with chunk popup functionality
    legend_html = generate_legend_html(chunk_full_texts)

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

    # Store chunk full texts for hover display
    print("Collecting chunk texts...")
    chunk_full_texts = {}
    chunk_nodes = [n for n, d in G_viz.nodes(data=True) if d.get('type') == 'chunk']

    for node in chunk_nodes:
        node_data = G_viz.nodes[node]
        chunk_text = node_data.get('text', '')
        if chunk_text:
            chunk_full_texts[node] = chunk_text
    print(f"Collected texts for {len(chunk_full_texts)} chunks")

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

        # Create label
        if node_type == 'entity':
            label = node_data.get('entity_text', node.split('::')[-1])
        elif node_type == 'theme':
            label = node_data.get('label', f"theme_{node.split('::')[-1]}")
        elif node_type == 'chunk':
            # Use chunk ID as label
            label = node
        else:
            label = node.split('::')[-1]

        # Highlight central entity (increased sizes: 50→70, 25→40)
        if node == entity_node:
            size = 70
            color = '#ff0000'  # Red for central node
        else:
            size = 40
            color = color_map.get(node_type, 'gray')

        # Prepare title (tooltip) - include full text for chunks
        if node_type == 'chunk':
            full_text = chunk_full_texts.get(node, '')
            # Escape HTML special characters and limit length for tooltip
            full_text_preview = full_text[:200].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            title = f"{node_type}: {node}<br><br>Preview: {full_text_preview}..."
        else:
            title = f"{node_type}: {node}"

        net.add_node(
            node,
            label=label,
            color=color,
            size=size,
            title=title,
            font={'size': 28, 'face': 'arial'}  # Increased from 20 to 28
        )

    # Add edges
    edge_color_map = {
        'CONTAINS': '#666666',
        'HAS_TOPIC': '#ab47bc',
        'MENTIONS': '#66bb6a',
        'CO_OCCURS': '#ef5350',
        'SEMANTIC_SIM': '#42a5f5',
        'CO_PRESENCE': '#9c27b0'
    }

    for edge in G_viz.edges(data=True):
        edge_type = edge[2].get('edge_type', 'unknown')
        weight = edge[2].get('weight', 1)

        # Normalize weight to [0, 1] if > 1 (for CO_OCCURS and CO_PRESENCE edges)
        if weight > 1:
            normalized_weight = min(weight / 10.0, 1.0)  # Divide by 10 and cap at 1
        else:
            normalized_weight = weight

        net.add_edge(
            edge[0],
            edge[1],
            title=f"{edge_type} (weight: {weight:.2f}, normalized: {normalized_weight:.2f})",
            label=f"{normalized_weight:.2f}",  # Display normalized weight on edge
            color=edge_color_map.get(edge_type, 'gray'),
            width=min(normalized_weight * 3, 5)  # Increased multiplier from 2 to 3
        )

    # Generate interactive legend with chunk popup functionality
    legend_html = generate_legend_html(chunk_full_texts)

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
    Display pre-computed graph statistics from the statistics directory.

    Instead of recalculating statistics from the graph, this function reads
    the summary file that was already generated during graph construction,
    avoiding redundant computation.

    Args:
        graph_path: Path to .gpickle file (used to find corresponding stats)
        use_postgres: Whether to load from PostgreSQL
        db_config: Database configuration
    """
    # Extract timestamp from graph path to find corresponding stats
    timestamp = extract_timestamp_from_path(graph_path) if graph_path else None

    if not timestamp:
        print("\n" + "="*60)
        print("ERROR: Could not extract timestamp from graph path")
        print("="*60)
        print("\nPlease provide a graph file with timestamp in format: graph_YYYYMMDD_HHMMSS.gpickle")
        print("Example: outputs/graphs/graph_20251118_155115.gpickle")
        return

    # Look for the pre-computed summary file
    stats_dir = Path("outputs/statistics")
    summary_file = stats_dir / f"summary_{timestamp}.txt"

    if summary_file.exists():
        print("\n" + "="*60)
        print("READING PRE-COMPUTED STATISTICS")
        print(f"Timestamp: {timestamp}")
        print("="*60 + "\n")

        # Read and display the summary file
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_content = f.read()

        print(summary_content)

        print("\n" + "="*60)
        print("STATISTICS FILES AVAILABLE:")
        print("="*60)
        print(f"  - Summary: {summary_file}")
        print(f"  - General stats: {stats_dir}/general_stats_{timestamp}.json")
        print(f"  - Topic stats: {stats_dir}/topic_stats_{timestamp}.json")
        print(f"  - Entity stats: {stats_dir}/entity_stats_{timestamp}.json")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("WARNING: Pre-computed statistics not found")
        print("="*60)
        print(f"\nExpected file: {summary_file}")
        print("\nStatistics are normally generated during graph build.")
        print("If you built this graph before the statistics feature was added,")
        print("please rebuild the graph to generate statistics.")
        print("="*60 + "\n")


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
