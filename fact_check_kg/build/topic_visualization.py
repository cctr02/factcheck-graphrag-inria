"""
Topic visualization module for Knowledge Graph.

Generates HTML reports showing hierarchical topic structure with:
- Supertopics (Level 2: broad themes)
- Topics (Level 1: fine-grained themes)
- Keywords and statistics for each topic
- Chunk assignments with probabilities

The visualization helps understand the thematic structure of the corpus
and the relationships between topics.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import networkx as nx
import numpy as np
from bertopic import BERTopic


def generate_topic_html_report(
    graph: nx.Graph,
    topic_model: BERTopic,
    topics: List[int],
    probabilities: np.ndarray,
    chunk_ids: List[str],
    chunk_texts: List[str],
    output_dir: str = "outputs/statistics",
    max_sample_chunks: int = 100,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Generate an HTML report visualizing the hierarchical topic structure.

    Creates a comprehensive HTML report showing:
    - Summary statistics (chunks, topics, supertopics)
    - Supertopic cards with child topics
    - Topic cards with keywords
    - Sample chunk assignments

    Args:
        graph: NetworkX graph containing theme and supertopic nodes
        topic_model: Trained BERTopic model
        topics: List of topic assignments per chunk
        probabilities: Topic probability matrix (n_chunks, n_topics)
        chunk_ids: List of chunk IDs
        chunk_texts: List of chunk texts
        output_dir: Output directory for HTML file
        max_sample_chunks: Maximum number of chunks to show in report
        logger: Optional logger instance

    Returns:
        Path to the generated HTML file

    Example:
        >>> path = generate_topic_html_report(graph, model, topics, probs, ids, texts)
        >>> print(f"Report saved to: {path}")
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect theme nodes
    theme_nodes = {}
    for node, data in graph.nodes(data=True):
        if data.get('type') == 'theme':
            topic_id = data.get('topic_id')
            theme_nodes[topic_id] = {
                'node': node,
                'label': data.get('label', f'Topic {topic_id}'),
                'keywords': data.get('keywords', []),
                'supertopic_id': data.get('supertopic_id', topic_id)
            }

    # Collect supertopic nodes
    supertopic_nodes = {}
    for node, data in graph.nodes(data=True):
        if data.get('type') == 'supertopic':
            super_id = data.get('supertopic_id')
            supertopic_nodes[super_id] = {
                'node': node,
                'label': data.get('label', f'Supertopic {super_id}'),
                'keywords': data.get('keywords', []),
                'child_topics': data.get('child_topics', []),
                'n_topics': data.get('n_topics', 0)
            }

    # Count chunks per topic
    topic_counts = {}
    for t in topics:
        if t != -1:
            topic_counts[t] = topic_counts.get(t, 0) + 1

    # Count chunks per supertopic
    supertopic_counts = {}
    for t in topics:
        if t != -1 and t in theme_nodes:
            super_id = theme_nodes[t]['supertopic_id']
            supertopic_counts[super_id] = supertopic_counts.get(super_id, 0) + 1

    # Calculate topic assignments for sample chunks
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(chunk_texts)), min(max_sample_chunks, len(chunk_texts)))
    sample_indices.sort()

    # Get top-3 topics per chunk
    chunk_assignments = []
    for idx in sample_indices:
        if probabilities is not None and probabilities.ndim > 1:
            chunk_probs = probabilities[idx]
            top_k_indices = np.argsort(chunk_probs)[-3:][::-1]
            assignments = [(int(t), float(chunk_probs[t])) for t in top_k_indices if chunk_probs[t] > 0.01]
        else:
            assignments = [(int(topics[idx]), 1.0)] if topics[idx] != -1 else []
        chunk_assignments.append((idx, assignments))

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Topic Report - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-box {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .supertopics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .supertopic-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            color: white;
        }}
        .supertopic-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .supertopic-id {{
            font-size: 1.4em;
            font-weight: bold;
        }}
        .supertopic-count {{
            background-color: rgba(255,255,255,0.3);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .supertopic-label {{
            font-size: 1.2em;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .child-topics {{
            background-color: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }}
        .child-topic-item {{
            background-color: rgba(255,255,255,0.2);
            padding: 8px 12px;
            margin: 5px;
            border-radius: 5px;
            display: inline-block;
            font-size: 0.9em;
        }}
        .topics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .topic-card {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-top: 4px solid #3498db;
        }}
        .topic-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .topic-id {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .topic-count {{
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .topic-label {{
            font-size: 1.1em;
            color: #27ae60;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .parent-supertopic {{
            font-size: 0.9em;
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 10px;
        }}
        .keywords {{
            margin-top: 10px;
        }}
        .keyword {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 5px 10px;
            margin: 3px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .chunks-section {{
            margin-top: 30px;
        }}
        .chunk-card {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chunk-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 2px solid #ecf0f1;
            font-size: 1.1em;
        }}
        .chunk-text {{
            color: #34495e;
            line-height: 1.6;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .assignment-item {{
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            background-color: #e8f5e9;
            border-left: 4px solid #27ae60;
        }}
        .assignment-name {{
            font-weight: bold;
            font-size: 0.95em;
        }}
        .assignment-prob {{
            color: #7f8c8d;
            font-size: 0.85em;
            margin-top: 3px;
        }}
        .progress-bar {{
            height: 6px;
            background-color: #ecf0f1;
            border-radius: 3px;
            margin-top: 5px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #27ae60, #2ecc71);
        }}
        .timestamp {{
            text-align: right;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <h1>Knowledge Graph Topic Report</h1>

    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-label">Total Chunks</div>
                <div class="stat-value">{len(chunk_texts)}</div>
            </div>
            <div class="stat-box" style="border-left-color: #667eea;">
                <div class="stat-label">Supertopics</div>
                <div class="stat-value">{len(supertopic_nodes)}</div>
            </div>
            <div class="stat-box" style="border-left-color: #27ae60;">
                <div class="stat-label">Topics</div>
                <div class="stat-value">{len(theme_nodes)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Outlier Chunks</div>
                <div class="stat-value">{sum(1 for t in topics if t == -1)}</div>
            </div>
        </div>
    </div>

    <h2>Supertopics (Broad Themes)</h2>
    <div class="supertopics-grid">
"""

    # Add supertopic cards
    for super_id in sorted(supertopic_nodes.keys(), key=lambda x: supertopic_counts.get(x, 0), reverse=True):
        info = supertopic_nodes[super_id]
        count = supertopic_counts.get(super_id, 0)

        html += f"""
        <div class="supertopic-card">
            <div class="supertopic-header">
                <div class="supertopic-id">Supertopic {super_id}</div>
                <div class="supertopic-count">{count} chunks</div>
            </div>
            <div class="supertopic-label">"{info['label']}"</div>
            <div style="font-size: 0.9em; margin-bottom: 10px;">
                Keywords: {', '.join(info['keywords'][:5])}
            </div>
            <div class="child-topics">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 0.9em;">
                    Contains {info['n_topics']} topics:
                </div>
"""
        for child_id in info['child_topics']:
            if child_id in theme_nodes:
                child_label = theme_nodes[child_id]['label']
                html += f'                <span class="child-topic-item">Topic {child_id}: {child_label}</span>\n'

        html += """            </div>
        </div>
"""

    html += """    </div>

    <h2>Topics (Fine-Grained)</h2>
    <div class="topics-grid">
"""

    # Add topic cards
    for topic_id in sorted(theme_nodes.keys(), key=lambda x: topic_counts.get(x, 0), reverse=True):
        info = theme_nodes[topic_id]
        count = topic_counts.get(topic_id, 0)
        super_id = info['supertopic_id']
        super_label = supertopic_nodes.get(super_id, {}).get('label', f'Supertopic {super_id}')

        html += f"""
        <div class="topic-card">
            <div class="topic-header">
                <div class="topic-id">Topic {topic_id}</div>
                <div class="topic-count">{count} chunks</div>
            </div>
            <div class="parent-supertopic">↑ {super_label}</div>
            <div class="topic-label">"{info['label']}"</div>
            <div class="keywords">
"""
        for keyword in info['keywords'][:10]:
            html += f'                <span class="keyword">{keyword}</span>\n'

        html += """            </div>
        </div>
"""

    html += f"""    </div>

    <h2>Sample Chunk Assignments ({len(sample_indices)} chunks)</h2>
    <div class="chunks-section">
"""

    # Add sample chunks
    for idx, assignments in chunk_assignments[:50]:  # Limit to 50 for performance
        chunk_id = chunk_ids[idx] if idx < len(chunk_ids) else f"chunk_{idx}"
        text_preview = chunk_texts[idx][:500] + "..." if len(chunk_texts[idx]) > 500 else chunk_texts[idx]

        html += f"""
        <div class="chunk-card">
            <div class="chunk-header">Chunk {idx + 1} <span style="color: #7f8c8d; font-weight: normal; font-size: 0.9em;">({chunk_id})</span></div>
            <div class="chunk-text">{text_preview}</div>
"""
        for topic_id, prob in assignments:
            if topic_id in theme_nodes:
                topic_label = theme_nodes[topic_id]['label']
                prob_percent = prob * 100
                html += f"""
            <div class="assignment-item">
                <div class="assignment-name">Topic {topic_id}: {topic_label}</div>
                <div class="assignment-prob">Probability: {prob:.3f} ({prob_percent:.1f}%)</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {min(prob_percent, 100)}%"></div>
                </div>
            </div>
"""

        html += """        </div>
"""

    html += f"""
    </div>

    <div class="timestamp">
        Generated: {timestamp}<br>
        {len(theme_nodes)} topics grouped into {len(supertopic_nodes)} supertopics
    </div>
</body>
</html>
"""

    # Save HTML file
    output_file = output_path / f"topic_visualization_{timestamp}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Topic visualization saved to: {output_file}")

    return str(output_file)
