"""
Relabel topics in an existing graph using LLM.

This script updates the labels of themes in an existing graph
without rebuilding everything.
"""

import pickle
import networkx as nx
import requests
from bertopic import BERTopic
import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fact_check_kg.db_connector import GraphDatabaseConnector
from fact_check_kg.config_loader import DatabaseConfig


def label_topic_with_llm(topic_words, model='llama3.1:latest'):
    """
    Generate a label for a topic using LLM.

    Args:
        topic_words: List of (word, score) tuples
        model: Ollama model to use

    Returns:
        Label string
    """
    top_words = [word for word, score in topic_words[:10]]

    try:
        prompt = f"""Based on these keywords, provide a short, clear topic label (2-4 words maximum):
Keywords: {', '.join(top_words)}

Topic label:"""

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'num_predict': 20
                }
            },
            timeout=300
        )

        if response.status_code == 200:
            label = response.json()['response'].strip()
            label = label.replace('\n', ' ').strip().strip('"\'')
            if len(label) > 50:
                label = ' '.join(label.split()[:4])
            return label.lower()

    except Exception as e:
        print(f"  LLM failed: {e}")

    # Fallback
    return " ".join(top_words[:3])


def relabel_graph(
    graph_path='knowledge_graph.gpickle',
    topic_model_path='knowledge_graph_topic_model.pkl',
    output_path='knowledge_graph.gpickle',
    use_postgres=False,
    db_config=None
):
    """
    Relabel all themes in the graph using LLM.

    Args:
        graph_path: Path to graph file (if not using postgres)
        topic_model_path: Path to topic model
        output_path: Output path for updated graph (ignored if using postgres)
        use_postgres: Whether to use PostgreSQL
        db_config: Database configuration
    """
    # Load graph
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
        db.connect()
        G = db.load_graph()
    else:
        print(f"Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

    print(f"Loading topic model from {topic_model_path}...")
    with open(topic_model_path, 'rb') as f:
        topic_model = pickle.load(f)

    # Find all theme nodes
    theme_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'theme']

    print(f"\nRelabeling {len(theme_nodes)} themes with LLM...")
    print("="*80)

    for theme_node in theme_nodes:
        topic_id = G.nodes[theme_node]['topic_id']

        # Get topic words
        topic_words = topic_model.get_topic(topic_id)

        if not topic_words:
            continue

        # Current label
        old_label = G.nodes[theme_node].get('label', '')

        # Generate new label with LLM
        new_label = label_topic_with_llm(topic_words)

        # Update graph
        G.nodes[theme_node]['label'] = new_label

        # Display
        keywords = [w for w, s in topic_words[:5]]
        print(f"\nTopic {topic_id}:")
        print(f"  Keywords: {', '.join(keywords)}")
        print(f"  Old label: '{old_label}'")
        print(f"  New label: '{new_label}'")

    # Save updated graph
    print(f"\n{'='*80}")
    if use_postgres:
        print(f"Saving updated graph to PostgreSQL...")
        db.clear_graph()
        db.save_graph(G)
        db.disconnect()
        print(f"✓ Done! Graph saved to PostgreSQL with LLM-generated labels.")
    else:
        print(f"Saving updated graph to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"✓ Done! Graph saved with LLM-generated labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relabel topics in knowledge graph")
    parser.add_argument(
        "--graph",
        type=str,
        default="knowledge_graph.gpickle",
        help="Path to graph file (if not using postgres)"
    )
    parser.add_argument(
        "--topic-model",
        type=str,
        default="knowledge_graph_topic_model.pkl",
        help="Path to topic model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="knowledge_graph.gpickle",
        help="Output path for updated graph (if not using postgres)"
    )
    parser.add_argument(
        "--use-postgres",
        action="store_true",
        help="Use PostgreSQL instead of pickle file"
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

    relabel_graph(
        graph_path=args.graph,
        topic_model_path=args.topic_model,
        output_path=args.output,
        use_postgres=args.use_postgres,
        db_config=db_config
    )
