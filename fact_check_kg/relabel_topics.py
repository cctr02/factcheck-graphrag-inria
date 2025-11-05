"""
Relabel topics in an existing graph using LLM.

This script updates the labels of themes in an existing graph
without rebuilding everything.
"""

import pickle
import networkx as nx
import requests
from bertopic import BERTopic


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
            timeout=15
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
    output_path='knowledge_graph.gpickle'
):
    """
    Relabel all themes in the graph using LLM.

    Args:
        graph_path: Path to graph file
        topic_model_path: Path to topic model
        output_path: Output path for updated graph
    """
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
    print(f"Saving updated graph to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)

    print(f"✓ Done! Graph saved with LLM-generated labels.")


if __name__ == "__main__":
    relabel_graph()
