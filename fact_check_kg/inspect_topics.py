"""
Inspect topics extracted by BERTopic.

Shows keywords and document counts for each topic.
"""

import pickle
from bertopic import BERTopic


def inspect_topics(topic_model_path: str = "knowledge_graph_topic_model.pkl"):
    """
    Display all topics with their keywords and counts.

    Args:
        topic_model_path: Path to saved BERTopic model
    """
    print(f"Loading topic model from {topic_model_path}...")
    with open(topic_model_path, 'rb') as f:
        topic_model = pickle.load(f)

    print("\n" + "="*80)
    print("TOPICS DISCOVERED BY BERTOPIC")
    print("="*80)

    # Get all topics
    topics = topic_model.get_topics()

    # Count excluding outliers
    n_topics = len(topics)
    if -1 in topics:
        n_topics -= 1

    print(f"\nTotal topics: {n_topics} (excluding outliers)")
    print()

    # For each topic
    for topic_id in sorted(topics.keys()):
        if topic_id == -1:  # Skip outliers
            print(f"\nTopic -1 (Outliers):")
            topic_freq = topic_model.get_topic_info()
            outlier_count = topic_freq[topic_freq.Topic == -1]['Count'].values
            if len(outlier_count) > 0:
                print(f"  Count: {outlier_count[0]} chunks")
            continue

        # Get topic words
        words = topic_model.get_topic(topic_id)
        if not words:
            continue

        top_words = [word for word, score in words[:10]]

        # Get count
        topic_info = topic_model.get_topic_info()
        count = topic_info[topic_info.Topic == topic_id]['Count'].values[0]

        print(f"\n{'='*80}")
        print(f"Topic {topic_id}:")
        print(f"  Count: {count} chunks")
        print(f"  Top 10 keywords: {', '.join(top_words)}")

        # Show word scores
        print(f"  Detailed scores:")
        for word, score in words[:5]:
            print(f"    - {word}: {score:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    inspect_topics()
