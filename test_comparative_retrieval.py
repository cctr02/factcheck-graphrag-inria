#!/usr/bin/env python3
"""
Comparative study: Graph-RAG vs Classic RAG

Generates 30 news-related queries and compares:
1. Graph-RAG (hybrid: semantic + theme + entity + topic distribution)
2. Classic RAG (pure semantic similarity)

For each query, computes the overlap between top-10 chunks from both methods.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import json
from datetime import datetime

# Import Graph-RAG retrieval
from fact_check_kg.retrieval import HybridRetriever
from fact_check_kg.config_loader import get_retrieval_config
from fact_check_kg.logger import get_logger
import spacy


def generate_test_queries():
    """Generate 30 news-related test queries."""
    queries = [
        # Politics
        "Trump indictment trial verdict legal proceedings",
        "Biden economic policy inflation interest rates",
        "European Union migration policy border control",
        "Ukraine war Russia military conflict ceasefire",
        "China Taiwan relations military tensions",

        # Climate & Environment
        "Climate change global warming carbon emissions",
        "Renewable energy solar wind power transition",
        "Extreme weather floods hurricanes wildfires",
        "Paris agreement climate goals net zero",
        "Amazon rainforest deforestation biodiversity loss",

        # Technology
        "Artificial intelligence ChatGPT regulation ethics",
        "Social media misinformation content moderation",
        "Cybersecurity data breach privacy protection",
        "Cryptocurrency bitcoin blockchain regulation",
        "5G technology network infrastructure rollout",

        # Health
        "COVID-19 pandemic vaccine booster immunity",
        "Mental health depression anxiety treatment",
        "Healthcare system reform insurance coverage",
        "Obesity epidemic diabetes public health",
        "Cancer treatment breakthrough immunotherapy",

        # Economy
        "Stock market crash recession economic crisis",
        "Unemployment rate job market labor shortage",
        "Housing market prices mortgage rates",
        "Federal Reserve interest rate inflation control",
        "Corporate earnings profit margins revenue",

        # Society
        "Gun violence school shooting legislation",
        "Immigration asylum refugees border policy",
        "Police reform criminal justice accountability",
        "Education reform school funding teachers",
        "Gender equality pay gap women rights"
    ]
    return queries


def classic_rag_retrieval(query, chunks, chunk_ids, embedding_model, top_k=30):
    """
    Classic RAG: Pure semantic similarity retrieval.

    Args:
        query: Query string
        chunks: List of chunk texts
        chunk_ids: Corresponding chunk IDs
        embedding_model: SentenceTransformer model
        top_k: Number of chunks to retrieve

    Returns:
        List of (chunk_id, similarity_score) tuples
    """
    # Encode query
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]

    # Encode all chunks
    chunk_embeddings = embedding_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    # Compute similarities
    similarities = chunk_embeddings @ query_embedding

    # Get top-K
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    results = [(chunk_ids[i], float(similarities[i])) for i in top_k_indices]

    return results


def graph_rag_retrieval(query, retriever, top_k=30):
    """
    Graph-RAG: Hybrid retrieval (semantic + theme + entity + topic distribution).

    Args:
        query: Query string
        retriever: HybridRetriever instance
        top_k: Number of chunks to retrieve

    Returns:
        List of (chunk_id, total_score) tuples
    """
    results = retriever.retrieve(query)

    # Extract top-K chunk IDs and scores
    top_results = [(r['chunk_id'], r['score_total']) for r in results[:top_k]]

    return top_results


def compute_overlap(results1, results2):
    """
    Compute overlap percentage between two result sets.

    Args:
        results1: List of (chunk_id, score) tuples
        results2: List of (chunk_id, score) tuples

    Returns:
        Overlap percentage (0-100)
    """
    ids1 = set(chunk_id for chunk_id, _ in results1)
    ids2 = set(chunk_id for chunk_id, _ in results2)

    overlap = len(ids1 & ids2)
    total = len(ids1)  # Should be same as len(ids2)

    return (overlap / total) * 100 if total > 0 else 0.0


def generate_html_report(results_data, overlaps, graph, queries, timestamp,
                         mean_overlap, median_overlap, std_overlap, min_overlap, max_overlap):
    """
    Generate a detailed HTML report showing one example query per overlap category.

    Args:
        results_data: List of result dictionaries from comparative study
        overlaps: List of overlap percentages
        graph: NetworkX graph containing chunk texts
        queries: List of query strings
        timestamp: Timestamp string for filename
        mean_overlap, median_overlap, std_overlap, min_overlap, max_overlap: Statistics
    """
    # Define overlap categories
    categories = [
        (0, 20, "0-20%"),
        (20, 40, "20-40%"),
        (40, 60, "40-60%"),
        (60, 80, "60-80%"),
        (80, 100, "80-100%")
    ]

    # Select one example query per category
    examples = []
    for min_val, max_val, label in categories:
        # Find queries in this category
        candidates = [
            (i, results_data[i], overlaps[i])
            for i in range(len(results_data))
            if min_val <= overlaps[i] < max_val or (max_val == 100 and overlaps[i] == 100)
        ]

        if candidates:
            # Pick the first candidate
            idx, result, overlap = candidates[0]
            examples.append({
                'category': label,
                'query_id': result['query_id'],
                'query': result['query'],
                'overlap': overlap,
                'graph_rag_top3': result['graph_rag_top3'],
                'classic_rag_top3': result['classic_rag_top3']
            })

    # Retrieve full result sets for each example
    for example in examples:
        query = example['query']
        query_id = example['query_id']

        # Find the full top-10 results for this query
        result = results_data[query_id - 1]

        # We need to re-run retrieval to get full top-10 lists
        # For now, store the top-3 chunk IDs we already have
        example['graph_rag_chunks'] = result['graph_rag_top3']
        example['classic_rag_chunks'] = result['classic_rag_top3']

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph-RAG vs Classic RAG - Comparative Study</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
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
        .summary table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .summary td {{
            padding: 8px;
            border-bottom: 1px solid #bdc3c7;
        }}
        .summary td:first-child {{
            font-weight: bold;
            width: 200px;
        }}
        .example {{
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .category-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .cat-0-20 {{ background-color: #e74c3c; color: white; }}
        .cat-20-40 {{ background-color: #e67e22; color: white; }}
        .cat-40-60 {{ background-color: #f39c12; color: white; }}
        .cat-60-80 {{ background-color: #2ecc71; color: white; }}
        .cat-80-100 {{ background-color: #27ae60; color: white; }}
        .query-text {{
            background-color: #3498db;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 1.1em;
        }}
        .overlap-stat {{
            font-size: 1.3em;
            color: #2c3e50;
            margin: 15px 0;
            font-weight: bold;
        }}
        .chunks-section {{
            margin-top: 20px;
        }}
        .chunk-group {{
            margin-bottom: 20px;
        }}
        .chunk-group h3 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .chunk {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 4px;
        }}
        .chunk-id {{
            font-weight: bold;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .chunk-text {{
            line-height: 1.6;
            color: #2c3e50;
        }}
        .graph-rag-only {{ border-left-color: #e74c3c; }}
        .classic-rag-only {{ border-left-color: #9b59b6; }}
        .both {{ border-left-color: #27ae60; }}
        .no-results {{
            color: #95a5a6;
            font-style: italic;
            padding: 10px;
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
    <h1>Graph-RAG vs Classic RAG: Comparative Study</h1>

    <div class="summary">
        <h2>Summary Statistics (30 Queries)</h2>
        <table>
            <tr>
                <td>Mean Overlap:</td>
                <td>{mean_overlap:.2f}%</td>
            </tr>
            <tr>
                <td>Median Overlap:</td>
                <td>{median_overlap:.2f}%</td>
            </tr>
            <tr>
                <td>Standard Deviation:</td>
                <td>{std_overlap:.2f}%</td>
            </tr>
            <tr>
                <td>Min Overlap:</td>
                <td>{min_overlap:.2f}%</td>
            </tr>
            <tr>
                <td>Max Overlap:</td>
                <td>{max_overlap:.2f}%</td>
            </tr>
        </table>
    </div>

    <h2>Detailed Examples by Overlap Category</h2>
    <p>The following sections show one representative query from each overlap category, displaying the chunks found exclusively by each method.</p>
"""

    # Add each example
    for example in examples:
        category_class = example['category'].replace('%', '').replace('-', '-')
        cat_class = f"cat-{category_class}"

        # Get chunk IDs for each group
        graph_ids = set(example['graph_rag_chunks'])
        classic_ids = set(example['classic_rag_chunks'])

        only_graph = graph_ids - classic_ids
        only_classic = classic_ids - graph_ids
        both = graph_ids & classic_ids

        html += f"""
    <div class="example">
        <span class="category-badge {cat_class}">Overlap: {example['category']}</span>
        <div class="query-text">
            <strong>Query #{example['query_id']}:</strong> {example['query']}
        </div>
        <div class="overlap-stat">
            Overlap: {example['overlap']:.1f}% ({int(len(both))} common chunks out of top-3 shown)
        </div>

        <div class="chunks-section">
            <div class="chunk-group">
                <h3>🔴 Found ONLY by Graph-RAG ({len(only_graph)} chunks):</h3>
"""

        if only_graph:
            for chunk_id in only_graph:
                chunk_text = graph.nodes.get(chunk_id, {}).get('text', 'N/A')
                html += f"""
                <div class="chunk graph-rag-only">
                    <div class="chunk-id">Chunk ID: {chunk_id}</div>
                    <div class="chunk-text">{chunk_text}</div>
                </div>
"""
        else:
            html += '                <div class="no-results">No chunks found exclusively by Graph-RAG</div>\n'

        html += """
            </div>

            <div class="chunk-group">
                <h3>🟣 Found ONLY by Classic RAG (""" + str(len(only_classic)) + """ chunks):</h3>
"""

        if only_classic:
            for chunk_id in only_classic:
                chunk_text = graph.nodes.get(chunk_id, {}).get('text', 'N/A')
                html += f"""
                <div class="chunk classic-rag-only">
                    <div class="chunk-id">Chunk ID: {chunk_id}</div>
                    <div class="chunk-text">{chunk_text}</div>
                </div>
"""
        else:
            html += '                <div class="no-results">No chunks found exclusively by Classic RAG</div>\n'

        html += """
            </div>

            <div class="chunk-group">
                <h3>🟢 Found by BOTH methods (""" + str(len(both)) + """ chunks):</h3>
"""

        if both:
            for chunk_id in both:
                chunk_text = graph.nodes.get(chunk_id, {}).get('text', 'N/A')
                html += f"""
                <div class="chunk both">
                    <div class="chunk-id">Chunk ID: {chunk_id}</div>
                    <div class="chunk-text">{chunk_text}</div>
                </div>
"""
        else:
            html += '                <div class="no-results">No common chunks</div>\n'

        html += """
            </div>
        </div>
    </div>
"""

    html += f"""
    <div class="timestamp">
        Generated on {timestamp}
    </div>
</body>
</html>
"""

    # Save HTML file
    output_file = f"outputs/comparative_study_detailed_{timestamp}.html"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Detailed HTML report saved to: {output_file}")


def main():
    print("=" * 80)
    print("COMPARATIVE STUDY: Graph-RAG vs Classic RAG")
    print("=" * 80)

    # Load configuration
    config = get_retrieval_config()
    logger = get_logger("comparative_test", "INFO")

    # Find latest graph
    from fact_check_kg.graph_manager import get_latest_graph_id, load_graph_by_id

    graph_id = get_latest_graph_id()
    if not graph_id:
        print("ERROR: No graphs found. Please build a graph first.")
        sys.exit(1)

    print(f"\nLoading latest graph: {graph_id}")
    graph, topic_model, metadata = load_graph_by_id(graph_id)

    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Extract all chunks from graph
    print("\nExtracting chunks from graph...")
    chunks = []
    chunk_ids = []
    for node, data in graph.nodes(data=True):
        if data.get('type') == 'chunk':
            chunks.append(data['text'])
            chunk_ids.append(node)

    print(f"Total chunks: {len(chunks)}")

    # Load models
    print("\nLoading models...")
    embedding_model = SentenceTransformer(config.models.embedding_model)
    nlp = spacy.load(config.models.ner_model)

    # Initialize Graph-RAG retriever
    print("Initializing Graph-RAG retriever...")
    retriever = HybridRetriever(
        graph=graph,
        topic_model=topic_model,
        embedding_model=embedding_model,
        nlp=nlp,
        entity_types=["PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "NORP"],
        alpha=config.retrieval.alpha,
        beta=config.retrieval.beta,
        gamma=config.retrieval.gamma,
        delta=config.retrieval.delta,
        top_k=30,
        graph_id=graph_id,
        cache_dir=str(Path(config.paths.graphs_dir)),
        config=config.to_dict(),
        logger=logger
    )

    # Generate test queries
    print("\nGenerating 30 test queries...")
    queries = generate_test_queries()
    print(f"Generated {len(queries)} queries")

    # Run comparative tests
    print("\n" + "=" * 80)
    print("RUNNING COMPARATIVE TESTS")
    print("=" * 80)

    results_data = []
    overlaps = []

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/30] Query: {query[:60]}...")

        # Graph-RAG retrieval
        print("  Running Graph-RAG...")
        graph_rag_results = graph_rag_retrieval(query, retriever, top_k=30)

        # Classic RAG retrieval
        print("  Running Classic RAG...")
        classic_rag_results = classic_rag_retrieval(query, chunks, chunk_ids, embedding_model, top_k=30)

        # Compute overlap
        overlap = compute_overlap(graph_rag_results, classic_rag_results)
        overlaps.append(overlap)

        print(f"  ✓ Overlap: {overlap:.1f}% ({int(overlap*30/100)} common chunks out of 30)")

        # Store results
        results_data.append({
            "query_id": i,
            "query": query,
            "overlap_percentage": overlap,
            "graph_rag_top3": [chunk_id for chunk_id, _ in graph_rag_results[:3]],
            "classic_rag_top3": [chunk_id for chunk_id, _ in classic_rag_results[:3]]
        })

    # Compute statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    mean_overlap = np.mean(overlaps)
    std_overlap = np.std(overlaps)
    min_overlap = np.min(overlaps)
    max_overlap = np.max(overlaps)
    median_overlap = np.median(overlaps)

    print(f"\nOverlap Statistics (30 queries):")
    print(f"  Mean overlap:     {mean_overlap:.2f}%")
    print(f"  Median overlap:   {median_overlap:.2f}%")
    print(f"  Std deviation:    {std_overlap:.2f}%")
    print(f"  Min overlap:      {min_overlap:.2f}%")
    print(f"  Max overlap:      {max_overlap:.2f}%")

    # Distribution
    print(f"\nOverlap Distribution:")
    bins = [0, 20, 40, 60, 80, 100]
    for i in range(len(bins) - 1):
        count = sum(1 for o in overlaps if bins[i] <= o < bins[i+1])
        print(f"  {bins[i]:3d}-{bins[i+1]:3d}%: {count:2d} queries {'█' * count}")

    # Perfect overlap count
    perfect_overlap = sum(1 for o in overlaps if o == 100.0)
    no_overlap = sum(1 for o in overlaps if o == 0.0)

    print(f"\nSpecial cases:")
    print(f"  Perfect overlap (100%): {perfect_overlap} queries")
    print(f"  No overlap (0%):        {no_overlap} queries")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/comparative_study_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "graph_id": graph_id,
        "total_queries": len(queries),
        "statistics": {
            "mean_overlap": float(mean_overlap),
            "median_overlap": float(median_overlap),
            "std_overlap": float(std_overlap),
            "min_overlap": float(min_overlap),
            "max_overlap": float(max_overlap)
        },
        "results": results_data
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_file}")

    # Generate HTML report with examples from each overlap category
    print("\nGenerating detailed HTML report...")
    generate_html_report(
        results_data,
        overlaps,
        graph,
        queries,
        timestamp,
        mean_overlap,
        median_overlap,
        std_overlap,
        min_overlap,
        max_overlap
    )

    print("\n" + "=" * 80)

    # Interpretation
    print("\nINTERPRETATION:")
    if mean_overlap > 70:
        print("  → High overlap: Graph-RAG and Classic RAG return similar results.")
        print("     The graph structure may not add much value for these queries.")
    elif mean_overlap > 40:
        print("  → Moderate overlap: Graph-RAG adds complementary information.")
        print("     The hybrid approach retrieves some different but relevant chunks.")
    else:
        print("  → Low overlap: Graph-RAG and Classic RAG diverge significantly.")
        print("     Graph structure provides substantial additional value.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
