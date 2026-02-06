"""
Common propagation logic for graph-based retrieval strategies.

This module provides BFS-based relevance propagation functions used by:
- Entity-based retrieval (Strategy 4)
- Thematic retrieval (Strategy 3)

Propagation Algorithm:
    Uses Breadth-First Search (BFS) to propagate relevance scores through
    the graph, with relevance decaying at each hop based on edge weights.

    For each node n reached from seed node s via path p:
        relevance(n) = max over all paths p of: ∏(edge_weight(e) for e in p)

    The propagation stops when:
    1. No more unvisited neighbors exist
    2. Propagated relevance falls below threshold

Entity Propagation:
    - Seed nodes: Entities extracted from query (relevance = 1.0)
    - Traversal edges: CO_OCCURS, SEMANTIC_SIM (entity-entity edges only)
    - Output: relevance scores for all reachable entities

Theme Propagation:
    - Seed nodes: Themes detected in query (relevance = topic probability)
    - Traversal edges: CO_PRESENCE, SEMANTIC_SIM (theme-theme edges only)
    - Output: relevance scores for all reachable themes

The propagated relevance is then used to score chunks:
    chunk_score = Σ(entity_relevance × MENTIONS_weight) for entity strategy
    chunk_score = Σ(theme_relevance × HAS_TOPIC_weight) for thematic strategy
"""

from typing import List, Dict, Tuple
from collections import deque
import networkx as nx


def propagate_relevance_entities(
    graph: nx.Graph,
    seed_entities: List[str],
    relevance_threshold: float
) -> Dict[str, float]:
    """
    Propagate relevance through entity graph using BFS.

    Starting from seed entities with relevance 1.0, propagates through
    CO_OCCURS and SEMANTIC_SIM edges, multiplying relevance by edge weight.

    Args:
        graph: Knowledge graph
        seed_entities: List of entity node IDs to start from
        relevance_threshold: Minimum relevance to continue traversal

    Returns:
        Dictionary mapping entity_node to final relevance score

    Example:
        >>> entity_relevance = propagate_relevance_entities(
        ...     graph, ["entity::PERSON::trump"], threshold=0.1
        ... )
        >>> entity_relevance["entity::PERSON::trump"]
        1.0
    """
    # Entity relevance scores
    entity_relevance = {}
    for entity in seed_entities:
        entity_relevance[entity] = 1.0  # Seed entities start with relevance 1.0

    # BFS queue: (entity_node, current_relevance)
    queue = deque([(entity, 1.0) for entity in seed_entities])
    visited = set(seed_entities)

    while queue:
        current_entity, current_relevance = queue.popleft()

        # Explore neighbors (other entities connected via CO_OCCURS or SEMANTIC_SIM)
        for neighbor in graph.neighbors(current_entity):
            if graph.nodes[neighbor].get('type') != 'entity':
                continue  # Skip non-entity nodes

            edge_data = graph[current_entity][neighbor]
            edge_type = edge_data.get('edge_type')

            if edge_type not in ['CO_OCCURS', 'SEMANTIC_SIM']:
                continue  # Only traverse entity-entity edges

            # Get edge weight
            edge_weight = edge_data.get('weight', 0)

            # Calculate propagated relevance
            propagated_relevance = current_relevance * edge_weight

            # Only continue if above threshold
            if propagated_relevance < relevance_threshold:
                continue

            # Update relevance (take maximum if entity already visited)
            if neighbor not in entity_relevance:
                entity_relevance[neighbor] = propagated_relevance
            else:
                entity_relevance[neighbor] = max(entity_relevance[neighbor], propagated_relevance)

            # Add to queue if not visited yet
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, propagated_relevance))

    return entity_relevance


def propagate_relevance_themes(
    graph: nx.Graph,
    seed_themes: List[Tuple[str, float]],
    relevance_threshold: float
) -> Dict[str, float]:
    """
    Propagate relevance through theme graph using BFS.

    Starting from seed themes with their initial probabilities, propagates through
    CO_PRESENCE and SEMANTIC_SIM edges, multiplying relevance by edge weight.

    Args:
        graph: Knowledge graph
        seed_themes: List of (theme_node, initial_probability) tuples
        relevance_threshold: Minimum relevance to continue traversal

    Returns:
        Dictionary mapping theme_node to final relevance score

    Example:
        >>> theme_relevance = propagate_relevance_themes(
        ...     graph, [("theme::0", 0.8), ("theme::1", 0.15)], threshold=0.1
        ... )
    """
    # Theme relevance scores
    theme_relevance = {}
    queue = deque()
    visited = set()

    for theme_node, initial_prob in seed_themes:
        theme_relevance[theme_node] = initial_prob
        queue.append((theme_node, initial_prob))
        visited.add(theme_node)

    while queue:
        current_theme, current_relevance = queue.popleft()

        # Explore neighbors (other themes connected via CO_PRESENCE or SEMANTIC_SIM)
        for neighbor in graph.neighbors(current_theme):
            if graph.nodes[neighbor].get('type') != 'theme':
                continue  # Skip non-theme nodes

            edge_data = graph[current_theme][neighbor]
            edge_type = edge_data.get('edge_type')

            if edge_type not in ['CO_PRESENCE', 'SEMANTIC_SIM']:
                continue  # Only traverse theme-theme edges

            # Get edge weight
            edge_weight = edge_data.get('weight', 0)

            # Calculate propagated relevance
            propagated_relevance = current_relevance * edge_weight

            # Only continue if above threshold
            if propagated_relevance < relevance_threshold:
                continue

            # Update relevance (take maximum if theme already visited)
            if neighbor not in theme_relevance:
                theme_relevance[neighbor] = propagated_relevance
            else:
                theme_relevance[neighbor] = max(theme_relevance[neighbor], propagated_relevance)

            # Add to queue if not visited yet
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, propagated_relevance))

    return theme_relevance
