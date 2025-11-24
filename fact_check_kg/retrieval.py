"""
Retrieval module for fact-checking using the Knowledge Graph.

This module implements a hybrid retrieval strategy combining:
1. Semantic retrieval (embedding similarity)
2. Thematic retrieval (via topic model and graph traversal)
3. Entity-based retrieval (via entity extraction and graph traversal)
4. Topic distribution retrieval (via Jensen-Shannon divergence)
"""

import networkx as nx
import numpy as np
import spacy
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from scipy.spatial.distance import jensenshannon
import logging


class HybridRetriever:
    """
    Hybrid retrieval system for fact-checking claims.

    Combines four retrieval strategies:
    - Semantic: Direct embedding similarity
    - Thematic: Topic classification + graph traversal
    - Entity: Entity extraction + graph traversal (3 paths)
    - Topic Distribution: Jensen-Shannon divergence between topic distributions

    The final score is a weighted combination of all four.
    """

    def __init__(
        self,
        graph: nx.Graph,
        topic_model: BERTopic,
        embedding_model: SentenceTransformer,
        nlp: spacy.Language,
        entity_types: List[str],
        alpha: float = 0.3,
        beta: float = 0.2,
        gamma: float = 0.3,
        delta: float = 0.2,
        top_k: int = 10,
        theme_threshold: float = 0.1,
        entity_relevance_threshold: float = 0.1,
        theme_relevance_threshold: float = 0.1,
        graph_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the hybrid retriever.

        Args:
            graph: Knowledge graph (from graph_builder)
            topic_model: Trained BERTopic model
            embedding_model: SentenceTransformer model
            nlp: spaCy model for entity extraction
            entity_types: List of entity types to extract
            alpha: Weight for semantic score (default: 0.3)
            beta: Weight for thematic score (default: 0.2)
            gamma: Weight for entity score (default: 0.3)
            delta: Weight for topic distribution score (default: 0.2)
            top_k: Number of chunks to return
            theme_threshold: Minimum confidence for theme detection
            entity_relevance_threshold: Minimum relevance to continue entity traversal
            theme_relevance_threshold: Minimum relevance to continue theme traversal
            graph_id: Optional graph ID for loading cached embeddings
            cache_dir: Optional directory containing FAISS cache
            config: Optional complete configuration dictionary
            logger: Optional logger instance
        """
        self.graph = graph
        self.topic_model = topic_model
        self.embedding_model = embedding_model
        self.nlp = nlp
        self.entity_types = entity_types
        self.config = config

        # Weights (should sum to 1.0)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Parameters
        self.top_k = top_k
        self.theme_threshold = theme_threshold
        self.entity_relevance_threshold = entity_relevance_threshold
        self.theme_relevance_threshold = theme_relevance_threshold

        # Cache parameters
        self.graph_id = graph_id
        self.cache_dir = cache_dir

        self.logger = logger or logging.getLogger(__name__)

        # Pre-compute chunk embeddings and topic distributions for efficiency
        self.logger.info("Pre-computing chunk embeddings and topic distributions...")
        self._precompute_chunk_embeddings(graph_id, cache_dir)
        self._precompute_chunk_topic_distributions()

    def _precompute_chunk_embeddings(self, graph_id: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Pre-compute embeddings for all chunks in the graph.

        If graph_id and cache_dir are provided, will try to load from FAISS cache first.
        Otherwise, computes embeddings on-the-fly.

        Args:
            graph_id: Optional graph ID for loading cached embeddings
            cache_dir: Optional directory containing FAISS cache
        """
        from pathlib import Path
        from fact_check_kg.embedding_cache import EmbeddingCache

        self.chunk_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'chunk'
        ]

        self.chunk_texts = [
            self.graph.nodes[n]['text']
            for n in self.chunk_nodes
        ]

        # Try to load from FAISS cache if available
        if graph_id and cache_dir:
            try:
                cache_path = Path(cache_dir)
                embedding_cache = EmbeddingCache(logger=self.logger)
                embedding_cache.load(cache_path, graph_id)

                # Get embeddings from cache
                cached_chunk_ids, cached_embeddings = embedding_cache.get_all_embeddings()

                # Verify chunk IDs match
                if cached_chunk_ids == self.chunk_nodes:
                    self.chunk_embeddings = cached_embeddings
                    self.logger.info(f"Loaded embeddings from FAISS cache for {len(self.chunk_nodes)} chunks")
                    return
                else:
                    self.logger.warning("Cached chunk IDs don't match current graph, recomputing embeddings...")
            except FileNotFoundError:
                self.logger.warning(f"FAISS cache not found for graph_id {graph_id}, computing embeddings...")
            except Exception as e:
                self.logger.warning(f"Error loading FAISS cache: {e}, computing embeddings...")

        # Fallback: compute embeddings on-the-fly
        self.logger.info("Computing embeddings for all chunks...")
        self.chunk_embeddings = self.embedding_model.encode(
            self.chunk_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.logger.info(f"Computed embeddings for {len(self.chunk_nodes)} chunks")

    def _precompute_chunk_topic_distributions(self):
        """Pre-compute topic probability distributions for all chunks."""
        # Get topic distributions from BERTopic
        _, chunk_topic_probs = self.topic_model.transform(self.chunk_texts)

        # Store as numpy array for efficient computation
        self.chunk_topic_distributions = chunk_topic_probs

        # Create mapping from chunk_id to distribution index
        self.chunk_to_dist_idx = {
            chunk_id: idx
            for idx, chunk_id in enumerate(self.chunk_nodes)
        }

        self.logger.info(f"Pre-computed topic distributions for {len(self.chunk_nodes)} chunks")
        self.logger.info(f"Number of topics: {chunk_topic_probs.shape[1]}")

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text (same as in graph_builder).

        Args:
            text: Input text

        Returns:
            List of (entity_text, entity_type) tuples
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity_text = ent.text.strip().lower()
                if len(entity_text) > 2:
                    entities.append((entity_text, ent.label_))

        return entities

    def _propagate_relevance_entities(
        self,
        seed_entities: List[str],
        relevance_threshold: float
    ) -> Dict[str, float]:
        """
        Propagate relevance through entity graph using BFS.

        Args:
            seed_entities: List of entity node IDs to start from
            relevance_threshold: Minimum relevance to continue traversal

        Returns:
            Dictionary mapping entity_node to final relevance score
        """
        from collections import deque

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
            for neighbor in self.graph.neighbors(current_entity):
                if self.graph.nodes[neighbor].get('type') != 'entity':
                    continue  # Skip non-entity nodes

                edge_data = self.graph[current_entity][neighbor]
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

    def _propagate_relevance_themes(
        self,
        seed_themes: List[Tuple[str, float]],
        relevance_threshold: float
    ) -> Dict[str, float]:
        """
        Propagate relevance through theme graph using BFS.

        Args:
            seed_themes: List of (theme_node, initial_probability) tuples
            relevance_threshold: Minimum relevance to continue traversal

        Returns:
            Dictionary mapping theme_node to final relevance score
        """
        from collections import deque

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
            for neighbor in self.graph.neighbors(current_theme):
                if self.graph.nodes[neighbor].get('type') != 'theme':
                    continue  # Skip non-theme nodes

                edge_data = self.graph[current_theme][neighbor]
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

    def retrieve_semantic(self, claim: str) -> Dict[str, float]:
        """
        Step 1: Semantic retrieval using embedding similarity.

        Args:
            claim: Claim text to verify

        Returns:
            Dictionary mapping chunk_id to semantic score (normalized)
        """
        self.logger.info("\n--- Step 1: Semantic Retrieval ---")

        # Embed claim
        claim_embedding = self.embedding_model.encode(
            [claim],
            normalize_embeddings=True
        )[0]

        # Compute similarities
        similarities = np.matmul(self.chunk_embeddings, claim_embedding)

        # Create score dictionary
        scores = {}
        for chunk_node, sim in zip(self.chunk_nodes, similarities):
            scores[chunk_node] = float(sim)

        # Normalize scores (min-max)
        if scores:
            min_score = min(scores.values())
            max_score = max(scores.values())
            if max_score > min_score:
                scores = {
                    k: (v - min_score) / (max_score - min_score)
                    for k, v in scores.items()
                }

        self.logger.info(f"Found {len(scores)} chunks with semantic scores")
        self.logger.info(f"Top score: {max(scores.values()):.3f}")

        return scores

    def retrieve_thematic(self, claim: str) -> Dict[str, float]:
        """
        Step 2: Thematic retrieval using topic model and graph traversal with relevance propagation.

        Uses BFS to propagate relevance through theme graph (CO_PRESENCE, SEMANTIC_SIM edges),
        then scores chunks based on theme relevance weighted by probabilities from HAS_TOPIC edges.

        Args:
            claim: Claim text to verify

        Returns:
            Dictionary mapping chunk_id to thematic score (normalized)
        """
        self.logger.info("\n--- Step 2: Thematic Retrieval ---")

        # Classify claim into topics
        topic_probs = self.topic_model.transform([claim])[1][0]

        # Detect themes above threshold
        detected_themes = [
            (idx, prob)
            for idx, prob in enumerate(topic_probs)
            if prob > self.theme_threshold
        ]

        self.logger.info(f"Detected {len(detected_themes)} themes in claim")
        for idx, prob in detected_themes:
            theme_label = self.graph.nodes.get(f"theme::{idx}", {}).get('label', f'topic_{idx}')
            self.logger.info(f"  Theme {idx} ({theme_label}): {prob:.3f}")

        # Build seed theme list with initial probabilities
        seed_themes = []
        for theme_id, prob in detected_themes:
            theme_node = f"theme::{theme_id}"
            if self.graph.has_node(theme_node):
                seed_themes.append((theme_node, prob))
            else:
                self.logger.debug(f"Theme not in graph: {theme_node}")

        if not seed_themes:
            self.logger.warning("No themes from claim found in graph")
            return {}

        self.logger.info(f"Found {len(seed_themes)} seed themes in graph")

        # Propagate relevance through theme graph
        theme_relevance = self._propagate_relevance_themes(
            seed_themes=seed_themes,
            relevance_threshold=self.theme_relevance_threshold
        )

        self.logger.info(f"Relevance propagated to {len(theme_relevance)} themes")

        # Score chunks based on theme relevance and HAS_TOPIC edge weights (probabilities)
        scores = defaultdict(float)

        for theme_node, relevance in theme_relevance.items():
            # Find chunks connected via HAS_TOPIC edges
            for neighbor in self.graph.neighbors(theme_node):
                edge_data = self.graph[theme_node][neighbor]

                # Only consider HAS_TOPIC edges pointing to chunks
                if (edge_data.get('edge_type') == 'HAS_TOPIC' and
                    self.graph.nodes[neighbor].get('type') == 'chunk'):

                    # Get probability weight from HAS_TOPIC edge
                    prob_weight = edge_data.get('weight', 0.0)

                    # Chunk score = theme relevance × HAS_TOPIC probability
                    chunk_score = relevance * prob_weight

                    # Accumulate scores (chunks can be associated with multiple themes)
                    scores[neighbor] += chunk_score

        # Normalize scores
        if scores:
            min_score = min(scores.values())
            max_score = max(scores.values())
            if max_score > min_score:
                scores = {
                    k: (v - min_score) / (max_score - min_score)
                    for k, v in scores.items()
                }

        self.logger.info(f"Found {len(scores)} chunks via thematic retrieval")
        if scores:
            self.logger.info(f"Top score: {max(scores.values()):.3f}")

        return dict(scores)

    def retrieve_entity(self, claim: str) -> Dict[str, float]:
        """
        Step 3: Entity-based retrieval using graph traversal with relevance propagation.

        Uses BFS to propagate relevance through entity graph (CO_OCCURS, SEMANTIC_SIM edges),
        then scores chunks based on entity relevance weighted by TF-IDF from MENTIONS edges.

        Args:
            claim: Claim text to verify

        Returns:
            Dictionary mapping chunk_id to entity score (normalized)
        """
        self.logger.info("\n--- Step 3: Entity-based Retrieval ---")

        # Extract entities from claim
        entities_claim = self.extract_entities(claim)

        self.logger.info(f"Extracted {len(entities_claim)} entities from claim:")
        for ent_text, ent_type in entities_claim:
            self.logger.info(f"  {ent_type}: {ent_text}")

        # Build seed entity list (only entities present in graph)
        seed_entities = []
        for entity_text, entity_type in entities_claim:
            entity_node = f"entity::{entity_type}::{entity_text}"
            if self.graph.has_node(entity_node):
                seed_entities.append(entity_node)
            else:
                self.logger.debug(f"Entity not in graph: {entity_node}")

        if not seed_entities:
            self.logger.warning("No entities from claim found in graph")
            return {}

        self.logger.info(f"Found {len(seed_entities)} seed entities in graph")

        # Propagate relevance through entity graph
        entity_relevance = self._propagate_relevance_entities(
            seed_entities=seed_entities,
            relevance_threshold=self.entity_relevance_threshold
        )

        self.logger.info(f"Relevance propagated to {len(entity_relevance)} entities")

        # Score chunks based on entity relevance and MENTIONS edge weights (TF-IDF)
        scores = defaultdict(float)

        for entity_node, relevance in entity_relevance.items():
            # Find chunks connected via MENTIONS edges
            for neighbor in self.graph.neighbors(entity_node):
                edge_data = self.graph[entity_node][neighbor]

                if (edge_data.get('edge_type') == 'MENTIONS' and
                    self.graph.nodes[neighbor].get('type') == 'chunk'):

                    # Get TF-IDF weight from MENTIONS edge
                    tf_idf_weight = edge_data.get('weight', 1.0)

                    # Chunk score = entity relevance × TF-IDF weight
                    chunk_score = relevance * tf_idf_weight

                    # Accumulate scores (chunks can be mentioned by multiple entities)
                    scores[neighbor] += chunk_score

        # Normalize scores
        if scores:
            min_score = min(scores.values())
            max_score = max(scores.values())
            if max_score > min_score:
                scores = {
                    k: (v - min_score) / (max_score - min_score)
                    for k, v in scores.items()
                }

        self.logger.info(f"Found {len(scores)} chunks via entity retrieval")
        if scores:
            self.logger.info(f"Top score: {max(scores.values()):.3f}")

        return dict(scores)

    def retrieve_topic_distribution(self, claim: str) -> Dict[str, float]:
        """
        Step 4: Topic distribution similarity using Jensen-Shannon divergence.

        Compares the topic probability distribution of the claim with each chunk's
        topic distribution. Uses Jensen-Shannon divergence (symmetric, bounded [0,1]).

        Args:
            claim: Claim text to verify

        Returns:
            Dictionary mapping chunk_id to distribution similarity score (normalized)
        """
        self.logger.info("\n--- Step 4: Topic Distribution Retrieval ---")

        # Get topic distribution for claim
        _, claim_topic_probs = self.topic_model.transform([claim])
        claim_distribution = claim_topic_probs[0]

        self.logger.info(f"Claim topic distribution shape: {claim_distribution.shape}")

        # Show top topics in claim
        top_topics = np.argsort(claim_distribution)[-5:][::-1]
        self.logger.info("Top 5 topics in claim:")
        for topic_id in top_topics:
            if claim_distribution[topic_id] > 0.01:
                theme_label = self.graph.nodes.get(f"theme::{topic_id}", {}).get('label', f'topic_{topic_id}')
                self.logger.info(f"  Topic {topic_id} ({theme_label}): {claim_distribution[topic_id]:.3f}")

        # Compute Jensen-Shannon divergence for each chunk
        scores = {}
        for chunk_id, dist_idx in self.chunk_to_dist_idx.items():
            chunk_distribution = self.chunk_topic_distributions[dist_idx]

            # Jensen-Shannon divergence (returns value in [0, 1])
            # Lower divergence = more similar distributions
            js_divergence = jensenshannon(claim_distribution, chunk_distribution)

            # Convert to similarity score (1 - divergence)
            # So higher score = more similar
            similarity = 1.0 - js_divergence

            scores[chunk_id] = float(similarity)

        # Normalize scores (min-max normalization)
        if scores:
            min_score = min(scores.values())
            max_score = max(scores.values())
            if max_score > min_score:
                scores = {
                    k: (v - min_score) / (max_score - min_score)
                    for k, v in scores.items()
                }

        self.logger.info(f"Found {len(scores)} chunks with topic distribution scores")
        if scores:
            self.logger.info(f"Top score: {max(scores.values()):.3f}")
            self.logger.info(f"Mean score: {np.mean(list(scores.values())):.3f}")

        return scores

    def retrieve(self, claim: str) -> List[Dict]:
        """
        Hybrid retrieval: combine all four strategies.

        Args:
            claim: Claim text to verify

        Returns:
            List of result dictionaries sorted by final score, each containing:
            - chunk_id: str
            - doc_id: str
            - text: str
            - score_total: float (final weighted score)
            - score_sem: float (semantic score)
            - score_theme: float (thematic score)
            - score_entity: float (entity score)
            - score_topic_dist: float (topic distribution score)
            - claim: str (the original claim)
            - config: Dict (complete configuration used for retrieval, included in first result only)
            - retrieval_timestamp: str (ISO format timestamp, included in first result only)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"RETRIEVAL FOR CLAIM: {claim}")
        self.logger.info("=" * 60)

        # Step 1: Semantic retrieval
        S1 = self.retrieve_semantic(claim)

        # Step 2: Thematic retrieval
        S2 = self.retrieve_thematic(claim)

        # Step 3: Entity retrieval
        S3 = self.retrieve_entity(claim)

        # Step 4: Topic distribution retrieval
        S4 = self.retrieve_topic_distribution(claim)

        # Fusion: combine scores
        all_chunks = set(S1.keys()) | set(S2.keys()) | set(S3.keys()) | set(S4.keys())

        self.logger.info("\n--- Fusion ---")
        self.logger.info(f"Total unique chunks: {len(all_chunks)}")
        self.logger.info(f"Weights: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, delta={self.delta}")

        results = []
        for chunk_id in all_chunks:
            score_sem = S1.get(chunk_id, 0)
            score_theme = S2.get(chunk_id, 0)
            score_entity = S3.get(chunk_id, 0)
            score_topic_dist = S4.get(chunk_id, 0)

            # Final weighted score
            final_score = (
                self.alpha * score_sem +
                self.beta * score_theme +
                self.gamma * score_entity +
                self.delta * score_topic_dist
            )

            # Get document ID
            doc_id = self.graph.nodes[chunk_id].get('doc_id', 'unknown')
            text = self.graph.nodes[chunk_id].get('text', '')

            results.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'text': text,
                'score_total': final_score,
                'score_sem': score_sem,
                'score_theme': score_theme,
                'score_entity': score_entity,
                'score_topic_dist': score_topic_dist
            })

        # Sort by total score
        results.sort(key=lambda x: x['score_total'], reverse=True)

        # Return top-K
        top_results = results[:self.top_k]

        # Add claim to all results
        for result in top_results:
            result['claim'] = claim

        self.logger.info(f"\nTop {len(top_results)} results:")
        for i, result in enumerate(top_results[:5], 1):
            self.logger.info(
                f"{i}. {result['chunk_id']} (doc: {result['doc_id']}) | "
                f"Total: {result['score_total']:.3f} | "
                f"Sem: {result['score_sem']:.3f} | "
                f"Theme: {result['score_theme']:.3f} | "
                f"Entity: {result['score_entity']:.3f} | "
                f"TopicDist: {result['score_topic_dist']:.3f}"
            )

        # Display text previews of top 3 results
        self.logger.info("\nText previews of top 3 results:")
        for i, result in enumerate(top_results[:3], 1):
            text_preview = result['text'][:150].replace('\n', ' ')
            if len(result['text']) > 150:
                text_preview += "..."
            self.logger.info(f"{i}. [{result['chunk_id']}] {text_preview}")

        self.logger.info("=" * 60)

        return top_results
