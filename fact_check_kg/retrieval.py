"""
Retrieval module for fact-checking using the Knowledge Graph.

This module implements a hybrid retrieval strategy combining:
1. Semantic retrieval (embedding similarity)
2. Thematic retrieval (via topic model and graph traversal)
3. Entity-based retrieval (via entity extraction and graph traversal)
"""

import networkx as nx
import numpy as np
import spacy
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import logging


class HybridRetriever:
    """
    Hybrid retrieval system for fact-checking claims.

    Combines three retrieval strategies:
    - Semantic: Direct embedding similarity
    - Thematic: Topic classification + graph traversal
    - Entity: Entity extraction + graph traversal (3 paths)

    The final score is a weighted combination of all three.
    """

    def __init__(
        self,
        graph: nx.Graph,
        topic_model: BERTopic,
        embedding_model: SentenceTransformer,
        nlp: spacy.Language,
        entity_types: List[str],
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        top_k: int = 10,
        theme_threshold: float = 0.1,
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
            alpha: Weight for semantic score (default: 0.4)
            beta: Weight for thematic score (default: 0.3)
            gamma: Weight for entity score (default: 0.3)
            top_k: Number of chunks to return
            theme_threshold: Minimum confidence for theme detection
            logger: Optional logger instance
        """
        self.graph = graph
        self.topic_model = topic_model
        self.embedding_model = embedding_model
        self.nlp = nlp
        self.entity_types = entity_types

        # Weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Parameters
        self.top_k = top_k
        self.theme_threshold = theme_threshold

        self.logger = logger or logging.getLogger(__name__)

        # Pre-compute chunk embeddings for efficiency
        self.logger.info("Pre-computing chunk embeddings...")
        self._precompute_chunk_embeddings()

    def _precompute_chunk_embeddings(self):
        """Pre-compute embeddings for all chunks in the graph."""
        self.chunk_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'chunk'
        ]

        self.chunk_texts = [
            self.graph.nodes[n]['text']
            for n in self.chunk_nodes
        ]

        self.chunk_embeddings = self.embedding_model.encode(
            self.chunk_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.logger.info(f"Pre-computed embeddings for {len(self.chunk_nodes)} chunks")

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
        Step 2: Thematic retrieval using topic model and graph traversal.

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

        # Traverse graph via themes
        scores = defaultdict(float)

        for theme_id, confidence_claim in detected_themes:
            theme_node = f"theme::{theme_id}"

            if not self.graph.has_node(theme_node):
                continue

            # Find chunks connected to this theme
            for neighbor in self.graph.neighbors(theme_node):
                edge_data = self.graph[theme_node][neighbor]

                # Only consider HAS_TOPIC edges pointing to chunks
                if (edge_data.get('edge_type') == 'HAS_TOPIC' and
                    self.graph.nodes[neighbor].get('type') == 'chunk'):

                    prob_chunk = edge_data['weight']

                    # Score = confidence in claim × probability in chunk
                    score = confidence_claim * prob_chunk

                    # Accumulate (if chunk has multiple matching themes)
                    scores[neighbor] += score

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
        Step 3: Entity-based retrieval using graph traversal.

        Two paths:
        - Direct: entity → MENTIONS → chunk (score = 1.0)
        - Co-occurrence: entity → CO_OCCURS → entity2 → MENTIONS → chunk (score = w * 0.8)

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

        scores = defaultdict(float)

        # Get max co-occur weight for normalization
        max_co_occur = 1
        for u, v, d in self.graph.edges(data=True):
            if d.get('edge_type') == 'CO_OCCURS':
                max_co_occur = max(max_co_occur, d.get('weight', 1))

        for entity_text, entity_type in entities_claim:
            entity_node = f"entity::{entity_type}::{entity_text}"

            if not self.graph.has_node(entity_node):
                self.logger.debug(f"Entity not in graph: {entity_node}")
                continue

            # PATH 1: Direct (entity → MENTIONS → chunk)
            for neighbor in self.graph.neighbors(entity_node):
                edge_data = self.graph[entity_node][neighbor]

                if (edge_data.get('edge_type') == 'MENTIONS' and
                    self.graph.nodes[neighbor].get('type') == 'chunk'):

                    # Direct match: maximum score
                    scores[neighbor] = max(scores[neighbor], 1.0)

            # PATH 2: Via co-occurrence (entity → CO_OCCURS → entity2 → MENTIONS → chunk)
            for co_entity in self.graph.neighbors(entity_node):
                edge_data = self.graph[entity_node][co_entity]

                if (edge_data.get('edge_type') == 'CO_OCCURS' and
                    self.graph.nodes[co_entity].get('type') == 'entity'):

                    weight_co = edge_data.get('weight', 1)

                    # Find chunks mentioning co-entity
                    for chunk in self.graph.neighbors(co_entity):
                        edge_data2 = self.graph[co_entity][chunk]

                        if (edge_data2.get('edge_type') == 'MENTIONS' and
                            self.graph.nodes[chunk].get('type') == 'chunk'):

                            # Score with penalty for 2-hop path
                            score = (weight_co / max_co_occur) * 0.8
                            scores[chunk] = max(scores[chunk], score)

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

    def retrieve(self, claim: str) -> List[Dict]:
        """
        Hybrid retrieval: combine all three strategies.

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

        # Fusion: combine scores
        all_chunks = set(S1.keys()) | set(S2.keys()) | set(S3.keys())

        self.logger.info("\n--- Fusion ---")
        self.logger.info(f"Total unique chunks: {len(all_chunks)}")
        self.logger.info(f"Weights: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")

        results = []
        for chunk_id in all_chunks:
            score_sem = S1.get(chunk_id, 0)
            score_theme = S2.get(chunk_id, 0)
            score_entity = S3.get(chunk_id, 0)

            # Final weighted score
            final_score = (
                self.alpha * score_sem +
                self.beta * score_theme +
                self.gamma * score_entity
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
                'score_entity': score_entity
            })

        # Sort by total score
        results.sort(key=lambda x: x['score_total'], reverse=True)

        # Return top-K
        top_results = results[:self.top_k]

        self.logger.info(f"\nTop {len(top_results)} results:")
        for i, result in enumerate(top_results[:5], 1):
            self.logger.info(
                f"{i}. {result['chunk_id']} (doc: {result['doc_id']}) | "
                f"Total: {result['score_total']:.3f} | "
                f"Sem: {result['score_sem']:.3f} | "
                f"Theme: {result['score_theme']:.3f} | "
                f"Entity: {result['score_entity']:.3f}"
            )

        self.logger.info("=" * 60)

        return top_results
