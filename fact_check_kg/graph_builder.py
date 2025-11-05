"""
Graph construction module for Knowledge Graph.

This module handles the construction of the multi-level knowledge graph:
- Level 1: Documents → Chunks
- Level 2: Chunks → Themes (BERTopic)
- Level 3: Chunks → Entities (spaCy NER)
- Relations: Entity ↔ Entity (CO_OCCURS, SEMANTIC_SIM)
- Relations: Entity ↔ Theme (RELATED_TO)
"""

import networkx as nx
import spacy
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import logging


class KnowledgeGraphBuilder:
    """
    Builder class for constructing a multi-level knowledge graph.

    The graph contains:
    - Document nodes
    - Chunk nodes (with text)
    - Theme nodes (from BERTopic)
    - Entity nodes (from spaCy NER)

    Edge types:
    - CONTAINS: Document → Chunk
    - HAS_TOPIC: Chunk → Theme (weighted by probability)
    - MENTIONS: Chunk → Entity
    - RELATED_TO: Entity → Theme (weighted by accumulated probabilities)
    - CO_OCCURS: Entity ↔ Entity (weighted by co-occurrence count)
    - SEMANTIC_SIM: Entity ↔ Entity (weighted by cosine similarity)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        ner_model: str = "en_core_web_sm",
        entity_types: List[str] = None,
        co_occur_threshold: int = 2,
        semantic_sim_threshold: float = 0.6,
        entity_related_threshold: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Knowledge Graph Builder.

        Args:
            embedding_model: SentenceTransformer model name
            ner_model: spaCy model name for NER
            entity_types: List of entity types to extract (PERSON, ORG, etc.)
            co_occur_threshold: Minimum co-occurrences to create edge
            semantic_sim_threshold: Minimum similarity to create edge
            entity_related_threshold: Minimum weight for entity-theme edge
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Initialize models
        self.logger.info(f"Loading SentenceTransformer model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        self.logger.info(f"Loading spaCy NER model: {ner_model}")
        self.nlp = spacy.load(ner_model)

        # Configuration
        self.entity_types = entity_types or ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'NORP']
        self.co_occur_threshold = co_occur_threshold
        self.semantic_sim_threshold = semantic_sim_threshold
        self.entity_related_threshold = entity_related_threshold

        # Graph
        self.graph = nx.Graph()

        # Topic model (will be trained during build)
        self.topic_model = None

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text using spaCy.

        Args:
            text: Input text

        Returns:
            List of (entity_text, entity_type) tuples

        Example:
            >>> entities = builder.extract_entities("Trump visited France")
            >>> entities
            [("trump", "PERSON"), ("france", "GPE")]
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Filter for relevant entity types
            if ent.label_ in self.entity_types:
                # Normalize entity text
                entity_text = ent.text.strip().lower()
                if len(entity_text) > 2:  # Filter very short entities
                    entities.append((entity_text, ent.label_))

        return entities

    def build_topic_model(
        self,
        chunk_texts: List[str],
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None
    ) -> Tuple[BERTopic, List[int], np.ndarray]:
        """
        Build topic model using BERTopic.

        Args:
            chunk_texts: List of chunk texts
            min_topic_size: Minimum size for a topic
            nr_topics: Number of topics (None = auto)

        Returns:
            Tuple of (topic_model, topics, probabilities)
            - topics: List of topic IDs for each chunk (-1 = outlier)
            - probabilities: Array of shape (n_chunks, n_topics)

        Example:
            >>> topic_model, topics, probs = builder.build_topic_model(chunks)
            >>> probs.shape
            (100, 10)  # 100 chunks, 10 topics
        """
        self.logger.info("Training BERTopic model...")

        # Create vectorizer with stop words
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,
            ngram_range=(1, 3),  # Include trigrams
            max_features=1000     # More features
        )

        # Initialize BERTopic with same embedding model and vectorizer
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=False
        )

        # Fit and transform
        topics, probabilities = topic_model.fit_transform(chunk_texts)

        self.logger.info(f"Found {len(set(topics)) - 1} topics (excluding outliers)")

        self.topic_model = topic_model
        return topic_model, topics, probabilities

    def label_topic_with_llm(
        self,
        topic_words: List[Tuple[str, float]],
        llm_client=None
    ) -> str:
        """
        Label a topic using LLM (Llama via Ollama).

        Args:
            topic_words: List of (word, score) tuples from BERTopic
            llm_client: LLM client (not used, kept for compatibility)

        Returns:
            Topic label string generated by LLM

        Example:
            >>> words = [("gaza", 0.8), ("israel", 0.7), ("conflict", 0.6)]
            >>> label = builder.label_topic_with_llm(words)
            >>> label
            "middle east conflict"
        """
        # Extract top words
        top_words = [word for word, score in topic_words[:10]]

        # Try Ollama with Llama
        try:
            import requests

            prompt = f"""Based on these keywords, provide a short, clear topic label (2-4 words maximum):
Keywords: {', '.join(top_words)}

Topic label:"""

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.1:latest',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,
                        'num_predict': 20  # Short responses
                    }
                },
                timeout=10
            )

            if response.status_code == 200:
                label = response.json()['response'].strip()
                # Clean up the response
                label = label.replace('\n', ' ').strip()
                # Remove quotes if present
                label = label.strip('"\'')
                # Limit length
                if len(label) > 50:
                    label = ' '.join(label.split()[:4])

                self.logger.debug(f"LLM labeled topic: {top_words[:3]} -> '{label}'")
                return label.lower()

        except Exception as e:
            self.logger.warning(f"LLM labeling failed: {e}, using fallback")

        # Fallback: concatenate top 3 words
        label = " ".join(top_words[:3])
        return label

    def build_graph(
        self,
        chunks: List[Dict],
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None
    ) -> nx.Graph:
        """
        Build the complete knowledge graph.

        Args:
            chunks: List of chunk dictionaries from data_loader
            min_topic_size: Minimum topic size for BERTopic
            nr_topics: Number of topics (None = auto)

        Returns:
            NetworkX graph with all nodes and edges

        Example:
            >>> chunks = [{"chunk_id": "chunk::0::0", "doc_id": "0", ...}]
            >>> graph = builder.build_graph(chunks)
            >>> graph.number_of_nodes()
            150
        """
        self.logger.info("=" * 60)
        self.logger.info("BUILDING KNOWLEDGE GRAPH")
        self.logger.info("=" * 60)

        # Extract data
        chunk_texts = [c['text'] for c in chunks]
        chunk_ids = [c['chunk_id'] for c in chunks]
        doc_ids = [c['doc_id'] for c in chunks]

        # Step 1: Add document and chunk nodes
        self.logger.info("\nStep 1: Adding documents and chunks...")
        self._add_documents_and_chunks(chunks)

        # Step 2: Topic modeling
        self.logger.info("\nStep 2: Running topic modeling...")
        topic_model, topics, probabilities = self.build_topic_model(
            chunk_texts,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics
        )

        # Step 3: Add themes and HAS_TOPIC edges
        self.logger.info("\nStep 3: Adding themes and HAS_TOPIC edges...")
        self._add_themes_and_edges(topic_model, topics, probabilities, chunk_ids)

        # Step 4: Extract entities and add MENTIONS edges
        self.logger.info("\nStep 4: Extracting entities and adding MENTIONS edges...")
        chunk_to_entities = self._add_entities_and_mentions(chunks)

        # Step 5: Add RELATED_TO edges (Entity ↔ Theme)
        self.logger.info("\nStep 5: Adding RELATED_TO edges (Entity ↔ Theme)...")
        self._add_entity_theme_relations(chunk_to_entities, topics, probabilities, chunk_ids)

        # Step 6: Add CO_OCCURS edges (Entity ↔ Entity)
        self.logger.info("\nStep 6: Adding CO_OCCURS edges (Entity ↔ Entity)...")
        self._add_entity_co_occurrences(chunk_to_entities)

        # Step 7: Add SEMANTIC_SIM edges (Entity ↔ Entity)
        self.logger.info("\nStep 7: Adding SEMANTIC_SIM edges (Entity ↔ Entity)...")
        self._add_entity_semantic_similarity()

        # Log statistics
        self._log_graph_statistics()

        return self.graph

    def _add_documents_and_chunks(self, chunks: List[Dict]):
        """Add document and chunk nodes with CONTAINS edges."""
        doc_ids_seen = set()

        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            doc_id = chunk['doc_id']
            text = chunk['text']

            # Add document node (once per document)
            doc_node = f"doc::{doc_id}"
            if doc_node not in doc_ids_seen:
                self.graph.add_node(
                    doc_node,
                    type="document",
                    doc_id=doc_id
                )
                doc_ids_seen.add(doc_node)

            # Add chunk node
            self.graph.add_node(
                chunk_id,
                type="chunk",
                doc_id=doc_id,
                text=text,
                chunk_idx=chunk['chunk_idx']
            )

            # Add CONTAINS edge
            self.graph.add_edge(doc_node, chunk_id, edge_type="CONTAINS")

    def _add_themes_and_edges(
        self,
        topic_model: BERTopic,
        topics: List[int],
        probabilities: np.ndarray,
        chunk_ids: List[str]
    ):
        """Add theme nodes and HAS_TOPIC edges."""
        # Get unique topics (excluding -1 which is outlier)
        unique_topics = set(topics)
        if -1 in unique_topics:
            unique_topics.remove(-1)

        for topic_id in unique_topics:
            # Get topic words
            topic_words = topic_model.get_topic(topic_id)

            if not topic_words:
                continue

            # Label topic (with LLM or simple concatenation)
            label = self.label_topic_with_llm(topic_words)

            # Add theme node
            theme_node = f"theme::{topic_id}"
            self.graph.add_node(
                theme_node,
                type="theme",
                topic_id=topic_id,
                label=label,
                keywords=[w for w, s in topic_words[:10]]
            )

        # Add HAS_TOPIC edges
        for idx, (chunk_id, topic_id) in enumerate(zip(chunk_ids, topics)):
            if topic_id == -1:  # Skip outliers
                continue

            theme_node = f"theme::{topic_id}"

            # Get probability for this chunk-topic pair
            prob = probabilities[idx, topic_id] if probabilities.ndim > 1 else probabilities[idx]

            # Add edge with probability weight
            self.graph.add_edge(
                chunk_id,
                theme_node,
                edge_type="HAS_TOPIC",
                weight=float(prob)
            )

    def _add_entities_and_mentions(self, chunks: List[Dict]) -> Dict[str, Set[str]]:
        """
        Extract entities and add MENTIONS edges.

        Returns:
            Dictionary mapping chunk_id to set of entity nodes
        """
        chunk_to_entities = defaultdict(set)

        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            text = chunk['text']

            # Extract entities
            entities = self.extract_entities(text)

            for entity_text, entity_type in entities:
                entity_node = f"entity::{entity_type}::{entity_text}"

                # Add entity node (if not exists)
                if not self.graph.has_node(entity_node):
                    self.graph.add_node(
                        entity_node,
                        type="entity",
                        entity_type=entity_type,
                        entity_text=entity_text
                    )

                # Add MENTIONS edge
                self.graph.add_edge(
                    chunk_id,
                    entity_node,
                    edge_type="MENTIONS"
                )

                # Track for later processing
                chunk_to_entities[chunk_id].add(entity_node)

        return chunk_to_entities

    def _add_entity_theme_relations(
        self,
        chunk_to_entities: Dict[str, Set[str]],
        topics: List[int],
        probabilities: np.ndarray,
        chunk_ids: List[str]
    ):
        """Add RELATED_TO edges between entities and themes."""
        entity_theme_weights = defaultdict(float)

        # Accumulate weights
        for idx, (chunk_id, topic_id) in enumerate(zip(chunk_ids, topics)):
            if topic_id == -1:  # Skip outliers
                continue

            theme_node = f"theme::{topic_id}"
            prob = probabilities[idx, topic_id] if probabilities.ndim > 1 else probabilities[idx]

            # For each entity in this chunk
            for entity_node in chunk_to_entities[chunk_id]:
                pair = (entity_node, theme_node)
                entity_theme_weights[pair] += float(prob)

        # Add edges above threshold
        for (entity_node, theme_node), weight in entity_theme_weights.items():
            if weight >= self.entity_related_threshold:
                self.graph.add_edge(
                    entity_node,
                    theme_node,
                    edge_type="RELATED_TO",
                    weight=weight
                )

    def _add_entity_co_occurrences(self, chunk_to_entities: Dict[str, Set[str]]):
        """Add CO_OCCURS edges between entities."""
        co_occur_counter = Counter()

        # Count co-occurrences
        for chunk_id, entities in chunk_to_entities.items():
            entities_list = list(entities)
            for i in range(len(entities_list)):
                for j in range(i + 1, len(entities_list)):
                    pair = tuple(sorted([entities_list[i], entities_list[j]]))
                    co_occur_counter[pair] += 1

        # Add edges above threshold
        for (entity1, entity2), count in co_occur_counter.items():
            if count >= self.co_occur_threshold:
                self.graph.add_edge(
                    entity1,
                    entity2,
                    edge_type="CO_OCCURS",
                    weight=count
                )

    def _add_entity_semantic_similarity(self):
        """Add SEMANTIC_SIM edges between entities based on embeddings."""
        # Get all entity nodes
        entity_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'entity'
        ]

        if len(entity_nodes) < 2:
            return

        # Extract entity texts
        entity_texts = [
            self.graph.nodes[n]['entity_text']
            for n in entity_nodes
        ]

        # Embed entities (just the name, not context - Option B)
        self.logger.info(f"Embedding {len(entity_texts)} entities...")
        embeddings = self.embedding_model.encode(
            entity_texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # Compute similarity matrix
        similarity_matrix = np.matmul(embeddings, embeddings.T)
        np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity

        # Add edges above threshold
        edge_count = 0
        for i in range(len(entity_nodes)):
            for j in range(i + 1, len(entity_nodes)):
                sim = similarity_matrix[i, j]
                if sim >= self.semantic_sim_threshold:
                    self.graph.add_edge(
                        entity_nodes[i],
                        entity_nodes[j],
                        edge_type="SEMANTIC_SIM",
                        weight=float(sim)
                    )
                    edge_count += 1

        self.logger.info(f"Added {edge_count} SEMANTIC_SIM edges")

    def _log_graph_statistics(self):
        """Log detailed graph statistics."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("GRAPH STATISTICS")
        self.logger.info("=" * 60)

        # Node counts by type
        node_types = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            node_types[data.get('type', 'unknown')] += 1

        self.logger.info(f"\nTotal nodes: {self.graph.number_of_nodes()}")
        for ntype, count in sorted(node_types.items()):
            self.logger.info(f"  {ntype}: {count}")

        # Edge counts by type
        edge_types = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            edge_types[data.get('edge_type', 'unknown')] += 1

        self.logger.info(f"\nTotal edges: {self.graph.number_of_edges()}")
        for etype, count in sorted(edge_types.items()):
            self.logger.info(f"  {etype}: {count}")

        self.logger.info("=" * 60)
