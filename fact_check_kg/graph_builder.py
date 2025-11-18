"""
Graph construction module for Knowledge Graph.

This module handles the construction of the multi-level knowledge graph:
- Level 1: Documents → Chunks
- Level 2: Chunks → Themes (BERTopic)
- Level 3: Chunks → Entities (spaCy NER)
- Relations: Entity ↔ Entity (CO_OCCURS, SEMANTIC_SIM)
"""

import networkx as nx
import spacy
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import logging
from tqdm import tqdm
import time as time


from fact_check_kg.db_connector import GraphDatabaseConnector


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
        topic_prob_threshold: float = 0.15,
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
            topic_prob_threshold: Minimum probability to create HAS_TOPIC edge
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
        self.topic_prob_threshold = topic_prob_threshold

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
                timeout=300
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

        # Step 5: Add CO_OCCURS edges (Entity ↔ Entity)
        self.logger.info("\nStep 5: Adding CO_OCCURS edges (Entity ↔ Entity)...")
        self._add_entity_co_occurrences(chunk_to_entities)

        # Step 6: Add SEMANTIC_SIM edges (Entity ↔ Entity)
        self.logger.info("\nStep 6: Adding SEMANTIC_SIM edges (Entity ↔ Entity)...")
        self._add_entity_semantic_similarity()

        # Step 7: Add SEMANTIC_SIM edges (Theme ↔ Theme)
        self.logger.info("\nStep 7: Adding SEMANTIC_SIM edges (Theme ↔ Theme)...")
        self._add_theme_semantic_similarity(topic_model)

        # Step 8: Add CO_PRESENCE edges (Theme ↔ Theme)
        self.logger.info("\nStep 8: Adding CO_PRESENCE edges (Theme ↔ Theme)...")
        self._add_theme_co_presence(chunk_ids)

        # Log statistics and save detailed reports
        self._log_graph_statistics()
        self._save_statistics_reports(topic_model, topics, chunks)

        return self.graph

    def _add_documents_and_chunks(self, chunks: List[Dict]):
        """Add document and chunk nodes with CONTAINS edges."""
        doc_ids_seen = set()

        for chunk in tqdm(chunks, desc="Adding documents and chunks", disable=self.logger.level > logging.INFO):
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

        # Step 1: Create theme nodes
        for topic_id in tqdm(unique_topics, desc="Creating theme nodes", disable=self.logger.level > logging.INFO):
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

        # Step 2: Add HAS_TOPIC edges for all topics above threshold
        edges_added = 0
        for idx, chunk_id in enumerate(tqdm(chunk_ids, desc="Adding HAS_TOPIC edges", disable=self.logger.level > logging.INFO)):
            # For each chunk, check ALL topics
            if probabilities.ndim > 1:
                # probabilities shape: (n_chunks, n_topics)
                chunk_probs = probabilities[idx]
            else:
                # If only 1D array, skip (shouldn't happen with calculate_probabilities=True)
                continue

            # Iterate through all topics
            for topic_id in unique_topics:
                # Get probability for this chunk-topic pair
                prob = chunk_probs[topic_id]

                # Only create edge if probability is above threshold
                if prob >= self.topic_prob_threshold:
                    theme_node = f"theme::{topic_id}"

                    # Add edge with probability weight
                    self.graph.add_edge(
                        chunk_id,
                        theme_node,
                        edge_type="HAS_TOPIC",
                        weight=float(prob)
                    )
                    edges_added += 1

        self.logger.info(f"Added {edges_added} HAS_TOPIC edges (threshold: {self.topic_prob_threshold})")

    def _add_entities_and_mentions(self, chunks: List[Dict]) -> Dict[str, Set[str]]:
        """
        Extract entities and add MENTIONS edges with TF-IDF weights.

        Returns:
            Dictionary mapping chunk_id to set of entity nodes
        """
        chunk_to_entities = defaultdict(set)
        # Track entity mentions: {entity_node: {chunk_id: count}}
        entity_mentions_in_chunks = defaultdict(lambda: defaultdict(int))

        # Step 1: Extract all entities and count term frequency (TF)
        for chunk in tqdm(chunks, desc="Extracting entities", disable=self.logger.level > logging.INFO):
            chunk_id = chunk['chunk_id']
            text = chunk['text']

            # Extract entities
            entities = self.extract_entities(text)

            # Count entity occurrences in this chunk (for TF)
            entity_counts = defaultdict(int)
            for entity_text, entity_type in entities:
                entity_node = f"entity::{entity_type}::{entity_text}"
                entity_counts[entity_node] += 1
                chunk_to_entities[chunk_id].add(entity_node)

            # Store counts
            for entity_node, count in entity_counts.items():
                entity_mentions_in_chunks[entity_node][chunk_id] = count

        # Step 2: Calculate IDF (Inverse Document Frequency)
        total_chunks = len(chunks)
        entity_idf = {}

        for entity_node, chunks_with_entity in entity_mentions_in_chunks.items():
            # IDF = log(total_chunks / number of chunks containing this entity)
            idf = np.log(total_chunks / len(chunks_with_entity))
            entity_idf[entity_node] = idf

        # Step 3: Add entity nodes and MENTIONS edges with TF-IDF weights
        for entity_node in tqdm(entity_mentions_in_chunks.keys(), desc="Adding entities and MENTIONS edges", disable=self.logger.level > logging.INFO):
            # Extract entity info from node name
            parts = entity_node.split("::")
            entity_type = parts[1]
            entity_text = "::".join(parts[2:])  # Handle entity text with "::"

            # Add entity node (if not exists)
            if not self.graph.has_node(entity_node):
                self.graph.add_node(
                    entity_node,
                    type="entity",
                    entity_type=entity_type,
                    entity_text=entity_text
                )

            # Add MENTIONS edges with TF-IDF weight
            idf = entity_idf[entity_node]
            for chunk_id, tf in entity_mentions_in_chunks[entity_node].items():
                # TF-IDF = TF * IDF
                tf_idf = tf * idf

                self.graph.add_edge(
                    chunk_id,
                    entity_node,
                    edge_type="MENTIONS",
                    weight=float(tf_idf),
                    tf=int(tf),
                    idf=float(idf)
                )

        return chunk_to_entities

    def _add_entity_co_occurrences(self, chunk_to_entities: Dict[str, Set[str]]):
        """
        Add CO_OCCURS edges between entities with sophisticated weighting.

        Weight calculation:
        - Raw co-occurrence count (higher = stronger relationship)
        - Normalized by entity global frequencies to penalize omnipresent entities
        - Formula: weight = co_occur_count * (1 / sqrt(freq(e1) * freq(e2)))

        This ensures:
        - Frequent co-occurrences get high weights
        - But entities that appear everywhere get penalized
        """
        co_occur_counter = Counter()
        entity_document_frequency = Counter()

        # Step 1: Count co-occurrences and entity frequencies across documents
        doc_to_entities = defaultdict(set)

        # Group chunks by document
        for chunk_id, entities in chunk_to_entities.items():
            # Extract doc_id from chunk_id (format: "chunk::doc_id::chunk_idx")
            doc_id = chunk_id.split("::")[1]
            doc_to_entities[doc_id].update(entities)

        # Count entity frequencies at document level
        for doc_id, entities in doc_to_entities.items():
            for entity in entities:
                entity_document_frequency[entity] += 1

        # Count co-occurrences at document level (not chunk level)
        for doc_id, entities in tqdm(doc_to_entities.items(), desc="Counting entity co-occurrences by document", disable=self.logger.level > logging.INFO):
            entities_list = list(entities)
            for i in range(len(entities_list)):
                for j in range(i + 1, len(entities_list)):
                    pair = tuple(sorted([entities_list[i], entities_list[j]]))
                    co_occur_counter[pair] += 1

        # Step 2: Calculate weights
        if not co_occur_counter:
            self.logger.warning("No entity co-occurrences found")
            return

        max_co_occur = max(co_occur_counter.values())

        # Add edges above threshold with weighted score
        edges_added = 0
        for (entity1, entity2), co_occur_count in tqdm(co_occur_counter.items(), desc="Adding CO_OCCURS edges", disable=self.logger.level > logging.INFO):
            if co_occur_count >= self.co_occur_threshold:
                # Entity frequencies (number of documents they appear in)
                freq1 = entity_document_frequency[entity1]
                freq2 = entity_document_frequency[entity2]

                # Inverse frequency weight (penalize entities that appear everywhere)
                # Using sqrt to soften the penalty
                inv_freq_weight = 1.0 / np.sqrt(freq1 * freq2)

                # Final weight: co-occurrence count weighted by entity rarity
                # High co-occurrence = high weight
                # But penalized if entities appear in too many documents
                weight = co_occur_count * inv_freq_weight

                self.graph.add_edge(
                    entity1,
                    entity2,
                    edge_type="CO_OCCURS",
                    weight=float(weight),
                    co_occur_count=int(co_occur_count),
                    entity1_freq=int(freq1),
                    entity2_freq=int(freq2),
                    inv_freq_weight=float(inv_freq_weight)
                )
                edges_added += 1

        self.logger.info(f"Added {edges_added} CO_OCCURS edges (threshold: {self.co_occur_threshold}, max_co_occur: {max_co_occur})")

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
        for i in tqdm(range(len(entity_nodes)), desc="Adding SEMANTIC_SIM edges", disable=self.logger.level > logging.INFO):
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

        self.logger.info(f"Added {edge_count} SEMANTIC_SIM edges (threshold: {self.semantic_sim_threshold})")

    def _add_theme_semantic_similarity(self, topic_model: BERTopic):
        """Add SEMANTIC_SIM edges between themes based on topic word embeddings."""
        # Get all theme nodes
        theme_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'theme'
        ]

        if len(theme_nodes) < 2:
            self.logger.warning("Less than 2 themes, skipping theme similarity")
            return

        # For each theme, create a representation from top keywords
        theme_representations = []
        for theme_node in theme_nodes:
            topic_id = self.graph.nodes[theme_node]['topic_id']
            topic_words = topic_model.get_topic(topic_id)

            if not topic_words:
                # Fallback: use label
                theme_representations.append(self.graph.nodes[theme_node].get('label', ''))
            else:
                # Concatenate top 5 keywords
                keywords = [word for word, _ in topic_words[:5]]
                theme_representations.append(' '.join(keywords))

        # Embed theme representations
        self.logger.info(f"Embedding {len(theme_representations)} themes...")
        theme_embeddings = self.embedding_model.encode(
            theme_representations,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # Compute similarity matrix
        similarity_matrix = np.matmul(theme_embeddings, theme_embeddings.T)
        np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity

        # Add edges above threshold (use same threshold as entities for consistency)
        edge_count = 0
        for i in tqdm(range(len(theme_nodes)), desc="Adding theme SEMANTIC_SIM edges", disable=self.logger.level > logging.INFO):
            for j in range(i + 1, len(theme_nodes)):
                sim = similarity_matrix[i, j]
                if sim >= self.semantic_sim_threshold:
                    self.graph.add_edge(
                        theme_nodes[i],
                        theme_nodes[j],
                        edge_type="SEMANTIC_SIM",
                        weight=float(sim)
                    )
                    edge_count += 1

        self.logger.info(f"Added {edge_count} theme SEMANTIC_SIM edges (threshold: {self.semantic_sim_threshold})")

    def _add_theme_co_presence(self, chunk_ids: List[str]):
        """
        Add CO_PRESENCE edges between themes based on co-occurrence in chunks.

        Similar to entity CO_OCCURS but for themes that appear together in chunks.
        Uses same weighting strategy: co_presence_count * (1 / sqrt(freq1 * freq2))
        """
        theme_co_presence_counter = Counter()
        theme_chunk_frequency = Counter()

        # Get all chunks and their connected themes
        chunk_to_themes = defaultdict(set)

        for chunk_id in tqdm(chunk_ids, desc="Collecting chunk-theme pairs", disable=self.logger.level > logging.INFO):
            # Find all themes connected to this chunk via HAS_TOPIC edges
            if not self.graph.has_node(chunk_id):
                continue

            for neighbor in self.graph.neighbors(chunk_id):
                edge_data = self.graph[chunk_id][neighbor]
                if (edge_data.get('edge_type') == 'HAS_TOPIC' and
                    self.graph.nodes[neighbor].get('type') == 'theme'):
                    chunk_to_themes[chunk_id].add(neighbor)
                    theme_chunk_frequency[neighbor] += 1

        # Count co-presences
        for chunk_id, themes in tqdm(chunk_to_themes.items(), desc="Counting theme co-presences", disable=self.logger.level > logging.INFO):
            themes_list = list(themes)
            for i in range(len(themes_list)):
                for j in range(i + 1, len(themes_list)):
                    pair = tuple(sorted([themes_list[i], themes_list[j]]))
                    theme_co_presence_counter[pair] += 1

        # Calculate weights and add edges
        if not theme_co_presence_counter:
            self.logger.warning("No theme co-presences found")
            return

        max_co_presence = max(theme_co_presence_counter.values())
        edges_added = 0

        for (theme1, theme2), co_presence_count in tqdm(theme_co_presence_counter.items(), desc="Adding CO_PRESENCE edges", disable=self.logger.level > logging.INFO):
            if co_presence_count >= self.co_occur_threshold:  # Use same threshold
                # Theme frequencies (number of chunks they appear in)
                freq1 = theme_chunk_frequency[theme1]
                freq2 = theme_chunk_frequency[theme2]

                # Inverse frequency weight
                inv_freq_weight = 1.0 / np.sqrt(freq1 * freq2)

                # Final weight
                weight = co_presence_count * inv_freq_weight

                self.graph.add_edge(
                    theme1,
                    theme2,
                    edge_type="CO_PRESENCE",
                    weight=float(weight),
                    co_presence_count=int(co_presence_count),
                    theme1_freq=int(freq1),
                    theme2_freq=int(freq2),
                    inv_freq_weight=float(inv_freq_weight)
                )
                edges_added += 1

        self.logger.info(f"Added {edges_added} CO_PRESENCE edges (threshold: {self.co_occur_threshold}, max_co_presence: {max_co_presence})")

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

    def _save_statistics_reports(
        self,
        topic_model: BERTopic,
        topics: List[int],
        chunks: List[Dict]
    ):
        """
        Save detailed statistics reports to outputs folder.

        Args:
            topic_model: Trained BERTopic model
            topics: List of topic assignments
            chunks: Original chunks data
        """
        import json
        from pathlib import Path
        from datetime import datetime

        # Create outputs directory
        stats_dir = Path("outputs/statistics")
        stats_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. General graph statistics
        general_stats = {
            "timestamp": timestamp,
            "total_documents": len(set(c['doc_id'] for c in chunks)),
            "total_chunks": len(chunks),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": {},
            "edges_by_type": {}
        }

        # Count nodes by type
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            general_stats["nodes_by_type"][node_type] = general_stats["nodes_by_type"].get(node_type, 0) + 1

        # Count edges by type
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            general_stats["edges_by_type"][edge_type] = general_stats["edges_by_type"].get(edge_type, 0) + 1

        # Save general statistics
        with open(stats_dir / f"general_stats_{timestamp}.json", 'w') as f:
            json.dump(general_stats, f, indent=2)
        self.logger.info(f"General statistics saved to: {stats_dir}/general_stats_{timestamp}.json")

        # 2. Topic statistics with document counts
        topic_stats = {
            "timestamp": timestamp,
            "total_topics": len(set(topics)) - (1 if -1 in topics else 0),
            "topics": []
        }

        unique_topics = set(topics)
        if -1 in unique_topics:
            unique_topics.remove(-1)

        for topic_id in sorted(unique_topics):
            topic_words = topic_model.get_topic(topic_id)
            if not topic_words:
                continue

            # Count documents for this topic
            topic_chunks = [i for i, t in enumerate(topics) if t == topic_id]
            topic_doc_ids = set(chunks[i]['doc_id'] for i in topic_chunks if i < len(chunks))

            # Get theme label from graph
            theme_node = f"theme::{topic_id}"
            theme_label = self.graph.nodes[theme_node].get('label', '') if self.graph.has_node(theme_node) else ''

            topic_info = {
                "topic_id": topic_id,
                "label": theme_label,
                "num_chunks": len(topic_chunks),
                "num_documents": len(topic_doc_ids),
                "top_words": [{"word": word, "score": float(score)} for word, score in topic_words[:10]],
                "keywords": [word for word, score in topic_words[:5]]
            }
            topic_stats["topics"].append(topic_info)

        # Save topic statistics
        with open(stats_dir / f"topic_stats_{timestamp}.json", 'w') as f:
            json.dump(topic_stats, f, indent=2)
        self.logger.info(f"Topic statistics saved to: {stats_dir}/topic_stats_{timestamp}.json")

        # 3. Entity statistics
        entity_stats = {
            "timestamp": timestamp,
            "total_entities": 0,
            "entities_by_type": {},
            "top_entities_by_degree": []
        }

        entities = [(n, self.graph.nodes[n], self.graph.degree(n))
                   for n, d in self.graph.nodes(data=True)
                   if d.get('type') == 'entity']

        entity_stats["total_entities"] = len(entities)

        # Count by entity type
        for node, data, degree in entities:
            entity_type = data.get('entity_type', 'unknown')
            entity_stats["entities_by_type"][entity_type] = entity_stats["entities_by_type"].get(entity_type, 0) + 1

        # Top entities by degree
        sorted_entities = sorted(entities, key=lambda x: x[2], reverse=True)[:50]
        for node, data, degree in sorted_entities:
            entity_stats["top_entities_by_degree"].append({
                "entity": data.get('entity_text', ''),
                "type": data.get('entity_type', ''),
                "degree": degree
            })

        # Save entity statistics
        with open(stats_dir / f"entity_stats_{timestamp}.json", 'w') as f:
            json.dump(entity_stats, f, indent=2)
        self.logger.info(f"Entity statistics saved to: {stats_dir}/entity_stats_{timestamp}.json")

        # 4. Create human-readable summary
        summary = f"""
KNOWLEDGE GRAPH STATISTICS REPORT
Generated: {timestamp}
{'=' * 80}

GENERAL STATISTICS
-----------------
Total Documents: {general_stats['total_documents']}
Total Chunks: {general_stats['total_chunks']}
Total Nodes: {general_stats['total_nodes']}
Total Edges: {general_stats['total_edges']}

Nodes by Type:
"""
        for node_type, count in general_stats["nodes_by_type"].items():
            summary += f"  - {node_type}: {count}\n"

        summary += "\nEdges by Type:\n"
        for edge_type, count in general_stats["edges_by_type"].items():
            summary += f"  - {edge_type}: {count}\n"

        summary += f"\n\nTOPIC STATISTICS\n{'-' * 17}\n"
        summary += f"Total Topics: {topic_stats['total_topics']}\n\n"

        for topic in topic_stats["topics"]:
            summary += f"Topic {topic['topic_id']}: {topic['label']}\n"
            summary += f"  Chunks: {topic['num_chunks']}, Documents: {topic['num_documents']}\n"
            summary += f"  Keywords: {', '.join(topic['keywords'])}\n\n"

        summary += f"\nENTITY STATISTICS\n{'-' * 17}\n"
        summary += f"Total Entities: {entity_stats['total_entities']}\n\n"
        summary += "Entities by Type:\n"
        for entity_type, count in entity_stats["entities_by_type"].items():
            summary += f"  - {entity_type}: {count}\n"

        summary += "\n\nTop 20 Entities by Degree:\n"
        for i, entity in enumerate(entity_stats["top_entities_by_degree"][:20], 1):
            summary += f"  {i}. {entity['entity']} ({entity['type']}) - Degree: {entity['degree']}\n"

        # Save summary
        with open(stats_dir / f"summary_{timestamp}.txt", 'w') as f:
            f.write(summary)
        self.logger.info(f"Summary report saved to: {stats_dir}/summary_{timestamp}.txt")

    def save_to_postgres(
        self,
        graph_id: str,
        host: str = "localhost",
        port: int = 5432,
        database: str = "knowledge_graph",
        user: str = "postgres",
        password: str = "postgres",
        metadata: dict = None
    ):
        """
        Save the graph to PostgreSQL database with timestamped tables.

        Args:
            graph_id: Graph ID (timestamp) for creating versioned tables
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username
            password: Password
            metadata: Optional metadata dictionary to store
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"SAVING GRAPH TO POSTGRESQL (Graph ID: {graph_id})")
        self.logger.info("=" * 60)

        # Initialize database connector with graph_id
        db = GraphDatabaseConnector(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            graph_id=graph_id,
            logger=self.logger
        )

        try:
            # Connect to database
            db.connect()

            # Create tables if they don't exist
            db.create_tables()

            # Save the graph with metadata
            db.save_graph(self.graph, metadata=metadata)

            self.logger.info("=" * 60)
            self.logger.info(f"GRAPH SAVED SUCCESSFULLY TO POSTGRESQL")
            self.logger.info(f"Tables: {db.graph_table}, {db.nodes_table}")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Failed to save graph to PostgreSQL: {e}")
            raise
        finally:
            db.disconnect()
