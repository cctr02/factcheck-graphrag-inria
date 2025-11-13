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
        host: str = "localhost",
        port: int = 5432,
        database: str = "knowledge_graph",
        user: str = "postgres",
        password: str = "postgres",
        clear_existing: bool = True
    ):
        """
        Save the graph to PostgreSQL database.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username
            password: Password
            clear_existing: Whether to clear existing data before saving
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("SAVING GRAPH TO POSTGRESQL")
        self.logger.info("=" * 60)

        # Initialize database connector
        db = GraphDatabaseConnector(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            logger=self.logger
        )

        try:
            # Connect to database
            db.connect()

            # Create tables if they don't exist
            db.create_tables()

            # Clear existing data if requested
            if clear_existing:
                db.clear_graph()

            # Save the graph
            db.save_graph(self.graph)

            self.logger.info("=" * 60)
            self.logger.info("GRAPH SAVED SUCCESSFULLY TO POSTGRESQL")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Failed to save graph to PostgreSQL: {e}")
            raise
        finally:
            db.disconnect()
