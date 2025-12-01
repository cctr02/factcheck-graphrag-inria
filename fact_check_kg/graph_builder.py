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
from nltk.stem.snowball import SnowballStemmer


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
        temporal_max_year_gap: int = 10,
        temporal_decay_rate: float = 0.3,
        topic_keywords_initial: int = 15,
        topic_keywords_final: int = 10,
        llm_endpoint: str = "http://localhost:11434/api/generate",
        llm_model: str = "llama3.1:latest",
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 20,
        llm_timeout: int = 300,
        llm_label_max_length: int = 50,
        llm_label_max_words: int = 4,
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
            temporal_max_year_gap: Maximum year difference for temporal similarity
            temporal_decay_rate: Decay rate for exponential temporal similarity
            topic_keywords_initial: Number of keywords to retrieve initially
            topic_keywords_final: Number of keywords after deduplication
            llm_endpoint: LLM API endpoint for topic labeling
            llm_model: LLM model name
            llm_temperature: LLM temperature parameter
            llm_max_tokens: Maximum tokens for LLM response
            llm_timeout: LLM request timeout in seconds
            llm_label_max_length: Maximum length for topic labels
            llm_label_max_words: Maximum words for truncated labels
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

        # Temporal similarity configuration
        self.temporal_max_year_gap = temporal_max_year_gap
        self.temporal_decay_rate = temporal_decay_rate

        # Topic keyword processing configuration
        self.topic_keywords_initial = topic_keywords_initial
        self.topic_keywords_final = topic_keywords_final

        # Initialize NLTK stemmer
        self.stemmer = SnowballStemmer('english')

        # LLM configuration
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_timeout = llm_timeout
        self.llm_label_max_length = llm_label_max_length
        self.llm_label_max_words = llm_label_max_words

        # Graph
        self.graph = nx.Graph()

        # Topic model (will be trained during build)
        self.topic_model = None

    def _log_step_completion(self, step_name: str, start_time: float, count: int = None):
        """
        Log completion of a step with elapsed time.

        Args:
            step_name: Name of the step
            start_time: Start time from time.time()
            count: Optional count of items processed
        """
        elapsed = time.time() - start_time
        if count is not None:
            if elapsed > 0.01:  # Avoid division by zero for very fast operations
                self.logger.info(f"[DONE] {step_name} completed: {count} items in {elapsed:.2f}s ({count/elapsed:.1f} items/s)")
            else:
                self.logger.info(f"[DONE] {step_name} completed: {count} items in <0.01s")
        else:
            self.logger.info(f"[DONE] {step_name} completed in {elapsed:.2f}s")

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text using spaCy.

        For DATE entities, only extract the year (4-digit format).

        Args:
            text: Input text

        Returns:
            List of (entity_text, entity_type) tuples

        Example:
            >>> entities = builder.extract_entities("Trump visited France in 2019")
            >>> entities
            [("trump", "PERSON"), ("france", "GPE"), ("2019", "DATE")]
        """
        import re

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Filter for relevant entity types
            if ent.label_ in self.entity_types:
                entity_text = ent.text.strip().lower()

                # Special handling for DATE: extract only the year
                if ent.label_ == "DATE":
                    # Look for 4-digit year in the entity text
                    year_match = re.search(r'\b(19|20)\d{2}\b', entity_text)
                    if year_match:
                        year = year_match.group(0)
                        entities.append((year, "DATE"))
                    # Skip dates without a clear year
                else:
                    # For non-DATE entities, keep as is
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
        start_time = time.time()
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

        n_topics = len(set(topics)) - 1  # excluding outliers
        self._log_step_completion(f"Training BERTopic model ({n_topics} topics found)", start_time, len(chunk_texts))

        self.topic_model = topic_model
        return topic_model, topics, probabilities

    def _process_topic_keywords(
        self,
        topic_words: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Process topic keywords with stemming and n-gram deduplication.

        Args:
            topic_words: List of (word, score) from BERTopic

        Returns:
            List of (word, score) tuples, deduplicated and stemmed
        """
        if not topic_words:
            return []

        # Step 1: Stem and merge variants (take top N initially)
        stem_scores = {}

        for word, score in topic_words[:self.topic_keywords_initial]:
            # Stem using NLTK SnowballStemmer
            # For multi-word phrases, stem each word individually
            words = word.split()
            stemmed_words = [self.stemmer.stem(w.lower()) for w in words]
            stem = " ".join(stemmed_words)

            # Merge scores for same stem (sum the scores)
            if stem not in stem_scores:
                stem_scores[stem] = 0
            stem_scores[stem] += score

        # Step 2: Filter redundant n-grams
        # Strategy: Keep longer n-grams, remove shorter ones that are substrings
        # Sort by length (descending) then score (descending)
        sorted_items = sorted(
            stem_scores.items(),
            key=lambda x: (len(x[0].split()), x[1]),  # Sort by word count, then score
            reverse=True
        )

        filtered = []
        for stem, score in sorted_items:
            is_redundant = False

            # Check if this stem is a substring of any already kept stem
            for kept_stem, kept_score in filtered:
                # Check if current stem is contained in a kept stem
                # Use word boundary checks to avoid false positives
                stem_words = set(stem.split())
                kept_words = set(kept_stem.split())

                # If all words in current stem are in the kept stem, it's redundant
                if stem_words.issubset(kept_words):
                    is_redundant = True
                    break

            if not is_redundant:
                # Before adding, remove any kept stems that are substrings of current stem
                to_remove = []
                for kept_stem, kept_score in filtered:
                    kept_words = set(kept_stem.split())
                    stem_words = set(stem.split())

                    # If kept stem is subset of current stem, mark for removal
                    if kept_words.issubset(stem_words) and kept_stem != stem:
                        to_remove.append((kept_stem, kept_score))

                # Remove redundant kept items
                for item in to_remove:
                    filtered.remove(item)

                filtered.append((stem, score))

        # Step 3: Re-sort by score and take top N
        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
        final_count = min(max(len(filtered_sorted), 3), self.topic_keywords_final)
        top_keywords = filtered_sorted[:final_count]

        # Return with stemmed forms
        result = [(stem, score) for stem, score in top_keywords]

        self.logger.debug(f"Processed keywords: {len(topic_words)} -> {len(result)} (removed {len(topic_words) - len(result)} redundant)")

        return result

    def label_topic_with_llm(
        self,
        topic_words: List[Tuple[str, float]],
        use_llm: bool = False
    ) -> str:
        """
        Label a topic using processed keywords (or optionally LLM).

        Args:
            topic_words: List of (word, score) tuples from BERTopic
            use_llm: Whether to use LLM for labeling (default: False)

        Returns:
            Topic label string

        Example:
            >>> words = [("gaza", 0.8), ("israel", 0.7), ("conflict", 0.6)]
            >>> label = builder.label_topic_with_llm(words, use_llm=False)
            >>> label
            "gaza israel conflict"
        """
        # Process keywords: lemmatize and remove redundant n-grams
        processed_keywords = self._process_topic_keywords(topic_words)

        # Extract top words from processed keywords
        top_words = [word for word, score in processed_keywords]

        # If LLM is not requested, use fallback immediately
        if not use_llm:
            label = " ".join(top_words[:3])
            self.logger.debug(f"Keyword-based label: {label}")
            return label

        # Try Ollama with Llama (only if use_llm=True)
        try:
            import requests

            prompt = f"""Based on these keywords, provide a short, clear topic label (2-4 words maximum):
Keywords: {', '.join(top_words)}

Topic label:"""

            response = requests.post(
                self.llm_endpoint,
                json={
                    'model': self.llm_model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': self.llm_temperature,
                        'num_predict': self.llm_max_tokens
                    }
                },
                timeout=self.llm_timeout
            )

            if response.status_code == 200:
                label = response.json()['response'].strip()
                # Clean up the response
                label = label.replace('\n', ' ').strip()
                # Remove quotes if present
                label = label.strip('"\'')
                # Limit length
                if len(label) > self.llm_label_max_length:
                    label = ' '.join(label.split()[:self.llm_label_max_words])

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
        nr_topics: Optional[int] = None,
        use_llm_labels: bool = False
    ) -> nx.Graph:
        """
        Build the complete knowledge graph.

        Args:
            chunks: List of chunk dictionaries from data_loader
            min_topic_size: Minimum topic size for BERTopic
            nr_topics: Number of topics (None = auto)
            use_llm_labels: Whether to use LLM for topic labeling (default: False)

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
        self._add_themes_and_edges(topic_model, topics, probabilities, chunk_ids, use_llm_labels)

        # Step 4: Extract entities and add MENTIONS edges
        self.logger.info("\nStep 4: Extracting entities and adding MENTIONS edges...")
        chunk_to_entities = self._add_entities_and_mentions(chunks)

        # Step 5: Add CO_OCCURS edges (Entity ↔ Entity)
        self.logger.info("\nStep 5: Adding CO_OCCURS edges (Entity ↔ Entity)...")
        self._add_entity_co_occurrences(chunk_to_entities)

        # Step 6: Add SEMANTIC_SIM edges (Entity ↔ Entity, excluding DATEs)
        self.logger.info("\nStep 6: Adding SEMANTIC_SIM edges (Entity ↔ Entity)...")
        self._add_entity_semantic_similarity()

        # Step 6b: Add TEMPORAL_DIST edges (DATE ↔ DATE)
        self.logger.info("\nStep 6b: Adding TEMPORAL_DIST edges (DATE ↔ DATE)...")
        self._add_date_temporal_similarity(
            max_year_gap=self.temporal_max_year_gap,
            decay_rate=self.temporal_decay_rate
        )

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
        start_time = time.time()
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

        self._log_step_completion("Adding documents and chunks", start_time, len(chunks))

    def _add_themes_and_edges(
        self,
        topic_model: BERTopic,
        topics: List[int],
        probabilities: np.ndarray,
        chunk_ids: List[str],
        use_llm_labels: bool = False
    ):
        """Add theme nodes and HAS_TOPIC edges."""
        # Get unique topics (excluding -1 which is outlier)
        unique_topics = set(topics)
        if -1 in unique_topics:
            unique_topics.remove(-1)

        # Step 1: Create theme nodes
        start_time_themes = time.time()
        for topic_id in tqdm(unique_topics, desc="Creating theme nodes", disable=self.logger.level > logging.INFO):
            # Get topic words
            topic_words = topic_model.get_topic(topic_id)

            if not topic_words:
                continue

            # Process keywords: lemmatize and deduplicate n-grams
            processed_keywords = self._process_topic_keywords(topic_words)

            # Label topic (with LLM or simple concatenation) using processed keywords
            label = self.label_topic_with_llm(topic_words, use_llm=use_llm_labels)

            # Add theme node with processed keywords
            theme_node = f"theme::{topic_id}"
            self.graph.add_node(
                theme_node,
                type="theme",
                topic_id=topic_id,
                label=label,
                keywords=[w for w, s in processed_keywords]  # Use processed keywords
            )

        self._log_step_completion("Creating theme nodes", start_time_themes, len(unique_topics))

        # Step 2: Add HAS_TOPIC edges for all topics above threshold
        start_time_edges = time.time()
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

        self._log_step_completion("Adding HAS_TOPIC edges", start_time_edges, edges_added)

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
        start_time_extract = time.time()
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

        self._log_step_completion("Extracting entities", start_time_extract, len(chunks))

        # Step 2: Calculate IDF (Inverse Document Frequency)
        total_chunks = len(chunks)
        entity_idf = {}

        for entity_node, chunks_with_entity in entity_mentions_in_chunks.items():
            # IDF = log(total_chunks / number of chunks containing this entity)
            idf = np.log(total_chunks / len(chunks_with_entity))
            entity_idf[entity_node] = idf

        # Step 3: Calculate all TF-IDF weights first to find max for normalization
        all_tf_idf_weights = []
        for entity_node, chunks_with_counts in entity_mentions_in_chunks.items():
            idf = entity_idf[entity_node]
            for chunk_id, tf in chunks_with_counts.items():
                tf_idf = tf * idf
                all_tf_idf_weights.append(tf_idf)

        # Find max TF-IDF for normalization
        max_tf_idf = max(all_tf_idf_weights) if all_tf_idf_weights else 1.0
        self.logger.info(f"Max TF-IDF weight: {max_tf_idf:.4f}, normalizing to [0, 1]")

        # Step 4: Add entity nodes and MENTIONS edges with normalized weights
        start_time_mentions = time.time()
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

            # Add MENTIONS edges with normalized weight (0 to 1)
            idf = entity_idf[entity_node]
            for chunk_id, tf in entity_mentions_in_chunks[entity_node].items():
                # TF-IDF = TF * IDF
                tf_idf = tf * idf
                # Normalize to [0, 1]
                normalized_weight = tf_idf / max_tf_idf

                self.graph.add_edge(
                    chunk_id,
                    entity_node,
                    edge_type="MENTIONS",
                    weight=float(normalized_weight)
                )

        self._log_step_completion("Adding entities and MENTIONS edges", start_time_mentions, len(entity_mentions_in_chunks))

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
        start_time_count = time.time()
        for doc_id, entities in tqdm(doc_to_entities.items(), desc="Counting entity co-occurrences by document", disable=self.logger.level > logging.INFO):
            entities_list = list(entities)
            for i in range(len(entities_list)):
                for j in range(i + 1, len(entities_list)):
                    pair = tuple(sorted([entities_list[i], entities_list[j]]))
                    co_occur_counter[pair] += 1

        self._log_step_completion("Counting entity co-occurrences", start_time_count, len(doc_to_entities))

        # Step 2: Calculate weights
        if not co_occur_counter:
            self.logger.warning("No entity co-occurrences found")
            return

        # Add edges above threshold with weighted score
        start_time_add = time.time()
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

        self._log_step_completion("Adding CO_OCCURS edges", start_time_add, edges_added)

    def _add_entity_semantic_similarity(self):
        """Add SEMANTIC_SIM edges between non-DATE entities based on embeddings."""
        start_time = time.time()

        # Get all entity nodes EXCEPT DATE entities
        entity_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'entity' and d.get('entity_type') != 'DATE'
        ]

        if len(entity_nodes) < 2:
            self.logger.info("Skipping entity semantic similarity (< 2 non-DATE entities)")
            return

        # Extract entity texts
        entity_texts = [
            self.graph.nodes[n]['entity_text']
            for n in entity_nodes
        ]

        # Embed entities (just the name, not context - Option B)
        self.logger.info(f"Embedding {len(entity_texts)} non-DATE entities...")
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

        self._log_step_completion("Adding entity SEMANTIC_SIM edges", start_time, edge_count)

    def _add_date_temporal_similarity(self, max_year_gap: int = 10, decay_rate: float = 0.3):
        """
        Add TEMPORAL_DIST edges between year entities with exponential decay.

        Uses an exponential negative decay function:
        weight = exp(-decay_rate * year_diff)

        Where:
        - year_diff = |year2 - year1|
        - decay_rate controls how fast the weight decreases (default: 0.3)

        Examples (with decay_rate=0.3):
        - Same year (diff=0): weight = 1.0
        - 1 year diff: weight ≈ 0.74
        - 2 years diff: weight ≈ 0.55
        - 3 years diff: weight ≈ 0.41
        - 5 years diff: weight ≈ 0.22
        - 10 years diff: weight ≈ 0.05

        Args:
            max_year_gap: Maximum year difference to create edges (default: 10)
            decay_rate: Decay rate for exponential function (default: 0.3)
        """
        start_time = time.time()

        # Get all DATE entity nodes (which are now years)
        date_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'entity' and d.get('entity_type') == 'DATE'
        ]

        if len(date_nodes) < 2:
            self.logger.info("Skipping date temporal similarity (< 2 years)")
            return

        self.logger.info(f"Processing temporal similarity for {len(date_nodes)} year entities...")
        self.logger.info(f"Using exponential decay: weight = exp(-{decay_rate} * year_diff), max_gap={max_year_gap}")

        # Extract years
        year_nodes = {}
        for node in date_nodes:
            year_text = self.graph.nodes[node]['entity_text']
            try:
                year = int(year_text)
                year_nodes[year] = node
            except ValueError:
                continue

        years_list = list(year_nodes.keys())

        # Connect all years within max_year_gap with exponential decay
        edge_count = 0
        for i in range(len(years_list)):
            for j in range(i + 1, len(years_list)):
                year1 = years_list[i]
                year2 = years_list[j]
                year_diff = abs(year2 - year1)

                # Only connect if within max_year_gap
                if year_diff <= max_year_gap:
                    node1 = year_nodes[year1]
                    node2 = year_nodes[year2]

                    # Exponential decay: weight = exp(-decay_rate * year_diff)
                    weight = float(np.exp(-decay_rate * year_diff))

                    self.graph.add_edge(
                        node1,
                        node2,
                        edge_type="TEMPORAL_DIST",
                        weight=weight,
                        year_diff=year_diff
                    )
                    edge_count += 1

        self._log_step_completion("Adding TEMPORAL_DIST edges (exponential decay)", start_time, edge_count)

    def _add_theme_semantic_similarity(self, topic_model: BERTopic):
        """Add SEMANTIC_SIM edges between themes based on topic word embeddings."""
        start_time = time.time()

        # Get all theme nodes
        theme_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'theme'
        ]

        if len(theme_nodes) < 2:
            self.logger.info("Skipping theme semantic similarity (< 2 themes)")
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

        self._log_step_completion("Adding theme SEMANTIC_SIM edges", start_time, edge_count)

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

        start_time_collect = time.time()
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

        self._log_step_completion("Collecting chunk-theme pairs", start_time_collect, len(chunk_ids))

        # Count co-presences
        start_time_count = time.time()
        for chunk_id, themes in tqdm(chunk_to_themes.items(), desc="Counting theme co-presences", disable=self.logger.level > logging.INFO):
            themes_list = list(themes)
            for i in range(len(themes_list)):
                for j in range(i + 1, len(themes_list)):
                    pair = tuple(sorted([themes_list[i], themes_list[j]]))
                    theme_co_presence_counter[pair] += 1

        self._log_step_completion("Counting theme co-presences", start_time_count, len(chunk_to_themes))

        # Calculate weights and add edges
        if not theme_co_presence_counter:
            self.logger.warning("No theme co-presences found")
            return

        edges_added = 0
        start_time_add = time.time()

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

        self._log_step_completion("Adding CO_PRESENCE edges", start_time_add, edges_added)

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

            # Get theme label and processed keywords from graph
            theme_node = f"theme::{topic_id}"
            if self.graph.has_node(theme_node):
                theme_label = self.graph.nodes[theme_node].get('label', '')
                processed_keywords = self.graph.nodes[theme_node].get('keywords', [])
            else:
                theme_label = ''
                processed_keywords = []

            topic_info = {
                "topic_id": topic_id,
                "label": theme_label,
                "num_chunks": len(topic_chunks),
                "num_documents": len(topic_doc_ids),
                "top_words": [{"word": word, "score": float(score)} for word, score in topic_words[:10]],
                "keywords": processed_keywords  # Use processed keywords from graph node
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
            "top_entities_by_degree": [],
            "top_entities_by_weight": []
        }

        # Calculate degree and total weight for each entity
        entities = []
        for n, d in self.graph.nodes(data=True):
            if d.get('type') == 'entity':
                degree = self.graph.degree(n)
                # Calculate total weight (sum of all edge weights)
                total_weight = sum(
                    self.graph[n][neighbor].get('weight', 1.0)
                    for neighbor in self.graph.neighbors(n)
                )
                entities.append((n, d, degree, total_weight))

        entity_stats["total_entities"] = len(entities)

        # Count by entity type
        for node, data, degree, weight in entities:
            entity_type = data.get('entity_type', 'unknown')
            entity_stats["entities_by_type"][entity_type] = entity_stats["entities_by_type"].get(entity_type, 0) + 1

        # Top entities by degree
        sorted_by_degree = sorted(entities, key=lambda x: x[2], reverse=True)[:50]
        for node, data, degree, weight in sorted_by_degree:
            entity_stats["top_entities_by_degree"].append({
                "entity": data.get('entity_text', ''),
                "type": data.get('entity_type', ''),
                "degree": degree,
                "total_weight": round(weight, 4)
            })

        # Top entities by total weight
        sorted_by_weight = sorted(entities, key=lambda x: x[3], reverse=True)[:50]
        for node, data, degree, weight in sorted_by_weight:
            entity_stats["top_entities_by_weight"].append({
                "entity": data.get('entity_text', ''),
                "type": data.get('entity_type', ''),
                "degree": degree,
                "total_weight": round(weight, 4)
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
            summary += f"  {i}. {entity['entity']} ({entity['type']}) - Degree: {entity['degree']}, Weight: {entity['total_weight']}\n"

        summary += "\n\nTop 20 Entities by Total Weight:\n"
        for i, entity in enumerate(entity_stats["top_entities_by_weight"][:20], 1):
            summary += f"  {i}. {entity['entity']} ({entity['type']}) - Weight: {entity['total_weight']}, Degree: {entity['degree']}\n"

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
        metadata: dict = None,
        graph: nx.Graph = None
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
            graph: Optional graph to save (if None, uses self.graph)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"SAVING GRAPH TO POSTGRESQL (Graph ID: {graph_id})")
        self.logger.info("=" * 60)

        # Use provided graph or self.graph
        graph_to_save = graph if graph is not None else self.graph

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
            db.save_graph(graph_to_save, metadata=metadata)

            self.logger.info("=" * 60)
            self.logger.info(f"GRAPH SAVED SUCCESSFULLY TO POSTGRESQL")
            self.logger.info(f"Tables: {db.graph_table}, {db.nodes_table}")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Failed to save graph to PostgreSQL: {e}")
            raise
        finally:
            db.disconnect()
