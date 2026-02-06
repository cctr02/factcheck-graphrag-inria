"""
Graph construction module for Knowledge Graph.

This module handles the construction of the multi-level knowledge graph:
- Level 1: Documents → Chunks (CONTAINS edges)
- Level 2: Chunks → Themes (HAS_TOPIC edges, BERTopic with LLM labeling)
- Level 3: Chunks → Entities (MENTIONS edges, spaCy NER with TF-IDF weights)
- Level 4: Chunks → Keyphrases (HAS_KEYPHRASE edges, KeyBERT with MMR diversity)
- Level 5: Claims → Documents (REVIEWS edges, fact-checking claims)

Inter-node relations:
- Entity ↔ Entity: CO_OCCURS (PMI-weighted), SEMANTIC_SIM (embedding similarity)
- DATE ↔ DATE: TEMPORAL_DIST (exponential decay by year difference)
- Theme ↔ Theme: CO_PRESENCE (co-occurrence in chunks), SEMANTIC_SIM (keyword similarity)
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
from keybert import KeyBERT
import threading
import sys
import types


from fact_check_kg.core.db_connector import GraphDatabaseConnector


class KnowledgeGraphBuilder:
    """
    Builder class for constructing a multi-level knowledge graph for fact-checking.

    Node types:
    - document: Source documents with metadata (text, review_date, language)
    - chunk: Text chunks from documents with positional info
    - theme: Topics from BERTopic with LLM-generated labels and keywords
    - entity: Named entities from spaCy NER (PERSON, ORG, GPE, LOC, DATE, EVENT, NORP)
    - keyphrase: Key phrases extracted via KeyBERT with MMR diversity
    - claim: Fact-checking claims linked to documents via REVIEWS edges

    Edge types:
    - CONTAINS: Document → Chunk (hierarchical structure)
    - HAS_TOPIC: Chunk → Theme (weighted by topic probability, top-3 per chunk)
    - HAS_KEYPHRASE: Chunk → Keyphrase (weighted by KeyBERT relevance score)
    - MENTIONS: Chunk → Entity (weighted by TF-IDF with sigmoid normalization)
    - CO_OCCURS: Entity ↔ Entity (weighted by PMI - Pointwise Mutual Information)
    - SEMANTIC_SIM: Entity ↔ Entity and Theme ↔ Theme (weighted by embedding cosine similarity)
    - TEMPORAL_DIST: DATE ↔ DATE (weighted by exponential decay based on year difference)
    - CO_PRESENCE: Theme ↔ Theme (weighted by inverse frequency of co-occurrence in chunks)
    - REVIEWS: Claim → Document (links claims to their reviewed documents)
    """

    def __init__(
        self,
        embedding_model: str,
        ner_model: str,
        entity_types: List[str],
        co_occur_threshold: int,
        semantic_sim_threshold: float,
        topic_prob_threshold: float,
        temporal_max_year_gap: int,
        temporal_decay_rate: float,
        topic_keywords_initial: int,
        topic_keywords_final: int,
        llm_endpoint: str,
        llm_model: str,
        # LLM batch labeling configuration
        llm_batch_prompt_template: str,
        llm_batch_size_default: int,
        llm_batch_size_min: int,
        llm_batch_size_max: int,
        llm_batch_size_step: int,
        llm_label_min_words: int,
        llm_label_max_words: int,
        llm_label_max_length: int,
        llm_max_keywords_per_topic: int,
        llm_retry_count: int,
        llm_retry_temperature: float,
        llm_batch_timeout: int,
        llm_success_streak_threshold: int,
        llm_hysteresis_decrease_factor: float,
        llm_failure_strategy: str,
        keyphrase_top_n: int,
        keyphrase_ngram_range: Tuple[int, int],
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
            llm_batch_prompt_template: Prompt template with {{topics_json}} placeholder
            llm_batch_size_default: Starting batch size for topic labeling
            llm_batch_size_min: Minimum batch size (floor)
            llm_batch_size_max: Maximum batch size (cap)
            llm_batch_size_step: Step for increasing batch size
            llm_label_min_words: Minimum words in label
            llm_label_max_words: Maximum words in label
            llm_label_max_length: Maximum character length for labels
            llm_max_keywords_per_topic: Number of keywords to send per topic
            llm_retry_count: Number of retries on failure
            llm_retry_temperature: Temperature for retry attempts
            llm_batch_timeout: Timeout for batch requests in seconds
            llm_success_streak_threshold: Consecutive successes before increasing batch size
            llm_hysteresis_decrease_factor: Factor to decrease batch size on failure
            llm_failure_strategy: "binary_split" or "single_fallback"
            keyphrase_top_n: Number of keyphrases to extract per chunk
            keyphrase_ngram_range: N-gram range for keyphrase extraction (min, max)
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Initialize models
        self.logger.info(f"Loading SentenceTransformer model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        self.logger.info(f"Loading spaCy NER model: {ner_model}")
        self.nlp = spacy.load(ner_model)

        # Configuration
        self.entity_types = entity_types
        self.co_occur_threshold = co_occur_threshold
        self.semantic_sim_threshold = semantic_sim_threshold
        self.topic_prob_threshold = topic_prob_threshold

        # Temporal similarity configuration
        self.temporal_max_year_gap = temporal_max_year_gap
        self.temporal_decay_rate = temporal_decay_rate

        # Topic keyword processing configuration
        self.topic_keywords_initial = topic_keywords_initial
        self.topic_keywords_final = topic_keywords_final

        # Initialize NLTK lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # LLM configuration
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model

        # LLM batch labeling configuration
        self.llm_batch_prompt_template = llm_batch_prompt_template
        self.llm_batch_size_default = llm_batch_size_default
        self.llm_batch_size_min = llm_batch_size_min
        self.llm_batch_size_max = llm_batch_size_max
        self.llm_batch_size_step = llm_batch_size_step
        self.llm_label_min_words = llm_label_min_words
        self.llm_label_max_words = llm_label_max_words
        self.llm_label_max_length = llm_label_max_length
        self.llm_max_keywords_per_topic = llm_max_keywords_per_topic
        self.llm_retry_count = llm_retry_count
        self.llm_retry_temperature = llm_retry_temperature
        self.llm_batch_timeout = llm_batch_timeout
        self.llm_success_streak_threshold = llm_success_streak_threshold
        self.llm_hysteresis_decrease_factor = llm_hysteresis_decrease_factor
        self.llm_failure_strategy = llm_failure_strategy

        # Auto-tuning state for batch labeling
        self._batch_size_current = llm_batch_size_default
        self._success_streak = 0

        # Graph
        self.graph = nx.Graph()

        # Topic model (will be trained during build)
        self.topic_model = None

        # KeyBERT for keyphrase extraction (reuses the same embedding model)
        self.logger.info("Initializing KeyBERT for keyphrase extraction...")
        self.keybert_model = KeyBERT(model=self.embedding_model)
        self.keyphrase_top_n = keyphrase_top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range

        # Build timing statistics
        self.build_timings = {}
        self.build_start_time = None

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

    def _show_clustering_progress(self, stop_event, n_samples):
        """
        Display a simulated progress bar during HDBSCAN clustering.

        Args:
            stop_event: Threading event to stop the progress bar
            n_samples: Number of samples to cluster
        """
        # Estimate time based on data size (~0.5-2 seconds per 1000 samples)
        estimated_time = max(30, int(n_samples * 0.001 * 60))  # in seconds

        with tqdm(total=100, desc="HDBSCAN Clustering", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]') as pbar:
            start = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - start
                # Simulated progress based on estimated time
                progress = min(95, int((elapsed / estimated_time) * 100))
                pbar.n = progress
                pbar.refresh()
                time.sleep(0.5)
            # Complete to 100% when done
            pbar.n = 100
            pbar.refresh()

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
        self.logger.info(f"Computing embeddings for {len(chunk_texts)} chunks...")

        # Create vectorizer with stop words
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        from hdbscan import HDBSCAN

        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,
            ngram_range=(1, 3),  # Include trigrams
            max_features=1000     # More features
        )

        # Custom UMAP with verbose output
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            verbose=True  # Show UMAP progress
        )

        # Create HDBSCAN model
        # prediction_data=True required for calculate_probabilities (multi-topic assignment)
        # core_dist_n_jobs=1 reduces memory spikes (sequential instead of parallel)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,  # Required for multi-topic probabilities
            core_dist_n_jobs=1  # Reduces memory spikes
        )

        # Initialize BERTopic with custom components
        # calculate_probabilities=True enables multi-topic assignment
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics,
            calculate_probabilities=True,  # Enable probabilities for each chunk-topic
            verbose=True  # Enable verbose mode for progress
        )

        # Fit and transform with progress logging
        self.logger.info(f"Starting BERTopic pipeline on {len(chunk_texts)} text chunks...")
        self.logger.info("  → Stage 1/4: Computing embeddings (GPU-accelerated)...")
        self.logger.info("  → Stage 2/4: UMAP dimensionality reduction (kNN-based)...")
        self.logger.info("  → Stage 3/4: HDBSCAN clustering (CPU, typically 30s-2min)...")

        # Wrapper to show message after HDBSCAN
        original_extract_topics = topic_model._extract_topics

        def extract_topics_with_logging(*args, **kwargs):
            self.logger.info("  → Stage 4/5: Computing c-TF-IDF representations (CPU-intensive, may take 1-5min)...")
            self.logger.info("              This step is working even if GPU usage is 0% - please be patient...")
            return original_extract_topics(*args, **kwargs)

        topic_model._extract_topics = extract_topics_with_logging

        topics, probabilities = topic_model.fit_transform(chunk_texts)

        # Restore original method to allow pickling
        topic_model._extract_topics = original_extract_topics

        self.logger.info("  → Stage 5/5: Topic probabilities computed!")
        self.logger.info("✓ BERTopic pipeline completed!")

        n_topics = len(set(topics)) - 1  # excluding outliers
        n_outliers = sum(1 for t in topics if t == -1)
        outlier_pct = (n_outliers / len(topics)) * 100 if topics else 0
        self.logger.info(f"  → Outliers (topic -1): {n_outliers}/{len(topics)} chunks ({outlier_pct:.1f}%)")
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

        # Step 1: take top N initial keywords
        candidates = topic_words[:self.topic_keywords_initial]

        # Step 2: substring deduplication (keep longer / more specific)
        sorted_words = sorted(candidates, key=lambda x: len(x[0]), reverse=True)

        filtered = []
        for word, score in sorted_words:
            word_lower = word.lower()
            # skip if substring of an already-kept word
            if any(word_lower in kept_word.lower() for kept_word, _ in filtered):
                continue
            filtered.append((word, score))

        # Step 3: lemmatization + score merging for equivalent forms
        lemma_scores: Dict[str, float] = {}
        for word, score in filtered:
            tokens = word.lower().split()
            lemmas = [self.lemmatizer.lemmatize(t) for t in tokens]
            lemma_phrase = " ".join(lemmas)
            lemma_scores[lemma_phrase] = lemma_scores.get(lemma_phrase, 0.0) + score

        # Step 4: sort by merged score descending
        final_sorted = sorted(lemma_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 5: keep top N final
        top_keywords = final_sorted[:self.topic_keywords_final]

        self.logger.debug(f"Processed keywords: {len(topic_words)} -> {len(top_keywords)}")
        return top_keywords

    def _normalize_label(self, label: str) -> str:
        """
        Normalize a topic label.

        Args:
            label: Raw label string

        Returns:
            Normalized label string
        """
        # Clean up the response
        label = label.replace('\n', ' ').strip()
        # Remove quotes if present
        label = label.strip('"\'')
        # Lowercase
        label = label.lower()
        # Limit length
        if len(label) > self.llm_label_max_length:
            label = ' '.join(label.split()[:self.llm_label_max_words])
        return label

    def _validate_label(self, label: str) -> bool:
        """
        Validate a label against constraints.

        Args:
            label: Label to validate

        Returns:
            True if valid, False otherwise
        """
        if not label:
            return False
        word_count = len(label.split())
        if word_count < self.llm_label_min_words:
            return False
        if word_count > self.llm_label_max_words:
            return False
        if len(label) > self.llm_label_max_length:
            return False
        return True

    def _parse_batch_response(
        self,
        response_text: str,
        expected_ids: Set[int]
    ) -> Tuple[Dict[int, str], Set[int]]:
        """
        Parse and validate batch LLM response.

        Args:
            response_text: Raw response text from LLM
            expected_ids: Set of topic IDs that should be in response

        Returns:
            Tuple of (labels_dict, missing_ids) where:
            - labels_dict: Dict mapping topic_id -> normalized label
            - missing_ids: Set of topic IDs not found or with invalid labels
        """
        import json
        import re

        labels = {}
        missing_ids = set(expected_ids)

        # Try to extract JSON from response (may have extra text)
        # Look for {"labels": ...} pattern
        json_match = re.search(r'\{[^{}]*"labels"\s*:\s*\[[^\]]*\][^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            # Try to find any JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if not json_match:
            self.logger.debug(f"No JSON found in response: {response_text[:200]}")
            return labels, missing_ids

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON parse error: {e}, response: {response_text[:200]}")
            return labels, missing_ids

        # Extract labels array
        labels_array = data.get('labels', [])
        if not isinstance(labels_array, list):
            self.logger.debug(f"'labels' is not an array: {type(labels_array)}")
            return labels, missing_ids

        for item in labels_array:
            if not isinstance(item, dict):
                continue

            topic_id = item.get('id')
            label = item.get('label', '')

            if topic_id is None:
                continue

            # Normalize and validate
            normalized_label = self._normalize_label(str(label))

            if topic_id in expected_ids and self._validate_label(normalized_label):
                labels[topic_id] = normalized_label
                missing_ids.discard(topic_id)
            else:
                self.logger.debug(f"Invalid label for topic {topic_id}: '{label}'")

        return labels, missing_ids

    def _call_llm_batch(
        self,
        topics_data: List[Dict],
        temperature: float
    ) -> str:
        """
        Call LLM with a batch of topics.

        Args:
            topics_data: List of {"id": topic_id, "keywords": [...]} dicts
            temperature: LLM temperature parameter

        Returns:
            Raw response text from LLM

        Raises:
            RuntimeError: If LLM call fails
        """
        import requests
        import json

        # Build prompt from template
        topics_json = json.dumps(topics_data, ensure_ascii=False)
        prompt = self.llm_batch_prompt_template.replace('{{topics_json}}', topics_json)

        try:
            response = requests.post(
                self.llm_endpoint,
                json={
                    'model': self.llm_model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': temperature,
                        'num_predict': 50 * len(topics_data)  # ~50 tokens per topic
                    }
                },
                timeout=self.llm_batch_timeout
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:200]}")

            return response.json()['response']

        except requests.exceptions.Timeout:
            raise RuntimeError(f"LLM batch request timed out after {self.llm_batch_timeout}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"LLM batch labeling failed (connection error): {e}. "
                f"Ensure Ollama is running at {self.llm_endpoint}"
            ) from e

    def _adjust_batch_size(self, success: bool):
        """
        Adjust batch size based on success/failure (auto-tuning).

        Args:
            success: Whether the last batch succeeded
        """
        old_size = self._batch_size_current

        if success:
            self._success_streak += 1
            if self._success_streak >= self.llm_success_streak_threshold:
                # Increase batch size
                new_size = min(
                    self._batch_size_current + self.llm_batch_size_step,
                    self.llm_batch_size_max
                )
                if new_size != self._batch_size_current:
                    self._batch_size_current = new_size
                    self.logger.info(f"[BATCH AUTO-TUNE] Increased batch size: {old_size} -> {new_size}")
                self._success_streak = 0  # Reset streak after increase
        else:
            self._success_streak = 0
            # Decrease batch size using hysteresis factor
            new_size = max(
                int(self._batch_size_current * self.llm_hysteresis_decrease_factor),
                self.llm_batch_size_min
            )
            if new_size != self._batch_size_current:
                self._batch_size_current = new_size
                self.logger.info(f"[BATCH AUTO-TUNE] Decreased batch size: {old_size} -> {new_size}")

    def _label_single_topic(
        self,
        topic_id: int,
        keywords: List[str]
    ) -> str:
        """
        Label a single topic (fallback for failed batches).

        Args:
            topic_id: Topic ID
            keywords: List of keywords for the topic

        Returns:
            Topic label string

        Raises:
            RuntimeError: If labeling fails after retries
        """
        topics_data = [{"id": topic_id, "keywords": keywords}]

        for attempt in range(self.llm_retry_count + 1):
            temperature = self.llm_retry_temperature if attempt > 0 else 0.3
            try:
                response_text = self._call_llm_batch(topics_data, temperature)
                labels, _ = self._parse_batch_response(response_text, {topic_id})

                if topic_id in labels:
                    return labels[topic_id]

            except RuntimeError as e:
                self.logger.debug(f"Single topic {topic_id} attempt {attempt + 1} failed: {e}")

        raise RuntimeError(f"Failed to label topic {topic_id} after {self.llm_retry_count + 1} attempts")

    def label_topics_batch(
        self,
        topics_to_label: List[Tuple[int, List[Tuple[str, float]]]]
    ) -> Dict[int, str]:
        """
        Label multiple topics in batches with auto-tuning.

        This method implements:
        - Batched LLM requests for efficiency
        - Auto-tuning of batch size based on success/failure
        - Retry logic with stricter settings
        - Binary split or single-fallback on persistent failures

        Args:
            topics_to_label: List of (topic_id, topic_words) tuples where
                           topic_words is List[(word, score)]

        Returns:
            Dictionary mapping topic_id -> label string

        Raises:
            RuntimeError: If labeling fails completely
        """
        if not topics_to_label:
            return {}

        # Prepare all topics data
        all_topics_data = []
        for topic_id, topic_words in topics_to_label:
            # Process and limit keywords
            processed = self._process_topic_keywords(topic_words)
            keywords = [w for w, s in processed[:self.llm_max_keywords_per_topic]]
            all_topics_data.append({
                "id": topic_id,
                "keywords": keywords
            })

        labels = {}
        pending = list(all_topics_data)

        self.logger.info(f"Labeling {len(pending)} topics in batches (starting batch size: {self._batch_size_current})")

        while pending:
            # Take a batch
            batch_size = min(self._batch_size_current, len(pending))
            batch = pending[:batch_size]
            pending = pending[batch_size:]

            expected_ids = {t["id"] for t in batch}
            batch_labels = {}
            batch_missing = set(expected_ids)
            success = False

            # Try the batch with retries
            for attempt in range(self.llm_retry_count + 1):
                temperature = self.llm_retry_temperature if attempt > 0 else 0.3

                try:
                    self.logger.debug(f"Batch attempt {attempt + 1}: {len(batch)} topics, temp={temperature}")
                    response_text = self._call_llm_batch(batch, temperature)
                    batch_labels, batch_missing = self._parse_batch_response(response_text, expected_ids)

                    if not batch_missing:
                        # Complete success
                        success = True
                        break
                    else:
                        self.logger.debug(f"Batch partial: got {len(batch_labels)}/{len(batch)}, missing: {batch_missing}")

                except RuntimeError as e:
                    self.logger.debug(f"Batch attempt {attempt + 1} failed: {e}")
                    continue

            # Update labels with what we got
            labels.update(batch_labels)

            # Handle missing topics
            if batch_missing:
                self.logger.debug(f"Handling {len(batch_missing)} missing topics with strategy: {self.llm_failure_strategy}")

                if self.llm_failure_strategy == "binary_split" and len(batch_missing) > 1:
                    # Re-add missing topics to pending for retry with smaller batch
                    missing_data = [t for t in batch if t["id"] in batch_missing]
                    pending = missing_data + pending
                    success = False
                else:
                    # Single fallback: label each missing topic individually
                    for topic_data in batch:
                        if topic_data["id"] in batch_missing:
                            try:
                                label = self._label_single_topic(
                                    topic_data["id"],
                                    topic_data["keywords"]
                                )
                                labels[topic_data["id"]] = label
                            except RuntimeError as e:
                                raise RuntimeError(
                                    f"Failed to label topic {topic_data['id']} even with single fallback: {e}"
                                ) from e

            # Adjust batch size based on outcome
            self._adjust_batch_size(success)

            self.logger.debug(f"Progress: {len(labels)}/{len(all_topics_data)} topics labeled")

        self.logger.info(f"Batch labeling complete: {len(labels)} topics labeled")
        return labels

    def build_graph(
        self,
        chunks: List[Dict],
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None,
        claims: Optional[List[Dict]] = None,
        claim_to_docs: Optional[Dict[str, List[str]]] = None,
        doc_metadata: Optional[Dict[str, Dict]] = None
    ) -> Tuple[nx.Graph, np.ndarray, List[str], Dict[str, Dict]]:
        """
        Build the complete knowledge graph through a 10-step pipeline.

        Pipeline steps:
        1. Add document and chunk nodes (CONTAINS edges)
        2. Train BERTopic model for topic modeling
        3. Add theme nodes with LLM labels and HAS_TOPIC edges (top-3 per chunk)
        4. Extract entities and add MENTIONS edges (TF-IDF weighted)
        4b. Extract keyphrases and add HAS_KEYPHRASE edges (KeyBERT)
        5. Add CO_OCCURS edges between entities (PMI-weighted)
        6. Add SEMANTIC_SIM edges between entities (embedding similarity)
        6b. Add TEMPORAL_DIST edges between DATE entities (exponential decay)
        7. Add SEMANTIC_SIM edges between themes (keyword embedding similarity)
        8. Add CO_PRESENCE edges between themes (co-occurrence in chunks)
        9. Add claim nodes and REVIEWS edges (if claims provided)
        10. Pre-compute claim retrieval results for Strategy 5 (if claims provided)

        Args:
            chunks: List of chunk dictionaries with keys:
                chunk_id, doc_id, text, chunk_idx, review_date, language
            min_topic_size: Minimum topic size for BERTopic clustering
            nr_topics: Number of topics (None = auto-detect)
            claims: Optional list of claim dictionaries with keys:
                claim_id, text, rating, claimant, claimant_norm, review_date, language
            claim_to_docs: Optional mapping of claim_id to list of doc_ids
            doc_metadata: Optional mapping of doc_id to metadata dict with keys:
                text (title + body), review_date, language

        Returns:
            Tuple of (graph, topic_probabilities, chunk_ids, claim_retrieval_results):
            - graph: NetworkX graph with all nodes and edges
            - topic_probabilities: numpy array of shape (n_chunks, n_topics)
            - chunk_ids: list of chunk IDs matching probability matrix rows
            - claim_retrieval_results: dict mapping claim_id to pre-computed results
        """
        self.logger.info("=" * 60)
        self.logger.info("BUILDING KNOWLEDGE GRAPH")
        self.logger.info("=" * 60)

        # Initialize timing
        self.build_start_time = time.time()
        self.build_timings = {}

        # Extract data
        chunk_texts = [c['text'] for c in chunks]
        chunk_ids = [c['chunk_id'] for c in chunks]
        doc_ids = [c['doc_id'] for c in chunks]

        # Step 1: Add document and chunk nodes
        self.logger.info("\nStep 1: Adding documents and chunks...")
        step_start = time.time()
        self._add_documents_and_chunks(chunks, doc_metadata)
        self.build_timings["Step 1: Documents & Chunks"] = time.time() - step_start

        # Step 2: Topic modeling
        self.logger.info("\nStep 2: Running topic modeling...")
        step_start = time.time()
        topic_model, topics, probabilities = self.build_topic_model(
            chunk_texts,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics
        )
        self.build_timings["Step 2: Topic Modeling"] = time.time() - step_start

        # Step 3: Add themes and HAS_TOPIC edges (requires LLM for labeling)
        self.logger.info("\nStep 3: Adding themes and HAS_TOPIC edges...")
        step_start = time.time()
        self._add_themes_and_edges(topic_model, topics, probabilities, chunk_ids)
        self.build_timings["Step 3: Themes & HAS_TOPIC"] = time.time() - step_start

        # Step 4: Extract entities and add MENTIONS edges
        self.logger.info("\nStep 4: Extracting entities and adding MENTIONS edges...")
        step_start = time.time()
        chunk_to_entities = self._add_entities_and_mentions(chunks)
        self.build_timings["Step 4: Entities & MENTIONS"] = time.time() - step_start

        # Step 4b: Extract keyphrases and add HAS_KEYPHRASE edges
        self.logger.info("\nStep 4b: Extracting keyphrases and adding HAS_KEYPHRASE edges...")
        step_start = time.time()
        self._add_keyphrases(chunks)
        self.build_timings["Step 4b: Keyphrases"] = time.time() - step_start

        # Step 5: Add CO_OCCURS edges (Entity ↔ Entity)
        self.logger.info("\nStep 5: Adding CO_OCCURS edges (Entity ↔ Entity)...")
        step_start = time.time()
        self._add_entity_co_occurrences(chunk_to_entities)
        self.build_timings["Step 5: CO_OCCURS"] = time.time() - step_start

        # Step 6: Add SEMANTIC_SIM edges (Entity ↔ Entity, excluding DATEs)
        self.logger.info("\nStep 6: Adding SEMANTIC_SIM edges (Entity ↔ Entity)...")
        step_start = time.time()
        self._add_entity_semantic_similarity()
        self.build_timings["Step 6: Entity SEMANTIC_SIM"] = time.time() - step_start

        # Step 6b: Add TEMPORAL_DIST edges (DATE ↔ DATE)
        self.logger.info("\nStep 6b: Adding TEMPORAL_DIST edges (DATE ↔ DATE)...")
        step_start = time.time()
        self._add_date_temporal_similarity(
            max_year_gap=self.temporal_max_year_gap,
            decay_rate=self.temporal_decay_rate
        )
        self.build_timings["Step 6b: TEMPORAL_DIST"] = time.time() - step_start

        # Step 7: Add SEMANTIC_SIM edges (Theme ↔ Theme)
        self.logger.info("\nStep 7: Adding SEMANTIC_SIM edges (Theme ↔ Theme)...")
        step_start = time.time()
        self._add_theme_semantic_similarity(topic_model)
        self.build_timings["Step 7: Theme SEMANTIC_SIM"] = time.time() - step_start

        # Step 8: Add CO_PRESENCE edges (Theme ↔ Theme)
        self.logger.info("\nStep 8: Adding CO_PRESENCE edges (Theme ↔ Theme)...")
        step_start = time.time()
        self._add_theme_co_presence(chunk_ids)
        self.build_timings["Step 8: CO_PRESENCE"] = time.time() - step_start

        # Step 9: Add claim nodes and REVIEWS edges (Claim → Document)
        if claims and claim_to_docs:
            self.logger.info("\nStep 9: Adding claims and REVIEWS edges (Claim → Document)...")
            step_start = time.time()
            self._add_claims_and_edges(claims, claim_to_docs)
            self.build_timings["Step 9: Claims & REVIEWS"] = time.time() - step_start

            # Step 10: Pre-compute claim retrieval (run 4 strategies for each claim)
            self.logger.info("\nStep 10: Pre-computing claim retrieval results...")
            step_start = time.time()
            claim_retrieval_results = self._precompute_claim_retrieval(
                claims, topic_model, probabilities, chunk_ids
            )
            self.build_timings["Step 10: Claim Pre-Retrieval"] = time.time() - step_start
        else:
            claim_retrieval_results = {}

        # Calculate total build time
        total_build_time = time.time() - self.build_start_time

        # Log statistics and save detailed reports
        self._log_graph_statistics()
        self._save_statistics_reports(topic_model, topics, chunks)
        self._save_timing_report(total_build_time)

        # Return graph, probabilities, chunk_ids, and claim_retrieval_results
        return self.graph, probabilities, chunk_ids, claim_retrieval_results

    def _add_documents_and_chunks(self, chunks: List[Dict], doc_metadata: Optional[Dict[str, Dict]] = None):
        """Add document and chunk nodes with CONTAINS edges.

        Args:
            chunks: List of chunk dictionaries
            doc_metadata: Optional mapping of doc_id to metadata dict with keys:
                text (title\\nbody), review_date, language
        """
        start_time = time.time()
        doc_ids_seen = set()

        for chunk in tqdm(chunks, desc="Adding documents and chunks", disable=self.logger.level > logging.INFO):
            chunk_id = chunk['chunk_id']
            doc_id = chunk['doc_id']
            text = chunk['text']

            # Add document node (once per document)
            doc_node = f"doc::{doc_id}"
            if doc_node not in doc_ids_seen:
                # Get document metadata if available
                meta = doc_metadata.get(doc_id, {}) if doc_metadata else {}
                doc_text = meta.get("text", "")
                review_date = meta.get("review_date", "")
                language = meta.get("language", "")

                self.graph.add_node(
                    doc_node,
                    type="document",
                    doc_id=doc_id,
                    text=doc_text,
                    review_date=review_date,
                    language=language
                )
                doc_ids_seen.add(doc_node)

            # Get chunk metadata
            chunk_review_date = chunk.get('review_date', '')
            chunk_language = chunk.get('language', '')

            # Add chunk node
            self.graph.add_node(
                chunk_id,
                type="chunk",
                doc_id=doc_id,
                text=text,
                chunk_idx=chunk['chunk_idx'],
                review_date=chunk_review_date,
                language=chunk_language
            )

            # Add CONTAINS edge
            self.graph.add_edge(doc_node, chunk_id, edge_type="CONTAINS")

        self._log_step_completion("Adding documents and chunks", start_time, len(chunks))

    def _add_claims_and_edges(
        self,
        claims: List[Dict],
        claim_to_docs: Dict[str, List[str]]
    ):
        """Add claim nodes and REVIEWS edges linking claims to documents.

        Args:
            claims: List of claim dictionaries with keys:
                claim_id, text, rating, claimant, claimant_norm, review_date, language
            claim_to_docs: Mapping of claim_id to list of doc_ids
        """
        start_time = time.time()
        edges_added = 0

        for claim in tqdm(claims, desc="Adding claims and REVIEWS edges", disable=self.logger.level > logging.INFO):
            claim_id = claim['claim_id']
            claim_node = f"claim::{claim_id}"

            # Add claim node with all attributes
            self.graph.add_node(
                claim_node,
                type="claim",
                claim_id=claim_id,
                text=claim.get('text', ''),
                rating=claim.get('rating', ''),
                claimant=claim.get('claimant', ''),
                claimant_norm=claim.get('claimant_norm', ''),
                review_date=claim.get('review_date', ''),
                language=claim.get('language', '')
            )

            # Add REVIEWS edges to linked documents
            doc_ids = claim_to_docs.get(claim_id, [])
            for doc_id in doc_ids:
                doc_node = f"doc::{doc_id}"
                if self.graph.has_node(doc_node):
                    self.graph.add_edge(
                        claim_node,
                        doc_node,
                        edge_type="REVIEWS"
                    )
                    edges_added += 1

        self._log_step_completion("Adding claims and REVIEWS edges", start_time, edges_added)

    def _add_themes_and_edges(
        self,
        topic_model: BERTopic,
        topics: List[int],
        probabilities: np.ndarray,
        chunk_ids: List[str]
    ):
        """Add theme nodes and HAS_TOPIC edges using batched LLM labeling."""
        # Get unique topics (excluding -1 which is outlier)
        unique_topics = set(topics)
        if -1 in unique_topics:
            unique_topics.remove(-1)

        # Step 1: Collect all topics to label
        start_time_themes = time.time()
        topics_to_label = []
        topic_keywords_map = {}  # Store processed keywords for later

        for topic_id in sorted(unique_topics):
            topic_words = topic_model.get_topic(topic_id)
            if not topic_words:
                continue
            topics_to_label.append((topic_id, topic_words))
            # Pre-process and store keywords
            processed_keywords = self._process_topic_keywords(topic_words)
            topic_keywords_map[topic_id] = processed_keywords

        # Step 2: Label all topics in batches (with auto-tuning)
        self.logger.info(f"Labeling {len(topics_to_label)} topics using batched LLM requests...")
        labels = self.label_topics_batch(topics_to_label)

        # Step 3: Create theme nodes with labels
        for topic_id in sorted(unique_topics):
            if topic_id not in labels:
                self.logger.warning(f"No label for topic {topic_id}, skipping node creation")
                continue

            processed_keywords = topic_keywords_map.get(topic_id, [])
            theme_node = f"theme::{topic_id}"
            self.graph.add_node(
                theme_node,
                type="theme",
                topic_id=topic_id,
                label=labels[topic_id],
                keywords=[w for w, _ in processed_keywords]
            )

        self._log_step_completion("Creating theme nodes (batched)", start_time_themes, len(unique_topics))

        # Step 2: Add HAS_TOPIC edges for all topics above threshold OR top-K topics
        start_time_edges = time.time()
        edges_added = 0
        top_k_topics = 3  # Assign top-3 topics to each chunk for better multi-label coverage

        for idx, chunk_id in enumerate(tqdm(chunk_ids, desc="Adding HAS_TOPIC edges", disable=self.logger.level > logging.INFO)):
            # Skip outliers (topic -1)
            if topics[idx] == -1:
                continue

            # Multi-topic assignment with probabilities
            if probabilities is not None and probabilities.ndim > 1:
                # probabilities shape: (n_chunks, n_topics)
                chunk_probs = probabilities[idx]

                # Strategy: Use TOP-K topics (with their probabilities as weights)
                # This ensures every chunk has multiple topics for CO_PRESENCE edges

                # Get top-K topic indices sorted by probability
                top_k_indices = np.argsort(chunk_probs)[-top_k_topics:][::-1]

                for topic_id in top_k_indices:
                    prob = chunk_probs[topic_id]

                    # Skip if topic is -1 (outlier) or probability is too low (< 1%)
                    if topic_id == -1 or prob < 0.01:
                        continue

                    theme_node = f"theme::{topic_id}"

                    # Add edge with probability weight
                    self.graph.add_edge(
                        chunk_id,
                        theme_node,
                        edge_type="HAS_TOPIC",
                        weight=float(prob)
                    )
                    edges_added += 1
            else:
                # Fallback: Hard assignment if no probabilities available
                theme_node = f"theme::{topics[idx]}"
                self.graph.add_edge(
                    chunk_id,
                    theme_node,
                    edge_type="HAS_TOPIC",
                    weight=1.0
                )
                edges_added += 1

        self._log_step_completion("Adding HAS_TOPIC edges (top-3 per chunk)", start_time_edges, edges_added)

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

        # Step 3: Calculate max_TF per chunk for local normalization
        chunk_max_tf = {}
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            # Get all TF values for entities in this chunk
            tf_values = []
            for entity_node, chunks_with_counts in entity_mentions_in_chunks.items():
                if chunk_id in chunks_with_counts:
                    tf_values.append(chunks_with_counts[chunk_id])

            # Store max TF for this chunk
            chunk_max_tf[chunk_id] = max(tf_values) if tf_values else 1.0

        self.logger.info(f"Calculated max TF for {len(chunk_max_tf)} chunks")

        # Step 4: Add entity nodes and MENTIONS edges with hybrid sigmoid weights
        start_time_mentions = time.time()
        scale = 3.0  # Sigmoid scale parameter (controls curve steepness)

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

            # Add MENTIONS edges with hybrid sigmoid weight
            idf = entity_idf[entity_node]
            for chunk_id, tf in entity_mentions_in_chunks[entity_node].items():
                # Normalize TF locally within chunk
                max_tf_in_chunk = chunk_max_tf.get(chunk_id, 1.0)
                normalized_tf = tf / max_tf_in_chunk

                # Calculate raw weight (local importance × global rarity)
                raw_weight = normalized_tf * idf

                # Apply sigmoid transformation to [0, 1]
                weight = 1.0 / (1.0 + np.exp(-raw_weight / scale))

                self.graph.add_edge(
                    chunk_id,
                    entity_node,
                    edge_type="MENTIONS",
                    weight=float(weight)
                )

        self._log_step_completion("Adding entities and MENTIONS edges", start_time_mentions, len(entity_mentions_in_chunks))

        return chunk_to_entities

    def _add_keyphrases(self, chunks: List[Dict]):
        """
        Extract keyphrases from chunks using KeyBERT and add keyphrase nodes with HAS_KEYPHRASE edges.

        For each chunk, extracts the top N most relevant keyphrases and creates:
        - Keyphrase nodes with the extracted phrase text
        - HAS_KEYPHRASE edges from chunk to keyphrase with relevance score as weight

        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'text' keys
        """
        start_time = time.time()
        keyphrase_count = 0
        edge_count = 0

        # Track unique keyphrases to avoid duplicates
        keyphrase_to_node = {}

        for chunk in tqdm(chunks, desc="Extracting keyphrases", disable=self.logger.level > logging.INFO):
            chunk_id = chunk['chunk_id']
            text = chunk['text']

            # Skip empty or very short texts
            if not text or len(text.strip()) < 20:
                continue

            try:
                # Extract keyphrases using KeyBERT
                keyphrases = self.keybert_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=self.keyphrase_ngram_range,
                    stop_words='english',
                    top_n=self.keyphrase_top_n,
                    use_mmr=True,  # Use Maximal Marginal Relevance for diversity
                    diversity=0.5
                )

                for phrase, score in keyphrases:
                    # Normalize keyphrase (lowercase, strip)
                    phrase_normalized = phrase.lower().strip()

                    # Skip very short phrases
                    if len(phrase_normalized) < 3:
                        continue

                    # Create or get keyphrase node
                    keyphrase_node = f"keyphrase::{phrase_normalized}"

                    if keyphrase_node not in keyphrase_to_node:
                        # Create new keyphrase node
                        self.graph.add_node(
                            keyphrase_node,
                            type="keyphrase",
                            text=phrase_normalized,
                            phrase=phrase  # Original casing
                        )
                        keyphrase_to_node[keyphrase_node] = True
                        keyphrase_count += 1

                    # Add HAS_KEYPHRASE edge with relevance score
                    self.graph.add_edge(
                        chunk_id,
                        keyphrase_node,
                        edge_type="HAS_KEYPHRASE",
                        weight=float(score)
                    )
                    edge_count += 1

            except Exception as e:
                # Log but don't fail on individual chunk errors
                self.logger.debug(f"KeyBERT extraction failed for {chunk_id}: {e}")
                continue

        self.logger.info(f"Created {keyphrase_count} unique keyphrase nodes")
        self._log_step_completion("Extracting keyphrases", start_time, edge_count)

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
        edges_skipped_date_date = 0

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

        # Add edges above threshold with PMI-based weighted score
        start_time_add = time.time()
        edges_added = 0
        total_docs = len(doc_to_entities)
        scale = 2.0  # Sigmoid scale parameter for PMI

        for (entity1, entity2), co_occur_count in tqdm(co_occur_counter.items(), desc="Adding CO_OCCURS edges", disable=self.logger.level > logging.INFO):
            if co_occur_count >= self.co_occur_threshold:
                # Skip DATE-DATE co-occurrences (they use TEMPORAL_DIST edges instead)
                entity1_type = self.graph.nodes[entity1].get('entity_type')
                entity2_type = self.graph.nodes[entity2].get('entity_type')

                if entity1_type == 'DATE' and entity2_type == 'DATE':
                    edges_skipped_date_date += 1
                    continue

                # Entity frequencies (number of documents they appear in)
                freq1 = entity_document_frequency[entity1]
                freq2 = entity_document_frequency[entity2]

                # Calculate PMI (Pointwise Mutual Information)
                # PMI = log((co_occur × total) / (freq1 × freq2))
                # Measures how much more often entities co-occur than expected by chance
                pmi = np.log((co_occur_count * total_docs) / (freq1 * freq2))

                # Apply sigmoid to map PMI to [0, 1]
                # High PMI (rare entities co-occurring) → high weight
                # Low/negative PMI (common entities, random co-occurrence) → low weight
                weight = 1.0 / (1.0 + np.exp(-pmi / scale))

                self.graph.add_edge(
                    entity1,
                    entity2,
                    edge_type="CO_OCCURS",
                    weight=float(weight),
                    co_occur_count=int(co_occur_count),
                    entity1_freq=int(freq1),
                    entity2_freq=int(freq2),
                    pmi=float(pmi)
                )
                edges_added += 1

        if edges_skipped_date_date > 0:
            self.logger.info(f"Skipped {edges_skipped_date_date} DATE-DATE co-occurrences (use TEMPORAL_DIST instead)")

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
                    # Clamp to [0, 1] to fix floating-point rounding errors
                    # If sim is slightly above 1.0 (e.g., 1.0000006), set to 1.0
                    if sim > 1.0 and sim <= 1.001:
                        sim = 1.0
                    elif sim > 1.001:
                        self.logger.warning(f"Similarity value {sim} > 1.001 detected, clamping to 1.0")
                        sim = 1.0

                    self.graph.add_edge(
                        entity_nodes[i],
                        entity_nodes[j],
                        edge_type="SEMANTIC_SIM",
                        weight=float(sim),
                        semantic_type="entity"  # Mark as entity-entity similarity
                    )
                    edge_count += 1

        self._log_step_completion("Adding entity SEMANTIC_SIM edges", start_time, edge_count)

    def _add_date_temporal_similarity(self, max_year_gap: int, decay_rate: float):
        """
        Add TEMPORAL_DIST edges between year entities with exponential decay.

        Uses an exponential negative decay function:
        weight = exp(-decay_rate * year_diff)

        Where:
        - year_diff = |year2 - year1|
        - decay_rate controls how fast the weight decreases

        Examples (with decay_rate=0.3):
        - Same year (diff=0): weight = 1.0
        - 1 year diff: weight ≈ 0.74
        - 2 years diff: weight ≈ 0.55
        - 3 years diff: weight ≈ 0.41
        - 5 years diff: weight ≈ 0.22
        - 10 years diff: weight ≈ 0.05

        Args:
            max_year_gap: Maximum year difference to create edges (from config)
            decay_rate: Decay rate for exponential function (from config)
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
                    # Clamp to [0, 1] to fix floating-point rounding errors
                    if sim > 1.0 and sim <= 1.001:
                        sim = 1.0
                    elif sim > 1.001:
                        self.logger.warning(f"Theme similarity value {sim} > 1.001 detected, clamping to 1.0")
                        sim = 1.0

                    self.graph.add_edge(
                        theme_nodes[i],
                        theme_nodes[j],
                        edge_type="SEMANTIC_SIM",
                        weight=float(sim),
                        semantic_type="theme"  # Mark as theme-theme similarity
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

        # Edge counts by type (with SEMANTIC_SIM breakdown)
        edge_types = defaultdict(int)
        semantic_sim_entity = 0
        semantic_sim_theme = 0

        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            if edge_type == 'SEMANTIC_SIM':
                # Separate SEMANTIC_SIM by semantic_type attribute
                sem_type = data.get('semantic_type', 'unknown')
                if sem_type == 'entity':
                    semantic_sim_entity += 1
                elif sem_type == 'theme':
                    semantic_sim_theme += 1
                else:
                    edge_types[edge_type] += 1
            else:
                edge_types[edge_type] += 1

        self.logger.info(f"\nTotal edges: {self.graph.number_of_edges()}")
        for etype, count in sorted(edge_types.items()):
            if etype == 'SEMANTIC_SIM':
                continue  # Skip, we'll show breakdown
            self.logger.info(f"  {etype}: {count}")

        # Show SEMANTIC_SIM breakdown
        if semantic_sim_entity > 0 or semantic_sim_theme > 0:
            total_sem = semantic_sim_entity + semantic_sim_theme
            self.logger.info(f"  SEMANTIC_SIM (total): {total_sem}")
            if semantic_sim_entity > 0:
                self.logger.info(f"    - Entity-Entity: {semantic_sim_entity}")
            if semantic_sim_theme > 0:
                self.logger.info(f"    - Theme-Theme: {semantic_sim_theme}")

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

        # Count edges by type (with SEMANTIC_SIM breakdown)
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            if edge_type == 'SEMANTIC_SIM':
                # Separate SEMANTIC_SIM by semantic_type
                sem_type = data.get('semantic_type', 'unknown')
                if sem_type == 'entity':
                    general_stats["edges_by_type"]["SEMANTIC_SIM_ENTITY"] = general_stats["edges_by_type"].get("SEMANTIC_SIM_ENTITY", 0) + 1
                elif sem_type == 'theme':
                    general_stats["edges_by_type"]["SEMANTIC_SIM_THEME"] = general_stats["edges_by_type"].get("SEMANTIC_SIM_THEME", 0) + 1
                else:
                    general_stats["edges_by_type"][edge_type] = general_stats["edges_by_type"].get(edge_type, 0) + 1
            else:
                general_stats["edges_by_type"][edge_type] = general_stats["edges_by_type"].get(edge_type, 0) + 1

        # Save general statistics
        with open(stats_dir / f"general_stats_{timestamp}.json", 'w') as f:
            json.dump(general_stats, f, indent=2)
        self.logger.info(f"General statistics saved to: {stats_dir}/general_stats_{timestamp}.json")

        # 2. Topic statistics with document counts (based on actual HAS_TOPIC edges)
        # Build mapping from themes to chunks using actual graph edges
        theme_to_chunks = defaultdict(set)
        chunks_with_topics = set()

        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'HAS_TOPIC':
                # u is chunk, v is theme
                if u.startswith('chunk::') and v.startswith('theme::'):
                    theme_to_chunks[v].add(u)
                    chunks_with_topics.add(u)
                elif v.startswith('chunk::') and u.startswith('theme::'):
                    theme_to_chunks[u].add(v)
                    chunks_with_topics.add(v)

        # Count outliers (chunks assigned to topic -1)
        num_outliers = sum(1 for t in topics if t == -1)
        # Count chunks without any topic (no HAS_TOPIC edges)
        all_chunk_nodes = {n for n, d in self.graph.nodes(data=True) if d.get('type') == 'chunk'}
        chunks_without_topics = all_chunk_nodes - chunks_with_topics

        topic_stats = {
            "timestamp": timestamp,
            "total_topics": len(set(topics)) - (1 if -1 in topics else 0),
            "total_chunks": len(chunks),
            "chunks_with_topics": len(chunks_with_topics),
            "chunks_without_topics": len(chunks_without_topics),
            "outlier_chunks": num_outliers,
            "topics": []
        }

        unique_topics = set(topics)
        if -1 in unique_topics:
            unique_topics.remove(-1)

        # Create chunk_id to doc_id mapping
        chunk_id_to_doc_id = {c['chunk_id']: c['doc_id'] for c in chunks}

        for topic_id in sorted(unique_topics):
            topic_words = topic_model.get_topic(topic_id)
            if not topic_words:
                continue

            # Count chunks and documents based on actual HAS_TOPIC edges
            theme_node = f"theme::{topic_id}"
            topic_chunk_ids = theme_to_chunks.get(theme_node, set())
            topic_doc_ids = set(chunk_id_to_doc_id.get(cid, '') for cid in topic_chunk_ids if cid in chunk_id_to_doc_id)
            topic_doc_ids.discard('')  # Remove empty doc_ids

            # Get theme label and processed keywords from graph
            if self.graph.has_node(theme_node):
                theme_label = self.graph.nodes[theme_node].get('label', '')
                processed_keywords = self.graph.nodes[theme_node].get('keywords', [])
            else:
                theme_label = ''
                processed_keywords = []

            topic_info = {
                "topic_id": topic_id,
                "label": theme_label,
                "num_chunks": len(topic_chunk_ids),
                "num_documents": len(topic_doc_ids),
                "top_words": [{"word": word, "score": float(score)} for word, score in topic_words[:10]],
                "keywords": processed_keywords  # Use processed keywords from graph node
            }
            topic_stats["topics"].append(topic_info)

        # Save topic statistics
        with open(stats_dir / f"topic_stats_{timestamp}.json", 'w') as f:
            json.dump(topic_stats, f, indent=2)
        self.logger.info(f"Topic statistics saved to: {stats_dir}/topic_stats_{timestamp}.json")
        self.logger.info(f"  - Chunks with topics: {topic_stats['chunks_with_topics']}/{topic_stats['total_chunks']}")
        self.logger.info(f"  - Chunks without topics: {topic_stats['chunks_without_topics']}")
        self.logger.info(f"  - Outlier chunks (topic -1): {topic_stats['outlier_chunks']}")

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

    def _save_timing_report(self, total_build_time: float):
        """
        Save build timing report showing time and percentage for each step.

        Args:
            total_build_time: Total time for the build process in seconds
        """
        import json
        from pathlib import Path
        from datetime import datetime

        stats_dir = Path("outputs/statistics")
        stats_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calculate percentages
        timing_report = {
            "timestamp": timestamp,
            "total_build_time_seconds": round(total_build_time, 2),
            "total_build_time_formatted": f"{int(total_build_time // 60)}m {int(total_build_time % 60)}s",
            "steps": []
        }

        self.logger.info("\n" + "=" * 60)
        self.logger.info("BUILD TIMING REPORT")
        self.logger.info("=" * 60)

        for step_name, step_time in self.build_timings.items():
            percentage = (step_time / total_build_time * 100) if total_build_time > 0 else 0
            timing_report["steps"].append({
                "step": step_name,
                "time_seconds": round(step_time, 2),
                "time_formatted": f"{int(step_time // 60)}m {step_time % 60:.1f}s" if step_time >= 60 else f"{step_time:.2f}s",
                "percentage": round(percentage, 1)
            })
            self.logger.info(f"  {step_name}: {step_time:.2f}s ({percentage:.1f}%)")

        self.logger.info(f"\n  TOTAL: {total_build_time:.2f}s ({int(total_build_time // 60)}m {int(total_build_time % 60)}s)")
        self.logger.info("=" * 60)

        # Save timing report
        with open(stats_dir / f"timing_report_{timestamp}.json", 'w') as f:
            json.dump(timing_report, f, indent=2)
        self.logger.info(f"Timing report saved to: {stats_dir}/timing_report_{timestamp}.json")

    def _precompute_claim_retrieval(
        self,
        claims: List[Dict],
        topic_model: BERTopic,
        probabilities: np.ndarray,
        chunk_ids: List[str]
    ) -> Dict[str, Dict]:
        """
        Pre-compute retrieval results for each claim using 4 strategies (OPTIMIZED).

        Uses batch processing for embeddings and topic transforms to significantly
        reduce computation time.

        Args:
            claims: List of claim dictionaries
            topic_model: Trained BERTopic model
            probabilities: Topic probability matrix for chunks
            chunk_ids: List of chunk IDs

        Returns:
            Dictionary mapping claim_id to retrieval results
        """
        from scipy.spatial.distance import jensenshannon
        from collections import defaultdict

        self.logger.info(f"Pre-computing retrieval for {len(claims)} claims (optimized batch mode)...")

        # Filter valid claims
        valid_claims = [(c['claim_id'], c.get('text', '')) for c in claims if c.get('text', '')]
        if not valid_claims:
            self.logger.warning("No valid claims to process")
            return {}

        claim_ids, claim_texts = zip(*valid_claims)
        self.logger.info(f"Processing {len(claim_texts)} valid claims")

        # BATCH 1: Pre-compute ALL chunk embeddings at once
        chunk_texts = [self.graph.nodes[cid].get('text', '') for cid in chunk_ids]
        self.logger.info("Computing chunk embeddings (batch)...")
        chunk_embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64
        )

        # BATCH 2: Pre-compute ALL claim embeddings at once
        self.logger.info("Computing claim embeddings (batch)...")
        claim_embeddings = self.embedding_model.encode(
            list(claim_texts),
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64
        )

        # BATCH 3: Compute ALL semantic similarities at once (claims x chunks matrix)
        self.logger.info("Computing semantic similarity matrix...")
        # Shape: (num_claims, num_chunks)
        all_S1_scores = np.matmul(claim_embeddings, chunk_embeddings.T)
        # Min-max normalize per claim (row-wise)
        S1_min = all_S1_scores.min(axis=1, keepdims=True)
        S1_max = all_S1_scores.max(axis=1, keepdims=True)
        S1_range = S1_max - S1_min
        S1_range[S1_range == 0] = 1  # Avoid division by zero
        all_S1_normalized = (all_S1_scores - S1_min) / S1_range

        # BATCH 4: Pre-compute ALL claim topic distributions at once
        self.logger.info("Computing claim topic distributions (batch)...")
        _, all_claim_topic_probs = topic_model.transform(list(claim_texts))

        # BATCH 5: Compute ALL JS divergence scores (vectorized)
        self.logger.info("Computing topic distribution similarities...")
        all_S2_scores = np.zeros((len(claim_texts), len(chunk_ids)))
        for claim_idx, claim_dist in enumerate(all_claim_topic_probs):
            for chunk_idx in range(len(chunk_ids)):
                chunk_dist = probabilities[chunk_idx]
                js_div = jensenshannon(claim_dist, chunk_dist)
                all_S2_scores[claim_idx, chunk_idx] = 1.0 - js_div
        # Min-max normalize per claim
        S2_min = all_S2_scores.min(axis=1, keepdims=True)
        S2_max = all_S2_scores.max(axis=1, keepdims=True)
        S2_range = S2_max - S2_min
        S2_range[S2_range == 0] = 1
        all_S2_normalized = (all_S2_scores - S2_min) / S2_range

        # Process S3 and S4 per claim (these depend on graph structure, harder to batch)
        self.logger.info("Computing thematic and entity scores per claim...")
        claim_retrieval_results = {}

        for idx, (claim_id, claim_text) in enumerate(tqdm(valid_claims, desc="Finalizing claim retrieval")):
            claim_node = f"claim::{claim_id}"

            # Get pre-computed S1 and S2 scores
            S1_scores = {cid: all_S1_normalized[idx, i] for i, cid in enumerate(chunk_ids)}
            S2_scores = {cid: all_S2_normalized[idx, i] for i, cid in enumerate(chunk_ids)}

            # Strategy 3: Thematic Graph Traversal (uses pre-computed topic probs)
            S3_scores = self._compute_thematic_scores_for_claim_with_probs(
                all_claim_topic_probs[idx]
            )
            # Normalize S3
            if S3_scores:
                min_s, max_s = min(S3_scores.values()), max(S3_scores.values())
                if max_s > min_s:
                    S3_scores = {k: (v - min_s) / (max_s - min_s) for k, v in S3_scores.items()}

            # Strategy 4: Entity Graph Traversal
            S4_scores = self._compute_entity_scores_for_claim(claim_text)
            # Normalize S4
            if S4_scores:
                min_s, max_s = min(S4_scores.values()), max(S4_scores.values())
                if max_s > min_s:
                    S4_scores = {k: (v - min_s) / (max_s - min_s) for k, v in S4_scores.items()}

            # Combine scores (equal weights: 0.25 each)
            all_chunks = set(S1_scores.keys()) | set(S2_scores.keys()) | set(S3_scores.keys()) | set(S4_scores.keys())
            combined_results = []

            for cid in all_chunks:
                s1 = S1_scores.get(cid, 0)
                s2 = S2_scores.get(cid, 0)
                s3 = S3_scores.get(cid, 0)
                s4 = S4_scores.get(cid, 0)
                total_score = 0.25 * s1 + 0.25 * s2 + 0.25 * s3 + 0.25 * s4

                combined_results.append({
                    "chunk_id": cid,
                    "score_total": round(float(total_score), 4),
                    "score_semantic": round(float(s1), 4),
                    "score_topic_dist": round(float(s2), 4),
                    "score_thematic": round(float(s3), 4),
                    "score_entity": round(float(s4), 4)
                })

            # Sort by total score and keep top 20
            combined_results.sort(key=lambda x: x['score_total'], reverse=True)
            top_results = combined_results[:20]

            claim_retrieval_results[claim_node] = {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "retrieval_results": top_results
            }

        self.logger.info(f"Pre-computed retrieval for {len(claim_retrieval_results)} claims")
        return claim_retrieval_results

    def _compute_thematic_scores_for_claim_with_probs(self, claim_topic_probs: np.ndarray) -> Dict[str, float]:
        """
        Compute thematic scores using pre-computed topic probabilities.

        Optimized version that takes pre-computed probs instead of recomputing.
        """
        from collections import deque, defaultdict

        # Detect themes above threshold (0.01)
        seed_themes = []
        for idx, prob in enumerate(claim_topic_probs):
            if prob > 0.01:
                theme_node = f"theme::{idx}"
                if self.graph.has_node(theme_node):
                    seed_themes.append((theme_node, prob))

        if not seed_themes:
            return {}

        # BFS propagation through theme graph
        theme_relevance = {}
        queue = deque()
        visited = set()

        for theme_node, prob in seed_themes:
            theme_relevance[theme_node] = prob
            queue.append((theme_node, prob))
            visited.add(theme_node)

        while queue:
            current_theme, current_rel = queue.popleft()
            for neighbor in self.graph.neighbors(current_theme):
                if self.graph.nodes[neighbor].get('type') != 'theme':
                    continue
                edge_data = self.graph[current_theme][neighbor]
                edge_type = edge_data.get('edge_type')
                if edge_type not in ['CO_PRESENCE', 'SEMANTIC_SIM']:
                    continue
                edge_weight = edge_data.get('weight', 0)
                propagated_rel = current_rel * edge_weight
                if propagated_rel < 0.1:
                    continue
                if neighbor not in theme_relevance:
                    theme_relevance[neighbor] = propagated_rel
                else:
                    theme_relevance[neighbor] = max(theme_relevance[neighbor], propagated_rel)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, propagated_rel))

        # Score chunks based on theme relevance
        scores = defaultdict(float)
        for theme_node, relevance in theme_relevance.items():
            for neighbor in self.graph.neighbors(theme_node):
                if self.graph.nodes[neighbor].get('type') == 'chunk':
                    edge_data = self.graph[theme_node][neighbor]
                    if edge_data.get('edge_type') == 'HAS_TOPIC':
                        weight = edge_data.get('weight', 1.0)
                        scores[neighbor] = max(scores[neighbor], relevance * weight)

        return dict(scores)

    def _compute_thematic_scores_for_claim(self, claim_text: str, topic_model: BERTopic) -> Dict[str, float]:
        """
        Compute thematic scores for a claim using theme graph traversal.

        Args:
            claim_text: The claim text
            topic_model: Trained BERTopic model

        Returns:
            Dictionary mapping chunk_id to thematic score
        """
        from collections import deque, defaultdict

        # Get topic probabilities for claim
        topic_probs = topic_model.transform([claim_text])[1][0]

        # Detect themes above threshold (0.01)
        seed_themes = []
        for idx, prob in enumerate(topic_probs):
            if prob > 0.01:
                theme_node = f"theme::{idx}"
                if self.graph.has_node(theme_node):
                    seed_themes.append((theme_node, prob))

        if not seed_themes:
            return {}

        # BFS propagation through theme graph
        theme_relevance = {}
        queue = deque()
        visited = set()

        for theme_node, prob in seed_themes:
            theme_relevance[theme_node] = prob
            queue.append((theme_node, prob))
            visited.add(theme_node)

        while queue:
            current_theme, current_rel = queue.popleft()
            for neighbor in self.graph.neighbors(current_theme):
                if self.graph.nodes[neighbor].get('type') != 'theme':
                    continue
                edge_data = self.graph[current_theme][neighbor]
                edge_type = edge_data.get('edge_type')
                if edge_type not in ['CO_PRESENCE', 'SEMANTIC_SIM']:
                    continue
                edge_weight = edge_data.get('weight', 0)
                propagated_rel = current_rel * edge_weight
                if propagated_rel < 0.1:
                    continue
                if neighbor not in theme_relevance:
                    theme_relevance[neighbor] = propagated_rel
                else:
                    theme_relevance[neighbor] = max(theme_relevance[neighbor], propagated_rel)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, propagated_rel))

        # Score chunks based on theme relevance
        scores = defaultdict(float)
        for theme_node, relevance in theme_relevance.items():
            for neighbor in self.graph.neighbors(theme_node):
                edge_data = self.graph[theme_node][neighbor]
                if (edge_data.get('edge_type') == 'HAS_TOPIC' and
                    self.graph.nodes[neighbor].get('type') == 'chunk'):
                    prob_weight = edge_data.get('weight', 0.0)
                    scores[neighbor] += relevance * prob_weight

        return dict(scores)

    def _compute_entity_scores_for_claim(self, claim_text: str) -> Dict[str, float]:
        """
        Compute entity scores for a claim using entity graph traversal.

        Args:
            claim_text: The claim text

        Returns:
            Dictionary mapping chunk_id to entity score
        """
        from collections import deque, defaultdict

        # Extract entities from claim
        doc = self.nlp(claim_text)
        claim_entities = []
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity_text = ent.text.strip().lower()
                if len(entity_text) > 2:
                    entity_node = f"entity::{ent.label_}::{entity_text}"
                    if self.graph.has_node(entity_node):
                        claim_entities.append(entity_node)

        if not claim_entities:
            return {}

        # BFS propagation through entity graph
        entity_relevance = {e: 1.0 for e in claim_entities}
        queue = deque([(e, 1.0) for e in claim_entities])
        visited = set(claim_entities)

        while queue:
            current_entity, current_rel = queue.popleft()
            for neighbor in self.graph.neighbors(current_entity):
                if self.graph.nodes[neighbor].get('type') != 'entity':
                    continue
                edge_data = self.graph[current_entity][neighbor]
                edge_type = edge_data.get('edge_type')
                if edge_type not in ['CO_OCCURS', 'SEMANTIC_SIM']:
                    continue
                edge_weight = edge_data.get('weight', 0)
                propagated_rel = current_rel * edge_weight
                if propagated_rel < 0.1:
                    continue
                if neighbor not in entity_relevance:
                    entity_relevance[neighbor] = propagated_rel
                else:
                    entity_relevance[neighbor] = max(entity_relevance[neighbor], propagated_rel)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, propagated_rel))

        # Score chunks based on entity relevance
        scores = defaultdict(float)
        for entity_node, relevance in entity_relevance.items():
            for neighbor in self.graph.neighbors(entity_node):
                edge_data = self.graph[entity_node][neighbor]
                if (edge_data.get('edge_type') == 'MENTIONS' and
                    self.graph.nodes[neighbor].get('type') == 'chunk'):
                    tfidf_weight = edge_data.get('weight', 1.0)
                    scores[neighbor] += relevance * tfidf_weight

        return dict(scores)

    def save_to_postgres(
        self,
        graph_id: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        metadata: dict = None,
        graph: nx.Graph = None
    ):
        """
        Save the graph to PostgreSQL database with timestamped tables.

        Args:
            graph_id: Graph ID (timestamp) for creating versioned tables
            host: PostgreSQL host (from config)
            port: PostgreSQL port (from config)
            database: Database name (from config)
            user: Username (from config)
            password: Password (from config)
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
