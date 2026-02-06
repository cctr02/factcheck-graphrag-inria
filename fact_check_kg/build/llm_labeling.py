"""
LLM-based topic labeling module.

This module handles batched topic labeling using Ollama LLM with:
- Keyword processing (deduplication, lemmatization)
- Batched LLM requests with auto-tuning
- Retry logic with binary split or single fallback strategies
"""

import json
import re
import logging
import requests
from typing import List, Dict, Tuple, Set, Optional
from nltk.stem import WordNetLemmatizer


class TopicLabeler:
    """
    LLM-based topic labeler with batched requests and auto-tuning.

    Handles the complete topic labeling pipeline:
    1. Keyword processing (deduplication, lemmatization)
    2. Batched LLM calls with retry logic
    3. Response parsing and validation
    4. Auto-tuning of batch size based on success/failure
    """

    def __init__(
        self,
        llm_endpoint: str,
        llm_model: str,
        prompt_template: str,
        batch_size_default: int,
        batch_size_min: int,
        batch_size_max: int,
        batch_size_step: int,
        label_min_words: int,
        label_max_words: int,
        label_max_length: int,
        max_keywords_per_topic: int,
        retry_count: int,
        retry_temperature: float,
        batch_timeout: int,
        success_streak_threshold: int,
        hysteresis_decrease_factor: float,
        failure_strategy: str,
        topic_keywords_initial: int,
        topic_keywords_final: int,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the topic labeler.

        Args:
            llm_endpoint: LLM API endpoint (e.g., Ollama)
            llm_model: LLM model name
            prompt_template: Prompt template with {{topics_json}} placeholder
            batch_size_default: Starting batch size
            batch_size_min: Minimum batch size (floor)
            batch_size_max: Maximum batch size (cap)
            batch_size_step: Step for increasing batch size
            label_min_words: Minimum words in label
            label_max_words: Maximum words in label
            label_max_length: Maximum character length for labels
            max_keywords_per_topic: Number of keywords to send per topic
            retry_count: Number of retries on failure
            retry_temperature: Temperature for retry attempts
            batch_timeout: Timeout for batch requests in seconds
            success_streak_threshold: Consecutive successes before increasing batch size
            hysteresis_decrease_factor: Factor to decrease batch size on failure
            failure_strategy: "binary_split" or "single_fallback"
            topic_keywords_initial: Number of keywords to retrieve initially
            topic_keywords_final: Number of keywords after deduplication
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # LLM configuration
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self.batch_timeout = batch_timeout

        # Batch size configuration
        self.batch_size_default = batch_size_default
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        self.batch_size_step = batch_size_step

        # Label constraints
        self.label_min_words = label_min_words
        self.label_max_words = label_max_words
        self.label_max_length = label_max_length

        # Keywords configuration
        self.max_keywords_per_topic = max_keywords_per_topic
        self.topic_keywords_initial = topic_keywords_initial
        self.topic_keywords_final = topic_keywords_final

        # Retry configuration
        self.retry_count = retry_count
        self.retry_temperature = retry_temperature

        # Auto-tuning configuration
        self.success_streak_threshold = success_streak_threshold
        self.hysteresis_decrease_factor = hysteresis_decrease_factor
        self.failure_strategy = failure_strategy

        # Auto-tuning state
        self._batch_size_current = batch_size_default
        self._success_streak = 0

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

    def process_topic_keywords(
        self,
        topic_words: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Process topic keywords with stemming and n-gram deduplication.

        Args:
            topic_words: List of (word, score) from BERTopic

        Returns:
            List of (word, score) tuples, deduplicated and lemmatized
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

        removed = min(len(topic_words), self.topic_keywords_initial) - len(top_keywords)
        self.logger.debug(
            f"Processed keywords: {min(len(topic_words), self.topic_keywords_initial)} -> "
            f"{len(top_keywords)} (removed {removed} redundant)"
        )
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
        if len(label) > self.label_max_length:
            label = ' '.join(label.split()[:self.label_max_words])
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
        if word_count < self.label_min_words:
            return False
        if word_count > self.label_max_words:
            return False
        if len(label) > self.label_max_length:
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
        # Build prompt from template
        topics_json = json.dumps(topics_data, ensure_ascii=False)
        prompt = self.prompt_template.replace('{{topics_json}}', topics_json)

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
                timeout=self.batch_timeout
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:200]}")

            return response.json()['response']

        except requests.exceptions.Timeout:
            raise RuntimeError(f"LLM batch request timed out after {self.batch_timeout}s")
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
            if self._success_streak >= self.success_streak_threshold:
                # Increase batch size
                new_size = min(
                    self._batch_size_current + self.batch_size_step,
                    self.batch_size_max
                )
                if new_size != self._batch_size_current:
                    self._batch_size_current = new_size
                    self.logger.info(f"[BATCH AUTO-TUNE] Increased batch size: {old_size} -> {new_size}")
                self._success_streak = 0  # Reset streak after increase
        else:
            self._success_streak = 0
            # Decrease batch size using hysteresis factor
            new_size = max(
                int(self._batch_size_current * self.hysteresis_decrease_factor),
                self.batch_size_min
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

        for attempt in range(self.retry_count + 1):
            temperature = self.retry_temperature if attempt > 0 else 0.3
            try:
                response_text = self._call_llm_batch(topics_data, temperature)
                labels, _ = self._parse_batch_response(response_text, {topic_id})

                if topic_id in labels:
                    return labels[topic_id]

            except RuntimeError as e:
                self.logger.debug(f"Single topic {topic_id} attempt {attempt + 1} failed: {e}")

        raise RuntimeError(f"Failed to label topic {topic_id} after {self.retry_count + 1} attempts")

    def label_topics_batch(
        self,
        topics_to_label: List[Tuple[int, List[Tuple[str, float]]]]
    ) -> Tuple[Dict[int, str], Dict[int, List[Tuple[str, float]]]]:
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
            Tuple of (labels_dict, processed_keywords_dict):
            - labels_dict: Dictionary mapping topic_id -> label string
            - processed_keywords_dict: Dictionary mapping topic_id -> processed keywords list

        Raises:
            RuntimeError: If labeling fails completely
        """
        if not topics_to_label:
            return {}, {}

        # Prepare all topics data and store processed keywords for reuse
        all_topics_data = []
        processed_keywords_map = {}  # topic_id -> processed keywords (for node creation)
        for topic_id, topic_words in topics_to_label:
            # Process and limit keywords (only once per topic)
            processed = self.process_topic_keywords(topic_words)
            processed_keywords_map[topic_id] = processed  # Store for caller
            keywords = [w for w, s in processed[:self.max_keywords_per_topic]]
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
            for attempt in range(self.retry_count + 1):
                temperature = self.retry_temperature if attempt > 0 else 0.3

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
                self.logger.debug(f"Handling {len(batch_missing)} missing topics with strategy: {self.failure_strategy}")

                if self.failure_strategy == "binary_split" and len(batch_missing) > 1:
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
        return labels, processed_keywords_map

    @property
    def current_batch_size(self) -> int:
        """Get current auto-tuned batch size."""
        return self._batch_size_current

    def reset_batch_size(self):
        """Reset batch size to default (e.g., for new labeling session)."""
        self._batch_size_current = self.batch_size_default
        self._success_streak = 0
