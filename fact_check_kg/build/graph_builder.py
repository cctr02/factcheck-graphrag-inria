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
from keybert import KeyBERT
import threading
import sys
import types


from fact_check_kg.core.db_connector import GraphDatabaseConnector
from fact_check_kg.build.llm_labeling import TopicLabeler
from fact_check_kg.build.graph_nodes_chunk_docs import add_documents_and_chunks, add_keyphrases
from fact_check_kg.build.graph_nodes_entity import (
    extract_entities,
    add_entities_and_mentions,
    add_entity_co_occurrences,
    add_entity_semantic_similarity,
    add_date_temporal_similarity
)
from fact_check_kg.build.graph_nodes_theme import (
    add_themes_and_edges,
    build_hierarchical_topics,
    add_theme_semantic_similarity,
    add_theme_co_presence
)
from fact_check_kg.build.graph_nodes_claims import (
    add_claims_and_edges,
    add_claim_semantic_similarity,
    precompute_claim_retrieval
)
from fact_check_kg.build.topic_visualization import generate_topic_html_report


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
        claim_sim_threshold: float,
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
            semantic_sim_threshold: Minimum similarity to create entity/theme SEMANTIC_SIM edge
            claim_sim_threshold: Minimum similarity to create CLAIM_SIM edge (typically lower)
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

        # Detect GPU availability
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            self.logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"  CUDA version: {torch.version.cuda}")
            self.logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.logger.warning("⚠ No GPU detected, using CPU (this will be slower)")

        # Initialize models with GPU support
        self.logger.info(f"Loading SentenceTransformer model: {embedding_model} on {self.device.upper()}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)

        # Enable GPU for spaCy BEFORE loading the model
        if self.device == 'cuda':
            self.logger.info("Enabling GPU acceleration for spaCy...")
            try:
                spacy.require_gpu()
                self.logger.info("✓ spaCy GPU enabled successfully")
            except Exception as e:
                self.logger.warning(f"⚠ Failed to enable spaCy GPU: {e}")
                self.logger.warning("  spaCy will run on CPU")

        self.logger.info(f"Loading spaCy NER model: {ner_model}")
        self.nlp = spacy.load(ner_model)

        # Configure spaCy for batch processing
        # Increase batch size for GPU processing
        if self.device == 'cuda':
            # Larger batches for GPU
            self.nlp.batch_size = 1000
        else:
            # Smaller batches for CPU
            self.nlp.batch_size = 50

        # Configuration
        self.entity_types = entity_types
        self.co_occur_threshold = co_occur_threshold
        self.semantic_sim_threshold = semantic_sim_threshold
        self.claim_sim_threshold = claim_sim_threshold
        self.topic_prob_threshold = topic_prob_threshold

        # Temporal similarity configuration
        self.temporal_max_year_gap = temporal_max_year_gap
        self.temporal_decay_rate = temporal_decay_rate

        # Topic keyword processing configuration
        self.topic_keywords_initial = topic_keywords_initial
        self.topic_keywords_final = topic_keywords_final

        # Initialize topic labeler (handles LLM-based labeling with batching and auto-tuning)
        self.logger.info("Initializing TopicLabeler for LLM-based topic labeling...")
        self.topic_labeler = TopicLabeler(
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            prompt_template=llm_batch_prompt_template,
            batch_size_default=llm_batch_size_default,
            batch_size_min=llm_batch_size_min,
            batch_size_max=llm_batch_size_max,
            batch_size_step=llm_batch_size_step,
            label_min_words=llm_label_min_words,
            label_max_words=llm_label_max_words,
            label_max_length=llm_label_max_length,
            max_keywords_per_topic=llm_max_keywords_per_topic,
            retry_count=llm_retry_count,
            retry_temperature=llm_retry_temperature,
            batch_timeout=llm_batch_timeout,
            success_streak_threshold=llm_success_streak_threshold,
            hysteresis_decrease_factor=llm_hysteresis_decrease_factor,
            failure_strategy=llm_failure_strategy,
            topic_keywords_initial=topic_keywords_initial,
            topic_keywords_final=topic_keywords_final,
            logger=self.logger
        )

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
        nr_topics: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[BERTopic, List[int], np.ndarray]:
        """
        Build topic model using BERTopic.

        Args:
            chunk_texts: List of chunk texts
            min_topic_size: Minimum size for a topic
            nr_topics: Number of topics (None = auto)
            embeddings: Pre-computed embeddings (None = compute from scratch)

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
        if embeddings is not None:
            self.logger.info(f"Using pre-computed embeddings: {embeddings.shape}")
        else:
            self.logger.info(f"Computing embeddings for {len(chunk_texts)} chunks...")

        # Create vectorizer with stop words
        from sklearn.feature_extraction.text import CountVectorizer
        from hdbscan import HDBSCAN

        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,
            ngram_range=(1, 3),  # Include trigrams
            max_features=1000     # More features
        )

        # Custom UMAP for dimensionality reduction with GPU support if available
        try:
            if self.device == 'cuda':
                # Try to use cuML UMAP (GPU-accelerated)
                from cuml.manifold import UMAP as cumlUMAP
                umap_model = cumlUMAP(
                    n_neighbors=15,
                    n_components=5,
                    min_dist=0.0,
                    metric='cosine',
                    verbose=True
                )
                self.logger.info("✓ Using GPU-accelerated UMAP (cuML)")
            else:
                raise ImportError("CPU mode")
        except ImportError:
            # Fallback to CPU UMAP
            from umap import UMAP
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                verbose=True
            )
            if self.device == 'cuda':
                self.logger.warning("⚠ cuML not available, using CPU UMAP (install rapids-cuml for GPU acceleration)")

        # Create HDBSCAN model
        # prediction_data=True required for calculate_probabilities (multi-topic assignment)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,  # Required for multi-topic probabilities
            core_dist_n_jobs=-1  # Use ALL CPU cores for parallel distance computation
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
        device_str = "GPU" if self.device == 'cuda' else "CPU"
        if embeddings is not None:
            self.logger.info("  → Stage 1/4: Embeddings SKIPPED (using pre-computed from checkpoint)")
        else:
            self.logger.info(f"  → Stage 1/4: Computing embeddings ({device_str}-accelerated)...")
        self.logger.info("  → Stage 2/4: UMAP dimensionality reduction (kNN-based)...")
        self.logger.info("  → Stage 3/4: HDBSCAN clustering (CPU, all cores)...")

        # Wrapper to show message after HDBSCAN
        original_extract_topics = topic_model._extract_topics

        def extract_topics_with_logging(*args, **kwargs):
            self.logger.info("  → Stage 4/5: Computing c-TF-IDF representations (CPU-intensive, may take 1-5min)...")
            self.logger.info("              This step is working even if GPU usage is 0% - please be patient...")
            return original_extract_topics(*args, **kwargs)

        topic_model._extract_topics = extract_topics_with_logging

        # Pass pre-computed embeddings if available, otherwise BERTopic computes them
        topics, probabilities = topic_model.fit_transform(
            chunk_texts,
            embeddings=embeddings  # None = BERTopic computes; ndarray = skip embedding step
        )

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

    def build_graph(
        self,
        chunks: List[Dict],
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None,
        claims: Optional[List[Dict]] = None,
        claim_to_docs: Optional[Dict[str, List[str]]] = None,
        doc_metadata: Optional[Dict[str, Dict]] = None,
        checkpoint_manager=None,
        graph_id: Optional[str] = None
    ) -> Tuple[nx.Graph, np.ndarray, List[str], Dict[str, Dict]]:
        """
        Build the complete knowledge graph through a 10-step pipeline.

        Pipeline steps:
        1. Add document and chunk nodes (CONTAINS edges)
        2a. Compute chunk embeddings (GPU)
        2b. Train BERTopic model (UMAP + HDBSCAN + c-TF-IDF)
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
            checkpoint_manager: Optional CheckpointManager for save/resume

        Returns:
            Tuple of (graph, topic_probabilities, chunk_ids, claim_retrieval_results):
            - graph: NetworkX graph with all nodes and edges
            - topic_probabilities: numpy array of shape (n_chunks, n_topics)
            - chunk_ids: list of chunk IDs matching probability matrix rows
            - claim_retrieval_results: dict mapping claim_id to pre-computed results
        """
        cm = checkpoint_manager  # shorthand

        self.logger.info("=" * 60)
        self.logger.info("BUILDING KNOWLEDGE GRAPH")
        if cm and cm.get_last_completed():
            self.logger.info(f"  RESUMING from checkpoint: {cm.get_last_completed()}")
        self.logger.info("=" * 60)

        # Initialize timing
        self.build_start_time = time.time()
        self.build_timings = {}

        # Extract data
        chunk_texts = [c['text'] for c in chunks]
        chunk_ids = [c['chunk_id'] for c in chunks]
        doc_ids = [c['doc_id'] for c in chunks]

        # ================================================================
        # Step 1: Add document and chunk nodes
        # ================================================================
        if cm and cm.should_skip("step_01_chunks"):
            self.logger.info("\n[SKIP] Step 1: Documents & Chunks (from checkpoint)")
            data = cm.load("step_01_chunks")
            self.graph = data['graph']
        else:
            self.logger.info("\nStep 1: Adding documents and chunks...")
            step_start = time.time()
            add_documents_and_chunks(self.graph, chunks, doc_metadata, self.logger, self._log_step_completion)
            self.build_timings["Step 1: Documents & Chunks"] = time.time() - step_start
            if cm:
                cm.save("step_01_chunks", {'graph': self.graph},
                        metadata={'n_docs': len(set(doc_ids)), 'n_chunks': len(chunks)})

        # ================================================================
        # Step 2a: Compute embeddings (GPU, fast - CRITICAL CHECKPOINT)
        # ================================================================
        if cm and cm.should_skip("step_02a_embeddings"):
            self.logger.info("\n[SKIP] Step 2a: Embeddings (from checkpoint)")
            data = cm.load("step_02a_embeddings")
            embeddings = data['embeddings']
            self.logger.info(f"  Loaded embeddings: {embeddings.shape}")
        else:
            self.logger.info("\nStep 2a: Computing chunk embeddings...")
            step_start = time.time()
            batch_size = 128 if self.device == 'cuda' else 32
            embeddings = self.embedding_model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device
            )
            self.build_timings["Step 2a: Embeddings"] = time.time() - step_start
            if cm:
                cm.save("step_02a_embeddings", {'embeddings': embeddings},
                        metadata={'shape': list(embeddings.shape)})
                self.logger.info("[CHECKPOINT] Embeddings saved - safe to interrupt during HDBSCAN!")

        # ================================================================
        # Step 2b: Topic modeling (UMAP + HDBSCAN + c-TF-IDF - CPU, slow)
        # ================================================================
        if cm and cm.should_skip("step_02b_topics"):
            self.logger.info("\n[SKIP] Step 2b: Topic modeling (from checkpoint)")
            data = cm.load("step_02b_topics")
            topic_model = data['topic_model']
            topics = data['topics']
            probabilities = data['probabilities']
            self.topic_model = topic_model
        else:
            self.logger.info("\nStep 2b: Running topic modeling (UMAP + HDBSCAN)...")
            step_start = time.time()
            topic_model, topics, probabilities = self.build_topic_model(
                chunk_texts,
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                embeddings=embeddings  # Use pre-computed embeddings!
            )
            self.build_timings["Step 2b: Topic Modeling"] = time.time() - step_start
            if cm:
                cm.save("step_02b_topics", {
                    'topic_model': topic_model,
                    'topics': topics,
                    'probabilities': probabilities
                }, metadata={'n_topics': len(set(topics)) - (1 if -1 in topics else 0)})

        # ================================================================
        # Step 3: Add themes + hierarchical topics
        # ================================================================
        if cm and cm.should_skip("step_03_graph_themes"):
            self.logger.info("\n[SKIP] Step 3: Themes & Hierarchical topics (from checkpoint)")
            data = cm.load("step_03_graph_themes")
            self.graph = data['graph']
            self.topic_to_supertopic = data.get('topic_to_supertopic', {})
        else:
            # Step 3: Add themes and HAS_TOPIC edges (requires LLM for labeling)
            self.logger.info("\nStep 3: Adding themes and HAS_TOPIC edges...")
            step_start = time.time()
            add_themes_and_edges(self.graph, topic_model, topics, probabilities, chunk_ids, self.topic_labeler, self.logger, self._log_step_completion)
            self.build_timings["Step 3: Themes & HAS_TOPIC"] = time.time() - step_start

            # Step 3b: Build hierarchical topic structure (supertopics)
            self.logger.info("\nStep 3b: Building hierarchical topics (supertopics)...")
            step_start = time.time()
            self.topic_to_supertopic = build_hierarchical_topics(
                self.graph, topic_model, chunk_texts, self.topic_labeler,
                nr_supertopics=None,  # Auto: ~40% of topics
                logger=self.logger, log_step_fn=self._log_step_completion
            )
            self.build_timings["Step 3b: Hierarchical Topics"] = time.time() - step_start
            if cm:
                cm.save("step_03_graph_themes", {
                    'graph': self.graph,
                    'topic_to_supertopic': self.topic_to_supertopic
                })

        # ================================================================
        # Step 4: Entities + Keyphrases
        # ================================================================
        if cm and cm.should_skip("step_04_graph_entities"):
            self.logger.info("\n[SKIP] Step 4: Entities & Keyphrases (from checkpoint)")
            data = cm.load("step_04_graph_entities")
            self.graph = data['graph']
            chunk_to_entities = data['chunk_to_entities']
        else:
            # Step 4: Extract entities and add MENTIONS edges
            self.logger.info("\nStep 4: Extracting entities and adding MENTIONS edges...")
            step_start = time.time()
            chunk_to_entities = add_entities_and_mentions(self.graph, chunks, self.nlp, self.entity_types, self.logger, self._log_step_completion)
            self.build_timings["Step 4: Entities & MENTIONS"] = time.time() - step_start

            # Step 4b: Extract keyphrases and add HAS_KEYPHRASE edges
            self.logger.info("\nStep 4b: Extracting keyphrases and adding HAS_KEYPHRASE edges...")
            step_start = time.time()
            add_keyphrases(self.graph, chunks, self.keybert_model, self.keyphrase_top_n, self.keyphrase_ngram_range, self.logger, self._log_step_completion)
            self.build_timings["Step 4b: Keyphrases"] = time.time() - step_start
            if cm:
                cm.save("step_04_graph_entities", {
                    'graph': self.graph,
                    'chunk_to_entities': dict(chunk_to_entities)  # Convert defaultdict
                })

        # ================================================================
        # Steps 5-8: All cross-edges
        # ================================================================
        if cm and cm.should_skip("step_05_graph_edges"):
            self.logger.info("\n[SKIP] Steps 5-8: Cross-edges (from checkpoint)")
            data = cm.load("step_05_graph_edges")
            self.graph = data['graph']
        else:
            # Step 5: Add CO_OCCURS edges (Entity ↔ Entity)
            self.logger.info("\nStep 5: Adding CO_OCCURS edges (Entity ↔ Entity)...")
            step_start = time.time()
            add_entity_co_occurrences(self.graph, chunk_to_entities, self.co_occur_threshold, self.logger, self._log_step_completion)
            self.build_timings["Step 5: CO_OCCURS"] = time.time() - step_start

            # Step 6: Add SEMANTIC_SIM edges (Entity ↔ Entity, excluding DATEs)
            self.logger.info("\nStep 6: Adding SEMANTIC_SIM edges (Entity ↔ Entity)...")
            step_start = time.time()
            add_entity_semantic_similarity(self.graph, self.embedding_model, self.semantic_sim_threshold, self.logger, self._log_step_completion)
            self.build_timings["Step 6: Entity SEMANTIC_SIM"] = time.time() - step_start

            # Step 6b: Add TEMPORAL_DIST edges (DATE ↔ DATE)
            self.logger.info("\nStep 6b: Adding TEMPORAL_DIST edges (DATE ↔ DATE)...")
            step_start = time.time()
            add_date_temporal_similarity(self.graph, self.temporal_max_year_gap, self.temporal_decay_rate, self.logger, self._log_step_completion)
            self.build_timings["Step 6b: TEMPORAL_DIST"] = time.time() - step_start

            # Step 7: Add SEMANTIC_SIM edges (Theme ↔ Theme)
            self.logger.info("\nStep 7: Adding SEMANTIC_SIM edges (Theme ↔ Theme)...")
            step_start = time.time()
            add_theme_semantic_similarity(self.graph, topic_model, self.embedding_model, self.semantic_sim_threshold, self.logger, self._log_step_completion)
            self.build_timings["Step 7: Theme SEMANTIC_SIM"] = time.time() - step_start

            # Step 8: Add CO_PRESENCE edges (Theme ↔ Theme)
            self.logger.info("\nStep 8: Adding CO_PRESENCE edges (Theme ↔ Theme)...")
            step_start = time.time()
            add_theme_co_presence(self.graph, chunk_ids, self.co_occur_threshold, self.logger, self._log_step_completion)
            self.build_timings["Step 8: CO_PRESENCE"] = time.time() - step_start
            if cm:
                cm.save("step_05_graph_edges", {'graph': self.graph})

        # ================================================================
        # Steps 9-10: Claims (optional)
        # ================================================================
        if claims and claim_to_docs:
            if cm and cm.should_skip("step_06_graph_claims"):
                self.logger.info("\n[SKIP] Steps 9-10: Claims (from checkpoint)")
                data = cm.load("step_06_graph_claims")
                self.graph = data['graph']
                claim_retrieval_results = data.get('claim_retrieval_results', {})
            else:
                self.logger.info("\nStep 9: Adding claims and REVIEWS edges (Claim → Document)...")
                step_start = time.time()
                add_claims_and_edges(self.graph, claims, claim_to_docs, self.logger, self._log_step_completion)
                self.build_timings["Step 9: Claims & REVIEWS"] = time.time() - step_start

                # Step 9b: Add CLAIM_SIM edges (semantic similarity between claims)
                self.logger.info("\nStep 9b: Adding CLAIM_SIM edges (Claim ↔ Claim semantic similarity)...")
                step_start = time.time()
                add_claim_semantic_similarity(
                    self.graph,
                    self.embedding_model,
                    sim_threshold=self.claim_sim_threshold,
                    top_k_per_claim=10,
                    logger=self.logger,
                    log_step_fn=self._log_step_completion
                )
                self.build_timings["Step 9b: Claim SEMANTIC_SIM"] = time.time() - step_start

                if cm:
                    cm.save("step_06_graph_claims", {'graph': self.graph})

        # Note: S5/S6/S7 claim retrieval is computed on-the-fly at retrieval time (no precompute)
        claim_retrieval_results = {}

        # Calculate total build time
        total_build_time = time.time() - self.build_start_time

        # Log statistics and save detailed reports
        self._log_graph_statistics()
        self._save_statistics_reports(topic_model, topics, probabilities, chunks, graph_id=graph_id)
        self._save_timing_report(total_build_time, graph_id=graph_id)

        # Return graph, probabilities, chunk_ids, and claim_retrieval_results
        return self.graph, probabilities, chunk_ids, claim_retrieval_results

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

        # Edge counts by type (with SEMANTIC_SIM and CLAIM_SIM breakdowns)
        edge_types = defaultdict(int)
        semantic_sim_entity = 0
        semantic_sim_theme = 0
        claim_sim_count = 0

        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            if edge_type == 'SEMANTIC_SIM':
                sem_type = data.get('semantic_type', 'unknown')
                if sem_type == 'entity':
                    semantic_sim_entity += 1
                elif sem_type == 'theme':
                    semantic_sim_theme += 1
                else:
                    edge_types[edge_type] += 1
            elif edge_type == 'CLAIM_SIM':
                claim_sim_count += 1
            else:
                edge_types[edge_type] += 1

        self.logger.info(f"\nTotal edges: {self.graph.number_of_edges()}")
        for etype, count in sorted(edge_types.items()):
            self.logger.info(f"  {etype}: {count}")

        # Show SEMANTIC_SIM breakdown
        if semantic_sim_entity > 0 or semantic_sim_theme > 0:
            total_sem = semantic_sim_entity + semantic_sim_theme
            self.logger.info(f"  SEMANTIC_SIM (total): {total_sem}")
            if semantic_sim_entity > 0:
                self.logger.info(f"    - Entity-Entity: {semantic_sim_entity}")
            if semantic_sim_theme > 0:
                self.logger.info(f"    - Theme-Theme: {semantic_sim_theme}")

        # Show CLAIM_SIM stats
        n_claims = node_types.get('claim', 0)
        self.logger.info(f"\nClaim statistics:")
        self.logger.info(f"  Claim nodes: {n_claims}")
        self.logger.info(f"  CLAIM_SIM edges (Claim↔Claim): {claim_sim_count}")
        if n_claims > 1:
            max_possible = n_claims * (n_claims - 1) // 2
            density = claim_sim_count / max_possible if max_possible > 0 else 0
            self.logger.info(f"  Claim graph density: {density:.4f} ({claim_sim_count}/{max_possible} max possible)")

        self.logger.info("=" * 60)

    def _save_statistics_reports(
        self,
        topic_model: BERTopic,
        topics: List[int],
        probabilities: np.ndarray,
        chunks: List[Dict],
        graph_id: Optional[str] = None
    ):
        """
        Save detailed statistics reports to outputs folder.

        Generates:
        - general_stats_{graph_id}.json: Node/edge counts by type
        - topic_stats_{graph_id}.json: Topic details with keywords
        - entity_stats_{graph_id}.json: Entity statistics
        - summary_{graph_id}.txt: Human-readable summary
        - topic_visualization_{graph_id}.html: Interactive topic visualization

        Args:
            topic_model: Trained BERTopic model
            topics: List of topic assignments
            probabilities: Topic probability matrix (n_chunks, n_topics)
            chunks: Original chunks data
            graph_id: Graph identifier (timestamp) shared across all outputs for this build
        """
        import json
        from pathlib import Path
        from datetime import datetime

        # Create outputs directory
        stats_dir = Path("outputs/statistics")
        stats_dir.mkdir(parents=True, exist_ok=True)

        timestamp = graph_id if graph_id is not None else datetime.now().strftime("%Y%m%d_%H%M%S")

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

        # Count edges by type (with SEMANTIC_SIM and CLAIM_SIM breakdowns)
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            if edge_type == 'SEMANTIC_SIM':
                sem_type = data.get('semantic_type', 'unknown')
                if sem_type == 'entity':
                    general_stats["edges_by_type"]["SEMANTIC_SIM_ENTITY"] = general_stats["edges_by_type"].get("SEMANTIC_SIM_ENTITY", 0) + 1
                elif sem_type == 'theme':
                    general_stats["edges_by_type"]["SEMANTIC_SIM_THEME"] = general_stats["edges_by_type"].get("SEMANTIC_SIM_THEME", 0) + 1
                else:
                    general_stats["edges_by_type"][edge_type] = general_stats["edges_by_type"].get(edge_type, 0) + 1
            else:
                general_stats["edges_by_type"][edge_type] = general_stats["edges_by_type"].get(edge_type, 0) + 1

        # Claim-specific statistics
        n_claims = general_stats["nodes_by_type"].get("claim", 0)
        n_claim_sim = general_stats["edges_by_type"].get("CLAIM_SIM", 0)
        max_possible = n_claims * (n_claims - 1) // 2 if n_claims > 1 else 0
        general_stats["claim_statistics"] = {
            "n_claim_nodes": n_claims,
            "n_claim_sim_edges": n_claim_sim,
            "max_possible_claim_pairs": max_possible,
            "claim_graph_density": round(n_claim_sim / max_possible, 6) if max_possible > 0 else 0.0
        }

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

        # Claim statistics section
        claim_stats_dict = general_stats.get("claim_statistics", {})
        summary += "\n\nCLAIM STATISTICS\n" + '-' * 17 + "\n"
        summary += f"  Claim nodes: {claim_stats_dict.get('n_claim_nodes', 0)}\n"
        summary += f"  CLAIM_SIM edges (Claim↔Claim): {claim_stats_dict.get('n_claim_sim_edges', 0)}\n"
        summary += f"  Max possible claim pairs: {claim_stats_dict.get('max_possible_claim_pairs', 0)}\n"
        summary += f"  Claim graph density: {claim_stats_dict.get('claim_graph_density', 0.0):.6f}\n"

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
        with open(stats_dir / f"summary_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
        self.logger.info(f"Summary report saved to: {stats_dir}/summary_{timestamp}.txt")

        # 5. Generate HTML topic visualization
        chunk_ids = [c['chunk_id'] for c in chunks]
        chunk_texts = [c['text'] for c in chunks]
        html_path = generate_topic_html_report(
            graph=self.graph,
            topic_model=topic_model,
            topics=topics,
            probabilities=probabilities,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            output_dir=str(stats_dir),
            max_sample_chunks=100,
            logger=self.logger,
            graph_id=timestamp
        )
        self.logger.info(f"Topic visualization saved to: {html_path}")

    def _save_timing_report(self, total_build_time: float, graph_id: Optional[str] = None):
        """
        Save build timing report showing time and percentage for each step.

        Args:
            total_build_time: Total time for the build process in seconds
            graph_id: Graph identifier (timestamp) shared across all outputs for this build
        """
        import json
        from pathlib import Path
        from datetime import datetime

        stats_dir = Path("outputs/statistics")
        stats_dir.mkdir(parents=True, exist_ok=True)

        timestamp = graph_id if graph_id is not None else datetime.now().strftime("%Y%m%d_%H%M%S")

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
