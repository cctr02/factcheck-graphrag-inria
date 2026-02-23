"""
Retrieval module for fact-checking using the Knowledge Graph.

This module implements a hybrid retrieval strategy combining:
1. Semantic Similarity (embedding similarity)
2. Topic Distribution Similarity (Jensen-Shannon divergence)
3. Thematic Graph Traversal (BFS over theme graph)
4. Entity Graph Traversal (BFS over entity graph)
5. Claim-Document Retrieval (semantic similarity with claims, then linked documents)
6. Claim-Topic Traversal (claim-based + topic distribution weighted by traversal)
7. Claim-KeyPhrase Similarity (claim-based + KeyBERT phrase similarity)
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
import time
import random

# Import strategy modules
from fact_check_kg.retrieval.retrieval_propagation import (
    propagate_relevance_entities,
    propagate_relevance_themes
)
from fact_check_kg.retrieval.retrieval_semantic import retrieve_semantic as _retrieve_semantic
from fact_check_kg.retrieval.retrieval_topic import retrieve_topic_distribution as _retrieve_topic_distribution
from fact_check_kg.retrieval.retrieval_thematic import retrieve_thematic as _retrieve_thematic
from fact_check_kg.retrieval.retrieval_entity import retrieve_entity as _retrieve_entity, extract_entities
from fact_check_kg.retrieval.retrieval_claims import retrieve_claim_document as _retrieve_claim_document
from fact_check_kg.retrieval.retrieval_claim_topic import retrieve_claim_topic_traversal as _retrieve_claim_topic_traversal
from fact_check_kg.retrieval.retrieval_claim_keyphrase import retrieve_claim_keyphrase_similarity as _retrieve_claim_keyphrase_similarity


class HybridRetriever:
    """
    Hybrid retrieval system for fact-checking claims.

    Combines seven retrieval strategies:
    - Strategy 1 (Semantic Similarity): Direct embedding similarity
    - Strategy 2 (Topic Distribution Similarity): Jensen-Shannon divergence between topic distributions
    - Strategy 3 (Thematic Graph Traversal): BFS over theme graph
    - Strategy 4 (Entity Graph Traversal): BFS over entity graph
    - Strategy 5 (Claim-Document Retrieval): Chunks from documents linked to semantically similar claims
    - Strategy 6 (Claim-Topic Traversal): Claim traversal with topic distribution weighting
    - Strategy 7 (Claim-KeyPhrase Similarity): Claim traversal with KeyBERT phrase matching

    The final score is a weighted combination of all strategies (weights must sum to 1.0).
    """

    def __init__(
        self,
        graph: nx.Graph,
        topic_model: BERTopic,
        embedding_model: SentenceTransformer,
        nlp: spacy.Language,
        entity_types: List[str],
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        epsilon: float,
        zeta: float = 0.0,
        eta: float = 0.0,
        top_k: int = 10,
        top_docs: int = 50,
        theme_threshold: float = 0.01,
        entity_relevance_threshold: float = 0.1,
        theme_relevance_threshold: float = 0.1,
        claim_top_k: int = 5,
        explore_similar_claims: bool = True,
        max_claim_expansion_depth: int = 1,
        keyphrase_top_n: int = 5,
        keybert_model = None,
        graph_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
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
            alpha: Weight for semantic score (S1)
            beta: Weight for topic distribution score (S2)
            gamma: Weight for thematic score (S3)
            delta: Weight for entity score (S4)
            epsilon: Weight for claim-document score (S5)
            zeta: Weight for claim-topic traversal score (S6)
            eta: Weight for claim-keyphrase score (S7)
            top_k: Number of chunks to return
            top_docs: Number of top documents to return per strategy and overall
            theme_threshold: Minimum confidence for theme detection
            entity_relevance_threshold: Minimum relevance to continue entity traversal
            theme_relevance_threshold: Minimum relevance to continue theme traversal
            claim_top_k: Number of top claims to consider for Strategies 5, 6, 7
            explore_similar_claims: If True, explore claims connected via SIMILAR_CLAIM edges in Strategy 5
            max_claim_expansion_depth: Maximum depth for exploring SIMILAR_CLAIM edges (default: 1)
            keyphrase_top_n: Number of keyphrases to extract for Strategy 7
            keybert_model: Optional KeyBERT model for Strategy 7 (lazy-loaded if None)
            graph_id: Optional graph ID for loading cached data
            cache_dir: Optional directory containing FAISS embeddings cache
            models_dir: Optional directory containing topic models and probabilities
            logger: Optional logger instance
        """
        self.graph = graph
        self.topic_model = topic_model
        self.embedding_model = embedding_model
        self.nlp = nlp
        self.entity_types = entity_types

        # Weights (should sum to 1.0)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta

        # Parameters
        self.top_k = top_k
        self.top_docs = top_docs
        self.theme_threshold = theme_threshold
        self.entity_relevance_threshold = entity_relevance_threshold
        self.theme_relevance_threshold = theme_relevance_threshold
        self.claim_top_k = claim_top_k
        self.explore_similar_claims = explore_similar_claims
        self.max_claim_expansion_depth = max_claim_expansion_depth
        self.keyphrase_top_n = keyphrase_top_n

        # KeyBERT model for Strategy 7 (lazy-loaded if not provided)
        self.keybert_model = keybert_model

        # Cache parameters
        self.graph_id = graph_id
        self.cache_dir = cache_dir
        self.models_dir = models_dir

        self.logger = logger or logging.getLogger(__name__)

        # Pre-compute chunk embeddings and topic distributions for efficiency
        self.logger.info("Pre-computing chunk embeddings and topic distributions...")
        self._precompute_chunk_embeddings(graph_id, cache_dir)
        self._precompute_chunk_topic_distributions(graph_id, models_dir)
        self._precompute_claim_embeddings()
        # Note: Claim retrieval results are no longer precomputed — S5/S6/S7 traverse
        # REVIEWS → CONTAINS edges on-the-fly at retrieval time.

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
        from fact_check_kg.core.embedding_cache import EmbeddingCache

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

    def _precompute_chunk_topic_distributions(self, graph_id: Optional[str] = None, models_dir: Optional[str] = None):
        """Pre-compute topic probability distributions for all chunks."""
        from pathlib import Path

        # Try to load pre-computed probabilities from models directory
        if graph_id and models_dir:
            try:
                models_path = Path(models_dir)
                probs_file = models_path / f"topic_probs_{graph_id}.npz"

                if probs_file.exists():
                    self.logger.info(f"Loading topic probabilities from cache: {probs_file}")
                    data = np.load(probs_file)
                    cached_chunk_ids = data['chunk_ids'].tolist()
                    cached_probs = data['probabilities']

                    # Verify chunk IDs match
                    if cached_chunk_ids == self.chunk_nodes:
                        self.chunk_topic_distributions = cached_probs

                        # Create mapping from chunk_id to distribution index
                        self.chunk_to_dist_idx = {
                            chunk_id: idx
                            for idx, chunk_id in enumerate(self.chunk_nodes)
                        }

                        self.logger.info(f"Loaded topic distributions for {len(self.chunk_nodes)} chunks")
                        self.logger.info(f"Number of topics: {cached_probs.shape[1]}")
                        return
                    else:
                        self.logger.warning("Cached chunk IDs don't match current graph, recomputing...")
                else:
                    self.logger.warning(f"Topic probabilities cache not found: {probs_file}")
            except Exception as e:
                self.logger.warning(f"Error loading topic probabilities cache: {e}")

        # Fallback: compute topic distributions (slow!)
        self.logger.info(f"Computing topic distributions for {len(self.chunk_texts)} chunks...")
        self.logger.info("This may take 2-5 minutes depending on number of topics. Please wait...")
        self.logger.warning("Consider rebuilding graph to generate cached probabilities for faster loading")

        _, chunk_topic_probs = self.topic_model.transform(self.chunk_texts)

        self.logger.info("Topic distributions computed successfully")

        # Store as numpy array for efficient computation
        self.chunk_topic_distributions = chunk_topic_probs

        # Create mapping from chunk_id to distribution index
        self.chunk_to_dist_idx = {
            chunk_id: idx
            for idx, chunk_id in enumerate(self.chunk_nodes)
        }

        self.logger.info(f"Pre-computed topic distributions for {len(self.chunk_nodes)} chunks")
        self.logger.info(f"Number of topics: {chunk_topic_probs.shape[1]}")

    def _precompute_claim_embeddings(self):
        """
        Pre-compute embeddings for all claim nodes in the graph.

        Used by Strategy 5 (Claim-Document retrieval) to find semantically
        similar claims to the query.
        """
        # Get all claim nodes
        self.claim_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'claim'
        ]

        if not self.claim_nodes:
            self.logger.warning("No claim nodes found in graph. Strategy 5 will be disabled.")
            self.claim_embeddings = None
            self.claim_texts = []
            return

        self.claim_texts = [
            self.graph.nodes[n].get('text', '')
            for n in self.claim_nodes
        ]

        # Compute embeddings for all claims
        self.logger.info(f"Computing embeddings for {len(self.claim_nodes)} claims...")
        self.claim_embeddings = self.embedding_model.encode(
            self.claim_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.logger.info(f"Pre-computed embeddings for {len(self.claim_nodes)} claims")

    def retrieve_semantic(self, claim: str) -> Dict[str, float]:
        """Strategy 1: Semantic Similarity using embedding similarity."""
        return _retrieve_semantic(
            claim=claim,
            embedding_model=self.embedding_model,
            chunk_embeddings=self.chunk_embeddings,
            chunk_nodes=self.chunk_nodes,
            logger=self.logger
        )

    def retrieve_thematic(self, claim: str) -> Tuple[Dict[str, float], List[Dict]]:
        """Strategy 3: Thematic Graph Traversal using topic model and graph traversal."""
        return _retrieve_thematic(
            claim=claim,
            graph=self.graph,
            topic_model=self.topic_model,
            theme_threshold=self.theme_threshold,
            theme_relevance_threshold=self.theme_relevance_threshold,
            logger=self.logger
        )

    def retrieve_entity(self, claim: str) -> Tuple[Dict[str, float], Dict]:
        """Strategy 4: Entity Graph Traversal using graph traversal with relevance propagation."""
        return _retrieve_entity(
            claim=claim,
            graph=self.graph,
            nlp=self.nlp,
            entity_types=set(self.entity_types),
            entity_relevance_threshold=self.entity_relevance_threshold,
            logger=self.logger
        )

    def retrieve_topic_distribution(self, claim: str) -> Dict[str, float]:
        """Strategy 2: Topic Distribution Similarity using Jensen-Shannon divergence."""
        return _retrieve_topic_distribution(
            claim=claim,
            graph=self.graph,
            topic_model=self.topic_model,
            chunk_topic_distributions=self.chunk_topic_distributions,
            chunk_to_dist_idx=self.chunk_to_dist_idx,
            logger=self.logger
        )

    def retrieve_claim_document(self, query: str) -> Tuple[Dict[str, float], List[Dict]]:
        """Strategy 5: Claim-Based Retrieval via REVIEWS → CONTAINS graph traversal."""
        return _retrieve_claim_document(
            query=query,
            graph=self.graph,
            embedding_model=self.embedding_model,
            claim_embeddings=self.claim_embeddings,
            claim_nodes=self.claim_nodes,
            claim_texts=self.claim_texts,
            claim_top_k=self.claim_top_k,
            logger=self.logger,
        )

    def retrieve_claim_topic_traversal(self, query: str) -> Tuple[Dict[str, float], List[Dict]]:
        """Strategy 6: Claim-Based Topic Distribution Traversal via REVIEWS → CONTAINS."""
        return _retrieve_claim_topic_traversal(
            query=query,
            graph=self.graph,
            topic_model=self.topic_model,
            embedding_model=self.embedding_model,
            chunk_topic_distributions=self.chunk_topic_distributions,
            chunk_to_dist_idx=self.chunk_to_dist_idx,
            claim_embeddings=self.claim_embeddings,
            claim_nodes=self.claim_nodes,
            claim_texts=self.claim_texts,
            claim_top_k=self.claim_top_k,
            logger=self.logger,
        )

    def retrieve_claim_keyphrase_similarity(self, query: str) -> Tuple[Dict[str, float], List[Dict]]:
        """Strategy 7: Claim-Based KeyBERT Phrase Similarity via REVIEWS → CONTAINS."""
        # Lazy-load KeyBERT model if not provided
        if self.keybert_model is None:
            from keybert import KeyBERT
            self.logger.info("Initializing KeyBERT model for Strategy 7...")
            self.keybert_model = KeyBERT(model=self.embedding_model)

        return _retrieve_claim_keyphrase_similarity(
            query=query,
            graph=self.graph,
            embedding_model=self.embedding_model,
            keybert_model=self.keybert_model,
            claim_embeddings=self.claim_embeddings,
            claim_nodes=self.claim_nodes,
            claim_texts=self.claim_texts,
            claim_top_k=self.claim_top_k,
            keyphrase_top_n=self.keyphrase_top_n,
            logger=self.logger,
        )

    def _aggregate_to_documents(
        self,
        chunk_results: List[Dict],
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Aggregate chunk-level scores to document-level scores using max pooling.

        Max pooling: the document score on each dimension equals the maximum score
        of any of its chunks on that dimension.  This prevents documents with more
        chunks from being systematically favoured — only the single best piece of
        evidence from each document contributes.

        doc_score_total    = max(chunk.score_total)  over all chunks in the doc
        doc_score_<strat>  = max(chunk.score_<strat>) over all chunks in the doc

        Returns:
            - doc_results: all documents with at least one scored chunk, sorted by
              doc_score_total descending (no length limit — caller truncates)
            - per_strategy_docs: dict mapping strategy key to list of docs (only those
              with a non-zero strategy score), sorted by that strategy score descending,
              each truncated to self.top_docs
        """
        from collections import defaultdict

        # Group chunks by document
        doc_chunks: Dict[str, List[Dict]] = defaultdict(list)
        for chunk in chunk_results:
            doc_chunks[chunk['doc_id']].append(chunk)

        doc_results = []
        for doc_id, chunks in doc_chunks.items():
            best_chunk = max(chunks, key=lambda c: c['score_total'])
            doc_results.append({
                'doc_id': doc_id,
                'doc_score_total': best_chunk['score_total'],
                'doc_score_sem': max(c['score_sem'] for c in chunks),
                'doc_score_topic_dist': max(c['score_topic_dist'] for c in chunks),
                'doc_score_theme': max(c['score_theme'] for c in chunks),
                'doc_score_entity': max(c['score_entity'] for c in chunks),
                'doc_score_claim_doc': max(c.get('score_claim_doc', 0.0) for c in chunks),
                'doc_score_claim_topic': max(c.get('score_claim_topic', 0.0) for c in chunks),
                'doc_score_claim_keyphrase': max(c.get('score_claim_keyphrase', 0.0) for c in chunks),
                'n_chunks_scored': len(chunks),
                'best_chunk_id': best_chunk['chunk_id'],
                'best_chunk_text': best_chunk['text'],
            })

        # Overall ranking by final score
        doc_results.sort(key=lambda d: d['doc_score_total'], reverse=True)

        # Per-strategy rankings (only docs with non-zero score for that strategy)
        strategy_keys = [
            ('S1_semantic',         'doc_score_sem'),
            ('S2_topic_dist',       'doc_score_topic_dist'),
            ('S3_thematic',         'doc_score_theme'),
            ('S4_entity',           'doc_score_entity'),
            ('S5_claim_doc',        'doc_score_claim_doc'),
            ('S6_claim_topic',      'doc_score_claim_topic'),
            ('S7_claim_keyphrase',  'doc_score_claim_keyphrase'),
        ]
        per_strategy_docs: Dict[str, List[Dict]] = {}
        for strategy_name, score_key in strategy_keys:
            per_strategy_docs[strategy_name] = sorted(
                [d for d in doc_results if d[score_key] > 0.0],
                key=lambda d: d[score_key],
                reverse=True
            )[:self.top_docs]

        return doc_results, per_strategy_docs

    def retrieve(self, claim: str) -> List[Dict]:
        """
        Hybrid retrieval: combine all seven strategies.

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
            - score_claim_doc: float (claim-document score)
            - score_claim_topic: float (claim-topic traversal score)
            - score_claim_keyphrase: float (claim-keyphrase score)
            - claim: str (the original claim)
            - config: Dict (complete configuration used for retrieval, included in first result only)
            - retrieval_timestamp: str (ISO format timestamp, included in first result only)
        """
        # Start timing the retrieval process
        start_time = time.time()
        strategy_timings = {}

        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"RETRIEVAL FOR CLAIM: {claim}")
        self.logger.info("=" * 60)

        # Strategy 1: Semantic Similarity
        t0 = time.time()
        S1 = self.retrieve_semantic(claim)
        strategy_timings['S1_semantic'] = time.time() - t0

        # Strategy 2: Topic Distribution Similarity
        t0 = time.time()
        S2 = self.retrieve_topic_distribution(claim)
        strategy_timings['S2_topic_dist'] = time.time() - t0

        # Strategy 3: Thematic Graph Traversal
        t0 = time.time()
        S3, detected_themes = self.retrieve_thematic(claim)
        strategy_timings['S3_thematic'] = time.time() - t0

        # Strategy 4: Entity Graph Traversal
        t0 = time.time()
        S4, entity_info_s4 = self.retrieve_entity(claim)
        strategy_timings['S4_entity'] = time.time() - t0

        # Strategy 5: Claim-Based Retrieval (using pre-computed results)
        t0 = time.time()
        S5, matched_claims = self.retrieve_claim_document(claim)
        strategy_timings['S5_claim_doc'] = time.time() - t0

        # Strategy 6: Claim-Based Topic Distribution Traversal
        t0 = time.time()
        S6, matched_claims_s6 = self.retrieve_claim_topic_traversal(claim)
        strategy_timings['S6_claim_topic'] = time.time() - t0

        # Strategy 7: Claim-Based KeyBERT Phrase Similarity
        t0 = time.time()
        S7, matched_claims_s7 = self.retrieve_claim_keyphrase_similarity(claim)
        strategy_timings['S7_claim_keyphrase'] = time.time() - t0

        # Fusion: combine scores
        all_chunks = set(S1.keys()) | set(S2.keys()) | set(S3.keys()) | set(S4.keys()) | set(S5.keys()) | set(S6.keys()) | set(S7.keys())

        self.logger.info("\n--- Fusion ---")
        self.logger.info(f"Total unique chunks: {len(all_chunks)}")
        self.logger.info(f"Weights: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, delta={self.delta}, epsilon={self.epsilon}, zeta={self.zeta}, eta={self.eta}")

        results = []
        for chunk_id in all_chunks:
            score_sem = S1.get(chunk_id, 0)
            score_topic_dist = S2.get(chunk_id, 0)
            score_theme = S3.get(chunk_id, 0)
            score_entity = S4.get(chunk_id, 0)
            score_claim_doc = S5.get(chunk_id, 0)
            score_claim_topic = S6.get(chunk_id, 0)
            score_claim_keyphrase = S7.get(chunk_id, 0)

            # Final weighted score
            final_score = (
                self.alpha * score_sem +
                self.beta * score_topic_dist +
                self.gamma * score_theme +
                self.delta * score_entity +
                self.epsilon * score_claim_doc +
                self.zeta * score_claim_topic +
                self.eta * score_claim_keyphrase
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
                'score_topic_dist': score_topic_dist,
                'score_claim_doc': score_claim_doc,
                'score_claim_topic': score_claim_topic,
                'score_claim_keyphrase': score_claim_keyphrase
            })

        # Sort by total score
        results.sort(key=lambda x: x['score_total'], reverse=True)

        # Document-level aggregation (uses ALL chunks, not just top-k)
        doc_results, per_strategy_doc_results = self._aggregate_to_documents(results)
        top_doc_results = doc_results[:self.top_docs]

        # Sample random chunks from outside top-K for comparison in reports
        remaining = results[self.top_k:]
        random_samples = random.sample(remaining, min(5, len(remaining))) if remaining else []

        # Return top-K
        top_results = results[:self.top_k]

        # Add claim, matched claims, detected themes, and entity info to results
        for i, result in enumerate(top_results):
            result['claim'] = claim
            if i == 0:
                if matched_claims:
                    result['matched_claims_s5'] = matched_claims
                if matched_claims_s6:
                    result['matched_claims_s6'] = matched_claims_s6
                if matched_claims_s7:
                    result['matched_claims_s7'] = matched_claims_s7
                if detected_themes:
                    result['detected_themes'] = detected_themes
                result['doc_results'] = top_doc_results
                result['per_strategy_doc_results'] = per_strategy_doc_results
                if entity_info_s4:
                    result['entity_info_s4'] = entity_info_s4
                if random_samples:
                    result['random_samples'] = random_samples

        # Log matched claims summary
        if matched_claims:
            self.logger.info(f"\n--- Matched Claims (S5) ---")
            for mc in matched_claims:
                source_tag = f"[{mc.get('source', 'direct')}]"
                self.logger.info(f"  {source_tag} Claim: {mc['claim_node']} (sim: {mc['query_similarity']:.3f})")
                preview = mc['claim_text'][:80] + "..." if len(mc['claim_text']) > 80 else mc['claim_text']
                self.logger.info(f"    Text: {preview}")
                self.logger.info(f"    Chunks found via REVIEWS: {mc['chunks_found']}")

        if matched_claims_s6:
            self.logger.info(f"\n--- Matched Claims (S6 Topic) ---")
            for mc in matched_claims_s6:
                source_tag = f"[{mc.get('source', 'direct')}]"
                self.logger.info(f"  {source_tag} Claim: {mc['claim_node']} (sim: {mc['query_similarity']:.3f})")
                preview = mc['claim_text'][:80] + "..." if len(mc['claim_text']) > 80 else mc['claim_text']
                self.logger.info(f"    Text: {preview}")
                self.logger.info(f"    Rating: {mc.get('rating', 'N/A')} | Chunks found: {mc['chunks_found']}")

        # S7 returns a dict with 'matched_claims' key
        s7_claims = matched_claims_s7.get('matched_claims', []) if isinstance(matched_claims_s7, dict) else matched_claims_s7
        if s7_claims:
            s7_keyphrases = matched_claims_s7.get('query_keyphrases', []) if isinstance(matched_claims_s7, dict) else []
            if s7_keyphrases:
                self.logger.info(f"\n--- S7 Query Keyphrases ---")
                for phrase, score in s7_keyphrases:
                    self.logger.info(f"  '{phrase}': {score:.3f}")
            self.logger.info(f"\n--- Matched Claims (S7 KeyPhrase) ---")
            for mc in s7_claims:
                source_tag = f"[{mc.get('source', 'direct')}]"
                self.logger.info(f"  {source_tag} Claim: {mc['claim_node']} (sim: {mc['query_similarity']:.3f})")
                preview = mc['claim_text'][:80] + "..." if len(mc['claim_text']) > 80 else mc['claim_text']
                self.logger.info(f"    Text: {preview}")
                self.logger.info(f"    Rating: {mc.get('rating', 'N/A')} | Chunks found: {mc['chunks_found']}")

        self.logger.info(f"\nTop {len(top_results)} results:")
        for i, result in enumerate(top_results[:5], 1):
            self.logger.info(
                f"{i}. {result['chunk_id']} (doc: {result['doc_id']}) | "
                f"Total: {result['score_total']:.3f} | "
                f"Sem: {result['score_sem']:.3f} | "
                f"Theme: {result['score_theme']:.3f} | "
                f"Entity: {result['score_entity']:.3f} | "
                f"TopicDist: {result['score_topic_dist']:.3f} | "
                f"ClaimDoc: {result['score_claim_doc']:.3f} | "
                f"ClaimTopic: {result['score_claim_topic']:.3f} | "
                f"ClaimKP: {result['score_claim_keyphrase']:.3f}"
            )

        # Display text previews of top 3 results
        self.logger.info("\nText previews of top 3 results:")
        for i, result in enumerate(top_results[:3], 1):
            text_preview = result['text'][:150].replace('\n', ' ')
            if len(result['text']) > 150:
                text_preview += "..."
            self.logger.info(f"{i}. [{result['chunk_id']}] {text_preview}")

        # Log document-level summary
        self.logger.info(f"\n--- Document-level results (max-pooling) ---")
        self.logger.info(f"  Total documents with at least one scored chunk: {len(doc_results)}")
        self.logger.info(f"  Top {min(self.top_docs, len(top_doc_results))} documents (by final score):")
        for i, doc in enumerate(top_doc_results[:5], 1):
            self.logger.info(
                f"  {i}. {doc['doc_id']} | total={doc['doc_score_total']:.3f} | "
                f"n_chunks={doc['n_chunks_scored']} | best={doc['best_chunk_id']}"
            )

        # Calculate and display retrieval time
        retrieval_time = time.time() - start_time
        strategy_timings['total'] = retrieval_time

        self.logger.info(f"\n--- Strategy Timings ---")
        self.logger.info(f"  S1 (Semantic):        {strategy_timings['S1_semantic']:.4f}s")
        self.logger.info(f"  S2 (Topic Dist):      {strategy_timings['S2_topic_dist']:.4f}s")
        self.logger.info(f"  S3 (Thematic):        {strategy_timings['S3_thematic']:.4f}s")
        self.logger.info(f"  S4 (Entity):          {strategy_timings['S4_entity']:.4f}s")
        self.logger.info(f"  S5 (Claim-Doc):       {strategy_timings['S5_claim_doc']:.4f}s")
        self.logger.info(f"  S6 (Claim-Topic):     {strategy_timings['S6_claim_topic']:.4f}s")
        self.logger.info(f"  S7 (Claim-KeyPhrase): {strategy_timings['S7_claim_keyphrase']:.4f}s")
        self.logger.info(f"  Total:                {retrieval_time:.4f}s")

        # Add timing info to first result
        if top_results:
            top_results[0]['strategy_timings'] = strategy_timings

        self.logger.info("=" * 60)

        return top_results
