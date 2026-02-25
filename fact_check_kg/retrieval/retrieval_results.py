"""
Enhanced retrieval results module with detailed scoring and ranking.

This module provides comprehensive result formatting, multiple ranking methods,
and detailed score breakdowns for retrieval results.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Maps per_strategy_doc_results strategy keys to the doc-level score field
_STRATEGY_SCORE_KEY: Dict[str, str] = {
    'S1_semantic':        'doc_score_sem',
    'S2_topic_dist':      'doc_score_topic_dist',
    'S3_thematic':        'doc_score_theme',
    'S4_entity':          'doc_score_entity',
    'S5_claim_doc':       'doc_score_claim_doc',
    'S6_claim_topic':     'doc_score_claim_topic',
    'S7_claim_keyphrase': 'doc_score_claim_keyphrase',
}


class RetrievalResultsManager:
    """
    Manager for retrieval results with detailed scoring and multiple rankings.

    Provides:
    - Detailed score breakdowns
    - Multiple ranking methods (total, by component, aggregation)
    - Export to JSON and text formats
    - Statistical analysis of results
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the results manager.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.results_dir = Path("outputs/retrieval_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_retrieval_filename(self, graph_id: str) -> str:
        """
        Get next available filename for retrieval results.

        Args:
            graph_id: Graph ID to use as base

        Returns:
            Filename base like "results_{graph_id}_1"
        """
        # Find existing retrieval files for this graph_id
        pattern = f"results_{graph_id}_*.json"
        existing_files = list(self.results_dir.glob(pattern))

        if not existing_files:
            return f"results_{graph_id}_1"

        # Extract numbers from filenames
        numbers = []
        for file in existing_files:
            # Extract number from filename like "results_20251118_155115_1.json"
            stem = file.stem  # Remove .json
            parts = stem.split('_')
            if len(parts) >= 4:  # results_YYYYMMDD_HHMMSS_N
                try:
                    numbers.append(int(parts[-1]))
                except ValueError:
                    continue

        # Get next number
        next_num = max(numbers) + 1 if numbers else 1
        return f"results_{graph_id}_{next_num}"

    def format_results_detailed(
        self,
        claim: str,
        results: List[Dict],
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        epsilon: float = 0.0,
        zeta: float = 0.0,
        eta: float = 0.0,
        config: dict = None,
        config_retrieval: dict = None,
        matched_claims: List[Dict] = None,
        matched_claims_s5: List[Dict] = None,
        matched_claims_s6: List[Dict] = None,
        matched_claims_s7: List[Dict] = None,
        detected_themes: List[Dict] = None,
        extracted_entities: Dict = None,
        random_samples: List[Dict] = None,
        doc_results: List[Dict] = None,
        per_strategy_doc_results: Dict[str, List[Dict]] = None
    ) -> Dict:
        """
        Format results with detailed scoring breakdown.

        Args:
            claim: The search claim
            results: List of result dictionaries
            alpha: Semantic weight (S1)
            beta: Topic distribution weight (S2)
            gamma: Thematic weight (S3)
            delta: Entity weight (S4)
            epsilon: Claim-document weight (S5)
            zeta: Claim-topic traversal weight (S6)
            eta: Claim-keyphrase weight (S7)
            config: Optional common configuration dict
            config_retrieval: Optional retrieval-specific configuration dict
            matched_claims: Optional list of matched claims (S5 query expansion)
            detected_themes: Optional list of themes detected from the query (S3)
            random_samples: Optional list of random chunks sampled outside top-K for comparison

        Returns:
            Formatted results dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        formatted = {
            "metadata": {
                "timestamp": timestamp,
                "claim": claim,
                "weights": {
                    "alpha_semantic": alpha,
                    "beta_topic_dist": beta,
                    "gamma_thematic": gamma,
                    "delta_entity": delta,
                    "epsilon_claim_doc": epsilon,
                    "zeta_claim_topic": zeta,
                    "eta_claim_keyphrase": eta
                },
                "total_results": len(results)
            },
            "results": []
        }

        # Add configurations to metadata if provided
        if config is not None:
            formatted["metadata"]["config"] = config
        if config_retrieval is not None:
            formatted["metadata"]["config_retrieval"] = config_retrieval

        # Add matched claims — prefer per-strategy lists (s5/s6/s7), fall back to legacy matched_claims
        def _format_claims(claims):
            return [
                {
                    "claim_id": c.get("claim_id", c.get("claim_node", "")),
                    "claim_text": c.get("claim_text", ""),
                    "similarity": round(c.get("similarity", c.get("query_similarity", 0.0)), 4),
                    "rating": c.get("rating", ""),
                    "chunks_found": c.get("chunks_found", 0),
                    "source": c.get("source", "direct"),
                }
                for c in claims
            ]

        effective_s5 = matched_claims_s5 or matched_claims  # backward-compat
        if effective_s5:
            formatted["metadata"]["matched_claims_s5"] = _format_claims(effective_s5)
        if matched_claims_s6:
            formatted["metadata"]["matched_claims_s6"] = _format_claims(matched_claims_s6)
        if matched_claims_s7:
            s7_list = matched_claims_s7.get('matched_claims', []) if isinstance(matched_claims_s7, dict) else matched_claims_s7
            formatted["metadata"]["matched_claims_s7"] = _format_claims(s7_list)

        # Add detected themes (S3 query classification) if provided
        if detected_themes:
            formatted["metadata"]["detected_themes"] = [
                {
                    "theme_node": t.get("theme_node", ""),
                    "topic_id": t.get("topic_id", -1),
                    "label": t.get("label", ""),
                    "probability": t.get("probability", 0.0),
                    "keywords": t.get("keywords", [])
                }
                for t in detected_themes
            ]

        # Add extracted entities (S4 entity extraction) if provided
        if extracted_entities:
            formatted["metadata"]["extracted_entities"] = {
                "extracted": [
                    {"text": text, "type": etype}
                    for text, etype in extracted_entities.get("extracted", [])
                ],
                "found_in_graph": extracted_entities.get("found_in_graph", []),
                "not_in_graph": extracted_entities.get("not_in_graph", []),
            }

        # Add random sample chunks for comparison if provided
        if random_samples:
            formatted["random_samples"] = [
                {
                    "chunk_id": r['chunk_id'],
                    "document_id": r['doc_id'],
                    "text_preview": r['text'][:200] + "..." if len(r['text']) > 200 else r['text'],
                    "scores": {
                        "total": round(r['score_total'], 4),
                        "semantic": round(r['score_sem'], 4),
                        "thematic": round(r['score_theme'], 4),
                        "entity": round(r['score_entity'], 4),
                        "topic_dist": round(r['score_topic_dist'], 4),
                        "claim_doc": round(r.get('score_claim_doc', 0), 4),
                        "claim_topic": round(r.get('score_claim_topic', 0), 4),
                        "claim_keyphrase": round(r.get('score_claim_keyphrase', 0), 4)
                    }
                }
                for r in random_samples
            ]

        # Document-level results
        if doc_results is not None:
            formatted["doc_results"] = [
                {
                    "rank": i,
                    "doc_id": d['doc_id'],
                    "n_chunks_scored": d['n_chunks_scored'],
                    "best_chunk_id": d['best_chunk_id'],
                    "best_chunk_text_preview": d['best_chunk_text'][:200] + "..." if len(d['best_chunk_text']) > 200 else d['best_chunk_text'],
                    "scores": {
                        "total": round(d['doc_score_total'], 4),
                        "semantic": round(d['doc_score_sem'], 4),
                        "topic_dist": round(d['doc_score_topic_dist'], 4),
                        "thematic": round(d['doc_score_theme'], 4),
                        "entity": round(d['doc_score_entity'], 4),
                        "claim_doc": round(d['doc_score_claim_doc'], 4),
                        "claim_topic": round(d['doc_score_claim_topic'], 4),
                        "claim_keyphrase": round(d['doc_score_claim_keyphrase'], 4),
                    }
                }
                for i, d in enumerate(doc_results, 1)
            ]

        if per_strategy_doc_results is not None:
            formatted["per_strategy_doc_results"] = {
                strategy: [
                    {
                        "rank": i,
                        "doc_id": d['doc_id'],
                        "n_chunks_scored": d['n_chunks_scored'],
                        "score": round(d[_STRATEGY_SCORE_KEY[strategy]], 4),
                        "best_chunk_id": d['best_chunk_id'],
                    }
                    for i, d in enumerate(docs, 1)
                ]
                for strategy, docs in per_strategy_doc_results.items()
            }

        for rank, result in enumerate(results, 1):
            formatted_result = {
                "rank": rank,
                "chunk_id": result['chunk_id'],
                "document_id": result['doc_id'],
                "text_preview": result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                "full_text": result['text'],  # Store full text
                "text_length": len(result['text']),
                "scores": {
                    "total": round(result['score_total'], 4),
                    "semantic": round(result['score_sem'], 4),
                    "thematic": round(result['score_theme'], 4),
                    "entity": round(result['score_entity'], 4),
                    "topic_dist": round(result['score_topic_dist'], 4),
                    "claim_doc": round(result.get('score_claim_doc', 0), 4),
                    "claim_topic": round(result.get('score_claim_topic', 0), 4),
                    "claim_keyphrase": round(result.get('score_claim_keyphrase', 0), 4)
                },
                "score_contributions": {
                    "semantic_contribution": round(result['score_sem'] * alpha, 4),
                    "thematic_contribution": round(result['score_theme'] * gamma, 4),
                    "entity_contribution": round(result['score_entity'] * delta, 4),
                    "topic_dist_contribution": round(result['score_topic_dist'] * beta, 4),
                    "claim_doc_contribution": round(result.get('score_claim_doc', 0) * epsilon, 4),
                    "claim_topic_contribution": round(result.get('score_claim_topic', 0) * zeta, 4),
                    "claim_keyphrase_contribution": round(result.get('score_claim_keyphrase', 0) * eta, 4)
                },
                "source_query": result.get('source_query', 'original_query'),
                "score_explanation": self._generate_score_explanation(result, alpha, beta, gamma, delta, epsilon, zeta, eta)
            }
            formatted["results"].append(formatted_result)

        return formatted

    def _generate_score_explanation(
        self,
        result: Dict,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        epsilon: float = 0.0,
        zeta: float = 0.0,
        eta: float = 0.0
    ) -> str:
        """
        Generate human-readable explanation of score calculation.

        Args:
            result: Result dictionary
            alpha: Semantic weight (S1)
            beta: Topic distribution weight (S2)
            gamma: Thematic weight (S3)
            delta: Entity weight (S4)
            epsilon: Claim-document weight (S5)
            zeta: Claim-topic traversal weight (S6)
            eta: Claim-keyphrase weight (S7)

        Returns:
            Score explanation string
        """
        total = result['score_total']

        explanation = f"Total score: {total:.4f} = "
        explanation += f"({result['score_sem']:.4f} × {alpha}) + "
        explanation += f"({result['score_theme']:.4f} × {gamma}) + "
        explanation += f"({result['score_entity']:.4f} × {delta}) + "
        explanation += f"({result['score_topic_dist']:.4f} × {beta}) + "
        explanation += f"({result.get('score_claim_doc', 0):.4f} × {epsilon}) + "
        explanation += f"({result.get('score_claim_topic', 0):.4f} × {zeta}) + "
        explanation += f"({result.get('score_claim_keyphrase', 0):.4f} × {eta})"

        # Identify dominant component
        components = [
            ("semantic", result['score_sem'] * alpha),
            ("topic_dist", result['score_topic_dist'] * beta),
            ("thematic", result['score_theme'] * gamma),
            ("entity", result['score_entity'] * delta),
            ("claim_doc", result.get('score_claim_doc', 0) * epsilon),
            ("claim_topic", result.get('score_claim_topic', 0) * zeta),
            ("claim_keyphrase", result.get('score_claim_keyphrase', 0) * eta),
        ]
        dominant = max(components, key=lambda x: x[1])
        explanation += f"\nDominant component: {dominant[0]} ({dominant[1]:.4f})"

        return explanation

    def create_multiple_rankings(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create multiple rankings based on different criteria.

        Args:
            results: List of result dictionaries

        Returns:
            Dictionary with different rankings
        """
        rankings = {}

        # 1. Ranking by total score (default)
        rankings["by_total_score"] = sorted(
            results,
            key=lambda x: x['score_total'],
            reverse=True
        )

        # 2. Ranking by semantic score
        rankings["by_semantic_score"] = sorted(
            results,
            key=lambda x: x['score_sem'],
            reverse=True
        )

        # 3. Ranking by thematic score
        rankings["by_thematic_score"] = sorted(
            results,
            key=lambda x: x['score_theme'],
            reverse=True
        )

        # 4. Ranking by entity score
        rankings["by_entity_score"] = sorted(
            results,
            key=lambda x: x['score_entity'],
            reverse=True
        )

        # 5. Ranking by topic distribution score
        rankings["by_topic_dist_score"] = sorted(
            results,
            key=lambda x: x['score_topic_dist'],
            reverse=True
        )

        # 6. Ranking by score variance (most balanced)
        rankings["by_balance"] = sorted(
            results,
            key=lambda x: self._calculate_score_variance(x),
            reverse=False  # Lower variance = more balanced
        )

        # 7. Ranking by maximum component score
        rankings["by_max_component"] = sorted(
            results,
            key=lambda x: max(x['score_sem'], x['score_theme'], x['score_entity'], x['score_topic_dist'], x.get('score_claim_doc', 0)),
            reverse=True
        )

        return rankings

    def _calculate_score_variance(self, result: Dict) -> float:
        """
        Calculate variance of score components.

        Args:
            result: Result dictionary

        Returns:
            Variance value
        """
        scores = [result['score_sem'], result['score_theme'], result['score_entity'], result['score_topic_dist'], result.get('score_claim_doc', 0)]
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance

    def generate_statistics(self, results: List[Dict]) -> Dict:
        """
        Generate statistical analysis of results.

        Args:
            results: List of result dictionaries

        Returns:
            Statistics dictionary
        """
        if not results:
            return {}

        stats = {
            "total_results": len(results),
            "score_statistics": {},
            "component_analysis": {}
        }

        # Score statistics
        total_scores = [r['score_total'] for r in results]
        sem_scores = [r['score_sem'] for r in results]
        theme_scores = [r['score_theme'] for r in results]
        entity_scores = [r['score_entity'] for r in results]
        topic_dist_scores = [r['score_topic_dist'] for r in results]
        claim_doc_scores = [r.get('score_claim_doc', 0) for r in results]
        claim_topic_scores = [r.get('score_claim_topic', 0) for r in results]
        claim_keyphrase_scores = [r.get('score_claim_keyphrase', 0) for r in results]

        for name, scores in [
            ("total", total_scores),
            ("semantic", sem_scores),
            ("thematic", theme_scores),
            ("entity", entity_scores),
            ("topic_dist", topic_dist_scores),
            ("claim_doc", claim_doc_scores),
            ("claim_topic", claim_topic_scores),
            ("claim_keyphrase", claim_keyphrase_scores)
        ]:
            stats["score_statistics"][name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "median": sorted(scores)[len(scores) // 2]
            }

        # Component dominance analysis
        for result in results:
            scores = {
                "semantic": result['score_sem'],
                "thematic": result['score_theme'],
                "entity": result['score_entity'],
                "topic_dist": result['score_topic_dist'],
                "claim_doc": result.get('score_claim_doc', 0),
                "claim_topic": result.get('score_claim_topic', 0),
                "claim_keyphrase": result.get('score_claim_keyphrase', 0)
            }
            dominant = max(scores, key=scores.get)
            stats["component_analysis"][dominant] = stats["component_analysis"].get(dominant, 0) + 1

        return stats

    def save_results(
        self,
        claim: str,
        results: List[Dict],
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        rankings: Dict[str, List[Dict]] = None,
        graph_id: str = None,
        custom_name: str = None,
        config: dict = None,
        config_retrieval: dict = None,
        matched_claims: List[Dict] = None,
        matched_claims_s5: List[Dict] = None,
        matched_claims_s6: List[Dict] = None,
        matched_claims_s7: List[Dict] = None,
        epsilon: float = 0.0,
        zeta: float = 0.0,
        eta: float = 0.0,
        detected_themes: List[Dict] = None,
        extracted_entities: Dict = None,
        random_samples: List[Dict] = None,
        doc_results: List[Dict] = None,
        per_strategy_doc_results: Dict[str, List[Dict]] = None
    ) -> Tuple[str, str]:
        """
        Save results to JSON and text files.

        Args:
            claim: The search claim
            results: List of result dictionaries
            alpha: Semantic weight (S1)
            beta: Topic distribution weight (S2)
            gamma: Thematic weight (S3)
            delta: Entity weight (S4)
            rankings: Optional pre-computed rankings
            graph_id: Graph ID to use as base timestamp
            custom_name: Custom name for this retrieval (e.g., "test1")
            config: Optional common configuration dict
            config_retrieval: Optional retrieval-specific configuration dict
            matched_claims: Optional list of matched claims for query expansion
            epsilon: Claim-document weight (S5)
            zeta: Claim-topic traversal weight (S6)
            eta: Claim-keyphrase weight (S7)
            detected_themes: Optional list of themes detected from the query (S3)
            random_samples: Optional list of random chunks sampled outside top-K for comparison

        Returns:
            Tuple of (json_path, text_path)
        """
        # Determine filename based on graph_id and custom_name
        if custom_name:
            # Use custom name: results_{graph_id}_{custom_name}
            filename_base = f"results_{graph_id}_{custom_name}" if graph_id else f"results_{custom_name}"
        elif graph_id:
            # Auto-increment: results_{graph_id}_1, results_{graph_id}_2, etc.
            filename_base = self._get_next_retrieval_filename(graph_id)
        else:
            # Fallback to timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"results_{timestamp}"

        # Format results (include configs if provided)
        formatted = self.format_results_detailed(
            claim, results, alpha, beta, gamma, delta,
            epsilon=epsilon, zeta=zeta, eta=eta,
            config=config, config_retrieval=config_retrieval,
            matched_claims=matched_claims,
            matched_claims_s5=matched_claims_s5,
            matched_claims_s6=matched_claims_s6,
            matched_claims_s7=matched_claims_s7,
            detected_themes=detected_themes,
            extracted_entities=extracted_entities,
            random_samples=random_samples,
            doc_results=doc_results,
            per_strategy_doc_results=per_strategy_doc_results,
        )

        # Create rankings if not provided
        if rankings is None:
            rankings = self.create_multiple_rankings(results)

        # Add rankings to formatted results
        formatted["rankings"] = {
            name: [
                {
                    "rank": i + 1,
                    "chunk_id": r['chunk_id'],
                    "score": round(r.get('score_total', 0), 4)
                }
                for i, r in enumerate(ranking[:10])  # Top 10 for each ranking
            ]
            for name, ranking in rankings.items()
        }

        # Add statistics
        formatted["statistics"] = self.generate_statistics(results)

        # Save JSON
        json_path = self.results_dir / f"{filename_base}.json"
        with open(json_path, 'w') as f:
            json.dump(formatted, f, indent=2)

        # Generate text report
        text_report = self._generate_text_report(formatted)
        text_path = self.results_dir / f"{filename_base}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)

        self.logger.info(f"Results saved to:")
        self.logger.info(f"  JSON: {json_path}")
        self.logger.info(f"  Text: {text_path}")

        return str(json_path), str(text_path)

    def _generate_text_report(
        self,
        formatted: Dict,
    ) -> str:
        """
        Generate human-readable text report.

        Structure:
          1. Header (claim, weights, counts)
          2. Query analysis (themes, entities, matched claims)
          3. Statistics (chunk-level score distribution)
          4. Per-strategy rankings: chunks + docs for each S1–S7
          5. Final hybrid aggregation: chunks + docs ranked by total score
          6. Random samples (if any)
          7. All retrieved chunks (full text)
        """
        H1 = "=" * 80   # major section separator
        H2 = "-" * 80   # minor separator

        # Strategy definitions: (key, label, weight_key, chunk_score_key, symbol)
        STRATEGIES = [
            ('S1_semantic',        'S1 Semantic',        'alpha_semantic',      'semantic',        'α'),
            ('S2_topic_dist',      'S2 Topic Dist',      'beta_topic_dist',     'topic_dist',      'β'),
            ('S3_thematic',        'S3 Thematic',        'gamma_thematic',      'thematic',        'γ'),
            ('S4_entity',          'S4 Entity',          'delta_entity',        'entity',          'δ'),
            ('S5_claim_doc',       'S5 Claim-Doc',       'epsilon_claim_doc',   'claim_doc',       'ε'),
            ('S6_claim_topic',     'S6 Claim-Topic',     'zeta_claim_topic',    'claim_topic',     'ζ'),
            ('S7_claim_keyphrase', 'S7 Claim-KP',        'eta_claim_keyphrase', 'claim_keyphrase', 'η'),
        ]
        # Short labels for score columns
        SCORE_COL = {
            'semantic': 'S1', 'topic_dist': 'S2', 'thematic': 'S3',
            'entity': 'S4', 'claim_doc': 'S5', 'claim_topic': 'S6', 'claim_keyphrase': 'S7',
        }

        report = []
        meta = formatted['metadata']
        weights = meta['weights']
        all_chunks = formatted.get("results", [])
        doc_res = formatted.get("doc_results", [])
        per_strat_docs = formatted.get("per_strategy_doc_results", {})

        # ── 1. HEADER ─────────────────────────────────────────────────────────
        report.append(H1)
        report.append("RETRIEVAL RESULTS REPORT")
        report.append(H1)
        report.append(f"Timestamp : {meta['timestamp']}")
        report.append(f"Claim     : {meta['claim']}")
        report.append("")
        report.append("Weights:")
        for _, label, wkey, _, sym in STRATEGIES:
            report.append(f"  {sym} {label:<22} {weights[wkey]}")
        report.append("")
        n_chunks = meta['total_results']
        n_docs = len(doc_res)
        report.append(f"Chunks returned (top_k): {n_chunks}   Documents scored: {n_docs}")
        report.append("")

        # ── 2. QUERY ANALYSIS ─────────────────────────────────────────────────
        report.append(H1)
        report.append("QUERY ANALYSIS")
        report.append(H1)

        # Detected themes
        themes = meta.get("detected_themes", [])
        if themes:
            report.append("DETECTED THEMES (S3)")
            report.append(H2)
            for i, t in enumerate(themes, 1):
                kw = ", ".join(t.get("keywords", [])) or "—"
                report.append(f"  #{i:2}  [p={t['probability']:.4f}]  {t['label']}  ({t['theme_node']})")
                report.append(f"        kw: {kw}")
            report.append("")

        # Extracted entities
        ent_meta = meta.get("extracted_entities")
        if ent_meta:
            report.append("EXTRACTED ENTITIES (S4)")
            report.append(H2)
            extracted = ent_meta.get("extracted", [])
            found = set(ent_meta.get("found_in_graph", []))
            not_found_set = set(ent_meta.get("not_in_graph", []))
            if extracted:
                for ent in extracted:
                    node_id = f"entity::{ent['type']}::{ent['text']}"
                    status = "[in graph]" if node_id in found else "[missing] "
                    report.append(f"  {status}  {ent['type']:<8}  {ent['text']}")
            else:
                report.append("  (no entities extracted from query)")
            report.append(f"  → {len(found)} found, {len(not_found_set)} missing")
            report.append("")

        # Matched claims (S5)
        s5_claims = meta.get("matched_claims_s5", [])
        if s5_claims:
            direct   = [c for c in s5_claims if c.get("source") == "direct"]
            expanded = [c for c in s5_claims if c.get("source") != "direct"]
            report.append("MATCHED CLAIMS (S5)")
            report.append(H2)
            report.append("  Direct semantic similarity:")
            if direct:
                for c in direct[:5]:
                    txt = c["claim_text"][:110] + "..." if len(c["claim_text"]) > 110 else c["claim_text"]
                    report.append(f"    sim={c['similarity']:.4f}  [{c.get('rating','N/A')}]  {txt}")
            else:
                report.append("    (none)")
            report.append("  Embedding-expanded neighbors:")
            if expanded:
                for c in expanded[:5]:
                    txt = c["claim_text"][:110] + "..." if len(c["claim_text"]) > 110 else c["claim_text"]
                    report.append(f"    sim={c['similarity']:.4f}  [{c.get('rating','N/A')}]  {txt}")
            else:
                report.append("    (none)")
            report.append("")

        # ── 3. STATISTICS ─────────────────────────────────────────────────────
        if "statistics" in formatted:
            stats = formatted["statistics"]
            report.append(H1)
            report.append(f"STATISTICS  (chunk-level, n={stats['total_results']})")
            report.append(H1)
            report.append(f"  {'Score':<16}  {'mean':>7}  {'max':>7}  {'min':>7}  {'median':>7}")
            report.append(f"  {H2[:60]}")
            for stype, v in stats["score_statistics"].items():
                report.append(
                    f"  {stype:<16}  {v['mean']:7.4f}  {v['max']:7.4f}  {v['min']:7.4f}  {v['median']:7.4f}"
                )
            dom = stats.get("component_analysis", {})
            if dom:
                n = stats["total_results"]
                dom_str = "  ".join(f"{k}={cnt} ({cnt/n*100:.0f}%)" for k, cnt in dom.items())
                report.append(f"\n  Dominant: {dom_str}")
            report.append("")

        # ── 4. PER-STRATEGY RANKINGS ──────────────────────────────────────────
        report.append(H1)
        report.append("PER-STRATEGY RANKINGS  (top 10 chunks / top 10 docs)")
        report.append(H1)

        for strat_key, label, wkey, chunk_score_key, sym in STRATEGIES:
            w = weights[wkey]
            rule_fill = "─" * max(1, 60 - len(label))
            report.append(f"\n── {label}  [{sym}={w}]  {rule_fill}")

            # --- Chunks ranked by this strategy's score ---
            sorted_chunks = sorted(
                all_chunks,
                key=lambda r: r['scores'].get(chunk_score_key, 0),
                reverse=True
            )
            non_zero_chunks = [r for r in sorted_chunks if r['scores'].get(chunk_score_key, 0) > 0]
            col = SCORE_COL[chunk_score_key]
            report.append(
                f"  Chunks by {col} score  ({len(non_zero_chunks)} non-zero / {len(all_chunks)} total):"
            )
            if non_zero_chunks:
                for rank, r in enumerate(non_zero_chunks[:10], 1):
                    sc = r['scores'].get(chunk_score_key, 0)
                    report.append(
                        f"    #{rank:<2}  {r['chunk_id']:<24}  doc={r['document_id']:<5}"
                        f"  {col}={sc:.4f}   total={r['scores']['total']:.4f}"
                    )
            else:
                report.append("    (no chunks scored by this strategy in top_k pool)")

            # --- Docs ranked by this strategy's pooled score ---
            strat_docs = per_strat_docs.get(strat_key, [])
            non_zero_docs = [d for d in strat_docs if d['score'] > 0]
            report.append(
                f"  Documents by {col} score  ({len(non_zero_docs)} non-zero / {len(strat_docs)} total):"
            )
            if non_zero_docs:
                for d in non_zero_docs[:10]:
                    report.append(
                        f"    #{d['rank']:<2}  doc={d['doc_id']:<5}  n={d['n_chunks_scored']:<3}"
                        f"  {col}={d['score']:.4f}"
                    )
            else:
                report.append("    (no documents scored by this strategy)")

        report.append("")

        # ── 5. FINAL HYBRID AGGREGATION ───────────────────────────────────────
        report.append(H1)
        report.append("FINAL HYBRID AGGREGATION")
        report.append(H1)

        # Chunk ranking by total score
        sorted_final_chunks = sorted(
            all_chunks, key=lambda r: r['scores']['total'], reverse=True
        )
        report.append(f"\nChunks  (top {len(sorted_final_chunks)}, ranked by total score):")
        report.append(
            f"  {'#':<3}  {'chunk_id':<24}  {'doc':<5}  {'total':>7}"
            f"  {'S1':>7}  {'S2':>7}  {'S3':>7}  {'S4':>7}  {'S5':>7}  {'S6':>7}  {'S7':>7}"
        )
        report.append(f"  {H2}")
        for rank, r in enumerate(sorted_final_chunks, 1):
            s = r['scores']
            report.append(
                f"  #{rank:<2}  {r['chunk_id']:<24}  {r['document_id']:<5}  {s['total']:7.4f}"
                f"  {s['semantic']:7.4f}  {s['topic_dist']:7.4f}  {s['thematic']:7.4f}"
                f"  {s['entity']:7.4f}  {s.get('claim_doc',0):7.4f}  {s.get('claim_topic',0):7.4f}"
                f"  {s.get('claim_keyphrase',0):7.4f}"
            )
            # Show a short text preview for the top 10
            if rank <= 10:
                preview = r.get('text_preview', '')
                if preview:
                    report.append(f"       > {preview[:120]}{'...' if len(preview) > 120 else ''}")

        # Document ranking by total doc score
        if doc_res:
            report.append(f"\nDocuments  (top {len(doc_res)}, ranked by doc total score):")
            report.append(
                f"  {'#':<3}  {'doc':<5}  {'n':>3}  {'total':>7}"
                f"  {'S1':>7}  {'S2':>7}  {'S3':>7}  {'S4':>7}  {'S5':>7}  {'S6':>7}  {'S7':>7}"
            )
            report.append(f"  {H2}")
            for d in doc_res:
                s = d['scores']
                report.append(
                    f"  #{d['rank']:<2}  {d['doc_id']:<5}  {d['n_chunks_scored']:3}  {s['total']:7.4f}"
                    f"  {s['semantic']:7.4f}  {s['topic_dist']:7.4f}  {s['thematic']:7.4f}"
                    f"  {s['entity']:7.4f}  {s['claim_doc']:7.4f}  {s['claim_topic']:7.4f}"
                    f"  {s['claim_keyphrase']:7.4f}"
                )
                preview = d.get('best_chunk_text_preview', '')
                if preview and d['rank'] <= 10:
                    report.append(f"       > {preview[:120]}{'...' if len(preview) > 120 else ''}")

        report.append("")

        # ── 6. RANDOM SAMPLES ─────────────────────────────────────────────────
        if "random_samples" in formatted:
            report.append(H1)
            report.append("RANDOM SAMPLES  (out-of-top-k, for comparison)")
            report.append(H1)
            report.append(
                f"  {'#':<3}  {'chunk_id':<24}  {'doc':<5}  {'total':>7}"
                f"  {'S1':>7}  {'S2':>7}  {'S3':>7}  {'S4':>7}  {'S5':>7}  {'S6':>7}  {'S7':>7}"
            )
            report.append(f"  {H2}")
            for i, sample in enumerate(formatted["random_samples"], 1):
                s = sample["scores"]
                report.append(
                    f"  #{i:<2}  {sample['chunk_id']:<24}  {sample['document_id']:<5}  {s['total']:7.4f}"
                    f"  {s['semantic']:7.4f}  {s['topic_dist']:7.4f}  {s['thematic']:7.4f}"
                    f"  {s['entity']:7.4f}  {s.get('claim_doc',0):7.4f}  {s.get('claim_topic',0):7.4f}"
                    f"  {s.get('claim_keyphrase',0):7.4f}"
                )
                preview = sample.get('text_preview', '')
                if preview:
                    report.append(f"       > {preview[:120]}{'...' if len(preview) > 120 else ''}")
            report.append("")

        # ── 7. ALL RETRIEVED CHUNKS (FULL TEXT) ───────────────────────────────
        report.append(H1)
        report.append("ALL RETRIEVED CHUNKS  (full text)")
        report.append(H1)
        for i, r in enumerate(sorted_final_chunks, 1):
            s = r['scores']
            report.append(
                f"\n#{i:<2}  {r['chunk_id']}   doc={r['document_id']}   total={s['total']:.4f}"
            )
            report.append(
                f"    S1={s['semantic']:.4f}  S2={s['topic_dist']:.4f}  S3={s['thematic']:.4f}"
                f"  S4={s['entity']:.4f}  S5={s.get('claim_doc',0):.4f}"
                f"  S6={s.get('claim_topic',0):.4f}  S7={s.get('claim_keyphrase',0):.4f}"
            )
            report.append(H2)
            full_text = r.get('full_text', r['text_preview'])
            report.append(full_text)

        report.append("\n" + H1)

        return "\n".join(report)

    def display_results_console(
        self,
        claim: str,
        results: List[Dict],
        top_n: int = 5
    ):
        """
        Display results in console with detailed breakdown.

        Args:
            claim: The search claim
            results: List of result dictionaries
            top_n: Number of top results to display
        """
        print("\n" + "=" * 80)
        print(f"RETRIEVAL RESULTS FOR: {claim}")
        print("=" * 80)

        if not results:
            print("No results found.")
            return

        print(f"\nShowing top {min(top_n, len(results))} results:\n")

        for i, result in enumerate(results[:top_n], 1):
            print(f"\n--- Result #{i} ---")
            print(f"Document ID: {result['doc_id']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"\nScores:")
            print(f"  Total:           {result['score_total']:.4f}")
            print(f"  Semantic:        {result['score_sem']:.4f}")
            print(f"  Thematic:        {result['score_theme']:.4f}")
            print(f"  Entity:          {result['score_entity']:.4f}")
            print(f"  Topic Dist:      {result['score_topic_dist']:.4f}")
            print(f"  Claim Doc:       {result.get('score_claim_doc', 0):.4f}")
            print(f"  Claim Topic:     {result.get('score_claim_topic', 0):.4f}")
            print(f"  Claim KeyPhrase: {result.get('score_claim_keyphrase', 0):.4f}")
            print(f"\nText preview:")
            text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            print(f"  {text_preview}")

            # Show which component dominated
            scores = {
                'semantic': result['score_sem'],
                'thematic': result['score_theme'],
                'entity': result['score_entity'],
                'topic_dist': result['score_topic_dist'],
                'claim_doc': result.get('score_claim_doc', 0),
                'claim_topic': result.get('score_claim_topic', 0),
                'claim_keyphrase': result.get('score_claim_keyphrase', 0)
            }
            dominant = max(scores, key=scores.get)
            print(f"\nDominant component: {dominant} ({scores[dominant]:.4f})")

        print("\n" + "=" * 80)
