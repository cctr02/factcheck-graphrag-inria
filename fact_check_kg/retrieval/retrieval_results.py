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
        random_samples: List[Dict] = None
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
        random_samples: List[Dict] = None
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
            random_samples=random_samples
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
        text_report = self._generate_text_report(formatted, rankings)
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
        rankings: Dict[str, List[Dict]]
    ) -> str:
        """
        Generate human-readable text report.

        Args:
            formatted: Formatted results dictionary
            rankings: Rankings dictionary

        Returns:
            Text report string
        """
        report = []

        # Header
        report.append("=" * 80)
        report.append("RETRIEVAL RESULTS REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {formatted['metadata']['timestamp']}")
        report.append(f"Claim: {formatted['metadata']['claim']}")
        report.append(f"\nWeights:")
        report.append(f"  Alpha (Semantic): {formatted['metadata']['weights']['alpha_semantic']}")
        report.append(f"  Beta (Topic Dist): {formatted['metadata']['weights']['beta_topic_dist']}")
        report.append(f"  Gamma (Thematic): {formatted['metadata']['weights']['gamma_thematic']}")
        report.append(f"  Delta (Entity): {formatted['metadata']['weights']['delta_entity']}")
        report.append(f"  Epsilon (Claim Doc): {formatted['metadata']['weights']['epsilon_claim_doc']}")
        report.append(f"  Zeta (Claim Topic): {formatted['metadata']['weights']['zeta_claim_topic']}")
        report.append(f"  Eta (Claim KeyPhrase): {formatted['metadata']['weights']['eta_claim_keyphrase']}")
        report.append(f"\nTotal Results: {formatted['metadata']['total_results']}")
        report.append("")

        # Detected themes (S3 query classification)
        if "detected_themes" in formatted.get("metadata", {}):
            report.append("DETECTED THEMES (Query Classification)")
            report.append("-" * 80)
            for i, theme in enumerate(formatted["metadata"]["detected_themes"], 1):
                keywords_str = ", ".join(theme.get("keywords", [])) if theme.get("keywords") else "—"
                report.append(f"  {i}. [prob={theme['probability']:.4f}] {theme['label']}  ({theme['theme_node']})")
                report.append(f"      Keywords: {keywords_str}")
            report.append("")

        # Extracted entities (S4)
        ent_meta = formatted.get("metadata", {}).get("extracted_entities")
        if ent_meta:
            report.append("EXTRACTED ENTITIES (S4 — Entity Graph Traversal)")
            report.append("-" * 80)
            extracted = ent_meta.get("extracted", [])
            found = set(ent_meta.get("found_in_graph", []))
            not_found = set(ent_meta.get("not_in_graph", []))
            if extracted:
                for ent in extracted:
                    node_id = f"entity::{ent['type']}::{ent['text']}"
                    status = "[in graph]" if node_id in found else "[missing] "
                    report.append(f"  {status} {ent['type']}: {ent['text']}")
            else:
                report.append("  (no entities extracted from query)")
            report.append(f"  → {len(found)} found in graph, {len(not_found)} missing")
            report.append("")

        # Matched claims — show S5 direct/graph (top 3 each), skip S6/S7 (same claim matching)
        s5_claims = formatted.get("metadata", {}).get("matched_claims_s5", [])
        if s5_claims:
            direct = [c for c in s5_claims if c.get("source") == "direct"]
            graph  = [c for c in s5_claims if c.get("source") != "direct"]
            report.append("MATCHED CLAIMS (S5 — Direct semantic similarity)")
            report.append("-" * 80)
            if direct:
                for c in direct[:3]:
                    preview = c["claim_text"][:120] + "..." if len(c["claim_text"]) > 120 else c["claim_text"]
                    report.append(f"  [direct] [sim={c['similarity']:.4f}] [rating={c.get('rating','N/A')}] {preview}")
            else:
                report.append("  (none)")
            report.append("")
            report.append("MATCHED CLAIMS (S5 — Embedding-expanded neighbors)")
            report.append("-" * 80)
            if graph:
                for c in graph[:3]:
                    preview = c["claim_text"][:120] + "..." if len(c["claim_text"]) > 120 else c["claim_text"]
                    report.append(f"  [embed]  [sim={c['similarity']:.4f}] [rating={c.get('rating','N/A')}] {preview}")
            else:
                report.append("  (none — all below expansion_threshold)")
            report.append("")

        # Statistics
        if "statistics" in formatted:
            stats = formatted["statistics"]
            report.append("STATISTICS")
            report.append("-" * 80)
            report.append(f"\nScore Statistics:")
            for score_type, values in stats["score_statistics"].items():
                report.append(f"\n  {score_type.capitalize()}:")
                report.append(f"    Mean: {values['mean']:.4f}")
                report.append(f"    Min: {values['min']:.4f}")
                report.append(f"    Max: {values['max']:.4f}")
                report.append(f"    Median: {values['median']:.4f}")

            report.append(f"\n\nComponent Dominance:")
            for component, count in stats["component_analysis"].items():
                percentage = (count / stats["total_results"]) * 100
                report.append(f"  {component.capitalize()}: {count} ({percentage:.1f}%)")
            report.append("")

        # Top 10 results by total score
        report.append("\nTOP 10 RESULTS (By Total Score)")
        report.append("-" * 80)
        for i, result in enumerate(formatted["results"][:10], 1):
            report.append(f"\n{i}. Document: {result['document_id']} | Chunk: {result['chunk_id']}")
            report.append(f"   Total Score: {result['scores']['total']:.4f}")
            report.append(f"   Component Scores:")
            report.append(f"     Semantic: {result['scores']['semantic']:.4f} (contrib: {result['score_contributions']['semantic_contribution']:.4f})")
            report.append(f"     Thematic: {result['scores']['thematic']:.4f} (contrib: {result['score_contributions']['thematic_contribution']:.4f})")
            report.append(f"     Entity: {result['scores']['entity']:.4f} (contrib: {result['score_contributions']['entity_contribution']:.4f})")
            report.append(f"     Topic Dist: {result['scores']['topic_dist']:.4f} (contrib: {result['score_contributions']['topic_dist_contribution']:.4f})")
            report.append(f"     Claim Doc:  {result['scores'].get('claim_doc', 0):.4f} (contrib: {result['score_contributions'].get('claim_doc_contribution', 0):.4f})")
            report.append(f"     Claim Topic: {result['scores'].get('claim_topic', 0):.4f} (contrib: {result['score_contributions'].get('claim_topic_contribution', 0):.4f})")
            report.append(f"     Claim KP:   {result['scores'].get('claim_keyphrase', 0):.4f} (contrib: {result['score_contributions'].get('claim_keyphrase_contribution', 0):.4f})")
            report.append(f"   Source: {result.get('source_query', 'original_query')}")
            report.append(f"   {result['score_explanation']}")
            report.append(f"   Text: {result['text_preview']}")

        # Random sample chunks for comparison
        if "random_samples" in formatted:
            report.append("\n\nRANDOM SAMPLE CHUNKS (for comparison)")
            report.append("-" * 80)
            for i, sample in enumerate(formatted["random_samples"], 1):
                s = sample["scores"]
                report.append(f"\n{i}. Document: {sample['document_id']} | Chunk: {sample['chunk_id']}")
                report.append(f"   Total Score: {s['total']:.4f}")
                report.append(f"   Scores: Sem={s['semantic']:.4f} | Theme={s['thematic']:.4f} | Entity={s['entity']:.4f} | TopicDist={s['topic_dist']:.4f} | ClaimDoc={s['claim_doc']:.4f} | ClaimTopic={s.get('claim_topic', 0):.4f} | ClaimKP={s.get('claim_keyphrase', 0):.4f}")
                report.append(f"   Text: {sample['text_preview']}")

        # Alternative rankings
        report.append("\n\nALTERNATIVE RANKINGS")
        report.append("-" * 80)

        for ranking_name, ranking_results in rankings.items():
            if ranking_name == "by_total_score":
                continue  # Already shown above

            report.append(f"\n{ranking_name.replace('_', ' ').title()} (Top 5):")
            for i, result in enumerate(ranking_results[:5], 1):
                report.append(f"  {i}. {result['chunk_id']} - Total: {result['score_total']:.4f}")

        # All chunks with full text
        report.append("\n\nALL RETRIEVED CHUNKS (Full Text)")
        report.append("=" * 80)
        for i, result in enumerate(formatted["results"], 1):
            report.append(f"\n--- Chunk #{i}: {result['chunk_id']} ---")
            report.append(f"Document ID: {result['document_id']}")
            report.append(f"Total Score: {result['scores']['total']:.4f}")
            report.append(f"\nFull Text:")
            # Get full text from formatted results (we need to add it)
            full_text = result.get('full_text', result['text_preview'])
            report.append(full_text)
            report.append("-" * 80)

        report.append("\n" + "=" * 80)

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
