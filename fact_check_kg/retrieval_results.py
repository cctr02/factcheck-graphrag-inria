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
        config: dict = None,
        config_retrieval: dict = None
    ) -> Dict:
        """
        Format results with detailed scoring breakdown.

        Args:
            claim: The search claim
            results: List of result dictionaries
            alpha: Semantic weight
            beta: Thematic weight
            gamma: Entity weight
            delta: Topic distribution weight
            config: Optional common configuration dict
            config_retrieval: Optional retrieval-specific configuration dict

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
                    "beta_thematic": beta,
                    "gamma_entity": gamma,
                    "delta_topic_dist": delta
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
                    "topic_dist": round(result['score_topic_dist'], 4)
                },
                "score_contributions": {
                    "semantic_contribution": round(result['score_sem'] * alpha, 4),
                    "thematic_contribution": round(result['score_theme'] * beta, 4),
                    "entity_contribution": round(result['score_entity'] * gamma, 4),
                    "topic_dist_contribution": round(result['score_topic_dist'] * delta, 4)
                },
                "score_explanation": self._generate_score_explanation(result, alpha, beta, gamma, delta)
            }
            formatted["results"].append(formatted_result)

        return formatted

    def _generate_score_explanation(
        self,
        result: Dict,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float
    ) -> str:
        """
        Generate human-readable explanation of score calculation.

        Args:
            result: Result dictionary
            alpha: Semantic weight
            beta: Thematic weight
            gamma: Entity weight
            delta: Topic distribution weight

        Returns:
            Score explanation string
        """
        sem_contrib = result['score_sem'] * alpha
        theme_contrib = result['score_theme'] * beta
        entity_contrib = result['score_entity'] * gamma
        topic_dist_contrib = result['score_topic_dist'] * delta
        total = result['score_total']

        explanation = f"Total score: {total:.4f} = "
        explanation += f"({result['score_sem']:.4f} × {alpha}) + "
        explanation += f"({result['score_theme']:.4f} × {beta}) + "
        explanation += f"({result['score_entity']:.4f} × {gamma}) + "
        explanation += f"({result['score_topic_dist']:.4f} × {delta})"

        # Identify dominant component
        components = [
            ("semantic", sem_contrib),
            ("thematic", theme_contrib),
            ("entity", entity_contrib),
            ("topic_dist", topic_dist_contrib)
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
            key=lambda x: max(x['score_sem'], x['score_theme'], x['score_entity'], x['score_topic_dist']),
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
        scores = [result['score_sem'], result['score_theme'], result['score_entity'], result['score_topic_dist']]
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

        for name, scores in [
            ("total", total_scores),
            ("semantic", sem_scores),
            ("thematic", theme_scores),
            ("entity", entity_scores),
            ("topic_dist", topic_dist_scores)
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
                "topic_dist": result['score_topic_dist']
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
        config_retrieval: dict = None
    ) -> Tuple[str, str]:
        """
        Save results to JSON and text files.

        Args:
            claim: The search claim
            results: List of result dictionaries
            alpha: Semantic weight
            beta: Thematic weight
            gamma: Entity weight
            delta: Topic distribution weight
            rankings: Optional pre-computed rankings
            graph_id: Graph ID to use as base timestamp
            custom_name: Custom name for this retrieval (e.g., "test1")
            config: Optional common configuration dict
            config_retrieval: Optional retrieval-specific configuration dict

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
            config=config, config_retrieval=config_retrieval
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
        report.append(f"  Beta (Thematic): {formatted['metadata']['weights']['beta_thematic']}")
        report.append(f"  Gamma (Entity): {formatted['metadata']['weights']['gamma_entity']}")
        report.append(f"  Delta (Topic Dist): {formatted['metadata']['weights']['delta_topic_dist']}")
        report.append(f"\nTotal Results: {formatted['metadata']['total_results']}")
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
            report.append(f"   {result['score_explanation']}")
            report.append(f"   Text: {result['text_preview']}")

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
            print(f"  Total:      {result['score_total']:.4f}")
            print(f"  Semantic:   {result['score_sem']:.4f}")
            print(f"  Thematic:   {result['score_theme']:.4f}")
            print(f"  Entity:     {result['score_entity']:.4f}")
            print(f"  Topic Dist: {result['score_topic_dist']:.4f}")
            print(f"\nText preview:")
            text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            print(f"  {text_preview}")

            # Show which component dominated
            scores = {
                'semantic': result['score_sem'],
                'thematic': result['score_theme'],
                'entity': result['score_entity'],
                'topic_dist': result['score_topic_dist']
            }
            dominant = max(scores, key=scores.get)
            print(f"\nDominant component: {dominant} ({scores[dominant]:.4f})")

        print("\n" + "=" * 80)
