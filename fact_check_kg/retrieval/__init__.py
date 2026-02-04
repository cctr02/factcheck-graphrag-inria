"""
Retrieval module.

Contains components for fact-checking retrieval:
- retrieval: Hybrid retrieval system combining semantic, thematic, and entity strategies
- retrieval_results: Results formatting and management
"""

from fact_check_kg.retrieval.retrieval import HybridRetriever
from fact_check_kg.retrieval.retrieval_results import RetrievalResultsManager

__all__ = [
    "HybridRetriever",
    "RetrievalResultsManager",
]
