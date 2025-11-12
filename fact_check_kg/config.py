"""
Configuration file for Knowledge Graph construction and retrieval.

This module contains all configurable parameters for the system.
"""

from dataclasses import dataclass


@dataclass
class GraphConfig:
    """Configuration for graph construction."""

    # Chunking parameters
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    ner_model: str = "en_core_web_sm"

    # Entity types to extract
    entity_types: list = None

    # Topic modeling parameters
    min_topic_size: int = 10
    nr_topics: int = None  # Auto if None

    # Thresholds
    co_occur_threshold: int = 2
    semantic_sim_threshold: float = 0.6
    entity_related_threshold: float = 0.5

    def __post_init__(self):
        """Set default values for mutable defaults."""
        if self.entity_types is None:
            self.entity_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'NORP']


@dataclass
class DatabaseConfig:
    """Configuration for PostgreSQL database connection."""

    # PostgreSQL connection parameters
    host: str = "localhost"
    port: int = 5432
    database: str = "knowledge_graph"
    user: str = "postgres"
    password: str = "postgres"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategy."""

    # Weight parameters for score fusion
    alpha: float = 0.4  # Semantic weight
    beta: float = 0.3   # Thematic weight
    gamma: float = 0.3  # Entity weight

    # Retrieval parameters
    top_k: int = 10
    theme_threshold: float = 0.1  # Min confidence for theme detection

    # Path penalties (already in edge weights, kept for reference)
    direct_path_weight: float = 1.0
    co_occur_path_weight: float = 0.8
    theme_path_weight: float = 0.6

    def __post_init__(self):
        """Validate weights sum to 1."""
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
