"""
Core shared modules.

Contains shared utilities used across the package:
- config_loader: Configuration loading and management
- db_connector: PostgreSQL database connector
- embedding_cache: FAISS-based embedding cache
- logger: Centralized logging system
"""

from fact_check_kg.core.config_loader import (
    Config,
    ConfigError,
    get_config,
    get_build_config,
    get_retrieval_config,
    reload_config,
    reload_build_config,
    reload_retrieval_config,
    load_raw_config,
)
from fact_check_kg.core.db_connector import GraphDatabaseConnector
from fact_check_kg.core.embedding_cache import EmbeddingCache
from fact_check_kg.core.logger import get_logger, log_section, log_stats, LoggerManager

__all__ = [
    # Config
    "Config",
    "ConfigError",
    "get_config",
    "get_build_config",
    "get_retrieval_config",
    "reload_config",
    "reload_build_config",
    "reload_retrieval_config",
    "load_raw_config",
    # Database
    "GraphDatabaseConnector",
    # Embedding
    "EmbeddingCache",
    # Logger
    "get_logger",
    "log_section",
    "log_stats",
    "LoggerManager",
]
