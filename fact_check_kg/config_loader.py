"""
Configuration loader for Knowledge Graph system.

This module loads and validates configuration from YAML file and provides
easy access to configuration parameters throughout the application.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


@dataclass
class PathsConfig:
    """Paths configuration."""
    data_dir: str
    csv_file: str
    outputs_dir: str
    graphs_dir: str
    visualizations_dir: str
    models_dir: str
    exports_dir: str
    logs_dir: str


@dataclass
class GraphConstructionConfig:
    """Graph construction parameters."""
    max_documents: int
    chunk_size: int
    chunk_overlap: int
    min_topic_size: int
    nr_topics: Optional[int]
    co_occurrence_threshold: int
    semantic_similarity_threshold: float
    entity_types: List[str]


@dataclass
class ModelsConfig:
    """Model configuration."""
    embedding_model: str
    ner_model: str
    llm_model: str
    llm_endpoint: str
    llm_temperature: float
    llm_max_tokens: int


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    database: str
    user: str
    password: str
    graph_table: str
    nodes_table: str
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    alpha: float
    beta: float
    gamma: float
    top_k: int
    theme_threshold: float
    direct_path_weight: float
    co_occurrence_path_weight: float
    theme_path_weight: float

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 0.01:
            raise ConfigError(f"Retrieval weights must sum to 1.0, got {total}")


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    default_width: str
    default_height: str
    background_color: str
    font_color: str
    min_node_size: int
    max_node_size: int
    max_nodes_full_view: int
    max_nodes_entity_view: int
    colors: Dict[str, str]
    edge_colors: Dict[str, str]
    physics: Dict[str, Any]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    log_to_file: bool
    log_to_console: bool
    max_log_size_mb: int
    backup_count: int
    module_levels: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """Export configuration."""
    graph_pickle_pattern: str
    topic_model_pattern: str
    visualization_pattern: str
    timestamp_format: str
    supported_formats: List[str]


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    batch_size: int
    n_jobs: int
    max_memory_mb: int
    enable_cache: bool
    cache_dir: str


@dataclass
class TestingConfig:
    """Testing configuration."""
    test_claims: List[str]
    test_max_docs: int
    test_chunk_size: int


class Config:
    """
    Main configuration class.

    Loads configuration from YAML file and provides access to all settings.

    Example:
        >>> config = Config()
        >>> print(config.graph_construction.chunk_size)
        500
        >>> print(config.paths.logs_dir)
        'logs'
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml file (default: ./config.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        # Load YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)

        # Parse configuration sections
        self._parse_config()

    def _parse_config(self):
        """Parse configuration sections into dataclass instances."""
        try:
            self.paths = PathsConfig(**self._raw_config['paths'])
            self.graph_construction = GraphConstructionConfig(**self._raw_config['graph_construction'])
            self.models = ModelsConfig(**self._raw_config['models'])
            self.database = DatabaseConfig(**self._raw_config['database'])
            self.retrieval = RetrievalConfig(**self._raw_config['retrieval'])
            self.visualization = VisualizationConfig(**self._raw_config['visualization'])
            self.logging = LoggingConfig(**self._raw_config['logging'])
            self.export = ExportConfig(**self._raw_config['export'])
            self.performance = PerformanceConfig(**self._raw_config['performance'])
            self.testing = TestingConfig(**self._raw_config['testing'])
        except KeyError as e:
            raise ConfigError(f"Missing required configuration key: {e}")
        except TypeError as e:
            raise ConfigError(f"Invalid configuration format: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., "paths.logs_dir")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = Config()
            >>> config.get("graph_construction.chunk_size")
            500
        """
        parts = key.split('.')
        value = self._raw_config

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default

        return value

    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self._raw_config

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(paths={self.paths}, graph_construction={self.graph_construction})"


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration instance (singleton).

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Config instance

    Example:
        >>> from fact_check_kg.config_loader import get_config
        >>> config = get_config()
        >>> print(config.paths.logs_dir)
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance


def reload_config(config_path: Optional[str] = None):
    """
    Reload configuration from file.

    Args:
        config_path: Path to config file
    """
    global _config_instance
    _config_instance = Config(config_path)
