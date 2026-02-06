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
    claims_csv_file: str
    outputs_dir: str
    graphs_dir: str
    visualizations_dir: str
    models_dir: str
    embeddings_dir: str
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
    topic_probability_threshold: float
    temporal_max_year_gap: int
    temporal_decay_rate: float
    topic_keywords_initial: int
    topic_keywords_final: int
    keyphrase_top_n: int
    keyphrase_ngram_min: int
    keyphrase_ngram_max: int
    entity_types: List[str]


@dataclass
class ModelsConfig:
    """Model configuration."""
    embedding_model: str
    embedding_dimension: int
    ner_model: str
    # LLM fields are optional (only used during build)
    llm_model: Optional[str] = None
    llm_endpoint: Optional[str] = None


@dataclass
class LLMBatchLabelingConfig:
    """LLM batch labeling configuration for topic labeling."""
    # Prompt template
    prompt_template: str

    # Batch size settings
    batch_size_default: int
    batch_size_min: int
    batch_size_max: int
    batch_size_step: int

    # Label constraints
    label_min_words: int
    label_max_words: int
    label_max_length: int

    # Keywords per topic
    max_keywords_per_topic: int

    # Retry settings
    retry_count: int
    retry_temperature: float

    # Timeout settings
    batch_timeout: int

    # Auto-tuning settings
    success_streak_threshold: int
    hysteresis_decrease_factor: float

    # Failure strategy
    failure_strategy: str

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size_min < 1:
            raise ConfigError("batch_size_min must be >= 1")
        if self.batch_size_max < self.batch_size_min:
            raise ConfigError("batch_size_max must be >= batch_size_min")
        if self.batch_size_default < self.batch_size_min or self.batch_size_default > self.batch_size_max:
            raise ConfigError("batch_size_default must be between batch_size_min and batch_size_max")
        if self.failure_strategy not in ("binary_split", "single_fallback"):
            raise ConfigError(f"Invalid failure_strategy: {self.failure_strategy}")


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
    delta: float
    epsilon: float
    zeta: float = 0.0  # Weight for claim-topic traversal (S6)
    eta: float = 0.0  # Weight for claim-keyphrase similarity (S7)
    entity_types: List[str] = None
    top_k: int = 10
    theme_threshold: float = 0.01
    entity_relevance_threshold: float = 0.1
    theme_relevance_threshold: float = 0.1
    direct_path_weight: float = 1.0
    co_occurrence_path_weight: float = 0.8
    theme_path_weight: float = 0.6
    claim_top_k: int = 5
    keyphrase_top_n: int = 5  # Number of keyphrases for Strategy 7

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.alpha + self.beta + self.gamma + self.delta + self.epsilon + self.zeta + self.eta
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
    timestamp_format: str
    supported_formats: List[str]
    # Optional fields (may differ between build and retrieval)
    graph_pickle_pattern: Optional[str] = None
    topic_model_pattern: Optional[str] = None
    visualization_pattern: Optional[str] = None


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


def _get_project_root() -> Path:
    """Get project root directory (3 levels up from this file: core -> fact_check_kg -> project)."""
    return Path(__file__).parent.parent.parent


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

    def __init__(self, config_path: Optional[str] = None, base_config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to specific config file (e.g., config_build.yaml)
            base_config_path: Path to base config file with common settings (default: ./config.yaml)
        """
        # Load base config first (common settings)
        if base_config_path is None:
            base_config_path = _get_project_root() / "config" / "config.yaml"
        else:
            base_config_path = Path(base_config_path)

        if base_config_path.exists():
            with open(base_config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}
        else:
            self._raw_config = {}

        # Load specific config and merge
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise ConfigError(f"Configuration file not found: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                specific_config = yaml.safe_load(f) or {}

            # Deep merge: specific config overrides base config
            self._raw_config = self._deep_merge(self._raw_config, specific_config)

        # Parse configuration sections
        self._parse_config()

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        Deep merge two dictionaries. Override values take precedence.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _parse_config(self):
        """Parse configuration sections into dataclass instances."""
        try:
            # Required sections (must be present)
            self.paths = PathsConfig(**self._raw_config['paths'])
            self.models = ModelsConfig(**self._raw_config['models'])
            self.database = DatabaseConfig(**self._raw_config['database'])
            self.logging = LoggingConfig(**self._raw_config['logging'])
            self.performance = PerformanceConfig(**self._raw_config['performance'])

            # Optional sections (may not be in all configs)
            if 'graph_construction' in self._raw_config:
                self.graph_construction = GraphConstructionConfig(**self._raw_config['graph_construction'])
            else:
                self.graph_construction = None

            if 'llm_batch_labeling' in self._raw_config:
                self.llm_batch_labeling = LLMBatchLabelingConfig(**self._raw_config['llm_batch_labeling'])
            else:
                self.llm_batch_labeling = None

            if 'retrieval' in self._raw_config:
                self.retrieval = RetrievalConfig(**self._raw_config['retrieval'])
            else:
                self.retrieval = None

            if 'visualization' in self._raw_config:
                self.visualization = VisualizationConfig(**self._raw_config['visualization'])
            else:
                self.visualization = None

            if 'export' in self._raw_config:
                self.export = ExportConfig(**self._raw_config['export'])
            else:
                self.export = None

            if 'active_graph' in self._raw_config:
                # active_graph is a simple dict, not a dataclass
                self.active_graph = self._raw_config['active_graph']
            else:
                self.active_graph = None

            # Testing config is optional
            if 'testing' in self._raw_config:
                self.testing = TestingConfig(**self._raw_config['testing'])
            else:
                self.testing = None
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


# Singleton instances
_config_build_instance: Optional[Config] = None
_config_retrieval_instance: Optional[Config] = None
_config_instance: Optional[Config] = None  # Legacy support


def get_build_config(config_path: Optional[str] = None) -> Config:
    """
    Get build configuration instance (singleton).

    Loads config.yaml (common settings) and merges with config_build.yaml (build-specific settings).

    Args:
        config_path: Path to build config file (default: ./config_build.yaml)

    Returns:
        Config instance for building graphs

    Example:
        >>> from fact_check_kg.core.config_loader import get_build_config
        >>> config = get_build_config()
        >>> print(config.graph_construction.chunk_size)
    """
    global _config_build_instance

    if _config_build_instance is None:
        if config_path is None:
            config_path = _get_project_root() / "config" / "config_build.yaml"
        # Load base config.yaml and merge with config_build.yaml
        _config_build_instance = Config(config_path=config_path)

    return _config_build_instance


def get_retrieval_config(config_path: Optional[str] = None) -> Config:
    """
    Get retrieval configuration instance (singleton).

    Loads config.yaml (common settings) and merges with config_retrieval.yaml (retrieval-specific settings).

    Args:
        config_path: Path to retrieval config file (default: ./config_retrieval.yaml)

    Returns:
        Config instance for retrieval

    Example:
        >>> from fact_check_kg.core.config_loader import get_retrieval_config
        >>> config = get_retrieval_config()
        >>> print(config.retrieval.top_k)
    """
    global _config_retrieval_instance

    if _config_retrieval_instance is None:
        if config_path is None:
            config_path = _get_project_root() / "config" / "config_retrieval.yaml"
        # Load base config.yaml and merge with config_retrieval.yaml
        _config_retrieval_instance = Config(config_path=config_path)

    return _config_retrieval_instance


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration instance (singleton).

    DEPRECATED: Use get_build_config() or get_retrieval_config() instead.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Config instance

    Example:
        >>> from fact_check_kg.core.config_loader import get_config
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


def load_raw_config(config_path: str) -> dict:
    """
    Load raw config from YAML file without merging or parsing.

    Args:
        config_path: Path to config file

    Returns:
        Raw configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def reload_build_config(config_path: Optional[str] = None):
    """
    Reload build configuration from file.

    Args:
        config_path: Path to config file
    """
    global _config_build_instance
    if config_path is None:
        config_path = _get_project_root() / "config" / "config_build.yaml"
    _config_build_instance = Config(config_path)


def reload_retrieval_config(config_path: Optional[str] = None):
    """
    Reload retrieval configuration from file.

    Args:
        config_path: Path to config file
    """
    global _config_retrieval_instance
    if config_path is None:
        config_path = _get_project_root() / "config" / "config_retrieval.yaml"
    _config_retrieval_instance = Config(config_path)
