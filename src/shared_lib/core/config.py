"""
Global configuration management and parameter validation.

This module provides centralized configuration management for all agents,
with optimized LLM configurations including timeouts, retries, and agent-specific settings.

MIGRATION TO GEMINI:
- Uses Google Gemini (gemini-2.5-flash-lite) as primary LLM provider
- Supports temperature, top_p, top_k parameters (unlike gpt-5-nano)
- Uses system instructions for prompts (Gemini best practice)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# OPTIMIZED LLM CONFIGURATIONS - GEMINI MIGRATION
# ============================================================================


@dataclass
class LLMConfig:
    """
    Base LLM configuration with optimized defaults for Google Gemini.

    Key optimizations:
    - timeout=30s (reduced from default 60s)
    - max_retries=2 (fail fast instead of many retries)
    - max_output_tokens=1500 (Gemini parameter name)

    Gemini Parameters:
    - temperature: Controls randomness (0.0-2.0, default 0.7)
    - top_p: Nucleus sampling (0.0-1.0)
    - top_k: Top-k sampling (1-40)

    References:
    - Authentication: https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb
    - System Instructions: https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/System_instructions.ipynb
    """

    # Model configuration (Gemini)
    model: str = "gemini-2.5-flash-lite"
    api_key: Optional[str] = None

    # Performance optimizations
    timeout: int = 30  # 30s timeout
    max_retries: int = 2  # Fail fast with only 2 retries

    # Generation parameters (Gemini naming convention)
    max_output_tokens: int = 1500  # Gemini uses max_output_tokens instead of max_tokens

    # Gemini generation parameters
    temperature: float = 0.7  # Default balanced temperature
    top_p: Optional[float] = None  # Nucleus sampling (optional)
    top_k: Optional[int] = None  # Top-k sampling (optional)

    # Response format (for JSON mode)
    response_mime_type: Optional[str] = None  # "application/json" for JSON mode

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found in environment. "
                    "Please set it in your .env file or pass it explicitly."
                )

    def to_gemini_kwargs(self) -> Dict[str, Any]:
        """
        Convert configuration to ChatGoogleGenerativeAI constructor kwargs.

        Follows Gemini API best practices:
        - Uses google_api_key parameter (not api_key)
        - Uses max_output_tokens (not max_tokens)
        - Supports temperature, top_p, top_k

        Returns:
            Dictionary ready for ChatGoogleGenerativeAI(**kwargs)
        """
        kwargs = {
            "model": self.model,
            "google_api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }

        # Add optional parameters
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.response_mime_type:
            kwargs["response_mime_type"] = self.response_mime_type

        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary (legacy compatibility).

        DEPRECATED: Use to_gemini_kwargs() instead.
        """
        return {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
        }


@dataclass
class FormatterLLMConfig(LLMConfig):
    """
    Configuration for formatter_agent.

    Optimized for narrative generation with larger token limit.
    Always uses JSON mode for structured output.
    """

    temperature: float = 0.3  # More deterministic for consistent formatting
    max_output_tokens: int = 2500  # Larger limit for executive summaries

    def __post_init__(self):
        """Set JSON mode for formatter (Gemini style)."""
        super().__post_init__()
        self.response_mime_type = "application/json"


@dataclass
class FilterLLMConfig(LLMConfig):
    """
    Configuration for filter_classifier.

    Optimized for fast filter parsing and semantic understanding.
    """

    temperature: float = 0.3  # Low temperature for consistent parsing


@dataclass
class GraphicLLMConfig(LLMConfig):
    """
    Configuration for graphic_classifier.

    Balanced temperature for consistent chart type classification.
    """

    temperature: float = 0.5  # Medium temperature for classification


@dataclass
class InsightLLMConfig(LLMConfig):
    """
    Configuration for insight_generator.

    FASE 3: Upgraded to gemini-2.5-flash (from flash-lite) for higher
    narrative quality. Temperature reduced to 0.4 for balanced output.
    Always uses JSON mode for structured insights.

    Note: Gemini 2.5 models with thinking require higher max_output_tokens
    because tokens are spent on reasoning before generating the final output.
    If max_output_tokens is too low, the model may return empty content
    with finish_reason='MAX_TOKENS'.
    """

    model: str = "gemini-2.5-flash"  # FASE 3: Upgraded from flash-lite
    max_output_tokens: int = 8192  # Higher limit for Gemini 2.5 thinking models
    temperature: float = 0.4  # FASE 3: Balanced between creativity and consistency

    def __post_init__(self):
        """Set JSON mode for insights (Gemini style)."""
        super().__post_init__()
        self.response_mime_type = "application/json"


# ============================================================================
# CONFIGURATION FACTORIES
# ============================================================================


def get_formatter_config(**overrides) -> FormatterLLMConfig:
    """
    Get formatter agent LLM configuration (Gemini).

    Supports environment variable overrides:
    - GEMINI_MODEL: Model name (default: gemini-2.5-flash-lite)
    - MAX_OUTPUT_TOKENS: Maximum output tokens
    - TEMPERATURE: Temperature for generation

    Args:
        **overrides: Explicit overrides for config values

    Returns:
        FormatterLLMConfig instance with Gemini optimizations

    Example:
        >>> config = get_formatter_config(max_output_tokens=3000)
        >>> llm = ChatGoogleGenerativeAI(**config.to_gemini_kwargs())
    """
    # Load env var overrides
    env_overrides = {}
    if os.getenv("MAX_OUTPUT_TOKENS"):
        env_overrides["max_output_tokens"] = int(os.getenv("MAX_OUTPUT_TOKENS"))
    if os.getenv("GEMINI_MODEL"):
        env_overrides["model"] = os.getenv("GEMINI_MODEL")
    if os.getenv("TEMPERATURE"):
        env_overrides["temperature"] = float(os.getenv("TEMPERATURE"))

    # Merge: defaults < env vars < explicit overrides
    config_dict = {**env_overrides, **overrides}
    return FormatterLLMConfig(**config_dict)


def get_filter_config(**overrides) -> FilterLLMConfig:
    """
    Get filter classifier LLM configuration (Gemini).

    Supports same environment variable overrides as get_formatter_config().

    Args:
        **overrides: Explicit overrides for config values

    Returns:
        FilterLLMConfig instance with Gemini optimizations
    """
    env_overrides = {}
    if os.getenv("MAX_OUTPUT_TOKENS"):
        env_overrides["max_output_tokens"] = int(os.getenv("MAX_OUTPUT_TOKENS"))
    if os.getenv("GEMINI_MODEL"):
        env_overrides["model"] = os.getenv("GEMINI_MODEL")
    if os.getenv("TEMPERATURE"):
        env_overrides["temperature"] = float(os.getenv("TEMPERATURE"))

    config_dict = {**env_overrides, **overrides}
    return FilterLLMConfig(**config_dict)


def get_graphic_config(**overrides) -> GraphicLLMConfig:
    """
    Get graphic classifier LLM configuration (Gemini).

    Supports same environment variable overrides as get_formatter_config().

    Args:
        **overrides: Explicit overrides for config values

    Returns:
        GraphicLLMConfig instance with Gemini optimizations
    """
    env_overrides = {}
    if os.getenv("MAX_OUTPUT_TOKENS"):
        env_overrides["max_output_tokens"] = int(os.getenv("MAX_OUTPUT_TOKENS"))
    if os.getenv("GEMINI_MODEL"):
        env_overrides["model"] = os.getenv("GEMINI_MODEL")
    if os.getenv("TEMPERATURE"):
        env_overrides["temperature"] = float(os.getenv("TEMPERATURE"))

    config_dict = {**env_overrides, **overrides}
    return GraphicLLMConfig(**config_dict)


def get_insight_config(**overrides) -> InsightLLMConfig:
    """
    Get insight generator LLM configuration (Gemini).

    Supports same environment variable overrides as get_formatter_config().

    Args:
        **overrides: Explicit overrides for config values

    Returns:
        InsightLLMConfig instance with Gemini optimizations
    """
    env_overrides = {}
    if os.getenv("MAX_OUTPUT_TOKENS"):
        env_overrides["max_output_tokens"] = int(os.getenv("MAX_OUTPUT_TOKENS"))
    if os.getenv("GEMINI_MODEL"):
        env_overrides["model"] = os.getenv("GEMINI_MODEL")
    if os.getenv("TEMPERATURE"):
        env_overrides["temperature"] = float(os.getenv("TEMPERATURE"))

    config_dict = {**env_overrides, **overrides}
    return InsightLLMConfig(**config_dict)


# ============================================================================
# LEGACY SUPPORT (BACKWARD COMPATIBILITY)
# ============================================================================
# Keep these for backward compatibility with existing code

# Import settings from graphic_classifier for legacy support
try:
    from src.graphic_classifier.core.settings import (
        OPENAI_API_KEY as _LEGACY_API_KEY,
        OPENAI_MODEL as _LEGACY_MODEL,
        ALIAS_PATH,
        DATASET_PATH,
        LOG_LEVEL,
        LOG_FILE,
        FUZZY_MATCH_THRESHOLD,
        SEMANTIC_MATCH_THRESHOLD,
        VALID_CHART_TYPES,
        VALID_AGGREGATIONS,
    )

    # Use legacy values for backward compatibility
    OPENAI_API_KEY = _LEGACY_API_KEY
    OPENAI_MODEL = _LEGACY_MODEL
except ImportError:
    # Fallback to env vars if graphic_classifier not available
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv(
        "OPENAI_MODEL", "gemini-2.5-flash-lite"
    )  # Legacy compatibility - now using Gemini
    ALIAS_PATH = "data/alias"
    DATASET_PATH = "data/datasets"
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/agent.log"
    FUZZY_MATCH_THRESHOLD = 0.75
    SEMANTIC_MATCH_THRESHOLD = 0.75
    VALID_CHART_TYPES = []
    VALID_AGGREGATIONS = []


@dataclass
class DataConfig:
    """Configuration for data sources (legacy compatibility)."""

    alias_path: str = ALIAS_PATH
    dataset_path: str = DATASET_PATH

    def validate(self) -> bool:
        """Validate that data files exist."""
        if not Path(self.alias_path).exists():
            raise FileNotFoundError(f"Alias file not found: {self.alias_path}")
        return True


@dataclass
class AliasMapperConfig:
    """Configuration for alias mapping (legacy compatibility)."""

    fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD
    semantic_threshold: float = SEMANTIC_MATCH_THRESHOLD
    use_semantic_matching: bool = False
    cache_enabled: bool = True

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not 0 <= self.fuzzy_threshold <= 1:
            raise ValueError(
                f"fuzzy_threshold must be between 0 and 1, got: {self.fuzzy_threshold}"
            )
        if not 0 <= self.semantic_threshold <= 1:
            raise ValueError(
                f"semantic_threshold must be between 0 and 1, got: {self.semantic_threshold}"
            )
        return True


@dataclass
class LoggingConfig:
    """Configuration for logging (legacy compatibility)."""

    level: str = LOG_LEVEL
    log_file: str = LOG_FILE
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class AgentConfig:
    """Main agent configuration aggregating all sub-configs (legacy compatibility)."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    alias_mapper: AliasMapperConfig = field(default_factory=AliasMapperConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Chart classification settings
    valid_chart_types: list = field(
        default_factory=lambda: VALID_CHART_TYPES.copy() if VALID_CHART_TYPES else []
    )
    valid_aggregations: list = field(
        default_factory=lambda: VALID_AGGREGATIONS.copy() if VALID_AGGREGATIONS else []
    )

    def validate_all(self) -> bool:
        """Validate all configurations."""
        self.data.validate()
        self.alias_mapper.validate()
        return True

    @classmethod
    def from_defaults(cls) -> "AgentConfig":
        """Create configuration with default values."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            "llm": self.llm.to_dict(),
            "data": {
                "alias_path": self.data.alias_path,
                "dataset_path": self.data.dataset_path,
            },
            "alias_mapper": {
                "fuzzy_threshold": self.alias_mapper.fuzzy_threshold,
                "semantic_threshold": self.alias_mapper.semantic_threshold,
                "use_semantic_matching": self.alias_mapper.use_semantic_matching,
            },
            "logging": {
                "level": self.logging.level,
                "log_file": self.logging.log_file,
            },
        }


# Global configuration instance (legacy compatibility)
config = AgentConfig.from_defaults()
