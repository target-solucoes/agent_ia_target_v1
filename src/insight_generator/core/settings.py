"""
Environment settings and configuration for insight_generator.

This module loads and validates all configuration parameters required
for the Insight Generator agent, including LLM settings, data paths,
and agent-specific configurations.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# ========== LLM Configuration ==========
# NOTE: This project is migrated to Gemini for insight generation.
# Keep legacy OPENAI_* vars for backward compatibility where still used.
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

# CRITICAL: OPENAI_API_KEY variable is used throughout codebase for backward compatibility
# but should PRIORITIZE GEMINI_API_KEY first, then fallback to OPENAI_API_KEY
OPENAI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv(
    "OPENAI_API_KEY"
)
OPENAI_MODEL: str = os.getenv("GEMINI_MODEL") or os.getenv(
    "OPENAI_MODEL", "gemini-2.5-flash"
)

# ========== FASE 3: Model Selection Configuration ==========
# Default model for insight generation (upgraded from flash-lite)
INSIGHT_MODEL_DEFAULT: str = os.getenv("INSIGHT_MODEL_DEFAULT", "gemini-2.5-flash")
# Lite model for simple queries (rankings, single metrics)
INSIGHT_MODEL_LITE: str = os.getenv("INSIGHT_MODEL_LITE", "gemini-2.5-flash-lite")
# Temperature: balanced between creativity and consistency
INSIGHT_TEMPERATURE_DEFAULT: float = float(os.getenv("INSIGHT_TEMPERATURE", "0.4"))

# Legacy settings (kept for backward compatibility)
# REASONING_EFFORT was used for GPT-5 models, not applicable to Gemini but kept for compatibility
REASONING_EFFORT: str = os.getenv("REASONING_EFFORT", "minimal")
MAX_COMPLETION_TOKENS: int = int(os.getenv("MAX_COMPLETION_TOKENS", "2000"))

# Rollout control (FASE 6): prompt mode selection
# - legacy: chart_type templates (current default)
# - dynamic: intent-based DynamicPromptBuilder (FASE 3)
INSIGHT_PROMPT_MODE: str = os.getenv("INSIGHT_PROMPT_MODE", "legacy").lower()

# ========== Data Paths ==========
DATASET_PATH: str = os.getenv(
    "DATASET_PATH",
    str(PROJECT_ROOT / "data" / "datasets" / "DadosComercial_resumido_v02.parquet"),
)

# ========== Insight Generator Configuration ==========
# Maximum number of insights to generate per query
MAX_INSIGHTS: int = int(os.getenv("MAX_INSIGHTS", "5"))

# Transparency validation threshold (minimum percentage of insights with numbers)
TRANSPARENCY_THRESHOLD: float = float(os.getenv("TRANSPARENCY_THRESHOLD", "0.7"))

# Default top N for ranking calculations (alinhado com CategoryLimiter)
DEFAULT_TOP_N: int = int(os.getenv("DEFAULT_TOP_N", "15"))

# Cache configuration
ENABLE_METRIC_CACHE: bool = os.getenv("ENABLE_METRIC_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# ========== Constants ==========
# Valid chart types supported by the insight generator
VALID_CHART_TYPES = [
    "bar_horizontal",
    "bar_vertical",
    "bar_vertical_composed",
    "bar_vertical_stacked",
    "line",
    "line_composed",
    "pie",
    "histogram",
]

# Status values
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"


def validate_settings() -> bool:
    """
    Validate that all critical settings are properly configured.

    Returns:
        bool: True if all settings are valid

    Raises:
        ValueError: If critical settings are invalid or missing
        FileNotFoundError: If required files are missing
    """
    # Validate LLM API key
    # Gemini is the default provider; allow legacy OpenAI for backward compatibility.
    if not GEMINI_API_KEY and not OPENAI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found in environment (and no OPENAI_API_KEY fallback). "
            "Please set GEMINI_API_KEY in .env file or environment variables."
        )

    # Validate reasoning effort (legacy GPT-5 parameter - not used with Gemini)
    # NOTE: This validation is kept for backward compatibility but REASONING_EFFORT
    # is not used with Gemini models (only with legacy GPT-5 configurations)
    if (
        OPENAI_API_KEY and not GEMINI_API_KEY
    ):  # Only validate if using pure OpenAI (legacy)
        valid_efforts = ["minimal", "low", "medium", "high"]
        if REASONING_EFFORT not in valid_efforts:
            raise ValueError(
                f"REASONING_EFFORT must be one of {valid_efforts}, got: {REASONING_EFFORT}"
            )

    # Validate max completion tokens (legacy parameter - now max_output_tokens for Gemini)
    if MAX_COMPLETION_TOKENS < 100 or MAX_COMPLETION_TOKENS > 10000:
        raise ValueError(
            f"MAX_COMPLETION_TOKENS must be between 100-10000, got: {MAX_COMPLETION_TOKENS}"
        )

    # Validate rollout prompt mode
    valid_prompt_modes = {"legacy", "dynamic"}
    if INSIGHT_PROMPT_MODE not in valid_prompt_modes:
        raise ValueError(
            f"INSIGHT_PROMPT_MODE must be one of {sorted(valid_prompt_modes)}, got: {INSIGHT_PROMPT_MODE}"
        )

    # Validate max insights
    if MAX_INSIGHTS < 1 or MAX_INSIGHTS > 10:
        raise ValueError(f"MAX_INSIGHTS must be between 1-10, got: {MAX_INSIGHTS}")

    # Validate transparency threshold
    if TRANSPARENCY_THRESHOLD < 0 or TRANSPARENCY_THRESHOLD > 1:
        raise ValueError(
            f"TRANSPARENCY_THRESHOLD must be between 0-1, got: {TRANSPARENCY_THRESHOLD}"
        )

    # Validate dataset path exists
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH}. "
            "Please check DATASET_PATH in .env file."
        )

    return True
