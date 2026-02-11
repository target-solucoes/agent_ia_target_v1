"""
Formatters module for insight_generator.

This module contains prompt builders and insight formatters.
"""

from .prompt_builder import build_prompt, build_system_prompt
from .insight_formatter import InsightFormatter
from .dynamic_prompt_builder import (
    DynamicPromptBuilder,
    build_dynamic_prompt,
    ANALYSIS_PERSONAS,
    FORMAT_RULES,
)

__all__ = [
    "build_prompt",
    "build_system_prompt",
    "InsightFormatter",
    "DynamicPromptBuilder",
    "build_dynamic_prompt",
    "ANALYSIS_PERSONAS",
    "FORMAT_RULES",
]
