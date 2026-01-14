"""
Agent module for DeepFake Detector.

This module contains the AI agent implementation that orchestrates
the detection process using LLM-powered reasoning and tool execution.

Key Components:
    - DeepFakeDetectorAgent: Main agent class with tool use
    - Prompts: System and user prompts for agent reasoning
"""

from deepfake_detector.agent.detector_agent import DeepFakeDetectorAgent
from deepfake_detector.agent.prompts import (
    SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    REASONING_PROMPT_TEMPLATE,
)

__all__ = [
    "DeepFakeDetectorAgent",
    "SYSTEM_PROMPT",
    "ANALYSIS_PROMPT_TEMPLATE",
    "REASONING_PROMPT_TEMPLATE",
]
