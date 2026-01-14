"""
DeepFake Detector - AI Agent-based DeepFake Video Detection System

This package provides a comprehensive solution for detecting DeepFake videos
using an AI agent architecture with multiple specialized tools for video analysis.

Key Features:
    - AI Agent with LLM-powered reasoning
    - Multi-modal analysis (visual, temporal, frequency)
    - Extensible tool architecture
    - Detailed reporting and visualization

Example Usage:
    >>> from deepfake_detector import DeepFakeDetector
    >>> detector = DeepFakeDetector()
    >>> result = detector.analyze("path/to/video.mp4")
    >>> print(result.verdict)
    'FAKE'
    >>> print(result.confidence)
    0.85

Author: Student
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Student"
__email__ = "student@university.edu"

from deepfake_detector.core.detector import DeepFakeDetector
from deepfake_detector.core.models import (
    AnalysisResult,
    DetectionVerdict,
    FrameAnalysis,
    VideoMetadata,
)
from deepfake_detector.agent.detector_agent import DeepFakeDetectorAgent

__all__ = [
    "DeepFakeDetector",
    "DeepFakeDetectorAgent",
    "AnalysisResult",
    "DetectionVerdict",
    "FrameAnalysis",
    "VideoMetadata",
    "__version__",
]
