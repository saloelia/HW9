"""
Core module for DeepFake Detector.

This module contains the main detector class and data models
that form the foundation of the detection system.
"""

from deepfake_detector.core.detector import DeepFakeDetector
from deepfake_detector.core.models import (
    AnalysisResult,
    DetectionVerdict,
    FrameAnalysis,
    VideoMetadata,
    FaceData,
    TemporalAnalysisResult,
    FrequencyAnalysisResult,
)

__all__ = [
    "DeepFakeDetector",
    "AnalysisResult",
    "DetectionVerdict",
    "FrameAnalysis",
    "VideoMetadata",
    "FaceData",
    "TemporalAnalysisResult",
    "FrequencyAnalysisResult",
]
