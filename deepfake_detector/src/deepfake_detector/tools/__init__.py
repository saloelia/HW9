"""
Tools module for DeepFake Detector.

This module contains all the analysis tools that the AI agent can use
to detect DeepFake videos. Each tool follows the Building Blocks pattern
with clear input/output/setup data.

Available Tools:
    - VideoExtractor: Extract frames from video files
    - FaceAnalyzer: Detect and analyze faces in frames
    - TemporalAnalyzer: Analyze temporal consistency
    - FrequencyAnalyzer: Analyze frequency domain patterns
    - OpticalFlowAnalyzer: Analyze pixel motion patterns
"""

from deepfake_detector.tools.video_extractor import VideoExtractor
from deepfake_detector.tools.face_analyzer import FaceAnalyzer
from deepfake_detector.tools.temporal_analyzer import TemporalAnalyzer
from deepfake_detector.tools.frequency_analyzer import FrequencyAnalyzer
from deepfake_detector.tools.optical_flow_analyzer import OpticalFlowAnalyzer
from deepfake_detector.tools.base import BaseTool, ToolResult

__all__ = [
    "BaseTool",
    "ToolResult",
    "VideoExtractor",
    "FaceAnalyzer",
    "TemporalAnalyzer",
    "FrequencyAnalyzer",
    "OpticalFlowAnalyzer",
]
