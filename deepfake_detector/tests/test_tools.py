"""
Unit tests for detection tools.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from deepfake_detector.tools.base import BaseTool, ToolResult, ToolConfig
from deepfake_detector.tools.video_extractor import VideoExtractor, VideoExtractorConfig
from deepfake_detector.tools.face_analyzer import FaceAnalyzer, FaceAnalyzerConfig
from deepfake_detector.tools.temporal_analyzer import TemporalAnalyzer
from deepfake_detector.tools.frequency_analyzer import FrequencyAnalyzer
from deepfake_detector.tools.optical_flow_analyzer import OpticalFlowAnalyzer
from deepfake_detector.core.models import FrameAnalysis, FaceData


class TestToolResult:
    """Tests for ToolResult class."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ToolResult.success_result({"key": "value"})

        assert result.success
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult.error_result("Something went wrong")

        assert not result.success
        assert result.data is None
        assert result.error == "Something went wrong"

    def test_error_result_default_message(self):
        """Test error result gets default message if none provided."""
        result = ToolResult(success=False)

        assert result.error == "Unknown error occurred"


class TestVideoExtractor:
    """Tests for VideoExtractor tool."""

    def test_initialization(self):
        """Test VideoExtractor initialization."""
        extractor = VideoExtractor()

        assert extractor.name == "video_extractor"
        assert extractor.config.max_frames == 30

    def test_initialization_with_config(self):
        """Test VideoExtractor with custom config."""
        config = VideoExtractorConfig(max_frames=50, sample_rate=10)
        extractor = VideoExtractor(config=config)

        assert extractor.config.max_frames == 50
        assert extractor.config.sample_rate == 10

    def test_get_schema(self):
        """Test get_schema returns valid schema."""
        extractor = VideoExtractor()
        schema = extractor.get_schema()

        assert schema["type"] == "object"
        assert "video_path" in schema["properties"]
        assert "video_path" in schema["required"]

    def test_execute_file_not_found(self):
        """Test execute with non-existent file."""
        extractor = VideoExtractor()
        result = extractor.execute(video_path="/nonexistent/path.mp4")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_execute_unsupported_format(self, tmp_path):
        """Test execute with unsupported format."""
        # Create a dummy file
        test_file = tmp_path / "test.xyz"
        test_file.write_text("dummy")

        extractor = VideoExtractor()
        result = extractor.execute(video_path=str(test_file))

        assert not result.success
        assert "unsupported" in result.error.lower()

    def test_tool_definition(self):
        """Test get_tool_definition method."""
        extractor = VideoExtractor()
        definition = extractor.get_tool_definition()

        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition


class TestFaceAnalyzer:
    """Tests for FaceAnalyzer tool."""

    def test_initialization(self):
        """Test FaceAnalyzer initialization."""
        analyzer = FaceAnalyzer()

        assert analyzer.name == "face_analyzer"
        assert analyzer.config.detection_confidence == 0.8

    def test_execute_empty_frames(self):
        """Test execute with empty frames list."""
        analyzer = FaceAnalyzer()
        result = analyzer.execute(frames=[])

        assert not result.success
        assert "no frames" in result.error.lower()

    def test_execute_with_frames(self):
        """Test execute with valid frames."""
        analyzer = FaceAnalyzer()

        # Create dummy RGB frames
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                  for _ in range(5)]

        result = analyzer.execute(frames=frames)

        assert result.success
        assert "frame_analyses" in result.data
        assert len(result.data["frame_analyses"]) == 5

    def test_get_schema(self):
        """Test get_schema returns valid schema."""
        analyzer = FaceAnalyzer()
        schema = analyzer.get_schema()

        assert schema["type"] == "object"
        assert "frames" in schema["properties"]
        assert "frames" in schema["required"]


class TestTemporalAnalyzer:
    """Tests for TemporalAnalyzer tool."""

    def test_initialization(self):
        """Test TemporalAnalyzer initialization."""
        analyzer = TemporalAnalyzer()

        assert analyzer.name == "temporal_analyzer"
        assert analyzer.config.normal_blink_rate_min == 10.0

    def test_execute_empty_analyses(self):
        """Test execute with empty frame analyses."""
        analyzer = TemporalAnalyzer()
        result = analyzer.execute(frame_analyses=[])

        assert not result.success

    def test_execute_with_analyses(self):
        """Test execute with valid frame analyses."""
        analyzer = TemporalAnalyzer()

        # Create mock frame analyses
        analyses = []
        for i in range(10):
            face = FaceData(
                bbox=(100, 100, 200, 200),
                confidence=0.9,
                landmarks=[(j, j) for j in range(68)],
            )
            analysis = FrameAnalysis(
                frame_number=i,
                timestamp=i * 0.1,
                faces=[face],
            )
            analyses.append(analysis)

        result = analyzer.execute(frame_analyses=analyses, fps=30.0)

        assert result.success
        assert "temporal_result" in result.data
        assert "overall_score" in result.data


class TestFrequencyAnalyzer:
    """Tests for FrequencyAnalyzer tool."""

    def test_initialization(self):
        """Test FrequencyAnalyzer initialization."""
        analyzer = FrequencyAnalyzer()

        assert analyzer.name == "frequency_analyzer"
        assert analyzer.config.fft_enabled

    def test_execute_empty_frames(self):
        """Test execute with empty frames."""
        analyzer = FrequencyAnalyzer()
        result = analyzer.execute(frames=[])

        assert not result.success

    def test_execute_with_frames(self):
        """Test execute with valid frames."""
        analyzer = FrequencyAnalyzer()

        # Create dummy frames
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                  for _ in range(5)]

        result = analyzer.execute(frames=frames)

        assert result.success
        assert "frequency_result" in result.data
        assert "overall_score" in result.data


class TestOpticalFlowAnalyzer:
    """Tests for OpticalFlowAnalyzer tool."""

    def test_initialization(self):
        """Test OpticalFlowAnalyzer initialization."""
        analyzer = OpticalFlowAnalyzer()

        assert analyzer.name == "optical_flow_analyzer"
        assert analyzer.config.algorithm == "farneback"

    def test_execute_insufficient_frames(self):
        """Test execute with less than 2 frames."""
        analyzer = OpticalFlowAnalyzer()

        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
        result = analyzer.execute(frames=frames)

        assert not result.success
        assert "at least 2 frames" in result.error.lower()

    def test_execute_with_frames(self):
        """Test execute with valid frames."""
        analyzer = OpticalFlowAnalyzer()

        # Create sequential frames with slight movement
        frames = []
        for i in range(5):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Add a moving square
            x = 10 + i * 5
            frame[40:60, x:x+20] = 255
            frames.append(frame)

        result = analyzer.execute(frames=frames)

        assert result.success
        assert "optical_flow_result" in result.data

    def test_visualize_flow(self):
        """Test flow visualization."""
        analyzer = OpticalFlowAnalyzer()

        # Create dummy flow
        flow = np.random.rand(100, 100, 2).astype(np.float32)
        visualization = analyzer.visualize_flow(flow)

        assert visualization.shape == (100, 100, 3)


class TestToolStats:
    """Tests for tool statistics tracking."""

    def test_stats_tracking(self):
        """Test that tool tracks execution statistics."""
        analyzer = FaceAnalyzer()

        # Execute multiple times
        for _ in range(3):
            frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
            analyzer.execute(frames=frames)

        stats = analyzer.get_stats()

        assert stats["name"] == "face_analyzer"
        assert stats["execution_count"] == 3
        assert stats["total_time"] > 0
