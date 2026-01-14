"""
Unit tests for data models.
"""

import pytest
from pathlib import Path
from datetime import datetime

from deepfake_detector.core.models import (
    DetectionVerdict,
    VideoMetadata,
    FaceData,
    FrameAnalysis,
    TemporalAnalysisResult,
    FrequencyAnalysisResult,
    OpticalFlowResult,
    AnalysisResult,
    AgentState,
)


class TestDetectionVerdict:
    """Tests for DetectionVerdict enum."""

    def test_verdict_values(self):
        """Test all verdict values exist."""
        assert DetectionVerdict.REAL == "REAL"
        assert DetectionVerdict.FAKE == "FAKE"
        assert DetectionVerdict.UNCERTAIN == "UNCERTAIN"
        assert DetectionVerdict.ERROR == "ERROR"

    def test_verdict_from_string(self):
        """Test creating verdict from string."""
        verdict = DetectionVerdict("FAKE")
        assert verdict == DetectionVerdict.FAKE


class TestVideoMetadata:
    """Tests for VideoMetadata model."""

    def test_basic_creation(self, tmp_path):
        """Test creating VideoMetadata."""
        metadata = VideoMetadata(
            file_path=tmp_path / "test.mp4",
            duration=10.0,
            fps=30.0,
            resolution=(1920, 1080),
            codec="h264",
            file_size=1000000,
        )

        assert metadata.duration == 10.0
        assert metadata.fps == 30.0
        assert metadata.resolution == (1920, 1080)

    def test_total_frames_calculation(self, tmp_path):
        """Test total_frames property."""
        metadata = VideoMetadata(
            file_path=tmp_path / "test.mp4",
            duration=10.0,
            fps=30.0,
            resolution=(640, 480),
            codec="h264",
            file_size=1000000,
        )

        assert metadata.total_frames == 300

    def test_aspect_ratio_calculation(self, tmp_path):
        """Test aspect_ratio property."""
        metadata = VideoMetadata(
            file_path=tmp_path / "test.mp4",
            duration=10.0,
            fps=30.0,
            resolution=(1920, 1080),
            codec="h264",
            file_size=1000000,
        )

        assert abs(metadata.aspect_ratio - 16/9) < 0.01

    def test_file_size_mb(self, tmp_path):
        """Test file_size_mb property."""
        metadata = VideoMetadata(
            file_path=tmp_path / "test.mp4",
            duration=10.0,
            fps=30.0,
            resolution=(640, 480),
            codec="h264",
            file_size=10485760,  # 10 MB
        )

        assert metadata.file_size_mb == 10.0


class TestFaceData:
    """Tests for FaceData model."""

    def test_basic_creation(self):
        """Test creating FaceData."""
        face = FaceData(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            face_id=0,
        )

        assert face.bbox == (100, 100, 200, 200)
        assert face.confidence == 0.95

    def test_center_calculation(self):
        """Test center property."""
        face = FaceData(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
        )

        center = face.center
        assert center == (200.0, 200.0)

    def test_area_calculation(self):
        """Test area property."""
        face = FaceData(
            bbox=(0, 0, 100, 100),
            confidence=0.9,
        )

        assert face.area == 10000

    def test_landmarks(self):
        """Test with landmarks."""
        landmarks = [(i, i) for i in range(68)]
        face = FaceData(
            bbox=(0, 0, 100, 100),
            confidence=0.9,
            landmarks=landmarks,
        )

        assert len(face.landmarks) == 68


class TestFrameAnalysis:
    """Tests for FrameAnalysis model."""

    def test_basic_creation(self):
        """Test creating FrameAnalysis."""
        analysis = FrameAnalysis(
            frame_number=0,
            timestamp=0.0,
            faces=[],
        )

        assert analysis.frame_number == 0
        assert not analysis.has_face

    def test_with_faces(self):
        """Test with detected faces."""
        face = FaceData(bbox=(0, 0, 100, 100), confidence=0.9)
        analysis = FrameAnalysis(
            frame_number=0,
            timestamp=0.0,
            faces=[face],
        )

        assert analysis.has_face
        assert analysis.primary_face == face

    def test_primary_face_selection(self):
        """Test primary face is selected by confidence and area."""
        face1 = FaceData(bbox=(0, 0, 50, 50), confidence=0.8)
        face2 = FaceData(bbox=(0, 0, 100, 100), confidence=0.9)

        analysis = FrameAnalysis(
            frame_number=0,
            timestamp=0.0,
            faces=[face1, face2],
        )

        assert analysis.primary_face == face2


class TestTemporalAnalysisResult:
    """Tests for TemporalAnalysisResult model."""

    def test_basic_creation(self):
        """Test creating TemporalAnalysisResult."""
        result = TemporalAnalysisResult(
            blink_rate=15.0,
            blink_pattern_score=0.8,
            eye_movement_score=0.7,
            facial_movement_consistency=0.75,
        )

        assert result.blink_rate == 15.0
        assert not result.is_suspicious

    def test_suspicious_detection(self):
        """Test is_suspicious property."""
        result = TemporalAnalysisResult(
            blink_pattern_score=0.3,
            eye_movement_score=0.3,
            facial_movement_consistency=0.3,
        )

        assert result.is_suspicious

    def test_overall_score(self):
        """Test overall_score calculation."""
        result = TemporalAnalysisResult(
            blink_pattern_score=1.0,
            eye_movement_score=1.0,
            facial_movement_consistency=1.0,
            head_pose_consistency=1.0,
        )

        assert result.overall_score == 1.0


class TestFrequencyAnalysisResult:
    """Tests for FrequencyAnalysisResult model."""

    def test_basic_creation(self):
        """Test creating FrequencyAnalysisResult."""
        result = FrequencyAnalysisResult(
            spectral_anomaly_score=0.8,
            noise_pattern_score=0.7,
            gan_fingerprint_score=0.85,
        )

        assert result.gan_fingerprint_score == 0.85
        assert not result.is_suspicious

    def test_suspicious_detection(self):
        """Test is_suspicious property."""
        result = FrequencyAnalysisResult(
            spectral_anomaly_score=0.3,
            noise_pattern_score=0.3,
            gan_fingerprint_score=0.3,
        )

        assert result.is_suspicious


class TestOpticalFlowResult:
    """Tests for OpticalFlowResult model."""

    def test_basic_creation(self):
        """Test creating OpticalFlowResult."""
        result = OpticalFlowResult(
            flow_consistency_score=0.8,
            boundary_artifact_score=0.7,
            temporal_coherence_score=0.75,
        )

        assert result.flow_consistency_score == 0.8

    def test_overall_score(self):
        """Test overall_score calculation."""
        result = OpticalFlowResult(
            flow_consistency_score=1.0,
            boundary_artifact_score=1.0,
            temporal_coherence_score=1.0,
        )

        assert result.overall_score == 1.0


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_basic_creation(self, tmp_path):
        """Test creating AnalysisResult."""
        metadata = VideoMetadata(
            file_path=tmp_path / "test.mp4",
            duration=10.0,
            fps=30.0,
            resolution=(640, 480),
            codec="h264",
            file_size=1000000,
        )

        result = AnalysisResult(
            analysis_id="test123",
            video_metadata=metadata,
            verdict=DetectionVerdict.FAKE,
            confidence=0.85,
            reasoning="Test reasoning",
        )

        assert result.verdict == DetectionVerdict.FAKE
        assert result.confidence == 0.85

    def test_get_summary(self, tmp_path):
        """Test get_summary method."""
        metadata = VideoMetadata(
            file_path=tmp_path / "test.mp4",
            duration=10.0,
            fps=30.0,
            resolution=(640, 480),
            codec="h264",
            file_size=1000000,
        )

        result = AnalysisResult(
            analysis_id="test123",
            video_metadata=metadata,
            verdict=DetectionVerdict.REAL,
            confidence=0.9,
        )

        summary = result.get_summary()
        assert summary["verdict"] == "REAL"
        assert summary["confidence"] == 0.9


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_basic_creation(self):
        """Test creating AgentState."""
        state = AgentState()

        assert state.current_step == "initialization"
        assert state.frames_processed == 0
        assert len(state.evidence_collected) == 0

    def test_add_evidence(self):
        """Test add_evidence method."""
        state = AgentState()
        state.add_evidence("Test evidence 1")
        state.add_evidence("Test evidence 2")
        state.add_evidence("Test evidence 1")  # Duplicate

        assert len(state.evidence_collected) == 2

    def test_add_error(self):
        """Test add_error method."""
        state = AgentState()
        state.add_error("Error 1")
        state.add_error("Error 2")

        assert len(state.errors) == 2

    def test_reset(self):
        """Test reset method."""
        state = AgentState()
        state.current_step = "analysis"
        state.add_evidence("Evidence")
        state.add_error("Error")

        state.reset()

        assert state.current_step == "initialization"
        assert len(state.evidence_collected) == 0
        assert len(state.errors) == 0
