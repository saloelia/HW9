"""
Data models for DeepFake Detector.

This module defines all data structures used throughout the detection system,
implementing the Building Blocks design pattern with clear Input/Output/Setup data.

Building Block Design:
    - Each model has clearly defined input data (attributes)
    - Output data is represented by computed properties or methods
    - Setup data includes configuration and defaults

Classes:
    DetectionVerdict: Enum for detection results
    VideoMetadata: Metadata about the analyzed video
    FaceData: Face detection and landmark data
    FrameAnalysis: Analysis results for a single frame
    TemporalAnalysisResult: Results from temporal analysis
    FrequencyAnalysisResult: Results from frequency analysis
    AnalysisResult: Complete analysis result with verdict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator


class DetectionVerdict(str, Enum):
    """
    Enumeration of possible detection verdicts.

    Attributes:
        REAL: Video appears to be authentic
        FAKE: Video appears to be a DeepFake
        UNCERTAIN: Unable to determine with confidence
        ERROR: Analysis could not be completed
    """

    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


class VideoMetadata(BaseModel):
    """
    Metadata about the video being analyzed.

    Input Data:
        - file_path: Path to the video file
        - duration: Video duration in seconds
        - fps: Frames per second
        - resolution: Video resolution (width, height)
        - codec: Video codec used
        - file_size: File size in bytes

    Output Data:
        - total_frames: Calculated total number of frames
        - aspect_ratio: Calculated aspect ratio
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_path: Path
    duration: float = Field(ge=0, description="Duration in seconds")
    fps: float = Field(ge=0, description="Frames per second")
    resolution: Tuple[int, int] = Field(description="(width, height)")
    codec: str = Field(default="unknown", description="Video codec")
    file_size: int = Field(ge=0, description="File size in bytes")
    creation_time: Optional[datetime] = None

    @property
    def total_frames(self) -> int:
        """Calculate total number of frames in the video."""
        return int(self.duration * self.fps)

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio of the video."""
        if self.resolution[1] == 0:
            return 0.0
        return self.resolution[0] / self.resolution[1]

    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)


class FaceData(BaseModel):
    """
    Face detection and landmark data for a single face.

    Input Data:
        - bbox: Bounding box coordinates (x, y, width, height)
        - confidence: Detection confidence score
        - landmarks: Facial landmark coordinates

    Output Data:
        - center: Calculated center point of the face
        - area: Calculated area of bounding box
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: Tuple[int, int, int, int] = Field(
        description="Bounding box (x, y, width, height)"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    landmarks: Optional[List[Tuple[float, float]]] = Field(
        default=None, description="Facial landmarks coordinates"
    )
    face_id: int = Field(default=0, description="Face identifier in frame")

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point of the face."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    @property
    def area(self) -> int:
        """Calculate area of the bounding box."""
        return self.bbox[2] * self.bbox[3]

    def get_eye_aspect_ratio(self) -> Optional[float]:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.

        Returns:
            EAR value if landmarks are available, None otherwise.
        """
        if self.landmarks is None or len(self.landmarks) < 68:
            return None

        # Eye landmarks indices (assuming 68-point model)
        left_eye = self.landmarks[36:42]
        right_eye = self.landmarks[42:48]

        def eye_ar(eye: List[Tuple[float, float]]) -> float:
            # Compute euclidean distances
            v1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            v2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            h = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

        left_ear = eye_ar(left_eye)
        right_ear = eye_ar(right_eye)
        return (left_ear + right_ear) / 2.0


class FrameAnalysis(BaseModel):
    """
    Analysis results for a single video frame.

    Input Data:
        - frame_number: Index of the frame in video
        - timestamp: Time position in video
        - faces: List of detected faces

    Output Data:
        - has_face: Whether any face was detected
        - primary_face: The main face in the frame
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_number: int = Field(ge=0, description="Frame index")
    timestamp: float = Field(ge=0.0, description="Timestamp in seconds")
    faces: List[FaceData] = Field(default_factory=list)
    quality_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Frame quality score"
    )
    blur_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Blur detection score"
    )
    anomaly_indicators: Dict[str, float] = Field(
        default_factory=dict, description="Anomaly indicators with scores"
    )

    @property
    def has_face(self) -> bool:
        """Check if any face was detected in this frame."""
        return len(self.faces) > 0

    @property
    def primary_face(self) -> Optional[FaceData]:
        """Get the primary (largest/most confident) face."""
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.confidence * f.area)


class TemporalAnalysisResult(BaseModel):
    """
    Results from temporal consistency analysis.

    Input Data:
        - blink_rate: Detected blink rate (blinks per minute)
        - blink_pattern_score: Score for natural blink patterns
        - eye_movement_score: Score for natural eye movements
        - facial_movement_consistency: Consistency of facial movements

    Output Data:
        - is_suspicious: Whether temporal patterns are suspicious
    """

    blink_rate: Optional[float] = Field(
        default=None, ge=0.0, description="Blinks per minute"
    )
    blink_pattern_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Natural blink pattern score"
    )
    eye_movement_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Natural eye movement score"
    )
    facial_movement_consistency: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Facial movement consistency"
    )
    lip_sync_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Audio-visual lip sync score"
    )
    head_pose_consistency: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Head pose consistency"
    )

    @property
    def is_suspicious(self) -> bool:
        """Determine if temporal patterns are suspicious."""
        scores = [
            self.blink_pattern_score,
            self.eye_movement_score,
            self.facial_movement_consistency,
        ]
        avg_score = sum(scores) / len(scores)
        return avg_score < 0.5

    @property
    def overall_score(self) -> float:
        """Calculate overall temporal analysis score."""
        weights = {
            "blink": 0.25,
            "eye": 0.25,
            "facial": 0.3,
            "head": 0.2,
        }
        return (
            self.blink_pattern_score * weights["blink"]
            + self.eye_movement_score * weights["eye"]
            + self.facial_movement_consistency * weights["facial"]
            + self.head_pose_consistency * weights["head"]
        )


class FrequencyAnalysisResult(BaseModel):
    """
    Results from frequency domain analysis.

    Input Data:
        - spectral_anomaly_score: Score for spectral anomalies
        - noise_pattern_score: Score for unnatural noise patterns
        - compression_artifact_score: Score for compression artifacts

    Output Data:
        - is_suspicious: Whether frequency patterns are suspicious
    """

    spectral_anomaly_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Spectral anomaly score"
    )
    noise_pattern_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Noise pattern naturalness"
    )
    compression_artifact_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Compression artifacts score"
    )
    high_frequency_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="High frequency component ratio"
    )
    gan_fingerprint_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="GAN fingerprint detection score"
    )

    @property
    def is_suspicious(self) -> bool:
        """Determine if frequency patterns are suspicious."""
        return self.overall_score < 0.5

    @property
    def overall_score(self) -> float:
        """Calculate overall frequency analysis score."""
        weights = {
            "spectral": 0.3,
            "noise": 0.25,
            "compression": 0.2,
            "gan": 0.25,
        }
        return (
            self.spectral_anomaly_score * weights["spectral"]
            + self.noise_pattern_score * weights["noise"]
            + self.compression_artifact_score * weights["compression"]
            + self.gan_fingerprint_score * weights["gan"]
        )


class OpticalFlowResult(BaseModel):
    """
    Results from optical flow analysis.

    Input Data:
        - flow_consistency_score: Score for flow consistency
        - boundary_artifact_score: Score for boundary artifacts
        - temporal_coherence_score: Score for temporal coherence
    """

    flow_consistency_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Optical flow consistency"
    )
    boundary_artifact_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Boundary artifacts naturalness"
    )
    temporal_coherence_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Temporal coherence"
    )

    @property
    def overall_score(self) -> float:
        """Calculate overall optical flow score."""
        return (
            self.flow_consistency_score * 0.4
            + self.boundary_artifact_score * 0.3
            + self.temporal_coherence_score * 0.3
        )


class AnalysisResult(BaseModel):
    """
    Complete analysis result with final verdict.

    This is the main output of the DeepFake detection system.

    Input Data:
        - video_metadata: Metadata about the analyzed video
        - frame_analyses: Analysis results for each frame
        - temporal_analysis: Temporal analysis results
        - frequency_analysis: Frequency analysis results
        - optical_flow: Optical flow analysis results

    Output Data:
        - verdict: Final detection verdict
        - confidence: Confidence level of the verdict
        - reasoning: LLM-generated reasoning for the verdict

    Setup Data:
        - analysis_timestamp: When analysis was performed
        - processing_time: Time taken for analysis
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Analysis identification
    analysis_id: str = Field(description="Unique analysis identifier")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    # Input data
    video_metadata: VideoMetadata
    frame_analyses: List[FrameAnalysis] = Field(default_factory=list)
    temporal_analysis: Optional[TemporalAnalysisResult] = None
    frequency_analysis: Optional[FrequencyAnalysisResult] = None
    optical_flow: Optional[OpticalFlowResult] = None

    # Output data
    verdict: DetectionVerdict = Field(default=DetectionVerdict.UNCERTAIN)
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Verdict confidence"
    )
    reasoning: str = Field(default="", description="LLM reasoning for verdict")
    evidence: List[str] = Field(
        default_factory=list, description="Evidence supporting the verdict"
    )

    # Metrics
    processing_time: float = Field(
        default=0.0, ge=0.0, description="Processing time in seconds"
    )
    frames_analyzed: int = Field(default=0, ge=0)
    faces_detected: int = Field(default=0, ge=0)

    # Token usage tracking
    total_tokens_used: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis result.

        Returns:
            Dictionary with key analysis metrics and verdict.
        """
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "video_file": str(self.video_metadata.file_path.name),
            "duration": round(self.video_metadata.duration, 2),
            "frames_analyzed": self.frames_analyzed,
            "faces_detected": self.faces_detected,
            "processing_time": round(self.processing_time, 2),
            "evidence_count": len(self.evidence),
            "tokens_used": self.total_tokens_used,
        }

    def get_detailed_scores(self) -> Dict[str, float]:
        """
        Get detailed scores from all analysis components.

        Returns:
            Dictionary with scores from each analysis method.
        """
        scores = {}

        if self.temporal_analysis:
            scores["temporal_overall"] = self.temporal_analysis.overall_score
            scores["blink_pattern"] = self.temporal_analysis.blink_pattern_score
            scores["eye_movement"] = self.temporal_analysis.eye_movement_score

        if self.frequency_analysis:
            scores["frequency_overall"] = self.frequency_analysis.overall_score
            scores["gan_fingerprint"] = self.frequency_analysis.gan_fingerprint_score

        if self.optical_flow:
            scores["optical_flow_overall"] = self.optical_flow.overall_score

        return scores


@dataclass
class AgentState:
    """
    Mutable state for the detection agent.

    This class tracks the agent's current state during analysis,
    following the Building Blocks pattern.

    Attributes:
        current_step: Current step in the analysis pipeline
        frames_processed: Number of frames processed so far
        evidence_collected: List of evidence found
        tool_results: Results from each tool execution
    """

    current_step: str = "initialization"
    frames_processed: int = 0
    evidence_collected: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    iteration_count: int = 0

    def add_evidence(self, evidence: str) -> None:
        """Add a piece of evidence to the collection."""
        if evidence not in self.evidence_collected:
            self.evidence_collected.append(evidence)

    def add_error(self, error: str) -> None:
        """Record an error that occurred during analysis."""
        self.errors.append(error)

    def update_tool_result(self, tool_name: str, result: Any) -> None:
        """Store result from a tool execution."""
        self.tool_results[tool_name] = result

    def reset(self) -> None:
        """Reset the agent state for a new analysis."""
        self.current_step = "initialization"
        self.frames_processed = 0
        self.evidence_collected = []
        self.tool_results = {}
        self.errors = []
        self.iteration_count = 0
