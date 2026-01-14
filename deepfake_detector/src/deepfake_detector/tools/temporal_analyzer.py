"""
Temporal analysis tool for DeepFake detection.

This module analyzes temporal patterns in video frames to detect
unnatural movements and behaviors that indicate DeepFake manipulation.

Key Detection Methods:
    1. Blink detection and rate analysis
    2. Eye movement pattern analysis
    3. Facial movement consistency
    4. Head pose tracking

Building Block Design:
    Input Data:
        - frame_analyses: List of FrameAnalysis from face analyzer
        - fps: Video frame rate

    Output Data:
        - temporal_result: TemporalAnalysisResult
        - anomalies: List of detected temporal anomalies

    Setup Data:
        - normal_blink_rate: Expected blink rate range
        - eye_movement_threshold: Threshold for eye movement anomalies
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from deepfake_detector.tools.base import BaseTool, ToolConfig, ToolResult
from deepfake_detector.core.models import FrameAnalysis, TemporalAnalysisResult


class TemporalAnalyzerConfig(ToolConfig):
    """Configuration for TemporalAnalyzer tool."""

    # Normal human blink rate: 15-20 per minute
    normal_blink_rate_min: float = 10.0
    normal_blink_rate_max: float = 30.0

    # Eye Aspect Ratio threshold for blink detection
    ear_threshold: float = 0.2

    # Consecutive frames below threshold to count as blink
    blink_consecutive_frames: int = 2

    # Movement analysis thresholds
    movement_smoothness_threshold: float = 0.7


class TemporalAnalyzer(BaseTool):
    """
    Tool for analyzing temporal patterns in video frames.

    This tool examines how facial features change over time to detect
    unnatural patterns that may indicate DeepFake manipulation.

    Detection Methods:
        1. Blink Detection: Analyzes eye aspect ratio over time to detect
           blinks and measure blink rate. DeepFakes often have abnormal
           blink patterns.

        2. Eye Movement Analysis: Tracks pupil positions and movements.
           Natural eye movements have specific patterns (saccades, fixations)
           that DeepFakes may not replicate correctly.

        3. Facial Movement Consistency: Analyzes how facial landmarks move
           relative to each other. Inconsistent movements suggest manipulation.

        4. Head Pose Tracking: Monitors head orientation changes for
           unnatural sudden movements or jitter.

    Example:
        >>> analyzer = TemporalAnalyzer()
        >>> result = analyzer.execute(
        ...     frame_analyses=frame_data,
        ...     fps=30.0
        ... )
        >>> if result.success:
        ...     print(f"Blink rate: {result.data['temporal_result'].blink_rate}")
    """

    def __init__(self, config: Optional[TemporalAnalyzerConfig] = None) -> None:
        """
        Initialize the TemporalAnalyzer.

        Args:
            config: Tool configuration (Setup Data)
        """
        super().__init__(
            name="temporal_analyzer",
            description=(
                "Analyze temporal patterns in video frames including "
                "blink rate, eye movements, and facial movement consistency. "
                "Returns scores indicating how natural the temporal patterns are. "
                "Lower scores suggest potential DeepFake manipulation."
            ),
            config=config or TemporalAnalyzerConfig(),
        )
        self.config: TemporalAnalyzerConfig = config or TemporalAnalyzerConfig()

    def execute(
        self,
        frame_analyses: List[FrameAnalysis],
        fps: float = 30.0,
        analyze_blinks: bool = True,
        analyze_eye_movement: bool = True,
        analyze_facial_movement: bool = True,
    ) -> ToolResult[Dict[str, Any]]:
        """
        Analyze temporal patterns in frame analyses.

        Args:
            frame_analyses: List of FrameAnalysis objects (Input Data)
            fps: Frames per second of the video
            analyze_blinks: Whether to analyze blink patterns
            analyze_eye_movement: Whether to analyze eye movements
            analyze_facial_movement: Whether to analyze facial movements

        Returns:
            ToolResult containing:
                - temporal_result: TemporalAnalysisResult object
                - anomalies: List of detected anomalies
                - detailed_scores: Dictionary of all computed scores
        """
        start = time.time()
        self._execution_count += 1

        try:
            if not frame_analyses:
                return ToolResult.error_result("No frame analyses provided")

            anomalies: List[str] = []
            detailed_scores: Dict[str, float] = {}

            # Extract EAR values for blink detection
            ear_values = self._extract_ear_values(frame_analyses)

            # Blink analysis
            blink_rate = None
            blink_pattern_score = 0.5

            if analyze_blinks and ear_values:
                blink_rate, blink_pattern_score = self._analyze_blinks(
                    ear_values, fps
                )
                detailed_scores["blink_rate"] = blink_rate if blink_rate else 0
                detailed_scores["blink_pattern_score"] = blink_pattern_score

                if blink_rate is not None:
                    if blink_rate < self.config.normal_blink_rate_min:
                        anomalies.append(
                            f"Abnormally low blink rate: {blink_rate:.1f}/min "
                            f"(normal: {self.config.normal_blink_rate_min}-"
                            f"{self.config.normal_blink_rate_max})"
                        )
                    elif blink_rate > self.config.normal_blink_rate_max:
                        anomalies.append(
                            f"Abnormally high blink rate: {blink_rate:.1f}/min"
                        )

            # Eye movement analysis
            eye_movement_score = 0.5
            if analyze_eye_movement:
                eye_movement_score = self._analyze_eye_movements(frame_analyses)
                detailed_scores["eye_movement_score"] = eye_movement_score

                if eye_movement_score < 0.4:
                    anomalies.append(
                        "Unnatural eye movement patterns detected"
                    )

            # Facial movement consistency
            facial_movement_score = 0.5
            head_pose_score = 0.5

            if analyze_facial_movement:
                facial_movement_score = self._analyze_facial_movements(
                    frame_analyses
                )
                head_pose_score = self._analyze_head_pose(frame_analyses)

                detailed_scores["facial_movement_score"] = facial_movement_score
                detailed_scores["head_pose_score"] = head_pose_score

                if facial_movement_score < 0.4:
                    anomalies.append(
                        "Inconsistent facial movements detected"
                    )

                if head_pose_score < 0.4:
                    anomalies.append(
                        "Unnatural head pose changes detected"
                    )

            # Create temporal analysis result
            temporal_result = TemporalAnalysisResult(
                blink_rate=blink_rate,
                blink_pattern_score=blink_pattern_score,
                eye_movement_score=eye_movement_score,
                facial_movement_consistency=facial_movement_score,
                head_pose_consistency=head_pose_score,
            )

            execution_time = time.time() - start
            self._total_execution_time += execution_time

            return ToolResult.success_result(
                data={
                    "temporal_result": temporal_result,
                    "anomalies": anomalies,
                    "detailed_scores": detailed_scores,
                    "overall_score": temporal_result.overall_score,
                },
                metadata={
                    "execution_time": round(execution_time, 3),
                    "frames_analyzed": len(frame_analyses),
                    "anomalies_found": len(anomalies),
                },
            )

        except Exception as e:
            execution_time = time.time() - start
            self._total_execution_time += execution_time
            return ToolResult.error_result(
                f"Error in temporal analysis: {str(e)}",
                metadata={"execution_time": round(execution_time, 3)},
            )

    def _extract_ear_values(
        self, frame_analyses: List[FrameAnalysis]
    ) -> List[Optional[float]]:
        """
        Extract Eye Aspect Ratio values from frame analyses.

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            List of EAR values (None if no face/landmarks detected)
        """
        ear_values: List[Optional[float]] = []

        for fa in frame_analyses:
            if fa.primary_face and fa.primary_face.landmarks:
                ear = fa.primary_face.get_eye_aspect_ratio()
                ear_values.append(ear)
            else:
                ear_values.append(None)

        return ear_values

    def _analyze_blinks(
        self, ear_values: List[Optional[float]], fps: float
    ) -> tuple[Optional[float], float]:
        """
        Analyze blink patterns from EAR values.

        Args:
            ear_values: List of Eye Aspect Ratio values
            fps: Frames per second

        Returns:
            Tuple of (blink_rate, pattern_score)
        """
        # Filter valid EAR values
        valid_ears = [e for e in ear_values if e is not None]

        if len(valid_ears) < 10:
            return None, 0.5

        # Detect blinks (EAR drops below threshold)
        blinks = 0
        in_blink = False
        blink_frames = 0
        blink_durations: List[int] = []

        for ear in valid_ears:
            if ear < self.config.ear_threshold:
                if not in_blink:
                    in_blink = True
                    blink_frames = 1
                else:
                    blink_frames += 1
            else:
                if in_blink and blink_frames >= self.config.blink_consecutive_frames:
                    blinks += 1
                    blink_durations.append(blink_frames)
                in_blink = False
                blink_frames = 0

        # Calculate blink rate (per minute)
        duration_seconds = len(valid_ears) / fps
        duration_minutes = duration_seconds / 60.0

        if duration_minutes > 0:
            blink_rate = blinks / duration_minutes
        else:
            return None, 0.5

        # Calculate pattern score
        # Regular blinks with consistent duration are more natural
        pattern_score = 0.5

        if blinks > 0:
            # Check if blink rate is in normal range
            if (
                self.config.normal_blink_rate_min
                <= blink_rate
                <= self.config.normal_blink_rate_max
            ):
                rate_score = 1.0
            else:
                deviation = min(
                    abs(blink_rate - self.config.normal_blink_rate_min),
                    abs(blink_rate - self.config.normal_blink_rate_max),
                )
                rate_score = max(0, 1.0 - deviation / 20.0)

            # Check blink duration consistency
            if blink_durations:
                duration_std = np.std(blink_durations)
                duration_score = max(0, 1.0 - duration_std / 5.0)
            else:
                duration_score = 0.5

            pattern_score = (rate_score + duration_score) / 2

        return blink_rate, pattern_score

    def _analyze_eye_movements(
        self, frame_analyses: List[FrameAnalysis]
    ) -> float:
        """
        Analyze eye movement patterns for naturalness.

        Natural eye movements include:
        - Saccades (rapid movements between fixation points)
        - Smooth pursuit (following moving objects)
        - Micro-saccades during fixation

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            Score indicating naturalness of eye movements (0-1)
        """
        # Extract eye positions from landmarks
        eye_positions: List[tuple[float, float]] = []

        for fa in frame_analyses:
            if fa.primary_face and fa.primary_face.landmarks:
                landmarks = fa.primary_face.landmarks
                if len(landmarks) >= 48:
                    # Average of eye landmark positions
                    left_eye = np.mean(landmarks[36:42], axis=0)
                    right_eye = np.mean(landmarks[42:48], axis=0)
                    eye_center = ((left_eye[0] + right_eye[0]) / 2,
                                  (left_eye[1] + right_eye[1]) / 2)
                    eye_positions.append(eye_center)

        if len(eye_positions) < 5:
            return 0.5

        # Calculate movement velocities
        velocities: List[float] = []
        for i in range(1, len(eye_positions)):
            dx = eye_positions[i][0] - eye_positions[i - 1][0]
            dy = eye_positions[i][1] - eye_positions[i - 1][1]
            velocity = np.sqrt(dx ** 2 + dy ** 2)
            velocities.append(velocity)

        if not velocities:
            return 0.5

        # Natural eye movements have varied velocities
        # (mix of saccades and fixations)
        velocity_std = np.std(velocities)
        velocity_mean = np.mean(velocities)

        # Check for natural variation
        cv = velocity_std / velocity_mean if velocity_mean > 0 else 0

        # Natural movements have coefficient of variation around 0.5-2.0
        if 0.3 <= cv <= 2.5:
            variation_score = 1.0
        else:
            variation_score = max(0, 1.0 - abs(cv - 1.0) / 2.0)

        # Check for unnatural stillness (too consistent)
        stillness_score = min(velocity_std / 2.0, 1.0)

        return (variation_score + stillness_score) / 2

    def _analyze_facial_movements(
        self, frame_analyses: List[FrameAnalysis]
    ) -> float:
        """
        Analyze facial movement consistency.

        Checks if different parts of the face move coherently.

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            Score indicating movement consistency (0-1)
        """
        if len(frame_analyses) < 3:
            return 0.5

        consistency_scores: List[float] = []

        for i in range(1, len(frame_analyses)):
            prev_fa = frame_analyses[i - 1]
            curr_fa = frame_analyses[i]

            if (
                prev_fa.primary_face
                and curr_fa.primary_face
                and prev_fa.primary_face.landmarks
                and curr_fa.primary_face.landmarks
            ):
                prev_landmarks = np.array(prev_fa.primary_face.landmarks)
                curr_landmarks = np.array(curr_fa.primary_face.landmarks)

                if len(prev_landmarks) == len(curr_landmarks):
                    # Calculate movement of different face regions
                    movements = curr_landmarks - prev_landmarks

                    # Check if movements are consistent across face
                    movement_mags = np.linalg.norm(movements, axis=1)

                    # Standard deviation of movement magnitudes
                    # Lower std = more consistent movement
                    movement_std = np.std(movement_mags)
                    movement_mean = np.mean(movement_mags)

                    if movement_mean > 0:
                        consistency = max(0, 1.0 - movement_std / movement_mean)
                    else:
                        consistency = 1.0

                    consistency_scores.append(consistency)

        if consistency_scores:
            return np.mean(consistency_scores)
        return 0.5

    def _analyze_head_pose(
        self, frame_analyses: List[FrameAnalysis]
    ) -> float:
        """
        Analyze head pose changes for naturalness.

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            Score indicating natural head pose changes (0-1)
        """
        # Extract face bounding box centers as proxy for head position
        positions: List[tuple[float, float]] = []

        for fa in frame_analyses:
            if fa.primary_face:
                positions.append(fa.primary_face.center)

        if len(positions) < 3:
            return 0.5

        # Calculate position changes
        changes: List[float] = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            change = np.sqrt(dx ** 2 + dy ** 2)
            changes.append(change)

        # Analyze change pattern
        # Natural head movements are smooth, not jittery
        change_std = np.std(changes)
        change_mean = np.mean(changes)

        # Check for jitter (high variation in small movements)
        if change_mean < 5:  # Small movements
            jitter_score = max(0, 1.0 - change_std / 3.0)
        else:
            jitter_score = max(0, 1.0 - change_std / change_mean)

        # Check for unnaturally smooth movement
        if change_std < 0.1 and change_mean > 0:
            smoothness_penalty = 0.3
        else:
            smoothness_penalty = 0

        return max(0, jitter_score - smoothness_penalty)

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool inputs."""
        return {
            "type": "object",
            "properties": {
                "frame_analyses": {
                    "type": "array",
                    "description": "List of FrameAnalysis objects from face analyzer",
                    "items": {"type": "object"},
                },
                "fps": {
                    "type": "number",
                    "description": "Frames per second of the video",
                    "default": 30.0,
                    "minimum": 1.0,
                },
                "analyze_blinks": {
                    "type": "boolean",
                    "description": "Whether to analyze blink patterns",
                    "default": True,
                },
                "analyze_eye_movement": {
                    "type": "boolean",
                    "description": "Whether to analyze eye movements",
                    "default": True,
                },
                "analyze_facial_movement": {
                    "type": "boolean",
                    "description": "Whether to analyze facial movements",
                    "default": True,
                },
            },
            "required": ["frame_analyses"],
        }
