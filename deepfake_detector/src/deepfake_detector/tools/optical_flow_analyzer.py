"""
Optical flow analysis tool for DeepFake detection.

This module analyzes pixel motion patterns between consecutive frames
to detect unnatural movements that indicate DeepFake manipulation.

Key Detection Methods:
    1. Flow consistency analysis
    2. Boundary artifact detection
    3. Temporal coherence checking

Building Block Design:
    Input Data:
        - frames: List of consecutive video frames

    Output Data:
        - optical_flow_result: OpticalFlowResult
        - flow_maps: Computed optical flow fields

    Setup Data:
        - algorithm: Flow computation algorithm
        - params: Algorithm-specific parameters
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deepfake_detector.tools.base import BaseTool, ToolConfig, ToolResult
from deepfake_detector.core.models import OpticalFlowResult


class OpticalFlowConfig(ToolConfig):
    """Configuration for OpticalFlowAnalyzer tool."""

    algorithm: str = "farneback"  # or "lucas_kanade"
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    anomaly_threshold: float = 0.65


class OpticalFlowAnalyzer(BaseTool):
    """
    Tool for optical flow analysis of video frames.

    This tool computes optical flow between consecutive frames and
    analyzes the flow patterns to detect DeepFake manipulation.

    Detection Methods:
        1. Flow Consistency: Natural videos have smooth, consistent
           optical flow. DeepFakes may show discontinuities or
           unnatural flow patterns at face boundaries.

        2. Boundary Artifacts: Analyzes flow patterns at face/background
           boundaries. Manipulation often causes flow discontinuities
           at these boundaries.

        3. Temporal Coherence: Checks if flow patterns are temporally
           consistent across multiple frames.

    Example:
        >>> analyzer = OpticalFlowAnalyzer()
        >>> result = analyzer.execute(frames=frame_list)
        >>> if result.success:
        ...     print(f"Flow consistency: "
        ...           f"{result.data['optical_flow_result'].flow_consistency_score}")
    """

    def __init__(self, config: Optional[OpticalFlowConfig] = None) -> None:
        """
        Initialize the OpticalFlowAnalyzer.

        Args:
            config: Tool configuration (Setup Data)
        """
        super().__init__(
            name="optical_flow_analyzer",
            description=(
                "Analyze optical flow patterns between video frames. "
                "Detects unnatural pixel motion, boundary artifacts, and "
                "temporal inconsistencies that indicate DeepFake manipulation."
            ),
            config=config or OpticalFlowConfig(),
        )
        self.config: OpticalFlowConfig = config or OpticalFlowConfig()

    def execute(
        self,
        frames: List[np.ndarray],
        face_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        compute_full_flow: bool = False,
    ) -> ToolResult[Dict[str, Any]]:
        """
        Analyze optical flow between consecutive frames.

        Args:
            frames: List of frame arrays (RGB format) (Input Data)
            face_regions: Optional face bounding boxes for boundary analysis
            compute_full_flow: Whether to return full flow maps

        Returns:
            ToolResult containing:
                - optical_flow_result: OpticalFlowResult object
                - anomalies: List of detected anomalies
                - flow_statistics: Flow computation statistics
        """
        start = time.time()
        self._execution_count += 1

        try:
            if len(frames) < 2:
                return ToolResult.error_result(
                    "At least 2 frames required for optical flow analysis"
                )

            anomalies: List[str] = []
            flow_statistics: Dict[str, Any] = {}

            # Convert frames to grayscale
            gray_frames = [
                cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames
            ]

            # Compute optical flow between consecutive frames
            flow_maps = []
            for i in range(len(gray_frames) - 1):
                flow = self._compute_flow(gray_frames[i], gray_frames[i + 1])
                flow_maps.append(flow)

            # Analyze flow consistency
            flow_consistency_score = self._analyze_flow_consistency(flow_maps)
            flow_statistics["consistency_score"] = flow_consistency_score

            if flow_consistency_score < 0.4:
                anomalies.append(
                    "Inconsistent optical flow patterns detected - "
                    "possible frame manipulation"
                )

            # Analyze boundary artifacts
            boundary_score = 0.5
            if face_regions:
                boundary_score = self._analyze_boundary_flow(
                    flow_maps, face_regions, frames[0].shape[:2]
                )
                flow_statistics["boundary_score"] = boundary_score

                if boundary_score < 0.4:
                    anomalies.append(
                        "Flow discontinuities at face boundaries - "
                        "possible face replacement"
                    )
            else:
                # Estimate face regions and analyze
                boundary_score = self._estimate_boundary_artifacts(flow_maps)
                flow_statistics["boundary_score"] = boundary_score

            # Analyze temporal coherence
            temporal_score = self._analyze_temporal_coherence(flow_maps)
            flow_statistics["temporal_coherence"] = temporal_score

            if temporal_score < 0.4:
                anomalies.append(
                    "Temporally incoherent flow patterns detected"
                )

            # Calculate flow magnitude statistics
            magnitudes = []
            for flow in flow_maps:
                mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
                magnitudes.append({
                    "mean": float(np.mean(mag)),
                    "std": float(np.std(mag)),
                    "max": float(np.max(mag)),
                })
            flow_statistics["magnitudes"] = magnitudes

            # Create optical flow result
            optical_flow_result = OpticalFlowResult(
                flow_consistency_score=flow_consistency_score,
                boundary_artifact_score=boundary_score,
                temporal_coherence_score=temporal_score,
            )

            execution_time = time.time() - start
            self._total_execution_time += execution_time

            result_data = {
                "optical_flow_result": optical_flow_result,
                "anomalies": anomalies,
                "flow_statistics": flow_statistics,
                "overall_score": optical_flow_result.overall_score,
            }

            if compute_full_flow:
                result_data["flow_maps"] = flow_maps

            return ToolResult.success_result(
                data=result_data,
                metadata={
                    "execution_time": round(execution_time, 3),
                    "frames_analyzed": len(frames),
                    "flow_pairs_computed": len(flow_maps),
                    "anomalies_found": len(anomalies),
                },
            )

        except Exception as e:
            execution_time = time.time() - start
            self._total_execution_time += execution_time
            return ToolResult.error_result(
                f"Error in optical flow analysis: {str(e)}",
                metadata={"execution_time": round(execution_time, 3)},
            )

    def _compute_flow(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> np.ndarray:
        """
        Compute optical flow between two frames.

        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)

        Returns:
            Optical flow array of shape (H, W, 2)
        """
        if self.config.algorithm == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                frame1,
                frame2,
                None,
                pyr_scale=self.config.pyr_scale,
                levels=self.config.levels,
                winsize=self.config.winsize,
                iterations=self.config.iterations,
                poly_n=self.config.poly_n,
                poly_sigma=self.config.poly_sigma,
                flags=0,
            )
        else:
            # Lucas-Kanade (sparse) - convert to dense representation
            # For simplicity, use Farneback as default
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

        return flow

    def _analyze_flow_consistency(self, flow_maps: List[np.ndarray]) -> float:
        """
        Analyze consistency of optical flow across frames.

        Args:
            flow_maps: List of optical flow arrays

        Returns:
            Consistency score (0-1, higher = more consistent)
        """
        if len(flow_maps) < 2:
            return 0.5

        consistency_scores = []

        for i in range(len(flow_maps) - 1):
            flow1 = flow_maps[i]
            flow2 = flow_maps[i + 1]

            # Calculate flow change
            flow_diff = flow2 - flow1
            diff_magnitude = np.sqrt(
                flow_diff[:, :, 0] ** 2 + flow_diff[:, :, 1] ** 2
            )

            # Calculate original flow magnitude
            flow1_magnitude = np.sqrt(
                flow1[:, :, 0] ** 2 + flow1[:, :, 1] ** 2
            )

            # Relative change
            mean_flow = np.mean(flow1_magnitude)
            if mean_flow > 0:
                relative_change = np.mean(diff_magnitude) / mean_flow
                # Lower change = more consistent
                score = max(0, 1.0 - relative_change)
            else:
                score = 0.5

            consistency_scores.append(score)

        return np.mean(consistency_scores)

    def _analyze_boundary_flow(
        self,
        flow_maps: List[np.ndarray],
        face_regions: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, int],
    ) -> float:
        """
        Analyze optical flow at face boundaries.

        Args:
            flow_maps: List of optical flow arrays
            face_regions: Face bounding boxes
            frame_shape: Frame dimensions (height, width)

        Returns:
            Score indicating natural boundary flow (0-1)
        """
        if not face_regions or not flow_maps:
            return 0.5

        boundary_scores = []

        for flow in flow_maps:
            for x, y, w, h in face_regions:
                # Extract boundary regions
                boundary_width = max(3, min(w, h) // 10)

                # Get flow inside and outside face boundary
                inside_flows = []
                outside_flows = []

                # Left boundary
                x1_in = max(0, x)
                x2_in = min(frame_shape[1], x + boundary_width)
                x1_out = max(0, x - boundary_width)
                x2_out = max(0, x)

                y1, y2 = max(0, y), min(frame_shape[0], y + h)

                if x2_in > x1_in and y2 > y1:
                    inside_flow = flow[y1:y2, x1_in:x2_in]
                    inside_flows.append(inside_flow)

                if x2_out > x1_out and y2 > y1:
                    outside_flow = flow[y1:y2, x1_out:x2_out]
                    outside_flows.append(outside_flow)

                # Calculate flow discontinuity
                if inside_flows and outside_flows:
                    inside_mean = np.mean([np.mean(f) for f in inside_flows])
                    outside_mean = np.mean([np.mean(f) for f in outside_flows])

                    # Large difference indicates boundary artifact
                    diff = abs(inside_mean - outside_mean)
                    score = max(0, 1.0 - diff / 5.0)
                    boundary_scores.append(score)

        return np.mean(boundary_scores) if boundary_scores else 0.5

    def _estimate_boundary_artifacts(
        self, flow_maps: List[np.ndarray]
    ) -> float:
        """
        Estimate boundary artifacts without explicit face regions.

        Looks for sudden flow discontinuities that might indicate
        manipulation boundaries.

        Args:
            flow_maps: List of optical flow arrays

        Returns:
            Score indicating natural flow (0-1)
        """
        artifact_scores = []

        for flow in flow_maps:
            # Calculate flow gradients
            flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

            # Compute gradients
            grad_x = np.abs(np.diff(flow_magnitude, axis=1))
            grad_y = np.abs(np.diff(flow_magnitude, axis=0))

            # Find strong gradient regions (potential boundaries)
            threshold = np.percentile(flow_magnitude, 90)

            # Count sharp transitions
            sharp_x = np.sum(grad_x > threshold)
            sharp_y = np.sum(grad_y > threshold)

            total_pixels = flow_magnitude.size
            sharp_ratio = (sharp_x + sharp_y) / total_pixels

            # Many sharp transitions suggest boundary artifacts
            score = max(0, 1.0 - sharp_ratio * 100)
            artifact_scores.append(score)

        return np.mean(artifact_scores) if artifact_scores else 0.5

    def _analyze_temporal_coherence(
        self, flow_maps: List[np.ndarray]
    ) -> float:
        """
        Analyze temporal coherence of optical flow.

        Natural videos have smoothly varying flow over time.
        Sudden changes indicate potential manipulation.

        Args:
            flow_maps: List of optical flow arrays

        Returns:
            Temporal coherence score (0-1)
        """
        if len(flow_maps) < 3:
            return 0.5

        coherence_scores = []

        for i in range(1, len(flow_maps) - 1):
            prev_flow = flow_maps[i - 1]
            curr_flow = flow_maps[i]
            next_flow = flow_maps[i + 1]

            # Predict current from neighbors (simple linear interpolation)
            predicted = (prev_flow + next_flow) / 2

            # Calculate prediction error
            error = curr_flow - predicted
            error_magnitude = np.sqrt(error[:, :, 0] ** 2 + error[:, :, 1] ** 2)

            # Calculate flow magnitude for normalization
            curr_magnitude = np.sqrt(
                curr_flow[:, :, 0] ** 2 + curr_flow[:, :, 1] ** 2
            )
            mean_magnitude = np.mean(curr_magnitude)

            if mean_magnitude > 0:
                relative_error = np.mean(error_magnitude) / mean_magnitude
                score = max(0, 1.0 - relative_error)
            else:
                score = 0.5

            coherence_scores.append(score)

        return np.mean(coherence_scores)

    def visualize_flow(
        self, flow: np.ndarray, scale: float = 1.0
    ) -> np.ndarray:
        """
        Visualize optical flow as HSV image.

        Args:
            flow: Optical flow array
            scale: Scaling factor for visualization

        Returns:
            RGB visualization of the flow
        """
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = cv2.normalize(mag * scale, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool inputs."""
        return {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "description": "List of consecutive video frames (RGB)",
                    "items": {"type": "array"},
                    "minItems": 2,
                },
                "face_regions": {
                    "type": "array",
                    "description": "Optional face bounding boxes (x, y, w, h)",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "compute_full_flow": {
                    "type": "boolean",
                    "description": "Whether to return full flow maps",
                    "default": False,
                },
            },
            "required": ["frames"],
        }
