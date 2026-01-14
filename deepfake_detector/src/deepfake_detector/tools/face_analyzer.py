"""
Face analysis tool for DeepFake detection.

This module provides face detection and landmark analysis capabilities
for identifying potential DeepFake indicators in facial regions.

Building Block Design:
    Input Data:
        - frames: List of video frames to analyze
        - detect_landmarks: Whether to detect facial landmarks

    Output Data:
        - face_data: List of FaceData for each frame
        - consistency_scores: Face consistency metrics
        - anomalies: Detected facial anomalies

    Setup Data:
        - detection_confidence: Minimum confidence for face detection
        - min_face_size: Minimum face size to detect
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deepfake_detector.tools.base import BaseTool, ToolConfig, ToolResult
from deepfake_detector.core.models import FaceData, FrameAnalysis


class FaceAnalyzerConfig(ToolConfig):
    """Configuration for FaceAnalyzer tool."""

    detection_confidence: float = 0.8
    min_face_size: int = 64
    landmarks_enabled: bool = True
    face_padding: float = 0.2
    max_faces_per_frame: int = 5


class FaceAnalyzer(BaseTool):
    """
    Tool for detecting and analyzing faces in video frames.

    This tool uses OpenCV's DNN-based face detector to find faces
    and analyze facial features for DeepFake indicators.

    Detection Methods:
        1. Face boundary consistency
        2. Facial landmark alignment
        3. Skin tone uniformity
        4. Edge artifact detection

    Example:
        >>> analyzer = FaceAnalyzer()
        >>> result = analyzer.execute(frames=frame_list)
        >>> if result.success:
        ...     for frame_analysis in result.data["frame_analyses"]:
        ...         print(f"Detected {len(frame_analysis.faces)} faces")
    """

    def __init__(self, config: Optional[FaceAnalyzerConfig] = None) -> None:
        """
        Initialize the FaceAnalyzer.

        Args:
            config: Tool configuration (Setup Data)
        """
        super().__init__(
            name="face_analyzer",
            description=(
                "Detect and analyze faces in video frames. "
                "Returns face locations, landmarks, and anomaly indicators. "
                "Can identify boundary artifacts and inconsistencies."
            ),
            config=config or FaceAnalyzerConfig(),
        )
        self.config: FaceAnalyzerConfig = config or FaceAnalyzerConfig()
        self._face_cascade = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize the face detection model."""
        # Use Haar Cascade as fallback (works without deep learning models)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def execute(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[List[float]] = None,
        detect_landmarks: bool = True,
        analyze_consistency: bool = True,
    ) -> ToolResult[Dict[str, Any]]:
        """
        Analyze faces in a list of frames.

        Args:
            frames: List of frame arrays (RGB format) (Input Data)
            timestamps: Optional list of frame timestamps
            detect_landmarks: Whether to detect facial landmarks
            analyze_consistency: Whether to analyze face consistency

        Returns:
            ToolResult containing:
                - frame_analyses: List of FrameAnalysis objects
                - consistency_score: Overall face consistency score
                - anomalies: List of detected anomalies
        """
        start = time.time()
        self._execution_count += 1

        try:
            if not frames:
                return ToolResult.error_result("No frames provided")

            timestamps = timestamps or [i * 0.1 for i in range(len(frames))]
            frame_analyses: List[FrameAnalysis] = []
            all_faces: List[List[FaceData]] = []

            # Process each frame
            for idx, frame in enumerate(frames):
                faces = self._detect_faces(frame)
                all_faces.append(faces)

                # Calculate frame quality metrics
                quality_score = self._calculate_quality(frame)
                blur_score = self._calculate_blur(frame)

                # Detect anomalies in faces
                anomalies = {}
                for face in faces:
                    face_anomalies = self._analyze_face_anomalies(frame, face)
                    for key, value in face_anomalies.items():
                        if key not in anomalies:
                            anomalies[key] = []
                        anomalies[key].append(value)

                # Average anomaly scores
                anomaly_indicators = {
                    k: sum(v) / len(v) for k, v in anomalies.items() if v
                }

                frame_analysis = FrameAnalysis(
                    frame_number=idx,
                    timestamp=timestamps[idx],
                    faces=faces,
                    quality_score=quality_score,
                    blur_score=blur_score,
                    anomaly_indicators=anomaly_indicators,
                )
                frame_analyses.append(frame_analysis)

            # Analyze temporal consistency of faces
            consistency_score = 0.5
            detected_anomalies: List[str] = []

            if analyze_consistency and len(all_faces) > 1:
                consistency_score = self._analyze_temporal_consistency(all_faces)

                if consistency_score < 0.4:
                    detected_anomalies.append(
                        "Low face consistency across frames - possible manipulation"
                    )

            # Check for boundary artifacts
            boundary_score = self._check_boundary_artifacts(frames, all_faces)
            if boundary_score < 0.5:
                detected_anomalies.append(
                    "Boundary artifacts detected around face regions"
                )

            execution_time = time.time() - start
            self._total_execution_time += execution_time

            total_faces = sum(len(fa.faces) for fa in frame_analyses)

            return ToolResult.success_result(
                data={
                    "frame_analyses": frame_analyses,
                    "consistency_score": consistency_score,
                    "boundary_score": boundary_score,
                    "anomalies": detected_anomalies,
                    "total_faces_detected": total_faces,
                },
                metadata={
                    "execution_time": round(execution_time, 3),
                    "frames_processed": len(frames),
                    "faces_detected": total_faces,
                },
            )

        except Exception as e:
            execution_time = time.time() - start
            self._total_execution_time += execution_time
            return ToolResult.error_result(
                f"Error analyzing faces: {str(e)}",
                metadata={"execution_time": round(execution_time, 3)},
            )

    def _detect_faces(self, frame: np.ndarray) -> List[FaceData]:
        """
        Detect faces in a single frame.

        Args:
            frame: Frame array in RGB format

        Returns:
            List of FaceData objects
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.config.min_face_size, self.config.min_face_size),
        )

        face_data_list: List[FaceData] = []

        for i, (x, y, w, h) in enumerate(faces[: self.config.max_faces_per_frame]):
            # Calculate confidence based on face size and aspect ratio
            aspect_ratio = w / h
            size_score = min(w * h / (frame.shape[0] * frame.shape[1] * 0.1), 1.0)
            aspect_score = 1.0 - abs(aspect_ratio - 1.0) * 0.5
            confidence = (size_score + aspect_score) / 2

            if confidence >= self.config.detection_confidence:
                # Generate pseudo-landmarks (simplified)
                landmarks = self._generate_landmarks(x, y, w, h) if self.config.landmarks_enabled else None

                face_data = FaceData(
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=float(confidence),
                    landmarks=landmarks,
                    face_id=i,
                )
                face_data_list.append(face_data)

        return face_data_list

    def _generate_landmarks(
        self, x: int, y: int, w: int, h: int
    ) -> List[Tuple[float, float]]:
        """
        Generate approximate facial landmarks based on face bounding box.

        This is a simplified landmark estimation. In production, use
        a proper landmark detection model like dlib or MediaPipe.

        Args:
            x, y, w, h: Face bounding box coordinates

        Returns:
            List of 68 facial landmark coordinates
        """
        landmarks = []

        # Generate approximate landmark positions
        # Jaw line (17 points)
        for i in range(17):
            lx = x + (w * i / 16)
            ly = y + h * (0.5 + 0.3 * np.sin(np.pi * i / 16))
            landmarks.append((lx, ly))

        # Eyebrows (10 points)
        for i in range(5):
            landmarks.append((x + w * (0.2 + 0.15 * i), y + h * 0.25))
        for i in range(5):
            landmarks.append((x + w * (0.55 + 0.15 * i), y + h * 0.25))

        # Nose (9 points)
        for i in range(9):
            landmarks.append((x + w * 0.5, y + h * (0.3 + 0.05 * i)))

        # Eyes (12 points)
        for i in range(6):
            landmarks.append((x + w * (0.25 + 0.05 * i), y + h * 0.35))
        for i in range(6):
            landmarks.append((x + w * (0.6 + 0.05 * i), y + h * 0.35))

        # Mouth (20 points)
        for i in range(20):
            angle = 2 * np.pi * i / 20
            lx = x + w * (0.5 + 0.1 * np.cos(angle))
            ly = y + h * (0.7 + 0.05 * np.sin(angle))
            landmarks.append((lx, ly))

        return landmarks[:68]  # Ensure exactly 68 landmarks

    def _calculate_quality(self, frame: np.ndarray) -> float:
        """Calculate overall frame quality score."""
        # Check brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2

        # Check contrast
        contrast = np.std(gray) / 128.0
        contrast_score = min(contrast, 1.0)

        return (brightness_score + contrast_score) / 2

    def _calculate_blur(self, frame: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize (higher variance = less blur)
        # Typical sharp images have variance > 100
        blur_score = 1.0 - min(laplacian_var / 500.0, 1.0)
        return blur_score

    def _analyze_face_anomalies(
        self, frame: np.ndarray, face: FaceData
    ) -> Dict[str, float]:
        """
        Analyze potential anomalies in a detected face.

        Args:
            frame: The frame containing the face
            face: FaceData object

        Returns:
            Dictionary of anomaly scores (higher = more suspicious)
        """
        anomalies = {}
        x, y, w, h = face.bbox

        # Extract face region with padding
        pad = int(min(w, h) * self.config.face_padding)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            return anomalies

        # Check color consistency
        anomalies["color_inconsistency"] = self._check_color_consistency(face_region)

        # Check edge artifacts
        anomalies["edge_artifacts"] = self._check_edge_artifacts(face_region)

        # Check texture uniformity
        anomalies["texture_anomaly"] = self._check_texture_uniformity(face_region)

        return anomalies

    def _check_color_consistency(self, face_region: np.ndarray) -> float:
        """Check for unnatural color transitions in face region."""
        # Convert to LAB color space
        lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)

        # Calculate color variance
        l_var = np.var(lab[:, :, 0])
        a_var = np.var(lab[:, :, 1])
        b_var = np.var(lab[:, :, 2])

        # High variance in chrominance might indicate manipulation
        chroma_var = (a_var + b_var) / 2
        anomaly_score = min(chroma_var / 500.0, 1.0)

        return anomaly_score

    def _check_edge_artifacts(self, face_region: np.ndarray) -> float:
        """Check for unnatural edges around face boundary."""
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)

        # Check edge density at boundaries
        h, w = edges.shape
        border_size = max(5, min(h, w) // 10)

        border_edge_density = (
            np.mean(edges[:border_size, :])
            + np.mean(edges[-border_size:, :])
            + np.mean(edges[:, :border_size])
            + np.mean(edges[:, -border_size:])
        ) / (4 * 255)

        center_edge_density = np.mean(
            edges[border_size:-border_size, border_size:-border_size]
        ) / 255

        # High border edge density relative to center is suspicious
        if center_edge_density > 0:
            ratio = border_edge_density / center_edge_density
            anomaly_score = min(ratio / 3.0, 1.0)
        else:
            anomaly_score = border_edge_density

        return anomaly_score

    def _check_texture_uniformity(self, face_region: np.ndarray) -> float:
        """Check for unnatural texture patterns."""
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)

        # Calculate local binary pattern-like feature
        # High uniformity in texture might indicate synthetic generation
        texture_var = np.var(gray)

        # Very low or very high variance is suspicious
        optimal_var = 1000
        deviation = abs(texture_var - optimal_var) / optimal_var
        anomaly_score = min(deviation, 1.0)

        return anomaly_score

    def _analyze_temporal_consistency(
        self, all_faces: List[List[FaceData]]
    ) -> float:
        """
        Analyze face consistency across frames.

        Args:
            all_faces: List of face lists for each frame

        Returns:
            Consistency score (higher = more consistent = likely real)
        """
        if len(all_faces) < 2:
            return 0.5

        # Track face positions across frames
        positions = []
        sizes = []

        for faces in all_faces:
            if faces:
                # Use the most confident face
                main_face = max(faces, key=lambda f: f.confidence)
                positions.append(main_face.center)
                sizes.append(main_face.area)

        if len(positions) < 2:
            return 0.5

        # Calculate position consistency
        position_changes = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            change = np.sqrt(dx ** 2 + dy ** 2)
            position_changes.append(change)

        # Smooth movement is expected, jittery movement is suspicious
        position_std = np.std(position_changes) if position_changes else 0
        position_score = max(0, 1.0 - position_std / 50.0)

        # Calculate size consistency
        size_std = np.std(sizes) if sizes else 0
        avg_size = np.mean(sizes) if sizes else 1
        size_cv = size_std / avg_size if avg_size > 0 else 0
        size_score = max(0, 1.0 - size_cv * 2)

        return (position_score + size_score) / 2

    def _check_boundary_artifacts(
        self,
        frames: List[np.ndarray],
        all_faces: List[List[FaceData]],
    ) -> float:
        """
        Check for boundary artifacts around faces across frames.

        Args:
            frames: List of video frames
            all_faces: Detected faces for each frame

        Returns:
            Score (higher = less artifacts = likely real)
        """
        artifact_scores = []

        for frame, faces in zip(frames, all_faces):
            for face in faces:
                x, y, w, h = face.bbox

                # Extract boundary region
                boundary_width = max(3, w // 20)

                # Get regions just inside and outside face boundary
                inner_left = frame[
                    max(0, y) : min(frame.shape[0], y + h),
                    max(0, x) : max(0, x + boundary_width),
                ]
                outer_left = frame[
                    max(0, y) : min(frame.shape[0], y + h),
                    max(0, x - boundary_width) : max(0, x),
                ]

                if inner_left.size > 0 and outer_left.size > 0:
                    # Calculate color difference at boundary
                    inner_mean = np.mean(inner_left, axis=(0, 1))
                    outer_mean = np.mean(outer_left, axis=(0, 1))
                    diff = np.linalg.norm(inner_mean - outer_mean)

                    # Normalize
                    score = max(0, 1.0 - diff / 100.0)
                    artifact_scores.append(score)

        if artifact_scores:
            return np.mean(artifact_scores)
        return 0.5

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool inputs."""
        return {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "description": "List of video frames to analyze (RGB format)",
                    "items": {"type": "array"},
                },
                "timestamps": {
                    "type": "array",
                    "description": "Optional timestamps for each frame",
                    "items": {"type": "number"},
                },
                "detect_landmarks": {
                    "type": "boolean",
                    "description": "Whether to detect facial landmarks",
                    "default": True,
                },
                "analyze_consistency": {
                    "type": "boolean",
                    "description": "Whether to analyze face consistency across frames",
                    "default": True,
                },
            },
            "required": ["frames"],
        }
