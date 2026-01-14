"""
Video extraction tool for DeepFake detection.

This module provides functionality to extract frames from video files
for subsequent analysis by other tools.

Building Block Design:
    Input Data:
        - video_path: Path to the video file
        - max_frames: Maximum number of frames to extract
        - sample_rate: Frame sampling rate

    Output Data:
        - frames: List of extracted frames as numpy arrays
        - metadata: Video metadata (duration, fps, resolution)

    Setup Data:
        - target_resolution: Resolution to resize frames to
        - supported_formats: List of supported video formats
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deepfake_detector.tools.base import BaseTool, ToolConfig, ToolResult
from deepfake_detector.core.models import VideoMetadata


class VideoExtractorConfig(ToolConfig):
    """Configuration for VideoExtractor tool."""

    target_resolution: Optional[Tuple[int, int]] = (640, 480)
    supported_formats: List[str] = ["mp4", "avi", "mov", "mkv", "webm"]
    max_frames: int = 30
    sample_rate: int = 5


class VideoExtractor(BaseTool):
    """
    Tool for extracting frames from video files.

    This tool reads video files and extracts frames at specified intervals
    for analysis by other detection tools.

    Example:
        >>> extractor = VideoExtractor()
        >>> result = extractor.execute(
        ...     video_path="video.mp4",
        ...     max_frames=20
        ... )
        >>> if result.success:
        ...     frames = result.data["frames"]
        ...     print(f"Extracted {len(frames)} frames")
    """

    def __init__(self, config: Optional[VideoExtractorConfig] = None) -> None:
        """
        Initialize the VideoExtractor.

        Args:
            config: Tool configuration (Setup Data)
        """
        super().__init__(
            name="video_extractor",
            description=(
                "Extract frames from a video file for analysis. "
                "Returns frames as images and video metadata including "
                "duration, fps, and resolution."
            ),
            config=config or VideoExtractorConfig(),
        )
        self.config: VideoExtractorConfig = config or VideoExtractorConfig()

    def execute(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        sample_rate: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> ToolResult[Dict[str, Any]]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file (Input Data)
            max_frames: Maximum frames to extract (optional)
            sample_rate: Extract every Nth frame (optional)
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            ToolResult containing:
                - frames: List of frame arrays
                - metadata: VideoMetadata object
                - frame_indices: List of original frame indices
        """
        start = time.time()
        self._execution_count += 1

        try:
            path = Path(video_path)

            # Validate file exists
            if not path.exists():
                return ToolResult.error_result(f"Video file not found: {video_path}")

            # Validate format
            ext = path.suffix.lower().lstrip(".")
            if ext not in self.config.supported_formats:
                return ToolResult.error_result(
                    f"Unsupported format: {ext}. "
                    f"Supported: {self.config.supported_formats}"
                )

            # Open video
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return ToolResult.error_result(f"Failed to open video: {video_path}")

            try:
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0

                # Get codec
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

                # Create metadata
                metadata = VideoMetadata(
                    file_path=path,
                    duration=duration,
                    fps=fps,
                    resolution=(width, height),
                    codec=codec,
                    file_size=path.stat().st_size,
                )

                # Calculate frame extraction parameters
                max_f = max_frames or self.config.max_frames
                rate = sample_rate or self.config.sample_rate

                # Calculate frame range
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps) if end_time else total_frames

                # Calculate which frames to extract
                frame_indices = list(range(start_frame, end_frame, rate))[:max_f]

                # Extract frames
                frames = []
                extracted_indices = []

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()

                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Resize if configured
                        if self.config.target_resolution:
                            frame_rgb = cv2.resize(
                                frame_rgb,
                                self.config.target_resolution,
                                interpolation=cv2.INTER_AREA,
                            )

                        frames.append(frame_rgb)
                        extracted_indices.append(idx)

            finally:
                cap.release()

            execution_time = time.time() - start
            self._total_execution_time += execution_time

            return ToolResult.success_result(
                data={
                    "frames": frames,
                    "metadata": metadata,
                    "frame_indices": extracted_indices,
                    "timestamps": [idx / fps for idx in extracted_indices],
                },
                metadata={
                    "execution_time": round(execution_time, 3),
                    "frames_extracted": len(frames),
                    "total_video_frames": total_frames,
                },
            )

        except Exception as e:
            execution_time = time.time() - start
            self._total_execution_time += execution_time
            return ToolResult.error_result(
                f"Error extracting frames: {str(e)}",
                metadata={"execution_time": round(execution_time, 3)},
            )

    def get_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool inputs.

        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Path to the video file to analyze",
                },
                "max_frames": {
                    "type": "integer",
                    "description": "Maximum number of frames to extract",
                    "default": self.config.max_frames,
                    "minimum": 1,
                    "maximum": 100,
                },
                "sample_rate": {
                    "type": "integer",
                    "description": "Extract every Nth frame",
                    "default": self.config.sample_rate,
                    "minimum": 1,
                },
                "start_time": {
                    "type": "number",
                    "description": "Start time in seconds",
                    "default": 0.0,
                    "minimum": 0.0,
                },
                "end_time": {
                    "type": "number",
                    "description": "End time in seconds (null for end of video)",
                    "default": None,
                },
            },
            "required": ["video_path"],
        }

    def get_video_info(self, video_path: str) -> ToolResult[VideoMetadata]:
        """
        Get video metadata without extracting frames.

        Args:
            video_path: Path to the video file

        Returns:
            ToolResult containing VideoMetadata
        """
        try:
            path = Path(video_path)
            if not path.exists():
                return ToolResult.error_result(f"Video not found: {video_path}")

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return ToolResult.error_result(f"Cannot open video: {video_path}")

            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0

                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

                metadata = VideoMetadata(
                    file_path=path,
                    duration=duration,
                    fps=fps,
                    resolution=(width, height),
                    codec=codec,
                    file_size=path.stat().st_size,
                )

                return ToolResult.success_result(metadata)

            finally:
                cap.release()

        except Exception as e:
            return ToolResult.error_result(f"Error getting video info: {str(e)}")
