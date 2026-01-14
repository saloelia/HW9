"""
Main DeepFake Detector class.

This module provides a high-level interface for DeepFake detection,
wrapping the agent and tools into an easy-to-use API.

Example:
    >>> from deepfake_detector import DeepFakeDetector
    >>> detector = DeepFakeDetector()
    >>> result = detector.analyze("video.mp4")
    >>> print(result.verdict)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from deepfake_detector.agent.detector_agent import (
    DeepFakeDetectorAgent,
    AgentConfig,
)
from deepfake_detector.core.models import AnalysisResult


logger = logging.getLogger(__name__)


class DeepFakeDetector:
    """
    High-level interface for DeepFake video detection.

    This class provides a simple API for analyzing videos and detecting
    DeepFake manipulation. It wraps the AI agent and handles configuration.

    Example:
        >>> detector = DeepFakeDetector()
        >>> result = detector.analyze("path/to/video.mp4")
        >>> if result.verdict == DetectionVerdict.FAKE:
        ...     print(f"DeepFake detected with {result.confidence:.0%} confidence")
        ...     print(f"Evidence: {result.evidence}")

    Attributes:
        agent: The underlying detection agent
        config: Detector configuration
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the DeepFake detector.

        Args:
            llm_provider: LLM provider to use ("anthropic" or "openai")
            model: Model identifier (uses default if None)
            api_key: API key for LLM (uses env var if None)
            verbose: Enable verbose logging
        """
        # Set up logging
        log_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Default models per provider
        default_models = {
            "anthropic": "claude-3-sonnet-20240229",
            "openai": "gpt-4-turbo",
        }

        config = AgentConfig(
            llm_provider=llm_provider,
            model=model or default_models.get(llm_provider, ""),
            verbose=verbose,
        )

        self.agent = DeepFakeDetectorAgent(config=config, api_key=api_key)
        self.config = config

        logger.info(
            f"DeepFake Detector initialized with {llm_provider} "
            f"({config.model})"
        )

    def analyze(
        self,
        video_path: str,
        save_results: bool = True,
    ) -> AnalysisResult:
        """
        Analyze a video for DeepFake manipulation.

        Args:
            video_path: Path to the video file to analyze
            save_results: Whether to save results to file

        Returns:
            AnalysisResult with verdict, confidence, and evidence

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is not supported
        """
        path = Path(video_path)

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        supported_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        if path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported: {supported_formats}"
            )

        return self.agent.analyze(
            video_path=str(path),
            save_results=save_results,
        )

    def analyze_batch(
        self,
        video_paths: list[str],
        save_results: bool = True,
    ) -> list[AnalysisResult]:
        """
        Analyze multiple videos.

        Args:
            video_paths: List of video file paths
            save_results: Whether to save results

        Returns:
            List of AnalysisResult objects
        """
        results = []
        for i, path in enumerate(video_paths):
            logger.info(f"Analyzing video {i+1}/{len(video_paths)}: {path}")
            try:
                result = self.analyze(path, save_results=save_results)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {path}: {e}")
        return results

    def get_version(self) -> str:
        """Get the detector version."""
        from deepfake_detector import __version__
        return __version__
