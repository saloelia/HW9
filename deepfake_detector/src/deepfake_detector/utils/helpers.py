"""
Helper functions for DeepFake Detector.

This module contains utility functions used across the detection system.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        format_str: Optional custom format string

    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=format_str,
        handlers=handlers,
    )

    return logging.getLogger("deepfake_detector")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path(__file__).parent.parent.parent.parent / "config" / "config.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path is None or not Path(config_path).exists():
        return get_default_config()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "agent": {
            "max_iterations": 10,
            "temperature": 0.3,
            "max_tokens": 4096,
            "verbose": True,
        },
        "video": {
            "max_frames": 30,
            "sample_rate": 5,
            "target_resolution": {"width": 640, "height": 480},
        },
        "detection": {
            "fake_threshold": 0.7,
            "min_evidence_count": 3,
        },
        "output": {
            "save_results": True,
            "output_dir": "results",
        },
    }


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get basic information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    import cv2

    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        info = {
            "path": str(path),
            "name": path.name,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "file_size": path.stat().st_size,
        }

        info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        info["file_size_mb"] = info["file_size"] / (1024 * 1024)

        return info

    finally:
        cap.release()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1:23" or "0:05")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def calculate_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of file hash
    """
    import hashlib

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def resize_frame(
    frame,
    target_width: int,
    target_height: int,
    keep_aspect: bool = True,
):
    """
    Resize a frame to target dimensions.

    Args:
        frame: Input frame (numpy array)
        target_width: Target width
        target_height: Target height
        keep_aspect: Whether to maintain aspect ratio

    Returns:
        Resized frame
    """
    import cv2

    if keep_aspect:
        h, w = frame.shape[:2]
        aspect = w / h
        target_aspect = target_width / target_height

        if aspect > target_aspect:
            new_w = target_width
            new_h = int(target_width / aspect)
        else:
            new_h = target_height
            new_w = int(target_height * aspect)

        return cv2.resize(frame, (new_w, new_h))
    else:
        return cv2.resize(frame, (target_width, target_height))
