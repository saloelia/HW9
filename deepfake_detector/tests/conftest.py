"""
Pytest configuration and fixtures for DeepFake Detector tests.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def sample_frame():
    """Create a sample RGB frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames():
    """Create a list of sample RGB frames."""
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)]


@pytest.fixture
def sample_gray_frame():
    """Create a sample grayscale frame."""
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)


@pytest.fixture
def mock_video_capture():
    """Create a mock cv2.VideoCapture object."""
    mock = Mock()
    mock.isOpened.return_value = True
    mock.get.side_effect = lambda prop: {
        0: 640,   # CAP_PROP_POS_MSEC
        1: 0,     # CAP_PROP_POS_FRAMES
        3: 640,   # CAP_PROP_FRAME_WIDTH
        4: 480,   # CAP_PROP_FRAME_HEIGHT
        5: 30.0,  # CAP_PROP_FPS
        6: 0,     # CAP_PROP_FOURCC
        7: 300,   # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return mock


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a temporary video file path (not a real video)."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy")
    return video_path


@pytest.fixture
def sample_face_bbox():
    """Create a sample face bounding box."""
    return (100, 100, 200, 200)  # x, y, width, height


@pytest.fixture
def sample_landmarks():
    """Create sample facial landmarks (68 points)."""
    return [(i * 5, i * 5) for i in range(68)]


@pytest.fixture
def config_dict():
    """Create a sample configuration dictionary."""
    return {
        "agent": {
            "max_iterations": 10,
            "temperature": 0.3,
            "max_tokens": 4096,
        },
        "video": {
            "max_frames": 30,
            "sample_rate": 5,
        },
        "detection": {
            "fake_threshold": 0.7,
        },
    }
