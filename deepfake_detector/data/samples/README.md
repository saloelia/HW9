# Sample Data Directory

This directory is for placing sample videos for testing the DeepFake detection system.

## Supported Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)

## Recommended Test Videos

For testing, you can use videos from:

1. **FaceForensics++ Dataset**: https://github.com/ondyari/FaceForensics
2. **Celeb-DF Dataset**: https://github.com/yuezunli/celeb-deepfakeforensics
3. **DFDC (DeepFake Detection Challenge)**: https://dfdc.ai/

## Directory Structure

```
data/
  samples/
    real/       # Place authentic videos here
    fake/       # Place deepfake videos here
```

## Usage

```python
from deepfake_detector import DeepFakeDetector

detector = DeepFakeDetector()
result = detector.analyze("data/samples/test_video.mp4")
print(result.verdict)
```
