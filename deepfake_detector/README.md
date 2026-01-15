# DeepFake Detector

An AI Agent-based DeepFake Video Detection System using LLM reasoning and computer vision tools.

## Overview

This project implements a sophisticated DeepFake detection system that combines:
- **AI Agent Architecture**: LLM-powered reasoning for analysis orchestration
- **Computer Vision Tools**: Specialized detection algorithms
- **Prompt Engineering**: Carefully crafted prompts for accurate detection

The system analyzes videos for signs of manipulation using multiple detection methods:
- Facial consistency analysis
- Temporal pattern analysis (blink rate, eye movements)
- Frequency domain analysis (GAN fingerprint detection)
- Optical flow analysis

## Features

- Multi-modal analysis combining visual, temporal, and frequency domain features
- AI agent with tool-use capabilities for intelligent analysis orchestration
- Detailed evidence collection and reasoning
- Support for multiple LLM providers (Anthropic Claude, OpenAI GPT-4)
- Comprehensive logging and result export
- Extensible tool architecture

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/student/deepfake-detector.git
cd deepfake-detector
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the package**
```bash
# Install with all dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Or install with notebook support
pip install -e ".[dev,notebook]"
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration

Edit `.env` file with your settings:

```bash
# LLM Provider (anthropic or openai)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key_here

# Detection settings
MAX_FRAMES_TO_ANALYZE=30
DETECTION_THRESHOLD=0.7
```

## Quick Start

### Basic Usage

```python
from deepfake_detector import DeepFakeDetector

# Initialize detector
detector = DeepFakeDetector()

# Analyze a video
result = detector.analyze("path/to/video.mp4")

# Check result
print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Evidence: {result.evidence}")
```

### Command Line Interface

```bash
# Analyze a single video
deepfake-detect path/to/video.mp4

# Analyze with verbose output
deepfake-detect path/to/video.mp4 --verbose

# Save results to specific directory
deepfake-detect path/to/video.mp4 --output results/
```

### Running Batch Experiments

Run analysis on all videos in the `data/samples/` directory:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run batch experiment on all sample videos
python demo.py --batch
```

**Directory Structure for Batch Experiments:**
```
data/samples/
├── real/           # Place authentic videos here
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── fake/           # Place deepfake videos here
    ├── deepfake1.mp4
    ├── deepfake2.mp4
    └── ...
```

The batch experiment will:
1. Automatically find all videos in `data/samples/real/` and `data/samples/fake/`
2. Analyze each video using the configured LLM
3. Compare predictions against ground truth (folder location)
4. Calculate accuracy metrics
5. Save detailed results to `results/batch_experiment_<timestamp>.json`

**Example Output:**
```
  Total Videos Analyzed: 10
  Overall Accuracy: 8/10 (80.0%)

  Real Video Detection:
    Correct: 4/5 (80.0%)

  Fake Video Detection:
    Correct: 4/5 (80.0%)

  Results saved to: results/batch_experiment_20240115_143012.json
```

### Using the Agent Directly

```python
from deepfake_detector.agent import DeepFakeDetectorAgent, AgentConfig

# Configure the agent
config = AgentConfig(
    llm_provider="anthropic",
    model="claude-3-sonnet-20240229",
    temperature=0.3,
)

# Create agent
agent = DeepFakeDetectorAgent(config=config)

# Run analysis
result = agent.analyze("video.mp4")
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
deepfake_detector/
├── src/deepfake_detector/
│   ├── agent/           # AI Agent implementation
│   │   ├── detector_agent.py   # Main agent class
│   │   └── prompts.py          # Prompt templates
│   ├── tools/           # Detection tools
│   │   ├── video_extractor.py  # Frame extraction
│   │   ├── face_analyzer.py    # Face detection & analysis
│   │   ├── temporal_analyzer.py # Temporal patterns
│   │   ├── frequency_analyzer.py # FFT analysis
│   │   └── optical_flow_analyzer.py # Motion analysis
│   ├── core/            # Core components
│   │   ├── detector.py         # High-level API
│   │   └── models.py           # Data models
│   └── utils/           # Utilities
├── tests/               # Unit tests
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks
└── config/              # Configuration files
```

### Detection Methods

1. **Face Analysis**
   - Face detection and tracking
   - Boundary artifact detection
   - Skin tone consistency analysis
   - Edge artifact detection

2. **Temporal Analysis**
   - Blink rate detection (normal: 15-20/min)
   - Eye movement pattern analysis
   - Facial movement consistency
   - Head pose tracking

3. **Frequency Analysis**
   - FFT-based spectral analysis
   - GAN fingerprint detection
   - Noise pattern analysis
   - Compression artifact detection

4. **Optical Flow Analysis**
   - Flow consistency measurement
   - Boundary discontinuity detection
   - Temporal coherence analysis

## API Reference

### DeepFakeDetector

Main class for video analysis.

```python
class DeepFakeDetector:
    def __init__(
        self,
        llm_provider: str = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = True,
    )

    def analyze(
        self,
        video_path: str,
        save_results: bool = True,
    ) -> AnalysisResult
```

### AnalysisResult

Result of video analysis.

```python
class AnalysisResult:
    verdict: DetectionVerdict  # REAL, FAKE, UNCERTAIN, ERROR
    confidence: float         # 0.0 to 1.0
    evidence: List[str]       # Supporting evidence
    reasoning: str            # LLM reasoning
    processing_time: float    # Seconds
```

## Configuration Guide

### config/config.yaml

```yaml
agent:
  max_iterations: 10
  temperature: 0.3
  max_tokens: 4096

video:
  max_frames: 30
  sample_rate: 5

detection:
  fake_threshold: 0.7
  weights:
    face_consistency: 0.25
    temporal_patterns: 0.25
    frequency_analysis: 0.30
    optical_flow: 0.20
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=deepfake_detector --cov-report=html

# Run specific test file
pytest tests/test_agent.py -v
```

## Troubleshooting

### Common Issues

1. **No API key found**
   - Ensure `.env` file exists with valid API key
   - Check environment variable is set correctly

2. **Video format not supported**
   - Supported formats: mp4, avi, mov, mkv, webm
   - Convert video using ffmpeg if needed

3. **Low detection confidence**
   - Try analyzing more frames (`MAX_FRAMES_TO_ANALYZE`)
   - Ensure video quality is sufficient

4. **Import errors**
   - Ensure all dependencies installed: `pip install -e ".[dev]"`
   - Check Python version (3.9+ required)

## Performance Considerations

- **Memory**: Each frame is loaded into memory. For long videos, increase `FRAME_SAMPLE_RATE`
- **Processing Time**: Typical analysis takes 30-60 seconds per video
- **API Costs**: LLM reasoning uses approximately 2000-4000 tokens per analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- Anthropic/OpenAI for LLM reasoning
- Research on DeepFake detection methods

## Contact

For questions or issues, please open a GitHub issue.
