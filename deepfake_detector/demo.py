#!/usr/bin/env python3
"""
DeepFake Detection System - Demo Script

This script demonstrates the capabilities of the DeepFake detection system.
It can be run with or without actual video files to showcase the system architecture.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepfake_detector.core.models import (
    DetectionVerdict,
    VideoMetadata,
    FaceData,
    FrameAnalysis,
    TemporalAnalysisResult,
    FrequencyAnalysisResult,
    OpticalFlowResult,
    AnalysisResult,
    AgentState,
)
from deepfake_detector.tools.face_analyzer import FaceAnalyzer
from deepfake_detector.tools.temporal_analyzer import TemporalAnalyzer
from deepfake_detector.tools.frequency_analyzer import FrequencyAnalyzer
from deepfake_detector.tools.optical_flow_analyzer import OpticalFlowAnalyzer
from deepfake_detector.agent.detector_agent import DeepFakeDetectorAgent, AgentConfig


def print_header():
    """Print demo header."""
    print("\n" + "=" * 70)
    print("        DEEPFAKE DETECTION SYSTEM - DEMONSTRATION")
    print("=" * 70)
    print("\nThis system uses AI agents with specialized tools to detect")
    print("manipulated videos through multi-modal analysis.\n")


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}\n")


def demo_tools():
    """Demonstrate individual detection tools."""
    print_section("1. Detection Tools Overview")

    tools = [
        ("Video Extractor", "Extracts frames from video files for analysis"),
        ("Face Analyzer", "Detects faces, extracts landmarks, checks boundaries"),
        ("Temporal Analyzer", "Analyzes blink rates, eye movements, consistency"),
        ("Frequency Analyzer", "FFT-based GAN fingerprint detection"),
        ("Optical Flow Analyzer", "Motion analysis between consecutive frames"),
    ]

    for name, description in tools:
        print(f"  [{name}]")
        print(f"     {description}\n")


def demo_face_analysis():
    """Demonstrate face analysis on synthetic data."""
    print_section("2. Face Analysis Demo")

    # Create synthetic frames
    print("  Creating synthetic test frames...")
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
              for _ in range(5)]

    analyzer = FaceAnalyzer()
    print(f"  Tool: {analyzer.name}")
    print(f"  Detection confidence threshold: {analyzer.config.detection_confidence}")

    # Execute analysis
    print("\n  Executing face analysis...")
    result = analyzer.execute(frames=frames)

    if result.success:
        print(f"  ✓ Analysis completed successfully")
        print(f"  ✓ Analyzed {len(result.data['frame_analyses'])} frames")

        # Show sample results
        for i, fa in enumerate(result.data['frame_analyses'][:3]):
            face_count = len(fa.get('faces', []))
            print(f"    Frame {i+1}: {face_count} face(s) detected")
    else:
        print(f"  Note: {result.error}")


def demo_temporal_analysis():
    """Demonstrate temporal analysis."""
    print_section("3. Temporal Analysis Demo")

    # Create mock frame analyses
    print("  Creating mock temporal data...")
    analyses = []
    for i in range(30):
        face = FaceData(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            landmarks=[(j, j) for j in range(68)],
        )
        analysis = FrameAnalysis(
            frame_number=i,
            timestamp=i / 30.0,
            faces=[face],
        )
        analyses.append(analysis)

    analyzer = TemporalAnalyzer()
    print(f"  Tool: {analyzer.name}")
    print(f"  Normal blink rate range: {analyzer.config.normal_blink_rate_min}-{analyzer.config.normal_blink_rate_max} blinks/min")

    print("\n  Executing temporal analysis...")
    result = analyzer.execute(frame_analyses=analyses, fps=30.0)

    if result.success:
        print(f"  ✓ Analysis completed successfully")
        print(f"  ✓ Overall temporal score: {result.data['overall_score']:.3f}")

        temporal = result.data['temporal_result']
        print(f"    - Blink pattern score: {temporal.blink_pattern_score:.3f}")
        print(f"    - Eye movement score: {temporal.eye_movement_score:.3f}")
        print(f"    - Movement consistency: {temporal.facial_movement_consistency:.3f}")
    else:
        print(f"  Note: {result.error}")


def demo_frequency_analysis():
    """Demonstrate frequency domain analysis."""
    print_section("4. Frequency Analysis Demo (GAN Fingerprint Detection)")

    # Create test frames
    print("  Creating test frames for FFT analysis...")
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
              for _ in range(5)]

    analyzer = FrequencyAnalyzer()
    print(f"  Tool: {analyzer.name}")
    print(f"  FFT enabled: {analyzer.config.fft_enabled}")
    print(f"  DCT enabled: {analyzer.config.dct_enabled}")

    print("\n  Executing frequency analysis...")
    result = analyzer.execute(frames=frames)

    if result.success:
        print(f"  ✓ Analysis completed successfully")
        print(f"  ✓ Overall frequency score: {result.data['overall_score']:.3f}")

        freq = result.data['frequency_result']
        print(f"    - Spectral anomaly score: {freq.spectral_anomaly_score:.3f}")
        print(f"    - GAN fingerprint score: {freq.gan_fingerprint_score:.3f}")
        print(f"    - Noise pattern score: {freq.noise_pattern_score:.3f}")
    else:
        print(f"  Note: {result.error}")


def demo_optical_flow():
    """Demonstrate optical flow analysis."""
    print_section("5. Optical Flow Analysis Demo")

    # Create frames with simulated motion
    print("  Creating frames with simulated motion...")
    frames = []
    for i in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        x = 10 + i * 5
        frame[40:60, x:x+20] = 255
        frames.append(frame)

    analyzer = OpticalFlowAnalyzer()
    print(f"  Tool: {analyzer.name}")
    print(f"  Algorithm: {analyzer.config.algorithm}")

    print("\n  Executing optical flow analysis...")
    result = analyzer.execute(frames=frames)

    if result.success:
        print(f"  ✓ Analysis completed successfully")
        print(f"  ✓ Overall optical flow score: {result.data['overall_score']:.3f}")

        flow = result.data['optical_flow_result']
        print(f"    - Flow consistency: {flow.flow_consistency_score:.3f}")
        print(f"    - Boundary artifacts: {flow.boundary_artifact_score:.3f}")
        print(f"    - Temporal coherence: {flow.temporal_coherence_score:.3f}")
    else:
        print(f"  Note: {result.error}")


def demo_agent():
    """Demonstrate the AI agent architecture."""
    print_section("6. AI Agent Architecture Demo")

    print("  Initializing DeepFake Detector Agent...")
    config = AgentConfig(
        llm_provider="anthropic",
        temperature=0.3,
        max_iterations=10,
    )
    agent = DeepFakeDetectorAgent(config=config)

    print(f"  ✓ Agent initialized with {len(agent.tools)} tools:")
    for tool_name in agent.tools:
        print(f"    - {tool_name}")

    print(f"\n  Agent Configuration:")
    print(f"    - LLM Provider: {agent.config.llm_provider}")
    print(f"    - Model: {agent.config.model}")
    print(f"    - Temperature: {agent.config.temperature}")
    print(f"    - Max iterations: {agent.config.max_iterations}")

    print(f"\n  Tool Definitions (for LLM tool use):")
    definitions = agent.get_tool_definitions()
    for defn in definitions[:2]:
        print(f"    - {defn['name']}: {defn['description'][:60]}...")


def demo_detection_simulation():
    """Simulate a complete detection workflow."""
    print_section("7. Complete Detection Workflow Simulation")

    print("  Simulating detection on a video file...\n")

    # Simulate workflow steps
    steps = [
        ("Video Loading", "Loading video and extracting metadata"),
        ("Frame Extraction", "Extracting 30 frames at 1 fps sample rate"),
        ("Face Detection", "Detecting faces in extracted frames"),
        ("Temporal Analysis", "Analyzing blink rates and movements"),
        ("Frequency Analysis", "Computing FFT for GAN fingerprints"),
        ("Optical Flow", "Analyzing motion between frames"),
        ("LLM Reasoning", "Agent synthesizing findings"),
        ("Final Verdict", "Generating detection result"),
    ]

    for i, (step, description) in enumerate(steps, 1):
        print(f"  Step {i}: {step}")
        print(f"         {description}")
        time.sleep(0.3)
        print(f"         ✓ Complete\n")

    # Simulate final result
    print("\n  " + "─" * 50)
    print("  DETECTION RESULT (Simulated)")
    print("  " + "─" * 50)

    # Simulate scores
    scores = {
        'face_consistency': 0.35,
        'temporal_score': 0.28,
        'frequency_score': 0.32,
        'optical_flow_score': 0.40,
    }

    weighted_avg = (
        scores['face_consistency'] * 0.25 +
        scores['temporal_score'] * 0.30 +
        scores['frequency_score'] * 0.25 +
        scores['optical_flow_score'] * 0.20
    )

    print(f"\n  Individual Scores:")
    for metric, score in scores.items():
        indicator = "⚠️ " if score < 0.5 else "✓ "
        print(f"    {indicator}{metric.replace('_', ' ').title()}: {score:.3f}")

    print(f"\n  Weighted Average: {weighted_avg:.3f}")
    print(f"\n  ╔════════════════════════════════════════════╗")
    print(f"  ║  VERDICT: FAKE                             ║")
    print(f"  ║  Confidence: 87.5%                         ║")
    print(f"  ╚════════════════════════════════════════════╝")

    print("\n  Evidence:")
    print("    - Abnormally low blink rate detected (5 blinks/min)")
    print("    - GAN fingerprint patterns in frequency domain")
    print("    - Boundary artifacts around face region")
    print("    - Inconsistent motion patterns detected")


def analyze_video(video_path: str):
    """Analyze an actual video file."""
    print_section("Analyzing Video File")

    path = Path(video_path)
    if not path.exists():
        print(f"  Error: Video file not found: {video_path}")
        return

    print(f"  Video: {path.name}")
    print(f"  Path: {path.absolute()}\n")

    # Initialize agent
    agent = DeepFakeDetectorAgent()

    print("  Running analysis...")
    print("  (This may take a few moments)\n")

    result = agent.analyze(str(path))

    print("  " + "─" * 50)
    print("  ANALYSIS RESULT")
    print("  " + "─" * 50)
    print(f"\n  Verdict: {result.verdict.value}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"\n  Reasoning:")
    print(f"    {result.reasoning}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="DeepFake Detection System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    # Run full demonstration
  python demo.py --quick            # Run quick demo
  python demo.py --video video.mp4  # Analyze a specific video
        """
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to video file to analyze"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick demo (tools overview only)"
    )

    args = parser.parse_args()

    print_header()

    if args.video:
        # Analyze specific video
        analyze_video(args.video)
    elif args.quick:
        # Quick demo
        demo_tools()
        demo_agent()
    else:
        # Full demonstration
        demo_tools()
        demo_face_analysis()
        demo_temporal_analysis()
        demo_frequency_analysis()
        demo_optical_flow()
        demo_agent()
        demo_detection_simulation()

    print("\n" + "=" * 70)
    print("                   Demo Complete!")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - README.md for usage instructions")
    print("  - docs/PRD.md for product requirements")
    print("  - docs/architecture.md for system design")
    print("  - notebooks/analysis.ipynb for experiments")
    print()


if __name__ == "__main__":
    main()
