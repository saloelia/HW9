"""
Command-line interface for DeepFake Detector.

This module provides the CLI entry point for the detection system.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from .core.detector import DeepFakeDetector
from .core.models import DetectionVerdict


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="deepfake-detector",
        description="AI-powered DeepFake video detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single video
  deepfake-detector analyze video.mp4

  # Analyze with JSON output
  deepfake-detector analyze video.mp4 --output json

  # Analyze with custom threshold
  deepfake-detector analyze video.mp4 --threshold 0.6

  # Batch analyze multiple videos
  deepfake-detector batch ./videos/

  # Show system info
  deepfake-detector info
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a video file for DeepFake detection"
    )
    analyze_parser.add_argument(
        "video",
        type=str,
        help="Path to the video file to analyze"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    analyze_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Detection threshold (0.0-1.0, default: 0.7)"
    )
    analyze_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    analyze_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use rule-based detection without LLM"
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Analyze multiple videos in a directory"
    )
    batch_parser.add_argument(
        "directory",
        type=str,
        help="Directory containing video files"
    )
    batch_parser.add_argument(
        "--output", "-o",
        type=str,
        default="results.json",
        help="Output file for results (default: results.json)"
    )
    batch_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search directory recursively"
    )

    # Info command
    subparsers.add_parser(
        "info",
        help="Show system information and available tools"
    )

    # Version command
    subparsers.add_parser(
        "version",
        help="Show version information"
    )

    return parser


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle the analyze command."""
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Analyzing: {video_path}")
        print(f"Threshold: {args.threshold}")
        print(f"Using LLM: {not args.no_llm}")
        print()

    # Initialize detector
    detector = DeepFakeDetector(
        fake_threshold=args.threshold,
        use_llm=not args.no_llm,
    )

    # Run analysis
    try:
        result = detector.analyze(str(video_path))
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1

    # Output results
    if args.output == "json":
        output = {
            "video": str(video_path.absolute()),
            "verdict": result.verdict.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "analysis_id": result.analysis_id,
            "processing_time": result.processing_time,
        }
        print(json.dumps(output, indent=2))
    else:
        # Text output
        print("=" * 60)
        print("DEEPFAKE DETECTION RESULT")
        print("=" * 60)
        print(f"\nVideo: {video_path.name}")
        print(f"Path: {video_path.absolute()}")
        print()

        # Verdict with color indication
        verdict_symbol = {
            DetectionVerdict.REAL: "AUTHENTIC",
            DetectionVerdict.FAKE: "DEEPFAKE DETECTED",
            DetectionVerdict.UNCERTAIN: "UNCERTAIN",
            DetectionVerdict.ERROR: "ERROR",
        }

        print(f"Verdict: {verdict_symbol.get(result.verdict, result.verdict.value)}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"\nReasoning:")
        print(f"  {result.reasoning}")

        if result.evidence:
            print(f"\nEvidence:")
            for item in result.evidence:
                print(f"  - {item}")

        print(f"\nProcessing Time: {result.processing_time:.2f}s")
        print(f"Analysis ID: {result.analysis_id}")

    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Handle the batch command."""
    directory = Path(args.directory)

    if not directory.exists():
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        return 1

    # Find video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if args.recursive:
        videos = [
            p for p in directory.rglob("*")
            if p.suffix.lower() in video_extensions
        ]
    else:
        videos = [
            p for p in directory.iterdir()
            if p.suffix.lower() in video_extensions
        ]

    if not videos:
        print("No video files found in directory.")
        return 0

    print(f"Found {len(videos)} video(s) to analyze...")
    print()

    # Initialize detector
    detector = DeepFakeDetector()

    results = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] Analyzing: {video.name}... ", end="", flush=True)
        try:
            result = detector.analyze(str(video))
            results.append({
                "video": str(video.absolute()),
                "verdict": result.verdict.value,
                "confidence": result.confidence,
            })
            print(f"{result.verdict.value} ({result.confidence:.1%})")
        except Exception as e:
            results.append({
                "video": str(video.absolute()),
                "verdict": "ERROR",
                "error": str(e),
            })
            print(f"ERROR: {e}")

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Summary
    fake_count = sum(1 for r in results if r.get("verdict") == "FAKE")
    real_count = sum(1 for r in results if r.get("verdict") == "REAL")
    error_count = sum(1 for r in results if r.get("verdict") == "ERROR")

    print(f"\nSummary:")
    print(f"  - Total videos: {len(results)}")
    print(f"  - Detected fakes: {fake_count}")
    print(f"  - Authentic: {real_count}")
    print(f"  - Errors: {error_count}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Handle the info command."""
    print("DeepFake Detection System")
    print("=" * 40)
    print()
    print("Detection Tools:")
    print("  - Video Extractor: Frame extraction from video files")
    print("  - Face Analyzer: Face detection and landmark analysis")
    print("  - Temporal Analyzer: Blink rate and movement analysis")
    print("  - Frequency Analyzer: FFT-based GAN fingerprint detection")
    print("  - Optical Flow: Motion consistency analysis")
    print()
    print("Supported Formats:")
    print("  - MP4, AVI, MOV, MKV, WebM")
    print()
    print("Detection Methods:")
    print("  - Blink rate analysis (normal: 10-30 blinks/min)")
    print("  - Eye movement tracking")
    print("  - GAN artifact detection in frequency domain")
    print("  - Optical flow consistency")
    print("  - Face boundary artifact detection")
    print()
    print("For more information, see the documentation:")
    print("  - README.md")
    print("  - docs/PRD.md")
    print("  - docs/architecture.md")

    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Handle the version command."""
    from . import __version__
    print(f"deepfake-detector version {__version__}")
    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    command_handlers = {
        "analyze": cmd_analyze,
        "batch": cmd_batch,
        "info": cmd_info,
        "version": cmd_version,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
