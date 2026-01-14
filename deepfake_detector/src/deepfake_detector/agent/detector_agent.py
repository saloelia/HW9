"""
DeepFake Detector Agent implementation.

This module contains the main AI agent that orchestrates DeepFake detection
using LLM-powered reasoning and specialized analysis tools.

The agent follows an iterative analysis process:
1. Extract video frames
2. Run detection tools
3. Reason about results
4. Make final determination

Building Block Design:
    Input Data:
        - video_path: Path to video to analyze

    Output Data:
        - AnalysisResult: Complete analysis with verdict

    Setup Data:
        - llm_provider: LLM provider to use
        - tools: Available detection tools
        - config: Agent configuration
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deepfake_detector.core.models import (
    AgentState,
    AnalysisResult,
    DetectionVerdict,
    VideoMetadata,
)
from deepfake_detector.tools.base import BaseTool, ToolResult
from deepfake_detector.tools.video_extractor import VideoExtractor
from deepfake_detector.tools.face_analyzer import FaceAnalyzer
from deepfake_detector.tools.temporal_analyzer import TemporalAnalyzer
from deepfake_detector.tools.frequency_analyzer import FrequencyAnalyzer
from deepfake_detector.tools.optical_flow_analyzer import OpticalFlowAnalyzer
from deepfake_detector.agent.prompts import (
    SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    REASONING_PROMPT_TEMPLATE,
)


logger = logging.getLogger(__name__)


class AgentConfig:
    """Configuration for the DeepFake detector agent."""

    def __init__(
        self,
        llm_provider: str = "anthropic",
        model: str = "claude-3-sonnet-20240229",
        max_iterations: int = 10,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        verbose: bool = True,
    ) -> None:
        """
        Initialize agent configuration.

        Args:
            llm_provider: LLM provider ("anthropic" or "openai")
            model: Model identifier
            max_iterations: Maximum reasoning iterations
            temperature: LLM temperature
            max_tokens: Maximum tokens per response
            verbose: Enable verbose logging
        """
        self.llm_provider = llm_provider
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose


class DeepFakeDetectorAgent:
    """
    AI Agent for DeepFake video detection.

    This agent orchestrates the detection process by:
    1. Managing detection tools
    2. Calling LLM for reasoning
    3. Synthesizing results into final verdict

    The agent supports two modes:
    - Autonomous: Agent decides which tools to run
    - Sequential: Runs all tools in predefined order

    Example:
        >>> agent = DeepFakeDetectorAgent()
        >>> result = agent.analyze("path/to/video.mp4")
        >>> print(f"Verdict: {result.verdict}")
        >>> print(f"Confidence: {result.confidence}")

    Attributes:
        config: Agent configuration
        tools: Dictionary of available tools
        state: Current agent state
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the DeepFake detector agent.

        Args:
            config: Agent configuration (Setup Data)
            api_key: Optional API key (defaults to environment variable)
        """
        self.config = config or AgentConfig()
        self._api_key = api_key or os.getenv(
            f"{self.config.llm_provider.upper()}_API_KEY"
        )
        self._client = None

        # Initialize tools
        self.tools: Dict[str, BaseTool] = {
            "video_extractor": VideoExtractor(),
            "face_analyzer": FaceAnalyzer(),
            "temporal_analyzer": TemporalAnalyzer(),
            "frequency_analyzer": FrequencyAnalyzer(),
            "optical_flow_analyzer": OpticalFlowAnalyzer(),
        }

        # Agent state
        self.state = AgentState()

        # Token tracking
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        # Initialize LLM client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the LLM client based on provider."""
        if not self._api_key:
            logger.warning(
                f"No API key found for {self.config.llm_provider}. "
                "Agent will run in offline mode (tools only, no LLM reasoning)."
            )
            return

        try:
            if self.config.llm_provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            elif self.config.llm_provider == "openai":
                import openai
                self._client = openai.OpenAI(api_key=self._api_key)
            else:
                raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")
        except ImportError as e:
            logger.warning(f"Could not import {self.config.llm_provider}: {e}")
            self._client = None

    def analyze(
        self,
        video_path: str,
        mode: str = "sequential",
        save_results: bool = True,
    ) -> AnalysisResult:
        """
        Analyze a video for DeepFake manipulation.

        This is the main entry point for video analysis.

        Args:
            video_path: Path to the video file (Input Data)
            mode: Analysis mode ("sequential" or "autonomous")
            save_results: Whether to save results to file

        Returns:
            AnalysisResult with verdict and evidence (Output Data)
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())[:8]

        logger.info(f"Starting analysis {analysis_id} for: {video_path}")
        self.state.reset()

        try:
            # Step 1: Extract frames
            self.state.current_step = "frame_extraction"
            extraction_result = self._extract_frames(video_path)

            if not extraction_result.success:
                return self._create_error_result(
                    analysis_id, video_path, extraction_result.error, start_time
                )

            frames = extraction_result.data["frames"]
            metadata = extraction_result.data["metadata"]
            timestamps = extraction_result.data["timestamps"]

            logger.info(f"Extracted {len(frames)} frames")

            # Step 2: Run analysis tools
            if mode == "sequential":
                analysis_results = self._run_sequential_analysis(frames, timestamps)
            else:
                analysis_results = self._run_autonomous_analysis(
                    frames, timestamps, metadata
                )

            # Step 3: Synthesize results
            self.state.current_step = "synthesis"
            verdict, confidence, reasoning = self._synthesize_results(
                analysis_results, metadata
            )

            # Create final result
            processing_time = time.time() - start_time

            result = AnalysisResult(
                analysis_id=analysis_id,
                analysis_timestamp=datetime.now(),
                video_metadata=metadata,
                frame_analyses=analysis_results.get("frame_analyses", []),
                temporal_analysis=analysis_results.get("temporal_result"),
                frequency_analysis=analysis_results.get("frequency_result"),
                optical_flow=analysis_results.get("optical_flow_result"),
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                evidence=self.state.evidence_collected,
                processing_time=processing_time,
                frames_analyzed=len(frames),
                faces_detected=analysis_results.get("total_faces", 0),
                total_tokens_used=self._total_prompt_tokens + self._total_completion_tokens,
                prompt_tokens=self._total_prompt_tokens,
                completion_tokens=self._total_completion_tokens,
            )

            if save_results:
                self._save_results(result)

            logger.info(
                f"Analysis complete: {verdict.value} "
                f"(confidence: {confidence:.2f}) in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return self._create_error_result(
                analysis_id, video_path, str(e), start_time
            )

    def _extract_frames(self, video_path: str) -> ToolResult:
        """Extract frames from video using the video extractor tool."""
        extractor = self.tools["video_extractor"]
        result = extractor.execute(video_path=video_path)
        self.state.update_tool_result("video_extractor", result)
        return result

    def _run_sequential_analysis(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
    ) -> Dict[str, Any]:
        """
        Run all analysis tools in sequence.

        Args:
            frames: Extracted video frames
            timestamps: Frame timestamps

        Returns:
            Dictionary with all analysis results
        """
        results: Dict[str, Any] = {}

        # Face Analysis
        self.state.current_step = "face_analysis"
        logger.info("Running face analysis...")

        face_result = self.tools["face_analyzer"].execute(
            frames=frames,
            timestamps=timestamps,
        )

        if face_result.success:
            results["frame_analyses"] = face_result.data["frame_analyses"]
            results["face_consistency"] = face_result.data["consistency_score"]
            results["total_faces"] = face_result.data["total_faces_detected"]

            for anomaly in face_result.data.get("anomalies", []):
                self.state.add_evidence(anomaly)

            self.state.update_tool_result("face_analyzer", face_result)

        # Temporal Analysis
        self.state.current_step = "temporal_analysis"
        logger.info("Running temporal analysis...")

        if "frame_analyses" in results:
            temporal_result = self.tools["temporal_analyzer"].execute(
                frame_analyses=results["frame_analyses"],
            )

            if temporal_result.success:
                results["temporal_result"] = temporal_result.data["temporal_result"]
                results["temporal_score"] = temporal_result.data["overall_score"]

                for anomaly in temporal_result.data.get("anomalies", []):
                    self.state.add_evidence(anomaly)

                self.state.update_tool_result("temporal_analyzer", temporal_result)

        # Frequency Analysis
        self.state.current_step = "frequency_analysis"
        logger.info("Running frequency analysis...")

        frequency_result = self.tools["frequency_analyzer"].execute(frames=frames)

        if frequency_result.success:
            results["frequency_result"] = frequency_result.data["frequency_result"]
            results["frequency_score"] = frequency_result.data["overall_score"]

            for anomaly in frequency_result.data.get("anomalies", []):
                self.state.add_evidence(anomaly)

            self.state.update_tool_result("frequency_analyzer", frequency_result)

        # Optical Flow Analysis
        self.state.current_step = "optical_flow_analysis"
        logger.info("Running optical flow analysis...")

        if len(frames) >= 2:
            flow_result = self.tools["optical_flow_analyzer"].execute(frames=frames)

            if flow_result.success:
                results["optical_flow_result"] = flow_result.data["optical_flow_result"]
                results["optical_flow_score"] = flow_result.data["overall_score"]

                for anomaly in flow_result.data.get("anomalies", []):
                    self.state.add_evidence(anomaly)

                self.state.update_tool_result("optical_flow_analyzer", flow_result)

        return results

    def _run_autonomous_analysis(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        metadata: VideoMetadata,
    ) -> Dict[str, Any]:
        """
        Run analysis with LLM-guided tool selection.

        The agent decides which tools to run based on intermediate results.

        Args:
            frames: Extracted video frames
            timestamps: Frame timestamps
            metadata: Video metadata

        Returns:
            Dictionary with all analysis results
        """
        # For now, fall back to sequential
        # Full autonomous mode requires LLM integration
        return self._run_sequential_analysis(frames, timestamps)

    def _synthesize_results(
        self,
        analysis_results: Dict[str, Any],
        metadata: VideoMetadata,
    ) -> Tuple[DetectionVerdict, float, str]:
        """
        Synthesize analysis results into final verdict.

        Uses LLM if available, otherwise uses rule-based synthesis.

        Args:
            analysis_results: Results from all analysis tools
            metadata: Video metadata

        Returns:
            Tuple of (verdict, confidence, reasoning)
        """
        if self._client:
            return self._llm_synthesis(analysis_results, metadata)
        else:
            return self._rule_based_synthesis(analysis_results)

    def _llm_synthesis(
        self,
        analysis_results: Dict[str, Any],
        metadata: VideoMetadata,
    ) -> Tuple[DetectionVerdict, float, str]:
        """
        Use LLM to synthesize results and make determination.

        Args:
            analysis_results: Results from analysis tools
            metadata: Video metadata

        Returns:
            Tuple of (verdict, confidence, reasoning)
        """
        # Prepare results summary
        summary_parts = []

        if "face_consistency" in analysis_results:
            summary_parts.append(
                f"- Face Consistency Score: {analysis_results['face_consistency']:.3f}"
            )

        if "temporal_score" in analysis_results:
            summary_parts.append(
                f"- Temporal Analysis Score: {analysis_results['temporal_score']:.3f}"
            )
            if analysis_results.get("temporal_result"):
                tr = analysis_results["temporal_result"]
                if tr.blink_rate:
                    summary_parts.append(f"  - Blink Rate: {tr.blink_rate:.1f}/min")

        if "frequency_score" in analysis_results:
            summary_parts.append(
                f"- Frequency Analysis Score: {analysis_results['frequency_score']:.3f}"
            )
            if analysis_results.get("frequency_result"):
                fr = analysis_results["frequency_result"]
                summary_parts.append(
                    f"  - GAN Fingerprint Score: {fr.gan_fingerprint_score:.3f}"
                )

        if "optical_flow_score" in analysis_results:
            summary_parts.append(
                f"- Optical Flow Score: {analysis_results['optical_flow_score']:.3f}"
            )

        tool_summary = "\n".join(summary_parts)
        evidence_summary = "\n".join(
            f"- {e}" for e in self.state.evidence_collected
        )

        # Calculate overall scores
        scores = []
        if "face_consistency" in analysis_results:
            scores.append(analysis_results["face_consistency"])
        if "temporal_score" in analysis_results:
            scores.append(analysis_results["temporal_score"])
        if "frequency_score" in analysis_results:
            scores.append(analysis_results["frequency_score"])
        if "optical_flow_score" in analysis_results:
            scores.append(analysis_results["optical_flow_score"])

        avg_score = sum(scores) / len(scores) if scores else 0.5
        score_summary = f"Average Analysis Score: {avg_score:.3f}"

        # Create prompt for LLM
        prompt = REASONING_PROMPT_TEMPLATE.format(
            tool_results_summary=tool_summary + "\n\nEvidence:\n" + evidence_summary,
            analysis_scores=score_summary,
        )

        try:
            response = self._call_llm(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}, using rule-based")
            return self._rule_based_synthesis(analysis_results)

    def _rule_based_synthesis(
        self, analysis_results: Dict[str, Any]
    ) -> Tuple[DetectionVerdict, float, str]:
        """
        Rule-based synthesis when LLM is not available.

        Uses weighted scoring of analysis results.

        Args:
            analysis_results: Results from analysis tools

        Returns:
            Tuple of (verdict, confidence, reasoning)
        """
        scores = []
        weights = []
        reasoning_parts = []

        # Face analysis (weight: 0.25)
        if "face_consistency" in analysis_results:
            score = analysis_results["face_consistency"]
            scores.append(score)
            weights.append(0.25)
            reasoning_parts.append(f"Face consistency: {score:.3f}")

        # Temporal analysis (weight: 0.25)
        if "temporal_score" in analysis_results:
            score = analysis_results["temporal_score"]
            scores.append(score)
            weights.append(0.25)
            reasoning_parts.append(f"Temporal patterns: {score:.3f}")

        # Frequency analysis (weight: 0.30 - most important for GAN detection)
        if "frequency_score" in analysis_results:
            score = analysis_results["frequency_score"]
            scores.append(score)
            weights.append(0.30)
            reasoning_parts.append(f"Frequency analysis: {score:.3f}")

        # Optical flow (weight: 0.20)
        if "optical_flow_score" in analysis_results:
            score = analysis_results["optical_flow_score"]
            scores.append(score)
            weights.append(0.20)
            reasoning_parts.append(f"Optical flow: {score:.3f}")

        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            weighted_score = 0.5

        # Determine verdict
        if weighted_score >= 0.7:
            verdict = DetectionVerdict.REAL
            confidence = min(weighted_score, 0.95)
        elif weighted_score <= 0.4:
            verdict = DetectionVerdict.FAKE
            confidence = min(1.0 - weighted_score, 0.95)
        else:
            verdict = DetectionVerdict.UNCERTAIN
            confidence = 0.5 + abs(weighted_score - 0.5)

        # Add evidence-based adjustments
        evidence_count = len(self.state.evidence_collected)
        if evidence_count >= 3 and verdict != DetectionVerdict.REAL:
            confidence = min(confidence + 0.1, 0.95)

        reasoning = (
            f"Rule-based analysis with weighted scoring.\n"
            f"Scores: {', '.join(reasoning_parts)}\n"
            f"Weighted average: {weighted_score:.3f}\n"
            f"Evidence items: {evidence_count}\n"
            f"Evidence: {'; '.join(self.state.evidence_collected[:5])}"
        )

        return verdict, confidence, reasoning

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: User prompt

        Returns:
            LLM response text
        """
        if not self._client:
            raise RuntimeError("LLM client not initialized")

        if self.config.llm_provider == "anthropic":
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            self._total_prompt_tokens += response.usage.input_tokens
            self._total_completion_tokens += response.usage.output_tokens
            return response.content[0].text

        elif self.config.llm_provider == "openai":
            response = self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            self._total_prompt_tokens += response.usage.prompt_tokens
            self._total_completion_tokens += response.usage.completion_tokens
            return response.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.config.llm_provider}")

    def _parse_llm_response(
        self, response: str
    ) -> Tuple[DetectionVerdict, float, str]:
        """
        Parse LLM response to extract verdict, confidence, and reasoning.

        Args:
            response: LLM response text

        Returns:
            Tuple of (verdict, confidence, reasoning)
        """
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                verdict_str = data.get("verdict", "UNCERTAIN").upper()
                verdict = DetectionVerdict(verdict_str)
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning", "")

                # Add evidence from LLM
                for evidence in data.get("evidence", []):
                    self.state.add_evidence(evidence)

                return verdict, confidence, reasoning

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        # Fallback: extract from text
        if "FAKE" in response.upper():
            verdict = DetectionVerdict.FAKE
            confidence = 0.7
        elif "REAL" in response.upper():
            verdict = DetectionVerdict.REAL
            confidence = 0.7
        else:
            verdict = DetectionVerdict.UNCERTAIN
            confidence = 0.5

        return verdict, confidence, response[:500]

    def _create_error_result(
        self,
        analysis_id: str,
        video_path: str,
        error: str,
        start_time: float,
    ) -> AnalysisResult:
        """Create an error result when analysis fails."""
        return AnalysisResult(
            analysis_id=analysis_id,
            video_metadata=VideoMetadata(
                file_path=Path(video_path),
                duration=0,
                fps=0,
                resolution=(0, 0),
                codec="unknown",
                file_size=0,
            ),
            verdict=DetectionVerdict.ERROR,
            confidence=0.0,
            reasoning=f"Analysis failed: {error}",
            processing_time=time.time() - start_time,
        )

    def _save_results(self, result: AnalysisResult) -> None:
        """Save analysis results to file."""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"analysis_{result.analysis_id}.json"

        with open(output_file, "w") as f:
            json.dump(result.get_summary(), f, indent=2, default=str)

        logger.info(f"Results saved to: {output_file}")

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions for LLM function calling."""
        return [tool.get_tool_definition() for tool in self.tools.values()]
