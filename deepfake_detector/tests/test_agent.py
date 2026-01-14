"""
Unit tests for the DeepFake detector agent.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from deepfake_detector.agent.detector_agent import (
    DeepFakeDetectorAgent,
    AgentConfig,
)
from deepfake_detector.agent.prompts import (
    SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    REASONING_PROMPT_TEMPLATE,
)
from deepfake_detector.core.models import (
    DetectionVerdict,
    AnalysisResult,
)


class TestAgentConfig:
    """Tests for AgentConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()

        assert config.llm_provider == "anthropic"
        assert config.temperature == 0.3
        assert config.max_iterations == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            llm_provider="openai",
            model="gpt-4",
            temperature=0.5,
            max_iterations=5,
        )

        assert config.llm_provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5


class TestDeepFakeDetectorAgent:
    """Tests for DeepFakeDetectorAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = DeepFakeDetectorAgent()

        assert agent.config is not None
        assert len(agent.tools) == 5
        assert "video_extractor" in agent.tools
        assert "face_analyzer" in agent.tools

    def test_initialization_with_config(self):
        """Test agent with custom config."""
        config = AgentConfig(temperature=0.5)
        agent = DeepFakeDetectorAgent(config=config)

        assert agent.config.temperature == 0.5

    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        agent = DeepFakeDetectorAgent()
        definitions = agent.get_tool_definitions()

        assert len(definitions) == 5
        for defn in definitions:
            assert "name" in defn
            assert "description" in defn
            assert "input_schema" in defn

    def test_state_initialization(self):
        """Test agent state is initialized correctly."""
        agent = DeepFakeDetectorAgent()

        assert agent.state.current_step == "initialization"
        assert agent.state.frames_processed == 0

    def test_rule_based_synthesis_fake(self):
        """Test rule-based synthesis for FAKE verdict."""
        agent = DeepFakeDetectorAgent()

        # Simulate low scores (indicating FAKE)
        analysis_results = {
            "face_consistency": 0.3,
            "temporal_score": 0.3,
            "frequency_score": 0.3,
            "optical_flow_score": 0.3,
        }

        verdict, confidence, reasoning = agent._rule_based_synthesis(analysis_results)

        assert verdict == DetectionVerdict.FAKE

    def test_rule_based_synthesis_real(self):
        """Test rule-based synthesis for REAL verdict."""
        agent = DeepFakeDetectorAgent()

        # Simulate high scores (indicating REAL)
        analysis_results = {
            "face_consistency": 0.9,
            "temporal_score": 0.85,
            "frequency_score": 0.9,
            "optical_flow_score": 0.85,
        }

        verdict, confidence, reasoning = agent._rule_based_synthesis(analysis_results)

        assert verdict == DetectionVerdict.REAL

    def test_rule_based_synthesis_uncertain(self):
        """Test rule-based synthesis for UNCERTAIN verdict."""
        agent = DeepFakeDetectorAgent()

        # Simulate mid-range scores
        analysis_results = {
            "face_consistency": 0.55,
            "temporal_score": 0.5,
            "frequency_score": 0.55,
            "optical_flow_score": 0.5,
        }

        verdict, confidence, reasoning = agent._rule_based_synthesis(analysis_results)

        assert verdict == DetectionVerdict.UNCERTAIN

    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        agent = DeepFakeDetectorAgent()

        response = '''
        Based on analysis:
        ```json
        {
            "verdict": "FAKE",
            "confidence": 0.85,
            "evidence": ["Low blink rate", "GAN fingerprints"],
            "reasoning": "Multiple indicators suggest manipulation"
        }
        ```
        '''

        verdict, confidence, reasoning = agent._parse_llm_response(response)

        assert verdict == DetectionVerdict.FAKE
        assert confidence == 0.85

    def test_parse_llm_response_no_json(self):
        """Test parsing response without JSON."""
        agent = DeepFakeDetectorAgent()

        response = "The video appears to be FAKE based on analysis."

        verdict, confidence, reasoning = agent._parse_llm_response(response)

        assert verdict == DetectionVerdict.FAKE
        assert confidence == 0.7

    def test_parse_llm_response_real_text(self):
        """Test parsing response indicating REAL."""
        agent = DeepFakeDetectorAgent()

        response = "This video is REAL and authentic."

        verdict, confidence, reasoning = agent._parse_llm_response(response)

        assert verdict == DetectionVerdict.REAL

    def test_create_error_result(self):
        """Test error result creation."""
        agent = DeepFakeDetectorAgent()

        result = agent._create_error_result(
            analysis_id="test123",
            video_path="/test/video.mp4",
            error="Test error message",
            start_time=0.0,
        )

        assert result.verdict == DetectionVerdict.ERROR
        assert result.confidence == 0.0
        assert "Test error message" in result.reasoning


class TestPrompts:
    """Tests for prompt templates."""

    def test_system_prompt_content(self):
        """Test system prompt contains key elements."""
        assert "DeepFake" in SYSTEM_PROMPT
        assert "face" in SYSTEM_PROMPT.lower()
        assert "temporal" in SYSTEM_PROMPT.lower()
        assert "frequency" in SYSTEM_PROMPT.lower()

    def test_analysis_prompt_template_formatting(self):
        """Test analysis prompt can be formatted."""
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            video_path="/test/video.mp4",
            duration=10.5,
            width=1920,
            height=1080,
            fps=30.0,
        )

        assert "/test/video.mp4" in prompt
        assert "10.50" in prompt
        assert "1920x1080" in prompt

    def test_reasoning_prompt_template_formatting(self):
        """Test reasoning prompt can be formatted."""
        prompt = REASONING_PROMPT_TEMPLATE.format(
            tool_results_summary="Face analysis: 0.8\nTemporal: 0.7",
            analysis_scores="Average: 0.75",
        )

        assert "Face analysis" in prompt
        assert "Average: 0.75" in prompt


class TestAgentIntegration:
    """Integration tests for agent behavior."""

    @pytest.fixture
    def mock_video_file(self, tmp_path):
        """Create a mock video file path."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")
        return str(video_path)

    def test_analyze_missing_file(self):
        """Test analyzing non-existent file."""
        agent = DeepFakeDetectorAgent()
        result = agent.analyze("/nonexistent/video.mp4")

        assert result.verdict == DetectionVerdict.ERROR

    def test_agent_state_reset_between_analyses(self):
        """Test that agent state resets between analyses."""
        agent = DeepFakeDetectorAgent()

        # First analysis
        agent.state.add_evidence("Test evidence")
        agent.state.current_step = "analysis"

        # Reset should happen in analyze()
        agent.state.reset()

        assert agent.state.current_step == "initialization"
        assert len(agent.state.evidence_collected) == 0
