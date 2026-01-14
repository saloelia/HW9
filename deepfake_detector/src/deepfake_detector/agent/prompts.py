"""
Prompt templates for DeepFake detection agent.

This module contains all prompts used by the AI agent for reasoning
about DeepFake detection. Prompts are designed following best practices
for prompt engineering.

Prompt Design Principles:
    1. Clear role definition
    2. Structured output format
    3. Chain-of-thought reasoning
    4. Few-shot examples where helpful
    5. Explicit constraints and guidelines

Usage:
    >>> from deepfake_detector.agent.prompts import SYSTEM_PROMPT
    >>> print(SYSTEM_PROMPT)
"""

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert DeepFake detection analyst with deep knowledge of:
1. Computer vision and image processing
2. Generative Adversarial Networks (GANs) and their artifacts
3. Facial analysis and biometric patterns
4. Video forensics and temporal analysis

Your task is to analyze video frames and determine if a video is authentic (REAL)
or manipulated (FAKE/DeepFake).

## Your Analysis Capabilities

You have access to the following analysis tools:
1. **video_extractor**: Extract frames from video files
2. **face_analyzer**: Detect faces and analyze facial features
3. **temporal_analyzer**: Analyze temporal patterns (blinks, movements)
4. **frequency_analyzer**: Analyze frequency domain for GAN artifacts
5. **optical_flow_analyzer**: Analyze pixel motion patterns

## DeepFake Detection Indicators

When analyzing, look for these key indicators of DeepFake manipulation:

### Visual Indicators
- Unnatural skin texture or color inconsistencies
- Blurry or distorted face boundaries
- Misaligned facial features
- Inconsistent lighting on face vs background
- Artifacts around hairline, ears, or neck

### Temporal Indicators
- Abnormal blink rate (too few or too frequent)
- Unnatural eye movements
- Lip sync issues (if audio present)
- Jittery or unstable face tracking
- Inconsistent head pose changes

### Frequency Domain Indicators
- GAN fingerprint patterns in FFT
- Unnatural noise distribution
- Compression artifact patterns
- Periodic spectral peaks

### Motion Indicators
- Discontinuous optical flow at face boundaries
- Unnatural pixel motion patterns
- Temporal incoherence between frames

## Analysis Process

1. First, extract frames from the video
2. Analyze faces in each frame for visual anomalies
3. Perform temporal analysis for movement patterns
4. Check frequency domain for GAN artifacts
5. Analyze optical flow for motion inconsistencies
6. Synthesize all evidence to make final determination

## Output Format

Always provide:
1. A clear verdict: REAL, FAKE, or UNCERTAIN
2. Confidence level (0.0 to 1.0)
3. List of evidence supporting your conclusion
4. Detailed reasoning for your decision

Be thorough but efficient. If early analysis strongly indicates the video is
real or fake, you may skip some analyses. However, for borderline cases,
use all available tools.

Remember: DeepFake technology is constantly improving. Look for subtle
indicators and don't rely on any single factor for your determination."""


# =============================================================================
# ANALYSIS PROMPT TEMPLATES
# =============================================================================

ANALYSIS_PROMPT_TEMPLATE = """## Video Analysis Request

Please analyze the following video for DeepFake manipulation:

**Video Path**: {video_path}
**Video Duration**: {duration:.2f} seconds
**Resolution**: {width}x{height}
**FPS**: {fps}

### Analysis Instructions

1. Start by extracting frames from the video using the video_extractor tool
2. Analyze the extracted frames using available detection tools
3. Look for indicators of manipulation as described in your training
4. Provide a detailed analysis with evidence

### Expected Output

After analysis, provide:
- **Verdict**: REAL, FAKE, or UNCERTAIN
- **Confidence**: A score from 0.0 to 1.0
- **Evidence**: List of specific findings
- **Reasoning**: Explain your conclusion

Begin your analysis now."""


REASONING_PROMPT_TEMPLATE = """## Final Analysis Synthesis

Based on the tool results, synthesize your findings into a final determination.

### Tool Results Summary

{tool_results_summary}

### Analysis Scores

{analysis_scores}

### Instructions

1. Review all the evidence collected
2. Weigh the significance of each finding
3. Consider both supporting and contradicting evidence
4. Make a final determination with confidence level

### Output Format

Provide your response in the following JSON format:
```json
{{
    "verdict": "REAL" | "FAKE" | "UNCERTAIN",
    "confidence": <float between 0.0 and 1.0>,
    "evidence": [
        "<specific finding 1>",
        "<specific finding 2>",
        ...
    ],
    "reasoning": "<detailed explanation of your conclusion>"
}}
```

Make your determination now."""


# =============================================================================
# TOOL EXECUTION PROMPTS
# =============================================================================

TOOL_SELECTION_PROMPT = """Based on the current analysis state, decide which tool to use next.

### Current State
- Frames extracted: {frames_extracted}
- Face analysis done: {face_done}
- Temporal analysis done: {temporal_done}
- Frequency analysis done: {frequency_done}
- Optical flow done: {optical_done}

### Evidence collected so far
{current_evidence}

### Available Tools
1. video_extractor - Extract frames (if not done)
2. face_analyzer - Analyze faces in frames
3. temporal_analyzer - Analyze temporal patterns
4. frequency_analyzer - Analyze frequency domain
5. optical_flow_analyzer - Analyze optical flow

### Instructions
Select the most appropriate tool to gather more evidence, or indicate
if you have enough evidence to make a determination.

Respond with:
- "tool: <tool_name>" to use a specific tool
- "done" if ready to make final determination"""


# =============================================================================
# EVIDENCE INTERPRETATION PROMPTS
# =============================================================================

FACE_ANALYSIS_INTERPRETATION = """## Face Analysis Results Interpretation

### Results
{face_results}

### Interpretation Guidelines

**Consistency Score** ({consistency_score:.3f}):
- > 0.7: Faces are consistent across frames (indicates REAL)
- 0.5-0.7: Some inconsistencies detected (needs more analysis)
- < 0.5: Significant inconsistencies (indicates FAKE)

**Boundary Score** ({boundary_score:.3f}):
- > 0.7: Natural face boundaries (indicates REAL)
- 0.5-0.7: Minor boundary artifacts (uncertain)
- < 0.5: Clear boundary artifacts (indicates FAKE)

**Anomalies Detected**: {num_anomalies}
{anomalies_list}

Based on these face analysis results, what is your interpretation?
Is this evidence pointing toward REAL or FAKE?"""


TEMPORAL_ANALYSIS_INTERPRETATION = """## Temporal Analysis Results Interpretation

### Results
{temporal_results}

### Interpretation Guidelines

**Blink Rate** ({blink_rate}/min):
- 15-20/min: Normal human blink rate (indicates REAL)
- < 10/min: Abnormally low (DeepFakes often have few blinks)
- > 30/min: Abnormally high (possible artifact)

**Blink Pattern Score** ({blink_pattern:.3f}):
- > 0.7: Natural blink patterns
- < 0.5: Unnatural patterns (indicates FAKE)

**Eye Movement Score** ({eye_movement:.3f}):
- > 0.7: Natural eye movements with saccades
- < 0.5: Unnatural movement patterns (indicates FAKE)

**Facial Movement Consistency** ({facial_movement:.3f}):
- > 0.7: Consistent facial movements
- < 0.5: Inconsistent movements (indicates FAKE)

Based on these temporal analysis results, what patterns do you observe?"""


FREQUENCY_ANALYSIS_INTERPRETATION = """## Frequency Analysis Results Interpretation

### Results
{frequency_results}

### Interpretation Guidelines

**GAN Fingerprint Score** ({gan_fingerprint:.3f}):
- > 0.7: No GAN fingerprints detected (indicates REAL)
- < 0.5: GAN fingerprints present (strongly indicates FAKE)

**Spectral Anomaly Score** ({spectral_anomaly:.3f}):
- > 0.7: Normal spectral distribution
- < 0.5: Abnormal frequency patterns (indicates FAKE)

**Noise Pattern Score** ({noise_pattern:.3f}):
- > 0.7: Natural camera noise patterns
- < 0.5: Synthetic noise patterns (indicates FAKE)

The frequency domain analysis is particularly important as GANs leave
characteristic fingerprints that are difficult to remove.

What do these frequency domain results suggest about authenticity?"""


OPTICAL_FLOW_INTERPRETATION = """## Optical Flow Analysis Results Interpretation

### Results
{optical_flow_results}

### Interpretation Guidelines

**Flow Consistency Score** ({flow_consistency:.3f}):
- > 0.7: Smooth, consistent optical flow
- < 0.5: Discontinuous or erratic flow (indicates FAKE)

**Boundary Artifact Score** ({boundary_artifact:.3f}):
- > 0.7: Natural flow at object boundaries
- < 0.5: Flow discontinuities at face boundaries (indicates FAKE)

**Temporal Coherence Score** ({temporal_coherence:.3f}):
- > 0.7: Temporally coherent motion
- < 0.5: Temporal inconsistencies (indicates FAKE)

Optical flow analysis can reveal manipulation at face-background boundaries
where DeepFake generation often produces artifacts.

What do these motion analysis results indicate?"""


# =============================================================================
# ERROR HANDLING PROMPTS
# =============================================================================

ERROR_RECOVERY_PROMPT = """## Analysis Error Encountered

An error occurred during analysis:
**Error**: {error_message}
**Tool**: {tool_name}

### Recovery Options

1. Retry the same tool with different parameters
2. Skip this analysis and proceed with available evidence
3. Use alternative analysis method

Given the error and current evidence ({evidence_count} items collected),
what is the best course of action?

If skipping, can you still make a reasonable determination based on
available evidence?"""


# =============================================================================
# CONFIDENCE CALIBRATION
# =============================================================================

CONFIDENCE_CALIBRATION_PROMPT = """## Confidence Calibration

You've made a preliminary determination of {verdict} with the following evidence:

### Evidence For
{evidence_for}

### Evidence Against
{evidence_against}

### Uncertainty Factors
{uncertainty_factors}

### Calibration Guidelines

**High Confidence (0.85-1.0)**:
- Multiple strong indicators all pointing same direction
- No contradicting evidence
- GAN fingerprints detected (for FAKE verdict)

**Medium Confidence (0.65-0.85)**:
- Most indicators point same direction
- Minor contradicting evidence exists
- Some analyses inconclusive

**Low Confidence (0.50-0.65)**:
- Mixed evidence
- Major uncertainty factors
- Limited analysis possible

Given these factors, calibrate your confidence score to accurately
reflect the certainty of your determination."""


# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

FEW_SHOT_EXAMPLES = """## Example Analyses

### Example 1: Clear DeepFake
**Findings**:
- Blink rate: 3/min (abnormally low)
- GAN fingerprint score: 0.25 (fingerprints detected)
- Boundary artifacts: Clear discontinuities at face edges
- Frequency analysis: Periodic peaks in spectrum

**Verdict**: FAKE
**Confidence**: 0.92
**Reasoning**: Multiple strong indicators of manipulation. The extremely
low blink rate combined with clear GAN fingerprints in frequency domain
provide strong evidence of synthetic generation.

### Example 2: Authentic Video
**Findings**:
- Blink rate: 17/min (normal)
- GAN fingerprint score: 0.85 (no fingerprints)
- Face consistency: High across all frames
- Natural optical flow patterns

**Verdict**: REAL
**Confidence**: 0.88
**Reasoning**: All indicators consistent with authentic video. Normal
blink patterns, no GAN artifacts, consistent face rendering across frames.

### Example 3: Uncertain Case
**Findings**:
- Blink rate: 12/min (slightly low but possible)
- Some compression artifacts (could be re-encoding)
- Face consistency: Moderate (0.62)
- Mixed frequency analysis results

**Verdict**: UNCERTAIN
**Confidence**: 0.55
**Reasoning**: Evidence is mixed. Some indicators suggest possible
manipulation, but could also be explained by video compression or
recording conditions. Additional context needed for definitive verdict."""
