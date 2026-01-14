"""
Frequency domain analysis tool for DeepFake detection.

This module analyzes frequency domain characteristics of video frames
to detect GAN-generated artifacts and unnatural patterns.

Key Detection Methods:
    1. FFT-based spectral analysis
    2. GAN fingerprint detection
    3. Noise pattern analysis
    4. Compression artifact detection

Building Block Design:
    Input Data:
        - frames: List of video frames to analyze

    Output Data:
        - frequency_result: FrequencyAnalysisResult
        - spectral_features: Extracted spectral features

    Setup Data:
        - fft_enabled: Whether to use FFT analysis
        - frequency_bands: Frequency bands to analyze
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import fftpack

from deepfake_detector.tools.base import BaseTool, ToolConfig, ToolResult
from deepfake_detector.core.models import FrequencyAnalysisResult


class FrequencyAnalyzerConfig(ToolConfig):
    """Configuration for FrequencyAnalyzer tool."""

    fft_enabled: bool = True
    # Frequency bands: (low_freq, high_freq)
    frequency_bands: Dict[str, Tuple[float, float]] = {
        "low": (0, 0.1),
        "mid": (0.1, 0.3),
        "high": (0.3, 0.5),
    }
    noise_threshold: float = 0.5
    gan_detection_enabled: bool = True


class FrequencyAnalyzer(BaseTool):
    """
    Tool for frequency domain analysis of video frames.

    This tool uses FFT and spectral analysis to detect artifacts
    commonly found in GAN-generated DeepFake videos.

    Detection Methods:
        1. Spectral Analysis: Examines frequency distribution.
           GANs often produce characteristic spectral patterns.

        2. GAN Fingerprint Detection: Looks for periodic patterns
           in the frequency domain that indicate GAN generation.

        3. Noise Pattern Analysis: Natural images have specific
           noise characteristics that GANs don't perfectly replicate.

        4. Compression Artifacts: Analyzes DCT coefficients for
           unnatural patterns from re-compression.

    Example:
        >>> analyzer = FrequencyAnalyzer()
        >>> result = analyzer.execute(frames=frame_list)
        >>> if result.success:
        ...     print(f"GAN fingerprint score: "
        ...           f"{result.data['frequency_result'].gan_fingerprint_score}")
    """

    def __init__(self, config: Optional[FrequencyAnalyzerConfig] = None) -> None:
        """
        Initialize the FrequencyAnalyzer.

        Args:
            config: Tool configuration (Setup Data)
        """
        super().__init__(
            name="frequency_analyzer",
            description=(
                "Analyze frequency domain characteristics of video frames. "
                "Detects GAN fingerprints, spectral anomalies, and unnatural "
                "noise patterns that indicate DeepFake manipulation."
            ),
            config=config or FrequencyAnalyzerConfig(),
        )
        self.config: FrequencyAnalyzerConfig = config or FrequencyAnalyzerConfig()

    def execute(
        self,
        frames: List[np.ndarray],
        face_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        analyze_gan_fingerprint: bool = True,
        analyze_noise: bool = True,
    ) -> ToolResult[Dict[str, Any]]:
        """
        Analyze frequency domain characteristics of frames.

        Args:
            frames: List of frame arrays (RGB format) (Input Data)
            face_regions: Optional list of face bounding boxes to focus analysis
            analyze_gan_fingerprint: Whether to analyze for GAN fingerprints
            analyze_noise: Whether to analyze noise patterns

        Returns:
            ToolResult containing:
                - frequency_result: FrequencyAnalysisResult object
                - spectral_features: Extracted features
                - anomalies: List of detected anomalies
        """
        start = time.time()
        self._execution_count += 1

        try:
            if not frames:
                return ToolResult.error_result("No frames provided")

            anomalies: List[str] = []
            spectral_features: Dict[str, Any] = {}

            # Spectral analysis
            spectral_scores = []
            for frame in frames:
                score = self._analyze_spectrum(frame)
                spectral_scores.append(score)

            avg_spectral_score = np.mean(spectral_scores)
            spectral_features["spectral_scores"] = spectral_scores

            if avg_spectral_score < 0.4:
                anomalies.append(
                    "Abnormal spectral distribution detected - "
                    "possible GAN generation"
                )

            # GAN fingerprint detection
            gan_fingerprint_score = 0.5
            if analyze_gan_fingerprint and self.config.gan_detection_enabled:
                gan_fingerprint_score = self._detect_gan_fingerprint(frames)
                spectral_features["gan_fingerprint_score"] = gan_fingerprint_score

                if gan_fingerprint_score < 0.4:
                    anomalies.append(
                        "GAN fingerprint patterns detected in frequency domain"
                    )

            # Noise pattern analysis
            noise_score = 0.5
            if analyze_noise:
                noise_score = self._analyze_noise_patterns(frames)
                spectral_features["noise_pattern_score"] = noise_score

                if noise_score < 0.4:
                    anomalies.append(
                        "Unnatural noise patterns detected - "
                        "inconsistent with natural camera noise"
                    )

            # Compression artifact analysis
            compression_score = self._analyze_compression_artifacts(frames)
            spectral_features["compression_score"] = compression_score

            if compression_score < 0.4:
                anomalies.append(
                    "Suspicious compression artifacts detected - "
                    "possible multiple re-encodings"
                )

            # High frequency analysis
            high_freq_ratio = self._analyze_high_frequencies(frames)
            spectral_features["high_frequency_ratio"] = high_freq_ratio

            # Create frequency analysis result
            frequency_result = FrequencyAnalysisResult(
                spectral_anomaly_score=avg_spectral_score,
                noise_pattern_score=noise_score,
                compression_artifact_score=compression_score,
                high_frequency_ratio=high_freq_ratio,
                gan_fingerprint_score=gan_fingerprint_score,
            )

            execution_time = time.time() - start
            self._total_execution_time += execution_time

            return ToolResult.success_result(
                data={
                    "frequency_result": frequency_result,
                    "spectral_features": spectral_features,
                    "anomalies": anomalies,
                    "overall_score": frequency_result.overall_score,
                },
                metadata={
                    "execution_time": round(execution_time, 3),
                    "frames_analyzed": len(frames),
                    "anomalies_found": len(anomalies),
                },
            )

        except Exception as e:
            execution_time = time.time() - start
            self._total_execution_time += execution_time
            return ToolResult.error_result(
                f"Error in frequency analysis: {str(e)}",
                metadata={"execution_time": round(execution_time, 3)},
            )

    def _analyze_spectrum(self, frame: np.ndarray) -> float:
        """
        Analyze spectral distribution of a frame.

        Args:
            frame: Frame array in RGB format

        Returns:
            Score indicating natural spectral distribution (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Apply FFT
        f_transform = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Analyze magnitude distribution
        # Natural images have specific frequency falloff (1/f noise)
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2

        # Create radial frequency bins
        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

        # Analyze power spectrum in radial bands
        band_powers = []
        num_bands = 10
        max_r = min(center_row, center_col)

        for i in range(num_bands):
            r_min = i * max_r / num_bands
            r_max = (i + 1) * max_r / num_bands
            mask = (r >= r_min) & (r < r_max)
            if np.any(mask):
                band_power = np.mean(magnitude[mask])
                band_powers.append(band_power)

        if len(band_powers) < 2:
            return 0.5

        # Check for natural 1/f falloff
        # Power should decrease with frequency
        band_powers = np.array(band_powers)
        expected_falloff = band_powers[0] / (np.arange(1, len(band_powers) + 1))

        # Calculate correlation with expected falloff
        correlation = np.corrcoef(band_powers, expected_falloff[:len(band_powers)])[0, 1]

        # Higher correlation = more natural
        score = (correlation + 1) / 2  # Map from [-1,1] to [0,1]

        return max(0, min(1, score))

    def _detect_gan_fingerprint(self, frames: List[np.ndarray]) -> float:
        """
        Detect GAN-specific fingerprints in frequency domain.

        GANs often produce characteristic periodic patterns in the
        frequency domain due to their architecture (upsampling, etc.).

        Args:
            frames: List of frame arrays

        Returns:
            Score indicating absence of GAN fingerprints (0-1)
            Lower score = more likely GAN-generated
        """
        fingerprint_scores = []

        for frame in frames[:10]:  # Analyze subset for efficiency
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Apply FFT
            f_transform = fftpack.fft2(gray)
            f_shift = fftpack.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))

            # Look for periodic peaks (GAN fingerprint)
            # GANs often produce grid-like patterns in frequency domain

            # Detect peaks
            magnitude_norm = (magnitude - magnitude.min()) / (
                magnitude.max() - magnitude.min() + 1e-8
            )

            # Count strong peaks outside center
            rows, cols = magnitude_norm.shape
            center_row, center_col = rows // 2, cols // 2
            center_mask_size = min(rows, cols) // 8

            # Mask out the center (DC component and low frequencies)
            magnitude_masked = magnitude_norm.copy()
            magnitude_masked[
                center_row - center_mask_size : center_row + center_mask_size,
                center_col - center_mask_size : center_col + center_mask_size,
            ] = 0

            # Count peaks above threshold
            threshold = 0.7
            peaks = magnitude_masked > threshold
            num_peaks = np.sum(peaks)

            # Many strong peaks outside center is suspicious
            # Natural images have fewer isolated peaks
            total_pixels = rows * cols
            peak_ratio = num_peaks / total_pixels

            # High peak ratio suggests GAN fingerprint
            if peak_ratio > 0.01:
                score = max(0, 1.0 - peak_ratio * 50)
            else:
                score = 1.0

            fingerprint_scores.append(score)

        return np.mean(fingerprint_scores) if fingerprint_scores else 0.5

    def _analyze_noise_patterns(self, frames: List[np.ndarray]) -> float:
        """
        Analyze noise patterns for naturalness.

        Natural camera noise has specific characteristics that
        GAN-generated images don't perfectly replicate.

        Args:
            frames: List of frame arrays

        Returns:
            Score indicating natural noise patterns (0-1)
        """
        noise_scores = []

        for frame in frames[:10]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Extract noise using high-pass filter
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray - blur

            # Analyze noise statistics
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))

            # Natural noise should be roughly Gaussian
            # Calculate kurtosis-like measure
            noise_normalized = (noise - np.mean(noise)) / (noise_std + 1e-8)
            kurtosis = np.mean(noise_normalized ** 4) - 3

            # Natural noise has kurtosis close to 0 (Gaussian)
            kurtosis_score = max(0, 1.0 - abs(kurtosis) / 10)

            # Check spatial correlation of noise
            # Natural noise is less spatially correlated
            noise_autocorr = cv2.filter2D(noise, -1, noise[:50, :50])
            autocorr_score = 1.0 - min(np.std(noise_autocorr) / 1000, 1.0)

            score = (kurtosis_score + autocorr_score) / 2
            noise_scores.append(score)

        return np.mean(noise_scores) if noise_scores else 0.5

    def _analyze_compression_artifacts(
        self, frames: List[np.ndarray]
    ) -> float:
        """
        Analyze compression artifacts.

        Multiple re-encodings leave traces that can indicate manipulation.

        Args:
            frames: List of frame arrays

        Returns:
            Score indicating normal compression (0-1)
        """
        artifact_scores = []

        for frame in frames[:10]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Analyze 8x8 block boundaries (JPEG/video compression)
            rows, cols = gray.shape
            block_diffs = []

            # Check horizontal block boundaries
            for i in range(8, rows, 8):
                diff = np.abs(gray[i, :] - gray[i - 1, :])
                block_diffs.extend(diff)

            # Check vertical block boundaries
            for j in range(8, cols, 8):
                diff = np.abs(gray[:, j] - gray[:, j - 1])
                block_diffs.extend(diff)

            if block_diffs:
                block_diff_mean = np.mean(block_diffs)

                # Compare with non-boundary pixels
                non_boundary_diffs = []
                for i in range(4, rows - 4, 8):
                    diff = np.abs(gray[i, :] - gray[i - 1, :])
                    non_boundary_diffs.extend(diff)

                if non_boundary_diffs:
                    non_boundary_mean = np.mean(non_boundary_diffs)

                    # Higher ratio of boundary to non-boundary = more artifacts
                    if non_boundary_mean > 0:
                        ratio = block_diff_mean / non_boundary_mean
                        # Ratio close to 1 is normal
                        score = max(0, 1.0 - abs(ratio - 1.0) * 0.5)
                    else:
                        score = 0.5
                else:
                    score = 0.5
            else:
                score = 0.5

            artifact_scores.append(score)

        return np.mean(artifact_scores) if artifact_scores else 0.5

    def _analyze_high_frequencies(self, frames: List[np.ndarray]) -> float:
        """
        Analyze high frequency content ratio.

        Args:
            frames: List of frame arrays

        Returns:
            Ratio of high frequency content (0-1)
        """
        high_freq_ratios = []

        for frame in frames[:10]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Apply FFT
            f_transform = fftpack.fft2(gray)
            f_shift = fftpack.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            # Calculate total energy
            total_energy = np.sum(magnitude ** 2)

            # Calculate high frequency energy (outer 30%)
            rows, cols = magnitude.shape
            center_row, center_col = rows // 2, cols // 2

            y, x = np.ogrid[:rows, :cols]
            r = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)
            max_r = np.sqrt(center_row ** 2 + center_col ** 2)

            high_freq_mask = r > (0.7 * max_r)
            high_freq_energy = np.sum(magnitude[high_freq_mask] ** 2)

            if total_energy > 0:
                ratio = high_freq_energy / total_energy
            else:
                ratio = 0

            high_freq_ratios.append(ratio)

        return np.mean(high_freq_ratios) if high_freq_ratios else 0

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool inputs."""
        return {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "description": "List of video frames to analyze (RGB format)",
                    "items": {"type": "array"},
                },
                "face_regions": {
                    "type": "array",
                    "description": "Optional face bounding boxes (x, y, w, h)",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "analyze_gan_fingerprint": {
                    "type": "boolean",
                    "description": "Whether to analyze for GAN fingerprints",
                    "default": True,
                },
                "analyze_noise": {
                    "type": "boolean",
                    "description": "Whether to analyze noise patterns",
                    "default": True,
                },
            },
            "required": ["frames"],
        }
