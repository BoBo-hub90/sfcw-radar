"""
Unit tests for the SFCW processing pipeline (RadarPipeline).

All tests run on synthetic NumPy data; no Pluto SDR or Raspberry Pi hardware is
required. The pipeline is constructed from the repository's real config so the
geometry (n_steps, range resolution, frequency grid) matches production.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make the src/ packages importable without installing the project.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from processing.pipeline import RadarPipeline  # noqa: E402

_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "radar_params.yaml")


@pytest.fixture
def pipeline() -> RadarPipeline:
    """A pipeline wired to the project's real radar config."""
    return RadarPipeline(_CONFIG_PATH)


# ----------------------------------------------------------------------- #
# phase_correction
# ----------------------------------------------------------------------- #

def test_phase_correction_identity(pipeline):
    """Dividing a sweep by itself yields all ones."""
    n = pipeline.n_steps
    rng = np.random.default_rng(0)
    s = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)

    out = pipeline.phase_correction(s, s)

    assert out.shape == (n,)
    np.testing.assert_allclose(out, np.ones(n, dtype=np.complex128), atol=1e-9)


def test_phase_correction_zero_reference_is_guarded(pipeline):
    """A zero in S_ref must not produce inf/nan; the divide is guarded."""
    n = pipeline.n_steps
    s_raw = np.ones(n, dtype=np.complex128)
    s_ref = np.ones(n, dtype=np.complex128)
    s_ref[5] = 0.0  # force a zero-division case

    out = pipeline.phase_correction(s_raw, s_ref)

    assert np.all(np.isfinite(out))


# ----------------------------------------------------------------------- #
# background_subtraction
# ----------------------------------------------------------------------- #

def test_background_subtraction_preserves_shape(pipeline):
    """Output shape matches the (n_sweeps, n_steps) input."""
    n_sweeps, n = 12, pipeline.n_steps
    rng = np.random.default_rng(1)
    S = (rng.standard_normal((n_sweeps, n))
         + 1j * rng.standard_normal((n_sweeps, n))).astype(np.complex128)

    out = pipeline.background_subtraction(S)

    assert out.shape == (n_sweeps, n)


def test_background_subtraction_removes_static_scene(pipeline):
    """A scene constant over slow time subtracts to (near) zero."""
    n_sweeps, n = pipeline.n_background_scans, pipeline.n_steps
    row = (np.arange(n) + 1j * np.arange(n)).astype(np.complex128)
    S = np.broadcast_to(row, (n_sweeps, n)).copy()  # identical every sweep

    out = pipeline.background_subtraction(S)

    np.testing.assert_allclose(out, np.zeros_like(S), atol=1e-9)


# ----------------------------------------------------------------------- #
# range_profile
# ----------------------------------------------------------------------- #

def test_range_profile_shapes(pipeline):
    """h_matrix is (n_sweeps, n_steps) and range_axis has length n_steps."""
    n_sweeps, n = 4, pipeline.n_steps
    S = np.ones((n_sweeps, n), dtype=np.complex128)

    h_matrix, range_axis = pipeline.range_profile(S)

    assert h_matrix.shape == (n_sweeps, n)
    assert range_axis.shape == (n,)


def test_range_profile_point_target_bin(pipeline):
    """A single complex tone places the IFFT peak at the expected range bin."""
    n = pipeline.n_steps
    target_bin = 40
    k = np.arange(n)
    # exp(-j 2π k b / N) inverse-transforms to a peak at index b.
    s_clean = np.exp(-1j * 2.0 * np.pi * k * target_bin / n).astype(np.complex128)

    h_matrix, range_axis = pipeline.range_profile(s_clean)
    peak_bin = int(np.argmax(np.abs(h_matrix[0])))

    assert peak_bin == target_bin


# ----------------------------------------------------------------------- #
# cfar_detect
# ----------------------------------------------------------------------- #

def test_cfar_detect_finds_target(pipeline):
    """A strong spike over a quiet floor is detected at its true range."""
    n = pipeline.n_steps
    target_bin = 40
    range_axis = np.arange(n) * pipeline.range_resolution_m

    profile = np.full(n, 0.01)
    profile[target_bin] = 10.0
    h_matrix = profile[np.newaxis, :].astype(np.complex128)

    result = pipeline.cfar_detect(h_matrix, range_axis)

    assert result["detected"] is True
    expected_range = target_bin * pipeline.range_resolution_m
    assert abs(result["target_range_m"] - expected_range) <= 0.1


def test_cfar_detect_empty_scene(pipeline):
    """A flat profile yields no detection and a sentinel range of -1.0."""
    n = pipeline.n_steps
    range_axis = np.arange(n) * pipeline.range_resolution_m
    h_matrix = np.ones((1, n), dtype=np.complex128)

    result = pipeline.cfar_detect(h_matrix, range_axis)

    assert result["detected"] is False
    assert result["target_range_m"] == -1.0


# ----------------------------------------------------------------------- #
# frequency_spectrum
# ----------------------------------------------------------------------- #

def test_frequency_spectrum_shape_and_axis(pipeline):
    """Spectrum is length n_steps and the LO axis spans 1.0–4.0 GHz."""
    n = pipeline.n_steps
    s = np.ones(n, dtype=np.complex128)

    spectrum = pipeline.frequency_spectrum(s)

    assert spectrum["magnitude_db"].shape == (n,)
    assert spectrum["freq_axis"].shape == (n,)
    assert spectrum["freq_axis"][0] == pytest.approx(1.0, abs=1e-6)
    assert spectrum["freq_axis"][-1] == pytest.approx(4.0, abs=1e-6)
