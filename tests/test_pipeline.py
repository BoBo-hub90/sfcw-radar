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


# ----------------------------------------------------------------------- #
# level_detect
# ----------------------------------------------------------------------- #

def test_level_detect_above_margin(pipeline):
    """A peak 5 dB over baseline clears the 1.5 dB margin and is detected."""
    # signal 5 dB above baseline, margin is 1.5 → detected
    result = pipeline.level_detect(signal_db=19.0, baseline_db=14.0)
    assert result["detected"] is True
    assert result["margin_above_baseline_db"] == 5.0


def test_level_detect_below_margin(pipeline):
    """A peak only 1 dB over baseline stays under the margin: no detection."""
    # signal only 1 dB above baseline → not detected
    result = pipeline.level_detect(signal_db=15.0, baseline_db=14.0)
    assert result["detected"] is False


def test_level_detect_returns_all_keys(pipeline):
    """level_detect returns every documented key."""
    result = pipeline.level_detect(signal_db=18.0, baseline_db=14.0)
    for key in ["detected", "signal_db", "baseline_db",
                "margin_above_baseline_db"]:
        assert key in result


# ----------------------------------------------------------------------- #
# energy_detect
# ----------------------------------------------------------------------- #

def test_energy_detect_steady_scene_no_motion(pipeline):
    """Identical-energy sweeps give a low ratio std: no motion detected."""
    # feed identical-energy sweeps → low std → not detected
    from collections import deque
    history = deque(maxlen=5)
    S = np.ones((1, 201), dtype=complex)
    bg = np.ones((10, 201), dtype=complex)
    for _ in range(5):
        result = pipeline.energy_detect(S, bg, history)
    assert result["detected"] is False
    assert result["energy_std"] < 0.05


def test_energy_detect_fluctuating_scene_motion(pipeline):
    """Varying-energy sweeps give a high ratio std: motion detected."""
    # feed varying-energy sweeps → high std → detected
    from collections import deque
    history = deque(maxlen=5)
    bg = np.ones((10, 201), dtype=complex)
    for amp in [1.0, 3.0, 0.5, 4.0, 2.0]:
        S = np.ones((1, 201), dtype=complex) * amp
        result = pipeline.energy_detect(S, bg, history)
    assert result["detected"] is True
    assert result["energy_std"] > 0.05


def test_energy_detect_returns_all_keys(pipeline):
    """energy_detect returns every documented key."""
    from collections import deque
    history = deque(maxlen=5)
    S = np.ones((1, 201), dtype=complex)
    bg = np.ones((10, 201), dtype=complex)
    result = pipeline.energy_detect(S, bg, history)
    for key in ["detected", "energy_ratio", "energy_std",
                "current_energy_db", "background_energy_db"]:
        assert key in result


# ----------------------------------------------------------------------- #
# doppler_process
# ----------------------------------------------------------------------- #

def test_doppler_process_output_keys(pipeline):
    """doppler_process returns every documented key for a moving target."""
    # synthetic moving target across sweeps
    n_sweeps, n_steps = 20, 201
    h = np.random.randn(n_sweeps, n_steps) + 1j*np.random.randn(n_sweeps, n_steps)
    range_axis = np.linspace(0, 10, n_steps)
    result = pipeline.doppler_process(h, range_axis)
    for key in ["velocity_ms", "mean_velocity_ms", "doppler_spectrogram",
                "doppler_v_axis", "doppler_t_axis", "target_range_m", "moving"]:
        assert key in result


def test_doppler_process_velocity_length(pipeline):
    """The per-sweep velocity series has one sample per slow-time sweep."""
    n_sweeps, n_steps = 20, 201
    h = np.random.randn(n_sweeps, n_steps) + 1j*np.random.randn(n_sweeps, n_steps)
    range_axis = np.linspace(0, 10, n_steps)
    result = pipeline.doppler_process(h, range_axis)
    assert len(result["velocity_ms"]) == n_sweeps
