"""
SFCW radar processing pipeline (detection variant).

Turns a stack of complex frequency sweeps into a range detection result. The
stages mirror a standard stepped-frequency through-wall flow:

    phase_correction       remove the system response (S_raw / S_ref)
    background_subtraction remove static clutter (walls, direct coupling)
    range_profile          IFFT across frequency -> range, per sweep
    cfar_detect            CA-CFAR on the mean magnitude range profile

Data conventions
----------------
- A sweep S(f) is a complex vector of length `n_steps`, one phasor per LO step,
  as produced by SFCWSweep.run_sweep().
- A sweep stack S_matrix has shape (n_sweeps, n_steps): axis 0 is slow time
  (sweep index), axis 1 is fast frequency (LO step).
- A range profile h has shape (n_sweeps, n_steps): axis 0 is slow time, axis 1
  is range bin.

Range mapping
-------------
With `n_steps` LO steps spaced by `delta_f`, an n_steps-point IFFT places range
bin k at R_k = c * k / (2 * n_steps * delta_f). The bin spacing
dR = c / (2 * n_steps * delta_f) is ~0.05 m here, and the unambiguous span
R_max = c / (2 * delta_f) is ~10 m, matching the config's range_resolution_m /
max_range_m.
"""

from __future__ import annotations

import numpy as np
import yaml

from utils.logger import get_logger

log = get_logger(__name__)

# Speed of light (m/s).
C = 299_792_458.0

# CA-CFAR parameters (cells counted per side of the cell under test).
CFAR_GUARD_CELLS = 2
CFAR_TRAINING_CELLS = 8
CFAR_PFA = 1e-3


class RadarPipeline:
    """Processing chain from raw SFCW sweeps to a CFAR range detection."""

    def __init__(self, config_path: str = "config/radar_params.yaml"):
        """
        Args:
            config_path: Path to the radar YAML config.
        """
        self.config = self._load_config(config_path)
        radar = self.config["radar"]
        proc = self.config["processing"]

        self.f_start = float(radar["f_start"])
        self.f_stop = float(radar["f_stop"])
        self.n_steps = int(radar["n_steps"])
        # Use the swept grid's actual spacing so range mapping matches the data.
        self.delta_f = (self.f_stop - self.f_start) / (self.n_steps - 1)

        self.n_background_scans = int(proc["n_background_scans"])

        # Range bin spacing and unambiguous span for an n_steps-point IFFT.
        self.range_resolution_m = C / (2.0 * self.n_steps * self.delta_f)
        self.r_max_m = C / (2.0 * self.delta_f)

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Stage 1 — phase / system-response correction
    # ------------------------------------------------------------------ #

    def phase_correction(
        self, S_raw: np.ndarray, S_ref: np.ndarray
    ) -> np.ndarray:
        """
        Divide out the system response: S_corrected = S_raw / S_ref.

        The element-wise complex division removes shared cable/PLL/IQ-imbalance
        terms present identically in both paths, leaving the antenna-path
        channel response.

        Args:
            S_raw: Measurement-path sweeps, shape (n_steps,) or (n_sweeps, n_steps).
            S_ref: Reference-path sweeps, same shape as S_raw.

        Returns:
            Complex array of the same shape as the inputs.
        """
        S_raw = np.asarray(S_raw)
        S_ref = np.asarray(S_ref)
        if S_raw.shape != S_ref.shape:
            raise ValueError(
                f"S_raw {S_raw.shape} and S_ref {S_ref.shape} must match"
            )

        # Guard against division by exact zeros in the reference.
        denom = S_ref.astype(np.complex128).copy()
        zeros = denom == 0
        if np.any(zeros):
            log.warning("phase_correction: %d zero reference sample(s) guarded",
                        int(np.count_nonzero(zeros)))
            denom[zeros] = np.finfo(np.float64).tiny
        return S_raw.astype(np.complex128) / denom

    # ------------------------------------------------------------------ #
    # Stage 2 — background / clutter subtraction
    # ------------------------------------------------------------------ #

    def background_subtraction(self, S_corrected: np.ndarray) -> np.ndarray:
        """
        Subtract the static clutter map estimated from the last sweeps.

        S_clean(f, t) = S_corrected(f, t) - mean_t'{ last n_background_scans }.
        Static reflectors (walls, direct TX->RX coupling) are constant across
        slow time, so their per-frequency mean is removed, leaving moving-target
        contributions.

        Args:
            S_corrected: Complex array of shape (n_sweeps, n_steps).

        Returns:
            Complex array of the same shape.
        """
        S_corrected = np.asarray(S_corrected, dtype=np.complex128)
        if S_corrected.ndim != 2:
            raise ValueError(
                "background_subtraction expects a 2-D (n_sweeps, n_steps) array"
            )

        n_sweeps = S_corrected.shape[0]
        n_bg = min(self.n_background_scans, n_sweeps)
        background = S_corrected[-n_bg:, :].mean(axis=0, keepdims=True)
        return S_corrected - background

    # ------------------------------------------------------------------ #
    # Stage 3 — range profile (IFFT across frequency)
    # ------------------------------------------------------------------ #

    def range_profile(
        self, S_clean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform frequency-domain sweeps into range profiles.

        A Hanning window is applied across the frequency axis to suppress range
        sidelobes, then an IFFT maps frequency steps to range bins.

        Args:
            S_clean: Complex array of shape (n_sweeps, n_steps) (a 1-D
                (n_steps,) array is accepted and treated as a single sweep).

        Returns:
            (h_matrix, range_axis):
              h_matrix   : complex array of shape (n_sweeps, n_steps), the range
                           profile per sweep (slow time x range bin).
              range_axis : float array of shape (n_steps,), 0..R_max in meters.
        """
        S_clean = np.atleast_2d(np.asarray(S_clean, dtype=np.complex128))
        n_freq = S_clean.shape[1]

        window = np.hanning(n_freq)
        h_matrix = np.fft.ifft(S_clean * window[np.newaxis, :], axis=1)

        range_axis = np.arange(n_freq) * self.range_resolution_m
        return h_matrix, range_axis

    # ------------------------------------------------------------------ #
    # Stage 4 — CA-CFAR detection
    # ------------------------------------------------------------------ #

    def cfar_detect(
        self, h_matrix: np.ndarray, range_axis: np.ndarray
    ) -> dict:
        """
        1-D Cell-Averaging CFAR along range on the mean magnitude profile.

        The detector forms P(R) = mean_t |h(R, t)| and, for each cell under test
        (CUT), estimates the local noise level from training cells on both sides
        (separated from the CUT by guard cells). The adaptive threshold is
        alpha * noise, where alpha = N * (Pfa^(-1/N) - 1) for N training cells,
        the standard CA-CFAR factor for a target false-alarm rate.

        Edge cells with no available training window keep an infinite threshold
        (never detect). The threshold factor is computed from the number of
        training cells actually used, so the false-alarm rate stays consistent
        near the array edges.

        Note: CA-CFAR theory assumes square-law (power) samples; here it is
        applied to magnitude, the common practical approximation.

        Args:
            h_matrix: Range profiles, shape (n_sweeps, n_steps).
            range_axis: Range axis in meters, shape (n_steps,).

        Returns:
            dict with keys:
              detected       : bool, True if any cell exceeds its threshold.
              target_range_m : float, range of the strongest detected cell, or
                               -1.0 if there is no detection.
              cfar_threshold : 1-D array (n_steps,), the adaptive threshold.
              range_profile  : 1-D array (n_steps,), the mean |h| profile.
        """
        h_matrix = np.atleast_2d(np.asarray(h_matrix))
        profile = np.abs(h_matrix).mean(axis=0)
        n = profile.size

        threshold = np.full(n, np.inf)
        guard = CFAR_GUARD_CELLS
        train = CFAR_TRAINING_CELLS

        for i in range(n):
            # Leading and lagging training windows, clipped to the array.
            lead = profile[max(0, i - guard - train): max(0, i - guard)]
            lag = profile[i + guard + 1: i + guard + 1 + train]
            training = np.concatenate((lead, lag))

            n_train = training.size
            if n_train == 0:
                continue  # leave threshold at +inf -> no detection

            noise = training.mean()
            alpha = n_train * (CFAR_PFA ** (-1.0 / n_train) - 1.0)
            threshold[i] = alpha * noise

        detections = profile > threshold
        if np.any(detections):
            det_idx = np.flatnonzero(detections)
            peak = det_idx[np.argmax(profile[det_idx])]
            detected = True
            target_range_m = float(range_axis[peak])
        else:
            detected = False
            target_range_m = -1.0

        return {
            "detected": detected,
            "target_range_m": target_range_m,
            "cfar_threshold": threshold,
            "range_profile": profile,
        }

    # ------------------------------------------------------------------ #
    # Diagnostic — frequency-domain spectrum
    # ------------------------------------------------------------------ #

    def frequency_spectrum(self, S_corrected: np.ndarray) -> dict:
        """
        Magnitude spectrum of the corrected sweep across the LO frequency grid.

        Useful for diagnostics: it shows how flat the system response is after
        correction and exposes dropouts or strong reflectors per LO step.

        Args:
            S_corrected: Complex array of shape (n_steps,) or
                (n_sweeps, n_steps). A 2-D input is averaged over sweeps first.

        Returns:
            dict with keys:
              freq_axis    : 1-D array (n_steps,), LO frequencies in GHz.
              magnitude_db : 1-D array (n_steps,), 20*log10(|S| + 1e-12) in dB.
        """
        S_corrected = np.asarray(S_corrected, dtype=np.complex128)
        if S_corrected.ndim == 2:
            S_corrected = S_corrected.mean(axis=0)

        magnitude_db = 20.0 * np.log10(np.abs(S_corrected) + 1e-12)
        freq_axis = np.linspace(self.f_start, self.f_stop, self.n_steps) / 1e9
        return {"freq_axis": freq_axis, "magnitude_db": magnitude_db}

    # ------------------------------------------------------------------ #
    # Stage 5 — full pipeline
    # ------------------------------------------------------------------ #

    def run(self, S_raw: np.ndarray, S_ref: np.ndarray) -> dict:
        """
        Run the full chain and return the CFAR detection result.

        phase_correction -> background_subtraction -> range_profile ->
        cfar_detect.

        Args:
            S_raw: Measurement-path sweep stack, shape (n_sweeps, n_steps).
            S_ref: Reference-path sweep stack, same shape.

        Returns:
            The detection dict from cfar_detect().
        """
        S_corrected = self.phase_correction(S_raw, S_ref)
        S_clean = self.background_subtraction(S_corrected)
        h_matrix, range_axis = self.range_profile(S_clean)
        return self.cfar_detect(h_matrix, range_axis)

    def run_no_bg(self, S_raw: np.ndarray, S_ref: np.ndarray) -> dict:
        """
        Run the chain without background subtraction.

        Identical to run() but the static-clutter removal stage is skipped:
        phase_correction -> range_profile -> cfar_detect. Useful when there is
        no stable clutter background to estimate (e.g. very few sweeps, or a
        single moving setup) and the subtraction would otherwise remove signal.

        Args:
            S_raw: Measurement-path sweep stack, shape (n_sweeps, n_steps).
            S_ref: Reference-path sweep stack, same shape.

        Returns:
            The detection dict from cfar_detect().
        """
        S_corrected = self.phase_correction(S_raw, S_ref)
        h_matrix, range_axis = self.range_profile(S_corrected)
        return self.cfar_detect(h_matrix, range_axis)
