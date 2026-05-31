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

from collections import deque

import numpy as np
import yaml

from utils.logger import get_logger

log = get_logger(__name__)

# Speed of light (m/s).
C = 299_792_458.0

# CA-CFAR defaults (cells counted per side of the cell under test). Used as a
# fallback when the config has no `cfar` section.
CFAR_GUARD_CELLS = 2
CFAR_TRAINING_CELLS = 8
CFAR_PFA = 1e-3

# Number of near-range bins to exclude from detection. The first few bins are
# dominated by TX->RX direct-coupling leakage (a strong, static return at ~0 m)
# that would otherwise mask real targets; they are blanked before peak finding
# and CFAR threshold crossing.
NEAR_RANGE_EXCLUDE_BINS = 10

# OS-CFAR order-statistic rank as a fraction of the training-cell count. The
# training cells are sorted ascending and the value at this percentile is used
# as the local noise estimate, which is robust when several cells contain
# target energy (unlike the mean used by CA-CFAR).
OS_CFAR_RANK_FRACTION = 0.75


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

        # Doppler / micro-Doppler parameters. The sweep period (slow-time
        # sampling interval) sets the velocity scaling; the motion threshold is
        # the |mean velocity| above which a target is flagged as moving.
        self.sweep_period_s = float(proc.get("sweep_period_s", 1.5))
        self.doppler_motion_threshold_ms = float(
            proc.get("doppler_motion_threshold_ms", 0.05)
        )

        # Minimum peak-to-mean ratio of the range profile below which the scene
        # is treated as flat (no target) and CFAR is skipped. Lowered for the
        # direct-coupling-dominated bring-up case (a leaky front end pins the
        # ratio near ~1.8 regardless of scene).
        self.peak_to_mean_threshold = float(
            proc.get("peak_to_mean_threshold", 1.3)
        )

        # Energy-detector threshold: the current/background mean-power ratio
        # above which a target (e.g. a person entering the scene) is declared.
        self.energy_threshold = float(proc.get("energy_threshold", 1.5))

        # Energy-variance threshold: the std deviation of the recent energy
        # ratios above which motion (a *changing* scene) is declared. This keys
        # on change rather than absolute level, so a static person/clutter that
        # merely raises the level does not trip it but movement does.
        self.energy_variance_threshold = float(
            proc.get("energy_variance_threshold", 0.05)
        )

        # CA-CFAR tuning from the optional `cfar` config section (falls back to
        # the module defaults when the section or a key is absent).
        cfar = self.config.get("cfar", {})
        self.cfar_guard_cells = int(cfar.get("guard_cells", CFAR_GUARD_CELLS))
        self.cfar_training_cells = int(
            cfar.get("training_cells", CFAR_TRAINING_CELLS)
        )
        self.cfar_pfa = float(cfar.get("false_alarm_rate", CFAR_PFA))

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

        # Remove the per-sweep DC offset (zero-frequency LO leakage) before the
        # IFFT so it does not pile up into the zero-range bin.
        S_clean = S_clean - np.mean(S_clean, axis=-1, keepdims=True)

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
        1-D Order-Statistics CFAR (OS-CFAR) along range on the mean profile.

        The detector forms P(R) = mean_t |h(R, t)| and, for each cell under test
        (CUT), gathers training cells on both sides (separated from the CUT by
        guard cells). Instead of averaging them (CA-CFAR), the training cells are
        sorted and the value at the OS_CFAR_RANK_FRACTION percentile is taken as
        the local noise estimate. This order-statistic is robust when several
        training cells already contain target energy, which is exactly the
        cluttered, direct-coupling-dominated case here. The adaptive threshold is
        alpha * noise, with alpha = N * (Pfa^(-1/N) - 1) for N training cells.

        Two extra guards are applied for the leaky bring-up front end:
          - Peak-to-mean guard: if the whole profile is nearly flat
            (peak / mean <= peak_to_mean_threshold) the scene carries no target,
            so CFAR is skipped and no detection is reported.
          - Near-range exclusion: the first NEAR_RANGE_EXCLUDE_BINS bins hold
            TX->RX direct-coupling leakage and are removed from both peak finding
            and threshold crossing.

        Edge cells with no available training window keep an infinite threshold
        (never detect). The threshold factor is computed from the number of
        training cells actually used, so the false-alarm rate stays consistent
        near the array edges.

        Note: CFAR theory assumes square-law (power) samples; here it is applied
        to magnitude, the common practical approximation.

        Args:
            h_matrix: Range profiles, shape (n_sweeps, n_steps).
            range_axis: Range axis in meters, shape (n_steps,).

        Returns:
            dict with keys:
              detected       : bool, True if any valid cell exceeds its threshold.
              target_range_m : float, range of the strongest detected cell, or
                               -1.0 if there is no detection.
              cfar_threshold : 1-D array (n_steps,), the adaptive threshold.
              range_profile  : 1-D array (n_steps,), the mean |h| profile.
              peak_to_mean   : float, max(profile) / mean(profile), a coarse
                               "how much does the peak stand out" measure.
        """
        h_matrix = np.atleast_2d(np.asarray(h_matrix))
        profile = np.abs(h_matrix).mean(axis=0)
        n = profile.size

        # Peak-to-mean guard: a flat profile (every bin similar) carries no
        # target, so skip CFAR entirely and report no detection. Only when the
        # peak stands clearly above the average is the adaptive CFAR worthwhile.
        mean_profile = float(np.mean(profile))
        peak = float(np.max(profile))
        peak_to_mean = peak / mean_profile if mean_profile > 0 else 0.0
        if peak_to_mean <= self.peak_to_mean_threshold:
            return {
                "detected": False,
                "target_range_m": -1.0,
                "cfar_threshold": np.full(n, np.inf),
                "range_profile": profile,
                "peak_to_mean": peak_to_mean,
            }

        # Near-range exclusion: blank the direct-coupling zone so only bins from
        # NEAR_RANGE_EXCLUDE_BINS onward are eligible for detection.
        valid_mask = np.zeros(n, dtype=bool)
        valid_mask[NEAR_RANGE_EXCLUDE_BINS:] = True

        threshold = np.full(n, np.inf)
        guard = self.cfar_guard_cells
        train = self.cfar_training_cells
        pfa = self.cfar_pfa

        for i in range(n):
            # Only compute a threshold for cells that are eligible to detect.
            if not valid_mask[i]:
                continue

            # Leading and lagging training windows, clipped to the array.
            lead = profile[max(0, i - guard - train): max(0, i - guard)]
            lag = profile[i + guard + 1: i + guard + 1 + train]
            training = np.concatenate((lead, lag))

            n_train = training.size
            if n_train == 0:
                continue  # leave threshold at +inf -> no detection

            # OS-CFAR noise estimate: the order statistic at the configured
            # rank. Sort ascending and index the percentile; clamp the rank to
            # the cells actually available so edge windows stay in bounds.
            sorted_training = np.sort(training)
            k = min(int(OS_CFAR_RANK_FRACTION * train), n_train - 1)
            noise = sorted_training[k]

            alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
            threshold[i] = alpha * noise

        # Threshold crossing, restricted to the valid (non-near-range) bins.
        detections = (profile > threshold) & valid_mask
        if np.any(detections):
            det_idx = np.flatnonzero(detections)
            peak_bin = det_idx[np.argmax(profile[det_idx])]
            detected = True
            target_range_m = float(range_axis[peak_bin])
        else:
            detected = False
            target_range_m = -1.0

        return {
            "detected": detected,
            "target_range_m": target_range_m,
            "cfar_threshold": threshold,
            "range_profile": profile,
            "peak_to_mean": peak_to_mean,
        }

    # ------------------------------------------------------------------ #
    # Stage 4c — energy detector (CFAR alternative)
    # ------------------------------------------------------------------ #

    def energy_detect(
        self,
        S_matrix: np.ndarray,
        background_matrix: np.ndarray,
        ratio_history: deque | None = None,
    ) -> dict:
        """
        Motion detector from the *variance* of the energy ratio over time.

        Rather than localising a target in range (CFAR), this tracks the total
        received power relative to the empty-scene warmup baseline and keys on
        how much that ratio *changes*, not its absolute level. A static scene
        (even a leaky front end or a stationary person that merely raises the
        level) holds a steady ratio, so its sliding-window std deviation stays
        low; real motion makes the ratio fluctuate and the std rise above
        `energy_variance_threshold`. This is robust where CFAR fails under heavy
        direct coupling.

        The current ratio is computed from this frame's mean power over the
        baseline, then appended to `ratio_history` (a caller-owned rolling
        window). Detection is std(ratio_history) > energy_variance_threshold.

        Power is averaged over every element of each stack (all sweeps and all
        LO steps), so the ratio is independent of how many sweeps each stack
        holds.

        Args:
            S_matrix: Current sweeps, shape (n_sweeps, n_steps).
            background_matrix: Empty-scene warmup sweeps, shape
                (n_background_scans, n_steps).
            ratio_history: Caller-owned rolling buffer (e.g. deque(maxlen=5)) of
                recent energy ratios. The freshly computed ratio is appended to
                it in place; its std deviation drives the decision. When None, a
                fresh single-element window is used (std 0 -> no detection).

        Returns:
            dict with keys:
              detected             : bool, energy_std > energy_variance_threshold.
              energy_ratio         : float, current / background mean power.
              energy_std           : float, std deviation of the recent ratios.
              current_energy_db    : float, 10*log10(current mean power).
              background_energy_db : float, 10*log10(background mean power).
        """
        S_matrix = np.asarray(S_matrix)
        background_matrix = np.asarray(background_matrix)

        current_energy = float(np.mean(np.abs(S_matrix) ** 2))
        background_energy = float(np.mean(np.abs(background_matrix) ** 2))

        # Guard against an all-zero / empty background to avoid div-by-zero.
        if background_energy > 0:
            energy_ratio = current_energy / background_energy
        else:
            log.warning("energy_detect: background energy is zero; "
                        "reporting ratio as inf")
            energy_ratio = float("inf")

        # Append this frame's ratio to the caller-owned rolling window and take
        # the std deviation over it. A short window (a few frames) makes the
        # detector responsive to onset/offset of motion.
        if ratio_history is None:
            ratio_history = deque([energy_ratio], maxlen=5)
        else:
            ratio_history.append(energy_ratio)

        # np.std on a window that still contains inf (zero-background guard)
        # would be nan; treat that as "no usable estimate" -> std 0.
        finite = [r for r in ratio_history if np.isfinite(r)]
        energy_std = float(np.std(finite)) if len(finite) > 1 else 0.0

        detected = bool(energy_std > self.energy_variance_threshold)

        current_energy_db = 10.0 * np.log10(current_energy + 1e-12)
        background_energy_db = 10.0 * np.log10(background_energy + 1e-12)

        return {
            "detected": detected,
            "energy_ratio": energy_ratio,
            "energy_std": energy_std,
            "current_energy_db": float(current_energy_db),
            "background_energy_db": float(background_energy_db),
        }

    # ------------------------------------------------------------------ #
    # Stage 4b — Doppler / micro-Doppler (slow-time phase)
    # ------------------------------------------------------------------ #

    def doppler_process(
        self, h_matrix: np.ndarray, range_axis: np.ndarray
    ) -> dict:
        """
        Estimate radial velocity and a micro-Doppler signature at the target bin.

        Where cfar_detect collapses slow time to a single magnitude profile,
        this looks *along* slow time at the strongest range bin and reads the
        target's motion out of the phase evolution there:

          1. Pick the target bin as the peak of the mean |h| range profile
             (the same cell cfar_detect reports).
          2. theta(t) = arctan2(Im, Re) of h at that bin, then np.unwrap to
             remove 2*pi jumps so the phase tracks continuously.
          3. A round-trip of range R contributes a phase 4*pi*R/lambda, so the
             radial velocity is v(t) = -(lambda / 4*pi) * dtheta/dt, evaluated
             at the carrier wavelength lambda = c / f0 with f0 the sweep's
             centre frequency. The derivative is taken over the slow-time axis
             (spacing = sweep_period_s) and lightly smoothed.
          4. A short-time Fourier transform of the complex bin signal gives the
             micro-Doppler spectrogram: a walking person shows a steady torso
             line plus swinging-limb sidebands, separating it from a rigid body.

        Args:
            h_matrix: Range profiles, shape (n_sweeps, n_steps).
            range_axis: Range axis in meters, shape (n_steps,).

        Returns:
            dict with keys:
              velocity_ms         : 1-D array (n_sweeps,), smoothed radial
                                    velocity per slow-time sample.
              mean_velocity_ms    : float, mean of velocity_ms.
              doppler_spectrogram : 2-D array (n_v, n_frames), micro-Doppler
                                    magnitude (|STFT|) at the target bin.
              doppler_v_axis      : 1-D array (n_v,), velocity axis (m/s) of the
                                    spectrogram rows (ascending).
              doppler_t_axis      : 1-D array (n_frames,), slow-time centre of
                                    each STFT frame in seconds.
              target_range_m      : float, range of the analysed bin (meters).
              moving              : bool, |mean_velocity_ms| exceeds the config
                                    motion threshold.
        """
        from scipy.ndimage import uniform_filter1d

        h_matrix = np.atleast_2d(np.asarray(h_matrix))
        n_sweeps = h_matrix.shape[0]

        # --- Step 1: target bin = peak of the mean magnitude range profile ---
        profile = np.abs(h_matrix).mean(axis=0)
        target_bin = int(np.argmax(profile))
        target_range_m = float(range_axis[target_bin])

        # Complex slow-time signal at the target bin.
        bin_signal = h_matrix[:, target_bin]

        # --- Step 2: unwrapped phase along slow time ---
        theta = np.unwrap(np.arctan2(bin_signal.imag, bin_signal.real))

        # --- Step 3: velocity from the phase derivative ---
        f0 = 0.5 * (self.f_start + self.f_stop)
        lam = C / f0
        dt = self.sweep_period_s

        if n_sweeps >= 2:
            dphidt = np.gradient(theta, dt)
        else:
            dphidt = np.zeros_like(theta)
        velocity = -(lam / (4.0 * np.pi)) * dphidt

        # Smooth the derivative noise. Clamp the window to the data length so a
        # short slow-time axis does not over-smooth or error out.
        smooth_size = min(5, max(1, n_sweeps))
        velocity = uniform_filter1d(velocity, size=smooth_size)
        mean_velocity_ms = float(np.mean(velocity))

        # --- Step 4: micro-Doppler STFT of the complex bin signal ---
        nperseg = max(1, min(16, n_sweeps // 2))
        if nperseg >= 2 and n_sweeps > nperseg:
            hop = max(1, nperseg // 4)
            win = np.hanning(nperseg)
            starts = list(range(0, n_sweeps - nperseg + 1, hop))
            cols = [
                np.fft.fftshift(np.fft.fft(bin_signal[i:i + nperseg] * win))
                for i in starts
            ]
            spec = np.abs(np.array(cols).T)  # (nperseg, n_frames)

            freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, dt))
            # Doppler frequency -> radial velocity, reordered ascending.
            v_axis = (-lam * freqs / 2.0)[::-1]
            spec = spec[::-1, :]
            t_axis = (np.array(starts) + nperseg / 2.0) * dt
        else:
            # Too few sweeps for a meaningful STFT; return empty arrays.
            spec = np.empty((0, 0))
            v_axis = np.empty(0)
            t_axis = np.empty(0)

        moving = bool(abs(mean_velocity_ms) > self.doppler_motion_threshold_ms)

        return {
            "velocity_ms": velocity,
            "mean_velocity_ms": mean_velocity_ms,
            "doppler_spectrogram": spec,
            "doppler_v_axis": v_axis,
            "doppler_t_axis": t_axis,
            "target_range_m": target_range_m,
            "moving": moving,
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

    def run(
        self, S_raw: np.ndarray, S_ref: np.ndarray, doppler: bool = False
    ) -> dict:
        """
        Run the full chain and return the CFAR detection result.

        phase_correction -> background_subtraction -> range_profile ->
        cfar_detect.

        Args:
            S_raw: Measurement-path sweep stack, shape (n_sweeps, n_steps).
            S_ref: Reference-path sweep stack, same shape.
            doppler: When True, also run doppler_process() on the same range
                profiles and merge its velocity / micro-Doppler keys into the
                returned dict.

        Returns:
            The detection dict from cfar_detect(), optionally extended with the
            doppler_process() keys when `doppler` is True.
        """
        S_corrected = self.phase_correction(S_raw, S_ref)
        S_clean = self.background_subtraction(S_corrected)
        h_matrix, range_axis = self.range_profile(S_clean)
        result = self.cfar_detect(h_matrix, range_axis)
        if doppler:
            result.update(self.doppler_process(h_matrix, range_axis))
        return result

    def run_no_bg(
        self, S_raw: np.ndarray, S_ref: np.ndarray, doppler: bool = False
    ) -> dict:
        """
        Run the chain without background subtraction.

        Identical to run() but the static-clutter removal stage is skipped:
        phase_correction -> range_profile -> cfar_detect. Useful when there is
        no stable clutter background to estimate (e.g. very few sweeps, or a
        single moving setup) and the subtraction would otherwise remove signal.

        Args:
            S_raw: Measurement-path sweep stack, shape (n_sweeps, n_steps).
            S_ref: Reference-path sweep stack, same shape.
            doppler: When True, also run doppler_process() on the same range
                profiles and merge its velocity / micro-Doppler keys into the
                returned dict.

        Returns:
            The detection dict from cfar_detect(), optionally extended with the
            doppler_process() keys when `doppler` is True.
        """
        S_corrected = self.phase_correction(S_raw, S_ref)
        h_matrix, range_axis = self.range_profile(S_corrected)
        result = self.cfar_detect(h_matrix, range_axis)
        if doppler:
            result.update(self.doppler_process(h_matrix, range_axis))
        return result
