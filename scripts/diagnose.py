#!/usr/bin/env python3
"""
diagnose.py — SFCW radar detection-chain diagnostic.

Runs a sequence of controlled, self-checking tests against the live
hardware + software chain (Pluto SDR -> sweep acquisition -> range processing)
and prints a clear PASS/FAIL result for each, so the radar can be *verified*
rather than guessed about (e.g. "is it really seeing anything, or is it all
TX->RX coupling?").

Test sequence (each prints its own header and verdict):
  1. RX level check      — is a signal present at the ADC, and not saturated?
  2. Single-sweep sanity — does the 201-step swept response actually vary?
  3. Range profile       — where does the energy concentrate in range?
  4. Stability (empty)   — is the empty-scene energy steady over ~15 s?
  5. Target response     — does inserting a metal reflector change the return?

A final summary collates the PASS/FAIL verdicts and gives a plain-language
conclusion about whether the chain is working.

This script needs the real ADALM-Pluto connected: it tunes the LO, transmits a
CW tone and captures RX buffers, so it cannot run on a development laptop
without the radio attached.

Usage:
    python scripts/diagnose.py
    python scripts/diagnose.py --ip ip:192.168.2.1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Make the src/ packages importable regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from pluto.device import PlutoDevice          # noqa: E402
from acquisition.sweep import SFCWSweep        # noqa: E402
from processing.pipeline import RadarPipeline  # noqa: E402

CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "radar_params.yaml")

# RF reference switch GPIO pin (BCM). The diagnostic only uses the measurement
# path (run_sweep), so off-Pi the switch falls back to its no-op stub.
SWITCH_GPIO_PIN = 17

# --- Test 1: RX level check ---
RX_LEVEL_MIN = 50.0       # below this the return is effectively absent
RX_LEVEL_MAX = 1800.0     # above this the 12-bit ADC is heading into clipping
RX_LEVEL_INDICES = [0, 100, 200]  # first / middle / last LO step

# --- Test 3: range profile ---
RANGE_TOP_N = 5

# --- Test 4: stability ---
STABILITY_SWEEPS = 10
STABILITY_TOTAL_S = 15.0
STABILITY_REL_STD_MAX = 0.10   # PASS if energy std stays under 10% of the mean

# --- Test 5: target response ---
TARGET_SWEEPS = 5
TARGET_RESPOND_HI = 1.2
TARGET_RESPOND_LO = 0.8


# ----------------------------------------------------------------------- #
# Small formatting / metric helpers
# ----------------------------------------------------------------------- #

def _print_header(number: int, title: str) -> None:
    """Print a clear, numbered section banner."""
    print()
    print("=" * 64)
    print(f"  TEST {number}: {title}")
    print("=" * 64)


def _verdict(passed: bool | None) -> str:
    """Map a tri-state result to a label (None == skipped)."""
    if passed is None:
        return "SKIP"
    return "PASS" if passed else "FAIL"


def _sweep_metrics(s_raw: np.ndarray, pipeline: RadarPipeline) -> tuple[float, float]:
    """
    Reduce one sweep to its two scalar diagnostics.

    Returns:
        (total_energy, peak_magnitude):
          total_energy   : sum |S|^2 over the 201 frequency steps.
          peak_magnitude : max |h| of the sweep's range profile.
    """
    energy = float(np.sum(np.abs(s_raw) ** 2))
    h_matrix, _ = pipeline.range_profile(s_raw)
    peak_magnitude = float(np.abs(h_matrix[0]).max())
    return energy, peak_magnitude


def _capture_mean_metrics(
    sweep: SFCWSweep, pipeline: RadarPipeline, n: int
) -> tuple[float, float]:
    """Capture n sweeps and return the mean (total_energy, peak_magnitude)."""
    energies, peaks = [], []
    for _ in range(n):
        s_raw = sweep.run_sweep()
        energy, peak = _sweep_metrics(s_raw, pipeline)
        energies.append(energy)
        peaks.append(peak)
    return float(np.mean(energies)), float(np.mean(peaks))


# ----------------------------------------------------------------------- #
# Test 1 — RX level check
# ----------------------------------------------------------------------- #

def test_rx_level(device: PlutoDevice) -> bool:
    """
    Sample the RX ADC under a live CW carrier at three LO steps.

    PASS when the peak |IQ| at every checked step sits in [RX_LEVEL_MIN,
    RX_LEVEL_MAX]: high enough that a return is actually present, low enough
    that the 12-bit converter is not saturating.
    """
    _print_header(1, "RX LEVEL CHECK")
    print("Starting the CW carrier and sampling the RX ADC at 3 LO steps.")

    freqs = device.frequency_vector()
    n = device.n_steps
    all_ok = True

    device.start_cw()
    try:
        for idx in RX_LEVEL_INDICES:
            step = min(idx, n - 1)          # guard if n_steps < 201
            device.recall_profile(step)
            time.sleep(0.01)               # let the LO hop settle before sampling
            lvl = device.check_rx_level()
            ok = RX_LEVEL_MIN <= lvl["max_magnitude"] <= RX_LEVEL_MAX
            all_ok = all_ok and ok
            print(
                f"  step {step:3d} ({freqs[step] / 1e9:.3f} GHz): "
                f"mean={lvl['mean_magnitude']:7.1f}  "
                f"max={lvl['max_magnitude']:7.1f}  "
                f"saturation={lvl['saturation_pct'] * 100:5.2f}%  "
                f"[{'ok' if ok else 'out of range'}]"
            )
    finally:
        device.stop_cw()

    print(f"\n  Expect peak |IQ| in [{RX_LEVEL_MIN:.0f}, {RX_LEVEL_MAX:.0f}] "
          f"(signal present, not saturated).")
    print(f"  RESULT: {_verdict(all_ok)}")
    return all_ok


# ----------------------------------------------------------------------- #
# Test 2 — single-sweep sanity
# ----------------------------------------------------------------------- #

def test_single_sweep(sweep: SFCWSweep) -> tuple[np.ndarray, bool]:
    """
    Run one full sweep and confirm the swept response is not flat/dead.

    PASS when the per-step magnitude has non-zero spread (std > 0), i.e. the LO
    steps see a real, frequency-dependent channel rather than a constant value.
    Returns the captured sweep so the next test can reuse it.
    """
    _print_header(2, "SINGLE SWEEP SANITY")
    print("Running one 201-step sweep and inspecting the raw response.")

    s_raw = sweep.run_sweep()
    mag = np.abs(s_raw)
    mag_std = float(np.std(mag))
    nonzero = int(np.count_nonzero(s_raw))

    print(f"  steps captured : {s_raw.size}")
    print(f"  |S| min        : {mag.min():.3f}")
    print(f"  |S| max        : {mag.max():.3f}")
    print(f"  |S| mean       : {mag.mean():.3f}")
    print(f"  |S| std        : {mag_std:.3f}")
    print(f"  non-zero steps : {nonzero}/{s_raw.size}")

    passed = mag_std > 0.0
    print("\n  Expect |S| to vary across steps (std > 0) — a real frequency "
          "response, not a flat/dead signal.")
    print(f"  RESULT: {_verdict(passed)}")
    return s_raw, passed


# ----------------------------------------------------------------------- #
# Test 3 — range profile check
# ----------------------------------------------------------------------- #

def test_range_profile(pipeline: RadarPipeline, s_raw: np.ndarray) -> bool:
    """
    IFFT the sweep to range and report where the energy concentrates.

    Prints the strongest RANGE_TOP_N range bins (with their distances) and the
    peak-to-mean ratio. PASS when a peak stands out at all (peak/mean > 1),
    i.e. the energy is localised in range rather than perfectly flat.
    """
    _print_header(3, "RANGE PROFILE CHECK")
    print("IFFT across frequency -> range; showing where the energy sits.")

    h_matrix, range_axis = pipeline.range_profile(s_raw)
    profile = np.abs(h_matrix).mean(axis=0)
    mean_profile = float(profile.mean())
    peak = float(profile.max())
    peak_to_mean = peak / mean_profile if mean_profile > 0 else 0.0

    top = np.argsort(profile)[::-1][:RANGE_TOP_N]
    print(f"  Top {RANGE_TOP_N} range bins by magnitude:")
    for rank, b in enumerate(top, start=1):
        print(f"    {rank}. bin {int(b):3d}  "
              f"range={range_axis[b]:5.2f} m  "
              f"magnitude={profile[b]:.4f}")
    print(f"  peak-to-mean ratio : {peak_to_mean:.2f}")

    passed = peak_to_mean > 1.0
    print("\n  A clear peak (peak/mean > 1) means energy concentrates in range.")
    print(f"  RESULT: {_verdict(passed)}")
    return passed


# ----------------------------------------------------------------------- #
# Test 4 — stability (empty scene)
# ----------------------------------------------------------------------- #

def test_stability(sweep: SFCWSweep, pipeline: RadarPipeline) -> bool:
    """
    Capture STABILITY_SWEEPS sweeps over ~STABILITY_TOTAL_S seconds, empty scene.

    Prints each sweep's total energy and peak range-bin magnitude, then their
    spread. PASS when the energy std stays under STABILITY_REL_STD_MAX of the
    mean: a steady empty-scene level means no random drift in the chain.
    """
    _print_header(4, "STABILITY TEST (empty scene)")
    print(f"Capturing {STABILITY_SWEEPS} sweeps over ~{STABILITY_TOTAL_S:.0f} s "
          f"with an EMPTY scene (do not move in front of the radar).")

    interval = STABILITY_TOTAL_S / STABILITY_SWEEPS
    energies, peaks = [], []
    for k in range(STABILITY_SWEEPS):
        t0 = time.time()
        s_raw = sweep.run_sweep()
        energy, peak = _sweep_metrics(s_raw, pipeline)
        energies.append(energy)
        peaks.append(peak)
        print(f"  sweep {k + 1:2d}/{STABILITY_SWEEPS}: "
              f"energy={energy:12.1f}  peak={peak:.4f}")
        # Pace the captures so the 10 sweeps span ~STABILITY_TOTAL_S seconds.
        if k < STABILITY_SWEEPS - 1:
            time.sleep(max(0.0, interval - (time.time() - t0)))

    energies = np.asarray(energies)
    peaks = np.asarray(peaks)
    e_mean = float(energies.mean())
    e_std = float(energies.std())
    rel_std = e_std / e_mean if e_mean > 0 else float("inf")

    print(f"\n  energy mean : {e_mean:.1f}")
    print(f"  energy std  : {e_std:.1f}  ({rel_std * 100:.1f}% of mean)")
    print(f"  peak  std   : {peaks.std():.4f}")

    passed = rel_std < STABILITY_REL_STD_MAX
    print(f"\n  Expect a steady empty-scene energy (std < "
          f"{STABILITY_REL_STD_MAX * 100:.0f}% of mean) — no random drift.")
    print(f"  RESULT: {_verdict(passed)}")
    return passed


# ----------------------------------------------------------------------- #
# Test 5 — target response (interactive)
# ----------------------------------------------------------------------- #

def test_target_response(sweep: SFCWSweep, pipeline: RadarPipeline) -> bool | None:
    """
    Compare the return with and without a metal reflector in front of the radar.

    Captures TARGET_SWEEPS sweeps in each condition and reports the
    metal/empty ratio of both total energy and peak magnitude. The radar is
    judged to RESPOND when either ratio moves more than 20% away from 1.0
    (outside [TARGET_RESPOND_LO, TARGET_RESPOND_HI]); a ratio that stays inside
    that band means it does not respond. Returns None if run non-interactively.
    """
    _print_header(5, "TARGET RESPONSE TEST (interactive)")

    try:
        input("  Place a metal sheet / corner reflector ~50 cm in front, "
              "then press ENTER...")
    except EOFError:
        print("  No interactive input available; skipping target-response test.")
        print(f"  RESULT: {_verdict(None)}")
        return None

    print(f"  Capturing {TARGET_SWEEPS} sweeps WITH the metal target...")
    e_metal, p_metal = _capture_mean_metrics(sweep, pipeline, TARGET_SWEEPS)
    print(f"    mean energy = {e_metal:.1f}   mean peak = {p_metal:.4f}")

    try:
        input("  Remove the metal, then press ENTER...")
    except EOFError:
        print("  No interactive input available; skipping second capture.")
        print(f"  RESULT: {_verdict(None)}")
        return None

    print(f"  Capturing {TARGET_SWEEPS} sweeps WITHOUT the metal target...")
    e_empty, p_empty = _capture_mean_metrics(sweep, pipeline, TARGET_SWEEPS)
    print(f"    mean energy = {e_empty:.1f}   mean peak = {p_empty:.4f}")

    energy_ratio = e_metal / e_empty if e_empty > 0 else float("inf")
    peak_ratio = p_metal / p_empty if p_empty > 0 else float("inf")
    print(f"\n  energy ratio (metal/empty) : {energy_ratio:.2f}")
    print(f"  peak   ratio (metal/empty) : {peak_ratio:.2f}")

    responds = (
        not (TARGET_RESPOND_LO <= energy_ratio <= TARGET_RESPOND_HI)
        or not (TARGET_RESPOND_LO <= peak_ratio <= TARGET_RESPOND_HI)
    )
    if responds:
        print("  VERDICT: radar RESPONDS to the target (>20% change) — working.")
    else:
        print(f"  VERDICT: radar does NOT respond (ratios within "
              f"{TARGET_RESPOND_LO}-{TARGET_RESPOND_HI}) — problem confirmed.")
    print(f"  RESULT: {_verdict(responds)}")
    return responds


# ----------------------------------------------------------------------- #
# Orchestration
# ----------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SFCW radar detection-chain diagnostic (PASS/FAIL tests)."
    )
    parser.add_argument(
        "--ip",
        default=None,
        help="Override the Pluto URI from the config (e.g. ip:192.168.2.1). "
             "A bare IP is accepted and prefixed with 'ip:'.",
    )
    return parser.parse_args()


def _normalize_uri(ip: str) -> str:
    """
    Accept a bare host/IP or a full pyadi URI, returning a usable URI.

    A value that already carries a scheme (contains ':', e.g. "ip:192.168.2.1",
    "usb:1.2.3", "local:") is passed through unchanged; a bare IPv4/hostname is
    prefixed with the network scheme "ip:".
    """
    return ip if ":" in ip else "ip:" + ip


def main() -> None:
    """Bring the radar up, run the test sequence, and print a summary."""
    args = parse_args()
    print("SFCW radar diagnostic — verifying the detection chain.")
    print("Note: test 5 is interactive (it asks you to place/remove a target).")

    device = PlutoDevice(CONFIG_PATH)
    if args.ip:
        device.uri = _normalize_uri(args.ip)
    pipeline = RadarPipeline(CONFIG_PATH)

    print(f"\nConnecting to Pluto at {device.uri} ...")
    try:
        device.connect()
        device.configure()
        device.store_fastlock_profiles(device.frequency_vector())
    except Exception as e:
        print(f"\nBring-up FAILED: {e}")
        print("This diagnostic needs the ADALM-Pluto connected and reachable.")
        device.disconnect()
        sys.exit(1)

    info = device.describe()
    print("\nBring-up OK. Key settings:")
    print(f"  sample_rate     : {info['sample_rate']} S/s")
    print(f"  rx_gain_db      : {info['rx_gain_db']}")
    print(f"  tx_hardwaregain : {info['tx_hardwaregain_db']}")
    print(f"  profiles stored : {info['n_profiles_stored']}")

    sweep = SFCWSweep(device, switch_gpio_pin=SWITCH_GPIO_PIN)

    results: dict[str, bool | None] = {}
    try:
        results["1. RX level"] = test_rx_level(device)
        s_raw, results["2. Single sweep"] = test_single_sweep(sweep)
        results["3. Range profile"] = test_range_profile(pipeline, s_raw)
        results["4. Stability"] = test_stability(sweep, pipeline)
        results["5. Target response"] = test_target_response(sweep, pipeline)
    finally:
        sweep.close()
        device.disconnect()

    # --- Summary ---
    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    for name, passed in results.items():
        print(f"  {name:20s}: {_verdict(passed)}")

    decided = [v for v in results.values() if v is not None]
    all_pass = bool(decided) and all(decided)
    print()
    if all_pass:
        print("CONCLUSION: the hardware + software chain is working — a signal is")
        print("present and not saturated, the swept response varies, the empty")
        print("scene is stable, and (if tested) the radar responds to a target.")
    else:
        failed = [n for n, v in results.items() if v is False]
        print("CONCLUSION: one or more checks did not pass: "
              + (", ".join(failed) if failed else "see above") + ".")
        print("Review each section above for the measured values and meaning.")

    # Non-zero exit if any decided test failed, so this is usable from scripts.
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
