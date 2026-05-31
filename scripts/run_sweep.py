#!/usr/bin/env python3
"""
run_sweep.py — End-to-end SFCW radar entry point.

Wires the three stages together:

    PlutoDevice   connect, configure, store fastlock profiles, hop the LO
    SFCWSweep     drive the 201-step sweep, return (S_raw, S_ref) per sweep
    RadarPipeline phase-correct -> background-subtract -> range -> CFAR detect

Flow:
  1. Connect and configure the Pluto, store fastlock profiles for all steps.
  2. Warm up by capturing n_background_scans sweeps to seed the rolling buffer.
  3. Loop: capture one referenced sweep, append it to the buffer, run the
     pipeline on the buffered stack, and print the detection result.
  4. Run until Ctrl+C, then disconnect cleanly.

The pipeline's background subtraction needs a 2-D (n_sweeps, n_steps) stack, but
each sweep is a single 1-D vector. A rolling buffer of the most recent
n_background_scans sweeps is therefore maintained and stacked on every iteration;
the newest sweep is detected against the clutter mean of that buffer.

With --debug, each iteration's mean range profile and CFAR threshold are written
to data/ as .npy files for offline inspection.

Usage:
    python scripts/run_sweep.py
    python scripts/run_sweep.py --debug
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np

# Make the src/ packages importable regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from pluto.device import PlutoDevice          # noqa: E402
from acquisition.sweep import SFCWSweep        # noqa: E402
from processing.pipeline import RadarPipeline  # noqa: E402

CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "radar_params.yaml")
DATA_DIR = os.path.join(_REPO_ROOT, "data")

# RF reference switch GPIO pin (BCM numbering).
SWITCH_GPIO_PIN = 17

log = logging.getLogger("run_sweep")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SFCW through-wall radar — live detection loop."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save each sweep's range_profile, cfar_threshold and frequency "
             "spectrum as .npy in data/.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Every 10 frames, save a PNG figure (range profile + spectrum) to data/.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Show the live result on the fullscreen touchscreen display (pygame).",
    )
    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="Run without the reference RF switch: capture the measurement path "
             "only and skip phase correction (use when GPIO 17 has no switch).",
    )
    return parser.parse_args()


# Save a figure at this frame interval when --plot is enabled.
PLOT_EVERY_N_FRAMES = 10


def save_debug_arrays(result: dict, spectrum: dict, frame_index: int) -> None:
    """Write the range profile, CFAR threshold and spectrum to data/*.npy."""
    os.makedirs(DATA_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"frame_{frame_index:06d}_{stamp}"
    np.save(os.path.join(DATA_DIR, f"{base}_range_profile.npy"),
            result["range_profile"])
    np.save(os.path.join(DATA_DIR, f"{base}_cfar_threshold.npy"),
            result["cfar_threshold"])
    np.save(os.path.join(DATA_DIR, f"{base}_freq_spectrum.npy"),
            spectrum["magnitude_db"])


def save_plot(
    result: dict, spectrum: dict, range_axis: np.ndarray, frame_index: int
) -> None:
    """
    Save a two-panel diagnostic PNG to data/ (headless: no plt.show()).

    Panel 1: mean range profile and the CFAR threshold versus range (meters).
    Panel 2: corrected-sweep magnitude spectrum (dB) versus LO frequency (GHz).
    """
    # Use a non-interactive backend so this works over SSH on the Pi.
    import matplotlib
    matplotlib.use("Agg")
    # Keep matplotlib's verbose font/backend debug logs out of the console.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt

    os.makedirs(DATA_DIR, exist_ok=True)

    # Edge cells carry an infinite CFAR threshold; mask them for plotting.
    threshold = np.array(result["cfar_threshold"], dtype=float)
    threshold[~np.isfinite(threshold)] = np.nan

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    ax1.plot(range_axis, result["range_profile"], label="range profile")
    ax1.plot(range_axis, threshold, "--", label="CFAR threshold")
    if result["detected"]:
        ax1.axvline(result["target_range_m"], color="r", alpha=0.6,
                    label=f"target {result['target_range_m']:.2f} m")
    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Mean |h|")
    ax1.set_title(f"Range profile — frame {frame_index}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(spectrum["freq_axis"], spectrum["magnitude_db"])
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title("Corrected sweep spectrum")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fig.savefig(os.path.join(DATA_DIR, f"frame_{frame_index:06d}_{stamp}.png"),
                dpi=100)
    plt.close(fig)


def main() -> None:
    """Set up the radar and run the live detection loop until Ctrl+C."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # --- Instantiate the three stages ---
    device = PlutoDevice(CONFIG_PATH)
    pipeline = RadarPipeline(CONFIG_PATH)
    n_background = pipeline.n_background_scans

    # --- Bring the radar up ---
    device.connect()
    device.configure()
    device.store_fastlock_profiles(device.frequency_vector())

    sweep = SFCWSweep(device, switch_gpio_pin=SWITCH_GPIO_PIN)

    # Optional touchscreen UI (pygame imported lazily so non-UI runs need no GUI).
    display = None
    if args.ui:
        from ui.display import RadarDisplay  # noqa: E402
        display = RadarDisplay()
        display.start()

    # Range axis (meters) for plotting, derived once from the pipeline geometry.
    range_axis = np.arange(pipeline.n_steps) * pipeline.range_resolution_m

    # Reference-path handling. With a real RF switch each sweep yields both the
    # measurement and reference paths and phase correction divides them out.
    # Without the switch (--no-ref) we capture the measurement path only and
    # substitute a flat reference of ones, so pipeline.run(S_raw, S_ref) is
    # unchanged but the S_raw / S_ref division becomes a no-op.
    fake_ref = np.ones(pipeline.n_steps, dtype=np.complex128)
    if args.no_ref:
        log.warning("Running without reference path — phase correction disabled")

    def capture_sweep() -> tuple[np.ndarray, np.ndarray]:
        """Capture one sweep, with or without the reference path."""
        if args.no_ref:
            return sweep.run_sweep(), fake_ref
        return sweep.run_sweep_with_reference()

    # Fewer warmup sweeps without the reference path for faster hardware testing.
    n_warmup = 5 if args.no_ref else n_background

    # Rolling buffers of the most recent sweeps (clutter background window).
    raw_buffer: deque[np.ndarray] = deque(maxlen=n_background)
    ref_buffer: deque[np.ndarray] = deque(maxlen=n_background)

    try:
        # --- Warmup: fill the background buffer ---
        print(f"Warming up: capturing {n_warmup} background sweeps...")
        for _ in range(n_warmup):
            s_raw, s_ref = capture_sweep()
            raw_buffer.append(s_raw)
            ref_buffer.append(s_ref)
        print("Warmup complete. Entering detection loop (Ctrl+C to stop).")

        # --- Continuous detection loop ---
        frame_index = 0
        while True:
            s_raw, s_ref = capture_sweep()
            raw_buffer.append(s_raw)
            ref_buffer.append(s_ref)

            S_raw = np.asarray(raw_buffer)
            S_ref = np.asarray(ref_buffer)
            result = pipeline.run(S_raw, S_ref)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] detected={str(result['detected']):5s}  "
                f"target_range_m={result['target_range_m']:.3f}"
            )

            # Frequency spectrum is needed by both --debug and --plot.
            if args.debug or args.plot:
                S_corrected = pipeline.phase_correction(S_raw, S_ref)
                spectrum = pipeline.frequency_spectrum(S_corrected)

                if args.debug:
                    save_debug_arrays(result, spectrum, frame_index)
                if args.plot and frame_index % PLOT_EVERY_N_FRAMES == 0:
                    save_plot(result, spectrum, range_axis, frame_index)

            if display is not None:
                display.update(result)
                # Exit if the user pressed STOP on the touchscreen.
                if not display.is_running():
                    print("STOP pressed on display.")
                    break

            frame_index += 1

    except KeyboardInterrupt:
        print()  # break the line after the ^C
    finally:
        if display is not None:
            display.stop()
        sweep.close()
        device.disconnect()
        print("Radar stopped.")


if __name__ == "__main__":
    main()
