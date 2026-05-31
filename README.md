# sfcw-radar
SFCW radar system for through-wall human detection using ADALM-Pluto SDR and Raspberry Pi 4

A stepped-frequency continuous-wave (SFCW) radar that synthesizes a wide
bandwidth from 201 narrowband LO steps (1–4 GHz), then range-processes the
swept channel response and runs detection. It is built to run headless on a
Raspberry Pi 4 with an optional WaveShare 5-inch HDMI touchscreen UI, and ships
three complementary detection modes (CFAR, signal-level, and energy variance).

## Hardware

| Component | Role |
|-----------|------|
| ADALM-Pluto SDR | TX/RX front end; LO is retuned per frequency step |
| Raspberry Pi 4 | Host controller; runs the sweep and processing headless |
| WaveShare 5-inch HDMI touchscreen (800×480) | Optional live UI (`--ui`) |
| RF reference switch (GPIO 17) | Optional; routes a through-line reference for phase correction |

Off-Pi the GPIO backend falls back to a no-op stub, so the full pipeline stays
importable and testable on a development laptop without any hardware attached.

## Radar Parameters

| Parameter | Value |
|-----------|-------|
| Frequency range | 1.0 – 4.0 GHz |
| Synthesized bandwidth | 3.0 GHz |
| Frequency step (Δf) | 15 MHz |
| Number of LO steps | 201 |
| TX power | −10 dBm |
| Pluto sample rate | 2.5 MSPS |
| Range resolution | ~0.05 m |
| Maximum unambiguous range | ~10 m |

These are defined in `config/radar_params.yaml` and read at startup; the range
resolution and maximum range follow directly from the swept bandwidth and step
size.

## Repository Structure

```
sfcw-radar/
├── config/
│   └── radar_params.yaml      # RF, Pluto and processing parameters
├── data/                      # --debug .npy dumps and --plot PNGs land here
├── docs/
├── notebooks/
│   ├── radar_analysis.ipynb   # offline analysis of captured sweeps
│   └── radar_motion_demo.py   # standalone Doppler velocity simulation
├── logs/                      # rotating runtime logs (git-ignored)
│   └── sfcw_radar.log
├── scripts/
│   └── run_sweep.py           # end-to-end entry point (live detection loop)
├── src/
│   ├── pluto/
│   │   └── device.py          # PlutoDevice: connect, fastlock hop, CW carrier
│   ├── acquisition/
│   │   └── sweep.py           # SFCWSweep: drive the 201-step sweep + RF switch
│   ├── processing/
│   │   └── pipeline.py        # RadarPipeline: phase corr → bg sub → range → detect
│   ├── ui/
│   │   └── display.py         # RadarDisplay: pygame touchscreen UI (white theme)
│   └── utils/
│       └── logger.py          # centralized logging (console + rotating file)
├── tests/
│   ├── test_pipeline.py       # pipeline + detection-method coverage
│   └── test_logger.py         # logging configuration coverage
├── requirements.txt
└── README.md
```

## Installation

```bash
# Python dependencies
pip install -r requirements.txt

# On the Raspberry Pi, install pygame from the system packages for the touchscreen UI
sudo apt-get install python3-pygame
```

The RF switch GPIO backend (`gpiozero` / `RPi.GPIO`) is only needed on the Pi;
off-Pi the sweep falls back to a no-op stub so the code stays importable.

## Usage

```bash
# Basic detection (CFAR)
python scripts/run_sweep.py
```

Flags (can be combined):

| Flag | Effect |
|------|--------|
| `--ui` | Launch fullscreen display on HDMI screen (800×480) |
| `--no-ref` | Run without RF reference switch (phase correction disabled) |
| `--no-bg` | Skip background subtraction (keep static clutter) |
| `--level` | Signal-level presence detection vs empty-room baseline |
| `--energy` | Variance-based motion detection |
| `--doppler` | Add Doppler velocity + micro-Doppler processing |
| `--debug` | Save range_profile/cfar_threshold/spectrum as .npy to data/ |
| `--plot` | Save diagnostic PNG plots to data/ |

Recommended for hardware bring-up without RF switch:

```bash
python scripts/run_sweep.py --ui --no-ref --level
```

Press Ctrl+C (or the on-screen close button with `--ui`) to stop cleanly.

## Detection Modes

The pipeline offers three detectors. They are complementary — CFAR and level
find presence, while energy finds motion — and can be selected per run.

- **CFAR** (default): range-based detection of peaks that rise above an adaptive
  OS-CFAR noise threshold. Reports the range of the strongest qualifying target.
  Best when a target produces a distinct range peak.
- **Level** (`--level`): compares the peak range-profile magnitude (dB) against
  an empty-room baseline captured during warmup, plus a fixed margin. Best for
  presence detection, including **stationary** targets behind walls that a
  motion detector would miss.
- **Energy** (`--energy`): tracks the variance of the current/background energy
  ratio over a short sliding window. Best for **moving** targets; a person who
  stops moving fades back to "no detection" after a few seconds.

## Configuration

All tunable parameters live in `config/radar_params.yaml`. The most useful knobs
for field tuning:

| Key | Effect |
|-----|--------|
| `processing.detection_margin_db` | Level-detector sensitivity: dB the peak must exceed the empty-room baseline |
| `processing.energy_variance_threshold` | Energy-detector sensitivity: ratio std-dev that counts as motion |
| `cfar.false_alarm_rate` | CFAR sensitivity: lower = stricter, fewer false alarms |
| `pluto.dwell_buffers` | Sweep speed vs SNR: buffers averaged per LO step (1 = fastest) |

## Logging

All modules log through the centralized configuration in `src/utils/logger.py`.
Logs are written to `logs/sfcw_radar.log` (rotating, 1 MB × 3 backups); the
console shows INFO and above while the file captures DEBUG and above.

Raise verbosity from code with `set_level("DEBUG")`, or run with the `--debug`
flag to enable DEBUG-level output at runtime.

## Testing

```bash
pytest
```

The suite under `tests/` (23 tests) uses synthetic NumPy data and requires no
Pluto or Pi hardware.
