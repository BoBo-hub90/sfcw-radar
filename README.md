# sfcw-radar
SFCW radar system for through-wall human detection using ADALM-Pluto SDR and Raspberry Pi 4

A stepped-frequency continuous-wave (SFCW) radar that synthesizes a wide
bandwidth from 201 narrowband LO steps (1–4 GHz), then range-processes the
swept channel response and runs CA-CFAR detection. It is built to run headless
on a Raspberry Pi 4 with an optional WaveShare 5-inch HDMI touchscreen UI.

## Repository Structure

```
sfcw-radar/
├── config/
│   └── radar_params.yaml      # RF, Pluto and processing parameters
├── data/                      # --debug .npy dumps and --plot PNGs land here
├── docs/
├── notebooks/
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
│   │   └── pipeline.py        # RadarPipeline: phase corr → bg sub → range → CFAR
│   ├── ui/
│   │   └── display.py         # RadarDisplay: pygame touchscreen UI (white theme)
│   └── utils/
│       └── logger.py          # centralized logging (console + rotating file)
├── tests/                     # pytest suite (synthetic data, no hardware)
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
python scripts/run_sweep.py                       # basic live detection loop
python scripts/run_sweep.py --ui                  # with the touchscreen display
python scripts/run_sweep.py --debug               # save range/CFAR/spectrum .npy to data/
python scripts/run_sweep.py --plot                # save diagnostic PNG plots to data/
python scripts/run_sweep.py --ui --debug --plot   # full: UI + .npy dumps + PNG plots
```

Press Ctrl+C (or the on-screen close button with `--ui`) to stop cleanly.

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

The suite under `tests/` uses synthetic NumPy data and requires no Pluto or Pi
hardware.
