"""
ADALM-Pluto SDR device controller for the SFCW radar.

A Stepped-Frequency Continuous-Wave (SFCW) radar tunes the local oscillator
(LO) across a discrete set of frequencies (here: 201 steps from 1 GHz to
4 GHz) and measures the complex baseband response at each step. The synthetic
wideband response is reconstructed by an inverse FFT across the steps.

The dominant cost of SFCW on an AD9361-based SDR is LO retuning: a full PLL
lock with VCO calibration takes hundreds of microseconds, which makes a 201-step
sweep far too slow for phase-stable vital-sign extraction. The AD9361 "fastlock"
feature solves this: the calibrated PLL state for a frequency is captured once,
then recalled in tens of microseconds.

Fastlock slot limitation
-------------------------
The AD9361 has only 8 internal fastlock slots (0..7), but this radar needs 201
profiles. The working pattern is therefore:

  1. store_fastlock_profiles(): for each frequency, tune the LO, capture the
     PLL state into internal slot 0, then read the slot's raw bytes out with
     `fastlock_save` and keep the string in host memory (201 strings).
  2. recall_profile(i): write the saved string back with `fastlock_load`
     (which targets slot 0) and apply it with `fastlock_recall`. This reuses a
     single hardware slot for all 201 steps.

This module uses pyadi-iio (`import adi`). Fastlock is not exposed as named
pyadi methods, so it is driven through pyadi's `_set_iio_attr` /
`_get_iio_attr_str` helpers, which operate on the AD9361 LO channels
(`altvoltage1` = TX LO, `altvoltage0` = RX LO, both output channels) rather
than opening a separate raw libiio context.

Usage:
    dev = PlutoDevice("config/radar_params.yaml")
    dev.connect()
    dev.configure()
    freqs = dev.frequency_vector()           # 201 steps, 1..4 GHz
    dev.store_fastlock_profiles(freqs)        # one-time calibration
    for i in range(len(freqs)):
        dev.recall_profile(i)                 # fast LO hop
        # ... transmit tone / capture RX buffer at this step ...
    dev.disconnect()
"""

from __future__ import annotations

import time

import numpy as np
import yaml
import adi

from utils.logger import get_logger

log = get_logger(__name__)

# AD9361 LO channels on the ad9361-phy device. Both are output channels.
TX_LO_CHANNEL = "altvoltage1"
RX_LO_CHANNEL = "altvoltage0"

# Single hardware fastlock slot reused for all profiles (see module docstring).
FASTLOCK_SLOT = 0

# Pluto's tx_hardwaregain_chan0 is an attenuation: 0 dB == maximum output
# (roughly 0 dBm at the SMA, frequency dependent). A requested TX power of
# P dBm is mapped to an attenuation of -P dB below this reference. The mapping
# is uncalibrated and approximate; treat tx_power_dbm as a relative setpoint.
TX_MAX_POWER_DBM = 0.0

# Default RX path gain. The radar config does not specify an RX gain, and phase
# stability for vital-sign extraction strongly favours a fixed manual gain over
# AGC, so a manual default is used here.
DEFAULT_RX_GAIN_DB = 40.0

# Pluto TX DAC expects int16 IQ with full scale at 2^14 = 16384; 2^15-1 clips.
# Using 2^14 leaves ~6 dB of headroom.
TX_FULL_SCALE = 2 ** 14

# Length of the cyclic CW buffer. A short constant buffer is sufficient: cyclic
# mode loops it continuously, producing an unmodulated carrier at the LO.
CW_BUFFER_SIZE = 1024

# Settle time (seconds) after tuning the LO before capturing its PLL state.
# Allows the full PLL lock to complete so the captured fastlock profile is valid.
LO_SETTLE_S = 0.005


class PlutoDevice:
    """ADALM-Pluto controller for SFCW operation with fastlock LO hopping."""

    def __init__(self, config_path: str = "config/radar_params.yaml"):
        """
        Args:
            config_path: Path to the YAML radar configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        radar = self.config["radar"]
        pluto = self.config["pluto"]

        self.f_start: float = float(radar["f_start"])
        self.f_stop: float = float(radar["f_stop"])
        self.n_steps: int = int(radar["n_steps"])
        self.tx_power_dbm: float = float(radar["tx_power_dbm"])

        self.uri: str = str(pluto["ip"])
        self.sample_rate: int = int(float(pluto["sample_rate"]))
        self.rx_buffer_size: int = int(pluto["rx_buffer_size"])

        self._sdr: adi.Pluto | None = None
        self._cw_active: bool = False

        # Captured fastlock profiles (host-side), one string per frequency step.
        self._tx_profiles: list[str] = []
        self._rx_profiles: list[str] = []
        self.frequencies: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Configuration loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_config(path: str) -> dict:
        """Load the radar YAML config into a dict."""
        log.info("Loading radar config: %s", path)
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def frequency_vector(self) -> np.ndarray:
        """Return the 201-step LO frequency grid (Hz), inclusive of endpoints."""
        return np.linspace(self.f_start, self.f_stop, self.n_steps)

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Open a connection to the Pluto at the configured URI."""
        log.info("Connecting to Pluto: %s", self.uri)
        try:
            self._sdr = adi.Pluto(self.uri)
        except Exception as e:
            log.error("Connection to %s failed: %s", self.uri, e)
            raise
        log.info("Connection established")

    def disconnect(self) -> None:
        """Stop TX and release the Pluto handle."""
        if self._sdr is None:
            return
        log.info("Disconnecting from Pluto")
        self.stop_cw()
        # pyadi-iio has no explicit close; dropping the reference is sufficient.
        self._sdr = None

    def __enter__(self) -> "PlutoDevice":
        self.connect()
        self.configure()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def _require_sdr(self) -> adi.Pluto:
        """Return the live SDR handle or raise if not connected."""
        if self._sdr is None:
            raise RuntimeError("Not connected; call connect() first")
        return self._sdr

    # ------------------------------------------------------------------ #
    # Channel configuration
    # ------------------------------------------------------------------ #

    def configure(self) -> None:
        """
        Apply baseband and RF parameters to the TX and RX channels.

        Note on bandwidth: the config's `radar.bandwidth` (3 GHz) is the
        *synthetic* SFCW sweep span, not the instantaneous channel bandwidth.
        The Pluto analog filter bandwidth must track the baseband sample rate
        (2.5 MS/s) instead, so it is set from `sample_rate`, not from
        `radar.bandwidth`.
        """
        s = self._require_sdr()

        log.info("Configuring Pluto channels for SFCW operation")

        # Baseband sample rate (shared TX/RX).
        s.sample_rate = self.sample_rate
        log.info("  sample_rate     : %d S/s", s.sample_rate)

        # Analog filter bandwidth tracks the sample rate, not the sweep span.
        analog_bw = self.sample_rate
        s.rx_rf_bandwidth = int(analog_bw)
        s.tx_rf_bandwidth = int(analog_bw)
        log.info("  rx/tx_rf_bw     : %d Hz (tracks sample_rate)", analog_bw)

        # Initial LO position at the start of the sweep band.
        s.tx_lo = int(self.f_start)
        s.rx_lo = int(self.f_start)
        log.info("  tx_lo / rx_lo   : %d Hz (sweep start)", int(self.f_start))

        # RX gain: fixed manual gain for phase stability.
        s.gain_control_mode_chan0 = "manual"
        s.rx_hardwaregain_chan0 = float(DEFAULT_RX_GAIN_DB)
        log.info("  rx_gain         : %.1f dB (manual)", s.rx_hardwaregain_chan0)

        # TX power: map requested dBm to Pluto's attenuation attribute.
        tx_atten = self.tx_power_dbm - TX_MAX_POWER_DBM  # e.g. -10 dBm -> -10 dB
        s.tx_hardwaregain_chan0 = float(tx_atten)
        log.info("  tx_hardwaregain : %.1f dB (target %.1f dBm)",
                 tx_atten, self.tx_power_dbm)

        # RX capture size.
        s.rx_buffer_size = int(self.rx_buffer_size)
        log.info("  rx_buffer_size  : %d", s.rx_buffer_size)

        log.info("Channel configuration complete")

    # ------------------------------------------------------------------ #
    # Fastlock profile management
    # ------------------------------------------------------------------ #

    def store_fastlock_profiles(self, frequencies: np.ndarray) -> None:
        """
        Capture an AD9361 fastlock profile for every frequency step.

        For each frequency the TX and RX LOs are tuned and allowed to lock, the
        PLL state is stored into a single hardware slot, and the slot's raw
        bytes are read back with `fastlock_save` into host memory. This sidesteps
        the 8-slot hardware limit so that all 201 steps can be recalled later.

        Args:
            frequencies: 1-D array of LO frequencies in Hz. Typically the output
                of frequency_vector().
        """
        s = self._require_sdr()

        frequencies = np.asarray(frequencies, dtype=float)
        self.frequencies = frequencies
        self._tx_profiles = []
        self._rx_profiles = []

        log.info("Storing fastlock profiles for %d frequency steps",
                 len(frequencies))

        for i, freq in enumerate(frequencies):
            f = int(freq)

            # Tune both LOs and let the PLL fully lock before capturing.
            s.tx_lo = f
            s.rx_lo = f
            time.sleep(LO_SETTLE_S)

            # Capture the current PLL state into the shared hardware slot, then
            # read the slot's bytes out to host memory.
            self._set_lo_attr(TX_LO_CHANNEL, "fastlock_store", FASTLOCK_SLOT)
            self._set_lo_attr(RX_LO_CHANNEL, "fastlock_store", FASTLOCK_SLOT)
            tx_profile = self._get_lo_attr(TX_LO_CHANNEL, "fastlock_save")
            rx_profile = self._get_lo_attr(RX_LO_CHANNEL, "fastlock_save")

            self._tx_profiles.append(tx_profile)
            self._rx_profiles.append(rx_profile)

            if i == 0 or (i + 1) % 50 == 0 or i == len(frequencies) - 1:
                log.info("  stored profile %d/%d at %.3f GHz",
                         i + 1, len(frequencies), freq / 1e9)

        log.info("Fastlock profile storage complete (%d profiles)",
                 len(self._tx_profiles))

    def recall_profile(self, index: int) -> None:
        """
        Recall a previously stored fastlock profile to hop the LO instantly.

        The saved TX/RX profile strings are loaded back into the shared hardware
        slot with `fastlock_load`, then applied with `fastlock_recall`. This is
        the fast path used inside the sweep loop.

        Args:
            index: Step index into the stored profiles (0..n_steps-1).
        """
        self._require_sdr()

        if not self._tx_profiles:
            raise RuntimeError(
                "No fastlock profiles stored; call store_fastlock_profiles() first"
            )
        if not 0 <= index < len(self._tx_profiles):
            raise IndexError(
                f"Profile index {index} out of range "
                f"[0, {len(self._tx_profiles)})"
            )

        # Load the saved bytes into the slot, then apply them.
        self._set_lo_attr(TX_LO_CHANNEL, "fastlock_load", self._tx_profiles[index])
        self._set_lo_attr(RX_LO_CHANNEL, "fastlock_load", self._rx_profiles[index])
        self._set_lo_attr(TX_LO_CHANNEL, "fastlock_recall", FASTLOCK_SLOT)
        self._set_lo_attr(RX_LO_CHANNEL, "fastlock_recall", FASTLOCK_SLOT)

    # ------------------------------------------------------------------ #
    # Transmission (CW carrier)
    # ------------------------------------------------------------------ #

    def start_cw(self) -> None:
        """
        Start a continuous-wave carrier via cyclic TX.

        Loads a constant full-scale complex value (a zero-IF / DC tone) into a
        cyclic TX buffer. With cyclic mode enabled the buffer loops forever, so
        an unmodulated carrier is emitted at whatever LO the fastlock recall has
        set. This must be running throughout an SFCW sweep so each step has a
        tone to measure; the LO is hopped under the carrier between steps.

        Idempotent: a no-op if the carrier is already running.
        """
        s = self._require_sdr()
        if self._cw_active:
            log.debug("CW carrier already active")
            return

        # Constant full-scale phasor; cyclic looping turns it into a pure tone.
        tone = np.full(CW_BUFFER_SIZE, TX_FULL_SCALE + 0j, dtype=np.complex64)

        s.tx_cyclic_buffer = True
        s.tx(tone)
        self._cw_active = True
        log.info("CW carrier started (cyclic TX, %d samples)", CW_BUFFER_SIZE)

    def stop_cw(self) -> None:
        """Stop the CW carrier and release the cyclic TX buffer."""
        if self._sdr is None or not self._cw_active:
            return
        try:
            self._sdr.tx_destroy_buffer()
        except Exception as e:
            log.warning("tx_destroy_buffer raised while stopping CW: %s", e)
        self._cw_active = False
        log.info("CW carrier stopped")

    # ------------------------------------------------------------------ #
    # Reception
    # ------------------------------------------------------------------ #

    def receive(self, num_buffers: int = 1) -> np.ndarray:
        """
        Capture one or more RX buffers of complex baseband IQ.

        Each buffer holds `rx_buffer_size` complex samples. At a fixed LO this is
        the per-step measurement primitive used by the SFCW sweep.

        Args:
            num_buffers: Number of consecutive buffers to capture.

        Returns:
            complex64 array of shape (num_buffers * rx_buffer_size,).
        """
        s = self._require_sdr()
        buffers = [s.rx() for _ in range(num_buffers)]
        return np.concatenate(buffers).astype(np.complex64)

    # ------------------------------------------------------------------ #
    # Low-level LO attribute access (pyadi-iio helpers)
    # ------------------------------------------------------------------ #

    def _set_lo_attr(self, channel: str, attr: str, value) -> None:
        """Write an attribute on an LO (altvoltage) output channel."""
        # Signature: _set_iio_attr(channel_name, attr_name, output, value)
        self._require_sdr()._set_iio_attr(channel, attr, True, value)

    def _get_lo_attr(self, channel: str, attr: str) -> str:
        """Read a string attribute from an LO (altvoltage) output channel."""
        # fastlock_save returns a slot+bytes string, so the string getter is used.
        return self._require_sdr()._get_iio_attr_str(channel, attr, True)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def describe(self) -> dict:
        """Return the current Pluto settings as a dict."""
        s = self._require_sdr()
        return {
            "uri": self.uri,
            "sample_rate": s.sample_rate,
            "tx_lo": s.tx_lo,
            "rx_lo": s.rx_lo,
            "rx_rf_bandwidth": s.rx_rf_bandwidth,
            "tx_rf_bandwidth": s.tx_rf_bandwidth,
            "rx_gain_db": s.rx_hardwaregain_chan0,
            "tx_hardwaregain_db": s.tx_hardwaregain_chan0,
            "rx_buffer_size": s.rx_buffer_size,
            "n_profiles_stored": len(self._tx_profiles),
        }
