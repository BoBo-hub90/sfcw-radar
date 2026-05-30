"""
SFCW sweep acquisition.

Drives a PlutoDevice through its 201-step frequency grid and collects one
complex phasor S[i] per step. The synthetic wideband channel response is the
vector S = [S[0], ..., S[n_steps-1]]; an inverse FFT across the steps yields the
range profile, and stacking sweeps over slow time feeds vital-sign extraction.

At each step the LO is hopped with the device's fastlock recall (fast), the RX
buffer is filled several times to dwell on the tone, and the complex samples are
averaged into a single phasor. Averaging across both fast time (samples within a
buffer) and several buffers suppresses noise while preserving the step's
amplitude and phase.

Optional reference path
-----------------------
A single-pole RF switch (default GPIO 17) can route the receiver between the
antenna ("measurement") and an internal reference ("reference") at each step.
Capturing both lets downstream processing divide out the system response
(cable/PLL/IQ-imbalance drift) via S_raw / S_ref.

GPIO backends
-------------
The RF switch is driven through RPi.GPIO or gpiozero on a Raspberry Pi. When
neither library is importable (e.g. a development laptop), a no-op stub backend
is used so the module stays importable and the sweep logic can be exercised
without hardware. The switch state is logged either way.

Usage:
    from pluto.device import PlutoDevice
    from acquisition.sweep import SFCWSweep

    dev = PlutoDevice("config/radar_params.yaml")
    dev.connect()
    dev.configure()
    dev.store_fastlock_profiles(dev.frequency_vector())

    sweep = SFCWSweep(dev, switch_gpio_pin=17)
    s_raw = sweep.run_sweep()                      # shape (201,)
    s_raw, s_ref = sweep.run_sweep_with_reference()
    slow_time = sweep.run_multi_sweep(n_sweeps=128) # shape (128, 201)
    sweep.close()
"""

from __future__ import annotations

import logging
import time

import numpy as np

from pluto.device import PlutoDevice

log = logging.getLogger(__name__)

# Minimum buffer fills per step (dwell), per requirements.
DEFAULT_DWELL_BUFFERS = 5

# One buffer is read and discarded after an LO hop / switch toggle to drop
# samples that were in flight before the new state settled.
WARMUP_BUFFERS = 1

# Settle time (seconds) after toggling the RF switch before capturing.
SWITCH_SETTLE_S = 0.001

# RF switch path labels.
PATH_MEASUREMENT = "measurement"
PATH_REFERENCE = "reference"


class _RFSwitch:
    """
    Single-pin RF switch controller with a graceful no-hardware fallback.

    Convention: the GPIO pin is driven LOW for the measurement path and HIGH for
    the reference path. Invert externally (or swap the cabling) if your switch
    uses the opposite polarity.
    """

    def __init__(self, pin: int):
        self.pin = pin
        self._state = PATH_MEASUREMENT
        self._backend, self._handle = self._init_backend(pin)
        log.info("RF switch on GPIO %d using %s backend", pin, self._backend)
        self.to_measurement()

    def _init_backend(self, pin: int):
        """Select gpiozero, then RPi.GPIO, then a no-op stub."""
        try:
            from gpiozero import DigitalOutputDevice

            return "gpiozero", DigitalOutputDevice(pin, initial_value=False)
        except Exception:
            pass
        try:
            import RPi.GPIO as GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            return "RPi.GPIO", GPIO
        except Exception:
            log.warning(
                "No GPIO backend available; RF switch is a no-op stub "
                "(reference path will measure the same signal as measurement)"
            )
            return "stub", None

    def _write(self, high: bool) -> None:
        if self._backend == "gpiozero":
            self._handle.on() if high else self._handle.off()
        elif self._backend == "RPi.GPIO":
            self._handle.output(self.pin, self._handle.HIGH if high else self._handle.LOW)
        # stub: nothing to do

    def to_measurement(self) -> None:
        """Route the receiver to the antenna (pin LOW)."""
        if self._state != PATH_MEASUREMENT:
            self._write(False)
            time.sleep(SWITCH_SETTLE_S)
        self._state = PATH_MEASUREMENT

    def to_reference(self) -> None:
        """Route the receiver to the internal reference (pin HIGH)."""
        if self._state != PATH_REFERENCE:
            self._write(True)
            time.sleep(SWITCH_SETTLE_S)
        self._state = PATH_REFERENCE

    def close(self) -> None:
        """Release the GPIO resource."""
        try:
            if self._backend == "gpiozero":
                self._handle.close()
            elif self._backend == "RPi.GPIO":
                self._handle.cleanup(self.pin)
        except Exception as e:
            log.warning("RF switch cleanup raised: %s", e)


class SFCWSweep:
    """Acquire SFCW frequency sweeps from a configured PlutoDevice."""

    def __init__(
        self,
        device: PlutoDevice,
        switch_gpio_pin: int = 17,
        dwell_buffers: int = DEFAULT_DWELL_BUFFERS,
    ):
        """
        Args:
            device: A PlutoDevice that is already connected, configured, and has
                fastlock profiles stored (store_fastlock_profiles()).
            switch_gpio_pin: BCM pin number controlling the reference RF switch.
            dwell_buffers: Buffer fills averaged per step (clamped to >= 5).
        """
        self.device = device
        self.n_steps = device.n_steps

        if dwell_buffers < DEFAULT_DWELL_BUFFERS:
            log.warning(
                "dwell_buffers=%d below minimum; clamping to %d",
                dwell_buffers, DEFAULT_DWELL_BUFFERS,
            )
            dwell_buffers = DEFAULT_DWELL_BUFFERS
        self.dwell_buffers = dwell_buffers

        self.switch = _RFSwitch(switch_gpio_pin)

    # ------------------------------------------------------------------ #
    # Per-step measurement
    # ------------------------------------------------------------------ #

    def _measure_phasor(self) -> complex:
        """
        Dwell on the current LO and return the mean complex IQ phasor.

        One warm-up buffer is discarded, then `dwell_buffers` buffers are
        captured and averaged across all of their samples.
        """
        if WARMUP_BUFFERS:
            self.device.receive(num_buffers=WARMUP_BUFFERS)
        samples = self.device.receive(num_buffers=self.dwell_buffers)
        return complex(np.mean(samples))

    # ------------------------------------------------------------------ #
    # Sweeps
    # ------------------------------------------------------------------ #

    def _sweep_into(self, out: np.ndarray) -> None:
        """
        Run one measurement-path sweep, writing a phasor per step into `out`.

        Assumes the CW carrier is already running; does not manage TX so it can
        be reused under a single continuous carrier across many sweeps.
        """
        self.switch.to_measurement()
        for i in range(self.n_steps):
            self.device.recall_profile(i)
            out[i] = self._measure_phasor()

    def run_sweep(self) -> np.ndarray:
        """
        Run one frequency sweep over all steps (measurement path only).

        Returns:
            Complex array S_raw of shape (n_steps,), one phasor per step.
        """
        s_raw = np.empty(self.n_steps, dtype=np.complex128)

        self.device.start_cw()
        try:
            self._sweep_into(s_raw)
        finally:
            self.device.stop_cw()

        log.info("Sweep complete: %d steps", self.n_steps)
        return s_raw

    def run_sweep_with_reference(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one sweep capturing both the reference and measurement paths.

        At each step the LO is hopped once, then the RF switch routes the
        reference first and the measurement second, so both phasors share the
        same LO state.

        Returns:
            (S_raw, S_ref), each a complex array of shape (n_steps,).
        """
        s_raw = np.empty(self.n_steps, dtype=np.complex128)
        s_ref = np.empty(self.n_steps, dtype=np.complex128)

        self.device.start_cw()
        try:
            for i in range(self.n_steps):
                self.device.recall_profile(i)

                self.switch.to_reference()
                s_ref[i] = self._measure_phasor()

                self.switch.to_measurement()
                s_raw[i] = self._measure_phasor()
        finally:
            self.device.stop_cw()

        log.info("Referenced sweep complete: %d steps", self.n_steps)
        return s_raw, s_ref

    def run_multi_sweep(self, n_sweeps: int) -> np.ndarray:
        """
        Run several consecutive sweeps to build the slow-time axis.

        Each row is one full frequency sweep; the row index is slow time, which
        carries the breathing/heartbeat modulation for vital-sign extraction.

        Args:
            n_sweeps: Number of sweeps (slow-time samples) to collect.

        Returns:
            Complex array of shape (n_sweeps, n_steps).
        """
        if n_sweeps < 1:
            raise ValueError("n_sweeps must be >= 1")

        data = np.empty((n_sweeps, self.n_steps), dtype=np.complex128)

        # Keep the carrier on continuously across all sweeps so slow time is
        # uninterrupted (no TX gaps between vital-sign samples).
        self.device.start_cw()
        try:
            for k in range(n_sweeps):
                self._sweep_into(data[k, :])
                if (k + 1) % 16 == 0 or k == n_sweeps - 1:
                    log.info("  multi-sweep %d/%d captured", k + 1, n_sweeps)
        finally:
            self.device.stop_cw()

        return data

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Release the RF switch GPIO. Does not disconnect the device."""
        self.switch.close()

    def __enter__(self) -> "SFCWSweep":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
