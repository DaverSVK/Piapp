"""Measured BMI270 constants: bias, noise and the detector thresholds.

Everything here comes from ``static_drift_config.json``, which was derived from
a 6.43 h / 2.15 M-sample bench run with the device flat and untouched, and
cross-checked against the quiet windows of real on-ice captures. Two blocks are
measurements and should only change when new static tests are run (``bias``,
``noise``, ``drift``); one block is tuning and is meant to be edited
(``thresholds``).

The distinction that matters most in that file: the ``*_lab`` thresholds are
k-sigma from the bench noise floor, and apply only to bench data. The defaults
are set from the on-ice quiet windows, which are 20-60x noisier - nothing on
the ice is ever as still as a table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "imu_static_drift_config"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "static_drift_config.json"

AXES = "xyz"


class SensorConfig:
    """The parts of static_drift_config.json the estimator uses."""

    def __init__(self, cfg: dict, lab: bool = False):
        self.raw = cfg

        b = cfg["bias"]
        self.gyro_bias_dps = np.array([b["gyro_dps"][k] for k in AXES])
        self.accel_bias_g = np.array([b["accel_g"][k] for k in AXES])

        n = cfg["noise"]
        self.k = float(n.get("sigma_multiplier_k", 6))
        self.gyro_std_dps = np.array([n["gyro_std_dps"][k] for k in AXES])
        self.accel_std_g = np.array([n["accel_std_g"][k] for k in AXES])

        t = cfg["thresholds"]
        suffix = "_lab" if lab else ""
        self.gyro_stationary_dps = float(
            t.get(f"gyro_stationary_dps{suffix}", t["gyro_stationary_dps"]))
        self.accel_stationary_dev_g = float(
            t.get(f"accel_stationary_dev_g{suffix}", t["accel_stationary_dev_g"]))
        self.gyro_deadband_dps = float(t["gyro_deadband_dps"])
        self.accel_deadband_g = float(t["accel_deadband_g"])
        self.hold = int(t["stationary_hold_samples"])
        self.rebias_max_gyro_norm_dps = float(
            t.get("rebias_max_gyro_norm_dps", 1.0))
        self.zupt = bool(t["zupt_enabled"])
        self.gyro_deadband_on = bool(t["gyro_deadband_enabled"])
        self.accel_deadband_on = bool(t["accel_deadband_enabled"])
        self.lab = lab

    def describe(self) -> str:
        return (f"bias gyro {np.round(self.gyro_bias_dps, 4)} dps, "
                f"accel {np.round(self.accel_bias_g, 5)} g\n"
                f"thresholds{' (lab)' if self.lab else ''}: stationary when "
                f"|gyro| < {self.gyro_stationary_dps:g} dps and "
                f"||a|-1g| < {self.accel_stationary_dev_g:g} g for "
                f"{self.hold} samples; gyro dead-band "
                f"{self.gyro_deadband_dps:g} dps "
                f"({'on' if self.gyro_deadband_on else 'off'}); "
                f"accel dead-band {self.accel_deadband_g:g} g "
                f"({'on' if self.accel_deadband_on else 'off'}); "
                f"ZUPT {'on' if self.zupt else 'off'}")

    def as_dict(self) -> dict:
        """The tuning that produced a result, for the bundle's provenance."""
        return {
            "gyroStationaryDps": self.gyro_stationary_dps,
            "accelStationaryDevG": self.accel_stationary_dev_g,
            "gyroDeadbandDps": self.gyro_deadband_dps if self.gyro_deadband_on else 0.0,
            "accelDeadbandG": self.accel_deadband_g if self.accel_deadband_on else 0.0,
            "stationaryHoldSamples": self.hold,
            "zuptEnabled": self.zupt,
            "lab": self.lab,
        }


def load_config(path: Path | str | None = None, lab: bool = False) -> SensorConfig:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not an {SCHEMA} file")
    return SensorConfig(cfg, lab=lab)
