"""Shared fixtures: synthetic captures written to tmp_path.

Everything here builds real CSV text in the exact format
tools/download_imu.py writes, so the tests exercise the same parsing path a
downloaded capture takes - no mocks, and no dependence on a device.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pi_app.storage import captures as captures_mod   # noqa: E402

HEADER = captures_mod.CSV_HEADER

# A still sensor reads +1 g on z with this firmware's sign convention, and the
# bench bias from static_drift_config.json.
REST_ACCEL_G = (-0.007515, -0.008061, 1.0 - 0.003859)
REST_GYRO_DPS = (0.073395, -0.094446, -0.095111)


def sample_row(t_ms, accel_g=REST_ACCEL_G, gyro_dps=REST_GYRO_DPS) -> str:
    return "%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f" % (t_ms, *accel_g, *gyro_dps)


def mark_row(t_ms) -> str:
    return "MARK,%u,,,,," % t_ms


def write_capture(path: Path, rows, header: str = HEADER,
                  bom: bool = False) -> Path:
    text = header + "\n" + "\n".join(rows) + "\n"
    path.write_text(text, encoding="utf-8-sig" if bom else "utf-8",
                    newline="")
    return path


def still_rows(n: int = 400, period_ms: int = 10, start_ms: int = 0,
               noise: float = 0.0, seed: int = 0) -> list[str]:
    """A capture of a sensor lying still, optionally with white noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a = np.array(REST_ACCEL_G)
        g = np.array(REST_GYRO_DPS)
        if noise:
            a = a + rng.normal(0.0, noise * 0.000841, 3)
            g = g + rng.normal(0.0, noise * 0.043205, 3)
        rows.append(sample_row(start_ms + i * period_ms, tuple(a), tuple(g)))
    return rows


def yaw_rows(rate_dps: float, seconds: float, period_ms: int = 10,
             start_ms: int = 0) -> list[str]:
    """A sensor turning about the world vertical at a constant rate.

    Level and stationary in translation, so the accelerometer keeps reading
    gravity on z while the z gyro reads the turn rate: exactly the case where
    the recovered heading should equal rate * time.
    """
    n = int(seconds * 1000 / period_ms)
    rows = []
    for i in range(n):
        rows.append(sample_row(
            start_ms + i * period_ms,
            accel_g=REST_ACCEL_G,
            gyro_dps=(REST_GYRO_DPS[0], REST_GYRO_DPS[1],
                      REST_GYRO_DPS[2] + rate_dps)))
    return rows


@pytest.fixture
def still_capture(tmp_path) -> Path:
    return write_capture(tmp_path / "run_0001.csv", still_rows(400))


@pytest.fixture
def meta_file():
    """Write a metadata sidecar next to a capture, with overrides applied."""
    def _write(csv_path: Path, **overrides) -> Path:
        meta = {
            "schema": captures_mod.META_SCHEMA,
            "version": captures_mod.META_VERSION,
            "deviceId": "DEV-07",
            "hardwareId": "d2f1a03b5c6e7890",
            "runId": 63,
            "startUnixMs": 1788877265000,
            "clockSyncAgeMs": 7123,
            "periodMs": 10,
            "accelFsG": 8.0,
            "gyroFsDps": 500.0,
            "marksMs": [],
        }
        meta.update(overrides)
        path = Path(csv_path).with_name(Path(csv_path).stem + ".meta.json")
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path
    return _write
