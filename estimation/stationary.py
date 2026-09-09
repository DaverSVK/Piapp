"""Dead-bands, stationary detection and per-run gyro re-biasing.

These are what make dead reckoning survive more than a few seconds. The gyro
bias is the dominant error term by a wide margin: at the measured 0.095 dps it
tilts the frame by 9.5 degrees in 100 s, which leaks 1.6 m/s^2 of gravity into
the horizontal channels and double-integrates to kilometres. The bench bias
removes most of it; the stationary detector then lets the filter observe and
correct what is left, every time the skater stands still.

The stationary flag drives all three of the filter's corrections (ZUPT, ZARU
and the tilt update), so its thresholds are the single most consequential
tuning in the pipeline. They live in static_drift_config.json.
"""

from __future__ import annotations

import numpy as np

from estimation.sensor_config import SensorConfig


def detect_stationary(accel_g: np.ndarray, gyro_dps: np.ndarray,
                      cfg: SensorConfig) -> np.ndarray:
    """Per-sample stationary mask: quiet gyro, and |a| close to 1 g.

    Both conditions are needed. A constant translation has a quiet gyro but
    accelerates nothing, so gravity alone is not enough to catch it; a rotation
    about the gravity vector leaves |a| at 1 g, so the accelerometer alone is
    not enough either.

    The flag only latches after ``cfg.hold`` consecutive quiet samples (30 ms
    at 100 Hz), which keeps a single quiet sample between two strides from
    zeroing a perfectly good velocity.
    """
    quiet = ((np.abs(gyro_dps).max(axis=1) < cfg.gyro_stationary_dps) &
             (np.abs(np.linalg.norm(accel_g, axis=1) - 1.0)
              < cfg.accel_stationary_dev_g))
    if cfg.hold <= 1:
        return quiet

    stat = np.zeros(len(quiet), dtype=bool)
    run = 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        stat[i] = run >= cfg.hold
    return stat


def rebias_gyro(t_ms: np.ndarray, gyro_dps: np.ndarray, cfg: SensorConfig,
                window_s: float = 3.0):
    """Re-estimate gyro bias from the first still seconds of this capture.

    Returns ``(bias_dps, mean_norm_dps, ok)``. Per-session bias beats the
    stored bench figure when the capture actually starts at rest - the bench
    value was measured on a different day at a different temperature.

    The window only qualifies if it is quiet in an absolute sense: the mean of
    a window that still contains motion *is* that motion, and subtracting it
    makes the path worse than doing nothing. When ``ok`` is False the caller
    keeps the bench bias, and the filter's ZARU updates recover the rest once
    the skater does stand still.
    """
    if len(t_ms) < 2:
        return np.zeros(3), float("inf"), False

    sel = (t_ms - t_ms[0]) <= window_s * 1000.0
    if sel.sum() < 20:
        return np.zeros(3), float("inf"), False

    seg = gyro_dps[sel]
    mean_norm = float(np.linalg.norm(seg, axis=1).mean())
    return (seg.mean(axis=0), mean_norm,
            mean_norm <= cfg.rebias_max_gyro_norm_dps)


def apply_deadbands(accel_g: np.ndarray, gyro_dps: np.ndarray,
                    cfg: SensorConfig):
    """Zero the channels that are indistinguishable from noise.

    The gyro dead-band is on by default at 6 sigma (0.27 dps): below it a rate
    is noise, and integrating noise is how a heading walks away.

    The accelerometer dead-band is off by default. At the same 6 sigma it is
    0.05 m/s^2, which clips far more real skating acceleration than it removes
    noise - and unlike a rate error, a small acceleration error does not
    accumulate into the attitude. Turn it on for bench and desk data.
    """
    if cfg.gyro_deadband_on:
        gyro_dps = np.where(np.abs(gyro_dps) < cfg.gyro_deadband_dps,
                            0.0, gyro_dps)
    if cfg.accel_deadband_on:
        accel_g = np.where(np.abs(accel_g) < cfg.accel_deadband_g, 0.0, accel_g)
    return accel_g, gyro_dps


def stationary_report(stationary: np.ndarray, period_s: float) -> dict:
    """How still the capture was, for the bundle's quality block."""
    n = len(stationary)
    if n == 0:
        return {"fraction": 0.0, "longest_s": 0.0, "intervals": 0}

    longest = run = 0
    intervals = 0
    prev = False
    for s in stationary:
        if s:
            run += 1
            if not prev:
                intervals += 1
            longest = max(longest, run)
        else:
            run = 0
        prev = bool(s)

    return {
        "fraction": float(stationary.mean()),
        "longest_s": longest * period_s,
        "intervals": intervals,
    }
