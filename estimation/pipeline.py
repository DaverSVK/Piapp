"""The whole estimation chain, in one call.

    capture.load  ->  sensor config  ->  gyro re-bias  ->  dead-bands
                  ->  stationary detection  ->  ESKF  ->  telemetry bundle

Deliberately free of any Bluetooth, serial or HTTP concern: it takes a CSV path
and returns a dict. That is what lets the same code run from the Pi app, from
the estimations.py command line, and from the tests, with nothing mocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from estimation import capture as capture_mod
from estimation import eskf, sensor_config, stationary as stat_mod
from estimation import telemetry as telemetry_mod


def parse_start_time(value: str | int) -> int:
    """An ISO-8601 timestamp or epoch value -> epoch milliseconds.

    Accepts seconds as readily as milliseconds, because both are called "epoch
    time" and mixing them up puts a session 55 years out - far enough that it
    is obvious, but only after someone has looked.
    """
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        ms = int(text)
        return ms * 1000 if ms < 10_000_000_000 else ms

    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)   # bare text means UTC
    return int(dt.timestamp() * 1000)


@dataclass
class Options:
    """Everything a caller may reasonably want to change."""

    device_id: str | None = None
    start_unix_ms: int | None = None
    heading_offset_deg: float = 0.0
    origin: tuple[float, float] = (0.0, 0.0)
    max_speed_kmh: float = telemetry_mod.DEFAULT_MAX_SPEED_KMH
    extended: bool = False
    planar: bool = True
    use_zaru: bool = True
    use_tilt: bool = True
    rebias_window_s: float = 3.0
    auto_q_scale: bool = True
    q_scale: float = 1.0
    lab_thresholds: bool = False
    config_path: Path | None = None


@dataclass
class Result:
    capture: capture_mod.Capture
    estimate: eskf.Estimate
    bundle: dict
    notes: list[str]


def _resolve_device_id(cap: capture_mod.Capture, opts: Options,
                       notes: list[str]) -> str:
    if opts.device_id:
        return opts.device_id
    if cap.device_id:
        return cap.device_id
    # Falling back to the hardware id keeps runs distinguishable rather than
    # silently merging two players under one blank name.
    hw = cap.meta.get("hardwareId")
    fallback = f"IMU-{hw[-8:]}" if hw else "IMU-UNKNOWN"
    notes.append(f"no device name in the capture; using {fallback} "
                 "(set one with --device-id, or name the board with SET_ID)")
    return fallback


def _resolve_start(cap: capture_mod.Capture, opts: Options,
                   notes: list[str]) -> int:
    if opts.start_unix_ms:
        if not cap.has_clock:
            # Worth saying loudly: every sampleAt in the bundle now rests on
            # this one number, and the device did not supply it.
            notes.append(
                "the capture carries no start time (recorded before the "
                "board's clock was set); every sampleAt is offset from the "
                "start time given by hand")
        return int(opts.start_unix_ms)
    if cap.has_clock:
        return cap.start_unix_ms
    raise ValueError(
        f"{cap.path.name} has no absolute start time: the board's clock had "
        "not been set when this capture was recorded, so sampleAt cannot be "
        "computed. Give --start-time, or connect the Pi app (or run "
        "tools/download_imu.py) before recording so the clock is set.")


def process(csv_path: Path | str, opts: Options | None = None) -> Result:
    """Run the full chain on one capture CSV."""
    opts = opts or Options()
    notes: list[str] = []

    cap = capture_mod.load(csv_path)
    cfg = sensor_config.load_config(opts.config_path, lab=opts.lab_thresholds)

    device_id = _resolve_device_id(cap, opts, notes)
    start_unix_ms = _resolve_start(cap, opts, notes)

    if cap.repairs:
        notes.extend(cap.repairs)
    if cap.skipped_rows:
        notes.append(f"{cap.skipped_rows} unparsable row(s) skipped")

    # Per-session gyro bias, when the capture actually starts at rest.
    bias_dps, mean_norm, ok = stat_mod.rebias_gyro(
        cap.t_ms, cap.gyro_dps, cfg, opts.rebias_window_s)
    if ok:
        gyro_bias = bias_dps
        notes.append(f"gyro re-biased from the first {opts.rebias_window_s:g} s "
                     f"({np.round(bias_dps, 4)} dps)")
    else:
        gyro_bias = cfg.gyro_bias_dps
        notes.append(
            f"start is not still enough to re-bias the gyro "
            f"(mean |gyro| {mean_norm:.2f} dps > "
            f"{cfg.rebias_max_gyro_norm_dps:g}); using the bench bias")
    rebias = {
        "applied": bool(ok),
        "windowS": opts.rebias_window_s,
        "meanGyroNormDps": (round(mean_norm, 3) if np.isfinite(mean_norm)
                            else None),
        "biasDps": [round(v, 5) for v in
                    (bias_dps if ok else cfg.gyro_bias_dps)],
    }

    accel_g, gyro_dps = stat_mod.apply_deadbands(cap.accel_g, cap.gyro_dps, cfg)
    stationary = stat_mod.detect_stationary(accel_g, gyro_dps, cfg)
    report = stat_mod.stationary_report(stationary, cap.median_period_s)

    if report["fraction"] == 0.0:
        notes.append(
            "the skater never stands still in this capture, so the filter gets "
            "no ZUPT/ZARU corrections at all - treat position as indicative only")

    # A skater's sensor is far noisier than the bench it was characterised on;
    # measuring how much and putting it in Q keeps the filter from trusting the
    # strapdown solution more than it deserves.
    q_a, q_g = (eskf.capture_noise_ratio(accel_g, gyro_dps, stationary, cfg)
                if opts.auto_q_scale else (opts.q_scale, opts.q_scale))

    noise = eskf.KFNoise(cfg, q_scale=q_a, q_scale_gyro=q_g)
    est = eskf.run(cap.t_ms, accel_g, gyro_dps, stationary, cfg, noise,
                   gyro_bias_dps=gyro_bias, use_zaru=opts.use_zaru,
                   use_tilt=opts.use_tilt, planar=opts.planar)

    config_summary = {
        **cfg.as_dict(),
        "qScaleAccel": round(q_a, 2),
        "qScaleGyro": round(q_g, 2),
        "planar": opts.planar,
        "zaru": opts.use_zaru,
        "tiltUpdate": opts.use_tilt,
        "maxSpeedKmh": opts.max_speed_kmh,
    }

    bundle = telemetry_mod.build_bundle(
        cap, est,
        device_id=device_id,
        start_unix_ms=start_unix_ms,
        stationary_report=report,
        config_summary=config_summary,
        gyro_rebias=rebias,
        heading_offset_deg=opts.heading_offset_deg,
        origin=opts.origin,
        max_speed_kmh=opts.max_speed_kmh,
        extended=opts.extended,
    )
    bundle["source"]["notes"] = notes

    return Result(capture=cap, estimate=est, bundle=bundle, notes=notes)
