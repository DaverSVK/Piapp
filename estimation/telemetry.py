"""Turn an estimator run into the telemetry the Smart Jersey backend receives.

The per-sample record is fixed by the backend:

    {"deviceId", "sampleAt", "speedKmh", "accelMps2", "turnRateDps",
     "pitchDeg", "rollDeg", "yawDeg", "posX", "posY"}

``--extended`` adds the fields the dummy sender also posts (impactG, distance,
timeOnIce, onIce, fall, collision, substitutionCount), so the aggregate blocks
stay self-consistent with what the backend already accepts.

What the numbers do and do not mean
-----------------------------------

``yawDeg``, ``posX`` and ``posY`` are **relative to where the capture started**.
A 6-axis IMU has no compass, so heading has no absolute reference: the capture
begins at heading 0 at the origin, and ``--heading-offset`` / ``--origin`` place
it on the rink if you know where the skater actually started.

Absolute position also drifts, and not slowly. The bench characterisation
(static_drift_config.json, ``error_budget``) records that the residual gyro bias
alone integrates to hundreds of metres over a couple of minutes, and no
calibration removes that term - only the ZUPTs do, and only when the skater
actually stops. Every bundle therefore carries ``finalPosSigmaM`` from the
filter's own covariance, and speed is clamped for reporting with the count of
clamped samples recorded. Speed and turn rate are far more trustworthy than
position, because both are corrected every time the skater stands still.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from estimation.capture import Capture
from estimation.eskf import Estimate

# Below this the velocity direction is noise, so a tangential projection onto it
# is meaningless. 0.3 m/s is about one slow step.
MIN_SPEED_FOR_TANGENT_MS = 0.3

# Reporting-only ceiling. Elite hockey tops out near 40 km/h; anything above
# this is filter divergence, and letting it through would put a fictional
# number in front of a coach.
DEFAULT_MAX_SPEED_KMH = 45.0

# An impact worth counting, in g of total specific force.
IMPACT_G_THRESHOLD = 4.0

BUNDLE_SCHEMA = "imu_telemetry_bundle"
BUNDLE_VERSION = 1


def iso_utc(value: datetime) -> str:
    """ISO-8601 UTC with milliseconds, the format the API expects."""
    return (value.astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"))


def _wrap360(deg: np.ndarray) -> np.ndarray:
    return np.mod(deg, 360.0)


def tangential_accel(est: Estimate) -> np.ndarray:
    """Acceleration along the direction of travel, in m/s^2.

    Positive means speeding up. This is what "acceleration" means for a skater;
    the raw magnitude would report a hard turn at constant speed as a large
    acceleration, which is true of the vector and useless to a coach.

    Below MIN_SPEED_FOR_TANGENT_MS there is no direction of travel to project
    onto, so it reports 0 - which is also the honest answer while standing
    still.
    """
    v_xy = est.vel[:, :2]
    a_xy = est.acc[:, :2]
    speed = np.linalg.norm(v_xy, axis=1)

    moving = speed >= MIN_SPEED_FOR_TANGENT_MS
    out = np.zeros(len(speed))
    if moving.any():
        unit = v_xy[moving] / speed[moving, None]
        out[moving] = np.einsum("ij,ij->i", a_xy[moving], unit)
    return out


def clamp_speed(speed_kmh: np.ndarray, ceiling: float):
    """Saturate implausible speeds for reporting, and count how many."""
    if ceiling <= 0:
        return speed_kmh, 0
    over = speed_kmh > ceiling
    if not over.any():
        return speed_kmh, 0
    out = speed_kmh.copy()
    out[over] = ceiling
    return out, int(over.sum())


def build_records(capture: Capture, est: Estimate, device_id: str,
                  start_unix_ms: int, heading_offset_deg: float = 0.0,
                  origin: tuple[float, float] = (0.0, 0.0),
                  max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
                  extended: bool = False):
    """One telemetry record per sample. Returns (records, quality)."""
    n = est.n
    speed_kmh_raw = est.speed_ms * 3.6
    speed_kmh, clamped = clamp_speed(speed_kmh_raw, max_speed_kmh)

    accel_mps2 = tangential_accel(est)
    turn_dps = est.yaw_rate_dps
    roll_deg = np.degrees(est.euler[:, 0])
    pitch_deg = np.degrees(est.euler[:, 1])
    yaw_deg = _wrap360(np.degrees(est.euler[:, 2]) + heading_offset_deg)
    pos_x = est.pos[:, 0] + origin[0]
    pos_y = est.pos[:, 1] + origin[1]

    t_ms = capture.t_ms
    base = datetime.fromtimestamp(start_unix_ms / 1000.0, tz=timezone.utc)
    t0 = float(t_ms[0])

    # Extended-only series.
    if extended:
        accel_mag_g = np.linalg.norm(capture.accel_g, axis=1)
        step = np.linalg.norm(np.diff(est.pos[:, :2], axis=0), axis=1)
        distance_m = np.concatenate([[0.0], np.cumsum(step)])

    records = []
    for i in range(n):
        sample_at = base + timedelta(milliseconds=float(t_ms[i]) - t0)
        rec = {
            "deviceId": device_id,
            "sampleAt": iso_utc(sample_at),
            "speedKmh": round(float(speed_kmh[i]), 2),
            "accelMps2": round(float(accel_mps2[i]), 2),
            "turnRateDps": round(float(turn_dps[i]), 2),
            "pitchDeg": round(float(pitch_deg[i]), 2),
            "rollDeg": round(float(roll_deg[i]), 2),
            "yawDeg": round(float(yaw_deg[i]), 2),
            "posX": round(float(pos_x[i]), 2),
            "posY": round(float(pos_y[i]), 2),
        }
        if extended:
            impact = float(accel_mag_g[i])
            rec.update({
                "impactG": round(impact, 2) if impact >= IMPACT_G_THRESHOLD else 0.0,
                "fall": False,
                "onIce": True,
                "distance": round(float(distance_m[i]), 2),
                "collision": impact >= IMPACT_G_THRESHOLD,
                "timeOnIce": round(float(est.t_s[i]), 2),
                "substitutionCount": 0,
            })
        records.append(rec)

    quality = {
        "clampedSamples": clamped,
        "maxSpeedKmhRaw": round(float(speed_kmh_raw.max()), 2) if n else 0.0,
    }
    return records, quality


def player_stats(device_id: str, records: list[dict], est: Estimate) -> list[dict]:
    """The aggregate block, with the field names send_stats.py already posts."""
    if not records:
        return []

    speeds = [r["speedKmh"] for r in records]
    accels = [abs(r["accelMps2"]) for r in records]
    turns = [abs(r["turnRateDps"]) for r in records]
    impacts = [r["impactG"] for r in records
               if r.get("impactG", 0) >= IMPACT_G_THRESHOLD]
    duration_s = float(est.t_s[-1]) if est.n else 0.0

    return [{
        "deviceId": device_id,
        "avgSpeedKmh": round(sum(speeds) / len(speeds), 2),
        "maxSpeedKmh": round(max(speeds), 2),
        "avgAccelMps2": round(sum(accels) / len(accels), 2),
        "maxAccelMps2": round(max(accels), 2),
        "distanceM": round(est.distance_m(), 2),
        "impactsCount": len(impacts),
        "maxImpactG": round(max(impacts), 2) if impacts else 0.0,
        "timeOnIceSec": round(duration_s),
        "shiftsCount": 1,
        "avgTurnRateDps": round(sum(turns) / len(turns), 2),
        # A sustained turn, not the per-sample noise: 20 deg/s is a real corner.
        "turnsCount": sum(1 for t in turns if t >= 20.0),
        "fallsCount": 0,
        "substitutionCount": 0,
    }]


def general_stats(summary: dict) -> dict:
    """The session-completing block, derived from the player stats."""
    return {
        "avgSpeedKmh": summary["avgSpeedKmh"],
        "avgAccelMps2": summary["avgAccelMps2"],
        "totalDistanceM": summary["distanceM"],
        "impactsCount": summary["impactsCount"],
        "maxImpactG": summary["maxImpactG"],
        "fallsCount": summary["fallsCount"],
    }


def build_bundle(capture: Capture, est: Estimate, *, device_id: str,
                 start_unix_ms: int, stationary_report: dict,
                 config_summary: dict, gyro_rebias: dict,
                 heading_offset_deg: float = 0.0,
                 origin: tuple[float, float] = (0.0, 0.0),
                 max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
                 extended: bool = False) -> dict:
    """Everything the publisher needs, in one JSON-serialisable object.

    Bundling the aggregates with the samples means publishing never has to
    re-run the estimator, so a retry after a network failure is a file upload
    rather than a recomputation.
    """
    records, tele_quality = build_records(
        capture, est, device_id, start_unix_ms,
        heading_offset_deg=heading_offset_deg, origin=origin,
        max_speed_kmh=max_speed_kmh, extended=extended)

    stats = player_stats(device_id, records, est)

    base = datetime.fromtimestamp(start_unix_ms / 1000.0, tz=timezone.utc)
    t0 = float(capture.t_ms[0]) if capture.n else 0.0
    marks = [{"tMs": int(m),
              "sampleAt": iso_utc(base + timedelta(milliseconds=m - t0))}
             for m in capture.marks_ms]

    meta = capture.meta
    quality = {
        "zuptCount": est.counts.get("zupt", 0),
        "zaruCount": est.counts.get("zaru", 0),
        "tiltCount": est.counts.get("tilt", 0),
        "timingGaps": est.counts.get("gaps", 0),
        "levellingSamples": est.counts.get("levelling_samples", 0),
        "initRollPitchDeg": [round(v, 2) for v in est.init_euler_deg[:2]],
        "stationaryFraction": round(stationary_report["fraction"], 4),
        "longestStationaryS": round(stationary_report["longest_s"], 2),
        "stationaryIntervals": stationary_report["intervals"],
        "finalPosSigmaM": round(est.final_pos_sigma_m, 1),
        "finalGyroBiasDps": [round(v, 4) for v in np.degrees(est.gyro_bias[-1])],
        "gyroRebias": gyro_rebias,
        "measuredRateHz": round(capture.measured_rate_hz, 2),
        "skippedRows": capture.skipped_rows,
        "timingRepairs": capture.repairs,
        "yawReference": (
            "relative to the capture start"
            if heading_offset_deg == 0.0
            else f"capture start + {heading_offset_deg:g} deg"),
        "positionReference": (
            "origin at the capture start"
            if origin == (0.0, 0.0)
            else f"origin at {origin}"),
        **tele_quality,
    }

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "version": BUNDLE_VERSION,
        "source": {
            "runCsv": capture.path.name,
            "deviceId": device_id,
            "hardwareId": meta.get("hardwareId"),
            "runId": meta.get("runId"),
            "slot": meta.get("slot"),
            "captureState": meta.get("state"),
            "startUnixMs": start_unix_ms,
            "startIso": iso_utc(base),
            "clockSyncAgeMs": meta.get("clockSyncAgeMs"),
            "sampleCount": est.n,
            "durationS": round(capture.duration_s, 2),
            "assumed": meta.get("assumed", []),
            "estimator": config_summary,
            "quality": quality,
        },
        "marks": marks,
        "telemetry": records,
        "playerStats": stats,
    }
    if stats:
        bundle["generalStats"] = general_stats(stats[0])
    return bundle
