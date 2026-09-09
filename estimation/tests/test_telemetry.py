"""The API contract: field names, units, timestamps, and the honesty guards."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from estimation import pipeline, telemetry as tele
from estimation.tests.conftest import (mark_row, still_rows, write_capture,
                                       yaw_rows)

# The exact ten keys the backend expects (thingy53_pi_app_flow.md section 25).
REQUIRED_KEYS = {
    "deviceId", "sampleAt", "speedKmh", "accelMps2", "turnRateDps",
    "pitchDeg", "rollDeg", "yawDeg", "posX", "posY",
}

START_MS = 1788877265000


def _bundle(tmp_path, rows, **kw):
    path = write_capture(tmp_path / "run_0063.csv", rows)
    opts = pipeline.Options(device_id="DEV-12", start_unix_ms=START_MS, **kw)
    return pipeline.process(path, opts).bundle


def test_records_carry_exactly_the_required_keys(tmp_path):
    bundle = _bundle(tmp_path, still_rows(200))
    assert set(bundle["telemetry"][0]) == REQUIRED_KEYS


def test_extended_adds_the_optional_backend_fields(tmp_path):
    bundle = _bundle(tmp_path, still_rows(200), extended=True)
    extra = {"impactG", "fall", "onIce", "distance", "collision",
             "timeOnIce", "substitutionCount"}
    assert set(bundle["telemetry"][0]) == REQUIRED_KEYS | extra


def test_sample_at_is_absolute_iso_utc_and_increasing(tmp_path):
    bundle = _bundle(tmp_path, still_rows(50))
    stamps = [r["sampleAt"] for r in bundle["telemetry"]]

    assert all(s.endswith("Z") for s in stamps)
    parsed = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps]
    assert parsed == sorted(parsed)
    assert all(p.tzinfo == timezone.utc for p in parsed)

    # The first sample is the capture start, and 10 ms separates samples.
    assert parsed[0] == datetime.fromtimestamp(START_MS / 1000, timezone.utc)
    assert (parsed[1] - parsed[0]).total_seconds() == pytest.approx(0.01)


def test_speed_is_km_per_hour(tmp_path):
    """Guards the 3.6 factor, the easiest thing in the pipeline to get wrong."""
    bundle = _bundle(tmp_path, still_rows(200))
    result = pipeline.process(
        write_capture(tmp_path / "run_0063.csv", still_rows(200)),
        pipeline.Options(device_id="DEV-12", start_unix_ms=START_MS))

    expected = result.estimate.speed_ms * 3.6
    got = [r["speedKmh"] for r in bundle["telemetry"]]
    assert got[10] == pytest.approx(expected[10], abs=0.01)


def test_standing_still_reports_no_speed_and_no_acceleration(tmp_path):
    bundle = _bundle(tmp_path, still_rows(400))
    tail = bundle["telemetry"][100:]

    assert max(r["speedKmh"] for r in tail) < 0.5
    # Below the tangent threshold there is no direction of travel to project
    # onto, so the honest answer is zero rather than a projection onto noise.
    assert all(r["accelMps2"] == 0.0 for r in tail)


def test_yaw_is_wrapped_into_zero_to_360(tmp_path):
    bundle = _bundle(tmp_path, yaw_rows(90.0, 10.0))
    yaws = [r["yawDeg"] for r in bundle["telemetry"]]

    assert all(0.0 <= y < 360.0 for y in yaws)
    assert max(yaws) > 180.0, "10 s at 90 deg/s must pass a full turn"


def test_heading_offset_and_origin_place_the_run(tmp_path):
    plain = _bundle(tmp_path, still_rows(100))
    moved = _bundle(tmp_path, still_rows(100),
                    heading_offset_deg=90.0, origin=(12.5, 28.3))

    assert moved["telemetry"][0]["yawDeg"] == pytest.approx(
        (plain["telemetry"][0]["yawDeg"] + 90.0) % 360.0, abs=0.01)
    assert moved["telemetry"][0]["posX"] == pytest.approx(
        plain["telemetry"][0]["posX"] + 12.5, abs=0.01)
    assert moved["telemetry"][0]["posY"] == pytest.approx(
        plain["telemetry"][0]["posY"] + 28.3, abs=0.01)
    assert "90" in moved["source"]["quality"]["yawReference"]


def test_speed_clamp_saturates_and_counts():
    speed = np.array([10.0, 50.0, 80.0])

    out, clamped = tele.clamp_speed(speed, 45.0)

    assert list(out) == [10.0, 45.0, 45.0]
    assert clamped == 2


def test_speed_clamp_can_be_disabled():
    speed = np.array([10.0, 500.0])
    out, clamped = tele.clamp_speed(speed, 0.0)
    assert list(out) == [10.0, 500.0]
    assert clamped == 0


def test_marks_get_absolute_timestamps(tmp_path):
    rows = still_rows(100)
    rows.insert(50, mark_row(500))
    bundle = _bundle(tmp_path, rows)

    assert len(bundle["marks"]) == 1
    mark = bundle["marks"][0]
    assert mark["tMs"] == 500
    expected = datetime.fromtimestamp((START_MS + 500) / 1000, timezone.utc)
    assert datetime.fromisoformat(mark["sampleAt"].replace("Z", "+00:00")) == expected


def test_bundle_carries_the_four_things_the_publisher_posts(tmp_path):
    bundle = _bundle(tmp_path, still_rows(200), extended=True)

    assert bundle["schema"] == tele.BUNDLE_SCHEMA
    assert bundle["telemetry"]
    assert bundle["playerStats"]
    assert bundle["generalStats"]
    assert bundle["source"]["startIso"].endswith("Z")


def test_player_stats_use_the_backend_field_names(tmp_path):
    bundle = _bundle(tmp_path, still_rows(200), extended=True)

    assert set(bundle["playerStats"][0]) == {
        "deviceId", "avgSpeedKmh", "maxSpeedKmh", "avgAccelMps2",
        "maxAccelMps2", "distanceM", "impactsCount", "maxImpactG",
        "timeOnIceSec", "shiftsCount", "avgTurnRateDps", "turnsCount",
        "fallsCount", "substitutionCount",
    }
    assert set(bundle["generalStats"]) == {
        "avgSpeedKmh", "avgAccelMps2", "totalDistanceM", "impactsCount",
        "maxImpactG", "fallsCount",
    }


def test_quality_block_reports_what_the_result_rests_on(tmp_path):
    bundle = _bundle(tmp_path, still_rows(400))
    q = bundle["source"]["quality"]

    for key in ("zuptCount", "stationaryFraction", "finalPosSigmaM",
                "clampedSamples", "yawReference", "gyroRebias",
                "measuredRateHz"):
        assert key in q, key
    assert "relative to the capture start" in q["yawReference"]


def test_a_capture_without_a_clock_is_refused_rather_than_guessed(tmp_path):
    path = write_capture(tmp_path / "run.csv", still_rows(100))

    with pytest.raises(ValueError, match="no absolute start time"):
        pipeline.process(path, pipeline.Options(device_id="DEV-12"))


def test_start_time_from_the_sidecar_is_used(tmp_path, meta_file):
    path = write_capture(tmp_path / "run_0063.csv", still_rows(100))
    meta_file(path, startUnixMs=START_MS, deviceId="DEV-09")

    bundle = pipeline.process(path, pipeline.Options()).bundle

    assert bundle["source"]["deviceId"] == "DEV-09"
    assert bundle["source"]["startUnixMs"] == START_MS


def test_tangential_acceleration_is_signed_along_the_path():
    """Speeding up is positive, slowing down negative - not a magnitude."""
    class FakeEstimate:
        vel = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        acc = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    out = tele.tangential_accel(FakeEstimate())

    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(-2.0)
    assert out[2] == 0.0, "no direction of travel below the speed threshold"
