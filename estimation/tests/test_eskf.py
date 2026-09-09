"""The filter itself: does it stay still, does it turn, does it stay sane."""

from __future__ import annotations

import math

import numpy as np
import pytest

from estimation import capture as cap_mod
from estimation import eskf, sensor_config, stationary as stat_mod
from estimation.tests.conftest import still_rows, write_capture, yaw_rows


@pytest.fixture(scope="module")
def cfg():
    return sensor_config.load_config()


def _run(path, cfg, **kw):
    cap = cap_mod.load(path)
    accel, gyro = stat_mod.apply_deadbands(cap.accel_g, cap.gyro_dps, cfg)
    stationary = stat_mod.detect_stationary(accel, gyro, cfg)
    noise = eskf.KFNoise(cfg)
    return cap, stationary, eskf.run(cap.t_ms, accel, gyro, stationary,
                                     cfg, noise, **kw)


# ------------------------------------------------------------------ quaternions


def test_quaternion_helpers_round_trip():
    for rpy in [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (-1.0, 0.4, -2.5)]:
        q = eskf.quat_from_euler(*rpy)
        back = eskf.matrix_to_euler(eskf.quat_to_matrix(q))
        assert back == pytest.approx(rpy, abs=1e-9)


def test_rotation_vector_is_exact_at_zero():
    q = eskf.quat_from_rotvec(np.zeros(3))
    assert q == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_small_rotation_matches_the_analytic_angle():
    v = np.array([0.0, 0.0, math.radians(30.0)])
    R = eskf.quat_to_matrix(eskf.quat_from_rotvec(v))
    assert eskf.matrix_to_euler(R)[2] == pytest.approx(math.radians(30.0))


# ------------------------------------------------------------------- behaviour


def test_a_still_capture_does_not_wander(tmp_path, cfg):
    """The whole point of ZUPT: standing still must produce no motion."""
    path = write_capture(tmp_path / "still.csv", still_rows(1000, noise=1.0))
    _, stationary, est = _run(path, cfg)

    assert stationary.mean() > 0.9, "a still bench capture must read as still"
    assert est.counts["zupt"] > 500
    assert est.counts["zaru"] > 500

    drift = np.linalg.norm(est.pos[-1][:2])
    assert drift < 0.05, f"10 s of standing still drifted {drift:.3f} m"
    assert est.speed_ms[-1] < 0.01


def test_zupt_off_drifts_further_than_zupt_on(tmp_path, cfg):
    """Guards the corrections against being quietly disabled."""
    path = write_capture(tmp_path / "still.csv", still_rows(1000, noise=1.0))

    cap = cap_mod.load(path)
    accel, gyro = stat_mod.apply_deadbands(cap.accel_g, cap.gyro_dps, cfg)
    stationary = stat_mod.detect_stationary(accel, gyro, cfg)
    noise = eskf.KFNoise(cfg)

    with_zupt = eskf.run(cap.t_ms, accel, gyro, stationary, cfg, noise)
    without = eskf.run(cap.t_ms, accel, gyro, np.zeros_like(stationary),
                       cfg, noise)

    assert (np.linalg.norm(with_zupt.pos[-1][:2])
            <= np.linalg.norm(without.pos[-1][:2]))


def test_gyro_bias_uncertainty_shrinks_while_still(tmp_path, cfg):
    """ZARU is the only thing that observes the vertical gyro bias."""
    path = write_capture(tmp_path / "still.csv", still_rows(1000, noise=1.0))
    _, _, est = _run(path, cfg)

    assert est.sigma[-1, 12:15].max() < est.sigma[0, 12:15].max()


def test_yaw_is_left_unobserved(tmp_path, cfg):
    """A 6-axis IMU cannot see heading; the covariance must admit it."""
    path = write_capture(tmp_path / "still.csv", still_rows(600, noise=1.0))
    _, _, est = _run(path, cfg)

    roll_pitch_sigma = math.degrees(max(est.sigma[-1, 6], est.sigma[-1, 7]))
    yaw_sigma = math.degrees(est.sigma[-1, 8])
    assert roll_pitch_sigma < 2.0, "tilt updates should pin roll and pitch"
    assert yaw_sigma > roll_pitch_sigma * 2, "yaw must stay far less certain"


def test_constant_turn_rate_recovers_the_heading(tmp_path, cfg):
    """Integrating a known rate for a known time must give the known angle."""
    rate, seconds = 45.0, 4.0
    path = write_capture(tmp_path / "turn.csv", yaw_rows(rate, seconds))
    _, _, est = _run(path, cfg)

    turned = math.degrees(est.euler[-1, 2])
    assert turned == pytest.approx(rate * seconds, rel=0.05)


def test_turn_rate_is_reported_about_the_world_vertical(tmp_path, cfg):
    path = write_capture(tmp_path / "turn.csv", yaw_rows(45.0, 4.0))
    _, _, est = _run(path, cfg)

    # Skip the settling transient at the start of the capture.
    assert np.median(est.yaw_rate_dps[100:]) == pytest.approx(45.0, abs=2.0)


def test_planar_update_pins_the_vertical_channel(tmp_path, cfg):
    """A rink is flat, so the sensor cannot climb.

    Uses a turning capture rather than a still one: while the skater is moving
    there are no ZUPTs, so the vertical channel is free to absorb tilt error -
    which is exactly the case the planar pseudo-measurement exists for. On a
    still capture the ZUPTs pin everything and the comparison would be noise.
    """
    path = write_capture(tmp_path / "turn.csv", yaw_rows(45.0, 30.0))
    cap = cap_mod.load(path)
    accel, gyro = stat_mod.apply_deadbands(cap.accel_g, cap.gyro_dps, cfg)
    stationary = stat_mod.detect_stationary(accel, gyro, cfg)
    assert not stationary.any(), "a turning capture must not read as still"

    noise = eskf.KFNoise(cfg)
    flat = eskf.run(cap.t_ms, accel, gyro, stationary, cfg, noise, planar=True)
    free = eskf.run(cap.t_ms, accel, gyro, stationary, cfg, noise, planar=False)

    assert flat.counts["planar"] > 0
    assert abs(flat.pos[-1, 2]) < abs(free.pos[-1, 2])
    # Within the pseudo-measurement's own 1-sigma of how far the sensor may
    # ride above the ice.
    assert abs(flat.pos[-1, 2]) < 0.5


def test_timing_gaps_are_carried_not_integrated(tmp_path, cfg):
    """A hole in the capture must not manufacture motion."""
    rows = still_rows(200) + still_rows(200, start_ms=10_000)
    path = write_capture(tmp_path / "gap.csv", rows)
    _, _, est = _run(path, cfg)

    assert est.counts["gaps"] == 1


def test_output_arrays_are_all_the_same_length(tmp_path, cfg):
    path = write_capture(tmp_path / "still.csv", still_rows(300))
    cap, _, est = _run(path, cfg)

    n = cap.n
    for name in ("pos", "vel", "acc", "euler", "gyro_world",
                 "accel_bias", "gyro_bias", "sigma", "stationary"):
        assert len(getattr(est, name)) == n, name


def test_covariance_stays_finite_and_non_negative(tmp_path, cfg):
    path = write_capture(tmp_path / "turn.csv", yaw_rows(120.0, 6.0))
    _, _, est = _run(path, cfg)

    assert np.all(np.isfinite(est.sigma))
    assert np.all(est.sigma >= 0.0)
