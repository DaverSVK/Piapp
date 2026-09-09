"""Dead-bands, the stationary gate, and per-run gyro re-biasing."""

from __future__ import annotations

import numpy as np
import pytest

from estimation import sensor_config, stationary as stat


@pytest.fixture(scope="module")
def cfg():
    return sensor_config.load_config()


def _rest(n, gyro_dps=0.0, accel_norm_g=1.0):
    accel = np.zeros((n, 3))
    accel[:, 2] = accel_norm_g
    gyro = np.full((n, 3), gyro_dps)
    return accel, gyro


def test_still_data_is_stationary(cfg):
    accel, gyro = _rest(50)
    mask = stat.detect_stationary(accel, gyro, cfg)

    # The first cfg.hold - 1 samples cannot latch: the flag needs that many
    # consecutive quiet samples behind it. Everything after them must be still.
    assert mask[cfg.hold - 1:].all()
    assert not mask[:cfg.hold - 1].any()


def test_a_fast_rotation_is_not_stationary(cfg):
    accel, gyro = _rest(50, gyro_dps=cfg.gyro_stationary_dps + 1.0)
    assert not stat.detect_stationary(accel, gyro, cfg).any()


def test_acceleration_alone_breaks_the_gate(cfg):
    """A quiet gyro is not enough: a shove has no rotation but is not rest."""
    accel, gyro = _rest(50, accel_norm_g=1.0 + cfg.accel_stationary_dev_g * 2)
    assert not stat.detect_stationary(accel, gyro, cfg).any()


def test_flag_latches_only_after_the_hold(cfg):
    n = 20
    accel, gyro = _rest(n)
    gyro[:5] = cfg.gyro_stationary_dps + 1.0     # moving, then still

    mask = stat.detect_stationary(accel, gyro, cfg)

    assert not mask[:5].any()
    # The first cfg.hold quiet samples do not latch; the next one does.
    assert not mask[5:5 + cfg.hold - 1].any()
    assert mask[5 + cfg.hold - 1]


def test_a_single_quiet_sample_does_not_latch(cfg):
    """One quiet sample between strides must not zero a good velocity."""
    accel, gyro = _rest(20, gyro_dps=cfg.gyro_stationary_dps + 1.0)
    gyro[10] = 0.0

    assert not stat.detect_stationary(accel, gyro, cfg).any()


def test_gyro_deadband_zeros_only_below_the_threshold(cfg):
    gyro = np.array([[cfg.gyro_deadband_dps * 0.5, cfg.gyro_deadband_dps * 2,
                      -cfg.gyro_deadband_dps * 0.5]])
    accel = np.zeros((1, 3))

    _, out = stat.apply_deadbands(accel, gyro, cfg)

    assert out[0, 0] == 0.0
    assert out[0, 1] == gyro[0, 1]
    assert out[0, 2] == 0.0


def test_accel_deadband_is_off_by_default(cfg):
    """At 6 sigma it clips real skating harder than it removes noise."""
    assert not cfg.accel_deadband_on
    accel = np.full((1, 3), cfg.accel_deadband_g * 0.5)
    out, _ = stat.apply_deadbands(accel, np.zeros((1, 3)), cfg)
    assert np.array_equal(out, accel)


def test_rebias_uses_a_quiet_start(cfg):
    n = 500
    t_ms = np.arange(n) * 10.0
    gyro = np.full((n, 3), 0.05)          # a small, constant, quiet bias

    bias, mean_norm, ok = stat.rebias_gyro(t_ms, gyro, cfg, window_s=3.0)

    assert ok
    assert bias == pytest.approx([0.05, 0.05, 0.05])
    assert mean_norm < cfg.rebias_max_gyro_norm_dps


def test_rebias_refuses_a_moving_start(cfg):
    """The mean of a moving window is the motion, not the bias."""
    n = 500
    t_ms = np.arange(n) * 10.0
    gyro = np.full((n, 3), 20.0)

    _, mean_norm, ok = stat.rebias_gyro(t_ms, gyro, cfg, window_s=3.0)

    assert not ok
    assert mean_norm > cfg.rebias_max_gyro_norm_dps


def test_rebias_refuses_too_short_a_window(cfg):
    t_ms = np.arange(5) * 10.0
    _, _, ok = stat.rebias_gyro(t_ms, np.zeros((5, 3)), cfg, window_s=3.0)
    assert not ok


def test_stationary_report_counts_intervals_and_longest(cfg):
    mask = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)

    report = stat.stationary_report(mask, period_s=0.01)

    assert report["intervals"] == 2
    assert report["longest_s"] == pytest.approx(0.03)
    assert report["fraction"] == pytest.approx(5 / 9)


def test_stationary_report_handles_an_empty_capture(cfg):
    report = stat.stationary_report(np.zeros(0, dtype=bool), 0.01)
    assert report == {"fraction": 0.0, "longest_s": 0.0, "intervals": 0}
