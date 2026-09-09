"""Loading captures: markers, timing repair, metadata, malformed input."""

from __future__ import annotations

import pytest

from estimation import capture as cap_mod
from estimation.tests.conftest import (HEADER, mark_row, sample_row,
                                       still_rows, write_capture)


def test_markers_are_kept_and_excluded_from_samples(tmp_path):
    rows = still_rows(5)
    rows.insert(2, mark_row(115))
    rows.append(mark_row(49338))
    path = write_capture(tmp_path / "run.csv", rows)

    cap = cap_mod.load(path)

    assert cap.n == 5, "marker rows must not become samples"
    assert cap.marks_ms == [115.0, 49338.0]
    assert cap.skipped_rows == 0


def test_marker_time_comes_from_the_second_column(tmp_path):
    # The downloader writes MARK,<t_ms>,,,,, - the time is in ax_g, not t_ms.
    path = write_capture(tmp_path / "run.csv", still_rows(3) + ["MARK,777,,,,,"])
    assert cap_mod.load(path).marks_ms == [777.0]


def test_stale_leading_sample_is_dropped(tmp_path):
    # run_0058 starts at 732 ms then restarts at 10; run_0065 starts at 6016.
    rows = [sample_row(6016)] + still_rows(20)
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    assert cap.n == 20
    assert cap.t_ms[0] == 0
    assert any("stale leading sample" in r for r in cap.repairs)


def test_all_zero_leading_sample_is_dropped(tmp_path):
    rows = ["0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000"]
    rows += still_rows(20, start_ms=10)
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    assert cap.n == 20
    assert any("all-zero" in r for r in cap.repairs)


def test_timestamps_end_strictly_increasing(tmp_path):
    rows = still_rows(10)
    rows.insert(5, sample_row(20))          # a repeat of an earlier timestamp
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    import numpy as np
    assert np.all(np.diff(cap.t_ms) > 0)


def test_bom_is_tolerated(tmp_path):
    path = write_capture(tmp_path / "run.csv", still_rows(5), bom=True)
    assert cap_mod.load(path).n == 5


def test_unparsable_rows_are_counted_not_fatal(tmp_path):
    rows = still_rows(5) + ["nonsense,rows,that,are,not,numbers,here"]
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    assert cap.n == 5
    assert cap.skipped_rows == 1


def test_missing_columns_raise(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("t_ms,ax_g,ay_g\n0,1,2\n", encoding="utf-8")

    with pytest.raises(cap_mod.CaptureError, match="missing columns"):
        cap_mod.load(path)


def test_too_few_samples_raise(tmp_path):
    path = write_capture(tmp_path / "run.csv", still_rows(1))
    with pytest.raises(cap_mod.CaptureError):
        cap_mod.load(path)


def test_units_and_derived_quantities(tmp_path):
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", still_rows(101)))

    assert cap.duration_s == pytest.approx(1.0)
    assert cap.measured_rate_hz == pytest.approx(100.0)
    # The CSV is in g and dps; the estimator wants SI.
    assert cap.accel_ms2[0][2] == pytest.approx(cap.accel_g[0][2] * 9.80665)
    assert cap.gyro_rad_s[0][0] == pytest.approx(
        cap.gyro_dps[0][0] * 3.141592653589793 / 180.0)


def test_metadata_is_read_from_the_sidecar(tmp_path, meta_file):
    path = write_capture(tmp_path / "run_0063.csv", still_rows(5))
    meta_file(path)

    cap = cap_mod.load(path)

    assert cap.device_id == "DEV-07"
    assert cap.start_unix_ms == 1788877265000
    assert cap.has_clock
    assert "startUnixMs" not in cap.meta["assumed"]


def test_missing_sidecar_falls_back_and_says_so(tmp_path):
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", still_rows(5)))

    assert not cap.has_clock
    assert cap.meta["accelFsG"] == cap_mod.DEFAULT_ACCEL_FS_G
    assert "sidecar missing" in cap.meta["assumed"]
    assert "startUnixMs" in cap.meta["assumed"]


def test_marks_are_relative_to_the_first_kept_sample(tmp_path):
    rows = [sample_row(6016)] + still_rows(20, start_ms=0)
    rows.append(mark_row(100))
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    # t_s is zeroed on the first kept sample (t_ms == 0), so a mark at 100 ms
    # lands at 0.1 s rather than being shifted by the dropped stale row.
    assert cap.marks_s() == [pytest.approx(0.1)]


# ------------------------------- stale prefixes from the previous capture


def test_a_multi_sample_stale_prefix_keeps_the_real_capture(tmp_path):
    """run_0099: starts 951, 1282, then restarts at 10.

    The wrong reading trusts the first two rows and then drops every sample
    below 1282 - 128 of them, the first 1.3 seconds of real motion.
    """
    rows = [sample_row(951), sample_row(1282)] + still_rows(200, start_ms=10)
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    assert cap.n == 200, "the real samples must survive, not the stale ones"
    assert cap.t_ms[0] == 10
    assert cap.t_ms[-1] == 10 + 199 * 10
    assert any("stale leading" in r for r in cap.repairs)
    assert not any("out-of-order" in r for r in cap.repairs)


def test_a_single_stale_sample_still_works(tmp_path):
    rows = [sample_row(6016)] + still_rows(50, start_ms=10)
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    assert cap.n == 50
    assert cap.t_ms[0] == 10


def test_a_clean_capture_is_left_alone(tmp_path):
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", still_rows(100)))

    assert cap.n == 100
    assert cap.repairs == []


def test_a_late_backward_step_drops_only_that_sample(tmp_path):
    """Far from the start, a backward step is one bad record, not a restart."""
    rows = still_rows(300)
    rows.insert(250, sample_row(50))       # a single stray, well past the prefix
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    assert cap.n == 300, "one stray dropped, the other 300 kept"
    assert any("out-of-order" in r for r in cap.repairs)
    assert not any("stale leading" in r for r in cap.repairs)


def test_the_stale_prefix_search_is_bounded(tmp_path):
    """A restart beyond the window is not a prefix - it is corruption."""
    rows = still_rows(cap_mod.STALE_PREFIX_MAX + 40, start_ms=100_000)
    rows += still_rows(100, start_ms=10)
    cap = cap_mod.load(write_capture(tmp_path / "run.csv", rows))

    # The long high-timestamp run wins; the later low block is out-of-order.
    assert cap.t_ms[0] == 100_000
    assert any("out-of-order" in r for r in cap.repairs)
