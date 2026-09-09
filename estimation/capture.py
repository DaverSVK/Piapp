"""Load a downloaded capture: the seven-column CSV plus its metadata sidecar.

The CSV is the format tools/download_imu.py and pi_app write, and the only
format the estimator reads - so a capture that arrived over Bluetooth and one
pulled over USB go down exactly the same path.

Two details matter and are handled here rather than left to callers:

* **Event markers.** They ride in the record stream as ``MARK,<t_ms>,,,,,``
  rows with the marker time in the second column. They are kept (in
  ``marks_ms``) and excluded from the sample arrays. Older analysis scripts in
  this repository dropped them silently by failing to parse the first column;
  losing them here would mean losing the only in-capture time reference the
  skater controls.

* **Timestamps that are not strictly increasing.** Captures routinely begin
  with one or more *stale* records left over from the previous capture, whose
  timestamps are ahead of everything that follows: run_0058 starts at 732 ms,
  run_0065 at 6016 ms, run_0099 at 951 then 1282 before restarting at 10.

  The fix has to drop the stale prefix, not the good samples after it. Trusting
  the first row and then discarding everything below it is the tempting reading
  and the wrong one - on run_0099 it throws away the first 1.3 seconds (128
  samples) to keep two records that belong to a different capture. So a
  backward step near the very start is read as "the real capture begins here".
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

G0 = 9.80665

COLUMNS = ("t_ms", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")
ACCEL_COLUMNS = ("ax_g", "ay_g", "az_g")
GYRO_COLUMNS = ("gx_dps", "gy_dps", "gz_dps")
MARKER_TOKEN = "MARK"

# Defaults for a capture whose sidecar is missing (anything downloaded before
# the sidecar existed). They match src/storage.h; a capture recorded with other
# full-scale ranges and no sidecar cannot be detected and will be scaled wrong,
# which is exactly why the sidecar records them.
DEFAULT_ACCEL_FS_G = 8.0
DEFAULT_GYRO_FS_DPS = 500.0
DEFAULT_PERIOD_MS = 10


class CaptureError(Exception):
    """The file is not a capture this pipeline can read."""


@dataclass
class Capture:
    """One capture, in SI units, ready for the estimator."""

    path: Path
    t_s: np.ndarray            # (N,) seconds since the first kept sample
    t_ms: np.ndarray           # (N,) device milliseconds, as recorded
    accel_g: np.ndarray        # (N, 3)
    gyro_dps: np.ndarray       # (N, 3)
    marks_ms: list[float] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    skipped_rows: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t_ms)

    @property
    def accel_ms2(self) -> np.ndarray:
        return self.accel_g * G0

    @property
    def gyro_rad_s(self) -> np.ndarray:
        return np.radians(self.gyro_dps)

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1] - self.t_s[0]) if self.n > 1 else 0.0

    @property
    def median_period_s(self) -> float:
        if self.n < 2:
            return DEFAULT_PERIOD_MS / 1000.0
        return float(np.median(np.diff(self.t_s)))

    @property
    def measured_rate_hz(self) -> float:
        p = self.median_period_s
        return 1.0 / p if p > 0 else 0.0

    @property
    def device_id(self) -> str:
        """The DEV-XX name, or "" when the capture predates the stamp."""
        return self.meta.get("deviceId") or ""

    @property
    def start_unix_ms(self) -> int:
        """UTC ms at t_ms == 0, or 0 when the clock was never set."""
        return int(self.meta.get("startUnixMs") or 0)

    def sample_unix_ms(self) -> np.ndarray:
        """Absolute time per sample. Requires a known start (see has_clock)."""
        return self.start_unix_ms + self.t_ms

    @property
    def has_clock(self) -> bool:
        return self.start_unix_ms > 0

    def marks_s(self) -> list[float]:
        """Marker times on the same zero as t_s."""
        t0 = float(self.t_ms[0]) if self.n else 0.0
        return [(m - t0) / 1000.0 for m in self.marks_ms]

    def summary(self) -> str:
        clock = ("starts " + str(self.start_unix_ms) if self.has_clock
                 else "no absolute start time")
        return (f"{self.path.name}: {self.n} samples, {self.duration_s:.1f} s "
                f"at {self.measured_rate_hz:.1f} Hz, {len(self.marks_ms)} marks, "
                f"{clock}")


def _num(text: str) -> float:
    """Parse a float, tolerating the punctuation spreadsheets introduce."""
    t = text.strip().replace("−", "-").replace("–", "-")
    t = t.replace("—", "-").replace(" ", "").replace(" ", "")
    return float(t)


def meta_path_for(csv_path: Path | str) -> Path:
    p = Path(csv_path)
    return p.with_name(p.stem + ".meta.json")


def load_meta(csv_path: Path | str, overrides: dict | None = None) -> dict:
    """Read the sidecar for a capture, filling in documented defaults.

    ``assumed`` lists every field that did not come from the device, so the
    telemetry bundle can say which of its numbers rest on a guess.
    """
    path = meta_path_for(csv_path)
    meta: dict = {}
    assumed: list[str] = []

    if path.exists():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise CaptureError(f"{path}: unreadable metadata ({exc})") from exc
    else:
        assumed.append("sidecar missing")

    for key, default in (("accelFsG", DEFAULT_ACCEL_FS_G),
                         ("gyroFsDps", DEFAULT_GYRO_FS_DPS),
                         ("periodMs", DEFAULT_PERIOD_MS)):
        if not meta.get(key):
            meta[key] = default
            assumed.append(key)

    for key, value in (overrides or {}).items():
        if value is not None:
            meta[key] = value
            assumed.append(f"{key} (given on the command line)")

    if not meta.get("startUnixMs"):
        meta["startUnixMs"] = 0
        assumed.append("startUnixMs")
    if not meta.get("deviceId"):
        assumed.append("deviceId")

    meta["assumed"] = assumed
    return meta


# How far into a capture a backward time step can still mean "the real capture
# starts here". The stale prefix is whatever was left in the firmware's record
# queue when the slot was claimed - a handful of samples, never seconds of them.
STALE_PREFIX_MAX = 64


def _stale_prefix(t_ms: np.ndarray) -> int:
    """Length of the leftover prefix from the previous capture, if any.

    A real capture's timestamps only ever increase, so a backward step is a
    restart. Near the beginning that means everything before it belongs to the
    previous capture; later it means one corrupt record, which the strictly
    increasing filter handles instead.
    """
    limit = min(len(t_ms), STALE_PREFIX_MAX + 1)
    for i in range(limit - 1, 0, -1):        # the last restart in the window
        if t_ms[i] <= t_ms[i - 1]:
            return i
    return 0


def _repair(t_ms: np.ndarray, accel: np.ndarray, gyro: np.ndarray):
    """Drop the stale prefix and any later non-monotonic rows; report both."""
    keep = np.ones(len(t_ms), dtype=bool)
    repairs: list[str] = []

    prefix = _stale_prefix(t_ms)
    if prefix:
        keep[:prefix] = False
        repairs.append(
            f"dropped {prefix} stale leading sample(s) left over from the "
            f"previous capture (t={t_ms[0]:.0f}..{t_ms[prefix - 1]:.0f} ms, "
            f"before the capture restarts at {t_ms[prefix]:.0f} ms)")

    first = prefix
    if (first < len(t_ms) and t_ms[first] == 0.0 and len(t_ms) - first > 1
            and not accel[first].any()):
        keep[first] = False
        repairs.append("dropped an all-zero leading sample")
        first += 1

    last = -math.inf
    out_of_order = 0
    for i in range(first, len(t_ms)):
        if not keep[i]:
            continue
        if t_ms[i] <= last:
            keep[i] = False
            out_of_order += 1
            continue
        last = t_ms[i]

    if out_of_order:
        repairs.append(f"dropped {out_of_order} out-of-order sample(s)")
    return t_ms[keep], accel[keep], gyro[keep], repairs


def load(path: Path | str, meta_overrides: dict | None = None) -> Capture:
    """Read one capture CSV (and its sidecar) into a Capture."""
    path = Path(path)
    times: list[float] = []
    accel: list[list[float]] = []
    gyro: list[list[float]] = []
    marks: list[float] = []
    skipped = 0

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise CaptureError(
                f"{path}: missing columns {missing}; expected {list(COLUMNS)}")

        for row in reader:
            raw_t = (row.get("t_ms") or "").strip()

            if raw_t.upper() == MARKER_TOKEN:
                # MARK,<t_ms>,,,,,  - the event time sits in the ax_g column.
                try:
                    marks.append(_num(row["ax_g"]))
                except (TypeError, ValueError, KeyError):
                    skipped += 1
                continue

            try:
                t = _num(raw_t)
                a = [_num(row[c]) for c in ACCEL_COLUMNS]
                g = [_num(row[c]) for c in GYRO_COLUMNS]
            except (TypeError, ValueError, KeyError):
                skipped += 1
                continue

            times.append(t)
            accel.append(a)
            gyro.append(g)

    if len(times) < 2:
        raise CaptureError(f"{path}: only {len(times)} usable sample(s)")

    t_ms, accel_g, gyro_dps, repairs = _repair(
        np.asarray(times, dtype=float),
        np.asarray(accel, dtype=float),
        np.asarray(gyro, dtype=float))

    if len(t_ms) < 2:
        raise CaptureError(f"{path}: no usable samples after timing repair")

    return Capture(
        path=path,
        t_s=(t_ms - t_ms[0]) / 1000.0,
        t_ms=t_ms,
        accel_g=accel_g,
        gyro_dps=gyro_dps,
        marks_ms=sorted(marks),
        repairs=repairs,
        skipped_rows=skipped,
        meta=load_meta(path, meta_overrides),
    )
