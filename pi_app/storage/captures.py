"""On-disk layout for downloaded captures, and the raw -> CSV decoder.

One capture lives as up to four files in a per-device, per-day directory:

    data/DEV-12/2026-09-08/
        run_0063.raw            packed struct imu_record bytes, exactly as
                                stored on the device
        run_0063.meta.json      everything the device knew about the capture
        run_0063.csv            the seven-column format every analysis tool
                                in this repo already reads
        run_0063.bundle.json    the telemetry the backend receives

The raw file is written first and kept forever. Everything downstream - CSV,
estimation, publishing - is reproducible from it, so a failed processing run or
a failed upload never costs a capture, and re-processing never needs the device
again.

Both transports share this module, so there is exactly one CSV writer: the USB
tool in tools/download_imu.py and the Bluetooth downloader in
pi_app/ble/downloader.py produce byte-identical output from the same bytes.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from pi_app.ble import protocol as proto

CSV_HEADER = "t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"

META_SCHEMA = "imu_capture_meta"
META_VERSION = 1

# Per-run lifecycle (thingy53_pi_app_flow.md section 30). Persisted alongside
# the capture so the app can show where every run got to after a restart.
ON_DEVICE = "ON_DEVICE"
DOWNLOADING = "DOWNLOADING"
DOWNLOADED = "DOWNLOADED"
PROCESSING = "PROCESSING"
PROCESSED = "PROCESSED"
PUBLISHING = "PUBLISHING"
PUBLISHED = "PUBLISHED"
ERROR = "ERROR"

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data"


def _safe_name(name: str) -> str:
    """A device name that is safe as a directory component.

    devid.c already refuses spaces and non-printables, but the name arrives
    from a device and lands in a path, so it is filtered here as well rather
    than trusted.
    """
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in name]
    out = "".join(keep).strip("._") or "unknown"
    return out[: proto.DEVID_SIZE]


def capture_dir(device_name: str, when: datetime | None = None,
                root: Path | str | None = None) -> Path:
    """data/<DEV-XX>/<YYYY-MM-DD>/ for a capture, created if needed."""
    base = Path(root) if root is not None else DEFAULT_ROOT
    day = (when or datetime.now(timezone.utc)).astimezone().strftime("%Y-%m-%d")
    path = base / _safe_name(device_name) / day
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_stem(run_id: int, partial: bool = False) -> str:
    return f"run_{run_id:04d}{'_partial' if partial else ''}"


def start_datetime(start_unix_ms: int) -> datetime | None:
    """UTC datetime for a capture start, or None when the clock was never set."""
    if not start_unix_ms:
        return None
    return datetime.fromtimestamp(start_unix_ms / 1000.0, tz=timezone.utc)


# ------------------------------------------------------------------- metadata


def meta_from_slot(info: proto.SlotInfo, source: str = "ble",
                   marks_ms: list[int] | None = None,
                   device_name: str | None = None) -> dict:
    """Build the sidecar metadata for one downloaded capture.

    device_name overrides the name stamped into the capture, which is only
    needed for captures recorded by firmware that predates the stamp.
    """
    name = info.name or device_name or ""
    return {
        "schema": META_SCHEMA,
        "version": META_VERSION,
        "deviceId": name,
        "hardwareId": info.hw_id_hex,
        "runId": info.run_id,
        "slot": info.slot,
        "state": info.state_name,
        "bootId": info.boot_id,
        "startUnixMs": info.start_unix_ms,
        "startUptimeMs": info.start_uptime_ms,
        "clockSyncAgeMs": info.sync_age_ms,
        "startIso": (dt.isoformat().replace("+00:00", "Z")
                     if (dt := start_datetime(info.start_unix_ms)) else None),
        "periodMs": info.period_ms or proto.PERIOD_MS,
        "durationMs": info.duration_ms,
        "recordCount": info.record_count,
        "droppedSamples": info.dropped_samples,
        "sensorErrors": info.sensor_errors,
        "dataCrc": info.data_crc,
        "slotFull": bool(info.flags & proto.SLOT_FLAG_FULL),
        # Not stored on the device (see src/storage.h): written here so a
        # capture on disk is self-describing even if the firmware's ranges
        # ever change.
        "accelFsG": proto.ACCEL_FS_G,
        "gyroFsDps": proto.GYRO_FS_DPS,
        "marksMs": sorted(marks_ms or []),
        "downloadedVia": source,
        "downloadedAt": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }


def write_meta(path: Path | str, meta: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def read_meta(path: Path | str) -> dict:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if meta.get("schema") != META_SCHEMA:
        raise ValueError(f"{path}: not an {META_SCHEMA} file")
    return meta


# ------------------------------------------------------------- raw -> CSV


def marks_in(data: bytes, count: int | None = None) -> list[int]:
    """Marker timestamps in a raw capture, in the order they were recorded."""
    if count is None:
        count = len(data) // proto.RECORD_SIZE
    out = []
    for i in range(count):
        (t,) = struct.unpack_from("<I", data, i * proto.RECORD_SIZE)
        if t & proto.MARKER_FLAG:
            out.append(t & proto.MARKER_T_MASK)
    return out


def write_csv(path: Path | str, data: bytes, count: int | None = None,
              accel_fs_g: float = proto.ACCEL_FS_G,
              gyro_fs_dps: float = proto.GYRO_FS_DPS) -> Path:
    """Decode packed records into the repository's seven-column CSV.

    Event markers travel in the record stream with a flagged timestamp; they
    are emitted as a valid seven-column row carrying the marker time in the
    second column:

        MARK,<marker_t_ms>,,,,,

    Every loader in estimation/ handles that row explicitly; the older analysis
    scripts skip it because the first column is not a number.
    """
    path = Path(path)
    if count is None:
        count = len(data) // proto.RECORD_SIZE

    a = accel_fs_g / 32768.0
    g = gyro_fs_dps / 32768.0
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(CSV_HEADER + "\n")
        for i in range(count):
            t, ax, ay, az, gx, gy, gz = struct.unpack_from(
                proto.RECORD_FMT, data, i * proto.RECORD_SIZE)
            if t & proto.MARKER_FLAG:
                f.write("MARK,%u,,,,,\n" % (t & proto.MARKER_T_MASK))
                continue
            f.write("%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
                t, ax * a, ay * a, az * a, gx * g, gy * g, gz * g))
    return path


def verify(data: bytes, info: proto.SlotInfo, trailer_crc: int | None = None) -> None:
    """Check a downloaded capture, raising ValueError with a plain reason.

    Two independent checks, and both matter: the trailer covers what the device
    actually sent (so it catches a mangled transfer even for a recovered
    INCOMPLETE slot, whose stored CRC is unknown), while info.data_crc covers
    what was stored (so it catches flash that has gone bad since).
    """
    expected = info.record_count * proto.RECORD_SIZE
    if len(data) != expected:
        raise ValueError(f"got {len(data)} bytes, expected {expected}")

    calc = zlib.crc32(data) & 0xFFFFFFFF
    if trailer_crc is not None and calc != trailer_crc:
        raise ValueError(f"transfer CRC mismatch (sent 0x{trailer_crc:08x}, "
                         f"computed 0x{calc:08x}) - retry the download")
    if info.state != proto.SLOT_INCOMPLETE and info.data_crc != calc:
        raise ValueError(f"stored CRC mismatch (header 0x{info.data_crc:08x}, "
                         f"computed 0x{calc:08x}) - the flash data is corrupt; "
                         f"erase slot {info.slot}")


def save_capture(info: proto.SlotInfo, data: bytes, source: str = "ble",
                 root: Path | str | None = None,
                 device_name: str | None = None) -> dict:
    """Write raw + meta + CSV for one verified capture.

    Returns a dict of the paths written plus the metadata, so a caller can hand
    the whole thing straight to the estimation pipeline.
    """
    name = info.name or device_name or "unknown"
    when = start_datetime(info.start_unix_ms) or datetime.now(timezone.utc)
    out = capture_dir(name, when, root)
    stem = run_stem(info.run_id, partial=info.state == proto.SLOT_INCOMPLETE)

    raw_path = out / f"{stem}.raw"
    raw_path.write_bytes(data)

    meta = meta_from_slot(info, source=source, marks_ms=marks_in(data, info.record_count),
                          device_name=device_name)
    meta_path = write_meta(out / f"{stem}.meta.json", meta)
    csv_path = write_csv(out / f"{stem}.csv", data, info.record_count,
                         meta["accelFsG"], meta["gyroFsDps"])

    return {
        "dir": out,
        "raw": raw_path,
        "meta_path": meta_path,
        "csv": csv_path,
        "bundle": out / f"{stem}.bundle.json",
        "meta": meta,
    }


# ------------------------------------------------------- interrupted downloads


def partial_dir(device_name: str, root: Path | str | None = None) -> Path:
    """Where a half-finished download lives until it is verified.

    Kept out of the dated capture directories so an unverified file can never
    be mistaken for a capture: only save_capture() creates those, and only
    after the CRC has been checked.
    """
    base = Path(root) if root is not None else DEFAULT_ROOT
    path = base / _safe_name(device_name) / ".partial"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_partial(device_name: str, slot: int, run_id: int,
                 root: Path | str | None = None) -> bytes:
    """Bytes already received for this capture, or b"" to start over.

    The run id is what makes resuming safe. A slot is a location, not a
    capture: the device recycles slots, so the same slot number can hold a
    different run tomorrow. Resuming into it by offset alone would splice two
    captures together and still pass a length check.
    """
    directory = partial_dir(device_name, root)
    state_file = directory / f"slot_{slot:02d}.json"
    data_file = directory / f"slot_{slot:02d}.raw"
    if not (state_file.exists() and data_file.exists()):
        return b""

    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except ValueError:
        return b""

    if state.get("runId") != run_id:
        return b""

    data = data_file.read_bytes()
    return data[: int(state.get("receivedBytes", len(data)))]


def save_partial(device_name: str, slot: int, run_id: int, data: bytes,
                 total_bytes: int, root: Path | str | None = None) -> Path:
    """Keep what arrived, so a dropped link costs seconds rather than a redo."""
    directory = partial_dir(device_name, root)
    (directory / f"slot_{slot:02d}.raw").write_bytes(data)
    state_file = directory / f"slot_{slot:02d}.json"
    state_file.write_text(json.dumps({
        "runId": run_id,
        "slot": slot,
        "receivedBytes": len(data),
        "nextOffset": len(data),
        "totalBytes": total_bytes,
    }, indent=2) + "\n", encoding="utf-8")
    return state_file


def clear_partial(device_name: str, slot: int,
                  root: Path | str | None = None) -> None:
    directory = partial_dir(device_name, root)
    for suffix in (".raw", ".json"):
        path = directory / f"slot_{slot:02d}{suffix}"
        if path.exists():
            path.unlink()


# --------------------------------------------------------------- run state


@dataclass
class RunState:
    """Where one run has got to, persisted next to the capture."""

    run_id: int
    device_id: str
    state: str = ON_DEVICE
    slot: int | None = None
    received_bytes: int = 0
    next_offset: int = 0
    total_bytes: int = 0
    error: str | None = None
    updated_at: str = ""

    def touch(self) -> "RunState":
        self.updated_at = (datetime.now(timezone.utc)
                           .isoformat(timespec="seconds").replace("+00:00", "Z"))
        return self

    @property
    def progress(self) -> float:
        if not self.total_bytes:
            return 0.0
        return min(1.0, self.received_bytes / self.total_bytes)


def state_path(directory: Path | str, run_id: int, partial: bool = False) -> Path:
    return Path(directory) / f"{run_stem(run_id, partial)}.state.json"


def save_state(path: Path | str, state: RunState) -> Path:
    path = Path(path)
    path.write_text(json.dumps(asdict(state.touch()), indent=2) + "\n",
                    encoding="utf-8")
    return path


def load_state(path: Path | str) -> RunState | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return RunState(**json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError) as exc:
        # A half-written state file must not make a capture unreachable; the
        # raw data is what matters and it is still there.
        print(f"{path}: ignoring unreadable run state ({exc})")
        return None


def discover(root: Path | str | None = None) -> list[dict]:
    """Every capture on disk, newest first.

    Reads only the sidecar metadata, so listing hundreds of runs stays cheap.
    """
    base = Path(root) if root is not None else DEFAULT_ROOT
    if not base.exists():
        return []

    out = []
    for meta_path in base.glob("*/*/*.meta.json"):
        try:
            meta = read_meta(meta_path)
        except (ValueError, OSError) as exc:
            print(f"{meta_path}: skipped ({exc})")
            continue
        stem = meta_path.name[: -len(".meta.json")]
        directory = meta_path.parent
        bundle = directory / f"{stem}.bundle.json"
        state = load_state(directory / f"{stem}.state.json")
        out.append({
            "meta": meta,
            "dir": directory,
            "stem": stem,
            "raw": directory / f"{stem}.raw",
            "csv": directory / f"{stem}.csv",
            "bundle": bundle,
            "state": (state.state if state else
                      (PUBLISHED if bundle.exists() else DOWNLOADED)),
        })
    out.sort(key=lambda r: (r["meta"].get("startUnixMs") or 0,
                            r["meta"].get("runId") or 0), reverse=True)
    return out
