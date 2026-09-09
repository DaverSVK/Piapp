"""Wire formats spoken by the Thingy:53 firmware.

This is the single source of truth for every struct that crosses the link, in
either direction and over either transport. It mirrors, byte for byte:

    src/app_ctrl.h   struct imu_status          (STATUS_FMT)
    src/ble.h        struct imu_live            (LIVE_FMT)
    src/storage.h    struct imu_slot_info       (SLOT_FMT)
    src/usb_dl.c     the 'I' reply              (INFO_FMT)
    src/dl_proto.h   struct imu_dl_cmd/_packet  (DL_CMD_FMT, DL_PKT_FMT)

Change a struct on one side and you must change it here; the sizes are asserted
at import time so a mismatch is a loud failure at start-up rather than silently
misdecoded telemetry. tools/imu_monitor.py imports this module too, so there is
exactly one copy of these formats in the repository.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# --------------------------------------------------------------------- UUIDs

SERVICE_UUID = "8e7c0f00-1d2b-4a3c-9e5f-a1b2c3d4e5f6"
CTRL_UUID = "8e7c0f01-1d2b-4a3c-9e5f-a1b2c3d4e5f6"
MARK_UUID = "8e7c0f02-1d2b-4a3c-9e5f-a1b2c3d4e5f6"
STATUS_UUID = "8e7c0f03-1d2b-4a3c-9e5f-a1b2c3d4e5f6"
LIVE_UUID = "8e7c0f04-1d2b-4a3c-9e5f-a1b2c3d4e5f6"
DLCTRL_UUID = "8e7c0f05-1d2b-4a3c-9e5f-a1b2c3d4e5f6"
DLDATA_UUID = "8e7c0f06-1d2b-4a3c-9e5f-a1b2c3d4e5f6"

BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"  # standard BAS

# ------------------------------------------------------------------- sensors

# src/storage.h. Not carried in a capture, so both sides hold a copy; the
# download writes them into the sidecar metadata so a stored capture is
# self-describing even if these ever change.
ACCEL_FS_G = 8.0
GYRO_FS_DPS = 500.0
SAMPLE_HZ = 100
PERIOD_MS = 10

RECORD_FMT = "<Ihhhhhh"
RECORD_SIZE = struct.calcsize(RECORD_FMT)

MARKER_FLAG = 0x80000000
MARKER_T_MASK = 0x7FFFFFFF

DEVID_SIZE = 16  # IMU_DEVID_SIZE

# -------------------------------------------------------------------- status

STATUS_PROTO = 1
STATUS_FMT = "<BBBBIIHHHBB"
STATUS_SIZE = struct.calcsize(STATUS_FMT)

ST_IDLE, ST_RECORDING, ST_ERROR = 0, 1, 2
STATE_NAMES = {ST_IDLE: "IDLE", ST_RECORDING: "RECORDING", ST_ERROR: "ERROR"}

F_IMU_OK = 0x01
F_STORAGE_OK = 0x02
F_LIVE = 0x04
F_BUSY = 0x08
F_FULL = 0x10
F_NOSPACE = 0x20
F_CLOCK = 0x40
F_XFER = 0x80

NO_SLOT = 0xFF

LIVE_FMT = "<IhhhhhhHH"
LIVE_SIZE = struct.calcsize(LIVE_FMT)
LIVE_MAX_HZ = 20

# ----------------------------------------------------------------- slot info

# struct imu_slot_info, src/storage.h.
SLOT_FMT = "<BBHI8sIIIIIIIIQII16s"
SLOT_SIZE = struct.calcsize(SLOT_FMT)

SLOT_EMPTY, SLOT_INCOMPLETE, SLOT_COMPLETE, SLOT_DOWNLOADED = 0, 1, 2, 3
SLOT_STATE_NAMES = {
    SLOT_EMPTY: "EMPTY",
    SLOT_INCOMPLETE: "INCOMPLETE",
    SLOT_COMPLETE: "COMPLETE",
    SLOT_DOWNLOADED: "DOWNLOADED",
}

SLOT_FLAG_FULL = 0x1

# --------------------------------------------------------------- device info

PROTO_VERSION = 3  # src/usb_dl.c PROTO_VERSION; also the BLE feature level
PING_REPLY = b"IMU3"

INFO_FMT = "<BBHI8sIIIIQ16s"
INFO_SIZE = struct.calcsize(INFO_FMT)

CLOCK_NEVER_SET = 0xFFFFFFFF  # device_info.clock_sync_age_ms

# --------------------------------------------------------- bulk transfer

DL_CMD_FMT = "<BBHI"
DL_CMD_SIZE = struct.calcsize(DL_CMD_FMT)

DL_LIST, DL_GET, DL_ABORT, DL_ACK, DL_ERASE = 0x01, 0x02, 0x03, 0x04, 0x05

DL_PKT_FMT = "<BBHI"
DL_PKT_SIZE = struct.calcsize(DL_PKT_FMT)

DL_T_SLOT_INFO, DL_T_DATA, DL_T_TRAILER, DL_T_ERROR = 0x01, 0x02, 0x03, 0x04

DL_ERRORS = {
    0x01: "device is busy (recording, or another transfer is running)",
    0x02: "no such slot",
    0x03: "slot holds nothing downloadable",
    0x04: "flash read failed",
    0x05: "malformed command",
}

# The firmware asserts these sizes at compile time; assert them here too, so a
# firmware/tool mismatch fails at import rather than halfway through a download.
assert STATUS_SIZE == 20, STATUS_SIZE
assert LIVE_SIZE == 20, LIVE_SIZE
assert SLOT_SIZE == 80, SLOT_SIZE
assert INFO_SIZE == 56, INFO_SIZE
assert RECORD_SIZE == 16, RECORD_SIZE
assert DL_CMD_SIZE == 8, DL_CMD_SIZE
assert DL_PKT_SIZE == 8, DL_PKT_SIZE


def _cstr(raw: bytes) -> str:
    """Decode a fixed-width NUL-padded field."""
    return raw.split(b"\x00", 1)[0].decode("ascii", "replace")


# ------------------------------------------------------------------ decoders


@dataclass
class Status:
    """One 20-byte status snapshot (struct imu_status)."""

    proto: int
    state: int
    slot: int
    flags: int
    elapsed_ms: int
    records: int
    markers: int
    dropped: int
    sensor_errors: int
    free_slots: int
    fill_pct: int

    @classmethod
    def unpack(cls, raw: bytes) -> "Status":
        if len(raw) != STATUS_SIZE:
            raise ValueError(f"status is {len(raw)} bytes, expected {STATUS_SIZE}")
        return cls(*struct.unpack(STATUS_FMT, raw))

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, f"?{self.state}")

    @property
    def recording(self) -> bool:
        return self.state == ST_RECORDING

    @property
    def clock_synced(self) -> bool:
        return bool(self.flags & F_CLOCK)

    @property
    def transferring(self) -> bool:
        return bool(self.flags & F_XFER)

    def problems(self) -> list[str]:
        """Everything the user should be told about, in plain words."""
        out = []
        if not self.flags & F_IMU_OK:
            out.append("IMU did not start")
        if not self.flags & F_STORAGE_OK:
            out.append("flash did not start")
        if self.flags & F_NOSPACE:
            out.append("no free slot: download or erase first")
        if self.flags & F_FULL:
            out.append("last capture filled its slot")
        if self.dropped:
            out.append(f"{self.dropped} dropped samples")
        if self.sensor_errors:
            out.append(f"{self.sensor_errors} sensor errors")
        return out


@dataclass
class Live:
    """One 20-byte decimated live sample (struct imu_live), scaled to SI-ish."""

    t_ms: int
    accel_g: tuple[float, float, float]
    gyro_dps: tuple[float, float, float]
    peak_a_g: float
    peak_g_dps: float

    @classmethod
    def unpack(cls, raw: bytes) -> "Live":
        if len(raw) != LIVE_SIZE:
            raise ValueError(f"live is {len(raw)} bytes, expected {LIVE_SIZE}")
        t, ax, ay, az, gx, gy, gz, pa, pg = struct.unpack(LIVE_FMT, raw)
        a = ACCEL_FS_G / 32768.0
        g = GYRO_FS_DPS / 32768.0
        return cls(
            t_ms=t,
            accel_g=(ax * a, ay * a, az * a),
            gyro_dps=(gx * g, gy * g, gz * g),
            peak_a_g=pa * a,
            peak_g_dps=pg * g,
        )


@dataclass
class SlotInfo:
    """One capture's metadata (struct imu_slot_info)."""

    slot: int
    state: int
    flags: int
    run_id: int
    hw_id: bytes
    boot_id: int
    start_uptime_ms: int
    duration_ms: int
    record_count: int
    period_ms: int
    dropped_samples: int
    sensor_errors: int
    data_crc: int
    start_unix_ms: int
    clock_sync_age_ms: int
    reserved: int
    name: str

    @classmethod
    def unpack(cls, raw: bytes) -> "SlotInfo":
        if len(raw) != SLOT_SIZE:
            raise ValueError(f"slot info is {len(raw)} bytes, expected {SLOT_SIZE}")
        f = struct.unpack(SLOT_FMT, raw)
        return cls(*f[:16], name=_cstr(f[16]))

    @property
    def state_name(self) -> str:
        return SLOT_STATE_NAMES.get(self.state, f"?{self.state}")

    @property
    def data_bytes(self) -> int:
        return self.record_count * RECORD_SIZE

    @property
    def hw_id_hex(self) -> str:
        return self.hw_id.hex()

    @property
    def has_clock(self) -> bool:
        """Whether this capture knows when it happened.

        False for anything recorded before the Pi ever synchronised the clock,
        and for every capture made by firmware older than protocol 3.
        """
        return self.start_unix_ms > 0

    @property
    def downloadable(self) -> bool:
        return self.state != SLOT_EMPTY

    @property
    def sync_age_ms(self) -> int | None:
        """How stale the wall clock was when this capture started.

        None when unknown. The pairing behind the clock rides the uptime
        crystal, so a large value means the stamped start time has had time to
        drift away from real UTC.
        """
        if self.clock_sync_age_ms == CLOCK_NEVER_SET:
            return None
        return self.clock_sync_age_ms


@dataclass
class DeviceInfo:
    """The 'I' reply / the same fields read over Bluetooth."""

    proto_ver: int
    slot_count: int
    reserved: int
    slot_max_records: int
    hw_id: bytes
    boot_id: int
    period_ms: int
    uptime_ms: int
    clock_sync_age_ms: int
    unix_ms: int
    name: str

    @classmethod
    def unpack(cls, raw: bytes) -> "DeviceInfo":
        if len(raw) != INFO_SIZE:
            raise ValueError(f"device info is {len(raw)} bytes, expected {INFO_SIZE}")
        f = struct.unpack(INFO_FMT, raw)
        return cls(*f[:10], name=_cstr(f[10]))

    @property
    def clock_synced(self) -> bool:
        return self.clock_sync_age_ms != CLOCK_NEVER_SET and self.unix_ms > 0

    @property
    def hw_id_hex(self) -> str:
        return self.hw_id.hex()


@dataclass
class Packet:
    """One bulk-transfer notification (struct imu_dl_packet plus payload)."""

    type: int
    slot: int
    seq: int
    offset: int
    payload: bytes = b""

    @classmethod
    def unpack(cls, raw: bytes) -> "Packet":
        if len(raw) < DL_PKT_SIZE:
            raise ValueError(f"packet is {len(raw)} bytes, expected at least {DL_PKT_SIZE}")
        t, slot, seq, off = struct.unpack(DL_PKT_FMT, raw[:DL_PKT_SIZE])
        return cls(t, slot, seq, off, bytes(raw[DL_PKT_SIZE:]))

    @property
    def error_text(self) -> str:
        code = self.payload[0] if self.payload else 0
        return DL_ERRORS.get(code, f"unknown error 0x{code:02x}")


# ------------------------------------------------------------------ encoders


def pack_cmd(op: int, slot: int = 0, offset: int = 0) -> bytes:
    """Build one bulk control command."""
    return struct.pack(DL_CMD_FMT, op, slot, 0, offset)


def cmd_set_time(unix_ms: int) -> str:
    """The ASCII control-characteristic command that sets the wall clock."""
    return f"TIME {int(unix_ms)}"


def cmd_set_id(name: str) -> str:
    return f"SET_ID {name}"


def valid_device_name(name: str) -> bool:
    """Mirror of name_ok() in src/devid.c, so the UI can refuse early."""
    if not 3 <= len(name) < DEVID_SIZE:
        return False
    return all(" " < c <= "~" for c in name)


def decode_records(data: bytes, offset: int = 0, count: int | None = None):
    """Yield (t_ms, accel_g, gyro_dps, is_marker) for packed capture records.

    Marker records carry MARKER_FLAG in t_ms and have zeroed sensor fields; the
    flag is stripped here so callers see the marker's capture-relative time.
    """
    if count is None:
        count = (len(data) - offset) // RECORD_SIZE
    a = ACCEL_FS_G / 32768.0
    g = GYRO_FS_DPS / 32768.0
    for i in range(count):
        t, ax, ay, az, gx, gy, gz = struct.unpack_from(RECORD_FMT, data, offset + i * RECORD_SIZE)
        if t & MARKER_FLAG:
            yield t & MARKER_T_MASK, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), True
        else:
            yield t, (ax * a, ay * a, az * a), (gx * g, gy * g, gz * g), False
