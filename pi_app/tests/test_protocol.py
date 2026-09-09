"""The wire formats, and the raw -> CSV path both transports share.

These are the contract with the firmware. A mismatch here means silently
misdecoded telemetry, so the struct sizes are asserted against the numbers the
C code asserts at compile time.
"""

from __future__ import annotations

import struct
import zlib

import pytest

from pi_app.ble import protocol as proto
from pi_app.storage import captures


def make_slot(**kw) -> proto.SlotInfo:
    f = {
        "slot": 3, "state": proto.SLOT_COMPLETE, "flags": 0, "run_id": 63,
        "hw_id": b"\xd2\xf1\xa0\x3b\x5c\x6e\x78\x90", "boot_id": 7,
        "start_uptime_ms": 48210, "duration_ms": 279040, "record_count": 27904,
        "period_ms": 10, "dropped_samples": 0, "sensor_errors": 0,
        "data_crc": 0xDEADBEEF, "start_unix_ms": 1788877265000,
        "clock_sync_age_ms": 7123, "reserved": 0, "name": b"DEV-12",
    }
    f.update(kw)
    return proto.SlotInfo.unpack(struct.pack(proto.SLOT_FMT, *f.values()))


# ------------------------------------------------------------------ sizes


def test_struct_sizes_match_the_firmware_build_asserts():
    # src/app_ctrl.h, src/ble.h, src/storage.h, src/usb_dl.c, src/dl_proto.h
    assert proto.STATUS_SIZE == 20
    assert proto.LIVE_SIZE == 20
    assert proto.SLOT_SIZE == 80
    assert proto.INFO_SIZE == 56
    assert proto.RECORD_SIZE == 16
    assert proto.DL_CMD_SIZE == 8
    assert proto.DL_PKT_SIZE == 8


# ----------------------------------------------------------------- decoding


def test_slot_info_round_trip():
    s = make_slot()

    assert s.run_id == 63
    assert s.name == "DEV-12"
    assert s.state_name == "COMPLETE"
    assert s.hw_id_hex == "d2f1a03b5c6e7890"
    assert s.data_bytes == 27904 * 16
    assert s.has_clock
    assert s.sync_age_ms == 7123


def test_a_capture_without_a_clock_says_so():
    s = make_slot(start_unix_ms=0, clock_sync_age_ms=proto.CLOCK_NEVER_SET)
    assert not s.has_clock
    assert s.sync_age_ms is None


def test_empty_slot_is_not_downloadable():
    assert not make_slot(state=proto.SLOT_EMPTY).downloadable
    # An interrupted capture still holds real data worth recovering.
    assert make_slot(state=proto.SLOT_INCOMPLETE).downloadable


def test_status_flags_decode():
    raw = struct.pack(proto.STATUS_FMT, proto.STATUS_PROTO,
                      proto.ST_RECORDING, 4,
                      proto.F_IMU_OK | proto.F_STORAGE_OK | proto.F_CLOCK,
                      12345, 1234, 2, 0, 0, 9, 42)
    st = proto.Status.unpack(raw)

    assert st.recording
    assert st.clock_synced
    assert not st.transferring
    assert st.problems() == []


def test_status_reports_problems_in_plain_words():
    raw = struct.pack(proto.STATUS_FMT, proto.STATUS_PROTO, proto.ST_IDLE,
                      proto.NO_SLOT, proto.F_IMU_OK | proto.F_NOSPACE,
                      0, 0, 0, 3, 0, 0, 0)
    problems = proto.Status.unpack(raw).problems()

    assert any("flash" in p for p in problems)
    assert any("no free slot" in p for p in problems)
    assert any("dropped" in p for p in problems)


def test_wrong_length_is_rejected_rather_than_misdecoded():
    for cls, size in ((proto.Status, proto.STATUS_SIZE),
                      (proto.Live, proto.LIVE_SIZE),
                      (proto.SlotInfo, proto.SLOT_SIZE),
                      (proto.DeviceInfo, proto.INFO_SIZE)):
        with pytest.raises(ValueError):
            cls.unpack(b"\x00" * (size - 1))


def test_live_sample_is_scaled_to_physical_units():
    raw = struct.pack(proto.LIVE_FMT, 1000, 0, 0, 4096, 0, 0, 0, 4096, 0)
    live = proto.Live.unpack(raw)

    # 4096 counts of a +-8 g full scale is 1 g.
    assert live.accel_g[2] == pytest.approx(1.0)
    assert live.peak_a_g == pytest.approx(1.0)


def test_device_info_reports_clock_state():
    def info(age, unix_ms):
        return proto.DeviceInfo.unpack(struct.pack(
            proto.INFO_FMT, 3, 16, 0, 27904, b"12345678", 7, 10, 5000,
            age, unix_ms, b"DEV-12"))

    assert info(7123, 1788877265000).clock_synced
    assert not info(proto.CLOCK_NEVER_SET, 0).clock_synced


# ------------------------------------------------------------------ commands


def test_bulk_command_packs_slot_and_offset():
    raw = proto.pack_cmd(proto.DL_GET, slot=3, offset=2818048)
    op, slot, _rsv, offset = struct.unpack(proto.DL_CMD_FMT, raw)

    assert (op, slot, offset) == (proto.DL_GET, 3, 2818048)


def test_packet_splits_header_from_payload():
    payload = b"\xaa" * 240
    raw = struct.pack(proto.DL_PKT_FMT, proto.DL_T_DATA, 3, 5, 1024) + payload
    pkt = proto.Packet.unpack(raw)

    assert (pkt.type, pkt.slot, pkt.seq, pkt.offset) == (proto.DL_T_DATA, 3, 5, 1024)
    assert pkt.payload == payload


def test_error_packets_carry_a_readable_reason():
    raw = struct.pack(proto.DL_PKT_FMT, proto.DL_T_ERROR, 3, 0, 0) + b"\x01"
    assert "busy" in proto.Packet.unpack(raw).error_text


def test_device_name_validation_mirrors_the_firmware():
    assert proto.valid_device_name("DEV-12")
    assert proto.valid_device_name("abc")
    assert not proto.valid_device_name("ab")            # too short
    assert not proto.valid_device_name("D" * 16)        # too long for the field
    assert not proto.valid_device_name("has space")
    assert not proto.valid_device_name("tab\there")


def test_set_time_command_is_ascii_and_fits_the_default_mtu():
    cmd = proto.cmd_set_time(1788877145123)

    assert cmd == "TIME 1788877145123"
    # 20 bytes is the ATT payload of the default 23-byte MTU; the clock has to
    # be settable before any MTU negotiation.
    assert len(cmd.encode("ascii")) <= 20


# ------------------------------------------------------------------ raw -> CSV


def _records():
    out = b""
    out += struct.pack(proto.RECORD_FMT, 0, 4096, 0, 0, 0, 0, 0)
    out += struct.pack(proto.RECORD_FMT, 10 | proto.MARKER_FLAG, 0, 0, 0, 0, 0, 0)
    out += struct.pack(proto.RECORD_FMT, 20, 0, 0, 0, 3277, 0, 0)
    return out


def test_csv_matches_the_format_every_analysis_tool_reads(tmp_path):
    path = captures.write_csv(tmp_path / "run.csv", _records())
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"
    assert lines[1].startswith("0,1.000000")            # 4096 counts = 1 g
    assert lines[2] == "MARK,10,,,,,"                   # the marker row
    assert lines[3].split(",")[4].startswith("50.0")    # 3277 counts = 50 dps


def test_markers_are_found_without_decoding_everything():
    assert captures.marks_in(_records()) == [10]


def test_verify_accepts_a_good_capture():
    data = _records()
    info = make_slot(record_count=3, data_crc=zlib.crc32(data) & 0xFFFFFFFF)
    captures.verify(data, info, zlib.crc32(data) & 0xFFFFFFFF)


def test_verify_catches_a_mangled_transfer():
    data = _records()
    info = make_slot(record_count=3, data_crc=zlib.crc32(data) & 0xFFFFFFFF)

    with pytest.raises(ValueError, match="transfer CRC"):
        captures.verify(data, info, 0x12345678)


def test_verify_catches_corrupt_stored_data():
    data = _records()
    info = make_slot(record_count=3, data_crc=0x12345678)

    with pytest.raises(ValueError, match="stored CRC"):
        captures.verify(data, info, zlib.crc32(data) & 0xFFFFFFFF)


def test_verify_skips_the_stored_crc_for_a_recovered_capture():
    """An INCOMPLETE slot never got a close header, so its CRC is unknown."""
    data = _records()
    info = make_slot(state=proto.SLOT_INCOMPLETE, record_count=3, data_crc=0)

    captures.verify(data, info, zlib.crc32(data) & 0xFFFFFFFF)


def test_verify_catches_a_short_transfer():
    info = make_slot(record_count=3)
    with pytest.raises(ValueError, match="expected"):
        captures.verify(_records()[:-16], info)


def test_save_capture_writes_raw_meta_and_csv(tmp_path):
    data = _records()
    info = make_slot(record_count=3, data_crc=zlib.crc32(data) & 0xFFFFFFFF)

    out = captures.save_capture(info, data, source="ble", root=tmp_path)

    assert out["raw"].read_bytes() == data
    meta = captures.read_meta(out["meta_path"])
    assert meta["deviceId"] == "DEV-12"
    assert meta["runId"] == 63
    assert meta["marksMs"] == [10]
    assert meta["accelFsG"] == 8.0
    assert meta["startIso"].endswith("Z")
    # data/<DEV-XX>/<date>/
    assert out["dir"].parent.name == "DEV-12"


def test_discover_lists_saved_captures(tmp_path):
    data = _records()
    info = make_slot(record_count=3, data_crc=zlib.crc32(data) & 0xFFFFFFFF)
    captures.save_capture(info, data, root=tmp_path)

    found = captures.discover(tmp_path)

    assert len(found) == 1
    assert found[0]["meta"]["runId"] == 63
    assert found[0]["state"] == captures.DOWNLOADED


def test_run_state_round_trips(tmp_path):
    state = captures.RunState(run_id=63, device_id="DEV-12", slot=3,
                              state=captures.DOWNLOADING,
                              received_bytes=2818048, next_offset=2818048,
                              total_bytes=446464)
    path = captures.save_state(tmp_path / "s.json", state)
    back = captures.load_state(path)

    assert back.next_offset == 2818048
    assert back.state == captures.DOWNLOADING
    assert back.updated_at


def test_unreadable_run_state_does_not_hide_the_capture(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json", encoding="utf-8")
    assert captures.load_state(path) is None


def test_device_names_are_made_safe_for_a_path():
    assert captures._safe_name("DEV-12") == "DEV-12"
    assert "/" not in captures._safe_name("a/b")
    assert captures._safe_name("") == "unknown"
