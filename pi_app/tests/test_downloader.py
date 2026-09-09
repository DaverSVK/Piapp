"""Reassembling a capture from the notification stream.

A dropped packet must never produce a short-but-plausible capture: that is a
file which looks perfectly well-formed and holds the wrong samples. These tests
exist to keep that failure loud.
"""

from __future__ import annotations

import struct
import zlib

import pytest

from pi_app.ble import protocol as proto
from pi_app.ble.downloader import Reassembler, TransferError, merge_resumed
from pi_app.tests.test_protocol import make_slot

CHUNK = 240          # whole records, as the firmware sends at a 247-byte MTU


def payload(n: int, fill: int = 0xAA) -> bytes:
    return bytes([fill]) * n


def pkt(type_, slot=3, seq=0, offset=0, data=b"") -> proto.Packet:
    return proto.Packet.unpack(
        struct.pack(proto.DL_PKT_FMT, type_, slot, seq, offset) + data)


def slot_info_packet(record_count: int, data: bytes, **kw) -> proto.Packet:
    info = make_slot(record_count=record_count,
                     data_crc=zlib.crc32(data) & 0xFFFFFFFF, **kw)
    raw = struct.pack(proto.SLOT_FMT, info.slot, info.state, info.flags,
                      info.run_id, info.hw_id, info.boot_id,
                      info.start_uptime_ms, info.duration_ms,
                      info.record_count, info.period_ms, info.dropped_samples,
                      info.sensor_errors, info.data_crc, info.start_unix_ms,
                      info.clock_sync_age_ms, info.reserved,
                      info.name.encode())
    return pkt(proto.DL_T_SLOT_INFO, slot=info.slot, data=raw)


def make_transfer(records: int = 60):
    """A capture and the DATA packets that would carry it."""
    data = b"".join(
        struct.pack(proto.RECORD_FMT, i * 10, i, -i, 4096, 1, 2, 3)
        for i in range(records))
    packets = []
    for offset in range(0, len(data), CHUNK):
        packets.append(pkt(proto.DL_T_DATA, seq=offset // CHUNK, offset=offset,
                           data=data[offset:offset + CHUNK]))
    return data, packets


def feed_all(asm: Reassembler, data: bytes, packets, skip=()) -> None:
    asm.feed(slot_info_packet(len(data) // proto.RECORD_SIZE, data))
    for i, packet in enumerate(packets):
        if i in skip:
            continue
        asm.feed(packet)


# ------------------------------------------------------------------ happy path


def test_a_clean_transfer_reassembles_exactly():
    data, packets = make_transfer()
    asm = Reassembler(slot=3)

    feed_all(asm, data, packets)
    asm.feed(pkt(proto.DL_T_TRAILER, offset=len(data),
                 data=struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF)))

    assert asm.complete
    assert asm.assemble() == data
    asm.verify_trailer(data)


def test_progress_tracks_the_expected_total():
    data, packets = make_transfer()
    asm = Reassembler(slot=3)
    asm.feed(slot_info_packet(len(data) // proto.RECORD_SIZE, data))

    assert asm.total == len(data)
    assert asm.progress == 0.0

    asm.feed(packets[0])
    assert 0.0 < asm.progress < 1.0

    for packet in packets[1:]:
        asm.feed(packet)
    assert asm.progress == pytest.approx(1.0)


def test_out_of_order_packets_are_placed_by_offset():
    """Notifications are not guaranteed to arrive in the order they were sent."""
    data, packets = make_transfer()
    asm = Reassembler(slot=3)

    asm.feed(slot_info_packet(len(data) // proto.RECORD_SIZE, data))
    for packet in reversed(packets):
        asm.feed(packet)

    assert asm.assemble() == data


def test_duplicate_packets_are_ignored_not_double_counted():
    data, packets = make_transfer()
    asm = Reassembler(slot=3)

    feed_all(asm, data, packets)
    before = asm.received
    asm.feed(packets[0])

    assert asm.received == before
    assert asm.assemble() == data


# ---------------------------------------------------------------- failures


def test_a_missing_packet_is_a_loud_gap_not_a_short_capture():
    data, packets = make_transfer()
    asm = Reassembler(slot=3)

    feed_all(asm, data, packets, skip=(2,))

    with pytest.raises(TransferError, match="missing"):
        asm.assemble()


def test_next_offset_points_at_the_first_missing_byte():
    data, packets = make_transfer()
    asm = Reassembler(slot=3)

    feed_all(asm, data, packets, skip=(2, 3))

    assert asm.next_offset() == 2 * CHUNK


def test_next_offset_is_the_end_when_nothing_is_missing():
    data, packets = make_transfer()
    asm = Reassembler(slot=3)
    feed_all(asm, data, packets)

    assert asm.next_offset() == len(data)


def test_a_corrupted_transfer_fails_the_trailer():
    data, packets = make_transfer()
    asm = Reassembler(slot=3)
    feed_all(asm, data, packets)
    asm.feed(pkt(proto.DL_T_TRAILER, data=struct.pack("<I", 0x12345678)))

    with pytest.raises(TransferError, match="CRC mismatch"):
        asm.verify_trailer(data)


def test_an_error_packet_stops_the_transfer_with_the_reason():
    asm = Reassembler(slot=3)

    with pytest.raises(TransferError, match="busy"):
        asm.feed(pkt(proto.DL_T_ERROR, data=bytes([0x01])))

    assert asm.error is not None


def test_assembling_without_a_header_is_refused():
    asm = Reassembler(slot=3)
    with pytest.raises(TransferError, match="capture header"):
        asm.assemble()


def test_verifying_without_a_trailer_is_refused():
    data, packets = make_transfer()
    asm = Reassembler(slot=3)
    feed_all(asm, data, packets)

    with pytest.raises(TransferError, match="trailer"):
        asm.verify_trailer(data)


# ------------------------------------------------------------------ resuming


def test_a_resumed_transfer_only_fetches_the_tail():
    data, packets = make_transfer()
    start = 2 * CHUNK

    asm = Reassembler(slot=3, start=start)
    asm.feed(slot_info_packet(len(data) // proto.RECORD_SIZE, data))

    assert asm.total == len(data) - start

    for packet in packets[2:]:
        asm.feed(packet)
    # The trailer covers only the bytes this transfer sent, i.e. the tail.
    asm.feed(pkt(proto.DL_T_TRAILER, offset=len(data),
                 data=struct.pack("<I", zlib.crc32(data[start:]) & 0xFFFFFFFF)))

    tail = asm.assemble()
    assert tail == data[start:]
    asm.verify_trailer(tail)


def test_merging_a_resumed_tail_reproduces_the_capture():
    data, _ = make_transfer()
    start = 2 * CHUNK

    merged = merge_resumed(data[:start], data[start:], start)

    assert merged == data


def test_merging_discards_an_overlap_in_favour_of_the_fresh_bytes():
    """A retry may resend bytes already held; the new ones win."""
    data, _ = make_transfer()
    start = 2 * CHUNK
    stale = data[:start] + b"\xff" * 32

    assert merge_resumed(stale, data[start:], start) == data


def test_resuming_past_what_is_held_is_refused():
    with pytest.raises(TransferError, match="only 10 bytes"):
        merge_resumed(b"\x00" * 10, b"tail", 20)


# ------------------------------------------------- the trailer is not the end


def test_a_trailer_with_a_hole_does_not_mean_complete():
    """The bug that cost a real capture.

    The device sends its trailer when *it* has finished sending. If a
    notification went missing on the way, treating that as completion ends the
    transfer with a hole, and the caller then has nothing to resume from.
    """
    data, packets = make_transfer()
    asm = Reassembler(slot=3)

    feed_all(asm, data, packets, skip=(2,))
    asm.feed(pkt(proto.DL_T_TRAILER, offset=len(data),
                 data=struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF)))

    assert asm.trailer_seen, "the device did finish sending"
    assert not asm.complete, "but the capture is not whole"


def test_progress_never_reads_full_while_bytes_are_missing():
    """A bar at 100% over an unusable capture is worse than no bar."""
    data, packets = make_transfer()
    asm = Reassembler(slot=3)

    feed_all(asm, data, packets, skip=(2,))

    assert asm.received == len(data) - CHUNK
    assert asm.contiguous == 2 * CHUNK      # the unbroken run stops at the gap
    assert asm.progress < 1.0


def test_a_second_round_fills_the_gap():
    """What the retry loop in connection.py does, in miniature."""
    data, packets = make_transfer()
    asm = Reassembler(slot=3)

    # Round one, as connection.py runs it: begin, GET, then packets arrive.
    assert asm.begin_round() == 0
    feed_all(asm, data, packets, skip=(2, 3))
    asm.feed(pkt(proto.DL_T_TRAILER, data=struct.pack("<I", 0)))
    assert not asm.complete

    # Round two asks from the first missing byte; the device resends from
    # there, so everything after the gap arrives again as duplicates.
    offset = asm.begin_round()
    assert offset == 2 * CHUNK
    assert not asm.trailer_seen, "a new round starts without a trailer"

    for packet in packets[2:]:
        asm.feed(packet)

    assert asm.complete
    assert asm.assemble() == data
    assert asm.rounds == 2


def test_rounds_converge_when_every_round_loses_something():
    """A lossy link still gets there, one gap at a time."""
    data, packets = make_transfer(records=200)
    asm = Reassembler(slot=3)
    asm.feed(slot_info_packet(len(data) // proto.RECORD_SIZE, data))

    drop = {2, 5, 9}
    for _ in range(4):
        if asm.complete:
            break
        offset = asm.begin_round()
        for i, packet in enumerate(packets):
            if packet.offset < offset or i in drop:
                continue
            asm.feed(packet)
        drop = {min(drop)} if len(drop) > 1 else set()

    assert asm.complete
    assert asm.assemble() == data
