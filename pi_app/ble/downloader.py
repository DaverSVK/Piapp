"""Reassembling a capture from the bulk notification stream.

Deliberately free of any bleak dependency: it is fed decoded packets and
produces bytes, so the awkward part of a download - what happens when packets
arrive out of order, or the link drops halfway - is testable without a device.

The transfer is:

    SLOT_INFO  ->  DATA ... DATA  ->  TRAILER

Data is placed by the offset each packet carries, never by arrival order, and
the transfer is only accepted once every byte between 0 and the expected length
has actually arrived. A dropped notification therefore surfaces as a named gap
rather than as a silently short capture - which, in a file that looks perfectly
well-formed, is the failure mode worth engineering against.

**The trailer is not a completion signal.** It says the device finished sending,
not that the client finished receiving, and notifications do go missing on a
busy host. Completion means "contiguous to the end", so a transfer whose trailer
arrived with a hole in the middle stays incomplete and the caller asks again
from ``next_offset()``. One Reassembler therefore spans however many ranged GETs
it takes: offsets are absolute, so a later range simply fills in.

Resuming across app restarts works the same way, from the bytes on disk.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

from pi_app.ble import protocol as proto


class TransferError(Exception):
    """The device refused the transfer, or it cannot be completed."""


@dataclass
class Reassembler:
    """Collects one capture's bytes from DATA packets.

    ``start`` is the offset the GET asked for, so a resumed transfer only
    describes the tail it is actually fetching.
    """

    slot: int
    start: int = 0
    info: proto.SlotInfo | None = None
    chunks: dict[int, bytes] = field(default_factory=dict)
    trailer_crc: int | None = None
    error: str | None = None
    rounds: int = 0
    _received: int = 0

    @property
    def total(self) -> int:
        """Bytes this transfer expects to receive (0 until SLOT_INFO lands)."""
        if self.info is None:
            return 0
        return max(0, self.info.data_bytes - self.start)

    @property
    def received(self) -> int:
        """Payload bytes held, gaps included. See `contiguous` for progress."""
        return self._received

    @property
    def contiguous(self) -> int:
        """Bytes usable so far: the unbroken run from `start`.

        This, not `received`, is what a progress bar should show. Counting every
        packet that arrived lets the bar reach 100% while a hole in the middle
        makes the capture unusable, which is worse than useless - it says the
        job is done when it is not.
        """
        return self.next_offset() - self.start

    @property
    def progress(self) -> float:
        return min(1.0, self.contiguous / self.total) if self.total else 0.0

    @property
    def trailer_seen(self) -> bool:
        """The device says it finished sending this range."""
        return self.trailer_crc is not None

    @property
    def complete(self) -> bool:
        """Every byte through to the end of the capture is held."""
        return self.info is not None and self.next_offset() >= self.info.data_bytes

    def feed(self, packet: proto.Packet) -> None:
        """Accept one decoded notification."""
        if packet.type == proto.DL_T_ERROR:
            self.error = packet.error_text
            raise TransferError(f"slot {packet.slot}: {self.error}")

        if packet.type == proto.DL_T_SLOT_INFO:
            self.info = proto.SlotInfo.unpack(packet.payload)
            return

        if packet.type == proto.DL_T_DATA:
            # Ignore a duplicate rather than double-counting it: a retransmit
            # after a stall is a normal event, not an error.
            if packet.offset in self.chunks:
                return
            self.chunks[packet.offset] = packet.payload
            self._received += len(packet.payload)
            return

        if packet.type == proto.DL_T_TRAILER:
            self.trailer_crc = int.from_bytes(packet.payload[:4], "little")
            return

    def begin_round(self) -> int:
        """Start another ranged GET. Returns the offset to ask for."""
        self.rounds += 1
        self.trailer_crc = None
        return self.next_offset()

    def next_offset(self) -> int:
        """Where a resume should continue from: the first byte still missing."""
        offset = self.start
        while offset in self.chunks:
            offset += len(self.chunks[offset])
        return offset

    def assemble(self) -> bytes:
        """The contiguous bytes received, raising on the first gap."""
        if self.info is None:
            raise TransferError("transfer ended before the capture header arrived")

        out = bytearray()
        offset = self.start
        while offset < self.info.data_bytes:
            chunk = self.chunks.get(offset)
            if chunk is None:
                raise TransferError(
                    f"missing {self.info.data_bytes - offset} bytes from offset "
                    f"{offset}; resume with GET slot {self.slot} {offset}")
            out += chunk
            offset += len(chunk)
        return bytes(out)

    def verify_trailer(self, data: bytes) -> None:
        """Check the CRC the device computed over what it actually sent."""
        if self.trailer_crc is None:
            raise TransferError("transfer ended before the trailer arrived")
        calc = zlib.crc32(data) & 0xFFFFFFFF
        if calc != self.trailer_crc:
            raise TransferError(
                f"transfer CRC mismatch (device sent 0x{self.trailer_crc:08x}, "
                f"computed 0x{calc:08x}) - retry the download")


def merge_resumed(previous: bytes, tail: bytes, start: int) -> bytes:
    """Join a resumed tail onto the bytes already held.

    ``start`` is where the tail begins, so previously received bytes past it
    (an overlap after a retry) are discarded in favour of the fresh ones.
    """
    if start > len(previous):
        raise TransferError(
            f"cannot resume at {start}: only {len(previous)} bytes are held")
    return previous[:start] + tail
