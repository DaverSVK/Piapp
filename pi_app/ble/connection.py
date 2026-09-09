"""The Bluetooth side of the Pi app, on its own thread.

Extends the pattern tools/imu_monitor.py established: everything asyncio and
bleak happens on one background thread, the UI hands it work through
non-blocking calls, and results come back as ``(kind, payload)`` tuples on a
queue. The two never touch each other's state, which is what keeps the window
responsive while a scan, a reconnect or a 436 KiB download is in flight.

The one thing that happens automatically on connect is the clock sync
(thingy53_pi_app_flow.md section 32): the board has no RTC, so without it every
capture recorded afterwards would have no absolute start time. Nothing else is
automatic - the user chooses what to download.

Events emitted:

    ("devices", [Found, ...])        a scan finished
    ("conn", "scanning"|"connecting"|"connected"|"disconnected")
    ("info", DeviceInfo)             read once on connect
    ("clock", bool)                  whether the sync took
    ("battery", int)                 percent, when the board reports it
    ("status", Status)               1 Hz plus on every change
    ("slots", [SlotInfo, ...])       a LIST completed
    ("progress", (slot, received, total))
    ("downloaded", dict)             paths + metadata from captures.save_capture
    ("log", str)                     something worth showing the user
    ("error", str)                   something that stopped what was asked for
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from pathlib import Path

from bleak import BleakClient

from pi_app.ble import protocol as proto
from pi_app.ble import scanner
from pi_app.ble.downloader import Reassembler, TransferError, merge_resumed
from pi_app.storage import captures

RETRY_DELAY = 2.0
IDLE_POLL = 0.2

# A whole slot is 436 KiB; at a 15 ms interval with a 247-byte MTU that is
# seconds, but a link that has gone quiet should not hang the app forever.
TRANSFER_TIMEOUT = 180.0
LIST_TIMEOUT = 15.0

# Slots the firmware exposes (SLOT_COUNT in src/storage.c).
MAX_SLOTS = 16

# How many ranged GETs one download may use to fill in dropped notifications.
# Each round costs only the bytes still missing, so this converges quickly; the
# cap exists so a genuinely broken link fails with a message instead of looping.
MAX_ROUNDS = 6

# After the device's trailer, how long to wait for stragglers before deciding
# the rest was lost. Notifications already delivered are long since queued, so
# this only needs to cover the host handing them over.
QUIET_AFTER_TRAILER = 1.5


class DeviceSession:
    """Owns the bleak loop and one connection at a time."""

    def __init__(self, events: queue.Queue, data_root: Path | str | None = None):
        self.events = events
        self.data_root = data_root
        self.connected = False
        self.device_name = ""

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._work: asyncio.Queue | None = None
        self._target: str | None = None
        self._want = threading.Event()
        self._quit = threading.Event()
        self._ready = threading.Event()

        # Set by the notification handler, drained by whatever is waiting.
        self._packets: asyncio.Queue | None = None
        self._abort = threading.Event()

    # -- called from the UI thread -------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ble",
                                        daemon=True)
        self._thread.start()
        self._ready.wait(5.0)

    def shutdown(self) -> None:
        self._quit.set()
        self._want.clear()
        if self._thread:
            self._thread.join(timeout=5.0)

    def scan(self, timeout: float = scanner.SCAN_TIMEOUT) -> None:
        self._submit(("scan", timeout))

    def connect(self, address: str) -> None:
        self._target = address
        self._want.set()

    def disconnect(self) -> None:
        self._want.clear()
        self._abort.set()

    def list_runs(self) -> None:
        self._submit(("list", None))

    def download(self, slot: int, run_id: int = 0, ack: bool = True) -> None:
        """Fetch one capture. run_id (from the last listing) lets an
        interrupted download resume instead of starting again."""
        self._submit(("download", (slot, run_id, ack)))

    def erase(self, slot: int) -> None:
        self._submit(("erase", slot))

    def send(self, command: str) -> None:
        """Write one ASCII control command (START / STOP / MARK / SET_ID)."""
        self._submit(("cmd", command))

    def cancel_transfer(self) -> None:
        self._abort.set()
        self._submit(("abort", None))

    # -- plumbing ------------------------------------------------------

    def _emit(self, kind: str, payload=None) -> None:
        self.events.put((kind, payload))

    def _submit(self, item) -> None:
        if self._loop is None or self._work is None:
            self._emit("log", "not ready")
            return
        self._loop.call_soon_threadsafe(self._work.put_nowait, item)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._work = asyncio.Queue()
        self._packets = asyncio.Queue()
        self._ready.set()
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        while not self._quit.is_set():
            if not self._want.is_set():
                # Scans are the one thing that works while disconnected.
                await self._drain_offline()
                continue
            try:
                await self._session()
            except Exception as exc:                    # noqa: BLE001
                self._emit("error", f"connection failed: {exc}")
            self.connected = False
            self._emit("conn", "disconnected")
            if self._want.is_set() and not self._quit.is_set():
                await asyncio.sleep(RETRY_DELAY)

    async def _drain_offline(self) -> None:
        try:
            item = await asyncio.wait_for(self._work.get(), IDLE_POLL)
        except asyncio.TimeoutError:
            return
        kind, arg = item
        if kind == "scan":
            await self._do_scan(arg)
        elif kind != "abort":
            self._emit("log", f"{kind}: not connected")

    async def _do_scan(self, timeout: float) -> None:
        self._emit("conn", "scanning")
        try:
            self._emit("devices", await scanner.scan(timeout))
        except Exception as exc:                        # noqa: BLE001
            self._emit("error", f"scan failed: {exc}")
        finally:
            if not self.connected:
                self._emit("conn", "disconnected")

    # -- one connection -------------------------------------------------

    async def _session(self) -> None:
        self._emit("conn", "connecting")
        device = await scanner.find_by_address(self._target)
        if device is None:
            self._emit("error", f"{self._target} is not in range")
            self._want.clear()
            return

        loop = asyncio.get_running_loop()
        gone = asyncio.Event()

        def on_disconnect(_client):
            loop.call_soon_threadsafe(gone.set)

        async with BleakClient(device, disconnected_callback=on_disconnect) as client:
            self.connected = True
            self.device_name = device.name or self._target
            self._emit("conn", "connected")
            self._emit("log", f"connected to {self.device_name}")

            await self._on_connect(client)

            while (self._want.is_set() and not gone.is_set()
                   and not self._quit.is_set()):
                try:
                    item = await asyncio.wait_for(self._work.get(), 0.25)
                except asyncio.TimeoutError:
                    continue
                await self._handle(client, item, gone)

    async def _on_connect(self, client: BleakClient) -> None:
        """The automatic part of connecting: identify, sync the clock, listen."""
        await client.start_notify(proto.STATUS_UUID, self._on_status)
        await client.start_notify(proto.DLDATA_UUID, self._on_packet)

        # The board has no RTC. Do this before anything else can start a
        # recording, so captures made during this session are timestamped.
        now_ms = time.time_ns() // 1_000_000
        await self._write(client, proto.cmd_set_time(now_ms))

        try:
            name = (await client.read_gatt_char(proto.CTRL_UUID)).decode("ascii")
            if name:
                self.device_name = name
                self._emit("log", f"device name: {name}")
        except Exception:                               # noqa: BLE001
            pass          # GET_ID needs protocol v3; older firmware just omits it

        await self._read_battery(client)

    async def _read_battery(self, client: BleakClient) -> None:
        try:
            raw = await client.read_gatt_char(proto.BATTERY_LEVEL_UUID)
            self._emit("battery", raw[0])
        except Exception:                               # noqa: BLE001
            pass          # a board without the fuel gauge simply has no level

    async def _handle(self, client: BleakClient, item, gone: asyncio.Event) -> None:
        kind, arg = item
        try:
            if kind == "scan":
                self._emit("log", "already connected; disconnect to rescan")
            elif kind == "cmd":
                await self._write(client, arg)
            elif kind == "list":
                await self._do_list(client)
            elif kind == "download":
                await self._do_download(client, *arg)
            elif kind == "erase":
                await self._bulk(client, proto.DL_ERASE, arg)
                self._emit("log", f"slot {arg} erased")
                await self._do_list(client)
            elif kind == "abort":
                await self._bulk(client, proto.DL_ABORT)
        except TransferError as exc:
            self._emit("error", str(exc))
        except Exception as exc:                        # noqa: BLE001
            self._emit("error", f"{kind} failed: {exc}")

    async def _write(self, client: BleakClient, command: str) -> None:
        try:
            await client.write_gatt_char(proto.CTRL_UUID,
                                         command.encode("ascii"), response=True)
            if command.startswith("TIME"):
                self._emit("clock", True)
            else:
                self._emit("log", f"{command} accepted")
        except Exception as exc:                        # noqa: BLE001
            # The firmware refuses with an ATT error - MARK when nothing is
            # recording, say - and bleak surfaces that as an exception.
            if command.startswith("TIME"):
                self._emit("clock", False)
            self._emit("error", f"{command} refused by the device ({exc})")

    async def _bulk(self, client: BleakClient, op: int, slot: int = 0,
                    offset: int = 0) -> None:
        await client.write_gatt_char(proto.DLCTRL_UUID,
                                     proto.pack_cmd(op, slot, offset),
                                     response=True)

    # -- notifications ---------------------------------------------------

    def _on_status(self, _char, data: bytearray) -> None:
        try:
            status = proto.Status.unpack(bytes(data))
        except ValueError as exc:
            self._emit("error", f"status: {exc} - firmware mismatch?")
            return
        if status.proto != proto.STATUS_PROTO:
            self._emit("error",
                       f"status protocol v{status.proto}, this app speaks "
                       f"v{proto.STATUS_PROTO} - update one of them")
            return
        self._emit("status", status)

    def _on_packet(self, _char, data: bytearray) -> None:
        # Runs on the bleak loop; hand it straight to whoever is collecting.
        if self._packets is not None:
            self._packets.put_nowait(bytes(data))

    # -- operations ------------------------------------------------------

    async def _do_list(self, client: BleakClient) -> None:
        slots: list[proto.SlotInfo] = []

        self._drain_packets()
        await self._bulk(client, proto.DL_LIST)

        deadline = time.monotonic() + LIST_TIMEOUT
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(self._packets.get(), 1.0)
            except asyncio.TimeoutError:
                # The listing has no explicit terminator - one packet per slot,
                # then silence. A quiet second after at least one slot means it
                # is over, which also copes with a board that has fewer slots
                # than this app assumes.
                if slots:
                    break
                continue

            pkt = proto.Packet.unpack(raw)
            if pkt.type == proto.DL_T_ERROR:
                raise TransferError(f"LIST refused: {pkt.error_text}")
            if pkt.type == proto.DL_T_SLOT_INFO:
                slots.append(proto.SlotInfo.unpack(pkt.payload))
                if len(slots) >= MAX_SLOTS:
                    break

        if not slots:
            raise TransferError("the device sent no slot listing")
        self._emit("slots", slots)

    def _drain_packets(self) -> None:
        """Discard anything left over from an aborted transfer."""
        while self._packets is not None and not self._packets.empty():
            self._packets.get_nowait()

    async def _do_download(self, client: BleakClient, slot: int, run_id: int,
                           ack: bool) -> None:
        """Fetch one capture, filling any gaps with further ranged GETs.

        Notifications go missing on a busy host, and the device has no way to
        know: it sends its trailer and considers the job done. So the trailer
        only ends a *round*. If the capture is not contiguous afterwards, the
        next round asks from the first missing byte, which is exactly what the
        protocol's GET <slot> <offset> is for.

        Every exit keeps the contiguous prefix on disk, so even a download that
        fails outright leaves the next attempt less to do.
        """
        self._abort.clear()
        self._drain_packets()

        # Continue an interrupted download rather than starting again. The run
        # id has to match: a slot is a location, and the device recycles them.
        previous = captures.load_partial(self.device_name, slot, run_id,
                                         self.data_root) if run_id else b""
        start = len(previous)
        if start:
            self._emit("log", f"resuming slot {slot} from {start} bytes")

        asm = Reassembler(slot=slot, start=start)
        single_round = True

        for attempt in range(MAX_ROUNDS):
            before = asm.contiguous
            offset = asm.begin_round()
            if attempt:
                single_round = False
                self._emit("log",
                           f"slot {slot}: {asm.total - before} bytes still "
                           f"missing, asking again from {offset}")
            await self._bulk(client, proto.DL_GET, slot, offset)

            await self._collect_round(client, asm, slot, start, previous)
            if asm.complete:
                break

            if attempt and asm.contiguous == before:
                # A whole round went by and not one more usable byte arrived.
                # Asking again would just hang on the same thing; stop and keep
                # what we have so the next attempt starts from here.
                self._emit("log", f"slot {slot}: no progress this round")
                break
        else:
            self._keep_partial(asm, previous, slot, start)
            raise TransferError(
                f"slot {slot}: still {asm.total - asm.contiguous} bytes short "
                f"after {MAX_ROUNDS} attempts. The {asm.contiguous} bytes "
                "received are kept, so downloading again continues from there.")

        if not asm.complete:
            self._keep_partial(asm, previous, slot, start)
            raise TransferError(
                f"slot {slot}: the device stopped sending with "
                f"{asm.total - asm.contiguous} bytes to go. What arrived is "
                "kept, so downloading again continues from there.")

        data = merge_resumed(previous, asm.assemble(), start)

        # The trailer covers the bytes one round sent, so it only means anything
        # when the whole thing arrived in one. Across rounds the capture's own
        # stored CRC is the check that matters, and captures.verify() applies it.
        if single_round and not start:
            asm.verify_trailer(data)
        captures.verify(data, asm.info)
        captures.clear_partial(self.device_name, slot, self.data_root)

        written = captures.save_capture(asm.info, data, source="ble",
                                        root=self.data_root,
                                        device_name=self.device_name)
        self._emit("log", f"saved {written['csv'].name} "
                          f"({asm.info.record_count} records)")

        # Only acknowledge after the bytes are on disk and verified: the ack is
        # what lets the device recycle the slot.
        if ack and asm.info.state == proto.SLOT_COMPLETE:
            await self._bulk(client, proto.DL_ACK, slot)
            self._emit("log", f"slot {slot} acknowledged; the device may reuse it")

        self._emit("downloaded", written)
        await self._do_list(client)

    async def _collect_round(self, client: BleakClient, asm: Reassembler,
                             slot: int, start: int, previous: bytes) -> None:
        """Drain packets until this round goes quiet or the capture is whole."""
        deadline = time.monotonic() + TRANSFER_TIMEOUT
        last_report = 0.0

        while not asm.complete:
            if self._abort.is_set():
                await self._bulk(client, proto.DL_ABORT)
                self._keep_partial(asm, previous, slot, start)
                raise TransferError("download cancelled")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._keep_partial(asm, previous, slot, start)
                raise TransferError(
                    f"slot {slot}: the device went quiet with "
                    f"{asm.total - asm.contiguous} bytes to go. What arrived "
                    "is kept, so downloading again continues from there.")

            try:
                raw = await asyncio.wait_for(self._packets.get(),
                                             min(QUIET_AFTER_TRAILER, remaining))
            except asyncio.TimeoutError:
                # The device said it had finished and nothing more is arriving,
                # so whatever is missing was lost. End the round and let the
                # caller ask again for the gap.
                if asm.trailer_seen:
                    return
                continue

            asm.feed(proto.Packet.unpack(raw))
            deadline = time.monotonic() + TRANSFER_TIMEOUT   # progress resets it

            now = time.monotonic()
            if now - last_report > 0.2:
                last_report = now
                self._emit("progress",
                           (slot, start + asm.contiguous, start + asm.total))

        self._emit("progress", (slot, start + asm.contiguous, start + asm.total))

    def _keep_partial(self, asm: Reassembler, previous: bytes, slot: int,
                      start: int) -> None:
        """Save whatever arrived before the link went quiet.

        Only the contiguous prefix is kept: bytes past a gap cannot be placed
        without the missing ones, and writing them would put the wrong samples
        at the wrong offsets.
        """
        if asm.info is None:
            return
        contiguous = asm.next_offset()
        if contiguous <= start:
            return
        chunk = bytearray()
        offset = start
        while offset < contiguous:
            chunk += asm.chunks[offset]
            offset += len(asm.chunks[offset])
        captures.save_partial(self.device_name, slot, asm.info.run_id,
                              previous[:start] + bytes(chunk),
                              asm.info.data_bytes, self.data_root)
