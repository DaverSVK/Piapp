#!/usr/bin/env python3
"""Smart Jersey - Raspberry Pi app for the Thingy:53 IMU recorders.

    python -m pi_app.app

Four screens, in the order the work happens:

    DEVICES     scan, and connect to one unit
    DEVICE      status, clock, battery, storage
    RECORDINGS  what is stored on the board; download one
    PROCESSING  estimate, review, publish

The only automatic action on connect is the clock sync - the board has no RTC,
and without it a capture has no absolute start time. Nothing is downloaded
without being asked for, and nothing is published without being asked for.

Everything slow (scanning, downloading, the ESKF, publishing) runs on a worker
thread and reports through one event queue, so the window never blocks. The
Bluetooth worker is pi_app/ble/connection.py; the processing worker is
pi_app/processing/pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import tkinter as tk
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation import pipeline as est_pipeline          # noqa: E402
from pi_app.ble import protocol as proto                 # noqa: E402
from pi_app.ble.connection import DeviceSession          # noqa: E402
from pi_app.processing import pipeline as work           # noqa: E402
from pi_app.storage import captures                      # noqa: E402

POLL_MS = 50
LOG_LINES = 300


def fmt_duration(ms: int) -> str:
    seconds = int(ms) // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


def fmt_bytes(n: int) -> str:
    return f"{n / 1024:.0f} KiB" if n < 1024 * 1024 else f"{n / 1048576:.2f} MiB"


class App(ttk.Frame):
    def __init__(self, master: tk.Tk, args: argparse.Namespace):
        super().__init__(master, padding=10)
        self.pack(fill="both", expand=True)

        self.args = args
        self.events: queue.Queue = queue.Queue()
        self.session = DeviceSession(self.events, data_root=args.data_dir)
        self.worker = work.Worker(self.events)

        self.devices: list = []
        self.captures: list = []
        self.slots: list[proto.SlotInfo] = []
        self.status: proto.Status | None = None
        self.selected_capture: dict | None = None
        self.last_bundle: Path | None = None
        self.battery: int | None = None
        self.clock_ok = False
        self._last_error: str | None = None

        self._build()
        self.session.start()
        self.after(POLL_MS, self._pump)
        self.session.scan()

    # ------------------------------------------------------------- layout

    def _build(self) -> None:
        self.master.title("Smart Jersey")

        header = ttk.Frame(self)
        header.pack(fill="x")
        self.title_var = tk.StringVar(value="SMART JERSEY")
        ttk.Label(header, textvariable=self.title_var,
                  font=("TkDefaultFont", 14, "bold")).pack(side="left")
        self.conn_var = tk.StringVar(value="disconnected")
        ttk.Label(header, textvariable=self.conn_var).pack(side="right")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, pady=(8, 6))
        self._build_devices()
        self._build_device()
        self._build_recordings()
        self._build_processing()

        self.log = tk.Text(self, height=7, wrap="word", state="disabled")
        self.log.pack(fill="x")

    def _build_devices(self) -> None:
        page = ttk.Frame(self.nb, padding=8)
        self.nb.add(page, text="Devices")

        self.device_list = ttk.Treeview(
            page, columns=("signal", "address"), show="tree headings", height=8)
        self.device_list.heading("#0", text="Device")
        self.device_list.heading("signal", text="Signal")
        self.device_list.heading("address", text="Address")
        self.device_list.column("#0", width=160)
        self.device_list.column("signal", width=90, anchor="e")
        self.device_list.pack(fill="both", expand=True)
        self.device_list.bind("<Double-1>", lambda _e: self._connect())

        row = ttk.Frame(page)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Rescan", command=self._rescan).pack(side="left")
        ttk.Button(row, text="Connect", command=self._connect).pack(side="left", padx=6)
        ttk.Button(row, text="Disconnect",
                   command=self._disconnect).pack(side="left")

    def _build_device(self) -> None:
        page = ttk.Frame(self.nb, padding=8)
        self.nb.add(page, text="Device")

        self.detail_vars = {}
        for label in ("Connection", "Clock", "Battery", "Storage",
                      "Recording", "Recorded runs"):
            row = ttk.Frame(page)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{label}:", width=16).pack(side="left")
            var = tk.StringVar(value="-")
            ttk.Label(row, textvariable=var).pack(side="left")
            self.detail_vars[label] = var

        row = ttk.Frame(page)
        row.pack(fill="x", pady=(10, 0))
        ttk.Button(row, text="Refresh",
                   command=self.session.list_runs).pack(side="left")
        ttk.Button(row, text="Rename",
                   command=self._rename).pack(side="left", padx=6)
        ttk.Button(row, text="Mark",
                   command=lambda: self.session.send("MARK")).pack(side="left")

    def _build_recordings(self) -> None:
        page = ttk.Frame(self.nb, padding=8)
        self.nb.add(page, text="Recordings")

        cols = ("started", "duration", "samples", "state")
        self.run_list = ttk.Treeview(page, columns=cols, show="tree headings",
                                     height=9)
        self.run_list.heading("#0", text="Run")
        for col, text, width in (("started", "Started", 150),
                                 ("duration", "Duration", 80),
                                 ("samples", "Samples", 90),
                                 ("state", "Status", 110)):
            self.run_list.heading(col, text=text)
            self.run_list.column(col, width=width,
                                 anchor="e" if col != "state" else "w")
        self.run_list.column("#0", width=80)
        self.run_list.pack(fill="both", expand=True)

        self.progress = ttk.Progressbar(page, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", pady=(8, 2))
        self.progress_var = tk.StringVar(value="")
        ttk.Label(page, textvariable=self.progress_var).pack(anchor="w")

        row = ttk.Frame(page)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Refresh",
                   command=self.session.list_runs).pack(side="left")
        ttk.Button(row, text="Download",
                   command=self._download).pack(side="left", padx=6)
        ttk.Button(row, text="Cancel",
                   command=self.session.cancel_transfer).pack(side="left")
        ttk.Button(row, text="Erase",
                   command=self._erase).pack(side="right")

    def _build_processing(self) -> None:
        page = ttk.Frame(self.nb, padding=8)
        self.nb.add(page, text="Processing")

        cols = ("device", "started", "state")
        self.capture_list = ttk.Treeview(page, columns=cols,
                                         show="tree headings", height=7)
        self.capture_list.heading("#0", text="Capture")
        for col, text, width in (("device", "Device", 100),
                                 ("started", "Started", 150),
                                 ("state", "Status", 110)):
            self.capture_list.heading(col, text=text)
            self.capture_list.column(col, width=width)
        self.capture_list.column("#0", width=140)
        self.capture_list.pack(fill="both", expand=True)
        self.capture_list.bind("<<TreeviewSelect>>", self._on_capture_selected)

        self.summary = tk.Text(page, height=8, wrap="none", state="disabled")
        self.summary.pack(fill="x", pady=(8, 0))

        row = ttk.Frame(page)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Refresh",
                   command=self._refresh_captures).pack(side="left")
        ttk.Button(row, text="Process",
                   command=self._process).pack(side="left", padx=6)
        ttk.Button(row, text="Publish",
                   command=self._publish).pack(side="left")
        ttk.Label(row, text="Session:").pack(side="left", padx=(16, 4))
        self.session_var = tk.StringVar(value=self.args.session_id or "")
        ttk.Entry(row, textvariable=self.session_var,
                  width=22).pack(side="left")
        self.dry_var = tk.BooleanVar(value=self.args.dry_run)
        ttk.Checkbutton(row, text="Dry run",
                        variable=self.dry_var).pack(side="left", padx=6)

        self._refresh_captures()

    # ------------------------------------------------------------ actions

    def _say(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        # Keep the buffer bounded: a long download logs a lot.
        if int(self.log.index("end-1c").split(".")[0]) > LOG_LINES:
            self.log.delete("1.0", "2.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _rescan(self) -> None:
        self.device_list.delete(*self.device_list.get_children())
        self.session.scan()

    def _selected_device(self):
        sel = self.device_list.selection()
        if not sel:
            self._say("pick a device first")
            return None
        index = int(sel[0])
        return self.devices[index] if index < len(self.devices) else None

    def _connect(self) -> None:
        found = self._selected_device()
        if found:
            self._say(f"connecting to {found.label} ...")
            self.session.connect(found.address)

    def _disconnect(self) -> None:
        self.session.disconnect()

    def _rename(self) -> None:
        name = simpledialog.askstring(
            "Rename device", "New device name (e.g. DEV-12):", parent=self)
        if not name:
            return
        if not proto.valid_device_name(name):
            messagebox.showerror(
                "Rename device",
                f"'{name}' will not fit or contains a space.\n"
                f"Use 3 to {proto.DEVID_SIZE - 1} printable characters, "
                "no spaces - for example DEV-12.")
            return
        self.session.send(proto.cmd_set_id(name))
        self._say("renamed; the scan list updates after you disconnect")

    def _selected_slot(self) -> proto.SlotInfo | None:
        sel = self.run_list.selection()
        if not sel:
            self._say("pick a recording first")
            return None
        slot = int(sel[0])
        return next((s for s in self.slots if s.slot == slot), None)

    def _download(self) -> None:
        info = self._selected_slot()
        if info is None:
            return
        if not info.downloadable:
            self._say(f"slot {info.slot} is empty")
            return
        self.progress["value"] = 0
        self.progress_var.set(f"starting slot {info.slot} ...")
        self.session.download(info.slot, info.run_id)

    def _erase(self) -> None:
        info = self._selected_slot()
        if info is None or not info.downloadable:
            return
        if info.state != proto.SLOT_DOWNLOADED and not messagebox.askyesno(
                "Erase recording",
                f"Run {info.run_id} has not been downloaded and acknowledged.\n"
                "Erasing it destroys the only copy. Continue?"):
            return
        self.session.erase(info.slot)

    def _refresh_captures(self) -> None:
        self.capture_list.delete(*self.capture_list.get_children())
        self.captures = captures.discover(self.args.data_dir)
        for i, entry in enumerate(self.captures):
            meta = entry["meta"]
            self.capture_list.insert(
                "", "end", iid=str(i), text=entry["stem"],
                values=(meta.get("deviceId", "?"),
                        (meta.get("startIso") or "-")[:19].replace("T", " "),
                        work.run_state_for(entry)))

    def _selected_capture(self) -> dict | None:
        sel = self.capture_list.selection()
        if not sel:
            self._say("pick a capture first")
            return None
        index = int(sel[0])
        if index >= len(self.captures):
            # The list was rebuilt under the selection; ask again rather than
            # acting on whatever now sits at that row.
            self._say("the capture list changed; pick again")
            return None
        return self.captures[index]

    def _on_capture_selected(self, _event) -> None:
        # Clearing the list fires this with nothing selected, so an unguarded
        # selection()[0] raises - and because the clear happens inside an event
        # handler, that used to take the whole event pump down with it.
        sel = self.capture_list.selection()
        if not sel:
            self.selected_capture = None
            return
        index = int(sel[0])
        if index >= len(self.captures):
            return
        entry = self.captures[index]
        self.selected_capture = entry
        if entry["bundle"].exists():
            try:
                bundle = json.loads(entry["bundle"].read_text(encoding="utf-8"))
                self._show_summary(work.summarise(bundle))
                self.last_bundle = entry["bundle"]
                return
            except ValueError:
                pass
        self._show_summary("Not processed yet.")

    def _show_summary(self, text: str) -> None:
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", text)
        self.summary.configure(state="disabled")

    def _process(self) -> None:
        entry = self._selected_capture()
        if entry is None:
            return

        start_ms = entry["meta"].get("startUnixMs") or 0
        if not start_ms:
            start_ms = self._ask_start_time(entry)
            if start_ms is None:
                return

        opts = est_pipeline.Options(
            extended=self.args.extended,
            start_unix_ms=start_ms or None)
        if self.worker.process(entry["csv"], opts, entry["bundle"]):
            self._show_summary(f"Processing {entry['stem']} ...")

    def _ask_start_time(self, entry: dict) -> int | None:
        """Get an absolute start for a capture recorded before a clock sync.

        The board has no RTC, so anything recorded before the app first
        connected carries no wall-clock time and every sampleAt would be
        fiction. Rather than refuse outright - the samples themselves are
        perfectly good - ask, defaulting to the download time less the
        capture's own duration. That is an estimate, and the bundle says so.
        """
        meta = entry["meta"]
        default = datetime.now(timezone.utc)
        downloaded = meta.get("downloadedAt")
        if downloaded:
            try:
                default = datetime.fromisoformat(downloaded.replace("Z", "+00:00"))
            except ValueError:
                pass
        default -= timedelta(milliseconds=meta.get("durationMs") or 0)

        answer = simpledialog.askstring(
            "When was this recorded?",
            f"{entry['stem']} was recorded before the board's clock was set, "
            "so it carries no start time.\n\n"
            "Enter when the recording started (ISO-8601 UTC, or epoch ms).\n"
            "The suggestion is the download time minus the capture length - "
            "an estimate.",
            initialvalue=default.strftime("%Y-%m-%dT%H:%M:%SZ"),
            parent=self)
        if not answer:
            return None
        try:
            return est_pipeline.parse_start_time(answer)
        except ValueError:
            messagebox.showerror(
                "When was this recorded?",
                f"Could not read '{answer}' as a time.\n\n"
                "Use ISO-8601 (2026-09-09T08:35:38Z) or epoch milliseconds.")
            return None

    def _publish(self) -> None:
        entry = self._selected_capture()
        if entry is None:
            return
        if not entry["bundle"].exists():
            self._say("process the capture first")
            return
        session_id = self.session_var.get().strip()
        if not session_id:
            messagebox.showerror("Publish",
                                 "Enter the server-side session id first.")
            return
        self.worker.publish(entry["bundle"], session_id,
                            dry_run=self.dry_var.get())

    # ------------------------------------------------------------- events

    def _pump(self) -> None:
        """Drain the worker events into the UI.

        The reschedule is in a finally: this loop must survive anything a
        handler throws. Losing it once means every later event is silently
        dropped and the window freezes on whatever it happened to be showing -
        which reads as a hung job rather than a bug, and is very hard to place.
        """
        try:
            while True:
                try:
                    kind, payload = self.events.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._on_event(kind, payload)
                except Exception as exc:              # noqa: BLE001
                    traceback.print_exc()
                    self._say(f"ERROR  displaying {kind}: {exc!r}")
        finally:
            self.after(POLL_MS, self._pump)

    def _on_event(self, kind: str, payload) -> None:
        if kind == "devices":
            self._on_devices(payload)
        elif kind == "conn":
            self.conn_var.set(payload)
            if payload == "connected":
                self.title_var.set(self.session.device_name or "SMART JERSEY")
                self.detail_vars["Connection"].set("Connected")
                self.session.list_runs()
                self.nb.select(1)
            elif payload == "disconnected":
                self.detail_vars["Connection"].set("Disconnected")
                self.detail_vars["Clock"].set("-")
                self.clock_ok = False
        elif kind == "clock":
            self.clock_ok = bool(payload)
            self.detail_vars["Clock"].set(
                "Synchronized" if payload else "NOT SET - captures will have "
                                               "no start time")
        elif kind == "battery":
            self.battery = payload
            self.detail_vars["Battery"].set(f"{payload} %")
        elif kind == "status":
            self._on_status(payload)
        elif kind == "slots":
            self._on_slots(payload)
        elif kind == "progress":
            self._on_progress(*payload)
        elif kind == "downloaded":
            self._say(f"saved {payload['csv']}")
            self._refresh_captures()
            self.nb.select(3)
        elif kind == "processed":
            self._show_summary(payload["summary"])
            self.last_bundle = payload["bundle"]
            self._say(f"wrote {payload['bundle'].name}")
            for note in payload["notes"]:
                self._say(f"  note: {note}")
            self._refresh_captures()
        elif kind == "published":
            self._say(payload["output"] or "published")
            self._refresh_captures()
        elif kind == "stage":
            self._say(f"{payload[0]}: {payload[1]}")
            if payload[0] == "error":
                # The summary box still says "Processing ..." at this point.
                # Leaving it there is how a failure reads as a five-minute hang.
                self._show_summary(self._last_error
                                   or f"{payload[1]}: failed. See the log below.")
        elif kind == "error":
            self._last_error = str(payload)
            self._say(f"ERROR  {payload}")
        elif kind == "log":
            self._say(payload)

    def _on_devices(self, found) -> None:
        self.devices = found
        self.device_list.delete(*self.device_list.get_children())
        for i, dev in enumerate(found):
            self.device_list.insert("", "end", iid=str(i), text=dev.label,
                                    values=(dev.signal, dev.address))
        self._say(f"{len(found)} recorder(s) in range"
                  if found else "no recorders in range")

    def _on_status(self, status: proto.Status) -> None:
        self.status = status
        self.detail_vars["Recording"].set(
            f"Yes - slot {status.slot}, {fmt_duration(status.elapsed_ms)}, "
            f"{status.records} records" if status.recording else "No")
        self.detail_vars["Storage"].set(
            f"{status.free_slots} slot(s) free, active slot {status.fill_pct}% full")
        if status.clock_synced != self.clock_ok:
            self.clock_ok = status.clock_synced
            self.detail_vars["Clock"].set(
                "Synchronized" if status.clock_synced else "NOT SET")
        for problem in status.problems():
            self._say(f"device: {problem}")

    def _on_slots(self, slots) -> None:
        self.slots = slots
        self.run_list.delete(*self.run_list.get_children())
        stored = 0
        for info in slots:
            if not info.downloadable:
                continue
            stored += 1
            started = captures.start_datetime(info.start_unix_ms)
            self.run_list.insert(
                "", "end", iid=str(info.slot), text=f"{info.run_id:04d}",
                values=(started.astimezone().strftime("%d %b %Y %H:%M:%S")
                        if started else "unknown",
                        fmt_duration(info.duration_ms),
                        f"{info.record_count:,}",
                        info.state_name))
        self.detail_vars["Recorded runs"].set(str(stored))

    def _on_progress(self, slot: int, received: int, total: int) -> None:
        self.progress["value"] = int(1000 * received / total) if total else 0
        pct = (100 * received / total) if total else 0
        self.progress_var.set(
            f"slot {slot}: {pct:.0f} %   {fmt_bytes(received)} / {fmt_bytes(total)}")

    def close(self) -> None:
        self.session.shutdown()
        self.master.destroy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=captures.DEFAULT_ROOT,
                    help="where captures are stored "
                         f"(default {captures.DEFAULT_ROOT})")
    ap.add_argument("--session-id", help="prefill the Smart Jersey session id")
    ap.add_argument("--dry-run", action="store_true",
                    help="start with publishing set to dry-run")
    ap.add_argument("--extended", action="store_true",
                    help="include the optional backend telemetry fields")
    args = ap.parse_args()

    root = tk.Tk()
    root.geometry("880x680")
    app = App(root, args)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
