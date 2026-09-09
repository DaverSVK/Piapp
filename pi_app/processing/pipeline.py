"""Estimation and publishing, off the UI thread.

Both are slow enough to freeze a window: the ESKF runs tens of thousands of
15x15 updates, and a publish is hundreds of HTTPS round trips. Each runs on a
worker thread and reports through the same event queue the Bluetooth session
uses, so the UI has one place to look.

The work itself lives elsewhere - the estimator in estimation/ (which knows
nothing about Bluetooth, disk layout or HTTP) and the upload in
pi_app/backend/publisher.py. This module is only the thread and the bookkeeping
around it.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from estimation import capture as capture_mod
from estimation import pipeline as est_pipeline
from pi_app.backend import publisher
from pi_app.storage import captures


class Worker:
    """Runs one job at a time, reporting to an event queue.

    Jobs are serialised deliberately: processing two captures at once on a Pi
    would only make both slower, and a single busy flag is something the UI can
    render honestly.
    """

    def __init__(self, events: queue.Queue):
        self.events = events
        self._lock = threading.Lock()
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def _emit(self, kind: str, payload=None) -> None:
        self.events.put((kind, payload))

    def _start(self, target, *args) -> bool:
        with self._lock:
            if self._busy:
                self._emit("log", "already working; wait for it to finish")
                return False
            self._busy = True
        threading.Thread(target=self._wrap, args=(target, args),
                         daemon=True).start()
        return True

    def _wrap(self, target, args) -> None:
        try:
            target(*args)
        finally:
            self._busy = False

    # -- processing ------------------------------------------------------

    def process(self, csv_path: Path, options: est_pipeline.Options | None = None,
                bundle_path: Path | None = None) -> bool:
        return self._start(self._process, Path(csv_path), options, bundle_path)

    def _process(self, csv_path: Path, options, bundle_path) -> None:
        self._emit("stage", ("processing", csv_path.name))
        try:
            result = est_pipeline.process(csv_path, options)
        except (capture_mod.CaptureError, ValueError, OSError) as exc:
            self._emit("error", f"{csv_path.name}: {exc}")
            self._emit("stage", ("error", csv_path.name))
            return

        dest = Path(bundle_path or csv_path.with_name(
            csv_path.stem + ".bundle.json"))
        dest.write_text(json.dumps(result.bundle, indent=2) + "\n",
                        encoding="utf-8")

        self._emit("processed", {
            "csv": csv_path,
            "bundle": dest,
            "summary": summarise(result.bundle),
            "notes": result.notes,
        })
        self._emit("stage", ("processed", csv_path.name))

    # -- publishing ------------------------------------------------------

    def publish(self, bundle_path: Path, session_id: str,
                dry_run: bool = False,
                burst_size: int = publisher.DEFAULT_BURST) -> bool:
        return self._start(self._publish, Path(bundle_path), session_id,
                           dry_run, burst_size)

    def _publish(self, bundle_path: Path, session_id: str, dry_run: bool,
                 burst_size: int) -> None:
        self._emit("stage", ("publishing", bundle_path.name))
        try:
            result = publisher.publish(bundle_path, session_id,
                                       dry_run=dry_run, burst_size=burst_size)
        except publisher.PublishError as exc:
            # A failed publish never costs the capture: the raw data, the CSV
            # and the bundle all stay on disk, so this is a retry, not a redo.
            self._emit("error", f"publish failed: {exc}")
            self._emit("stage", ("error", bundle_path.name))
            return

        self._emit("published", {"bundle": result.bundle,
                                 "output": result.output})
        self._emit("stage", ("published", bundle_path.name))


def summarise(bundle: dict) -> str:
    """The few numbers worth putting on a screen after processing."""
    src = bundle.get("source", {})
    q = src.get("quality", {})
    stats = (bundle.get("playerStats") or [{}])[0]

    lines = [
        f"Duration:   {src.get('durationS', 0):.2f} s",
        f"Max speed:  {stats.get('maxSpeedKmh', 0):.1f} km/h",
        f"Avg speed:  {stats.get('avgSpeedKmh', 0):.1f} km/h",
        f"Distance:   {stats.get('distanceM', 0):.0f} m",
        f"Samples:    {len(bundle.get('telemetry', [])):,}",
        f"Still:      {q.get('stationaryFraction', 0) * 100:.0f}% "
        f"({q.get('zuptCount', 0)} ZUPTs)",
        f"Position:   +/-{q.get('finalPosSigmaM', 0):.1f} m by the end",
    ]
    if q.get("clampedSamples"):
        lines.append(f"WARNING:    {q['clampedSamples']} speed sample(s) "
                     "clamped - the filter diverged")
    return "\n".join(lines)


def run_state_for(entry: dict) -> str:
    """Where a discovered capture has got to, from what is on disk."""
    if entry["bundle"].exists():
        return captures.PROCESSED
    if entry["csv"].exists():
        return captures.DOWNLOADED
    return captures.ON_DEVICE
