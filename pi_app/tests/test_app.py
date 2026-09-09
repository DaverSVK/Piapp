"""The event pump and the capture list.

These exist because of a real freeze: refreshing the capture list fires
``<<TreeviewSelect>>`` with nothing selected, the handler indexed ``[0]`` of an
empty tuple, and the resulting IndexError escaped the pump - which then never
rescheduled itself. Every later event was silently dropped and the window sat
on "Processing run_0101 ..." while the job had in fact finished.

Nothing here needs a device or a network; it drives the real widgets.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock

import pytest

tk = pytest.importorskip("tkinter", reason="the app needs tkinter")


@pytest.fixture(scope="module")
def root():
    """One Tk root for the module.

    Tk does not reliably support creating a second root in the same process -
    the interpreter teardown leaves it unable to find its library - so the
    root is shared and only the App frame is rebuilt per test.
    """
    try:
        r = tk.Tk()
    except tk.TclError as exc:                      # headless CI
        pytest.skip(f"no display: {exc}")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def app(root, tmp_path, monkeypatch):
    """A real App on a hidden root, with the Bluetooth worker stubbed out."""
    from pi_app import app as app_mod

    # The session would otherwise open a Bluetooth adapter and start scanning.
    # A Mock stands in for the whole of it: the UI only ever calls methods on
    # it, and every call is meant to be non-blocking anyway.
    def make_session(events, data_root=None):
        session = MagicMock()
        session.events = events
        session.data_root = data_root
        session.device_name = "DEV-TEST"
        session.connected = False
        return session

    monkeypatch.setattr(app_mod, "DeviceSession", make_session)

    args = app_mod.argparse.Namespace(
        data_dir=tmp_path, session_id=None, dry_run=True, extended=False)
    instance = app_mod.App(root, args)
    yield instance
    instance.destroy()


def _pump_once(app) -> None:
    """Run one drain, the way the scheduled callback does."""
    app._pump()


def test_refreshing_the_capture_list_does_not_raise(app):
    """The exact call that used to kill the pump."""
    app._refresh_captures()
    app._refresh_captures()          # again, now that the list has been cleared


def test_clearing_a_selected_list_is_survivable(app, tmp_path):
    """Selecting a row and then rebuilding the list fired the IndexError."""
    app.capture_list.insert("", "end", iid="0", text="run_0001",
                            values=("DEV-TEST", "-", "DOWNLOADED"))
    app.captures = [{"meta": {}, "stem": "run_0001",
                     "bundle": tmp_path / "nope.json"}]
    app.capture_list.selection_set("0")

    # delete() fires <<TreeviewSelect>> with an empty selection, synchronously.
    app._refresh_captures()

    assert app.selected_capture is None


def test_a_handler_that_throws_does_not_stop_the_pump(app):
    """One bad event must cost one event, not every event after it."""
    boom = []

    def explode(kind, payload):
        boom.append(kind)
        raise RuntimeError("handler blew up")

    app._on_event = explode

    app.events.put(("log", "first"))
    app.events.put(("log", "second"))
    _pump_once(app)

    # Both events were delivered despite the first handler raising, and the
    # pump scheduled itself again rather than dying.
    assert boom == ["log", "log"]


def test_events_still_render_after_a_bad_one(app):
    seen = []
    real = app._on_event

    def flaky(kind, payload):
        if kind == "bad":
            raise ValueError("nope")
        seen.append(kind)
        real(kind, payload)

    app._on_event = flaky
    app.events.put(("bad", None))
    app.events.put(("log", "still alive"))
    _pump_once(app)

    assert seen == ["log"]


def test_selecting_nothing_is_reported_not_raised(app):
    app.capture_list.selection_remove(*app.capture_list.get_children())
    assert app._selected_capture() is None


def test_a_stale_selection_index_is_refused(app, tmp_path):
    """A row can outlive the list entry it pointed at."""
    app.capture_list.insert("", "end", iid="7", text="run_0007", values=("", "", ""))
    app.capture_list.selection_set("7")
    app.captures = []                                # the list shrank

    assert app._selected_capture() is None


def test_processing_summary_is_replaced_on_failure(app):
    """A failure must not leave the panel saying 'Processing ...' forever."""
    app._show_summary("Processing run_0101 ...")
    app.events.put(("error", "run_0101.csv: something went wrong"))
    app.events.put(("stage", ("error", "run_0101.csv")))
    _pump_once(app)

    shown = app.summary.get("1.0", "end").strip()
    assert "Processing" not in shown
    assert "something went wrong" in shown
