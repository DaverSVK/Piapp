"""Publishing a bundle: the request sequence, batching, and refusals.

Runs the real sender as a subprocess in --dry-run, so it exercises the actual
argument parsing and request construction rather than a stand-in for them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from estimation import pipeline as est_pipeline
from estimation import telemetry as tele
from estimation.tests.conftest import still_rows, write_capture

REPO_ROOT = Path(__file__).resolve().parents[2]
SENDER = REPO_ROOT / "API_to_backend" / "send_stats.py"

START_MS = 1788877265000


@pytest.fixture
def bundle_file(tmp_path):
    csv = write_capture(tmp_path / "run_0063.csv", still_rows(450))
    result = est_pipeline.process(csv, est_pipeline.Options(
        device_id="DEV-12", start_unix_ms=START_MS, extended=True))
    path = tmp_path / "run_0063.bundle.json"
    path.write_text(json.dumps(result.bundle), encoding="utf-8")
    return path


def send(*args, expect_ok: bool = True):
    proc = subprocess.run([sys.executable, str(SENDER), *args],
                          cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=120)
    if expect_ok:
        assert proc.returncode == 0, proc.stderr
    return proc


def posts(output: str) -> list[str]:
    return [line.split()[1] for line in output.splitlines()
            if line.startswith("POST ")]


def test_a_bundle_posts_the_backend_sequence(bundle_file):
    proc = send("--session-id", "S1", "--bundle", str(bundle_file),
                "--burst-size", "200", "--dry-run")

    urls = posts(proc.stdout)
    assert urls[0].endswith("/connect")
    assert urls[-1].endswith("/general-stats")
    assert urls[-2].endswith("/playerStats")
    assert all(u.endswith("/telemetry") for u in urls[1:-2])
    assert all("/sessions/S1/" in u for u in urls)


def test_telemetry_is_split_into_bursts(bundle_file):
    total = len(json.loads(bundle_file.read_text())["telemetry"])

    proc = send("--session-id", "S1", "--bundle", str(bundle_file),
                "--burst-size", "100", "--dry-run")

    telemetry_posts = [u for u in posts(proc.stdout) if u.endswith("/telemetry")]
    assert len(telemetry_posts) == -(-total // 100)   # ceiling division


def test_the_records_sent_are_the_records_in_the_bundle(bundle_file):
    bundle = json.loads(bundle_file.read_text())
    proc = send("--session-id", "S1", "--bundle", str(bundle_file),
                "--burst-size", "1000", "--dry-run")

    # The dry run prints each body as indented JSON after its POST line.
    body = proc.stdout.split("/telemetry\n", 1)[1]
    decoder = json.JSONDecoder()
    sent, _ = decoder.raw_decode(body.lstrip())

    assert sent[0] == bundle["telemetry"][0]
    assert len(sent) == len(bundle["telemetry"])


def test_connect_uses_the_capture_start_not_now(bundle_file):
    """The session must bracket when the skating happened, not when it uploaded."""
    proc = send("--session-id", "S1", "--bundle", str(bundle_file), "--dry-run")

    body = proc.stdout.split("/connect\n", 1)[1].lstrip().splitlines()[0]
    assert body.strip().strip('"').startswith("2026-")


def test_keep_session_open_omits_the_general_stats(bundle_file):
    proc = send("--session-id", "S1", "--bundle", str(bundle_file),
                "--dry-run", "--keep-session-open")

    assert not any(u.endswith("/general-stats") for u in posts(proc.stdout))


def test_skip_connect_omits_the_connect(bundle_file):
    proc = send("--session-id", "S1", "--bundle", str(bundle_file),
                "--dry-run", "--skip-connect")

    assert not any(u.endswith("/connect") for u in posts(proc.stdout))


def test_a_file_that_is_not_a_bundle_is_refused(tmp_path):
    path = tmp_path / "nope.json"
    path.write_text('{"schema": "something-else"}', encoding="utf-8")

    proc = send("--session-id", "S1", "--bundle", str(path), "--dry-run",
                expect_ok=False)

    assert proc.returncode != 0
    assert "not a imu_telemetry_bundle" in (proc.stdout + proc.stderr)


def test_a_missing_bundle_is_refused(tmp_path):
    proc = send("--session-id", "S1", "--bundle", str(tmp_path / "gone.json"),
                "--dry-run", expect_ok=False)

    assert proc.returncode != 0
    assert "cannot read bundle" in (proc.stdout + proc.stderr)


def test_dummy_mode_still_works():
    """The generator stays: it is the quickest credentials smoke test."""
    proc = send("--session-id", "S1", "--samples", "10", "--dry-run")

    assert any(u.endswith("/telemetry") for u in posts(proc.stdout))


def test_the_bundle_summary_warns_about_a_clamped_result(tmp_path):
    csv = write_capture(tmp_path / "run_0063.csv", still_rows(300))
    result = est_pipeline.process(csv, est_pipeline.Options(
        device_id="DEV-12", start_unix_ms=START_MS))
    # Force the honesty guard: pretend the filter diverged.
    result.bundle["source"]["quality"]["clampedSamples"] = 12
    result.bundle["source"]["quality"]["finalPosSigmaM"] = 120.0
    path = tmp_path / "b.json"
    path.write_text(json.dumps(result.bundle), encoding="utf-8")

    proc = send("--session-id", "S1", "--bundle", str(path), "--dry-run")

    assert "clamped" in proc.stdout
    assert "position uncertainty" in proc.stdout


def test_bundle_schema_constants_agree():
    """The writer and the reader must name the same schema."""
    source = SENDER.read_text(encoding="utf-8")
    assert f'BUNDLE_SCHEMA = "{tele.BUNDLE_SCHEMA}"' in source
