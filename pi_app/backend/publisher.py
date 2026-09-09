"""Sending a finished bundle to the Smart Jersey backend.

Runs API_to_backend/send_stats.py as a subprocess rather than importing it. Two
reasons: the sender is also the supported command-line tool, so driving it this
way means the app and a person at a terminal exercise exactly the same code
path; and its OAuth credentials come from the environment, which a subprocess
inherits without the app ever holding them.

A failed publish is a retry, never a redo: the raw capture, the CSV and the
bundle all stay on disk, so nothing has to be re-downloaded or re-estimated.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SENDER = REPO_ROOT / "API_to_backend" / "send_stats.py"

# Enough samples per POST to keep the request count sane (a full capture is
# ~28,000 records), small enough that one failure loses little progress.
DEFAULT_BURST = 200

# A whole session upload, over a rink's wifi.
TIMEOUT_S = 600


class PublishError(Exception):
    """The upload did not happen. The bundle on disk is untouched."""


@dataclass
class Result:
    bundle: Path
    output: str
    dry_run: bool


def check_bundle(path: Path | str) -> dict:
    """Read a bundle and confirm it is one, before opening a connection."""
    path = Path(path)
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"{path.name}: cannot read bundle ({exc})") from exc

    if bundle.get("schema") != "imu_telemetry_bundle":
        raise PublishError(f"{path.name} is not a telemetry bundle. "
                           "Process the capture first.")
    if not bundle.get("telemetry"):
        raise PublishError(f"{path.name} holds no telemetry samples.")
    return bundle


def publish(bundle_path: Path | str, session_id: str, *, dry_run: bool = False,
            burst_size: int = DEFAULT_BURST,
            keep_session_open: bool = False) -> Result:
    """Post connect -> telemetry -> playerStats -> general-stats."""
    bundle_path = Path(bundle_path)
    check_bundle(bundle_path)

    if not session_id.strip():
        raise PublishError("a server-side session id is required")

    cmd = [sys.executable, str(SENDER),
           "--session-id", session_id.strip(),
           "--bundle", str(bundle_path),
           "--burst-size", str(burst_size)]
    if dry_run:
        cmd.append("--dry-run")
    if keep_session_open:
        cmd.append("--keep-session-open")

    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise PublishError(
            f"the upload did not finish within {TIMEOUT_S} s; "
            "the bundle is unchanged, so retrying is safe") from exc
    except OSError as exc:
        raise PublishError(f"could not run the sender: {exc}") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise PublishError(detail[-1] if detail
                           else f"the sender exited with {proc.returncode}")

    return Result(bundle=bundle_path, output=proc.stdout.strip(),
                  dry_run=dry_run)
