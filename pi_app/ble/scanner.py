"""Finding Thingy:53 recorders.

Filters on the service UUID rather than the name: a board renamed with SET_ID
advertises as "DEV-12", so matching on a fixed name would hide exactly the
devices that have been set up properly. The UUID lives in the scan response,
which an active scan (bleak's default) collects along with the name.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from bleak import BleakScanner

from pi_app.ble import protocol as proto

SCAN_TIMEOUT = 6.0


@dataclass
class Found:
    """One nearby recorder, as the device list shows it."""

    name: str
    address: str
    rssi: int | None = None

    @property
    def label(self) -> str:
        return self.name or self.address

    @property
    def signal(self) -> str:
        return f"{self.rssi} dBm" if self.rssi is not None else "-"


def _is_ours(adv) -> bool:
    uuids = [u.lower() for u in (adv.service_uuids or [])]
    return proto.SERVICE_UUID.lower() in uuids


async def scan(timeout: float = SCAN_TIMEOUT) -> list[Found]:
    """Nearby recorders, strongest signal first."""
    seen = await BleakScanner.discover(timeout=timeout, return_adv=True)

    out = []
    for device, adv in seen.values():
        if not _is_ours(adv):
            continue
        out.append(Found(name=adv.local_name or device.name or "",
                         address=device.address,
                         rssi=getattr(adv, "rssi", None)))

    out.sort(key=lambda f: (f.rssi if f.rssi is not None else -999), reverse=True)
    return out


async def find_by_address(address: str, timeout: float = SCAN_TIMEOUT):
    return await BleakScanner.find_device_by_address(address, timeout=timeout)


def scan_blocking(timeout: float = SCAN_TIMEOUT) -> list[Found]:
    """For scripts and tests, where there is no event loop to join."""
    return asyncio.run(scan(timeout))
