"""Conversion between seconds and ``HH:MM:SS.mmm`` timecodes."""

from __future__ import annotations

import re

_TC = re.compile(r"^\s*(-)?(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*$")


def to_seconds(value: str | float | int | None) -> float | None:
    """Parse ``HH:MM:SS.mmm``, ``MM:SS.mmm`` or a plain number of seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _TC.match(value)
    if not m:
        return float(value)
    sign, h, mnt, s = m.groups()
    secs = int(h or 0) * 3600 + int(mnt) * 60 + float(s)
    return -secs if sign else secs


def to_tc(seconds: float | None) -> str | None:
    """Format seconds as ``HH:MM:SS.mmm``."""
    if seconds is None:
        return None
    sign = "-" if seconds < 0 else ""
    ms = round(abs(seconds) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
