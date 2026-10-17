"""Concert gain: one constant increase, as large as possible without clipping in any piece."""

from __future__ import annotations

import logging

import numpy as np

from . import media
from .plan import Piece, Plan

log = logging.getLogger(__name__)


def fade_curve(n: int, fade_in: int, fade_out: int) -> np.ndarray:
    """Linear fades, matching ffmpeg ``afade`` with ``curve=tri``."""
    g = np.ones(n, dtype=np.float32)
    if fade_in > 0:
        k = min(fade_in, n)
        g[:k] = np.linspace(0, 1, fade_in, endpoint=False, dtype=np.float32)[:k]
    if fade_out > 0:
        k = min(fade_out, n)
        g[n - k:] *= np.linspace(1, 0, fade_out, endpoint=False, dtype=np.float32)[fade_out - k:]
    return g


def peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20 * np.log10(p) if p > 0 else -np.inf


def measure_piece(plan: Plan, piece: Piece) -> float:
    """Sample peak (dBFS) of the recorder audio of ``piece``, after fades."""
    a = plan.audio[0]
    sr, ch = a.sample_rate, max(1, a.channels)
    z0 = plan.audio_time(piece, piece.start)
    x = media.decode_group(plan.audio, sr, z0, z0 + piece.duration, channels=ch)
    g = fade_curve(len(x), int(plan.fade_in(piece) * sr), int(plan.fade_out(piece) * sr))
    x = x * (g[:, None] if x.ndim == 2 else g)
    return peak_dbfs(x)


def update_gain(plan: Plan) -> None:
    """Measure every piece and set the one concert gain: the largest increase that does
    not clip during any piece (applause and talk between pieces may clip)."""
    peaks = []
    for p in plan.enabled_pieces():
        peak = measure_piece(plan, p)
        p.detected["peak_dbfs"] = round(peak, 2) if np.isfinite(peak) else None
        if np.isfinite(peak):
            peaks.append((peak, p))
    if not plan.settings.gain_db_auto and plan.settings.gain_db is not None:
        return
    if not peaks:
        plan.settings.gain_db = 0.0
        return
    loudest, piece = max(peaks, key=lambda x: x[0])
    plan.settings.gain_db = plan.settings.peak_dbfs - loudest
    plan.settings.gain_db_auto = True
    plan.settings.gain_limited_by = piece.basename
    log.info("concert gain %+.2f dB (limited by %s, peak %.2f dBFS)", plan.settings.gain_db,
             piece.basename, loudest)
