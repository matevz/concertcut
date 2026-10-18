"""Concert gain: one constant increase, as large as possible without clipping in any piece."""

from __future__ import annotations

import logging

import numpy as np

from . import media
from .config import Config, Piece, flac_rate

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


def peak_of(audio: list[media.MediaFile], z0: float, duration: float, rate: int | None = None) -> float:
    """Sample peak (dBFS) of the recording from recorder time z0, at the FLAC sample rate
    (resampling can raise peaks a little, so they are measured after it)."""
    a = audio[0]
    sr = rate or flac_rate(audio)
    x = media.decode_group(audio, sr, z0, z0 + duration, channels=max(1, a.channels), hq=sr != a.sample_rate)
    return peak_dbfs(x)


def measure_piece(config: Config, piece: Piece) -> float:
    """Sample peak (dBFS) of the recorder audio of ``piece``, after fades."""
    a = config.audio[0]
    sr, ch = config.flac_rate, max(1, a.channels)
    z0 = config.audio_time(piece, piece.start)
    x = media.decode_group(config.audio, sr, z0, z0 + piece.duration, channels=ch, hq=sr != a.sample_rate)
    g = fade_curve(len(x), int(config.fade_in(piece) * sr), int(config.fade_out(piece) * sr))
    x = x * (g[:, None] if x.ndim == 2 else g)
    return peak_dbfs(x)


def update_gain(config: Config, known: dict[int, float] | None = None) -> None:
    """Measure every piece and set the one concert gain: the largest increase that does
    not clip during any piece (applause and talk between pieces may clip).

    ``known``: peaks already measured, by piece number (not measured again).
    """
    peaks = []
    for p in config.enabled_pieces():
        peak = known[p.number] if known and p.number in known and not (config.settings.fade_in_s or
                                                                         config.settings.fade_out_s) \
            else measure_piece(config, p)
        p.detected["peak_dbfs"] = round(peak, 2) if np.isfinite(peak) else None
        if np.isfinite(peak):
            peaks.append((peak, p))
    if not config.settings.gain_db_auto and config.settings.gain_db is not None:
        return
    if not peaks:
        config.settings.gain_db = 0.0
        return
    loudest, piece = max(peaks, key=lambda x: x[0])
    config.settings.gain_db = config.settings.peak_dbfs - loudest
    config.settings.gain_db_auto = True
    config.settings.gain_limited_by = piece.basename
    log.info("concert gain %+.2f dB (limited by %s, peak %.2f dBFS)", config.settings.gain_db,
             piece.basename, loudest)
