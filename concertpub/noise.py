"""Detection and removal of a constant mains hum / buzz (e.g. an amplifier near the stage).

The buzz shows up as narrow peaks at multiples of the mains frequency (50 Hz in Europe,
60 Hz elsewhere). It is measured in the quietest stretches of the recording, where the
music does not mask it, and removed with very narrow notch filters at exactly those
frequencies, which leaves the rest of the spectrum untouched.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import welch

from . import media

log = logging.getLogger(__name__)

SR = 16000  # enough for harmonics up to FMAX
FMAX = 4000.0
MIN_PROMINENCE_DB = 5.0  # a harmonic is removed when it stands this much above its neighbours
NOTABLE_DB = 12.0  # a buzz is reported when harmonics stand this much above


def quiet_windows(energy_db: np.ndarray, fps: int = 100, win_s: float = 2.0, n: int = 20,
                  skip_s: float = 30.0) -> list[float]:
    """Start times (s) of the n quietest non-overlapping windows of the recording."""
    win = int(win_s * fps)
    if len(energy_db) < win * 2:
        return []
    mean = np.convolve(energy_db, np.ones(win) / win, mode="valid")
    lo, hi = int(skip_s * fps), len(mean) - int(skip_s * fps)
    picks: list[int] = []
    for i in np.argsort(mean):
        if lo <= i < hi and all(abs(i - j) >= win for j in picks):
            picks.append(int(i))
            if len(picks) == n:
                break
    return sorted(i / fps for i in picks)


def spectrum(audio: list[media.MediaFile], starts: list[float], win_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Average power spectrum (frequencies, dB) of the given recording windows, 0.5 Hz resolution."""
    x = np.concatenate([media.decode_group(audio, SR, t, t + win_s) for t in starts])
    f, p = welch(x, fs=SR, nperseg=int(SR * 2))
    return f, 10 * np.log10(p + 1e-20)


def _prominence(freqs: np.ndarray, psd_db: np.ndarray, centre: float,
                window: float = 1.5) -> tuple[float, float]:
    """(peak frequency, dB above the neighbouring frequencies) of a narrow peak near ``centre``."""
    near = (freqs > centre - window) & (freqs < centre + window)
    side = ((freqs > centre - 15) & (freqs < centre - 4)) | ((freqs > centre + 4) & (freqs < centre + 15))
    if not near.any() or not side.any():
        return centre, 0.0
    i = int(np.argmax(np.where(near, psd_db, -np.inf)))
    peak = freqs[i]
    if 0 < i < len(psd_db) - 1:  # sub-bin position of the peak (parabolic interpolation)
        a, b, c = psd_db[i - 1], psd_db[i], psd_db[i + 1]
        den = a - 2 * b + c
        if den < 0:
            peak = freqs[i] + 0.5 * (a - c) / den * (freqs[1] - freqs[0])
    return float(peak), float(psd_db[i] - np.median(psd_db[side]))


def find_hum(freqs: np.ndarray, psd_db: np.ndarray) -> dict | None:
    """The mains hum in a spectrum: {"mains_hz", "harmonics": [{"hz", "db"}...], "strongest_db"}.

    The exact mains frequency (it drifts a little around 50/60 Hz) is fitted from all clear
    harmonics, and every harmonic is then placed at an exact multiple of it.
    """
    best = None
    for mains in (50.0, 60.0):
        ks = np.arange(1, int(FMAX // mains) + 1)
        found = [(k, *_prominence(freqs, psd_db, k * mains)) for k in ks]
        clear = [(k, f) for k, f, db in found if db >= NOTABLE_DB]
        if not clear:
            continue
        kk = np.array([k for k, _ in clear], float)
        ff = np.array([f for _, f in clear], float)
        f0 = float((kk * ff).sum() / (kk * kk).sum())  # least squares: f_k = k * f0
        harmonics = []
        for k in ks:
            _, db = _prominence(freqs, psd_db, k * f0)
            if db >= MIN_PROMINENCE_DB:
                harmonics.append({"hz": round(float(k * f0), 3), "db": round(db, 1)})
        if len(harmonics) >= 2 and (best is None or len(harmonics) > len(best["harmonics"])):
            best = {"mains_hz": mains, "fundamental_hz": round(f0, 4), "harmonics": harmonics,
                    "strongest_db": max(h["db"] for h in harmonics)}
    return best


def detect(audio: list[media.MediaFile], energy_db: np.ndarray, fps: int = 100) -> dict:
    """Noise info for the config: {"hum": {...} or None, "measured_on_s": [...]}."""
    starts = quiet_windows(energy_db, fps)
    if not starts:
        return {"hum": None, "measured_on_s": []}
    f, db = spectrum(audio, starts)
    hum = find_hum(f, db)
    if hum:  # how far the mains frequency wanders: measured in each silence separately
        each = [fit_fundamental(*spectrum(audio, [t]), hum) for t in starts]
        each = [v for v in each if v is not None]
        if len(each) >= 3:
            lo, hi = np.percentile(each, [10, 90])
            hum["drift_hz"] = round(float((hi - lo) / 2), 4)
    if hum:
        log.info("constant buzz found: %.0f Hz mains hum, %d harmonics up to %.0f Hz (up to %.0f dB above "
                 "the background)", hum["mains_hz"], len(hum["harmonics"]), hum["harmonics"][-1]["hz"],
                 hum["strongest_db"])
    return {"hum": hum, "measured_on_s": [round(t, 1) for t in starts]}


FIT_DB = 8.0  # harmonics used to measure the exact mains frequency in a stretch


def fit_fundamental(freqs: np.ndarray, psd_db: np.ndarray, hum: dict) -> float | None:
    """The exact mains frequency in this stretch (it drifts during a concert); None if the
    buzz's harmonics cannot be measured reliably here (e.g. under loud music).

    Two steps: a rough fit from the low harmonics, then a refinement with all harmonics,
    each searched only very close to where it must be - so the singers' own overtones,
    which can lie next to the buzz harmonics, are not mistaken for them.
    """
    f0 = hum.get("fundamental_hz", hum["mains_hz"])
    ks = sorted({round(h["hz"] / hum.get("fundamental_hz", hum["mains_hz"])) for h in hum["harmonics"]})
    for kmax, window, need in ((20, 1.5, 3), (None, 0.6, 4)):
        used_k, used_f = [], []
        for k in ks:
            if kmax and k > kmax:
                break
            peak, db = _prominence(freqs, psd_db, k * f0, window)
            if db >= FIT_DB:
                used_k.append(k)
                used_f.append(peak)
        if len(used_k) < need:
            return None
        kk, ff = np.array(used_k, float), np.array(used_f, float)
        f0 = float((kk * ff).sum() / (kk * kk).sum())
    return f0


def harmonic_numbers(hum: dict) -> list[int]:
    """Every harmonic up to the highest one found: weak ones become audible in quiet passages."""
    f0 = hum.get("fundamental_hz", hum["mains_hz"])
    return list(range(1, max(round(h["hz"] / f0) for h in hum["harmonics"]) + 1))


def _notches(hum: dict, sr: int) -> np.ndarray:
    """Second-order sections of the notch filters: one per harmonic, at exact multiples of the
    mains frequency measured in the silences, wide enough to cover how far it drifted there."""
    from scipy.signal import iirnotch, tf2sos

    f0 = hum.get("fundamental_hz", hum["mains_hz"])
    drift = hum.get("drift_hz", 0.0)
    sections = []
    for k in harmonic_numbers(hum):
        f = k * f0
        if f >= sr / 2 - 100:
            break
        width = max(1.5, f * 0.002, 4 * k * drift + 0.5)  # Hz; one (forwards) pass, so a bit wider
        b, a = iirnotch(f, f / width, fs=sr)
        sections.append(tf2sos(b, a))
    return np.vstack(sections) if sections else np.zeros((0, 6))


def dehum(x: np.ndarray, sr: int, hum: dict, workers: int | None = None, chunk_s: float = 30.0,
          margin_s: float = 2.0, fade_s: float = 0.05) -> np.ndarray:
    """Remove the mains hum from ``x`` (samples, or samples x channels) with one fixed set of
    notch filters (see :func:`_notches`), the same for the whole piece and every piece.

    The filter runs forwards only. Filtering forwards and backwards would avoid any phase
    shift, but then each narrow notch "rings" *before* sudden sounds too: a note starting
    next to a notch frequency (e.g. a sung 498 Hz next to the 500 Hz notch) would put a
    short buzz at exactly the mains frequencies just before the note. Forwards only, a little
    ringing can follow an onset, where the note itself masks it.

    Only to use all CPU threads, the audio is filtered in ``chunk_s`` chunks, each starting
    ``margin_s`` earlier so the filters have settled, joined with a short crossfade; since
    every chunk gets the identical filter, the result equals filtering the piece in one go.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    from scipy.signal import sosfilt

    if not hum or len(x) == 0:
        return x
    sos = _notches(hum, sr)
    if not len(sos):
        return x
    x2 = x.reshape(len(x), -1).astype(np.float64)
    n, chunk, margin, fade = len(x2), int(chunk_s * sr), int(margin_s * sr), int(fade_s * sr)
    hop = chunk - fade
    starts = list(range(0, n, hop)) if n > chunk else [0]

    def filtered(s0):  # the filter releases Python's lock, so chunks run truly in parallel
        s1 = min(n, s0 + chunk)
        a = max(0, s0 - margin)
        return sosfilt(sos, x2[a:s1], axis=0)[s0 - a:]

    with ThreadPoolExecutor(max_workers=max(1, workers or os.cpu_count() or 1)) as ex:
        parts = list(ex.map(filtered, starts))
    out = np.zeros_like(x2)
    for i, (s0, y) in enumerate(zip(starts, parts)):
        s1 = s0 + len(y)
        w = np.ones(len(y))
        if i > 0:
            k = min(fade, len(w))
            w[:k] = np.linspace(0.0, 1.0, k, endpoint=False)
        if s1 < n:
            k = min(fade, len(w))
            w[len(w) - k:] *= np.linspace(1.0, 0.0, k, endpoint=False)
        out[s0:s1] += y * w[:, None]
    return out.reshape(x.shape).astype(np.float32)


def describe(hum: dict) -> str:
    hs = hum["harmonics"]
    drift = f", drifting +-{hum['drift_hz']:.3f} Hz" if hum.get("drift_hz") else ""
    return (f"{hum['mains_hz']:.0f} Hz mains hum ({hum.get('fundamental_hz', hum['mains_hz']):.2f} Hz{drift}) with "
            f"{len(hs)} harmonic(s) up to {hs[-1]['hz']:.0f} Hz, up to {hum['strongest_db']:.0f} dB above the background")


def state(hum: dict | None, removed: bool) -> str:
    """Whether a buzz was found and is removed, for the lines that show the concert gain."""
    if not hum:
        return "no buzz found"
    return "buzz removed" if removed else "buzz not removed"
