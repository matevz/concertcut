"""Synchronisation of the recorder (Zoom) audio to the camera timeline.

1. Coarse: onset-strength envelopes (100 Hz) of each camera file are cross-correlated
   with the envelope of the whole recorder stream. This places camera files relative to
   the recorder (and to each other, which also catches gaps between camera files).
2. Fine: GCC-PHAT on 16 kHz waveforms at anchors spread over the concert gives the
   offset with sub-millisecond resolution; a robust line fit over the anchors gives the
   offset at t = 0 and the clock drift between the two devices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import media
from .media import MediaFile

log = logging.getLogger(__name__)

ENV_SR = 8000
ENV_HOP = 80  # 100 Hz envelope
ENV_RATE = ENV_SR / ENV_HOP
FINE_SR = 16000


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def onset_envelope(x: np.ndarray, sr: int = ENV_SR, hop: int = ENV_HOP) -> np.ndarray:
    """Positive log-energy flux per hop, z-normalised."""
    if len(x) < hop * 4:
        return np.zeros(max(1, len(x) // hop), dtype=np.float32)
    sos = butter(4, [150, min(3500, sr / 2 - 100)], btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64))
    n = len(y) // hop
    e = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    flux = np.maximum(np.diff(20 * np.log10(e + 1e-6), prepend=0.0), 0.0)
    flux -= flux.mean()
    return (flux / (flux.std() + 1e-9)).astype(np.float32)


@dataclass
class Lag:
    lag: float  # in samples/frames of the input signals
    peak: float
    ratio: float  # best peak / best peak outside the exclusion zone

    @property
    def confident(self) -> bool:
        return self.ratio >= 1.4


def xcorr_lag(a: np.ndarray, b: np.ndarray, exclude: int = 100,
              min_lag: int | None = None, max_lag: int | None = None) -> Lag:
    """Lag ``k`` maximising sum_n a[n] * b[n + k] (i.e. ``a`` starts at b[k])."""
    n = _next_pow2(len(a) + len(b))
    r = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)
    lags = np.arange(n)
    lags[lags > n - len(a)] -= n  # negative lags wrap around
    valid = (lags > -len(a)) & (lags < len(b))
    if min_lag is not None:
        valid &= lags >= min_lag
    if max_lag is not None:
        valid &= lags <= max_lag
    rv = np.where(valid, r, -np.inf)
    i = int(np.argmax(rv))
    best = rv[i]
    far = np.abs(lags - lags[i]) > exclude
    second = np.max(np.where(far & valid, r, -np.inf)) if np.any(far & valid) else 0.0
    ratio = float(best / second) if second > 0 else float("inf")
    return Lag(float(lags[i]), float(best), ratio)


def gcc_phat(x: np.ndarray, y: np.ndarray, sr: int, max_shift: float) -> tuple[float, float]:
    """Delay d (s) such that y[n + d*sr] ~= x[n], searched in [0, 2*max_shift]; + sharpness."""
    n = _next_pow2(len(x) + len(y))
    X, Y = np.fft.rfft(x, n), np.fft.rfft(y, n)
    cross = np.conj(X) * Y
    freqs = np.fft.rfftfreq(n, 1 / sr)
    cross *= ((freqs > 120) & (freqs < 6000))  # ignore rumble and hiss
    r = np.fft.irfft(cross / (np.abs(cross) + 1e-12), n)
    kmax = int(2 * max_shift * sr)
    seg = r[: kmax + 1]
    k = int(np.argmax(seg))
    frac = 0.0
    if 0 < k < len(seg) - 1:  # parabolic interpolation
        a, b, c = seg[k - 1], seg[k], seg[k + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0
    sharp = float(seg[k] / (np.median(np.abs(seg)) + 1e-12))
    return (k + frac) / sr, sharp


def robust_line(t: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                tol: float | None = None) -> tuple[float, float, np.ndarray]:
    """Fit y = a + b t with the Theil-Sen estimator (median of pairwise slopes), which
    is insensitive to the occasional wrong measurement. Returns a, b, inlier mask.

    Inliers are within ``tol`` (s) of the line; by default 3 robust standard deviations
    of the residuals, at least 2 ms. ``w`` is accepted for API compatibility and ignored.
    """
    from scipy.stats import theilslopes

    mask = np.ones(len(t), bool)
    if len(t) == 1:
        return float(y[0]), 0.0, mask
    for _ in range(3):
        with np.errstate(invalid="ignore"):  # its confidence interval (unused) can be NaN
            b, a, _, _ = theilslopes(y[mask], t[mask])
        res = y - (a + b * t)
        lim = tol if tol is not None else max(0.002, 3 * 1.4826 * float(np.median(np.abs(res[mask]))))
        new = np.abs(res) < lim
        if new.sum() < 2 or np.array_equal(new, mask):
            break
        mask = new
    return float(a), float(b), mask


@dataclass
class SyncResult:
    audio_start: float  # timeline position of recorder time 0 (at t = 0, drift-free)
    drift: float  # seconds of offset change per second of timeline
    confidence: float
    anchors: list[dict]

    def offset_at(self, t: float) -> float:
        return self.audio_start + self.drift * t


def place_cameras(video: list[MediaFile], zoom_env: np.ndarray) -> tuple[list[float | None], list[Lag]]:
    """Coarse recorder time (s) at the start of each camera file, None if not found."""
    found, lags = [], []
    for f in video:
        x = media.decode(f.path, ENV_SR, 0.0, f.duration, 1, f.audio_index)
        env = onset_envelope(x)
        lag = xcorr_lag(env, zoom_env, exclude=int(ENV_RATE))
        lags.append(lag)
        ok = lag.confident and len(env) > ENV_RATE * 20
        found.append(lag.lag / ENV_RATE if ok else None)
        log.info("camera %s: recorder time at start %s (peak ratio %.2f)", f.name,
                 f"{lag.lag / ENV_RATE:.2f} s" if ok else "not found", lag.ratio)
    return found, lags


def apply_camera_placement(video: list[MediaFile], z: list[float | None],
                           snap: float = 0.5) -> float | None:
    """Re-place camera files on the timeline using the recorder as a common clock.

    Returns the coarse audio_start (timeline time of recorder time 0) or None.
    """
    meta = [f.start for f in video]
    for i in range(1, len(video)):
        prev, cur = video[i - 1], video[i]
        if z[i] is not None and z[i - 1] is not None:
            gap = (z[i] - z[i - 1]) - prev.duration
            cur.start = prev.end if abs(gap) < snap else prev.start + prev.duration + gap
            if abs(gap) >= snap:
                log.warning("gap of %.2f s between camera files %s and %s", gap, prev.name, cur.name)
        else:
            cur.start = prev.start + (meta[i] - meta[i - 1])
    shift = min(f.start for f in video)
    for f in video:
        f.start -= shift
    starts = [f.start - zi for f, zi in zip(video, z) if zi is not None]
    return float(np.median(starts)) if starts else None


def fine_sync(video: list[MediaFile], audio: list[MediaFile], coarse: float,
              times: list[float] | None = None, window: float = 30.0,
              max_shift: float = 0.3) -> SyncResult:
    span_v, span_a = media.group_span(video), media.group_span(audio)
    lo = max(window / 2, coarse + window / 2 + max_shift)
    hi = min(span_v, coarse + span_a) - window / 2 - max_shift
    if times is None:
        times = list(np.arange(lo, hi, 90.0)) if hi > lo else []
        # make sure every run of camera files gets anchors of its own
        for files in _blocks(video):
            b0, b1 = max(lo, files[0].start + window / 2), min(hi, files[-1].end - window / 2)
            if b1 >= b0:
                times += list(np.linspace(b0, b1, 3 if b1 - b0 > window else 1))
        times.sort()
    anchors = []
    for t in times:
        if not lo <= t <= hi:
            continue
        x = media.decode_group(video, FINE_SR, t - window / 2, t + window / 2)
        z0 = t - coarse - window / 2 - max_shift
        y = media.decode_group(audio, FINE_SR, z0, z0 + window + 2 * max_shift)
        if np.sqrt(np.mean(x ** 2)) < 1e-4 or np.sqrt(np.mean(y ** 2)) < 1e-4:
            continue
        d, sharp = gcc_phat(x, y, FINE_SR, max_shift)
        anchors.append({"t": float(t), "offset": float(coarse + max_shift - d), "sharpness": float(sharp)})
    good = [a for a in anchors if a["sharpness"] > 5]
    if not good:
        log.warning("fine sync failed, using coarse offset only")
        return SyncResult(coarse, 0.0, 0.0, anchors)
    t = np.array([a["t"] for a in good])
    y = np.array([a["offset"] for a in good])
    w = np.array([min(a["sharpness"], 50.0) for a in good])
    # Camera files after a gap were placed from the coarse envelope only; fit the line on
    # the largest run of camera files, then move every other run onto that line.
    blocks = _blocks(video)
    which = np.array([_block_of(blocks, x) for x in t])
    main = int(np.bincount(which[which >= 0]).argmax()) if np.any(which >= 0) else -1
    sel_main = which == main
    a0, b, _ = robust_line(t[sel_main], y[sel_main], w[sel_main]) if sel_main.sum() else robust_line(t, y, w)
    for k, files in enumerate(blocks):
        sel = which == k
        if k == main or not sel.any():
            continue
        resid = float(np.median(y[sel] - (a0 + b * t[sel])))
        if abs(resid) > 0.0005:
            log.info("moving camera files %s by %.1f ms", ", ".join(f.name for f in files), -resid * 1000)
            for f in files:
                f.start -= resid
            y[sel] -= resid
            for a, s_ in zip(good, sel):
                if s_:
                    a["offset"] -= resid
    a0, b, mask = robust_line(t, y, w)
    for a, m in zip(good, mask):
        a["inlier"] = bool(m)
    conf = float(mask.sum() / max(1, len(anchors)))
    log.info("sync: audio_start %.4f s, drift %.2f ppm, %d/%d anchors agree", a0, b * 1e6,
             mask.sum(), len(anchors))
    return SyncResult(a0, b, conf, anchors)


def _blocks(video: list[MediaFile]) -> list[list[MediaFile]]:
    """Runs of camera files that follow each other without a gap."""
    out: list[list[MediaFile]] = []
    for f in sorted(video, key=lambda f: f.start):
        if out and abs(f.start - out[-1][-1].end) < 1e-3:
            out[-1].append(f)
        else:
            out.append([f])
    return out


def _block_of(blocks: list[list[MediaFile]], t: float) -> int:
    for k, files in enumerate(blocks):
        if files[0].start <= t < files[-1].end:
            return k
    return -1


def synchronise(video: list[MediaFile], audio: list[MediaFile]) -> SyncResult:
    span_a = media.group_span(audio)
    log.info("decoding recorder audio for coarse sync (%.0f min)", span_a / 60)
    zoom = media.decode_group(audio, ENV_SR, 0.0, span_a)
    zoom_env = onset_envelope(zoom)
    del zoom
    meta_contiguous = [i > 0 and abs(f.start - video[i - 1].end) < 1e-6 for i, f in enumerate(video)]
    z, _ = place_cameras(video, zoom_env)
    # Drop camera files that are not in the recorder audio and not continuous with a file
    # that is (e.g. a short test clip recorded hours before the concert).
    keep = [zi is not None for zi in z]
    changed = True
    while changed:
        changed = False
        for i in range(len(video)):
            if keep[i]:
                continue
            if (i > 0 and keep[i - 1] and meta_contiguous[i]) or \
                    (i + 1 < len(video) and keep[i + 1] and meta_contiguous[i + 1]):
                keep[i] = changed = True
    for f, k in zip(video, keep):
        if not k:
            log.warning("ignoring camera file %s: not found in the recorder audio", f.name)
    video[:] = [f for f, k in zip(video, keep) if k]
    z = [zi for zi, k in zip(z, keep) if k]
    if not video:
        raise RuntimeError("could not find the recorder audio in any camera file - are the camera "
                           "and recorder files from the same concert?")
    coarse = apply_camera_placement(video, z)
    if coarse is None:
        raise RuntimeError("could not find the recorder audio in any camera file")
    return fine_sync(video, audio, coarse)


def piece_offsets(video: list[MediaFile], audio: list[MediaFile], res: SyncResult,
                  ranges: list[tuple[float, float]], window: float = 30.0, max_shift: float = 0.3,
                  max_dev: float = 0.03) -> list[dict]:
    """Offset (timeline position of recorder time 0) measured inside each piece.

    Several GCC-PHAT anchors per piece; the median of those within ``max_dev`` of the
    global line is used. Falls back to the line when a piece has no usable anchor.
    """
    span_v, span_a = media.group_span(video), media.group_span(audio)
    out = []
    for t0, t1 in ranges:
        w = min(window, max(4.0, t1 - t0))
        n = max(1, int((t1 - t0) // w))
        centers = [t0 + (t1 - t0) * (k + 0.5) / n for k in range(n)]
        vals = []
        for t in centers:
            base = res.offset_at(t)
            z0 = t - base - w / 2 - max_shift
            if t - w / 2 < 0 or t + w / 2 > span_v or z0 < 0 or z0 + w + 2 * max_shift > span_a:
                continue
            x = media.decode_group(video, FINE_SR, t - w / 2, t + w / 2)
            y = media.decode_group(audio, FINE_SR, z0, z0 + w + 2 * max_shift)
            if np.sqrt(np.mean(x ** 2)) < 1e-4 or np.sqrt(np.mean(y ** 2)) < 1e-4:
                continue
            d, sharp = gcc_phat(x, y, FINE_SR, max_shift)
            off = base + max_shift - d
            if sharp > 5 and abs(off - base) < max_dev:
                vals.append(off)
        mid = (t0 + t1) / 2
        if vals:
            out.append({"offset": float(np.median(vals)), "anchors": len(vals),
                        "spread_ms": round(float(np.std(vals)) * 1000, 1)})
        else:
            out.append({"offset": res.offset_at(mid), "anchors": 0, "spread_ms": None})
    return out
