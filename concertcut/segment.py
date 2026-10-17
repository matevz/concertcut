"""Turning frame-wise class scores into piece boundaries (pure numpy, no I/O).

Works on recorder stream time. Steps:
  label frames (music / applause / speech / other) -> clean short runs ->
  find pitch cues (the short note a choir gets before each piece) ->
  group music runs into pieces at applause, speech and pitch cues (silence alone,
  e.g. rests or quiet passages, never splits a piece) -> drop short regions ->
  pieces after the final ovation are encores -> match the number of programme items ->
  refine edges on the energy envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

MUSIC, APPLAUSE, SPEECH, OTHER = 0, 1, 2, 3
LABEL_NAMES = ["music", "applause", "speech", "other"]
RATE = 10  # label frames per second


@dataclass
class SegmentParams:
    music_min: float = 0.2
    applause_min: float = 0.2
    speech_min: float = 0.3
    smooth_s: float = 1.1
    min_run_s: float = 1.0
    hard_applause_s: float = 1.5  # applause in a gap that separates pieces
    hard_speech_s: float = 2.5
    # pitch cue: short, clearly "music" sound followed by silence before the piece starts
    cue_min_s: float = 1.5
    cue_max_s: float = 10.0
    cue_score: float = 0.4
    cue_gap_s: float = 4.0
    min_piece_s: float = 30.0
    split_gap_s: float = 2.5  # smallest silent gap used when more pieces are needed
    ovation_s: float = 20.0  # applause this long after a piece ends the programme
    lead_in_s: float = 3.0  # quiet hall kept before the first note (shorter if not quiet)
    applause_tail_s: float = 1.5  # applause still audible after its label ends
    speech_tail_s: float = 0.3
    quiet_db: float = 8.0  # "quiet" = at most this much above the hall's noise floor
    tail_db: float = 3.0
    tail_margin_s: float = 0.5
    max_tail_s: float = 6.0
    separator_search_s: float = 60.0  # how far after the music to look for applause


@dataclass
class Region:
    start: float
    end: float
    confidence: float = 0.0
    start_reason: str = ""
    end_reason: str = ""
    encore: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def downsample(scores: np.ndarray, src_fps: int, dst_fps: int = RATE) -> np.ndarray:
    f = src_fps // dst_fps
    n = len(scores) // f
    return scores[: n * f].reshape(n, f, -1).mean(axis=1)


def label_frames(scores: np.ndarray, p: SegmentParams) -> np.ndarray:
    """scores: (frames @ RATE, 3) columns music, applause, speech."""
    k = max(1, int(round(p.smooth_s * RATE)) | 1)
    s = median_filter(scores, size=(k, 1), mode="nearest")
    m, a, sp = s[:, 0], s[:, 1], s[:, 2]
    lab = np.full(len(s), OTHER, dtype=np.int8)
    lab[(sp >= p.speech_min) & (sp > m)] = SPEECH
    lab[(a >= p.applause_min) & (a > m) & (a >= sp)] = APPLAUSE
    lab[(m >= p.music_min) & (m >= a) & (m >= sp)] = MUSIC
    return lab


def runs(lab: np.ndarray) -> list[list[int]]:
    """[[label, start, end), ...]"""
    out = []
    if len(lab) == 0:
        return out
    edges = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(lab)]
    return [[int(lab[s]), int(s), int(e)] for s, e in zip(starts, ends)]


def clean_runs(lab: np.ndarray, min_len: int) -> np.ndarray:
    """Absorb runs shorter than ``min_len`` frames into their longer neighbour."""
    lab = lab.copy()
    while True:
        r = runs(lab)
        if len(r) <= 1:
            return lab
        lens = [e - s for _, s, e in r]
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            return lab
        left = r[i - 1] if i > 0 else None
        right = r[i + 1] if i + 1 < len(r) else None
        nb = max((x for x in (left, right) if x), key=lambda x: x[2] - x[1])
        lab[r[i][1]:r[i][2]] = nb[0]


def find_cues(lab: np.ndarray, music: np.ndarray, p: SegmentParams) -> list[tuple[int, int, int]]:
    """Pitch cues: the note a choir is given before a piece.

    A short, confident music run, quiet before it and clearly quieter after it, followed
    within 40 s (and without applause or speech) by the piece. The singers may hum the
    note back first: music runs as confident as the cue itself are still part of the cue.
    Returns (cue start, end of the cue tone, start of the piece) frames.
    """
    mus = [(s, e) for l, s, e in runs(lab) if l == MUSIC]
    pre_n, post_n = int(3 * RATE), int(1.5 * p.cue_gap_s * RATE)
    cues = []
    for i, (s, e) in enumerate(mus):
        dur = (e - s) / RATE
        level = float(music[s:e].mean())
        if not p.cue_min_s <= dur <= p.cue_max_s or level < p.cue_score:
            continue
        if s < pre_n or e + post_n > len(lab):
            continue
        if music[s - pre_n:s].mean() > 0.2 or music[e:e + post_n].mean() > 0.5 * level:
            continue
        piece = next(((s2, e2) for s2, e2 in mus[i + 1:]
                      if music[s2:e2].mean() < p.cue_score or (e2 - s2) / RATE >= 5), None)
        if piece is None or (piece[0] - e) / RATE > 40:
            continue
        gap = lab[e:piece[0]]
        if (gap == APPLAUSE).sum() / RATE >= p.hard_applause_s or (gap == SPEECH).sum() / RATE >= p.hard_speech_s:
            continue  # a cue is followed by silence, not by applause or talking
        cues.append((s, e, piece[0]))
    return cues


@dataclass
class Boundary:
    """What lies between two consecutive pieces (or music runs)."""
    start: int
    end: int
    applause: float
    speech: float
    cue: bool

    @property
    def dur(self) -> float:
        return (self.end - self.start) / RATE

    @property
    def hard(self) -> bool:
        return self.cue or self.applause > 0 or self.speech > 0

    def strength(self, p: SegmentParams) -> float:
        return (2 * min(self.applause, 30) + min(self.speech, 30) + (5 if self.cue else 0)
                + 0.01 * self.dur)


def _boundary(lab: np.ndarray, a: int, b: int, cues: list[tuple[int, int, int]], p: SegmentParams) -> Boundary:
    g = lab[a:b]
    app = float((g == APPLAUSE).sum()) / RATE
    sp = float((g == SPEECH).sum()) / RATE
    return Boundary(a, b, app if app >= p.hard_applause_s else 0.0, sp if sp >= p.hard_speech_s else 0.0,
                    any(a <= cs and ce <= b + 1 for cs, _, ce in cues))


@dataclass
class _Piece:
    a: int
    b: int
    soft: list[Boundary]  # silent gaps inside the piece


def find_regions(lab: np.ndarray, music: np.ndarray, p: SegmentParams,
                 expected: int | None = None
                 ) -> tuple[list[tuple[int, int]], list[str], list[bool], list[tuple[int, int, int] | None]]:
    """Music regions (frame ranges at RATE).

    Returns regions, warnings, encore flags and the pitch cue preceding each region.
    """
    warnings: list[str] = []
    cues = find_cues(lab, music, p)
    lab2 = lab.copy()
    for s, _, e in cues:
        lab2[s:e] = OTHER
    mus = [(s, e) for l, s, e in runs(lab2) if l == MUSIC]
    if not mus:
        return [], ["no music detected"], [], []

    pieces = [_Piece(mus[0][0], mus[0][1], [])]
    for (_, e0), (s1, e1) in zip(mus, mus[1:]):
        bd = _boundary(lab2, e0, s1, cues, p)
        if bd.hard:
            pieces.append(_Piece(s1, e1, []))
        else:
            pieces[-1].b = e1
            pieces[-1].soft.append(bd)

    def music_len(a: int, b: int) -> float:
        return float((lab2[a:b] == MUSIC).sum()) / RATE

    # A short region separated from its neighbour only by a "cue" (no applause/speech)
    # was a loud chord, not a pitch cue: merge it back instead of dropping it.
    k = 0
    while k < len(pieces):
        pc = pieces[k]
        if music_len(pc.a, pc.b) >= p.min_piece_s:
            k += 1
            continue
        prev_ok = k > 0 and not _boundary(lab2, pieces[k - 1].b, pc.a, [], p).hard
        next_ok = k + 1 < len(pieces) and not _boundary(lab2, pc.b, pieces[k + 1].a, [], p).hard
        if prev_ok:
            q = pieces[k - 1]
            q.soft.append(_boundary(lab2, q.b, pc.a, [], p))
            q.b, q.soft = pc.b, q.soft + pc.soft
            del pieces[k]
        elif next_ok:
            q = pieces[k + 1]
            q.soft = pc.soft + [_boundary(lab2, pc.b, q.a, [], p)] + q.soft
            q.a = pc.a
            del pieces[k]
        else:
            k += 1
    kept = [pc for pc in pieces if music_len(pc.a, pc.b) >= p.min_piece_s]
    if len(kept) < len(pieces):
        warnings.append(f"ignored {len(pieces) - len(kept)} short music region(s) under {p.min_piece_s:.0f} s")
    pieces = kept
    if not pieces:
        return [], warnings + ["no music region long enough"], [], []

    def boundaries() -> list[Boundary]:
        return [_boundary(lab2, x.b, y.a, cues, p) for x, y in zip(pieces, pieces[1:])]

    # final ovation: pieces after it are encores
    n_main = len(pieces)
    bds = boundaries()
    if len(bds) >= 2:
        j = int(np.argmax([b.applause for b in bds]))
        others = [b.applause for k, b in enumerate(bds) if k != j]
        if bds[j].applause >= p.ovation_s and bds[j].applause >= 1.5 * max(np.median(others), 1.0) \
                and j + 1 >= len(pieces) / 2:
            cand = j + 1
            if not expected or abs(cand - expected) <= abs(len(pieces) - expected):
                n_main = cand

    main, encores = pieces[:n_main], pieces[n_main:]
    if expected:
        merged = 0
        while len(main) > expected:
            # nobody applauds in the middle of a piece: only merge across cues / speech
            bd = [_boundary(lab2, x.b, y.a, cues, p) for x, y in zip(main, main[1:])]
            cand = [k for k, b in enumerate(bd) if b.applause == 0]
            if not cand:
                break
            k = min(cand, key=lambda k: bd[k].strength(p))
            main[k:k + 2] = [_Piece(main[k].a, main[k + 1].b, main[k].soft + [bd[k]] + main[k + 1].soft)]
            merged += 1
        if merged:
            warnings.append(f"merged {merged} region(s) at the weakest boundaries to match the programme")
        split = 0
        while len(main) < expected:
            best = None
            for k, pc in enumerate(main):
                for g in pc.soft:
                    if g.dur >= p.split_gap_s and music_len(pc.a, g.start) >= p.min_piece_s \
                            and music_len(g.end, pc.b) >= p.min_piece_s and (best is None or g.dur > best[1].dur):
                        best = (k, g)
            if best is None:
                break
            k, g = best
            pc = main[k]
            main[k:k + 1] = [_Piece(pc.a, g.start, [x for x in pc.soft if x.end <= g.start]),
                             _Piece(g.end, pc.b, [x for x in pc.soft if x.start >= g.end])]
            split += 1
        if split:
            warnings.append(f"split {split} region(s) at silent gaps to match the programme")
        if len(main) > expected:
            warnings.append(f"found {len(main)} pieces but the programme lists {expected}; the last "
                            f"{len(main) - expected} are named 'Encore' - check the order")
        if len(main) < expected:
            warnings.append(f"found {len(main)} pieces but the programme lists {expected}; names may be "
                            "shifted - check")
    if encores:
        warnings.append(f"{len(encores)} piece(s) after the final applause treated as encore(s)")

    allp = main + encores
    cue_before = []
    for pc in allp:
        prev = [c for c in cues if c[2] <= pc.a + 1 and (pc.a - c[2]) / RATE <= 30]
        cue_before.append(prev[-1] if prev else None)
    return ([(pc.a, pc.b) for pc in allp], warnings, [False] * len(main) + [True] * len(encores),
            cue_before)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    return uniform_filter1d(x, size=max(1, n), mode="nearest")


def refine(regions: list[tuple[int, int]], lab: np.ndarray, music: np.ndarray,
           energy_db: np.ndarray, hf_db: np.ndarray, env_fps: int, total: float,
           p: SegmentParams, cues: list[tuple[int, int, int] | None] | None = None) -> list[Region]:
    """Exact edges in seconds using the energy envelopes (both at ``env_fps``)."""
    e = _smooth(energy_db, int(0.05 * env_fps))
    hf = _smooth(hf_db, int(0.03 * env_fps))
    n_env = len(e)

    def idx(t: float) -> int:
        return int(np.clip(round(t * env_fps), 0, n_env - 1))

    out: list[Region] = []
    for k, (a, b) in enumerate(regions):
        s0, e0 = a / RATE, b / RATE
        r = Region(start=s0, end=e0)
        cue = cues[k] if cues else None
        # Hard limit: nothing of the pitch tone, the previous piece or far before.
        limit = max(0.0, s0 - 30.0)
        if cue:
            limit = max(limit, cue[1] / RATE)
        if out:
            limit = max(limit, out[-1].end)
        # applause/speech fades out after the detector stops labelling it: keep clear
        b0 = max(0, int(limit * RATE))
        back = lab[b0:a]
        for kind, tail in ((APPLAUSE, p.applause_tail_s), (SPEECH, p.speech_tail_s)):
            hits = np.flatnonzero(back == kind)
            if len(hits):
                limit = max(limit, (b0 + hits[-1] + 1) / RATE + tail)
        lo = idx(limit)
        floor = np.percentile(e[lo:idx(s0)], 10) if idx(s0) - lo > env_fps else np.percentile(e, 5)
        thr = floor + p.quiet_db

        def blocked(i: int) -> bool:
            return lab[min(len(lab) - 1, int(i / env_fps * RATE))] in (APPLAUSE, SPEECH)

        # --- first note: soft openings are often not scored as music yet, so walk back
        # from the detected start through sound until a real pause (>= 0.75 s at the
        # hall's noise floor), applause/speech, or the limit.
        i = idx(s0)
        while i < min(n_env - 1, idx(s0 + 1.0)) and e[i] <= thr:
            i += 1  # first audible frame at the detected start
        onset_i, quiet = i, 0
        while i > lo:
            i -= 1
            if blocked(i):
                break
            if e[i] <= thr:
                quiet += 1
                if quiet >= 0.75 * env_fps:
                    break
            else:
                quiet, onset_i = 0, i
        onset = onset_i / env_fps
        # --- lead-in: keep up to lead_in_s of quiet hall before the first note; stop at
        # anything audible (pitch tone still ringing, noise), applause or speech.
        start_i = onset_i
        j = onset_i - 1
        while j >= max(lo, idx(onset - p.lead_in_s)) and e[j] <= thr and not blocked(j):
            start_i = j
            j -= 1
        r.start = max(limit, start_i / env_fps)
        r.start_reason = (("after pitch cue, " if cue else "")
                          + f"first note {s0 - onset:.1f} s before detector, {onset - r.start:.1f} s lead-in")
        # --- end: where the audience starts to clap. Without applause before the next
        # piece: just before the next pitch cue or speech; otherwise the end of the reverb.
        nxt_a = regions[k + 1][0] if k + 1 < len(regions) else len(lab)
        if cues and k + 1 < len(regions) and cues[k + 1]:
            nxt_a = min(nxt_a, cues[k + 1][0])
        search_end = min(len(lab), nxt_a, b + int(p.separator_search_s * RATE))
        window = lab[b:search_end]
        app = np.flatnonzero(window == APPLAUSE)
        spk = np.flatnonzero(window == SPEECH)
        if len(app):
            st, kind = (b + app[0]) / RATE, "applause"
        elif len(spk):
            st, kind = (b + spk[0]) / RATE, "speech"
        else:
            st, kind = None, ""
        if st is not None:
            j0, j1 = idx(max(s0, st - 1.5)), idx(st + 0.5)
            step = max(1, int(0.1 * env_fps))
            if kind == "applause" and j1 - j0 > step:
                rise = hf[j0 + step:j1] - hf[j0:j1 - step]
                st = (j0 + int(np.argmax(rise))) / env_fps
            end, r.end_reason = st - 0.05, kind
        elif nxt_a < len(lab):
            end, r.end_reason = nxt_a / RATE - 0.5, "next piece"
        else:
            t0, t1 = idx(e0), idx(e0 + p.max_tail_s)
            below = np.flatnonzero(e[t0:t1] < np.percentile(e[t0:t1], 10) + p.tail_db) if t1 - t0 > env_fps else []
            end = (t0 + below[0]) / env_fps + p.tail_margin_s if len(below) else e0 + p.max_tail_s
            r.end_reason = "silence"
        if end >= total:
            end, r.end_reason = total, "end of recording"
        r.end = max(r.start + 1.0, end)
        sep_t = st
        if out and r.start < out[-1].end:
            r.start = out[-1].end
        m = float(music[a:b].mean()) if b > a else 0.0
        clear_start = bool(cue) or k == 0 or (k > 0 and out and out[-1].end_reason in ("applause", "speech"))
        r.confidence = round(min(1.0, m / 0.35) * (0.6 + 0.2 * clear_start + 0.2 * (sep_t is not None)), 3)
        out.append(r)
    for k in range(len(out) - 1):
        if out[k].end > out[k + 1].start:
            out[k].end = out[k + 1].start
    return out


def segment(scores_100fps: np.ndarray, energy_db: np.ndarray, hf_db: np.ndarray, env_fps: int,
            total: float, expected: int | None = None,
            p: SegmentParams | None = None) -> tuple[list[Region], list[str], np.ndarray]:
    """Full pipeline. Returns regions (stream seconds), warnings, frame labels @ RATE."""
    p = p or SegmentParams()
    sc = downsample(scores_100fps, 100)
    lab = clean_runs(label_frames(sc, p), int(p.min_run_s * RATE))
    regions, warnings, encore, cues = find_regions(lab, sc[:, 0], p, expected)
    out = refine(regions, lab, sc[:, 0], energy_db, hf_db, env_fps, total, p, cues)
    for r, enc in zip(out, encore):
        r.encore = enc
    return out, warnings, lab


def energy_envelopes(x: np.ndarray, sr: int, fps: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Broadband and high-frequency (2-8 kHz) RMS level in dB at ``fps``."""
    from scipy.signal import butter, sosfilt

    hop = sr // fps
    n = len(x) // hop
    if n == 0:
        return np.zeros(0), np.zeros(0)

    def rms_db(y: np.ndarray) -> np.ndarray:
        fr = y[: n * hop].reshape(n, hop).astype(np.float64)
        return 10 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)

    sos = butter(4, [2000, min(8000, sr / 2 - 100)], btype="band", fs=sr, output="sos")
    return rms_db(x), rms_db(sosfilt(sos, x))
