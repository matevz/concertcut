"""Media discovery, probing (ffprobe) and decoding (ffmpeg) of split recordings.

Cameras and field recorders split long recordings into several files. Each file is
placed on its group's own stream time axis (0 = start of the first file) using
recording metadata (camera ``creation_time``, BWF ``time_reference``); files whose
metadata says they follow each other are snapped together back to back.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".mov", ".mts", ".m2ts", ".mkv", ".avi"}
AUDIO_EXT = {".wav", ".flac", ".bwf", ".aif", ".aiff"}
IGNORE_AUDIO = re.compile(r"normali[sz]ed|render|mix", re.I)
# Files whose recorded start differs from "previous file end" by less than this are
# treated as a continuous (split) recording.
CONTIGUOUS_TOL_VIDEO = 3.0  # creation_time has 1 s resolution
CONTIGUOUS_TOL_AUDIO = 0.1


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise MediaError(
            f"{', '.join(missing)} not found. Install it with: sudo apt install ffmpeg"
        )


@dataclass
class MediaFile:
    path: str
    duration: float
    start: float = 0.0  # position on the group's stream time axis (seconds)
    audio_index: int = -1  # absolute stream index of the first audio stream
    video_index: int = -1
    sample_rate: int = 0
    channels: int = 0
    bits: int = 0
    fps: Fraction | None = None
    width: int = 0
    height: int = 0
    sar: Fraction = Fraction(1)
    progressive: bool = True
    size: int = 0
    creation_time: datetime | None = None
    time_reference: int | None = None  # BWF: samples since midnight

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def name(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> dict:
        from .timecode import to_tc

        return {"file": self.path, "start": to_tc(self.start), "duration": to_tc(self.duration)}


def _parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def probe(path: str | Path) -> MediaFile:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    fmt = info.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    mf = MediaFile(
        path=str(Path(path).resolve()),
        duration=float(fmt.get("duration", 0.0)),
        size=int(fmt.get("size", 0) or 0),
        creation_time=_parse_time(tags.get("creation_time")),
    )
    if mf.creation_time is None and tags.get("date") and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tags["date"]):
        # Broadcast WAV: date and time are separate fields
        mf.creation_time = _parse_time(f"{tags['date']}T{tags.get('creation_time') or '00:00:00'}")
    if "time_reference" in tags:
        try:
            mf.time_reference = int(tags["time_reference"])
        except ValueError:
            pass
    for st in info.get("streams", []):
        kind = st.get("codec_type")
        if kind == "audio" and mf.audio_index < 0:
            mf.audio_index = int(st["index"])
            mf.sample_rate = int(st.get("sample_rate", 0))
            mf.channels = int(st.get("channels", 0))
            mf.bits = int(st.get("bits_per_raw_sample") or st.get("bits_per_sample") or 0)
        elif kind == "video" and mf.video_index < 0 and not st.get("disposition", {}).get("attached_pic"):
            mf.video_index = int(st["index"])
            mf.width, mf.height = int(st.get("width", 0)), int(st.get("height", 0))
            rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
            mf.fps = Fraction(rate) if rate != "0/0" else None
            sar = st.get("sample_aspect_ratio", "1:1")
            if sar and sar != "0:1":
                num, den = sar.split(":")
                mf.sar = Fraction(int(num), int(den))
            mf.progressive = st.get("field_order", "progressive") in ("progressive", "unknown")
    return mf


def place_group(files: list[MediaFile], kind: str) -> list[MediaFile]:
    """Assign ``start`` to each file of a split recording (sorted by name)."""
    if not files:
        return files
    files[0].start = 0.0
    tol = CONTIGUOUS_TOL_AUDIO if kind == "audio" else CONTIGUOUS_TOL_VIDEO
    for prev, cur in zip(files, files[1:]):
        delta = None  # recorded start of `cur` relative to start of `prev`
        if kind == "audio" and prev.time_reference is not None and cur.time_reference is not None \
                and prev.sample_rate and prev.sample_rate == cur.sample_rate:
            delta = (cur.time_reference - prev.time_reference) / prev.sample_rate
            if delta < 0:  # recording crossed midnight
                delta += 86400
        elif prev.creation_time and cur.creation_time:
            delta = (cur.creation_time - prev.creation_time).total_seconds()
        # Some cameras stamp the end of the recording instead of the start, hence
        # the second comparison. The audio sync later verifies camera placement.
        if delta is None or delta < prev.duration or abs(delta - prev.duration) <= tol \
                or (kind == "video" and abs(delta - cur.duration) <= tol):
            cur.start = prev.end
        else:
            cur.start = prev.start + delta
            log.warning("%s starts %.1f s after the end of %s (separate recording, gap kept)",
                        cur.name, delta - prev.duration, prev.name)
    return files


def _group_key(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[-_]?\d{1,4}$", "", stem)


def discover(concert_dir: str | Path) -> tuple[list[Path], list[Path]]:
    """Find camera video files and the main field-recorder audio group."""
    d = Path(concert_dir)
    files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith("."))
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    audios = [p for p in files if p.suffix.lower() in AUDIO_EXT and not IGNORE_AUDIO.search(p.name)]
    groups: dict[str, list[Path]] = {}
    for p in audios:
        groups.setdefault(_group_key(p.name), []).append(p)
    if len(groups) > 1:
        best = max(groups.values(), key=lambda g: sum(p.stat().st_size for p in g))
        log.warning("several audio recordings found (%s); using %s (largest). Use --audio to choose.",
                    ", ".join(groups), _group_key(best[0].name))
        audios = best
    return videos, audios


def load_group(paths: list[str | Path], kind: str) -> list[MediaFile]:
    files = [probe(p) for p in paths]
    for f in files:
        if kind == "video" and f.video_index < 0:
            raise MediaError(f"{f.path} has no video stream")
        if kind == "audio" and f.audio_index < 0:
            raise MediaError(f"{f.path} has no audio stream")
    return place_group(files, kind)


def decode(path: str, sr: int, start: float, duration: float, channels: int = 1,
           stream: int | None = None, hq: bool = False) -> np.ndarray:
    """Decode an audio excerpt to float32, shape (samples,) or (samples, channels).

    ``hq``: resample with soxr (as the FLAC export does) instead of ffmpeg's default.
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-t", f"{duration:.6f}", "-i", path, "-map", f"0:{stream}" if stream is not None else "0:a:0",
            "-vn", "-ac", str(channels)]
    cmd += ["-af", f"aresample={sr}:resampler=soxr:precision=28"] if hq else ["-ar", str(sr)]
    cmd += ["-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    return x if channels == 1 else x.reshape(-1, channels)


def decode_group(files: list[MediaFile], sr: int, t0: float, t1: float,
                 channels: int = 1, hq: bool = False) -> np.ndarray:
    """Decode the stream-time window [t0, t1) of a file group; gaps are silent."""
    n = max(0, int(round((t1 - t0) * sr)))
    out = np.zeros((n, channels) if channels > 1 else n, dtype=np.float32)
    for f in files:
        a, b = max(t0, f.start), min(t1, f.end)
        if b <= a:
            continue
        x = decode(f.path, sr, a - f.start, b - a, channels, f.audio_index, hq)
        i = int(round((a - t0) * sr))
        m = min(len(x), n - i)
        out[i:i + m] = x[:m]
    return out


def group_span(files: list[MediaFile]) -> float:
    return max(f.end for f in files) if files else 0.0


def segments(files: list[MediaFile], t0: float, t1: float) -> list[tuple[MediaFile | None, float, float]]:
    """Split the stream window [t0, t1) into (file, in_point, duration) pieces.

    ``file`` is None for gaps not covered by any file.
    """
    out: list[tuple[MediaFile | None, float, float]] = []
    t = t0
    for f in sorted(files, key=lambda f: f.start):
        if f.end <= t or f.start >= t1:
            continue
        if f.start > t + 1e-6:
            out.append((None, 0.0, f.start - t))
            t = f.start
        b = min(t1, f.end)
        out.append((f, t - f.start, b - t))
        t = b
        if t >= t1 - 1e-6:
            break
    if t < t1 - 1e-6:
        out.append((None, 0.0, t1 - t))
    return out
