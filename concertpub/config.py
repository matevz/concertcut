"""The editable config file (``concertpub.json``) shared by all pipeline steps.

All times are on the *timeline*: 0 = first frame of the first camera file, with the
camera files at their recorded positions. The Zoom/recorder stream time ``z`` for a
timeline time ``t`` inside a piece is ``z = t - (sync.audio_start + piece.audio_offset_ms/1000)``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .media import MediaFile
from .timecode import to_seconds, to_tc

VERSION = 1
_BAD_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


CD_RATE, CD_BITS = 44100, 16


def flac_rate(audio: list[MediaFile], override: int | None = None) -> int:
    """FLAC sample rate: CD quality (44.1 kHz) unless the recording's rate is lower."""
    if override:
        return override
    src = audio[0].sample_rate if audio else CD_RATE
    return min(src, CD_RATE) if src else CD_RATE


def safe_filename(s: str) -> str:
    s = unicodedata.normalize("NFC", _BAD_CHARS.sub("", s))  # composed č, š, ž (see programme.clean_text)
    return re.sub(r"\s+", " ", s).strip(" .")


@dataclass
class Settings:
    peak_dbfs: float = -0.1
    # One constant gain for the whole concert: the largest increase that keeps the
    # loudest piece at peak_dbfs. gain_db_auto: re-measure after edits to the pieces.
    gain_db: float | None = None
    gain_db_auto: bool = True
    gain_limited_by: str = ""  # the loudest piece, which sets the concert gain
    mp4_peak_dbfs: float | None = None  # None = same as peak_dbfs
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0
    flac_sample_rate: int | None = None  # None = CD quality (44.1 kHz) or the source rate if lower
    x264_crf: int = 22  # matches the quality of a Kdenlive render (CRF 23, faster preset)
    x264_preset: str = "slow"
    aac_bitrate: str = "320k"

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if d["gain_db"] is not None:
            d["gain_db"] = round(d["gain_db"], 2)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Piece:
    number: int
    composer: str
    title: str
    start: float
    end: float
    enabled: bool = True
    audio_offset_ms: float = 0.0
    fade_in_s: float | None = None
    fade_out_s: float | None = None
    detected: dict = field(default_factory=dict)

    @property
    def basename(self) -> str:
        if self.composer:
            return safe_filename(f"{self.number:02d} {self.composer} - {self.title}")
        return safe_filename(f"{self.number:02d} {self.title}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "enabled": self.enabled,
            "composer": self.composer,
            "title": self.title,
            "start": to_tc(self.start),
            "end": to_tc(self.end),
            "audio_offset_ms": round(self.audio_offset_ms, 2),
            "fade_in_s": self.fade_in_s,
            "fade_out_s": self.fade_out_s,
            "detected": self.detected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Piece":
        return cls(
            number=int(d["number"]), composer=d.get("composer", ""), title=d.get("title", ""),
            start=to_seconds(d["start"]), end=to_seconds(d["end"]),
            enabled=bool(d.get("enabled", True)),
            audio_offset_ms=float(d.get("audio_offset_ms") or 0.0),
            fade_in_s=d.get("fade_in_s"), fade_out_s=d.get("fade_out_s"),
            detected=d.get("detected", {}),
        )


def _file_to_dict(f: MediaFile) -> dict:
    d = {"file": f.path, "start": to_tc(f.start), "duration": to_tc(f.duration),
         "stream_audio": f.audio_index}
    if f.video_index >= 0:
        d.update(stream_video=f.video_index, width=f.width, height=f.height,
                 fps=f"{f.fps.numerator}/{f.fps.denominator}" if f.fps else None,
                 sar=f"{f.sar.numerator}/{f.sar.denominator}", progressive=f.progressive)
    if f.sample_rate:
        d.update(sample_rate=f.sample_rate, channels=f.channels, bits=f.bits)
    if f.size:
        d["size"] = f.size
    return d


def _file_from_dict(d: dict) -> MediaFile:
    return MediaFile(
        path=d["file"], start=to_seconds(d.get("start", 0)), duration=to_seconds(d["duration"]),
        audio_index=int(d.get("stream_audio", -1)), video_index=int(d.get("stream_video", -1)),
        sample_rate=int(d.get("sample_rate", 0)), channels=int(d.get("channels", 0)),
        bits=int(d.get("bits", 0)), fps=Fraction(d["fps"]) if d.get("fps") else None,
        width=int(d.get("width", 0)), height=int(d.get("height", 0)),
        sar=Fraction(d.get("sar", "1/1")), progressive=bool(d.get("progressive", True)),
        size=int(d.get("size", 0)),
    )


@dataclass
class Config:
    concert: dict
    video: list[MediaFile]
    audio: list[MediaFile]
    audio_start: float
    pieces: list[Piece]
    sync: dict = field(default_factory=dict)  # drift_ppm, confidence, anchors...
    programme: dict = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)
    warnings: list[str] = field(default_factory=list)
    path: Path | None = None

    # -- geometry --------------------------------------------------------------------
    @property
    def flac_rate(self) -> int:
        return flac_rate(self.audio, self.settings.flac_sample_rate)

    @property
    def camera_is_recorder(self) -> bool:
        """No separate recorder: the camera's own sound is the recording."""
        return bool(self.video) and [f.path for f in self.audio] == [f.path for f in self.video]

    @property
    def fps(self) -> Fraction:
        for f in self.video:
            if f.fps:
                return f.fps
        return Fraction(25)

    @property
    def concert_dir(self) -> Path:
        return Path(self.concert["dir"])

    def audio_time(self, piece: Piece, t: float) -> float:
        """Recorder stream time for timeline time ``t`` inside ``piece``."""
        return t - self.audio_start - piece.audio_offset_ms / 1000.0

    @property
    def project_shift(self) -> float:
        """Seconds to add to timeline times in editor projects so nothing starts before 0
        (when the recorder was started before the camera and a piece begins before it)."""
        starts = [p.start for p in self.enabled_pieces()] + [f.start for f in self.video]
        return max(0.0, -min(starts)) if starts else 0.0

    @property
    def gain_db(self) -> float:
        return self.settings.gain_db or 0.0

    @property
    def project_audio_start(self) -> float:
        """Single recorder offset for editor projects with a continuous recorder track:
        the median of the per-piece offsets (they differ by the clock drift, a few ms)."""
        offs = sorted(self.audio_start + p.audio_offset_ms / 1000.0 for p in self.enabled_pieces())
        return offs[len(offs) // 2] if offs else self.audio_start

    def recorder_layout(self) -> list[tuple[MediaFile, float, float, float]]:
        """Recorder files on the (shifted) project timeline: (file, position, in point, duration)."""
        a, shift = self.project_audio_start, self.project_shift
        out = []
        for f in sorted(self.audio, key=lambda f: f.start):
            pos, inp = a + f.start + shift, 0.0
            if pos < 0:
                pos, inp = 0.0, -pos
            if f.duration - inp > 0:
                out.append((f, pos, inp, f.duration - inp))
        return out

    def enabled_pieces(self, only: set[int] | None = None) -> list[Piece]:
        return [p for p in self.pieces if p.enabled and (not only or p.number in only)]

    def fade_in(self, p: Piece) -> float:
        return self.settings.fade_in_s if p.fade_in_s is None else float(p.fade_in_s)

    def fade_out(self, p: Piece) -> float:
        return self.settings.fade_out_s if p.fade_out_s is None else float(p.fade_out_s)

    # -- io ----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "concert": self.concert,
            "sources": {
                "video": [_file_to_dict(f) for f in self.video],
                "audio": [_file_to_dict(f) for f in self.audio],
            },
            "programme": self.programme,
            "sync": {"audio_start": to_tc(self.audio_start), **self.sync},
            "settings": self.settings.to_dict(),
            "warnings": self.warnings,
            "pieces": [p.to_dict() for p in self.pieces],
        }

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != VERSION:
            raise ValueError(f"{path}: unsupported config version {d.get('version')}")
        sync = dict(d.get("sync", {}))
        audio_start = to_seconds(sync.pop("audio_start", 0.0))
        return cls(
            concert=d["concert"],
            video=[_file_from_dict(x) for x in d["sources"]["video"]],
            audio=[_file_from_dict(x) for x in d["sources"]["audio"]],
            audio_start=audio_start,
            pieces=[Piece.from_dict(x) for x in d.get("pieces", [])],
            sync=sync,
            programme=d.get("programme", {}),
            settings=Settings.from_dict(d.get("settings", {})),
            warnings=d.get("warnings", []),
            path=path,
        )
