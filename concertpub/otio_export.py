"""OpenTimelineIO export (File > OpenTimelineIO Import in Kdenlive 25.04+, or other NLEs).

Same tracks, clips and markers as the Kdenlive project, without filters: the concert
gain is written into the marker names instead.
"""

from __future__ import annotations

from pathlib import Path

import opentimelineio as otio

from . import media
from .config import Config


def _rt(seconds: float, rate: float) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(round(seconds * rate), rate)


def _range(start: float, dur: float, rate: float) -> otio.opentime.TimeRange:
    return otio.opentime.TimeRange(_rt(start, rate), _rt(dur, rate))


def _ref(f: media.MediaFile, rate: float) -> otio.schema.ExternalReference:
    return otio.schema.ExternalReference(
        target_url=Path(f.path).as_uri(), available_range=_range(0, f.duration, rate))


def _fill(track: otio.schema.Track, items: list[tuple[float, float, otio.schema.Clip]], rate: float) -> None:
    pos = 0
    for start, dur, clip in sorted(items, key=lambda x: x[0]):
        s = round(start * rate)
        if s > pos:
            track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(
                otio.opentime.RationalTime(0, rate), otio.opentime.RationalTime(s - pos, rate))))
        track.append(clip)
        pos = s + round(dur * rate)


def build(config: Config) -> otio.schema.Timeline:
    rate = float(config.fps)
    tl = otio.schema.Timeline(name=config.concert.get("name", "concert"))
    tl.global_start_time = otio.opentime.RationalTime(0, rate)

    vid = otio.schema.Track(name="Camera", kind=otio.schema.TrackKind.Video)
    cam_a = otio.schema.Track(name="Camera sound", kind=otio.schema.TrackKind.Audio)
    cam_a.enabled = False
    rec = otio.schema.Track(name="Camera sound" if config.camera_is_recorder else "Recorder",
                            kind=otio.schema.TrackKind.Audio)

    shift = config.project_shift
    v_items, a_items = [], []
    for f in config.video:
        v_items.append((f.start + shift, f.duration, otio.schema.Clip(
            name=f.name, media_reference=_ref(f, rate), source_range=_range(0, f.duration, rate))))
        if f.audio_index >= 0 and not config.camera_is_recorder:
            a_items.append((f.start + shift, f.duration, otio.schema.Clip(
                name=f.name, media_reference=_ref(f, rate), source_range=_range(0, f.duration, rate))))
    _fill(vid, v_items, rate)
    _fill(cam_a, a_items, rate)

    r_items = []
    for f, pos, inp, dur in config.recorder_layout():
        r_items.append((pos, dur, otio.schema.Clip(name=f.name, media_reference=_ref(f, rate),
                                                    source_range=_range(inp, dur, rate))))
    for p in config.enabled_pieces():
        gain = f" [gain {config.gain_db:+.2f} dB]"
        tl.tracks.markers.append(otio.schema.Marker(
            name=p.basename + gain, marked_range=_range(p.start + shift, p.duration, rate),
            color=otio.schema.MarkerColor.RED))
    _fill(rec, r_items, rate)
    tl.tracks.extend([t for t in (rec, cam_a, vid) if len(t) or t is rec])
    return tl


def write(config: Config, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(build(config), str(path))
    return path
