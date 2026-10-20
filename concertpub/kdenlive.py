"""Kdenlive project (document version 1.1, Kdenlive 23.04+ with sequences) from a config.

Track layout (bottom to top, like a typical concert project):
  audio: recorder, whole files at one sync offset, volume filter at the concert gain
  audio: camera sound (muted track)
  video: camera files
  video: empty
Guides "NN Composer - Title" / "NN end" mark every piece for render-by-guides.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
import uuid as uuidlib
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from . import media
from .media import MediaFile
from .config import Config

log = logging.getLogger(__name__)

KDENLIVE_VERSION = "26.04.2"
MLT_VERSION = "7.39.0"

_PROFILES = {
    (1920, 1080): "atsc_1080p", (1280, 720): "atsc_720p", (3840, 2160): "uhd_2160p",
}
_RATES = {Fraction(25): "25", Fraction(50): "50", Fraction(30): "30", Fraction(60): "60",
          Fraction(24): "24", Fraction(30000, 1001): "2997", Fraction(60000, 1001): "5994",
          Fraction(24000, 1001): "2398"}


def profile_name(w: int, h: int, fps: Fraction) -> str:
    base, rate = _PROFILES.get((w, h)), _RATES.get(fps)
    return f"{base}_{rate}" if base and rate else ""


class _Doc:
    def __init__(self, fps: Fraction):
        self.fps = fps
        self.counters: dict[str, int] = {}

    def next_id(self, kind: str) -> str:
        n = self.counters.get(kind, 0)
        self.counters[kind] = n + 1
        return f"{kind}{n}"

    def frames(self, seconds: float) -> int:
        return int(round(seconds * self.fps))

    def tc(self, frames: int) -> str:
        ms = int(round(frames * 1000 * self.fps.denominator / self.fps.numerator))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _props(el: ET.Element, props: dict) -> ET.Element:
    for k, v in props.items():
        if v is None:
            continue
        p = ET.SubElement(el, "property", name=k)
        p.text = str(v)
    return el


def _uuid() -> str:
    return "{" + str(uuidlib.uuid4()) + "}"


def build(config: Config) -> ET.ElementTree:
    fps = config.fps
    doc = _Doc(fps)
    v0 = next((f for f in config.video if f.video_index >= 0), None)
    width, height = (v0.width, v0.height) if v0 else (1920, 1080)
    sar = v0.sar if v0 else Fraction(1)
    dar = Fraction(width, height) * sar
    seq_uuid = _uuid()

    # ---- timeline content ---------------------------------------------------------
    shift = config.project_shift
    cam_entries = []  # (file, timeline_frame, length_frames)
    cursor = 0
    for f in sorted(config.video, key=lambda f: f.start):
        pos = max(cursor, doc.frames(f.start + shift))
        length = max(1, doc.frames(f.duration))
        cam_entries.append((f, pos, length))
        cursor = pos + length

    # Continuous recorder track, like a hand-made project: whole files at one sync
    # offset, each with a volume filter at the constant concert gain.
    guides = []
    for p in config.enabled_pieces():
        guides.append({"comment": p.basename, "duration": 0, "pos": doc.frames(p.start + shift), "type": 0})
        guides.append({"comment": f"{p.number:02d} end", "duration": 0, "pos": doc.frames(p.end + shift),
                       "type": 0})
    zoom_entries = []  # (file, timeline_frame, in_frame, length_frames, level, gain)
    cursor = 0
    for f, pos_s, in_s, dur_s in config.recorder_layout():
        pos, in_f, n = doc.frames(pos_s), doc.frames(in_s), doc.frames(dur_s)
        skip = max(0, cursor - pos)  # never overlap the previous file
        pos, in_f, n = pos + skip, in_f + skip, n - skip
        if n <= 0:
            continue
        level = f"{doc.tc(in_f)}={config.gain_db:.2f}"  # constant, like a hand-made project
        zoom_entries.append((f, pos, in_f, n, level, config.gain_db))
        cursor = pos + n
    total = max([pos + n for _, pos, n in cam_entries] + [e[1] + e[3] for e in zoom_entries] + [1])

    # ---- document ---------------------------------------------------------------------
    root = ET.Element("mlt", {"LC_NUMERIC": "C", "producer": "main_bin",
                              "root": str(config.concert_dir), "version": MLT_VERSION})
    ET.SubElement(root, "profile", {
        "colorspace": "709" if height >= 720 else "601",
        "description": f"{width}x{height} {float(fps):g} fps",
        "display_aspect_den": str(dar.denominator), "display_aspect_num": str(dar.numerator),
        "frame_rate_den": str(fps.denominator), "frame_rate_num": str(fps.numerator),
        "height": str(height), "progressive": "1" if (not v0 or v0.progressive) else "0",
        "sample_aspect_den": str(sar.denominator), "sample_aspect_num": str(sar.numerator),
        "width": str(width),
    })
    black = ET.SubElement(root, "producer", {"id": "producer0", "in": doc.tc(0), "out": doc.tc(total - 1)})
    doc.counters["producer"] = 1
    _props(black, {"length": 2147483647, "eof": "continue", "resource": "black", "aspect_ratio": 1,
                   "mlt_service": "color", "kdenlive:playlistid": "black_track",
                   "mlt_image_format": "rgba", "set.test_audio": 0})

    # bin clip ids: 2 = sequence folder, 3 = sequence, 4.. = source files
    sources: list[MediaFile] = []
    for f in list(config.video) + list(config.audio):  # each file once in the bin
        if f.path not in {x.path for x in sources}:
            sources.append(f)
    clip_id = {f.path: 4 + i for i, f in enumerate(sources)}
    control = {f.path: _uuid() for f in sources}

    def chain_props(f: MediaFile, service: str) -> dict:
        is_video = f.video_index >= 0
        return {
            "length": max(1, doc.frames(f.duration)), "eof": "pause", "resource": f.path,
            "mlt_service": service, "seekable": 1,
            "audio_index": f.audio_index, "video_index": f.video_index,
            "vstream": 0 if is_video else None, "astream": 0 if f.audio_index >= 0 else None,
            "kdenlive:id": clip_id[f.path], "kdenlive:control_uuid": control[f.path],
            "kdenlive:clip_type": 0 if is_video and f.audio_index >= 0 else (2 if is_video else 1),
            "kdenlive:folderid": -1, "kdenlive:file_size": f.size or None,
            "kdenlive:clipname": f.name, "mute_on_pause": 0,
        }

    def track_chain(f: MediaFile, audio: bool) -> str:
        cid = doc.next_id("chain")
        el = ET.SubElement(root, "chain", {"id": cid, "out": doc.tc(max(1, doc.frames(f.duration)) - 1)})
        _props(el, chain_props(f, "avformat-novalidate"))
        _props(el, {"set.test_audio": 0 if audio else 1, "set.test_image": 1 if audio else 0})
        return cid

    def audio_track_filters(el: ET.Element) -> None:
        for props in (
            {"window": 75, "max_gain": "20dB", "channel_mask": -1, "mlt_service": "volume",
             "internal_added": 237, "disable": 1},
            {"channel": -1, "mlt_service": "panner", "internal_added": 237, "start": 0.5, "disable": 1},
            {"iec_scale": 0, "mlt_service": "audiolevel", "internal_added": 237, "dbpeak": 1, "disable": 1},
        ):
            _props(ET.SubElement(el, "filter", id=doc.next_id("filter")), props)

    def track(entries: list[tuple], audio: bool, hide: str, chains: dict,
              gain_filters: bool = False) -> str:
        pl = ET.SubElement(root, "playlist", id=doc.next_id("playlist"))
        if audio:
            _props(pl, {"kdenlive:audio_track": 1})
        pos = 0
        for e in entries:
            f, at, in_f, n = e[0], e[1], e[2], e[3]
            if at > pos:
                ET.SubElement(pl, "blank", length=doc.tc(at - pos))
            ent = ET.SubElement(pl, "entry", {"in": doc.tc(in_f), "out": doc.tc(in_f + n - 1),
                                              "producer": chains[id(f)]})
            if gain_filters:
                level, g = e[4], e[5]
                max_gain = max(20, math.ceil(g) + 2)
                # in/out like Kdenlive writes them: keyframe positions are interpreted
                # relative to the clip, which does not start at 0 when it is trimmed
                flt = ET.SubElement(ent, "filter", {"id": doc.next_id("filter"), "in": doc.tc(in_f),
                                                    "out": doc.tc(in_f + n - 1)})
                _props(flt, {
                    "window": 75, "max_gain": f"{max_gain}dB", "level": level,
                    "channel_mask": -1, "mlt_service": "volume", "kdenlive_id": "volume",
                    "kdenlive:collapsed": 0})
            pos = at + n
        pl2 = ET.SubElement(root, "playlist", id=doc.next_id("playlist"))
        if audio:
            _props(pl2, {"kdenlive:audio_track": 1})
        tr = ET.SubElement(root, "tractor", {"id": doc.next_id("tractor"), "in": doc.tc(0),
                                              "out": doc.tc(total - 1)})
        tprops = {"kdenlive:trackheight": 75, "kdenlive:timeline_active": 1, "kdenlive:collapsed": 0,
                  "kdenlive:thumbs_format": "", "kdenlive:audio_rec": ""}
        _props(tr, ({"kdenlive:audio_track": 1} if audio else {}) | tprops)
        ET.SubElement(tr, "track", hide=hide, producer=pl.get("id"))
        ET.SubElement(tr, "track", hide=hide, producer=pl2.get("id"))
        if audio:
            audio_track_filters(tr)
        return tr.get("id")

    # recorder track
    zoom_chains = {id(f): track_chain(f, True) for f in config.audio
                   if any(e[0] is f for e in zoom_entries)}
    t_zoom = track(zoom_entries, True, "video", zoom_chains, gain_filters=True)
    # camera sound (muted)
    # camera sound for reference - unless it already is the recording (no separate recorder)
    cam_with_audio = [] if config.camera_is_recorder else \
        [(f, pos, 0, n) for f, pos, n in cam_entries if f.audio_index >= 0]
    cam_a_chains = {id(f): track_chain(f, True) for f, *_ in cam_with_audio}
    t_cam_a = track(cam_with_audio, True, "both", cam_a_chains)
    # camera video
    cam_v = [(f, pos, 0, n) for f, pos, n in cam_entries]
    cam_v_chains = {id(f): track_chain(f, False) for f, *_ in cam_v}
    t_cam_v = track(cam_v, False, "audio", cam_v_chains)
    t_empty = track([], False, "audio", {})

    # ---- sequence ------------------------------------------------------------------
    guides.sort(key=lambda g: g["pos"])
    guides_json = json.dumps(guides, ensure_ascii=False, indent=4)
    seq = ET.SubElement(root, "tractor", {"id": seq_uuid, "in": doc.tc(0), "out": doc.tc(total - 1)})
    _props(seq, {
        "kdenlive:uuid": seq_uuid, "kdenlive:clipname": "Sequence 1",
        "kdenlive:sequenceproperties.hasAudio": 1, "kdenlive:sequenceproperties.hasVideo": 1,
        "kdenlive:sequenceproperties.activeTrack": 0, "kdenlive:sequenceproperties.tracksCount": 4,
        "kdenlive:sequenceproperties.documentuuid": seq_uuid, "kdenlive:control_uuid": seq_uuid,
        "kdenlive:duration": doc.tc(total), "kdenlive:maxduration": total,
        "kdenlive:producer_type": 17, "kdenlive:id": 3, "kdenlive:clip_type": 0,
        "kdenlive:file_size": 0, "kdenlive:folderid": 2,
        "kdenlive:sequenceproperties.audioTarget": 1, "kdenlive:sequenceproperties.videoTarget": 2,
        "kdenlive:sequenceproperties.disablepreview": 0, "kdenlive:sequenceproperties.position": 0,
        "kdenlive:sequenceproperties.scrollPos": 0, "kdenlive:sequenceproperties.tracks": 4,
        "kdenlive:sequenceproperties.verticalzoom": 1, "kdenlive:sequenceproperties.zonein": 0,
        "kdenlive:sequenceproperties.zoneout": 75, "kdenlive:sequenceproperties.zoom": 12,
        "kdenlive:sequenceproperties.guides": guides_json, "kdenlive:markers": guides_json,
    })
    for tid in ("producer0", t_zoom, t_cam_a, t_cam_v, t_empty):
        ET.SubElement(seq, "track", producer=tid)
    for b in (1, 2):
        _props(ET.SubElement(seq, "transition", id=doc.next_id("transition")), {
            "a_track": 0, "b_track": b, "mlt_service": "mix", "kdenlive_id": "mix", "internal_added": 237,
            "always_active": 1, "accepts_blanks": 1, "sum": 1})
    for b in (3, 4):
        _props(ET.SubElement(seq, "transition", id=doc.next_id("transition")), {
            "a_track": 0, "b_track": b, "compositing": 0, "distort": 0, "rotate_center": 0,
            "mlt_service": "qtblend", "kdenlive_id": "qtblend", "internal_added": 237, "always_active": 1})
    _props(ET.SubElement(seq, "filter", id=doc.next_id("filter")), {
        "window": 75, "max_gain": "20dB", "channel_mask": -1, "mlt_service": "volume",
        "internal_added": 237, "disable": 1})
    _props(ET.SubElement(seq, "filter", id=doc.next_id("filter")), {
        "channel": -1, "mlt_service": "panner", "internal_added": 237, "start": 0.5, "disable": 1})

    # ---- bin -------------------------------------------------------------------------
    bin_chains = []
    for f in sources:
        cid = doc.next_id("chain")
        el = ET.SubElement(root, "chain", {"id": cid, "out": doc.tc(max(1, doc.frames(f.duration)) - 1)})
        _props(el, chain_props(f, "avformat"))
        bin_chains.append((cid, f))
    main_bin = ET.SubElement(root, "playlist", id="main_bin")
    pname = profile_name(width, height, fps)
    _props(main_bin, {
        "kdenlive:folder.-1.2": "Sequences", "kdenlive:sequenceFolder": 2,
        "kdenlive:docproperties.audioChannels": 2, "kdenlive:docproperties.binsort": 0,
        "kdenlive:docproperties.documentid": int(time.time() * 1000),
        "kdenlive:docproperties.enableTimelineZone": 0, "kdenlive:docproperties.enableexternalproxy": 0,
        "kdenlive:docproperties.enableproxy": 0, "kdenlive:docproperties.generateimageproxy": 0,
        "kdenlive:docproperties.generateproxy": 0,
        "kdenlive:docproperties.kdenliveversion": KDENLIVE_VERSION,
        "kdenlive:docproperties.profile": pname or None,
        "kdenlive:docproperties.seekOffset": 30000,
        "kdenlive:docproperties.sessionid": _uuid(),
        "kdenlive:docproperties.uuid": seq_uuid, "kdenlive:docproperties.version": "1.1",
        "kdenlive:expandedFolders": "", "kdenlive:binZoom": 4,
        "kdenlive:extraBins": "project_bin:-1:0", "kdenlive:documentnotes": "",
        "kdenlive:documentnotesversion": 2,
        "kdenlive:docproperties.opensequences": seq_uuid,
        "kdenlive:docproperties.activetimeline": seq_uuid,
        "xml_retain": 1,
    })
    ET.SubElement(main_bin, "entry", {"in": doc.tc(0), "out": doc.tc(0), "producer": seq_uuid})
    for cid, f in bin_chains:
        ET.SubElement(main_bin, "entry", {"in": doc.tc(0), "out": doc.tc(max(1, doc.frames(f.duration)) - 1),
                                          "producer": cid})
    proj = ET.SubElement(root, "tractor", {"id": doc.next_id("tractor"), "in": doc.tc(0),
                                            "out": doc.tc(total - 1)})
    _props(proj, {"kdenlive:projectTractor": 1})
    ET.SubElement(proj, "track", {"in": doc.tc(0), "out": doc.tc(total - 1), "producer": seq_uuid})
    ET.indent(root, space=" ")
    return ET.ElementTree(root)


def default_path(config: Config) -> Path:
    return config.concert_dir / f"{config.concert['name']}.concertpub.kdenlive"


def write_project(config: Config, path: str | Path | None = None) -> tuple[Path, Path]:
    """Write the Kdenlive project and the same timeline as .otio; an existing project is
    kept as .bak (it may contain changes made in Kdenlive)."""
    from . import otio_export

    path = Path(path) if path else default_path(config)
    if path.exists():
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        log.info("previous project kept as %s", backup)
    write(config, path)
    return path, otio_export.write(config, path.with_suffix(".otio"))


def write(config: Config, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build(config).write(path, encoding="utf-8", xml_declaration=True)
    return path
