"""The ``analyze`` step: sources + programme -> plan (sync, pieces, names, gains)."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from . import detect, gain, media, programme as prog, segment, sync
from .plan import Piece, Plan, Settings
from .timecode import to_tc

log = logging.getLogger(__name__)

ENCORE_TITLE = "Encore"


class RereadProgramme(Exception):
    """Raised by ``confirm_programme`` to read the programme PDF again."""


class RereadMedia(Exception):
    """Raised by ``confirm_matching`` to read the video/audio files again (sync + detection)."""


@dataclass
class Match:
    """A music segment (timeline seconds) and the programme item it is named after."""
    start: float
    end: float
    item: prog.ProgrammeItem | None  # None: encore / piece not in the programme
    region: segment.Region
    offset: float | None = None  # measured sync offset (s), None = measure it
    name: tuple[str, str] | None = None  # (composer, title) kept from an edited plan
    enabled: bool = True

    @property
    def composer(self) -> str:
        if self.name:
            return self.name[0]
        return self.item.composer_or_folk if self.item else ""

    @property
    def title(self) -> str:
        if self.name:
            return self.name[1]
        return self.item.title if self.item else ENCORE_TITLE


ProgrammeCallback = Callable[[prog.Programme], prog.Programme]
EnterProgrammeCallback = Callable[[str], "prog.Programme | None"]
MatchingCallback = Callable[[list[Match], "prog.Programme | None"], list[Match]]


def _concert_info(concert_dir: Path, pr: prog.Programme | None, artist: str | None) -> dict:
    name = concert_dir.name
    m = re.match(r"(\d{4})(\d{2})(\d{2})", name)
    info = {"name": name, "dir": str(concert_dir.resolve())}
    info["date"] = f"{m[1]}-{m[2]}-{m[3]}" if m else ""
    info["album"] = (pr.concert_title if pr else "") or name
    info["artist"] = artist or (pr.performer if pr else "")
    return info


def _cache_key(files: list[media.MediaFile]) -> str:
    h = hashlib.sha1()
    for f in files:
        h.update(f"{Path(f.path).name}:{f.size}:{f.duration:.3f};".encode())
    return h.hexdigest()


def detection_scores(audio: list[media.MediaFile], cache: Path | None,
                     model_dir: str | Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-group scores (100 fps) and energy envelopes (100 fps) of the recorder stream."""
    key = _cache_key(audio)
    if cache and cache.exists():
        d = np.load(cache)
        if str(d["key"]) == key:
            log.info("using cached detection results %s", cache)
            return d["scores"], d["energy"], d["hf"]
    span = media.group_span(audio)
    log.info("decoding recorder audio at 32 kHz (%.0f min)", span / 60)
    x = media.decode_group(audio, detect.SR, 0.0, span)
    energy, hf = segment.energy_envelopes(x, detect.SR, 100)
    log.info("running the music/applause/speech detector (CPU)")
    det = detect.Detector(model_dir)
    last = [-1]

    def progress(frac: float) -> None:
        pct = int(frac * 100) // 10 * 10
        if pct != last[0]:
            last[0] = pct
            log.info("  detector %d%%", pct)

    scores = det.group_scores(x, progress=progress)
    n = min(len(scores), len(energy))
    scores, energy, hf = scores[:n], energy[:n], hf[:n]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, key=key, scores=scores, energy=energy, hf=hf)
    return scores, energy, hf


def _load_media(concert_dir: Path, video: list[str] | None,
                audio: list[str] | None) -> tuple[list[media.MediaFile], list[media.MediaFile]]:
    vids, auds = media.discover(concert_dir)
    vids = [Path(v) for v in video] if video else vids
    auds = [Path(a) for a in audio] if audio else auds
    if not vids and not auds:
        raise media.MediaError(f"no camera or recorder files found in {concert_dir} (use --video/--audio)")
    vfiles = media.load_group(vids, "video") if vids else []
    if vids:
        log.info("camera: %s", ", ".join(p.name for p in vids))
    if auds:
        log.info("recorder: %s", ", ".join(p.name for p in auds))
        afiles = media.load_group(auds, "audio")
    else:
        # video only: the camera's own sound is the recording
        silent = [f.name for f in vfiles if f.audio_index < 0]
        if silent:
            raise media.MediaError(f"no recorder files and no sound in {', '.join(silent)}")
        log.info("no recorder files: using the camera's own sound")
        afiles = [replace(f) for f in vfiles]
        return vfiles, afiles
    if not vfiles:
        log.info("no camera files: audio only (FLAC export, no MP4)")
        return vfiles, afiles
    vdays = {f.creation_time.date() for f in vfiles if f.creation_time}
    adays = {f.creation_time.date() for f in afiles if f.creation_time}
    if vdays and adays and min(abs((v - a).days) for v in vdays for a in adays) > 1:
        log.warning("camera files are dated %s but recorder files %s - are they from the same concert?",
                    ", ".join(map(str, sorted(vdays))), ", ".join(map(str, sorted(adays))))
    return vfiles, afiles


def _read_programme(concert_dir: Path, programme_path: str | Path | None, skip: set[int] | None,
                    enter_programme: EnterProgrammeCallback | None) -> tuple[prog.Programme | None, str]:
    pr, problem = None, ""
    pdf = Path(programme_path) if programme_path else prog.find_pdf(concert_dir)
    if pdf:
        try:
            pr = prog.load(pdf)
            log.info("programme %s: %d pieces", pdf.name, len(pr.items))
        except prog.ProgrammeError:
            problem = f"no pieces could be read from {pdf.name} (a scanned PDF has no text)"
    else:
        problem = "no programme PDF found"
    if pr is None:
        if enter_programme:
            pr = enter_programme(problem)
        if pr is None:
            log.warning("%s; pieces will only be numbered (use --programme)", problem)
            return None, ""
    if skip:
        unknown = skip - {i.number for i in pr.items}
        if unknown:
            raise prog.ProgrammeError(f"--skip: no programme item(s) {sorted(unknown)}")
        for i in pr.items:
            if i.number in skip:
                log.info("not performed: %02d %s - %s", i.number, i.composer_or_folk, i.title)
        pr.items = [i for i in pr.items if i.number not in skip]
    return pr, str(pdf) if pdf else pr.source


def _programme_from_plan(plan: Plan) -> prog.Programme | None:
    d = plan.programme
    if not d.get("items"):
        return None
    header = [x for x in (d.get("concert_title", ""), d.get("performer", "")) if x]
    return prog.Programme(items=[prog.ProgrammeItem.from_dict(x) for x in d["items"]], header=header,
                          source=d.get("source", ""))


def _signature(pr: prog.Programme | None) -> list[tuple]:
    return [(i.number, i.composer, i.title) for i in pr.items] if pr else []


def _assign_in_order(matches: list[Match], pr: prog.Programme | None) -> list[Match]:
    """Name the segments after the programme in order; encores stay unnamed."""
    items = list(pr.items) if pr else []
    k = 0
    for m in matches:
        m.name = None
        m.item = items[k] if not m.region.encore and k < len(items) else None
        k += 0 if m.region.encore else 1
    return matches


def _detect(vfiles: list[media.MediaFile], afiles: list[media.MediaFile], cache: Path,
            model_dir: str | Path | None, params: segment.SegmentParams | None,
            pr: prog.Programme | None) -> tuple[sync.SyncResult, list[Match], list[str]]:
    warnings: list[str] = []
    if not vfiles or [f.path for f in afiles] == [f.path for f in vfiles]:
        sres = sync.SyncResult(0.0, 0.0, 1.0, [])  # one device: nothing to sync
    else:
        sres = sync.synchronise(vfiles, afiles)
    if sres.confidence < 0.5:
        warnings.append(f"low sync confidence ({sres.confidence:.2f}) - check audio/video sync")
    scores, energy, hf = detection_scores(afiles, cache, model_dir)
    regions, seg_warnings, _ = segment.segment(scores, energy, hf, 100, media.group_span(afiles),
                                               len(pr.items) if pr else None, params)
    warnings += seg_warnings
    a0, b = sres.audio_start, sres.drift
    to_tl = lambda z: (z + a0) / (1 - b)  # noqa: E731  recorder time -> timeline: t = z + a0 + b t
    matches = [Match(to_tl(r.start), to_tl(r.end), None, r) for r in regions]
    return sres, _assign_in_order(matches, pr), warnings


def _from_plan(plan: Plan, pr: prog.Programme | None,
               programme_changed: bool) -> tuple[sync.SyncResult, list[Match], list[str]]:
    """The pieces of an existing plan, including the user's edits to it."""
    d = plan.sync
    sres = sync.SyncResult(plan.audio_start, float(d.get("drift_ppm", 0.0)) * 1e-6,
                           float(d.get("confidence", 0.0)), [])
    by_no = {i.number: i for i in pr.items} if pr else {}
    matches = []
    for p in plan.pieces:
        det = p.detected
        r = segment.Region(0.0, 0.0, confidence=det.get("confidence", 0.0),
                           start_reason=det.get("start_reason", ""), end_reason=det.get("end_reason", ""),
                           encore=bool(det.get("encore", False)), warnings=[])
        matches.append(Match(p.start, p.end, by_no.get(det.get("programme_number")), r,
                             offset=plan.audio_start + p.audio_offset_ms / 1000.0,
                             name=(p.composer, p.title), enabled=p.enabled))
    if programme_changed:
        _assign_in_order(matches, pr)
    return sres, matches, [w for w in plan.warnings if not w.startswith("no recording matched")]


def analyze(concert_dir: str | Path, plan_path: str | Path | None = None,
            programme_path: str | Path | None = None, video: list[str] | None = None,
            audio: list[str] | None = None, model_dir: str | Path | None = None,
            artist: str | None = None, settings: dict | None = None,
            params: segment.SegmentParams | None = None, skip: set[int] | None = None,
            confirm_programme: ProgrammeCallback | None = None,
            confirm_matching: MatchingCallback | None = None,
            enter_programme: EnterProgrammeCallback | None = None) -> Plan:
    """Analyse a concert folder and write the plan.

    An existing plan is reused: its programme and pieces (with the user's edits) are the
    starting point, and the same confirmations are asked as on the first run. To start
    from scratch, delete the plan file.

    ``settings``: settings to change (e.g. ``{"peak_dbfs": -0.5}``).
    ``skip``: programme numbers that were listed but not performed.
    ``confirm_programme(programme) -> programme``: lets the caller (e.g. the CLI, asking
    the user) correct the performed programme; may raise RereadProgramme.
    ``confirm_matching(matches, programme) -> matches``: lets the caller check and fix
    which segment is which piece; may raise RereadMedia.
    ``enter_programme(reason) -> programme | None``: asked when no programme could be
    determined (no PDF, a scanned PDF, nothing recognisable); None = continue without names.
    """
    media.require_ffmpeg()
    concert_dir = Path(concert_dir)
    plan_path = Path(plan_path) if plan_path else concert_dir / "concertcut.json"
    cache = plan_path.with_suffix(".cache.npz")
    prev = Plan.load(plan_path) if plan_path.exists() else None
    if prev:
        log.info("%s already exists, loading existing preferences (delete it to start from scratch)",
                 plan_path)

    # programme
    pr = _programme_from_plan(prev) if prev else None
    source = prev.programme.get("source", "") if pr else ""
    if pr is None:
        pr, source = _read_programme(concert_dir, programme_path, skip, enter_programme)
    saved = _signature(_programme_from_plan(prev)) if prev else None
    while pr and confirm_programme:
        try:
            pr = confirm_programme(pr)
            break
        except RereadProgramme:
            pr, source = _read_programme(concert_dir, programme_path, skip, enter_programme)

    # media: sync, detection, segments
    if prev:
        vfiles, afiles = prev.video, prev.audio
        sres, matches, warnings = _from_plan(prev, pr, _signature(pr) != saved)
    else:
        vfiles, afiles = _load_media(concert_dir, video, audio)
        sres, matches, warnings = _detect(vfiles, afiles, cache, model_dir, params, pr)
    while confirm_matching:
        try:
            matches = confirm_matching(matches, pr)
            break
        except RereadMedia:
            vfiles, afiles = _load_media(concert_dir, video, audio)
            sres, matches, warnings = _detect(vfiles, afiles, cache, model_dir, params, pr)

    one_device = not vfiles or [f.path for f in afiles] == [f.path for f in vfiles]
    todo = [m for m in matches if m.offset is None]
    if todo and one_device:
        for m in todo:
            m.offset = sres.audio_start
    elif todo:
        log.info("measuring the sync inside %d piece(s)", len(todo))
        for m, off in zip(todo, sync.piece_offsets(vfiles, afiles, sres, [(m.start, m.end) for m in todo])):
            m.offset = off["offset"]
            m.region.warnings.extend([] if off["anchors"] else
                                     ["no sync measurement inside the piece; using the global line"])
    a0 = sres.audio_start
    items = list(pr.items) if pr else []
    pieces: list[Piece] = []
    for n, m in enumerate(matches, 1):  # track numbers follow the performance
        r = m.region
        # without a programme, pieces are just numbered; otherwise unknown ones are encores
        title = m.title if (m.name or m.item or items or r.encore) else f"Piece {n}"
        p = Piece(number=n, composer=m.composer, title=title, start=round(m.start, 3), end=round(m.end, 3),
                  enabled=m.enabled, audio_offset_ms=(m.offset - a0) * 1000.0,
                  detected={"confidence": r.confidence, "start_reason": r.start_reason,
                            "end_reason": r.end_reason, "encore": m.item is None and not m.name,
                            "programme_number": m.item.number if m.item else None,
                            "warnings": list(r.warnings)})
        if vfiles and (m.start < 0 or m.end > media.group_span(vfiles)):
            p.detected["warnings"].append("piece is not fully covered by the camera (black video added)")
        pieces.append(p)
    matched = {m.item.number for m in matches if m.item}
    unmatched = [i for i in items if i.number not in matched]
    if unmatched:
        warnings.append("no recording matched to programme item(s) "
                        + ", ".join(f"{i.number:02d}" for i in unmatched))

    base = prev.settings if prev else Settings()
    for k, v in (settings or {}).items():
        setattr(base, k, v)
    sync_info = {"drift_ppm": round(sres.drift * 1e6, 3), "confidence": round(sres.confidence, 3),
                 "anchors": [{"t": to_tc(a["t"]), "offset": round(a["offset"], 5),
                              "sharpness": round(a["sharpness"], 1), "inlier": a.get("inlier", False)}
                             for a in sres.anchors] if sres.anchors else (prev.sync.get("anchors", [])
                                                                          if prev else [])}
    plan = Plan(
        concert=_concert_info(concert_dir, pr, artist or (prev.concert.get("artist") if prev else None)),
        video=vfiles, audio=afiles, audio_start=a0, pieces=pieces, sync=sync_info,
        programme={**(pr.to_dict() if pr else {}), "source": source,
                   "not_performed": sorted(set(skip or []) | set(prev.programme.get("not_performed", [])
                                                                  if prev else []))},
        settings=base, warnings=warnings, path=plan_path,
    )
    gain.update_gain(plan)
    for p in plan.pieces:
        log.info("%s  %s - %s  peak %s dBFS  (%s)", p.basename, to_tc(p.start), to_tc(p.end),
                 p.detected.get("peak_dbfs"), p.detected.get("end_reason"))
    for w in warnings:
        log.warning("%s", w)
    plan.save()
    log.info("plan written to %s", plan_path)
    return plan
