"""Command line: concertcut analyze|render|kdenlive|run|programme."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from . import kdenlive, otio_export, programme as prog
from .media import MediaError
from .plan import Plan

PLAN_NAME = "concertcut.json"

log = logging.getLogger("concertcut")


def _only(s: str | None) -> set[int] | None:
    return {int(x) for x in s.split(",")} if s else None


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("concert_dir", type=Path, help="folder with camera files, recorder WAVs and programme PDF")
    p.add_argument("--plan", type=Path, help="plan file to write (default: <concert_dir>/concertcut.json)")
    p.add_argument("--programme", type=Path, help="programme PDF, text file or JSON list")
    p.add_argument("--video", nargs="+", help="camera files (default: all video files in the folder)")
    p.add_argument("--audio", nargs="+", help="recorder files (default: largest WAV group)")
    p.add_argument("--skip", help="programme numbers listed but not performed, e.g. 6 or 6,8")
    p.add_argument("--artist", help="performer for file tags (default: from the programme)")
    p.add_argument("--model-dir", type=Path, help="where the detector model is stored")
    p.add_argument("--peak", type=float, help="target peak of the loudest piece in dBFS (default -0.1)")
    p.add_argument("--fade-in", type=float, help="fade-in in seconds (default: none)")
    p.add_argument("--fade-out", type=float, help="fade-out in seconds (default: none)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="do not ask to confirm the programme and the matching (for scripts)")


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", type=Path, help="output folder (default: <concert_dir>/render)")
    p.add_argument("--only", help="comma separated piece numbers, e.g. 2,5")
    p.add_argument("--audio-only", action="store_true")
    p.add_argument("--video-only", action="store_true")
    p.add_argument("-j", "--jobs", type=int, default=2, help="parallel ffmpeg processes")
    p.add_argument("--crf", type=int, help="x264 quality (default from plan: 18)")
    p.add_argument("--preset", help="x264 preset (default from plan: slow)")
    p.add_argument("--flac-rate", type=int, help="resample FLAC to this rate (default: keep source)")
    p.add_argument("--dry-run", action="store_true", help="print ffmpeg commands only")
    p.add_argument("--force", action="store_true",
                   help="render all files again (default: only new or changed pieces)")


def _add_project_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", type=Path,
                   help="Kdenlive project to write (default: <concert_dir>/<name>.concertcut.kdenlive)")


def _plan_path(p: Path) -> Path:
    """A plan file, or a concert folder containing concertcut.json."""
    return p / PLAN_NAME if p.is_dir() else p


def cmd_analyze(a) -> Plan:
    from .analyze import analyze

    plan_path = a.plan or a.concert_dir / PLAN_NAME
    settings = {k: v for k, v in (("peak_dbfs", a.peak), ("fade_in_s", a.fade_in),
                                  ("fade_out_s", a.fade_out)) if v is not None}
    ask = not a.yes and sys.stdin.isatty()
    if not ask and not a.yes:
        log.info("not a terminal: accepting the programme and the matching without asking")
    from . import interactive

    return analyze(a.concert_dir, plan_path, a.programme, a.video, a.audio, a.model_dir,
                   a.artist, settings, skip=_only(a.skip),
                   confirm_programme=interactive.confirm_programme if ask else None,
                   confirm_matching=interactive.confirm_matching if ask else None,
                   enter_programme=interactive.enter_programme if ask else None)


def cmd_render(a, plan: Plan | None = None) -> None:
    from .render import render

    plan = plan or Plan.load(_plan_path(a.plan))
    s = plan.settings
    if a.crf is not None:
        s.x264_crf = a.crf
    if a.preset:
        s.x264_preset = a.preset
    if a.flac_rate:
        s.flac_sample_rate = a.flac_rate
    render(plan, a.out_dir, _only(a.only), audio=not a.video_only, video=not a.audio_only,
           jobs=a.jobs, force=a.force, dry_run=a.dry_run)


def cmd_project(a, plan: Plan | None = None) -> None:
    plan = plan or Plan.load(_plan_path(a.plan))
    path = a.project or plan.concert_dir / f"{plan.concert['name']}.concertcut.kdenlive"
    if path.exists():  # keep the previous version, it may contain changes made in Kdenlive
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        log.info("previous project kept as %s", backup)
    kdenlive.write(plan, path)
    otio_path = otio_export.write(plan, path.with_suffix(".otio"))
    log.info("wrote %s and %s", path, otio_path)


def cmd_programme(a) -> None:
    pr = prog.load(a.file)
    if pr.concert_title:
        print(f"# {pr.concert_title}" + (f" - {pr.performer}" if pr.performer else ""))
    for it in pr.items:
        flag = " *" if it.premiere else ""
        print(f"{it.number:02d} {it.composer_or_folk} - {it.title}{flag}")
        print(f"     title: {it.title_caps}" + (f" | years: {it.years}" if it.years else ""))
        for n in it.notes:
            print(f"     note:  {n}")
    for after in pr.intermission_after:
        print(f"-- intermission after piece {after}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="concertcut", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="sync, detect pieces, name them and write the plan")
    _add_analyze_args(p)

    p = sub.add_parser("render", help="export FLAC/MP4 per piece from a plan")
    p.add_argument("plan", type=Path, help="concertcut.json, or the concert folder containing it")
    _add_render_args(p)

    p = sub.add_parser("kdenlive", help="write the Kdenlive project (+ .otio) from a plan")
    p.add_argument("plan", type=Path, help="concertcut.json, or the concert folder containing it")
    _add_project_args(p)

    p = sub.add_parser("run", help="analyze + kdenlive + render")
    _add_analyze_args(p)
    _add_render_args(p)
    _add_project_args(p)

    p = sub.add_parser("programme", help="show how a programme PDF/text is parsed")
    p.add_argument("file", type=Path)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        if a.cmd == "analyze":
            cmd_analyze(a)
        elif a.cmd == "render":
            cmd_render(a)
        elif a.cmd == "kdenlive":
            cmd_project(a)
        elif a.cmd == "programme":
            cmd_programme(a)
        elif a.cmd == "run":
            plan = cmd_analyze(a)
            cmd_project(a, plan)
            cmd_render(a, plan)
    except (MediaError, prog.ProgrammeError, RuntimeError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
