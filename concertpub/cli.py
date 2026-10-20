"""Command line: concertpub analyze|render|kdenlive|run|programme."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import kdenlive, programme as prog
from .media import MediaError
from .config import Config

CONFIG_NAME = "concertpub.json"

log = logging.getLogger("concertpub")


def _only(s: str | None) -> set[int] | None:
    return {int(x) for x in s.split(",")} if s else None


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("concert_dir", type=Path, help="folder with camera files, recorder WAVs and programme PDF")
    p.add_argument("--config", type=Path, help="config file to write (default: <concert_dir>/concertpub.json)")
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
    _add_project_args(p)  # where 'k' in the matching question (and 'run') writes the project


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", type=Path, help="output folder (default: <concert_dir>/render)")
    p.add_argument("--only", help="comma separated piece numbers, e.g. 2,5")
    p.add_argument("--audio-only", action="store_true")
    p.add_argument("--video-only", action="store_true")
    p.add_argument("-j", "--jobs", type=int, default=2, help="parallel ffmpeg processes")
    p.add_argument("--crf", type=int, help="x264 quality, lower = better and bigger (default 22)")
    p.add_argument("--preset", help="x264 preset (default from config: slow)")
    p.add_argument("--flac-rate", type=int,
                   help="FLAC sample rate (default: 44100, CD quality, or the recording's rate if lower)")
    p.add_argument("--dry-run", action="store_true", help="print ffmpeg commands only")
    p.add_argument("--force", action="store_true",
                   help="render all files again (default: only new or changed pieces)")


def _add_project_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", type=Path,
                   help="Kdenlive project to write (default: <concert_dir>/<name>.concertpub.kdenlive)")


def _config_path(p: Path) -> Path:
    """A config file, or a concert folder containing concertpub.json."""
    return p / CONFIG_NAME if p.is_dir() else p


def cmd_analyze(a) -> Config:
    from .analyze import analyze

    config_path = a.config or a.concert_dir / CONFIG_NAME
    settings = {k: v for k, v in (("peak_dbfs", a.peak), ("fade_in_s", a.fade_in),
                                  ("fade_out_s", a.fade_out)) if v is not None}
    ask = not a.yes and sys.stdin.isatty()
    if not ask and not a.yes:
        log.info("not a terminal: accepting the programme and the matching without asking")
    from . import interactive

    return analyze(a.concert_dir, config_path, a.programme, a.video, a.audio, a.model_dir,
                   a.artist, settings, skip=_only(a.skip),
                   confirm_programme=interactive.confirm_programme if ask else None,
                   confirm_matching=interactive.confirm_matching if ask else None,
                   enter_programme=interactive.enter_programme if ask else None, project_path=a.project)


def cmd_render(a, config: Config | None = None) -> None:
    from .render import render

    config = config or Config.load(_config_path(a.config))
    s = config.settings
    if a.crf is not None:
        s.x264_crf = a.crf
    if a.preset:
        s.x264_preset = a.preset
    if a.flac_rate:
        s.flac_sample_rate = a.flac_rate
    render(config, a.out_dir, _only(a.only), audio=not a.video_only, video=not a.audio_only,
           jobs=a.jobs, force=a.force, dry_run=a.dry_run)


def cmd_project(a, config: Config | None = None) -> None:
    config = config or Config.load(_config_path(a.config))
    path, otio_path = kdenlive.write_project(config, a.project)
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
    ap = argparse.ArgumentParser(
        prog="concertpub", description="Cut a concert recording into one file per piece.",
        epilog="concertpub <folder> is short for 'concertpub run <folder>': analyze (asks two "
               "questions), write the Kdenlive project, render. Use 'concertpub <command> --help' "
               "for the options of each command.")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="sync, detect pieces, name them and write the config")
    _add_analyze_args(p)

    p = sub.add_parser("render", help="export FLAC/MP4 per piece from a config")
    p.add_argument("config", type=Path, help="concertpub.json, or the concert folder containing it")
    _add_render_args(p)

    p = sub.add_parser("kdenlive", help="write the Kdenlive project (+ .otio) from a config")
    p.add_argument("config", type=Path, help="concertpub.json, or the concert folder containing it")
    _add_project_args(p)

    p = sub.add_parser("run", help="analyze + kdenlive + render")
    _add_analyze_args(p)
    _add_render_args(p)

    p = sub.add_parser("programme", help="show how a programme PDF/text is parsed")
    p.add_argument("file", type=Path)

    argv = list(sys.argv[1:] if argv is None else argv)
    # "concertpub <folder>" is short for "concertpub run <folder>": the whole flow
    first = next((i for i, x in enumerate(argv) if not x.startswith("-")), None)
    if first is not None and argv[first] not in sub.choices:
        argv.insert(first, "run")
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
            config = cmd_analyze(a)
            cmd_project(a, config)
            cmd_render(a, config)
    except (MediaError, prog.ProgrammeError, RuntimeError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
