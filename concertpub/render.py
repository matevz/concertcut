"""Export of each piece to FLAC (recorder audio) and MP4 (camera video + recorder audio)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import gain, media
from .config import Config, Piece

log = logging.getLogger(__name__)


def _layout(channels: int) -> str:
    return {1: "mono", 2: "stereo"}.get(channels, f"{channels}c")


def _audio_inputs(config: Config, piece: Piece, first_index: int,
                  prepared: Path | None = None) -> tuple[list[str], str]:
    """ffmpeg input args and a filter chain producing [aud] for the piece's recorder audio.

    ``prepared``: the piece's audio already decoded and cleaned (buzz removed) as raw float32
    at the recording's own sample rate; used instead of the recorder files.
    """
    a0 = config.audio[0]
    if prepared is not None:
        args = ["-f", "f32le", "-ar", str(a0.sample_rate), "-ac", str(max(1, a0.channels)), "-i", str(prepared)]
        return args, "", f"[{first_index}:a]"
    sr, ch = a0.sample_rate, max(1, a0.channels)
    z0 = config.audio_time(piece, piece.start)
    args, labels, filt = [], [], []
    for k, (f, inp, dur) in enumerate(media.segments(config.audio, z0, z0 + piece.duration)):
        i = first_index + k
        if f is None:
            args += ["-f", "lavfi", "-t", f"{dur:.6f}", "-i", f"anullsrc=r={sr}:cl={_layout(ch)}"]
            src = f"[{i}:a]"
        else:
            if inp > 0:
                args += ["-ss", f"{inp:.6f}"]
            args += ["-t", f"{dur:.6f}", "-i", f.path]
            src = f"[{i}:{f.audio_index}]"
        filt.append(f"{src}aformat=sample_rates={sr}:channel_layouts={_layout(ch)}[a{k}]")
        labels.append(f"[a{k}]")
    if len(labels) > 1:
        filt.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[acat]")
        cur = "[acat]"
    else:
        cur = labels[0]
    return args, ";".join(filt), cur


def _audio_post(config: Config, piece: Piece, src: str, gain_db: float, out: str, extra: str = "") -> str:
    d = piece.duration
    fi, fo = config.fade_in(piece), config.fade_out(piece)
    chain = [f"volume={gain_db:.4f}dB:precision=double"]
    if fi > 0:
        chain.append(f"afade=t=in:st=0:d={fi:.3f}:curve=tri")
    if fo > 0:
        chain.append(f"afade=t=out:st={max(0.0, d - fo):.6f}:d={fo:.3f}:curve=tri")
    if extra:
        chain.append(extra)
    return f"{src}{','.join(chain)}{out}"


def _metadata(config: Config, piece: Piece) -> list[str]:
    c = config.concert
    tags = {
        "title": piece.title,
        "artist": c.get("artist") or piece.composer,
        "album_artist": c.get("artist", ""),
        "composer": piece.composer,
        "album": c.get("album", ""),
        "track": str(piece.number),
        "date": c.get("date", ""),
    }
    # extra tags added by hand in concertpub.json, e.g. "concert": {"tags": {"genre": "Choral"}}
    tags.update({str(k): str(v) for k, v in (c.get("tags") or {}).items()})
    out = []
    for k, v in tags.items():
        if v:
            out += ["-metadata", f"{k}={unicodedata.normalize('NFC', v)}"]
    return out


def _join(*parts: str) -> str:
    return ";".join(p for p in parts if p)


def flac_command(config: Config, piece: Piece, out: Path, prepared: Path | None = None) -> list[str]:
    # CD quality (44.1 kHz / 16-bit), or the recording's own quality if that is lower:
    # soxr resampling, triangular dither to 16 bits, maximum lossless compression
    args, filt, cur = _audio_inputs(config, piece, 0, prepared)
    convert = (f"aresample={config.flac_rate}:resampler=soxr:precision=28:osf=s16:"
               f"dither_method=triangular")
    filt = _join(filt, _audio_post(config, piece, cur, config.gain_db, "[out]", convert))
    return ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *args, "-filter_complex", filt,
            "-map", "[out]", "-c:a", "flac", "-compression_level", "12", "-sample_fmt", "s16",
            *_metadata(config, piece), str(out)]


def mp4_command(config: Config, piece: Piece, out: Path, prepared: Path | None = None) -> list[str]:
    s = config.settings
    v0 = next(f for f in config.video if f.video_index >= 0)
    fps = config.fps
    args, filt, labels = [], [], []
    for k, (f, inp, dur) in enumerate(media.segments(config.video, piece.start, piece.end)):
        if f is None:
            args += ["-f", "lavfi", "-t", f"{dur:.6f}", "-i",
                     f"color=c=black:s={v0.width}x{v0.height}:r={fps.numerator}/{fps.denominator}"]
            src = f"[{k}:v]"
        else:
            if inp > 0:
                args += ["-ss", f"{inp:.6f}"]
            args += ["-t", f"{dur:.6f}", "-i", f.path]
            src = f"[{k}:{f.video_index}]"
        filt.append(f"{src}fps={fps.numerator}/{fps.denominator},setsar={v0.sar.numerator}/{v0.sar.denominator},"
                    f"format=yuv420p,setpts=PTS-STARTPTS[v{k}]")
        labels.append(f"[v{k}]")
    if len(labels) > 1:
        filt.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vout]")
        vout = "[vout]"
    else:
        vout = labels[0]
    a_args, a_filt, cur = _audio_inputs(config, piece, len(labels), prepared)
    g = config.gain_db
    if s.mp4_peak_dbfs is not None:
        g += s.mp4_peak_dbfs - s.peak_dbfs
    if a_filt:
        filt.append(a_filt)
    # CD sample rate (or the recording's, if lower) like the FLAC; AAC needs no dither
    filt.append(_audio_post(config, piece, cur, g, "[aout]",
                            f"aresample={config.flac_rate}:resampler=soxr:precision=28"))
    return ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *args, *a_args,
            "-filter_complex", ";".join(filt), "-map", vout, "-map", "[aout]",
            "-t", f"{piece.duration:.6f}",
            "-c:v", "libx264", "-preset", s.x264_preset, "-crf", str(s.x264_crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", s.aac_bitrate, "-movflags", "+faststart",
            *_metadata(config, piece), str(out)]


def _run(cmd: list[str], out: Path, threads: int | None = None) -> float:
    log.info("writing %s", out)
    t = time.monotonic()
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    # encoder threads are set here, not in cmd, so they do not count as a change of the output
    cmd = cmd[:-1] + (["-threads", str(threads)] if threads else []) + [str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {out.name}:\n{e.stderr[-3000:]}") from None
    tmp.replace(out)
    return time.monotonic() - t


MANIFEST = ".concertpub-render.json"


def _fingerprint(config: Config, cmd: list[str], piece: Piece | None = None) -> str:
    """Everything that defines an output: the ffmpeg command (sources, cut points, gain,
    encoding settings, tags), the size of every source file, and - when the audio was
    prepared in Python (buzz removal) - the piece's cut and the buzz being removed."""
    h = hashlib.sha1(json.dumps(cmd[:-1]).encode())
    if piece is not None and config.active_hum:
        h.update(json.dumps([piece.start, piece.end, piece.audio_offset_ms, config.active_hum]).encode())
    for f in config.video + config.audio:
        h.update(f"{f.path}:{f.size}".encode())
    return h.hexdigest()


def render(config: Config, out_dir: str | Path | None = None, only: set[int] | None = None,
           audio: bool = True, video: bool = True, jobs: int = 2, force: bool = False,
           dry_run: bool = False) -> list[Path]:
    """Export the pieces. Files that are already up to date are skipped; files whose piece
    or settings changed since they were written are rendered again (``force``: all)."""
    out_dir = Path(out_dir) if out_dir else config.concert_dir / "render"
    pieces = config.enabled_pieces(only)
    if video and not config.video:
        log.info("no camera files: MP4 export skipped")
        video = False
    tmp = out_dir / ".concertpub-tmp"
    prepared: dict[int, Path] = {}
    if config.active_hum:  # buzz removal: each piece to render is cleaned once, for FLAC and MP4
        prepared = {p.number: tmp / f"{p.number:02d}.f32" for p in pieces}
    if not dry_run:
        if config.settings.gain_db_auto or config.settings.gain_db is None:
            # measured on the recording as it is (with a buzz, if any): quick, and removing
            # the buzz only lowers the peaks, so the cleaned files cannot clip
            log.info("measuring the pieces' peaks for the concert gain")
            gain.update_gain(config)
        if config.path:
            config.save()
    tasks: list[tuple[list[str], Path, Piece]] = []  # all FLAC files first, then the (slow) MP4 files

    if audio:
        for p in pieces:
            out = out_dir / "audio" / f"{p.basename}.flac"
            tasks.append((flac_command(config, p, out, prepared.get(p.number)), out, p))
    if video:
        for p in pieces:
            out = out_dir / "video" / f"{p.basename}.mp4"
            tasks.append((mp4_command(config, p, out, prepared.get(p.number)), out, p))
    manifest_path = out_dir / MANIFEST
    manifest: dict[str, str] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    todo = []
    for cmd, out, p in tasks:
        key, fp = str(out.relative_to(out_dir)), _fingerprint(config, cmd, p)
        if out.exists() and not force and manifest.get(key) == fp:
            log.info("up to date: %s", out.name)
            continue
        if dry_run:
            print(shlex.join(cmd))
            continue
        if out.exists():
            log.info("%s: %s", "re-rendering" if not force else "overwriting",
                     out.name if manifest.get(key) else f"{out.name} (not made by concertpub)")
        out.parent.mkdir(parents=True, exist_ok=True)
        todo.append((cmd, out, key, fp, p))
    if prepared and todo:
        need = list({id(t[4]): t[4] for t in todo}.values())  # only pieces that are rendered now
        tmp.mkdir(parents=True, exist_ok=True)
        cpus = os.cpu_count() or 1
        # a few pieces at a time (decoding one overlaps with filtering another), the rest of
        # the CPU threads filter the blocks of each piece in parallel
        at_once = max(1, min(len(need), cpus // 4))
        threads = max(1, cpus // at_once)
        log.info("removing the buzz from %d piece(s): %d at a time, %d thread(s) each",
                 len(need), at_once, threads)

        def prepare(p):
            x, _ = gain.piece_audio(config, p, threads)  # buzz removed at the recording's own rate
            x.astype(np.float32).tofile(prepared[p.number])

        with ThreadPoolExecutor(max_workers=at_once) as ex:
            list(ex.map(prepare, need))
    if not only:
        current = {str(out.relative_to(out_dir)) for _, out, _ in tasks}
        for key in sorted(set(manifest) - current):
            if (out_dir / key).exists():
                log.warning("no longer in the config (delete it if not needed): %s", out_dir / key)
    done: list[Path] = []
    errors: list[str] = []
    for kind, ext in (("FLAC", ".flac"), ("MP4", ".mp4")):  # audio phase finishes before video starts
        phase = [t for t in todo if t[1].suffix == ext]
        if not phase:
            continue
        parallel = max(1, min(jobs, len(phase)))
        if kind == "MP4":  # share the CPU between the x264 encoders running at the same time
            threads = max(1, (os.cpu_count() or 1) // parallel)
            log.info("rendering %d %s file(s): %d at a time, %d encoder thread(s) each (%d CPU threads)",
                     len(phase), kind, parallel, threads, os.cpu_count() or 1)
        else:  # the FLAC encoder is single-threaded
            threads = None
            log.info("rendering %d %s file(s): %d at a time, 1 thread each", len(phase), kind, parallel)
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(_run, cmd, out, threads): (out, key, fp) for cmd, out, key, fp, _ in phase}
            for fut in as_completed(futs):
                out, key, fp = futs[fut]
                try:
                    fut.result()
                    done.append(out)
                    manifest[key] = fp
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
                except RuntimeError as e:
                    log.error("%s", e)
                    errors.append(str(out))
    shutil.rmtree(tmp, ignore_errors=True)
    if errors:
        raise RuntimeError(f"{len(errors)} export(s) failed: {', '.join(errors)}")
    return done
