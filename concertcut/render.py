"""Export of each piece to FLAC (recorder audio) and MP4 (camera video + recorder audio)."""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import gain, media
from .plan import Piece, Plan

log = logging.getLogger(__name__)


def _layout(channels: int) -> str:
    return {1: "mono", 2: "stereo"}.get(channels, f"{channels}c")


def _audio_inputs(plan: Plan, piece: Piece, first_index: int) -> tuple[list[str], str]:
    """ffmpeg input args and a filter chain producing [aud] for the piece's recorder audio."""
    a0 = plan.audio[0]
    sr, ch = a0.sample_rate, max(1, a0.channels)
    z0 = plan.audio_time(piece, piece.start)
    args, labels, filt = [], [], []
    for k, (f, inp, dur) in enumerate(media.segments(plan.audio, z0, z0 + piece.duration)):
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


def _audio_post(plan: Plan, piece: Piece, src: str, gain_db: float, out: str, extra: str = "") -> str:
    d = piece.duration
    fi, fo = plan.fade_in(piece), plan.fade_out(piece)
    chain = [f"volume={gain_db:.4f}dB:precision=double"]
    if fi > 0:
        chain.append(f"afade=t=in:st=0:d={fi:.3f}:curve=tri")
    if fo > 0:
        chain.append(f"afade=t=out:st={max(0.0, d - fo):.6f}:d={fo:.3f}:curve=tri")
    if extra:
        chain.append(extra)
    return f"{src}{','.join(chain)}{out}"


def _metadata(plan: Plan, piece: Piece) -> list[str]:
    c = plan.concert
    tags = {
        "title": piece.title,
        "artist": c.get("artist") or piece.composer,
        "album_artist": c.get("artist", ""),
        "composer": piece.composer,
        "album": c.get("album", ""),
        "track": str(piece.number),
        "date": c.get("date", ""),
    }
    out = []
    for k, v in tags.items():
        if v:
            out += ["-metadata", f"{k}={v}"]
    return out


def flac_command(plan: Plan, piece: Piece, out: Path) -> list[str]:
    a0 = plan.audio[0]
    args, filt, cur = _audio_inputs(plan, piece, 0)
    sr = plan.settings.flac_sample_rate
    extra = f"aresample={sr}" if sr and sr != a0.sample_rate else ""
    filt += ";" + _audio_post(plan, piece, cur, plan.gain_db, "[out]", extra)
    # 24-bit unless the source is 16-bit PCM (lossy camera sound has no bit depth: use 24)
    fmt = ["-sample_fmt", "s16"] if a0.bits == 16 else ["-sample_fmt", "s32", "-bits_per_raw_sample", "24"]
    return ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *args, "-filter_complex", filt,
            "-map", "[out]", "-c:a", "flac", *fmt, *_metadata(plan, piece), str(out)]


def mp4_command(plan: Plan, piece: Piece, out: Path) -> list[str]:
    s = plan.settings
    v0 = next(f for f in plan.video if f.video_index >= 0)
    fps = plan.fps
    args, filt, labels = [], [], []
    for k, (f, inp, dur) in enumerate(media.segments(plan.video, piece.start, piece.end)):
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
    a_args, a_filt, cur = _audio_inputs(plan, piece, len(labels))
    g = plan.gain_db
    if s.mp4_peak_dbfs is not None:
        g += s.mp4_peak_dbfs - s.peak_dbfs
    filt.append(a_filt)
    filt.append(_audio_post(plan, piece, cur, g, "[aout]", "aresample=48000"))
    return ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *args, *a_args,
            "-filter_complex", ";".join(filt), "-map", vout, "-map", "[aout]",
            "-t", f"{piece.duration:.6f}",
            "-c:v", "libx264", "-preset", s.x264_preset, "-crf", str(s.x264_crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", s.aac_bitrate, "-movflags", "+faststart",
            *_metadata(plan, piece), str(out)]


def _run(cmd: list[str], out: Path) -> float:
    t = time.monotonic()
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    cmd = cmd[:-1] + [str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {out.name}:\n{e.stderr[-3000:]}") from None
    tmp.replace(out)
    return time.monotonic() - t


MANIFEST = ".concertcut-render.json"


def _fingerprint(plan: Plan, cmd: list[str]) -> str:
    """Everything that defines an output: the ffmpeg command (sources, cut points, gain,
    encoding settings, tags) and the size of every source file."""
    h = hashlib.sha1(json.dumps(cmd[:-1]).encode())
    for f in plan.video + plan.audio:
        h.update(f"{f.path}:{f.size}".encode())
    return h.hexdigest()


def render(plan: Plan, out_dir: str | Path | None = None, only: set[int] | None = None,
           audio: bool = True, video: bool = True, jobs: int = 2, force: bool = False,
           dry_run: bool = False) -> list[Path]:
    """Export the pieces. Files that are already up to date are skipped; files whose piece
    or settings changed since they were written are rendered again (``force``: all)."""
    out_dir = Path(out_dir) if out_dir else plan.concert_dir / "render"
    pieces = plan.enabled_pieces(only)
    if video and not plan.video:
        log.info("no camera files: MP4 export skipped")
        video = False
    if not dry_run:
        if plan.settings.gain_db_auto or plan.settings.gain_db is None:
            log.info("measuring the pieces' peaks for the concert gain")
            gain.update_gain(plan)
        if plan.path:
            plan.save()
    tasks: list[tuple[list[str], Path]] = []
    for p in pieces:
        if audio:
            out = out_dir / "audio" / f"{p.basename}.flac"
            tasks.append((flac_command(plan, p, out), out))
        if video:
            out = out_dir / "video" / f"{p.basename}.mp4"
            tasks.append((mp4_command(plan, p, out), out))
    manifest_path = out_dir / MANIFEST
    manifest: dict[str, str] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    todo = []
    for cmd, out in tasks:
        key, fp = str(out.relative_to(out_dir)), _fingerprint(plan, cmd)
        if out.exists() and not force and manifest.get(key) == fp:
            log.info("up to date: %s", out.name)
            continue
        if dry_run:
            print(shlex.join(cmd))
            continue
        if out.exists():
            log.info("%s: %s", "re-rendering" if not force else "overwriting",
                     out.name if manifest.get(key) else f"{out.name} (not made by this plan)")
        out.parent.mkdir(parents=True, exist_ok=True)
        todo.append((cmd, out, key, fp))
    if not only:
        current = {str(out.relative_to(out_dir)) for _, out in tasks}
        for key in sorted(set(manifest) - current):
            if (out_dir / key).exists():
                log.warning("no longer in the plan (delete it if not needed): %s", out_dir / key)
    done: list[Path] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(_run, cmd, out): (out, key, fp) for cmd, out, key, fp in todo}
        for fut in as_completed(futs):
            out, key, fp = futs[fut]
            try:
                secs = fut.result()
                log.info("wrote %s (%.0f s)", out, secs)
                done.append(out)
                manifest[key] = fp
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            except RuntimeError as e:
                log.error("%s", e)
                errors.append(str(out))
    if errors:
        raise RuntimeError(f"{len(errors)} export(s) failed: {', '.join(errors)}")
    return done
