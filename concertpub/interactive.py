"""Terminal prompts used by the CLI to confirm the performed programme and the matching
of detected segments to pieces. The analysis core only sees the callbacks."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import noise, programme as prog
from .analyze import BackToProgramme, Match, PreviewKdenlive, RereadMedia, RereadProgramme, concert_gain


class Aborted(SystemExit):
    pass


def _ask(prompt: str, input_fn=input) -> str:
    try:
        return input_fn(prompt).strip()
    except EOFError:
        raise Aborted("aborted (no input)") from None


def _numbers(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split()]


def _dur(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def enter_programme(reason: str, input_fn=input, out=print) -> prog.Programme | None:
    """Ask for the performed pieces when no programme could be read."""
    out(f"\nThe programme could not be determined: {reason}.")
    out("Type the performed pieces, one per line, as 'NN Composer - Title'")
    out("(e.g. '01 Andrej Misson - Mlinar in smrt'). Finish with an empty line.")
    out("Press Enter right away to continue without names.")
    while True:
        lines = []
        while True:
            line = _ask(f"{len(lines) + 1:3d}> ", input_fn)
            if not line:
                break
            lines.append(line)
        if not lines:
            return None
        try:
            return prog.parse_entered(lines)
        except prog.ProgrammeError as e:
            out(f"  {e} - please type the list again")


def numbering_notes(numbers: list[int]) -> list[str]:
    """What is unusual about programme numbers that are not simply 1, 2, 3, ..."""
    notes = []
    dupes = sorted({n for n in numbers if numbers.count(n) > 1})
    if dupes:
        notes.append("used more than once: " + ", ".join(map(str, dupes)))
    present = set(numbers)
    missing = [n for n in range(1, max(numbers, default=0) + 1) if n not in present]
    if missing:
        notes.append("missing: " + ", ".join(map(str, missing)) + " (not performed?)")
    swaps = [f"{a} before {b}" for a, b in zip(numbers, numbers[1:]) if a > b]
    if swaps:
        notes.append("not in order: " + ", ".join(swaps))
    return notes


def show_programme(pr: prog.Programme, out=print) -> None:
    out(f"\nProgramme{' - ' + pr.concert_title if pr.concert_title else ''}:")
    for it in pr.items:
        out(f"  {it.number:2d}. {it.composer_or_folk} - {it.title}")
    notes = numbering_notes([it.number for it in pr.items])
    if notes:
        out("")
        out("  WARNING: the numbering is not 1, 2, 3, ... - " + "; ".join(notes) + ".")
        out("  Tracks are numbered in the order performed; check that this is right.")
        out("")


def _edit_programme(pr: prog.Programme, editor: str | None = None) -> prog.Programme:
    """Edit the list as text: one piece per line, 'number | composer | title'."""
    lines = ["# One performed piece per line, in performance order: number | composer | title",
             "# Delete lines of pieces that were not performed, add lines for extra pieces.", ""]
    lines += [f"{it.number} | {it.composer} | {it.title}" for it in pr.items]
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        path = Path(f.name)
    try:
        cmd = shlex.split(editor or os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano")
        subprocess.run(cmd + [str(path)], check=False)
        items = []
        by_no = {it.number: it for it in pr.items}
        for line in prog.clean_text(path.read_text(encoding="utf-8")).splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [x.strip() for x in line.split("|")]
            if len(parts) != 3:
                print(f"  ignored line (need 'number | composer | title'): {line}")
                continue
            no = int(parts[0]) if parts[0].isdigit() else max([0] + [i.number for i in items]) + 1
            base = by_no.get(no)
            it = prog.ProgrammeItem(number=no, composer=parts[1], title_caps=base.title_caps if base else "",
                                    subtitle=parts[2], notes=base.notes if base else [],
                                    years=base.years if base else "")
            items.append(it)
        pr.items = items
        return pr
    finally:
        path.unlink(missing_ok=True)


def confirm_programme(pr: prog.Programme, input_fn=input, out=print) -> prog.Programme:
    while True:
        show_programme(pr, out)
        a = _ask("Was it performed exactly like this? [Y]es / [s]kip pieces / [o]rder / [e]dit / "
                 "[r]e-read the PDF / [q]uit: ", input_fn).lower()
        if a in ("", "y", "yes"):
            return pr
        if a.startswith("q"):
            raise Aborted("aborted")
        if a in ("r", "re-read", "reread"):
            raise RereadProgramme()
        nos = {it.number for it in pr.items}
        if a.startswith("s"):
            rest = a[1:].strip() or _ask("Numbers of pieces that were NOT performed (e.g. 6 or 6,8): ", input_fn)
            try:
                drop = set(_numbers(rest))
            except ValueError:
                out("  please enter numbers")
                continue
            if drop - nos:
                out(f"  unknown number(s): {sorted(drop - nos)}")
                continue
            pr.items = [it for it in pr.items if it.number not in drop]
        elif a.startswith("o"):
            rest = a[1:].strip() or _ask("Performed order as programme numbers (e.g. 1 2 4 3 5): ", input_fn)
            try:
                order = _numbers(rest)
            except ValueError:
                out("  please enter numbers")
                continue
            if set(order) - nos or len(set(order)) != len(order):
                out("  every number must be in the programme and appear once")
                continue
            by_no = {it.number: it for it in pr.items}
            pr.items = [by_no[n] for n in order]
        elif a.startswith("e"):
            pr = _edit_programme(pr)
        else:
            out("  ?")


def _clock(t: float) -> str:
    """Position in a file: M:SS.s, or H:MM:SS.s from one hour on."""
    tenths = int(round(t * 10))
    h, rest = divmod(tenths, 36000)
    m, rest = divmod(rest, 600)
    sec = rest / 10
    return f"{h}:{m:02d}:{sec:04.1f}" if h else f"{m}:{sec:04.1f}"


def _span_text(spans: list[dict]) -> str:
    if not spans:
        return "-"
    first, last = spans[0], spans[-1]
    if len(spans) == 1:
        return f"{first['file']}  {_clock(first['from'])} - {_clock(first['to'])}"
    return f"{first['file']} {_clock(first['from'])} - {last['file']} {_clock(last['to'])}"


def show_matching(matches: list[Match], pr: prog.Programme | None, out=print) -> None:
    stale = [m for m in matches if m.peak_stale]
    if stale:  # can take a while (full-quality decoding): say so, and measure in parallel
        out(f"\nmeasuring the peaks of {len(stale)} piece(s) ...")
        t = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, min(len(stale), os.cpu_count() or 1))) as ex:
            list(ex.map(lambda m: m.peak(), stale))
        out(f"done ({time.monotonic() - t:.0f} s)")
    gain, loudest, by_user = concert_gain(matches)
    out("\nDetected pieces (positions in the source files):")
    out(f"  {'#':>2}  {'len':>5}  {'lead-in':>7}  {'peak':>9}  {'ends at':<10}  piece")
    for k, m in enumerate(matches, 1):
        r = m.region
        name = (f"[{m.item.number:2d}] {m.composer} - {m.title}" if m.item else
                f"[ +] {m.title} (not in programme)")
        lead = "-" if m.lead_in is None else f"{m.lead_in:.1f} s"
        pk = m.peak() if m.sources else None
        peak = "-" if pk is None else f"{pk:.1f} dB" + ("*" if m is loudest else " ")
        out(f"  {k:2d}  {_dur(m.end - m.start):>5}  {lead:>7}  {peak:>9}  {r.end_reason:<10}  {name}")
        for kind, spans in m.spans().items():
            out(f"  {'':2}  {'':5}  {'':7}  {'':9}  {kind:<10}  {_span_text(spans)}")
    if pr:
        used = {m.item.number for m in matches if m.item}
        missing = [it for it in pr.items if it.number not in used]
        for it in missing:
            out(f"  not matched: [{it.number:2d}] {it.composer_or_folk} - {it.title}")
    smap = next((m.sources for m in matches if m.sources), None)
    buzz_line = None
    if smap is not None and smap.hum:
        out("")
        out(f"  WARNING: a constant buzz was found in the recording: {noise.describe(smap.hum)}.")
        buzz_line = ("  Buzz removal: ON - it is filtered out when rendering ('h' to turn it off)."
                     if smap.hum_removal else "  Buzz removal: off - press 'h' to filter it out when rendering.")
    if gain is None and buzz_line:
        out(buzz_line)
    if gain is not None:
        k = matches.index(loudest) + 1 if loudest else None
        if by_user:
            out(f"\n  Concert gain: {gain:+.2f} dB (set by you; 'g auto' for automatic)")
            if buzz_line:
                out(buzz_line)
            clipping = [str(i) for i, m in enumerate(matches, 1)
                        if m.enabled and m.peak() is not None and m.peak() + gain > 0.0]
            if clipping:
                out("")
                out(f"  WARNING: with this gain, piece(s) {', '.join(clipping)} will clip.")
                out("")
        else:
            target = loudest.sources.target_dbfs
            out(f"\n  Concert gain: {gain:+.2f} dB (automatic: the loudest piece, {k}*, peaks at "
                f"{target:.1f} dBFS)")
            if buzz_line:
                out(buzz_line)
    out("")  # an empty line before the question


def confirm_matching(matches: list[Match], pr: prog.Programme | None, input_fn=input,
                     out=print) -> list[Match]:
    by_no = {it.number: it for it in (pr.items if pr else [])}
    while True:
        show_matching(matches, pr, out)
        a = _ask("Is this matching correct? [Y]es / [m]erge segments / [d]rop segment / "
                 "[a]ssign pieces / [l]ead-in / [g]ain / [h]um/buzz removal / [k]denlive project / "
                 "[b]ack to the programme / "
                 "[r]e-read the video and audio files / [q]uit: ", input_fn).lower()
        if a in ("", "y", "yes"):
            return matches
        if a.startswith("q"):
            raise Aborted("aborted")
        if a in ("r", "re-read", "reread"):
            raise RereadMedia()
        if a in ("b", "back"):
            raise BackToProgramme()
        if a in ("k", "kdenlive"):
            raise PreviewKdenlive()
        try:
            if a.startswith("m"):
                rest = a[1:].strip() or _ask("Merge which two adjacent segments (e.g. 6 7): ", input_fn)
                i, j = sorted(_numbers(rest))
                if j != i + 1 or not 1 <= i < len(matches):
                    out("  give two adjacent segment numbers")
                    continue
                first, second = matches[i - 1], matches[j - 1]
                first.region.end_reason = second.region.end_reason
                matches[i - 1:j] = [replace(first, end=second.end, item=first.item or second.item,
                                            offset=None, name=None)]
            elif a.startswith("d"):
                rest = a[1:].strip() or _ask("Drop which segment (e.g. 10): ", input_fn)
                (i,) = _numbers(rest)
                if not 1 <= i <= len(matches):
                    out("  no such segment")
                    continue
                del matches[i - 1]
            elif a in ("h", "hum", "buzz"):
                smaps = {id(m.sources): m.sources for m in matches if m.sources}
                hum = next((sm.hum for sm in smaps.values() if sm.hum), None)
                if not hum:
                    out("  no constant buzz was found in this recording")
                    continue
                state = not any(sm.hum_removal for sm in smaps.values())
                for sm in smaps.values():
                    sm.hum_removal = state
                out(f"  buzz removal {'on' if state else 'off'}")
            elif a.startswith("g"):
                rest = a[1:].strip() or _ask("Concert gain in dB (e.g. 8 or 6.5), or 'auto': ", input_fn)
                smaps = {id(m.sources): m.sources for m in matches if m.sources}
                if not smaps:
                    out("  no recording to measure")
                    continue
                value = None if rest in ("auto", "a") else float(rest.replace("db", "").strip())
                for smap in smaps.values():
                    smap.gain_db = value
            elif a.startswith("l"):
                rest = a[1:].strip() or _ask("Lead-in before the first note: segment and seconds "
                                             "(e.g. 3 2.5; * 3 for all): ", input_fn)
                parts = rest.replace(",", " ").split()
                if len(parts) != 2:
                    out("  give a segment number (or *) and seconds")
                    continue
                secs = float(parts[1])
                if secs < 0:
                    out("  the lead-in cannot be negative")
                    continue
                targets = range(len(matches)) if parts[0] == "*" else [int(parts[0]) - 1]
                for i in targets:
                    if not 0 <= i < len(matches):
                        raise ValueError("no such segment")
                    m = matches[i]
                    if m.first_note is None:
                        out(f"  segment {i + 1}: the first note is not known, edit start in concertpub.json")
                        continue
                    start = m.first_note - secs
                    if i > 0 and start < matches[i - 1].end:
                        start = matches[i - 1].end
                        out(f"  segment {i + 1}: lead-in limited to {m.first_note - start:.1f} s "
                            "(the previous piece ends there)")
                    matches[i] = replace(m, start=start)
            elif a.startswith("a"):
                rest = a[1:].strip() or _ask(
                    f"Programme number for each of the {len(matches)} segments, + for a piece not in the "
                    "programme (encore), - to drop (e.g. 1 2 3 4 5 7 8 9 +): ", input_fn)
                toks = rest.replace(",", " ").split()
                if len(toks) != len(matches):
                    out(f"  need {len(matches)} entries, got {len(toks)}")
                    continue
                new = []
                for t, m in zip(toks, matches):
                    if t == "-":
                        continue
                    if t == "+":
                        new.append(replace(m, item=None, name=None))
                    elif t.isdigit() and int(t) in by_no:
                        new.append(replace(m, item=by_no[int(t)], name=None))
                    else:
                        raise ValueError(f"unknown programme number {t}")
                matches = new
            else:
                out("  ?")
        except ValueError as e:
            out(f"  {e}")
