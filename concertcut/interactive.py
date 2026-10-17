"""Terminal prompts used by the CLI to confirm the performed programme and the matching
of detected segments to pieces. The analysis core only sees the callbacks."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from . import programme as prog
from .analyze import Match, RereadMedia, RereadProgramme
from .timecode import to_tc


class Aborted(SystemExit):
    pass


def _ask(prompt: str, input_fn=input) -> str:
    try:
        return input_fn(prompt).strip()
    except EOFError:
        raise Aborted("aborted (no input)") from None


def _numbers(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split()]


def _tc(t: float) -> str:
    return to_tc(t)[:-2]  # HH:MM:SS.s


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


def show_programme(pr: prog.Programme, out=print) -> None:
    out(f"\nProgramme{' - ' + pr.concert_title if pr.concert_title else ''}:")
    for it in pr.items:
        out(f"  {it.number:2d}. {it.composer_or_folk} - {it.title}")


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
        for line in path.read_text(encoding="utf-8").splitlines():
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


def show_matching(matches: list[Match], pr: prog.Programme | None, out=print) -> None:
    out("\nDetected pieces (timeline):")
    out(f"  {'#':>2}  {'start':>10}  {'end':>10}  {'len':>5}  {'ends at':<10}  piece")
    for k, m in enumerate(matches, 1):
        r = m.region
        name = (f"[{m.item.number:2d}] {m.composer} - {m.title}" if m.item else
                f"[ +] {m.title} (not in programme)")
        cue = " (after pitch)" if "pitch cue" in r.start_reason else ""
        out(f"  {k:2d}  {_tc(m.start):>10}  {_tc(m.end):>10}  {_dur(m.end - m.start):>5}  "
            f"{r.end_reason:<10}  {name}{cue}")
    if pr:
        used = {m.item.number for m in matches if m.item}
        missing = [it for it in pr.items if it.number not in used]
        for it in missing:
            out(f"  not matched: [{it.number:2d}] {it.composer_or_folk} - {it.title}")


def confirm_matching(matches: list[Match], pr: prog.Programme | None, input_fn=input,
                     out=print) -> list[Match]:
    by_no = {it.number: it for it in (pr.items if pr else [])}
    while True:
        show_matching(matches, pr, out)
        a = _ask("Is this matching correct? [Y]es / [m]erge segments / [d]rop segment / "
                 "[a]ssign pieces / [r]e-read the video and audio files / [q]uit: ", input_fn).lower()
        if a in ("", "y", "yes"):
            return matches
        if a.startswith("q"):
            raise Aborted("aborted")
        if a in ("r", "re-read", "reread"):
            raise RereadMedia()
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
