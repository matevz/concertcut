"""Rule-based parsing of concert programmes (PDF or plain text).

Expected shape of each item (blank lines between items are optional)::

    Andrej Misson (1960):            <- composer line, years in parentheses
    ŠPANCIRA V MALNU MALNAR MLAD     <- title in capitals
    Mlinar in smrt                   <- optional subtitle (used as the name)
    Priredba ljudske pesmi iz ...    <- optional notes

A capitals title without a composer line before it is a folk song.
"""

from __future__ import annotations

import json
import re
import unicodedata
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_FOLK_COMPOSER = "Traditional"

_YEARS = r"\(\s*(?:r\.\s*|\*\s*)?(\d{4}(?:\s*[-–—]\s*(?:\d{4})?)?)\s*\)"
COMPOSER_RE = re.compile(rf"^(?P<name>[^\W\d_][^()\d]*?)\s*{_YEARS}\s*:?\s*(?P<rest>.*)$")
# Words recognised when *reading* a programme. Programmes may be printed in Slovenian or
# English, so both are listed; everything concertpub writes or asks is in English.
SECTION_RE = re.compile(r"^\s*(program|programme|spored|sporedi)\s*:?\s*$", re.I)
INTERMISSION_RE = re.compile(r"^\s*[-–—*]*\s*(odmor|pavza|intermission|pause|pausa)\b", re.I)
NOTE_PREFIXES = ("priredba", "prir.", "ljudsk", "besedilo", "po ", "glasba", "solist", "orgle", "klavir",
                 "harmonizacija",  # Slovenian
                 "arr.", "arr ", "arranged", "text", "words", "lyrics", "music", "traditional", "folk",
                 "solo", "organ", "piano", "after ")  # English
FOOTNOTE_RE = re.compile(r"^\s*\*+\s*\S")


@dataclass
class ProgrammeItem:
    number: int
    composer: str = ""
    years: str = ""
    title_caps: str = ""
    subtitle: str = ""
    notes: list[str] = field(default_factory=list)
    premiere: bool = False

    @property
    def title(self) -> str:
        return self.subtitle or sentence_case(self.title_caps)

    @property
    def composer_or_folk(self) -> str:
        return self.composer or DEFAULT_FOLK_COMPOSER

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProgrammeItem":
        d = dict(d)
        if isinstance(d.get("notes"), str):
            d["notes"] = [d["notes"]] if d["notes"] else []
        d.pop("note", None)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Programme:
    items: list[ProgrammeItem]
    header: list[str] = field(default_factory=list)  # lines before the PROGRAM heading
    legend: list[str] = field(default_factory=list)  # footnotes like "*krstna izvedba"
    intermission_after: list[int] = field(default_factory=list)
    source: str = ""

    @property
    def concert_title(self) -> str:
        for line in self.header:
            if line.strip():
                return sentence_case(line.strip()) if is_caps(line) else line.strip()
        return ""

    @property
    def performer(self) -> str:
        for line in self.header:
            if re.search(r"\b(zbor|choir|chor|ensemble|ansambel|kor)\b", line, re.I):
                return line.strip().rstrip(",")
        return ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "parser": "rules",
            "concert_title": self.concert_title,
            "performer": self.performer,
            "intermission_after": self.intermission_after,
            "items": [i.to_dict() for i in self.items],
        }


# ----------------------------------------------------------------------------- text

_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
              "\ufb05": "st", "\ufb06": "st"}
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"))  # soft hyphen, zero-width


def clean_text(s: str) -> str:
    """Normalise text from PDFs, files and the keyboard, wherever it was made.

    macOS often writes letters like č, š, ž decomposed (c + combining caron, Unicode NFD);
    Linux and Windows use the composed form (NFC). They look the same but are different
    bytes, so file names would differ. Also: PDF ligatures (ﬁ -> fi), soft hyphens and
    zero-width characters are removed, non-breaking spaces become spaces.
    """
    for lig, rep in _LIGATURES.items():
        s = s.replace(lig, rep)
    s = s.translate(_INVISIBLE).replace("\u00a0", " ").replace("\u202f", " ")
    return unicodedata.normalize("NFC", s)



def letters(s: str) -> list[str]:
    return [c for c in s if c.isalpha()]


def is_caps(line: str, min_ratio: float = 0.7, min_letters: int = 3) -> bool:
    ls = letters(line)
    return len(ls) >= min_letters and sum(c.isupper() for c in ls) / len(ls) >= min_ratio


def sentence_case(s: str) -> str:
    """'ŠPANCIRA V MALNU MALNAR MLAD' -> 'Špancira v malnu malnar mlad'."""
    s = re.sub(r"\s+", " ", s.strip())
    if not is_caps(s):
        return s
    low = s.lower()
    for i, c in enumerate(low):
        if c.isalpha():
            return low[:i] + c.upper() + low[i + 1:]
    return low


def clean_title(s: str) -> tuple[str, bool]:
    """Strip footnote asterisks; return (title, had_asterisk)."""
    star = "*" in s
    return re.sub(r"\s+", " ", s.replace("*", "")).strip(" ,;"), star


def is_note(line: str) -> bool:
    s = line.strip()
    if s.startswith("(") and s.endswith(")"):
        return True
    return s.lower().startswith(NOTE_PREFIXES)


def is_prose(line: str) -> bool:
    s = line.strip()
    return len(s) >= 90 or (len(s) >= 60 and len(s.split()) >= 8 and not is_caps(s))


def pdf_to_text(pdf: str | Path) -> str:
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True,
                              text=True, check=True).stdout
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)


def _split_columns(lines: list[str]) -> list[str]:
    """pdftotext -layout may put two columns on one line, separated by wide gaps."""
    out = []
    for line in lines:
        parts = [p for p in re.split(r"\s{6,}", line.strip()) if p]
        out.extend(parts or [""])
    return out


# ----------------------------------------------------------------------------- parse


class ProgrammeError(ValueError):
    pass


def parse_text(text: str, source: str = "") -> Programme:
    text = clean_text(text)
    raw = _split_columns(text.replace("\f", "\n").splitlines())
    start = next((i for i, l in enumerate(raw) if SECTION_RE.match(l)), None)
    header = raw[:start] if start is not None else []
    body = raw[start + 1:] if start is not None else raw

    items: list[ProgrammeItem] = []
    legend: list[str] = []
    intermissions: list[int] = []
    cur: ProgrammeItem | None = None

    def new_item() -> ProgrammeItem:
        nonlocal cur
        cur = ProgrammeItem(number=len(items) + 1)
        items.append(cur)
        return cur

    nonempty = [l.strip() for l in body if l.strip()]
    for k, s in enumerate(nonempty):
        # The programme ends where running prose (e.g. the ensemble's bio) begins.
        if len(s) >= 90 or all(is_prose(x) for x in nonempty[k:k + 3]) and len(nonempty[k:k + 3]) == 3:
            break
        if INTERMISSION_RE.match(s):
            intermissions.append(len(items))
            cur = None
            continue
        if FOOTNOTE_RE.match(s) and not is_caps(s):
            legend.append(s)
            continue
        if cur is not None and cur.title_caps and s.lower().startswith(NOTE_PREFIXES):
            cur.notes.append(s)
            continue
        m = COMPOSER_RE.match(s)
        if m:
            # A composer always starts a new item unless the current one has no title yet
            # (e.g. composer printed before the title), or is still waiting for a title.
            if cur is None or cur.title_caps or cur.composer:
                new_item()
            cur.composer = re.sub(r"\s+", " ", m.group("name")).strip(" ,:")
            cur.years = re.sub(r"\s+", "", m.group(2))
            rest = m.group("rest").strip()
            if rest:
                cur.title_caps, star = clean_title(rest)
                cur.premiere |= star
            continue
        if is_caps(s):
            if cur is None or cur.title_caps:
                new_item()
            cur.title_caps, star = clean_title(s)
            cur.premiere |= star
            continue
        if cur is None:
            continue  # stray text before the first item
        if not cur.title_caps:
            # Normal-case title layout (no capitals) - accept the first plain line as title.
            title, star = clean_title(s)
            cur.title_caps, cur.premiere = title, cur.premiere or star
        elif not cur.subtitle and not cur.notes and not is_note(s):
            cur.subtitle, star = clean_title(s)
            cur.premiere |= star
        else:
            cur.notes.append(s)

    items = [i for i in items if i.title_caps or i.subtitle]
    for n, it in enumerate(items, 1):
        it.number = n
    if not items:
        raise ProgrammeError(
            "No pieces found in the programme. Extracted text was:\n" + text[:3000]
            + "\nUse --programme-text with an edited copy, or --programme-json."
        )
    return Programme(items=items, header=[h for h in header if h.strip()], legend=legend,
                     intermission_after=intermissions, source=source)


ENTERED_RE = re.compile(r"^\s*(?:(\d+)\s*[.):]?\s+)?(.+?)\s*$")


def parse_entered(lines: list[str]) -> Programme:
    """Pieces typed by the user, one per line: ``NN Composer - Title``.

    The number may be left out (then lines are numbered in order); without " - " the line
    is a title without composer. Raises ProgrammeError naming the first bad line.
    """
    items: list[ProgrammeItem] = []
    for line in lines:
        line = clean_text(line)
        if not line.strip():
            continue
        m = ENTERED_RE.match(line)
        no, rest = m.group(1), m.group(2)
        composer, sep, title = rest.partition(" - ")
        if not sep:
            composer, title = "", rest
        composer, title = composer.strip(), title.strip()
        if not title:
            raise ProgrammeError(f"no title in line: {line!r}")
        number = int(no) if no else (items[-1].number + 1 if items else 1)
        if any(i.number == number for i in items):
            raise ProgrammeError(f"number {number} is used twice")
        items.append(ProgrammeItem(number=number, composer=composer, subtitle=title))
    if not items:
        raise ProgrammeError("no pieces entered")
    return Programme(items=items, source="entered")


def load(path: str | Path) -> Programme:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return parse_text(pdf_to_text(p), source=str(p))
    if p.suffix.lower() == ".json":
        d = json.loads(clean_text(p.read_text(encoding="utf-8")))
        rows = d["items"] if isinstance(d, dict) else d
        items = [ProgrammeItem.from_dict({"number": n, **r}) for n, r in enumerate(rows, 1)]
        return Programme(items=items, source=str(p),
                         header=[d.get("concert_title", "")] if isinstance(d, dict) else [])
    return parse_text(p.read_text(encoding="utf-8"), source=str(p))


def find_pdf(concert_dir: str | Path) -> Path | None:
    pdfs = sorted(p for p in Path(concert_dir).iterdir() if p.suffix.lower() == ".pdf")
    named = [p for p in pdfs if re.search(r"program|spored", p.name, re.I)]
    if named:
        return named[0]
    return pdfs[0] if len(pdfs) == 1 else None
