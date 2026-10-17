# Concert Cut

Turns the recording of a concert — camera video, a separate field recorder (e.g. a Zoom
H6) and the printed programme — into one file per piece:

```
render/audio/01 Uršula Jašovec - Grajska gospa in berač.flac
render/video/01 Uršula Jašovec - Grajska gospa in berač.mp4
...
```

plus a Kdenlive project to review or fine-tune the result. Mainly built for classical
concerts: applause and spoken narration between pieces are left out, pieces are named
after the programme, and the whole concert gets one constant gain.

![A concert in Kdenlive: the camera files on V1, the camera sound (muted) on A1, the field recorder on A2, and a guide at the start and end of every piece](screenshot.png)

What it does:

- **Sync** the recorder audio to the camera, also across split files (4 GB camera
  chunks, 2 GB WAV chunks) and with clock drift between the devices.
- **Find the pieces**: a pretrained audio classifier (PANNs, AudioSet) labels music,
  applause and speech; pieces start at the first note (after the pitch is given, with up
  to 3 s of quiet hall before it) and end where the audience starts to clap.
- **Name them** from the programme PDF (rule-based parser, no network) after you have
  confirmed which pieces were actually performed and how the recording matches them.
- **Raise the volume** of the whole concert by one constant amount — as much as possible
  without clipping in any piece. No fades, compression or other processing.
- **Export** FLAC (recorder audio, 24-bit, source sample rate) and MP4 (camera video +
  recorder audio), and a **Kdenlive** project (plus an OpenTimelineIO file).

## Requirements

- Linux (other platforms untested), Python 3.11+, [uv](https://docs.astral.sh/uv/)
- `ffmpeg` / `ffprobe`; optional `pdftotext` (poppler-utils) for better PDF text extraction
- No GPU needed: the detector runs on the CPU (about a minute for a 90-minute concert)

```bash
sudo apt install ffmpeg poppler-utils
git clone <this repository> && cd concertcut
uv sync            # installs a CPU-only build of PyTorch
```

The detector weights (PANNs CNN14, ~330 MB) are downloaded on first use to
`~/.cache/concertcut` (change with `--model-dir` or `$CONCERTCUT_MODEL_DIR`).

## Usage

Put the camera files, the recorder WAVs and the programme PDF in one folder, then run the
steps below. A folder with **only a recorder** (no camera) works too — you get FLAC files
and a Kdenlive project with just the audio — and so does a folder with **only camera
files**: the camera's own sound is then the recording (detection, FLAC, and the sound of
the MP4).



```bash
# 1. analyse -> <concert>/concertcut.json
uv run concertcut analyze /path/to/20261016_concert

# 2. optional: Kdenlive project for review -> <concert>/<name>.concertcut.kdenlive (+ .otio)
uv run concertcut kdenlive /path/to/20261016_concert

# 3. export -> <concert>/render/audio/*.flac and <concert>/render/video/*.mp4
uv run concertcut render /path/to/20261016_concert

# all three at once
uv run concertcut run /path/to/20261016_concert

# check how a programme PDF is parsed
uv run concertcut programme programme.pdf
```

`analyze` asks two questions in the terminal:

```
Programme - Bolečina in lepota:
   1. Uršula Jašovec - Grajska gospa in berač
   ...
Was it performed exactly like this? [Y]es / [s]kip pieces / [o]rder / [e]dit / [r]e-read the PDF / [q]uit:

Detected pieces (camera timeline):
   #       start         end    len  ends at     piece
   1  00:16:39.4  00:20:38.8   3:59  applause    [ 1] Uršula Jašovec - Grajska gospa in berač (after pitch)
   ...
Is this matching correct? [Y]es / [m]erge segments / [d]rop segment / [a]ssign pieces /
                           [r]e-read the video and audio files / [q]uit:
```

1. **Performed programme** — skip pieces that were not sung (`s 6`), change the order
   (`o 1 2 4 3 5`) or edit the list in `$EDITOR` (`e`).
2. **Matching** — merge two segments (`m 6 7`), drop one (`d 10`) or assign programme
   numbers to all segments (`a 1 2 3 4 5 7 8 9 +`; `+` = encore / not in the programme,
   `-` = drop).

If no programme can be determined — no PDF in the folder, a scanned PDF without text, or
nothing recognisable in it — `analyze` asks you to type the performed pieces instead, one
per line as `NN Composer - Title` (e.g. `02 Andrej Misson - Mlinar in smrt`), finished
with an empty line. A line without ` - ` is a title without composer; pressing Enter right
away continues without names (pieces are then numbered only).

Track numbers follow the performance (01, 02, …, encores last, named *Encore*; without a
programme, pieces are named *Piece 1*, *Piece 2*, …).
`--skip 6` answers the first question up front; `--yes` accepts everything without
asking (for scripts).

Useful options: `render --only 2,5`, `--audio-only`, `--video-only`, `--preset medium`
(faster video encoding), `--flac-rate 48000`, `--dry-run` (print the ffmpeg commands),
`-j` (parallel jobs).

### The plan file: `concertcut.json`

`analyze` writes everything it found and you confirmed to `concertcut.json` in the concert
folder; `kdenlive` and `render` take that file (or just the concert folder). You can edit it:
piece names, `start`/`end`, `enabled`, and `settings` (e.g. `gain_db`, `x264_crf`). All
times are `HH:MM:SS.mmm` on the camera timeline (0 = first frame of the first camera file).

**Running again is always safe.** When `concertcut.json` already exists, `analyze` loads
it (*"concertcut.json already exists, loading existing preferences"*) and asks the same two
questions, starting from what you confirmed or edited last time — this takes seconds. Both
questions have an extra option to start over for that part:

- `r` at the programme question reads the programme PDF again,
- `r` at the matching question reads the video and audio files again (sync and detection;
  the detector results are cached in `concertcut.cache.npz`, so this takes about a minute).

To start from scratch, delete `concertcut.json`.

`kdenlive` always writes a fresh project and keeps the previous one as `….kdenlive.bak`.
`render` only renders files that are missing or whose piece or settings changed since they
were written (it keeps a fingerprint of every output in `render/.concertcut-render.json`);
`--force` renders everything again. The concert gain (`settings.gain_db`) is re-measured
after changes unless you set it yourself and set `settings.gain_db_auto` to `false`.

### The Kdenlive project

Written for Kdenlive 26.04 (document version 1.1, works with 23.04+ in principle):

- V1: the camera files; A2: the camera sound (muted track); A1: the recorder files, whole
  and back to back at one sync offset, each with a volume filter at the concert gain
- guides `NN Composer - Title` at the start and `NN end` at the end of every piece, so
  Kdenlive's *render by guides* works as well
- the project profile (resolution, frame rate) is taken from the camera files

If a future Kdenlive version cannot open it, import the `.otio` file instead
(File → OpenTimelineIO Import): same clips and markers, without the volume filters.

## How it works

| step | module | method |
|---|---|---|
| find files | `media.py` | camera files and the largest WAV group (`*normalized*` ignored); split files placed by camera `creation_time` and BWF `time_reference`; warns when camera and recorder dates differ |
| sync | `sync.py` | onset-envelope cross-correlation of each camera file against the whole recording (coarse, also places camera files after gaps); GCC-PHAT every 90 s and a Theil–Sen line fit (offset + clock drift); the offset is measured again inside every piece for the exports |
| programme | `programme.py` | rule-based: after the `PROGRAM` heading, `Composer (year):` / CAPS title / subtitle / notes; name = subtitle, else the CAPS title in sentence case; composer missing → *Traditional* |
| pieces | `detect.py`, `segment.py` | PANNs CNN14 frame-wise AudioSet scores → music / applause / speech; pitch cues (the short note before each piece) separate pieces sung without applause; silence alone never splits a piece; pieces after the final ovation are encores; the number of pieces is matched to the programme |
| boundaries | `segment.py` | start: the first audible note (walking back from the detector through soft openings), plus up to 3 s of quiet hall (shorter if the pitch tone still rings or there is noise); end: the onset of applause, or just before the next piece |
| gain | `gain.py` | one constant gain: the loudest piece peaks at `--peak` (default −0.1 dBFS) |
| export | `render.py` | ffmpeg; FLAC 24-bit from the recorder; MP4 = camera video (x264, CRF 18) + recorder audio (AAC 320k) |
| project | `kdenlive.py`, `otio_export.py` | Kdenlive 1.1 document and OpenTimelineIO timeline |

The analysis core takes callbacks for the two confirmations (`concertcut.analyze`), so it
can be driven by something other than the terminal, e.g. a web service.

## Development

```bash
uv run pytest
```

The tests use synthetic signals and need neither ffmpeg nor the model. GitHub Actions runs
them on Python 3.11, 3.12 and 3.13 for every push and pull request
(`.github/workflows/ci.yml`).

## Credits and license

- Sound event detection: [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
  (Kong et al., *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern
  Recognition*, 2020) via `panns-inference`. The model weights are downloaded from
  [Zenodo](https://zenodo.org/record/3987831) and are licensed CC BY 4.0.
- Timeline export: [OpenTimelineIO](https://opentimelineio.readthedocs.io/).

concertcut is released under the [MIT License](LICENSE).
