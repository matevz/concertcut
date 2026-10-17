"""Frame-wise music / applause / speech probabilities with PANNs CNN14 (AudioSet).

Runs on CPU. Model weights (~330 MB) and the AudioSet label list are downloaded once
into the model directory (``--model-dir``, ``$CONCERTCUT_MODEL_DIR`` or
``~/.cache/concertcut``).
"""

from __future__ import annotations

import csv
import importlib.util
import logging
import os
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SR = 32000
FPS = 100  # PANNs frame rate (hop 320 @ 32 kHz)
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
CHECKPOINT_NAME = "Cnn14_DecisionLevelMax_mAP=0.385.pth"
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
LABELS_NAME = "class_labels_indices.csv"

GROUPS = {
    "music": ["Music", "Singing", "Choir", "Vocal music", "A capella", "Chant", "Song",
              "Musical instrument", "Organ", "Piano", "Keyboard (musical)", "Harmonic",
              "Christian music", "Classical music", "Lullaby", "Folk music"],
    "applause": ["Applause", "Clapping", "Cheering", "Crowd", "Hands"],
    "speech": ["Speech", "Narration, monologue", "Male speech, man speaking",
               "Female speech, woman speaking", "Conversation", "Speech synthesizer"],
}
GROUP_NAMES = list(GROUPS)


def default_model_dir() -> Path:
    return Path(os.environ.get("CONCERTCUT_MODEL_DIR", Path.home() / ".cache" / "concertcut"))


def _download(url: str, dest: Path, min_size: int = 0) -> Path:
    if dest.exists() and dest.stat().st_size >= min_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, step = 0, 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done * 10 // total > step:
                step = done * 10 // total
                log.info("  %d%%", step * 10)
    tmp.replace(dest)
    return dest


def load_labels(model_dir: Path) -> list[str]:
    path = _download(LABELS_URL, model_dir / LABELS_NAME)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    return [r[2] for r in rows]


def group_indices(labels: list[str]) -> dict[str, list[int]]:
    idx = {name: i for i, name in enumerate(labels)}
    out = {}
    for g, names in GROUPS.items():
        out[g] = [idx[n] for n in names if n in idx]
        missing = [n for n in names if n not in idx]
        if missing:
            log.debug("AudioSet labels not found for %s: %s", g, missing)
    return out


def _import_panns_models():
    """Import panns_inference.models without running the package __init__ (which
    downloads files into ~/panns_data on import)."""
    if "panns_inference.models" in sys.modules:
        return sys.modules["panns_inference.models"]
    spec = importlib.util.find_spec("panns_inference")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("panns-inference is not installed")
    pkg = types.ModuleType("panns_inference")
    pkg.__path__ = list(spec.submodule_search_locations)
    sys.modules.setdefault("panns_inference", pkg)
    import matplotlib

    matplotlib.use("Agg")  # models.py imports pyplot
    from panns_inference import models  # noqa: E402

    return models


class Detector:
    def __init__(self, model_dir: str | Path | None = None, threads: int | None = None):
        import torch

        self.torch = torch
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.labels = load_labels(self.model_dir)
        self.groups = group_indices(self.labels)
        ckpt = _download(CHECKPOINT_URL, self.model_dir / CHECKPOINT_NAME, min_size=300_000_000)
        models = _import_panns_models()
        self.model = models.Cnn14_DecisionLevelMax(
            sample_rate=SR, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
            classes_num=len(self.labels), interpolate_mode="nearest")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        torch.set_num_threads(threads or os.cpu_count() or 4)

    def framewise(self, x: np.ndarray) -> np.ndarray:
        """(frames, classes) probabilities for a mono 32 kHz excerpt."""
        with self.torch.no_grad():
            out = self.model(self.torch.from_numpy(x[None, :].astype(np.float32)), None)
        return out["framewise_output"][0].numpy()

    def group_scores(self, audio: np.ndarray, chunk_s: float = 60.0, overlap_s: float = 3.0,
                     progress=None) -> np.ndarray:
        """(frames @ 100 fps, len(GROUP_NAMES)) max-probability per class group."""
        n_frames = len(audio) * FPS // SR
        out = np.zeros((n_frames, len(GROUP_NAMES)), dtype=np.float32)
        chunk, ov = int(chunk_s * SR), int(overlap_s * SR)
        pos = 0
        while pos < len(audio):
            a = max(0, pos - ov)
            b = min(len(audio), pos + chunk + ov)
            seg = audio[a:b]
            if len(seg) < SR:  # model needs at least ~1 s
                seg = np.pad(seg, (0, SR - len(seg)))
            fw = self.framewise(seg)
            g = np.stack([fw[:, self.groups[k]].max(axis=1) for k in GROUP_NAMES], axis=1)
            fa = a * FPS // SR
            keep0 = (pos - a) * FPS // SR
            keep1 = min(len(g), (min(len(audio), pos + chunk) - a) * FPS // SR)
            dst0 = fa + keep0
            m = min(keep1 - keep0, n_frames - dst0)
            if m > 0:
                out[dst0:dst0 + m] = g[keep0:keep0 + m]
            pos += chunk
            if progress:
                progress(min(pos, len(audio)) / len(audio))
        return out
