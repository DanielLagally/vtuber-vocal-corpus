"""Speaker embeddings: one voice fingerprint per clip, one per talent.

A speaker-recognition network maps a clip to a vector (an embedding) arranged
so that clips of the same voice land close together and different voices far
apart. Its geometry is learned from thousands of speakers, which makes the
distance between two talents' embeddings the strongest available measure of
"these two sound alike" — much stronger than any handful of acoustic features.
It is also a black box: it cannot say *why* two voices are close. The
interpretable features answer that; this answers *whether*.

**Embeddings are never published.** An embedding is exactly the input a
zero-shot voice-cloning system conditions on, so a published per-talent vector
would be a ready-made cloning key. They are written under ``data/`` (ignored by
git) and only similarity scores derived from them leave this machine.

The production encoder is SpeechBrain's ECAPA-TDNN trained on VoxCeleb, a
general speaker model — deliberately *not* one fine-tuned on this corpus for
similarity: a model trained to separate these 80 talents learns to push
similar voices apart, which is the opposite of what a similarity ranking needs.

See reference/measurement.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .acoustics import load_mono

#: Every encoder here takes 16 kHz mono.
EMBED_SR = 16_000
DEFAULT_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_MODEL_DIR = Path("data/models/spkrec-ecapa-voxceleb")
DEFAULT_OUT_DIR = Path("data/embeddings")


class Encoder(Protocol):
    name: str

    def encode(self, wave: np.ndarray, sr: int) -> np.ndarray: ...


class SpeechBrainEncoder:
    """ECAPA-TDNN (VoxCeleb). Loaded lazily; runs on CUDA / MPS when present."""

    def __init__(
        self,
        source: str = DEFAULT_MODEL_SOURCE,
        savedir: Path | str = DEFAULT_MODEL_DIR,
        device: str | None = None,
    ):
        self.name = source
        self._source = source
        self._savedir = str(savedir)
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            device = self._device
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            self._model = EncoderClassifier.from_hparams(
                source=self._source, savedir=self._savedir, run_opts={"device": device}
            )
        return self._model

    def encode(self, wave: np.ndarray, sr: int) -> np.ndarray:
        import torch

        if sr != EMBED_SR:
            raise ValueError(f"encoder expects {EMBED_SR} Hz, got {sr}")
        model = self._load()
        with torch.no_grad():
            batch = torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0)
            out = model.encode_batch(batch)
        return out.squeeze().detach().cpu().numpy().astype(np.float64)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def embed_clip(path: Path | str, encoder: Encoder) -> np.ndarray:
    """Unit-length embedding of one clip, from the audio the features used."""
    wave, sr = load_mono(path, EMBED_SR)
    return _unit(np.asarray(encoder.encode(wave, sr), dtype=np.float64))


@dataclass(frozen=True)
class Embeddings:
    ids: tuple[str, ...]
    vectors: np.ndarray  # (n, dim), unit rows
    model: str

    def get(self, clip_id: str) -> np.ndarray | None:
        try:
            return self.vectors[self.ids.index(clip_id)]
        except ValueError:
            return None


def embed_records(
    records: list[dict],
    encoder: Encoder,
    *,
    existing: Embeddings | None = None,
    audio_exists=lambda path: Path(path).is_file(),
) -> Embeddings:
    """Embeddings for every record whose audio is on disk.

    Clips already in ``existing`` (from the same model) are reused rather than
    re-encoded. A clip whose audio is gone is left out — a gap, never a zero
    vector, which would sit at a fake, equal distance from everyone.
    """
    reuse = existing if existing is not None and existing.model == encoder.name else None
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for record in records:
        clip_id = str(record.get("id", ""))
        previous = reuse.get(clip_id) if reuse is not None else None
        if previous is not None:
            ids.append(clip_id)
            rows.append(previous)
            continue
        audio = record.get("source_audio")
        if record.get("legacy") or not audio or not audio_exists(audio):
            continue
        ids.append(clip_id)
        rows.append(embed_clip(audio, encoder))
    vectors = np.vstack(rows) if rows else np.zeros((0, 0))
    return Embeddings(tuple(ids), vectors, encoder.name)


def write_embeddings(out_dir: Path | str, talent: str, embeddings: Embeddings) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{talent}.npz"
    np.savez(
        path,
        ids=np.array(embeddings.ids, dtype=str),
        vectors=embeddings.vectors,
        model=np.array(embeddings.model),
    )
    return path


def load_embeddings(path: Path | str) -> Embeddings:
    with np.load(path, allow_pickle=False) as data:
        return Embeddings(
            tuple(str(i) for i in data["ids"]), np.array(data["vectors"]), str(data["model"])
        )


def usable(record: dict) -> bool:
    """A clip that may speak for its talent: re-measured and QC-passing."""
    return not record.get("legacy") and bool((record.get("qc") or {}).get("pass"))


def talent_centroid(records: list[dict], embeddings: Embeddings) -> np.ndarray | None:
    """The talent's voice: the re-normalised mean of usable clips' embeddings.

    ``None`` when no usable clip was embedded — no invented position.
    """
    rows = [
        vector
        for record in records
        if usable(record)
        and (vector := embeddings.get(str(record.get("id", "")))) is not None
    ]
    if not rows:
        return None
    centroid = np.mean(np.vstack(rows), axis=0)
    return _unit(centroid) if math.isfinite(float(np.linalg.norm(centroid))) else None
