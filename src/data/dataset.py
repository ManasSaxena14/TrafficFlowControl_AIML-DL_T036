"""Read the Phase 2 outputs. Every model in Phases 3-6 loads data through here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_split(name: str, protocol: str = "strict", root: Path = PROCESSED, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load a split. Pass keys=[...] to read only some arrays (X is ~1.7 GB for train)."""
    with np.load(root / protocol / f"{name}.npz") as z:
        return {k: z[k] for k in (keys or z.files)}


def load_scaler(protocol: str = "strict", root: Path = PROCESSED) -> dict:
    return json.loads((root / protocol / "scaler.json").read_text())


def load_adjacency(root: Path = PROCESSED) -> dict[str, np.ndarray]:
    with np.load(root / "adj.npz") as z:
        return {k: z[k] for k in z.files}


def inverse_scale(z: np.ndarray, scaler: dict) -> np.ndarray:
    """Back to mph. Do this before every metric so errors are in mph."""
    return z * scaler["sd"] + scaler["mu"]
