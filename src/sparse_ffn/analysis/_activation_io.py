"""Shared, memory-conscious utilities for reading activation extraction runs."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def list_layer_dirs(run_dir: str | Path) -> list[Path]:
    """Return layer directories in numeric order."""
    root = Path(run_dir)
    layer_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("layer_")]
    layer_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
    if not layer_dirs:
        raise FileNotFoundError(f"No layer directories found under {root}")
    return layer_dirs


def activation_shards(layer_dir: str | Path) -> list[Path]:
    """Return sorted NPZ activation shards for one layer."""
    files = sorted(Path(layer_dir).glob("batch_*.npz"))
    if not files:
        raise FileNotFoundError(f"No activation shards found under {layer_dir}")
    return files


def inspect_layer(layer_dir: str | Path) -> tuple[int, int]:
    """Return (number of token rows, number of neurons), validating shard shapes."""
    total_rows = 0
    n_neurons: int | None = None
    for path in activation_shards(layer_dir):
        with np.load(path) as shard:
            if "activations" not in shard.files:
                raise ValueError(f"Missing 'activations' array in {path}")
            values = shard["activations"]
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D activation array in {path}; got {values.shape}")
        if n_neurons is None:
            n_neurons = int(values.shape[1])
        elif int(values.shape[1]) != n_neurons:
            raise ValueError(f"Inconsistent neuron dimension in {path}")
        total_rows += int(values.shape[0])
    if n_neurons is None or total_rows == 0:
        raise ValueError(f"No activation rows found under {layer_dir}")
    return total_rows, n_neurons


def choose_row_indices(n_rows: int, max_rows: int | None, seed: int) -> np.ndarray:
    """Choose deterministic global row indices; return all rows when max_rows is unset."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if max_rows is None or max_rows <= 0 or max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False).astype(np.int64))


def load_selected_rows(
    layer_dir: str | Path,
    row_indices: np.ndarray,
    *,
    columns: np.ndarray | None = None,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Load selected global token rows, optionally selecting neuron columns.

    This reads compressed shards sequentially and only keeps the requested rows
    in memory. ``row_indices`` must be sorted and unique.
    """
    requested = np.asarray(row_indices, dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if requested.size and (np.any(np.diff(requested) <= 0) or requested[0] < 0):
        raise ValueError("row_indices must be sorted, unique, and non-negative")

    parts: list[np.ndarray] = []
    global_offset = 0
    cursor = 0
    for path in activation_shards(layer_dir):
        with np.load(path) as shard:
            values = np.asarray(shard["activations"])
            if values.ndim != 2:
                raise ValueError(f"Expected 2-D activations in {path}; got {values.shape}")
            end = global_offset + values.shape[0]
            start_cursor = cursor
            while cursor < requested.size and requested[cursor] < end:
                cursor += 1
            if cursor > start_cursor:
                local_rows = requested[start_cursor:cursor] - global_offset
                selected = values[local_rows]
                if columns is not None:
                    selected = selected[:, columns]
                parts.append(np.asarray(selected, dtype=dtype))
            global_offset = end
            if cursor >= requested.size:
                break

    if cursor != requested.size:
        raise IndexError(
            f"Requested row index {int(requested[cursor])} is outside the activation data "
            f"(observed {global_offset} rows)."
        )
    if not parts:
        n_features = len(columns) if columns is not None else inspect_layer(layer_dir)[1]
        return np.empty((0, n_features), dtype=dtype)
    return np.concatenate(parts, axis=0)


def read_manifest(run_dir: str | Path) -> dict:
    """Read and validate the extraction manifest."""
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Extraction manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Activation run is not complete: status={manifest.get('status')!r}")
    return manifest
