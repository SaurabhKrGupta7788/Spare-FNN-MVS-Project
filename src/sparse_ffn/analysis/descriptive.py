"""Descriptive analysis of post-ReLU FFN activations.

This is Phase 1 of the project. It describes the empirical activation
 distribution without fitting inferential or latent-variable models yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from sparse_ffn.config import resolve_path

logger = logging.getLogger(__name__)


def _activation_files(layer_dir: Path) -> list[Path]:
    files = sorted(layer_dir.glob("batch_*.npz"))
    if not files:
        raise FileNotFoundError(f"No activation shards found in {layer_dir}")
    return files


def _summarize_layer(
    layer_dir: Path,
    *,
    histogram_sample_size: int,
    random_seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Compute streaming descriptive statistics for one layer.

    Returns a layer summary, token-wise sparsity, neuron-wise active frequency,
    and a bounded random sample of positive activation magnitudes.
    """
    if histogram_sample_size <= 0:
        raise ValueError("histogram_sample_size must be positive")
    rng = np.random.default_rng(random_seed)
    files = _activation_files(layer_dir)

    n_tokens = 0
    n_neurons: int | None = None
    total_values = 0
    positive_count = 0
    zero_count = 0
    activation_sum = 0.0
    activation_sum_sq = 0.0
    positive_sum = 0.0
    positive_sum_sq = 0.0
    positive_samples: list[np.ndarray] = []
    token_sparsity_chunks: list[np.ndarray] = []
    block_id_chunks: list[np.ndarray] = []
    active_counts: np.ndarray | None = None

    # A bounded sample is enough for descriptive histograms and approximate
    # positive-value quantiles; the full activation matrix is never concatenated.
    sample_per_shard = max(1000, histogram_sample_size // max(1, len(files)))

    for shard_path in files:
        with np.load(shard_path) as shard:
            values = np.asarray(shard["activations"])
            block_ids = (
                np.asarray(shard["block_ids"], dtype=np.int64)
                if "block_ids" in shard.files
                else np.full(values.shape[0], -1, dtype=np.int64)
            )
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D activation matrix in {shard_path}, got {values.shape}")
        if n_neurons is None:
            n_neurons = int(values.shape[1])
            active_counts = np.zeros(n_neurons, dtype=np.int64)
        elif values.shape[1] != n_neurons:
            raise ValueError(f"Inconsistent neuron dimension in {shard_path}")

        # OPT uses ReLU, so active values are strictly greater than zero.
        active_mask = values > 0
        chunk_tokens = int(values.shape[0])
        n_tokens += chunk_tokens
        total_values += int(values.size)
        positive_in_chunk = int(active_mask.sum())
        positive_count += positive_in_chunk
        zero_count += int(values.size - positive_in_chunk)
        activation_sum += float(values.sum(dtype=np.float64))
        activation_sum_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        positive_values = values[active_mask].astype(np.float32, copy=False)
        if positive_values.size:
            positive_sum += float(positive_values.sum(dtype=np.float64))
            positive_sum_sq += float(np.square(positive_values, dtype=np.float64).sum(dtype=np.float64))
            take = min(sample_per_shard, positive_values.size)
            if take < positive_values.size:
                selected = rng.choice(positive_values.size, size=take, replace=False)
                positive_values = positive_values[selected]
            positive_samples.append(positive_values)

        token_sparsity_chunks.append(1.0 - active_mask.mean(axis=1, dtype=np.float64))
        if block_ids.shape[0] != values.shape[0]:
            raise ValueError(f"block_ids length does not match activation rows in {shard_path}")
        block_id_chunks.append(block_ids)
        assert active_counts is not None
        active_counts += active_mask.sum(axis=0, dtype=np.int64)

    if not n_tokens or not total_values or n_neurons is None or active_counts is None:
        raise ValueError(f"No activation values were found under {layer_dir}")

    token_sparsity = np.concatenate(token_sparsity_chunks).astype(np.float32)
    block_ids_all = np.concatenate(block_id_chunks).astype(np.int64)
    neuron_frequency = (active_counts / n_tokens).astype(np.float32)
    if positive_samples:
        positive_sample = np.concatenate(positive_samples)
        if positive_sample.size > histogram_sample_size:
            chosen = rng.choice(positive_sample.size, size=histogram_sample_size, replace=False)
            positive_sample = positive_sample[chosen]
    else:
        positive_sample = np.asarray([], dtype=np.float32)

    mean_activation = activation_sum / total_values
    variance = max(0.0, activation_sum_sq / total_values - mean_activation**2)
    positive_mean = positive_sum / positive_count if positive_count else 0.0
    positive_variance = (
        max(0.0, positive_sum_sq / positive_count - positive_mean**2) if positive_count else 0.0
    )

    summary: dict[str, Any] = {
        "layer": layer_dir.name,
        "n_tokens": n_tokens,
        "n_neurons": n_neurons,
        "n_activation_values": total_values,
        "zero_count": zero_count,
        "positive_count": positive_count,
        "zero_fraction": zero_count / total_values,
        "active_fraction": positive_count / total_values,
        "mean_token_sparsity": float(token_sparsity.mean()),
        "std_token_sparsity": float(token_sparsity.std(ddof=1)) if token_sparsity.size > 1 else 0.0,
        "median_token_sparsity": float(np.median(token_sparsity)),
        "token_sparsity_p05": float(np.quantile(token_sparsity, 0.05)),
        "token_sparsity_p95": float(np.quantile(token_sparsity, 0.95)),
        "mean_neuron_activation_frequency": float(neuron_frequency.mean()),
        "median_neuron_activation_frequency": float(np.median(neuron_frequency)),
        "mean_activation": mean_activation,
        "std_activation": float(np.sqrt(variance)),
        "positive_mean": positive_mean,
        "positive_std": float(np.sqrt(positive_variance)),
        "positive_p50_sample": float(np.quantile(positive_sample, 0.50)) if positive_sample.size else 0.0,
        "positive_p90_sample": float(np.quantile(positive_sample, 0.90)) if positive_sample.size else 0.0,
        "positive_p99_sample": float(np.quantile(positive_sample, 0.99)) if positive_sample.size else 0.0,
        "positive_sample_skewness": float(skew(positive_sample)) if positive_sample.size > 2 else 0.0,
        "positive_sample_excess_kurtosis": float(kurtosis(positive_sample)) if positive_sample.size > 3 else 0.0,
        "positive_quantiles_are_sampled": True,
        "positive_sample_size": int(positive_sample.size),
    }
    return summary, token_sparsity, neuron_frequency, positive_sample, block_ids_all


def _plot_layer_sparsity(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["mean_token_sparsity"] * 100.0)
    ax.set_xticks(x, summary["layer"], rotation=45, ha="right")
    ax.set_ylabel("Mean inactive neurons per token (%)")
    ax.set_xlabel("FFN layer")
    ax.set_title("Mean activation sparsity by FFN layer")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, summary["mean_token_sparsity"] * 100.0, strict=True):
        ax.annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_sparsity.png", dpi=180)
    plt.close(fig)


def _plot_distributions(
    layer: str,
    token_sparsity: np.ndarray,
    neuron_frequency: np.ndarray,
    positive_sample: np.ndarray,
    output_dir: Path,
    bins: int,
) -> None:
    layer_dir = output_dir / layer
    layer_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(token_sparsity * 100.0, bins=bins, density=True, alpha=0.85)
    ax.set_xlabel("Inactive neurons per token (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"Token-wise sparsity distribution — {layer}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(layer_dir / "token_sparsity_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(neuron_frequency * 100.0, bins=bins, density=True, alpha=0.85)
    ax.set_xlabel("Tokens on which neuron is active (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"Neuron activation-frequency distribution — {layer}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(layer_dir / "neuron_activation_frequency.png", dpi=180)
    plt.close(fig)

    if positive_sample.size:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(positive_sample, bins=bins, density=True, alpha=0.85)
        ax.set_xlabel("Positive activation magnitude")
        ax.set_ylabel("Density")
        ax.set_title(f"Positive activation distribution — {layer}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(layer_dir / "positive_activation_distribution.png", dpi=180)
        plt.close(fig)


def find_latest_complete_run(activations_root: str | Path, split: str) -> Path:
    """Return the most recently modified completed extraction run for a split."""
    split_dir = resolve_path(activations_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"No activation directory exists for split {split!r}: {split_dir}")
    candidates: list[tuple[float, Path]] = []
    for path in split_dir.iterdir():
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "complete":
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(f"No completed activation run found under {split_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def run_descriptive_analysis(
    *,
    activations_root: str | Path,
    analysis_root: str | Path,
    split: str = "train",
    run_dir: str | Path | None = None,
    histogram_sample_size: int = 100_000,
    random_seed: int = 42,
    histogram_bins: int = 60,
) -> Path:
    """Generate descriptive tables and plots for every extracted FFN layer."""
    selected_run = Path(run_dir).resolve() if run_dir else find_latest_complete_run(activations_root, split)
    manifest_path = selected_run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Extraction manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Extraction run is not complete (status={manifest.get('status')!r}).")

    output_dir = resolve_path(analysis_root) / split / selected_run.name / "descriptive"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dirs = sorted(path for path in selected_run.iterdir() if path.is_dir() and path.name.startswith("layer_"))
    if not layer_dirs:
        raise FileNotFoundError(f"No layer directories found in {selected_run}")

    summaries: list[dict[str, Any]] = []
    block_sparsity_rows: list[pd.DataFrame] = []
    neuron_frequency_rows: list[pd.DataFrame] = []
    for layer_index, layer_dir in enumerate(layer_dirs):
        logger.info("Describing %s", layer_dir.name)
        summary, token_sparsity, neuron_frequency, positive_sample, block_ids = _summarize_layer(
            layer_dir,
            histogram_sample_size=histogram_sample_size,
            random_seed=random_seed + layer_index,
        )
        summaries.append(summary)
        block_sparsity_rows.append(
            pd.DataFrame({"block_id": block_ids, "token_sparsity": token_sparsity})
            .groupby("block_id", as_index=False)
            .agg(
                mean_sparsity=("token_sparsity", "mean"),
                std_sparsity=("token_sparsity", "std"),
                token_count=("token_sparsity", "size"),
            )
            .assign(layer=layer_dir.name)
        )
        neuron_frequency_rows.append(
            pd.DataFrame({"layer": layer_dir.name, "neuron_index": np.arange(neuron_frequency.size),
                          "activation_frequency": neuron_frequency})
        )
        _plot_distributions(
            layer_dir.name,
            token_sparsity,
            neuron_frequency,
            positive_sample,
            output_dir / "plots",
            bins=histogram_bins,
        )

    summary_frame = pd.DataFrame(summaries).sort_values("layer")
    summary_frame.to_csv(output_dir / "layer_summary.csv", index=False)
    pd.concat(block_sparsity_rows, ignore_index=True).to_csv(
        output_dir / "block_sparsity_by_layer.csv", index=False
    )
    pd.concat(neuron_frequency_rows, ignore_index=True).to_csv(
        output_dir / "neuron_activation_frequency.csv", index=False
    )
    _plot_layer_sparsity(summary_frame, output_dir / "plots")
    report = {
        "phase": "phase_1_descriptive_analysis",
        "split": split,
        "run_dir": str(selected_run),
        "model_id": manifest.get("model_id"),
        "activation_function": manifest.get("activation_function"),
        "layers": summaries,
        "interpretation_note": (
            "This report is descriptive. It does not establish statistical significance, "
            "neuron independence, or low-rank structure. Those are later research questions."
        ),
    }
    (output_dir / "descriptive_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Descriptive report and plots saved to %s", output_dir)
    return output_dir
