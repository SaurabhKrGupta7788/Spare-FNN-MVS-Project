"""RQ2: joint, marginal and conditional activation distributions.

The analysis uses a pre-specified, reproducible subset of neurons selected
from a prevalence range. It estimates empirical pairwise distributions,
phi correlations and an exploratory hierarchical clustering of neurons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from sparse_ffn.analysis._activation_io import (
    activation_shards,
    choose_row_indices,
    list_layer_dirs,
    load_selected_rows,
    read_manifest,
)

logger = logging.getLogger(__name__)


def _activation_prevalence(layer_dir: Path) -> tuple[np.ndarray, int]:
    """Calculate each neuron's empirical active frequency by streaming shards."""
    counts: np.ndarray | None = None
    n_rows = 0
    for path in activation_shards(layer_dir):
        with np.load(path) as shard:
            values = np.asarray(shard["activations"])
        if values.ndim != 2:
            raise ValueError(f"Expected 2-D activations in {path}; got {values.shape}")
        if counts is None:
            counts = np.zeros(values.shape[1], dtype=np.int64)
        elif values.shape[1] != counts.size:
            raise ValueError(f"Inconsistent neuron dimension in {path}")
        counts += np.count_nonzero(values > 0, axis=0)
        n_rows += values.shape[0]
    if counts is None or n_rows == 0:
        raise ValueError(f"No activation data found in {layer_dir}")
    return counts / n_rows, n_rows


def _pairwise_tables(indicators: np.ndarray, neuron_ids: np.ndarray):
    """Return marginal and pairwise joint/conditional probability tables."""
    n, m = indicators.shape
    marginal = indicators.mean(axis=0, dtype=np.float64)
    joint = indicators.astype(np.float64).T @ indicators.astype(np.float64) / n
    denominator = np.sqrt(
        marginal[:, None]
        * (1.0 - marginal[:, None])
        * marginal[None, :]
        * (1.0 - marginal[None, :])
    )
    covariance = joint - np.outer(marginal, marginal)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(joint),
        where=denominator > 0,
    )
    np.fill_diagonal(correlation, 1.0)

    i_idx, j_idx = np.triu_indices(m, k=1)
    p_i = marginal[i_idx]
    p_j = marginal[j_idx]
    p_11 = joint[i_idx, j_idx]
    p_10 = p_i - p_11
    p_01 = p_j - p_11
    p_00 = 1.0 - p_i - p_j + p_11
    pairwise = pd.DataFrame(
        {
            "neuron_i": neuron_ids[i_idx],
            "neuron_j": neuron_ids[j_idx],
            "p_i_active": p_i,
            "p_j_active": p_j,
            "p_11_both_active": p_11,
            "p_10_i_only_active": p_10,
            "p_01_j_only_active": p_01,
            "p_00_both_inactive": p_00,
            "p11_under_independence": p_i * p_j,
            "joint_minus_independence": p_11 - p_i * p_j,
            "p_i_active_given_j_active": np.divide(
                p_11, p_j, out=np.full_like(p_11, np.nan), where=p_j > 0
            ),
            "p_j_active_given_i_active": np.divide(
                p_11, p_i, out=np.full_like(p_11, np.nan), where=p_i > 0
            ),
            "phi_correlation": correlation[i_idx, j_idx],
        }
    )
    marginal_table = pd.DataFrame(
        {
            "neuron_index": neuron_ids,
            "active_count": indicators.sum(axis=0),
            "activation_frequency": marginal,
            "inactive_frequency": 1.0 - marginal,
        }
    )
    return marginal_table, pairwise, covariance, correlation


def run_neuron_dependence_analysis(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    max_neurons: int = 128,
    max_tokens: int = 50_000,
    min_activation_frequency: float = 0.01,
    max_activation_frequency: float = 0.99,
    random_seed: int = 42,
    cluster_distance_threshold: float = 0.7,
) -> Path:
    """Run RQ2 for each layer using empirical joint/marginal/conditional distributions."""
    if max_neurons < 2:
        raise ValueError("max_neurons must be at least 2")
    if not 0 <= min_activation_frequency < max_activation_frequency <= 1:
        raise ValueError("Activation-frequency bounds must satisfy 0 <= min < max <= 1")
    if cluster_distance_threshold <= 0:
        raise ValueError("cluster_distance_threshold must be positive")
    manifest = read_manifest(run_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    layer_reports: list[dict] = []

    for layer_index, layer_dir in enumerate(list_layer_dirs(run_dir)):
        logger.info("RQ2 dependence analysis: %s", layer_dir.name)
        prevalence, n_rows = _activation_prevalence(layer_dir)
        eligible = np.flatnonzero(
            (prevalence >= min_activation_frequency)
            & (prevalence <= max_activation_frequency)
        )
        if eligible.size < 2:
            raise ValueError(
                f"{layer_dir.name} has only {eligible.size} eligible neurons in the requested "
                f"prevalence interval [{min_activation_frequency}, {max_activation_frequency}]. "
                "Widen the interval or increase the activation sample."
            )
        rng = np.random.default_rng(random_seed + layer_index)
        n_selected = min(max_neurons, eligible.size)
        neuron_ids = np.sort(rng.choice(eligible, size=n_selected, replace=False))
        row_indices = choose_row_indices(n_rows, max_tokens, random_seed + 10_000 + layer_index)
        values = load_selected_rows(layer_dir, row_indices, columns=neuron_ids, dtype=np.float32)
        indicators = values > 0
        marginal, pairwise, covariance, correlation = _pairwise_tables(indicators, neuron_ids)

        layer_output = target / layer_dir.name
        layer_output.mkdir(parents=True, exist_ok=True)
        marginal.to_csv(layer_output / "marginal_activation_probabilities.csv", index=False)
        pairwise.to_csv(layer_output / "joint_conditional_pairwise_probabilities.csv", index=False)
        pd.DataFrame(covariance, index=neuron_ids, columns=neuron_ids).rename_axis(
            "neuron_index"
        ).to_csv(layer_output / "activation_covariance_matrix.csv")
        pd.DataFrame(correlation, index=neuron_ids, columns=neuron_ids).rename_axis(
            "neuron_index"
        ).to_csv(layer_output / "phi_correlation_matrix.csv")
        pd.DataFrame({"global_token_row_index": row_indices}).to_csv(
            layer_output / "sampled_token_row_indices.csv", index=False
        )

        # Hierarchical clustering uses correlation distance (1 - phi). Positive
        # correlations are close; negative correlations are distant.
        distance = np.clip(1.0 - correlation, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        hierarchy = linkage(condensed, method="average")
        cluster_ids = fcluster(
            hierarchy, t=cluster_distance_threshold, criterion="distance"
        )
        pd.DataFrame(
            {"neuron_index": neuron_ids, "cluster_id": cluster_ids}
        ).to_csv(layer_output / "cluster_assignments.csv", index=False)
        np.save(layer_output / "hierarchical_linkage.npy", hierarchy)

        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
        ax.set_title(f"Neuron phi-correlation: {layer_dir.name}")
        ax.set_xlabel("Selected neuron index")
        ax.set_ylabel("Selected neuron index")
        fig.colorbar(image, ax=ax, label="Phi correlation")
        fig.tight_layout()
        fig.savefig(layer_output / "phi_correlation_heatmap.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        dendrogram(hierarchy, no_labels=n_selected > 80, ax=ax, color_threshold=cluster_distance_threshold)
        ax.set_title(f"Hierarchical clustering of neuron activation patterns: {layer_dir.name}")
        ax.set_xlabel("Selected neurons")
        ax.set_ylabel("Correlation distance (1 - phi)")
        fig.tight_layout()
        fig.savefig(layer_output / "neuron_dendrogram.png", dpi=180)
        plt.close(fig)

        upper = correlation[np.triu_indices(n_selected, k=1)]
        report = {
            "layer": layer_dir.name,
            "n_tokens_used": int(indicators.shape[0]),
        "sampled_token_row_indices_file": "sampled_token_row_indices.csv",
            "n_neurons_in_layer": int(prevalence.size),
            "n_neurons_eligible": int(eligible.size),
            "n_neurons_selected": int(n_selected),
            "prevalence_interval": [min_activation_frequency, max_activation_frequency],
            "mean_pairwise_phi": float(np.mean(upper)),
            "median_pairwise_phi": float(np.median(upper)),
            "fraction_positive_phi": float(np.mean(upper > 0)),
            "number_of_clusters": int(np.unique(cluster_ids).size),
            "cluster_distance_threshold": cluster_distance_threshold,
            "selection_note": (
                "Neurons were sampled reproducibly from those whose empirical activation "
                "frequency fell inside the configured prevalence interval."
            ),
        }
        (layer_output / "dependence_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        layer_reports.append(report)

    summary = {
        "phase": "rq2_neuron_dependence",
        "run_dir": str(Path(run_dir).resolve()),
        "model_id": manifest.get("model_id"),
        "activation_function": manifest.get("activation_function"),
        "random_seed": random_seed,
        "method": "Empirical joint, marginal and conditional distributions; phi correlation; hierarchical cluster analysis.",
        "interpretation_note": (
            "These are pairwise descriptive dependence measures on a selected neuron subset. "
            "They do not prove mutual independence/dependence of the complete neuron vector, "
            "and clusters are exploratory summaries rather than biological or semantic labels."
        ),
        "layers": layer_reports,
    }
    (target / "rq2_neuron_dependence_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("RQ2 neuron dependence outputs saved to %s", target)
    return target


def analyze_neuron_dependence(*args, **kwargs) -> Path:
    """Compatibility alias for the original scaffold entry point."""
    return run_neuron_dependence_analysis(*args, **kwargs)
