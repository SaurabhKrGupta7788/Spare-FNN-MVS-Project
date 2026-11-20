"""RQ3: PCA and exploratory factor analysis of FFN activation matrices.

PCA is applied to centred (not variance-standardised) activations and is used
as a descriptive covariance-based method. Gaussian factor analysis is fitted
on a reproducible subset of neurons as an exploratory model; ReLU activations
are zero-inflated, so normality and model adequacy must be considered before
interpreting latent factors.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.exceptions import ConvergenceWarning

from sparse_ffn.analysis._activation_io import (
    choose_row_indices,
    inspect_layer,
    list_layer_dirs,
    load_selected_rows,
    read_manifest,
)

logger = logging.getLogger(__name__)


def _component_count_for_variance(cumulative: np.ndarray, target: float) -> int | None:
    positions = np.flatnonzero(cumulative >= target)
    return int(positions[0] + 1) if positions.size else None


def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """Participation ratio: (sum lambda_i)^2 / sum (lambda_i^2)."""
    vals = np.maximum(eigenvalues, 0)
    sum_vals = float(np.sum(vals))
    sum_sq = float(np.sum(vals**2))
    return float(sum_vals**2 / max(sum_sq, 1e-15))


def compute_entropy_effective_rank(eigenvalues: np.ndarray) -> float:
    """Entropy-based effective rank (Roy & Vetterli): exp(-sum p_i ln p_i)."""
    vals = np.maximum(eigenvalues, 0)
    total = np.sum(vals)
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def compute_subspace_stability(x: np.ndarray, n_components: int = 10, seed: int = 42) -> float:
    """Split-half cross-sample subspace cosine similarity between top principal components."""
    n = x.shape[0]
    if n < 4:
        return float(np.nan)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    x1, x2 = x[perm[:half]], x[perm[half:]]
    k = min(n_components, x1.shape[0] - 1, x1.shape[1], x2.shape[0] - 1)
    if k < 1:
        return float(np.nan)
    p1 = PCA(n_components=k, random_state=seed).fit(x1)
    p2 = PCA(n_components=k, random_state=seed).fit(x2)
    overlap = p1.components_ @ p2.components_.T
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    return float(np.mean(np.clip(singular_values, 0.0, 1.0)))



def run_latent_structure_analysis(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    max_samples: int = 5_000,
    pca_components: int = 64,
    factor_analysis_max_neurons: int = 256,
    factor_counts: tuple[int, ...] | list[int] = (2, 5, 10),
    min_activation_frequency: float = 0.01,
    max_activation_frequency: float = 0.99,
    rotation: str | None = "varimax",
    max_iter: int = 200,
    random_seed: int = 42,
) -> Path:
    """Run covariance PCA and exploratory Gaussian factor analysis per layer."""
    if max_samples < 3:
        raise ValueError("max_samples must be at least 3")
    if pca_components < 1 or factor_analysis_max_neurons < 2:
        raise ValueError("pca_components must be >= 1 and factor_analysis_max_neurons >= 2")
    if not factor_counts or any(k < 1 for k in factor_counts):
        raise ValueError("factor_counts must contain positive integers")
    if rotation not in {None, "varimax", "quartimax"}:
        raise ValueError("rotation must be None, 'varimax', or 'quartimax'")
    if not 0 <= min_activation_frequency < max_activation_frequency <= 1:
        raise ValueError("Activation-frequency bounds must satisfy 0 <= min < max <= 1")

    manifest = read_manifest(run_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    first_layer = list_layer_dirs(run_dir)[0]
    n_rows, _ = inspect_layer(first_layer)
    row_indices = choose_row_indices(n_rows, max_samples, random_seed)
    pca_rows: list[dict] = []
    pca_summaries: list[dict] = []
    factor_rows: list[dict] = []
    assumption_rows: list[dict] = []

    for layer_index, layer_dir in enumerate(list_layer_dirs(run_dir)):
        logger.info("RQ3 PCA and factor analysis: %s", layer_dir.name)
        n_layer_rows, n_features = inspect_layer(layer_dir)
        if n_layer_rows != n_rows:
            raise ValueError(
                f"All layers must contain the same token rows; {layer_dir.name} has "
                f"{n_layer_rows}, expected {n_rows}."
            )
        x = load_selected_rows(layer_dir, row_indices, dtype=np.float32)
        x = np.nan_to_num(x, copy=False)
        n_samples = x.shape[0]
        n_pca = min(pca_components, n_samples - 1, n_features)
        if n_pca < 1:
            raise ValueError(f"Not enough observations/features for PCA in {layer_dir.name}")

        # PCA centres columns internally. Do not standardise: the proposal calls
        # for covariance-based PCA on the measured activation scale.
        pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=random_seed)
        pca.fit(x)
        ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        cumulative = np.cumsum(ratios)
        for component_index, (eigenvalue, ratio, cumulative_ratio) in enumerate(
            zip(pca.explained_variance_, ratios, cumulative), start=1
        ):
            pca_rows.append(
                {
                    "layer": layer_dir.name,
                    "component": component_index,
                    "eigenvalue": float(eigenvalue),
                    "explained_variance_ratio": float(ratio),
                    "cumulative_variance_ratio": float(cumulative_ratio),
                }
            )
        layer_output = target / layer_dir.name
        layer_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"global_token_row_index": row_indices}).to_csv(
            layer_output / "sampled_token_row_indices.csv", index=False
        )
        np.savez_compressed(
            layer_output / "pca_components.npz",
            components=pca.components_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
            explained_variance=pca.explained_variance_.astype(np.float32),
            explained_variance_ratio=ratios.astype(np.float32),
        )
        pr = compute_participation_ratio(pca.explained_variance_)
        erank = compute_entropy_effective_rank(pca.explained_variance_)
        stability_k05 = compute_subspace_stability(x, n_components=5, seed=random_seed + layer_index)
        stability_k10 = compute_subspace_stability(x, n_components=10, seed=random_seed + layer_index)

        pca_summary = {
            "layer": layer_dir.name,
            "n_samples": int(n_samples),
            "n_neurons": int(n_features),
            "pca_components_computed": int(n_pca),
            "variance_explained_by_computed_components": float(cumulative[-1]),
            "participation_ratio": float(pr),
            "entropy_effective_rank": float(erank),
            "components_for_80_percent": _component_count_for_variance(cumulative, 0.80),
            "components_for_90_percent": _component_count_for_variance(cumulative, 0.90),
            "components_for_95_percent": _component_count_for_variance(cumulative, 0.95),
            "reconstruction_error_k02": float(1.0 - cumulative[min(1, n_pca - 1)]),
            "reconstruction_error_k05": float(1.0 - cumulative[min(4, n_pca - 1)]),
            "reconstruction_error_k10": float(1.0 - cumulative[min(9, n_pca - 1)]),
            "reconstruction_error_k20": float(1.0 - cumulative[min(19, n_pca - 1)]),
            "subspace_stability_k05": stability_k05,
            "subspace_stability_k10": stability_k10,
        }
        pca_summaries.append(pca_summary)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        axes[0].plot(np.arange(1, n_pca + 1), pca.explained_variance_, marker=".", linewidth=1)
        axes[0].set_title(f"Scree plot: {layer_dir.name}")
        axes[0].set_xlabel("Principal component")
        axes[0].set_ylabel("Eigenvalue")
        axes[1].plot(np.arange(1, n_pca + 1), cumulative, marker=".", linewidth=1)
        axes[1].axhline(0.8, linestyle="--", linewidth=0.8)
        axes[1].axhline(0.9, linestyle="--", linewidth=0.8)
        axes[1].axhline(0.95, linestyle="--", linewidth=0.8)
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title("Cumulative variance explained")
        axes[1].set_xlabel("Number of components")
        axes[1].set_ylabel("Cumulative proportion")
        fig.tight_layout()
        fig.savefig(layer_output / "pca_scree_and_cumulative_variance.png", dpi=180)
        plt.close(fig)

        # Factor analysis is fitted on a prevalence-filtered, random subset of
        # neurons. This is an explicit computational limit, not a claim that
        # every neuron has been modelled by the Gaussian factor model.
        activation_frequency = (x > 0).mean(axis=0)
        eligible = np.flatnonzero(
            (activation_frequency >= min_activation_frequency)
            & (activation_frequency <= max_activation_frequency)
        )
        if eligible.size < 2:
            assumption_rows.append(
                {
                    "layer": layer_dir.name,
                    "status": "skipped",
                    "reason": "Fewer than two neurons passed the activation-frequency filter.",
                    "zero_fraction": float(np.mean(x == 0)),
                    "mean_neuron_skewness_in_factor_sample": None,
                }
            )
            continue
        rng = np.random.default_rng(random_seed + 50_000 + layer_index)
        n_fa_features = min(factor_analysis_max_neurons, eligible.size)
        selected_neurons = np.sort(rng.choice(eligible, size=n_fa_features, replace=False))
        x_fa = x[:, selected_neurons]
        pd.DataFrame(
            {
                "neuron_index": selected_neurons,
                "activation_frequency_in_analysis_sample": activation_frequency[selected_neurons],
            }
        ).to_csv(layer_output / "factor_analysis_neuron_selection.csv", index=False)

        for k in sorted(set(int(value) for value in factor_counts)):
            if k >= min(x_fa.shape):
                factor_rows.append(
                    {"layer": layer_dir.name, "n_factors": k, "status": "skipped",
                     "reason": f"n_factors must be smaller than min(samples, features)={min(x_fa.shape)}"}
                )
                continue
            model = FactorAnalysis(
                n_components=k,
                rotation=rotation,
                max_iter=max_iter,
                random_state=random_seed,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_fa)
            loadings = model.components_.T
            communalities = np.square(loadings).sum(axis=1)
            empirical_covariance = np.cov(x_fa, rowvar=False, ddof=1)
            reproduced_covariance = loadings @ loadings.T + np.diag(model.noise_variance_)
            denominator = float(np.linalg.norm(empirical_covariance, ord="fro"))
            relative_covariance_error = float(
                np.linalg.norm(empirical_covariance - reproduced_covariance, ord="fro")
                / max(denominator, np.finfo(float).eps)
            )
            loglike_history = np.asarray(model.loglike_, dtype=np.float64)
            converged = bool(
                len(loglike_history) < max_iter
                and not any(issubclass(item.category, ConvergenceWarning) for item in caught)
            )
            record = {
                "layer": layer_dir.name,
                "n_factors": k,
                "status": "fitted",
                "iterations": int(len(loglike_history)),
                "converged": converged,
                "final_log_likelihood": float(loglike_history[-1]) if loglike_history.size else None,
                "relative_covariance_reproduction_error": relative_covariance_error,
                "mean_communality": float(np.mean(communalities)),
                "rotation": rotation or "none",
            }
            factor_rows.append(record)
            loading_frame = pd.DataFrame(
                loadings,
                index=selected_neurons,
                columns=[f"factor_{j + 1:02d}" for j in range(k)],
            )
            loading_frame.rename_axis("neuron_index").to_csv(
                layer_output / f"factor_loadings_k{k:02d}.csv"
            )
            fig, ax = plt.subplots(figsize=(max(6, k * 0.9), 7))
            image = ax.imshow(loadings, aspect="auto", cmap="coolwarm", interpolation="nearest")
            ax.set_title(f"Factor loadings: {layer_dir.name}, K={k}")
            ax.set_xlabel("Factor")
            ax.set_ylabel("Selected neuron position")
            ax.set_xticks(np.arange(k), [f"F{i + 1}" for i in range(k)])
            fig.colorbar(image, ax=ax, label="Loading")
            fig.tight_layout()
            fig.savefig(layer_output / f"factor_loadings_heatmap_k{k:02d}.png", dpi=180)
            plt.close(fig)
            pd.DataFrame(
                {
                    "neuron_index": selected_neurons,
                    "communality": communalities,
                    "uniqueness": model.noise_variance_,
                }
            ).to_csv(layer_output / f"factor_communalities_k{k:02d}.csv", index=False)

        assumption_rows.append(
            {
                "layer": layer_dir.name,
                "status": "evaluated",
                "n_samples": int(n_samples),
                "n_neurons_in_layer": int(n_features),
                "n_neurons_used_for_factor_analysis": int(n_fa_features),
                "zero_fraction_in_factor_sample": float(np.mean(x_fa == 0)),
                "mean_neuron_skewness_in_factor_sample": float(
                    np.nanmean(pd.DataFrame(x_fa).skew(axis=0).to_numpy())
                ),
                "factor_analysis_note": (
                    "Gaussian factor analysis is exploratory. ReLU activations have a point mass "
                    "at zero and may be non-normal; interpret loadings only after checking fit "
                    "and distributional adequacy."
                ),
            }
        )

    pd.DataFrame(pca_rows).to_csv(target / "pca_eigenvalues.csv", index=False)
    summary_df = pd.DataFrame(pca_summaries)
    summary_df.to_csv(target / "pca_layer_summary.csv", index=False)
    eff_cols = [
        "layer",
        "pca_components_computed",
        "participation_ratio",
        "entropy_effective_rank",
        "components_for_80_percent",
        "components_for_90_percent",
        "components_for_95_percent",
        "reconstruction_error_k05",
        "reconstruction_error_k10",
        "subspace_stability_k05",
        "subspace_stability_k10",
    ]
    summary_df[eff_cols].to_csv(target / "effective_dimensionality_summary.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(target / "factor_analysis_model_summary.csv", index=False)
    pd.DataFrame(assumption_rows).to_csv(target / "factor_analysis_assumption_notes.csv", index=False)

    summary = {
        "phase": "rq3_latent_structure",
        "run_dir": str(Path(run_dir).resolve()),
        "model_id": manifest.get("model_id"),
        "n_token_rows_available": int(n_rows),
        "n_token_rows_used": int(row_indices.size),
        "pca_method": "PCA on centred, unstandardised activation variables (covariance-based).",
        "factor_analysis_method": "Exploratory Gaussian FactorAnalysis with configured rotation on a prevalence-filtered neuron subset.",
        "factor_counts_requested": sorted(set(int(value) for value in factor_counts)),
        "interpretation_note": (
            "PCA describes covariance-based variation and does not require a multivariate-normal "
            "data model. Gaussian factor analysis is an approximation for these zero-inflated ReLU "
            "activations; the factor model is fitted to a selected neuron subset and must not be "
            "interpreted as a validated generative model for all neurons."
        ),
        "layers": pca_summaries,
    }
    (target / "rq3_latent_structure_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("RQ3 PCA and factor analysis outputs saved to %s", target)
    return target


def analyze_latent_structure(*args, **kwargs) -> Path:
    """Compatibility alias for the original scaffold entry point."""
    return run_latent_structure_analysis(*args, **kwargs)
