"""RQ1: profile analysis of block-level sparsity across FFN layers.

Each text block is a subject and each FFN layer is a repeated measurement.
The primary profile-analysis question is flatness: are the mean sparsity
measurements equal across all layers?
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
from scipy.stats import f, friedmanchisquare, t, wilcoxon

logger = logging.getLogger(__name__)


def load_sparsity_profile(descriptive_dir: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load the Phase 1 block-by-layer table as a complete subject-by-layer matrix."""
    source = Path(descriptive_dir) / "block_sparsity_by_layer.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"Phase 1 block summary not found: {source}. Run analyze-descriptive first."
        )
    frame = pd.read_csv(source)
    required = {"block_id", "mean_sparsity", "layer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    if frame["block_id"].isna().any() or frame["layer"].isna().any():
        raise ValueError("block_id and layer must not contain missing values")

    wide = frame.pivot(index="block_id", columns="layer", values="mean_sparsity")
    wide = wide.reindex(sorted(wide.columns, key=lambda x: int(str(x).split("_")[-1])), axis=1)
    wide = wide.dropna(axis=0, how="any")
    values = wide.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError(
            f"Profile analysis needs at least 3 complete blocks and 2 layers; got {values.shape}."
        )
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Sparsity values must be finite and lie between 0 and 1.")
    return values, [str(c) for c in wide.columns], wide.index.to_numpy()


def _hotelling_flatness_test(profile: np.ndarray) -> dict[str, float | int | str]:
    """One-sample Hotelling T-squared test on layer-versus-reference contrasts.

    H0 is profile flatness: all layer means are equal. This is a multivariate
    test on p-1 within-block contrasts, rather than treating the layers as
    independent samples.
    """
    n, p = profile.shape
    q = p - 1
    if n <= q:
        raise ValueError(
            f"Flatness test needs more blocks than contrasts (n={n}, contrasts={q})."
        )
    contrasts = profile[:, 1:] - profile[:, [0]]
    mean_difference = contrasts.mean(axis=0)
    covariance = np.atleast_2d(np.cov(contrasts, rowvar=False, ddof=1))
    rank = int(np.linalg.matrix_rank(covariance))
    if rank < q:
        raise ValueError(
            "The contrast covariance matrix is singular. Collect more independent text blocks "
            "or inspect duplicate/constant layer profiles before applying the classical test."
        )
    t_squared = float(n * mean_difference @ np.linalg.solve(covariance, mean_difference))
    f_statistic = float((n - q) * t_squared / (q * (n - 1)))
    p_value = float(f.sf(f_statistic, q, n - q))
    return {
        "test": "Hotelling T-squared profile-flatness test",
        "null_hypothesis": "The mean sparsity is equal across all FFN layers.",
        "alternative_hypothesis": "At least one layer mean differs from the common profile level.",
        "n_blocks": int(n),
        "n_layers": int(p),
        "contrast_dimension": int(q),
        "covariance_rank": rank,
        "hotelling_t_squared": t_squared,
        "f_statistic": f_statistic,
        "df1": int(q),
        "df2": int(n - q),
        "p_value": p_value,
        "alpha": 0.05,
        "decision": "reject_flat_profile" if p_value < 0.05 else "do_not_reject_flat_profile",
        "assumption_note": (
            "The classical Hotelling T-squared reference distribution assumes independent "
            "sampling units and an approximately multivariate-normal contrast vector. "
            "Text blocks are the sampling units; neighbouring tokens are not treated as replicates."
        ),
    }


def _repeated_measures_anova(profile: np.ndarray, layer_names: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Repeated-measures ANOVA comparing layer sparsity across text blocks."""
    n, p = profile.shape
    grand_mean = np.mean(profile)
    block_means = np.mean(profile, axis=1)
    layer_means = np.mean(profile, axis=0)

    ss_total = float(np.sum((profile - grand_mean) ** 2))
    ss_blocks = float(p * np.sum((block_means - grand_mean) ** 2))
    ss_layers = float(n * np.sum((layer_means - grand_mean) ** 2))
    ss_error = float(ss_total - ss_blocks - ss_layers)

    df_layers = p - 1
    df_blocks = n - 1
    df_error = df_layers * df_blocks

    ms_layers = ss_layers / max(df_layers, 1)
    ms_error = ss_error / max(df_error, 1)
    f_stat = float(ms_layers / max(ms_error, 1e-15))
    p_value = float(f.sf(f_stat, df_layers, df_error))

    # Greenhouse-Geisser epsilon for sphericity correction
    cov = np.cov(profile, rowvar=False, ddof=1)
    mean_var = np.mean(np.diag(cov))
    mean_cov = (np.sum(cov) - np.sum(np.diag(cov))) / (p * (p - 1)) if p > 1 else 0.0
    numerator = p**2 * (mean_var - mean_cov) ** 2
    denominator = (p - 1) * (
        np.sum(cov**2) - 2 * p * np.sum(np.mean(cov, axis=0) ** 2) + p**2 * (np.mean(cov) ** 2)
    )
    epsilon = float(np.clip(numerator / max(denominator, 1e-15), 1.0 / (p - 1), 1.0))
    df_gg1 = epsilon * df_layers
    df_gg2 = epsilon * df_error
    p_value_gg = float(f.sf(f_stat, df_gg1, df_gg2))

    table = pd.DataFrame(
        [
            {"source": "Layers (Within-subjects)", "SS": ss_layers, "df": df_layers, "MS": ms_layers, "F": f_stat, "p_uncorrected": p_value, "p_greenhouse_geisser": p_value_gg, "epsilon_gg": epsilon},
            {"source": "Blocks (Subjects)", "SS": ss_blocks, "df": df_blocks, "MS": ss_blocks / max(df_blocks, 1), "F": np.nan, "p_uncorrected": np.nan, "p_greenhouse_geisser": np.nan, "epsilon_gg": epsilon},
            {"source": "Error (Interaction)", "SS": ss_error, "df": df_error, "MS": ms_error, "F": np.nan, "p_uncorrected": np.nan, "p_greenhouse_geisser": np.nan, "epsilon_gg": epsilon},
            {"source": "Total", "SS": ss_total, "df": n * p - 1, "MS": np.nan, "F": np.nan, "p_uncorrected": np.nan, "p_greenhouse_geisser": np.nan, "epsilon_gg": epsilon},
        ]
    )

    report = {
        "test": "Repeated-Measures ANOVA",
        "ss_layers": ss_layers,
        "ss_blocks": ss_blocks,
        "ss_error": ss_error,
        "df_layers": df_layers,
        "df_error": df_error,
        "f_statistic": f_stat,
        "p_value_uncorrected": p_value,
        "greenhouse_geisser_epsilon": epsilon,
        "p_value_greenhouse_geisser": p_value_gg,
        "decision": "reject_equal_layer_sparsity" if p_value_gg < 0.05 else "do_not_reject",
    }
    return report, table


def _friedman_test(profile: np.ndarray, layer_names: list[str]) -> dict[str, Any]:
    """Non-parametric Friedman test for repeated measures across layers."""
    res = friedmanchisquare(*[profile[:, j] for j in range(profile.shape[1])])
    return {
        "test": "Friedman test",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "df": int(profile.shape[1] - 1),
        "decision": "reject_equal_layer_sparsity" if res.pvalue < 0.05 else "do_not_reject",
        "null_hypothesis": "The distribution of activation sparsity is identical across all FFN layers.",
    }


def _posthoc_layer_tests(profile: np.ndarray, layer_names: list[str]) -> pd.DataFrame:
    """Pairwise post-hoc paired comparisons between adjacent layers and against Layer 0."""
    p = profile.shape[1]
    comparisons = []
    # Test adjacent layers
    for j in range(p - 1):
        diff = profile[:, j + 1] - profile[:, j]
        t_stat = float(np.mean(diff) / max(np.std(diff, ddof=1) / np.sqrt(len(diff)), 1e-15))
        p_val = float(2 * t.sf(abs(t_stat), df=len(diff) - 1))
        comparisons.append(
            {
                "comparison_type": "adjacent",
                "layer_a": layer_names[j],
                "layer_b": layer_names[j + 1],
                "mean_difference": float(np.mean(diff)),
                "t_statistic": t_stat,
                "p_raw": p_val,
            }
        )
    # Test against layer 0
    for j in range(1, p):
        diff = profile[:, j] - profile[:, 0]
        t_stat = float(np.mean(diff) / max(np.std(diff, ddof=1) / np.sqrt(len(diff)), 1e-15))
        p_val = float(2 * t.sf(abs(t_stat), df=len(diff) - 1))
        comparisons.append(
            {
                "comparison_type": "against_layer_00",
                "layer_a": layer_names[0],
                "layer_b": layer_names[j],
                "mean_difference": float(np.mean(diff)),
                "t_statistic": t_stat,
                "p_raw": p_val,
            }
        )
    df = pd.DataFrame(comparisons)
    # Holm-Bonferroni correction
    m = len(df)
    order = np.argsort(df["p_raw"].to_numpy())
    adj_p = np.zeros(m)
    current_max = 0.0
    for rank, idx in enumerate(order):
        p_adj = min(1.0, df.loc[idx, "p_raw"] * (m - rank))
        current_max = max(current_max, p_adj)
        adj_p[idx] = current_max
    df["p_holm_adjusted"] = adj_p
    df["significant_alpha_0_05"] = df["p_holm_adjusted"] < 0.05
    return df


def run_profile_analysis(
    *,
    descriptive_dir: str | Path,
    output_dir: str | Path,
    confidence_level: float = 0.95,
) -> Path:
    """Run RQ1 profile flatness analysis and save tables, plot, and report."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie between 0 and 1")
    profile, layer_names, block_ids = load_sparsity_profile(descriptive_dir)
    n, p = profile.shape
    means = profile.mean(axis=0)
    standard_deviations = profile.std(axis=0, ddof=1)
    standard_errors = standard_deviations / np.sqrt(n)
    critical = float(t.ppf((1 + confidence_level) / 2, df=n - 1))
    summary = pd.DataFrame(
        {
            "layer": layer_names,
            "n_blocks": n,
            "mean_sparsity": means,
            "standard_deviation": standard_deviations,
            "standard_error": standard_errors,
            "ci_lower": means - critical * standard_errors,
            "ci_upper": means + critical * standard_errors,
        }
    )
    report = _hotelling_flatness_test(profile)
    report["confidence_level"] = confidence_level
    report["descriptive_dir"] = str(Path(descriptive_dir).resolve())

    # Repeated-Measures ANOVA & Friedman test
    rm_report, rm_table = _repeated_measures_anova(profile, layer_names)
    report["repeated_measures_anova"] = rm_report

    friedman_report = _friedman_test(profile, layer_names)
    report["friedman_test"] = friedman_report

    posthoc_table = _posthoc_layer_tests(profile, layer_names)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target / "profile_layer_summary.csv", index=False)
    rm_table.to_csv(target / "repeated_measures_anova.csv", index=False)
    posthoc_table.to_csv(target / "posthoc_layer_comparisons.csv", index=False)
    pd.DataFrame(profile, index=block_ids, columns=layer_names).rename_axis("block_id").to_csv(
        target / "block_by_layer_sparsity_matrix.csv"
    )
    (target / "profile_analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(p)
    ax.errorbar(
        x,
        means,
        yerr=critical * standard_errors,
        marker="o",
        capsize=3,
        linewidth=1.6,
    )
    ax.set_xticks(x, layer_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean block-level activation sparsity")
    ax.set_xlabel("FFN layer")
    ax.set_title("Mean sparsity profile across FFN layers")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target / "mean_sparsity_profile.png", dpi=180)
    plt.close(fig)

    logger.info("RQ1 profile analysis saved to %s", target)
    return target


# Backwards-compatible, descriptive name used by the earlier scaffold.
def compare_layer_sparsity(*args: Any, **kwargs: Any) -> Path:
    """Alias for :func:`run_profile_analysis`."""
    return run_profile_analysis(*args, **kwargs)
