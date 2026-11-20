import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_ffn.analysis.rq4_weight_rank import (
    compute_spectral_entropy_effective_rank,
    compute_spectral_participation_ratio,
    load_or_extract_ffn_weights,
    run_weight_rank_analysis,
)


def test_effective_rank_boundary_cases():
    # 1. Single non-zero eigenvalue
    s_single = np.array([10.0, 0.0, 0.0, 0.0])
    assert np.isclose(compute_spectral_entropy_effective_rank(s_single), 1.0)
    assert np.isclose(compute_spectral_participation_ratio(s_single), 1.0)

    # 2. Four equal eigenvalues
    s_equal = np.array([5.0, 5.0, 5.0, 5.0])
    assert np.isclose(compute_spectral_entropy_effective_rank(s_equal), 4.0)
    assert np.isclose(compute_spectral_participation_ratio(s_equal), 4.0)


def test_rq4_pipeline_end_to_end(tmp_path: Path):
    # Setup dummy RQ3 latent structure directory
    rq3_dir = tmp_path / "rq3_latent_structure"
    rq3_dir.mkdir(parents=True)

    layers = [f"layer_{i:02d}" for i in range(4)]
    rq3_summary = pd.DataFrame(
        {
            "layer": layers,
            "n_samples": [100] * 4,
            "n_neurons": [128] * 4,
            "pca_components_computed": [16] * 4,
            "variance_explained_by_computed_components": [0.85] * 4,
            "participation_ratio": [1.5, 3.2, 4.8, 6.1],
            "entropy_effective_rank": [2.1, 4.3, 5.9, 7.8],
            "components_for_80_percent": [1, 2, 3, 4],
            "components_for_90_percent": [2, 4, 6, 8],
            "components_for_95_percent": [3, 5, 8, 10],
            "reconstruction_error_k05": [0.15, 0.20, 0.25, 0.30],
            "reconstruction_error_k10": [0.08, 0.12, 0.15, 0.18],
            "subspace_stability_k05": [0.92, 0.88, 0.85, 0.81],
            "subspace_stability_k10": [0.89, 0.84, 0.80, 0.76],
        }
    )
    rq3_summary.to_csv(rq3_dir / "pca_layer_summary.csv", index=False)

    out_dir = tmp_path / "rq4_weight_rank"
    result = run_weight_rank_analysis(
        analysis_dir=tmp_path,
        output_dir=out_dir,
        seed=101,
    )

    assert (out_dir / "weight_svd_summary.csv").is_file()
    assert (out_dir / "weight_singular_value_spectra.csv").is_file()
    assert (out_dir / "activation_vs_weight_dimensionality.csv").is_file()
    assert (out_dir / "activation_vs_weight_rank.png").is_file()
    assert (out_dir / "rq4_weight_rank_report.json").is_file()

    report = json.loads((out_dir / "rq4_weight_rank_report.json").read_text())
    metrics = report["correlation_metrics"]
    assert -1.0 <= metrics["spearman_correlation_entropy_effective_rank"] <= 1.0
    assert -1.0 <= metrics["pearson_correlation_entropy_effective_rank"] <= 1.0
