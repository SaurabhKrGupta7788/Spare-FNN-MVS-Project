import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_ffn.analysis.rq3_latent_structure import run_latent_structure_analysis


def test_rq3_pca_and_factor_analysis_outputs(tmp_path: Path):
    rng = np.random.default_rng(12)
    n_samples, n_features = 180, 14
    latent = rng.normal(size=(n_samples, 3))
    weights = rng.normal(size=(3, n_features))
    dense = latent @ weights + rng.normal(scale=0.12, size=(n_samples, n_features))
    # ReLU-like sparse observations, with enough non-zero prevalence for factor analysis.
    activations = np.maximum(dense + 0.7, 0).astype(np.float32)

    run_dir = tmp_path / "run"
    (run_dir / "layer_00").mkdir(parents=True)
    (run_dir / "layer_01").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "complete", "model_id": "test/opt"}), encoding="utf-8"
    )
    for layer in ["layer_00", "layer_01"]:
        np.savez_compressed(run_dir / layer / "batch_000000.npz", activations=activations[:90])
        np.savez_compressed(run_dir / layer / "batch_000001.npz", activations=activations[90:])

    output = run_latent_structure_analysis(
        run_dir=run_dir,
        output_dir=tmp_path / "rq3",
        max_samples=150,
        pca_components=8,
        factor_analysis_max_neurons=12,
        factor_counts=[2, 3],
        min_activation_frequency=0.01,
        max_activation_frequency=0.99,
        max_iter=100,
    )
    assert (output / "pca_eigenvalues.csv").is_file()
    assert (output / "pca_layer_summary.csv").is_file()
    assert (output / "factor_analysis_model_summary.csv").is_file()
    assert (output / "rq3_latent_structure_report.json").is_file()
    assert (output / "layer_00" / "pca_scree_and_cumulative_variance.png").is_file()
    pca = pd.read_csv(output / "pca_eigenvalues.csv")
    assert set(pca.layer) == {"layer_00", "layer_01"}
    assert pca.groupby("layer").component.max().eq(8).all()
    report = json.loads((output / "rq3_latent_structure_report.json").read_text())
    assert report["n_token_rows_used"] == 150
