import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_ffn.analysis.rq2_neuron_dependence import (
    _pairwise_tables,
    run_neuron_dependence_analysis,
)


def test_pairwise_tables_capture_joint_marginal_and_conditional_probabilities():
    indicators = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    )
    marginal, pairs, covariance, correlation = _pairwise_tables(
        indicators, np.array([10, 11, 12])
    )
    pair_10_11 = pairs[(pairs.neuron_i == 10) & (pairs.neuron_j == 11)].iloc[0]
    assert np.isclose(pair_10_11.p_11_both_active, 0.6)
    assert np.isclose(pair_10_11.p11_under_independence, 0.36)
    assert np.isclose(pair_10_11.p_i_active_given_j_active, 1.0)
    assert correlation[0, 1] > 0.9
    assert covariance[0, 1] > 0
    assert len(marginal) == 3


def test_rq2_pipeline_writes_pairwise_and_cluster_outputs(tmp_path: Path):
    run_dir = tmp_path / "run"
    layer_dir = run_dir / "layer_00"
    layer_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "complete", "model_id": "test/opt", "activation_function": "relu"}),
        encoding="utf-8",
    )
    rng = np.random.default_rng(4)
    latent = rng.random(300) > 0.7
    data = np.column_stack(
        [latent, latent, rng.random(300) > 0.5, rng.random(300) > 0.3]
    ).astype(np.float32)
    np.savez_compressed(layer_dir / "batch_000000.npz", activations=data[:150])
    np.savez_compressed(layer_dir / "batch_000001.npz", activations=data[150:])

    output = run_neuron_dependence_analysis(
        run_dir=run_dir,
        output_dir=tmp_path / "rq2",
        max_neurons=4,
        max_tokens=300,
        min_activation_frequency=0.01,
        max_activation_frequency=0.99,
    )
    layer_out = output / "layer_00"
    assert (layer_out / "marginal_activation_probabilities.csv").is_file()
    assert (layer_out / "joint_conditional_pairwise_probabilities.csv").is_file()
    assert (layer_out / "activation_covariance_matrix.csv").is_file()
    assert (layer_out / "phi_correlation_matrix.csv").is_file()
    assert (layer_out / "sampled_token_row_indices.csv").is_file()
    assert (layer_out / "cluster_assignments.csv").is_file()
    assert (layer_out / "neuron_dendrogram.png").is_file()
    report = json.loads((layer_out / "dependence_report.json").read_text())
    assert report["n_neurons_selected"] == 4
    pairs = pd.read_csv(layer_out / "joint_conditional_pairwise_probabilities.csv")
    assert len(pairs) == 6
