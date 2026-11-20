import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_ffn.analysis.rq1_layer_comparison import (
    _hotelling_flatness_test,
    load_sparsity_profile,
    run_profile_analysis,
)


def test_hotelling_profile_test_does_not_reject_flat_profile():
    rng = np.random.default_rng(7)
    # Every layer has the same population mean; blocks share a random effect.
    block_effect = rng.normal(0, 0.015, size=(50, 1))
    profile = 0.8 + block_effect + rng.normal(0, 0.02, size=(50, 6))
    result = _hotelling_flatness_test(profile)
    assert result["p_value"] > 0.05
    assert result["n_blocks"] == 50
    assert result["n_layers"] == 6


def test_hotelling_profile_test_detects_nonflat_profile():
    rng = np.random.default_rng(9)
    profile = rng.normal(0, 0.01, size=(60, 5)) + np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    result = _hotelling_flatness_test(profile)
    assert result["p_value"] < 0.001
    assert result["decision"] == "reject_flat_profile"


def test_profile_pipeline_reads_phase1_table_and_writes_outputs(tmp_path: Path):
    descriptive = tmp_path / "descriptive"
    descriptive.mkdir()
    rows = []
    rng = np.random.default_rng(11)
    for block in range(20):
        for layer_idx in range(4):
            rows.append(
                {
                    "block_id": block,
                    "layer": f"layer_{layer_idx:02d}",
                    "mean_sparsity": 0.6 + 0.03 * layer_idx + rng.normal(0, 0.01),
                }
            )
    pd.DataFrame(rows).to_csv(descriptive / "block_sparsity_by_layer.csv", index=False)

    profile, layers, block_ids = load_sparsity_profile(descriptive)
    assert profile.shape == (20, 4)
    assert layers == ["layer_00", "layer_01", "layer_02", "layer_03"]
    assert len(block_ids) == 20

    output = run_profile_analysis(descriptive_dir=descriptive, output_dir=tmp_path / "rq1")
    assert (output / "profile_layer_summary.csv").is_file()
    assert (output / "block_by_layer_sparsity_matrix.csv").is_file()
    assert (output / "mean_sparsity_profile.png").is_file()
    report = json.loads((output / "profile_analysis_report.json").read_text())
    assert report["test"] == "Hotelling T-squared profile-flatness test"
