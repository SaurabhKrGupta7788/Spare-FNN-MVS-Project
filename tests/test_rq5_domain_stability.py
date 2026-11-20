import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_ffn.analysis.rq5_domain_stability import (
    box_m_test,
    compute_manova,
    hotelling_two_sample_t2,
    run_domain_stability_analysis,
)


def test_hotelling_two_sample_t2_same_distribution():
    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=0.0, scale=1.0, size=(100, 4))
    x2 = rng.normal(loc=0.0, scale=1.0, size=(100, 4))
    res = hotelling_two_sample_t2(x1, x2)
    assert res["p_value"] > 0.05
    assert res["decision"] == "do_not_reject"


def test_hotelling_two_sample_t2_different_means():
    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=0.0, scale=1.0, size=(100, 4))
    x2 = rng.normal(loc=2.0, scale=1.0, size=(100, 4))
    res = hotelling_two_sample_t2(x1, x2)
    assert res["p_value"] < 0.001
    assert res["decision"] == "reject_equal_means"


def test_manova_and_box_m():
    rng = np.random.default_rng(42)
    domain_data = {
        "domain_a": rng.normal(loc=0.0, scale=1.0, size=(80, 5)),
        "domain_b": rng.normal(loc=0.5, scale=1.0, size=(80, 5)),
        "domain_c": rng.normal(loc=1.0, scale=1.0, size=(80, 5)),
    }
    manova = compute_manova(domain_data)
    assert 0.0 < manova["wilks_lambda"] <= 1.0
    assert manova["pillai_trace"] > 0.0
    assert manova["p_value"] < 0.05

    box = box_m_test(domain_data)
    assert box["chi2_statistic"] >= 0.0
    assert 0.0 <= box["p_value"] <= 1.0


def test_rq5_pipeline_end_to_end(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "layer_00").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "complete", "model_id": "test/opt"}), encoding="utf-8"
    )

    rng = np.random.default_rng(123)
    acts = np.maximum(rng.normal(loc=0.0, scale=1.0, size=(120, 32)), 0.0).astype(np.float32)
    np.savez_compressed(run_dir / "layer_00" / "batch_000000.npz", activations=acts)

    out_dir = tmp_path / "rq5_domain_stability"
    result = run_domain_stability_analysis(
        run_dir=run_dir,
        output_dir=out_dir,
        max_samples=100,
        pca_reduced_components=4,
        target_layer="layer_00",
        random_seed=42,
    )

    assert (out_dir / "domain_mean_vectors.csv").is_file()
    assert (out_dir / "domain_covariance_matrices.csv").is_file()
    assert (out_dir / "pairwise_hotelling_t2.csv").is_file()
    assert (out_dir / "manova_summary.csv").is_file()
    assert (out_dir / "box_m_test.csv").is_file()
    assert (out_dir / "lda_classification_report.csv").is_file()
    assert (out_dir / "lda_domain_separation.png").is_file()
    assert (out_dir / "rq5_domain_stability_report.json").is_file()

    report = json.loads((out_dir / "rq5_domain_stability_report.json").read_text())
    assert report["pca_reduced_components"] == 4
    assert len(report["pairwise_hotelling_t2"]) == 3
