from pathlib import Path

import numpy as np

from sparse_ffn.analysis.descriptive import _summarize_layer


def test_summarize_layer_computes_sparsity_and_frequency(tmp_path: Path):
    layer_dir = tmp_path / "layer_00"
    layer_dir.mkdir()
    a = np.array([[0, 1, 0], [2, 0, 0]], dtype=np.float16)
    b = np.array([[3, 4, 0]], dtype=np.float16)
    np.savez_compressed(layer_dir / "batch_000000.npz", activations=a)
    np.savez_compressed(layer_dir / "batch_000001.npz", activations=b)

    summary, token_sparsity, neuron_frequency, positives, block_ids = _summarize_layer(
        layer_dir,
        histogram_sample_size=100,
        random_seed=7,
    )

    assert summary["n_tokens"] == 3
    assert summary["n_neurons"] == 3
    assert summary["zero_count"] == 5
    assert summary["positive_count"] == 4
    assert np.isclose(summary["zero_fraction"], 5 / 9)
    np.testing.assert_allclose(token_sparsity, [2 / 3, 2 / 3, 1 / 3], rtol=1e-6)
    np.testing.assert_allclose(neuron_frequency, [2 / 3, 2 / 3, 0.0], rtol=1e-6)
    np.testing.assert_array_equal(np.sort(positives), [1, 2, 3, 4])
    np.testing.assert_array_equal(block_ids, [-1, -1, -1])


def test_summarize_layer_requires_shards(tmp_path: Path):
    layer_dir = tmp_path / "layer_00"
    layer_dir.mkdir()
    try:
        _summarize_layer(layer_dir, histogram_sample_size=100, random_seed=1)
    except FileNotFoundError as exc:
        assert "No activation shards" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected FileNotFoundError")


def test_full_descriptive_pipeline_creates_expected_artifacts(tmp_path: Path):
    import json

    from sparse_ffn.analysis.descriptive import run_descriptive_analysis

    run_dir = tmp_path / "activations" / "train" / "pilot_run"
    layer_dir = run_dir / "layer_00"
    layer_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "complete", "model_id": "test/opt", "activation_function": "relu"}),
        encoding="utf-8",
    )
    np.savez_compressed(
        layer_dir / "batch_000000.npz",
        activations=np.array([[0, 1, 0], [2, 0, 0], [3, 4, 0]], dtype=np.float16),
    )

    output = run_descriptive_analysis(
        activations_root=tmp_path / "activations",
        analysis_root=tmp_path / "analysis",
        split="train",
        histogram_sample_size=100,
        random_seed=42,
        histogram_bins=10,
    )

    assert (output / "layer_summary.csv").is_file()
    assert (output / "block_sparsity_by_layer.csv").is_file()
    assert (output / "neuron_activation_frequency.csv").is_file()
    assert (output / "descriptive_report.json").is_file()
    assert (output / "plots" / "layer_sparsity.png").is_file()
    assert (output / "plots" / "layer_00" / "positive_activation_distribution.png").is_file()
