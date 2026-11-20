"""Command-line interface for the sparse FFN MVS project."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from sparse_ffn.analysis.descriptive import find_latest_complete_run, run_descriptive_analysis
from sparse_ffn.analysis.rq1_layer_comparison import run_profile_analysis
from sparse_ffn.analysis.rq2_neuron_dependence import run_neuron_dependence_analysis
from sparse_ffn.analysis.rq3_latent_structure import run_latent_structure_analysis
from sparse_ffn.analysis.rq4_weight_rank import run_weight_rank_analysis
from sparse_ffn.analysis.rq5_domain_stability import run_domain_stability_analysis
from sparse_ffn.config import load_config
from sparse_ffn.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _prepare(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from sparse_ffn.data.wikitext import prepare_wikitext2

    hf = _section(config, "huggingface")
    data = _section(config, "data")
    result = prepare_wikitext2(
        dataset_id=hf.get("dataset_id", "Salesforce/wikitext"),
        dataset_config=hf.get("dataset_config", "wikitext-2-raw-v1"),
        model_id=hf.get("model_id", "facebook/opt-125m"),
        revision=hf.get("revision", "main"),
        cache_dir=hf.get("cache_dir"),
        processed_dir=data.get("processed_dir", "data/processed/wikitext2"),
        block_size=int(data.get("block_size", 128)),
        tokenize_batch_size=int(data.get("tokenize_batch_size", 1000)),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


def _extract(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    # Import Hugging Face/PyTorch-dependent code only when extraction is requested.
    from sparse_ffn.extraction import extract_activations

    hf = _section(config, "huggingface")
    data = _section(config, "data")
    extraction = _section(config, "extraction")
    max_batches = args.max_batches if args.max_batches is not None else int(extraction.get("max_batches", 20))
    run_dir = extract_activations(
        model_id=hf.get("model_id", "facebook/opt-125m"),
        revision=hf.get("revision", "main"),
        cache_dir=hf.get("cache_dir"),
        processed_dir=data.get("processed_dir", "data/processed/wikitext2"),
        output_dir=extraction.get("output_dir", "artifacts/activations"),
        split=args.split,
        batch_size=int(args.batch_size or extraction.get("batch_size", 2)),
        num_workers=int(extraction.get("num_workers", 0)),
        max_batches=max_batches,
        storage_dtype=args.storage_dtype or extraction.get("storage_dtype", "float16"),
        device=args.device or extraction.get("device", "auto"),
    )
    print(f"Activation run saved to: {run_dir}")
    return run_dir


def _analyze(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    extraction = _section(config, "extraction")
    analysis = _section(config, "analysis")
    output = run_descriptive_analysis(
        activations_root=extraction.get("output_dir", "artifacts/activations"),
        analysis_root=analysis.get("output_dir", "artifacts/analysis"),
        split=args.split,
        run_dir=args.run_dir,
        histogram_sample_size=int(analysis.get("histogram_sample_size", 100_000)),
        random_seed=int(analysis.get("random_seed", 42)),
        histogram_bins=int(analysis.get("histogram_bins", 60)),
    )
    print(f"Descriptive analysis saved to: {output}")
    return output



def _resolve_analysis_run(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve the extraction run and ensure Phase 1 summaries exist."""
    extraction = _section(config, "extraction")
    analysis = _section(config, "analysis")
    run_dir = Path(args.run_dir).resolve() if getattr(args, "run_dir", None) else find_latest_complete_run(
        extraction.get("output_dir", "artifacts/activations"), args.split
    )
    descriptive_dir = (
        Path(analysis.get("output_dir", "artifacts/analysis")).resolve()
        / args.split
        / run_dir.name
        / "descriptive"
    )
    if not (descriptive_dir / "block_sparsity_by_layer.csv").is_file():
        descriptive_dir = run_descriptive_analysis(
            activations_root=extraction.get("output_dir", "artifacts/activations"),
            analysis_root=analysis.get("output_dir", "artifacts/analysis"),
            split=args.split,
            run_dir=run_dir,
            histogram_sample_size=int(analysis.get("histogram_sample_size", 100_000)),
            random_seed=int(analysis.get("random_seed", 42)),
            histogram_bins=int(analysis.get("histogram_bins", 60)),
        )
    return run_dir, descriptive_dir


def _run_rq1(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    run_dir, descriptive_dir = _resolve_analysis_run(args, config)
    analysis = _section(config, "analysis")
    output = (
        Path(analysis.get("output_dir", "artifacts/analysis")).resolve()
        / args.split / run_dir.name / "rq1_profile_analysis"
    )
    result = run_profile_analysis(
        descriptive_dir=descriptive_dir,
        output_dir=output,
        confidence_level=float(_section(config, "rq1").get("confidence_level", 0.95)),
    )
    print(f"RQ1 profile-analysis outputs saved to: {result}")
    return result


def _run_rq2(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    run_dir, _ = _resolve_analysis_run(args, config)
    analysis = _section(config, "analysis")
    settings = _section(config, "rq2")
    output = (
        Path(analysis.get("output_dir", "artifacts/analysis")).resolve()
        / args.split / run_dir.name / "rq2_neuron_dependence"
    )
    result = run_neuron_dependence_analysis(
        run_dir=run_dir,
        output_dir=output,
        max_neurons=int(settings.get("max_neurons", 128)),
        max_tokens=int(settings.get("max_tokens", 50_000)),
        min_activation_frequency=float(settings.get("min_activation_frequency", 0.01)),
        max_activation_frequency=float(settings.get("max_activation_frequency", 0.99)),
        random_seed=int(settings.get("random_seed", 42)),
        cluster_distance_threshold=float(settings.get("cluster_distance_threshold", 0.7)),
    )
    print(f"RQ2 neuron-dependence outputs saved to: {result}")
    return result


def _run_rq3(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    run_dir, _ = _resolve_analysis_run(args, config)
    analysis = _section(config, "analysis")
    settings = _section(config, "rq3")
    output = (
        Path(analysis.get("output_dir", "artifacts/analysis")).resolve()
        / args.split / run_dir.name / "rq3_latent_structure"
    )
    result = run_latent_structure_analysis(
        run_dir=run_dir,
        output_dir=output,
        max_samples=int(settings.get("max_samples", 5_000)),
        pca_components=int(settings.get("pca_components", 64)),
        factor_analysis_max_neurons=int(settings.get("factor_analysis_max_neurons", 256)),
        factor_counts=tuple(int(value) for value in settings.get("factor_counts", [2, 5, 10])),
        min_activation_frequency=float(settings.get("min_activation_frequency", 0.01)),
        max_activation_frequency=float(settings.get("max_activation_frequency", 0.99)),
        rotation=settings.get("rotation", "varimax"),
        max_iter=int(settings.get("max_iter", 200)),
        random_seed=int(settings.get("random_seed", 42)),
    )
    print(f"RQ3 latent-structure outputs saved to: {result}")
    return result


def _run_rq4(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    run_dir, descriptive_dir = _resolve_analysis_run(args, config)
    analysis = _section(config, "analysis")
    settings = _section(config, "rq4")
    analysis_dir = (
        Path(analysis.get("output_dir", "artifacts/analysis")).resolve()
        / args.split / run_dir.name
    )
    # Ensure RQ3 outputs exist
    if not (analysis_dir / "rq3_latent_structure" / "pca_layer_summary.csv").is_file():
        _run_rq3(args, config)
    output = analysis_dir / "rq4_weight_rank"
    result = run_weight_rank_analysis(
        analysis_dir=analysis_dir,
        output_dir=output,
        weights_path=getattr(args, "weights_path", None) or settings.get("weights_path"),
        seed=int(settings.get("seed", 42)),
    )
    print(f"RQ4 weight-rank outputs saved to: {result}")
    return result


def _run_rq5(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    run_dir, _ = _resolve_analysis_run(args, config)
    analysis = _section(config, "analysis")
    settings = _section(config, "rq5")
    output = (
        Path(analysis.get("output_dir", "artifacts/analysis")).resolve()
        / args.split / run_dir.name / "rq5_domain_stability"
    )
    result = run_domain_stability_analysis(
        run_dir=run_dir,
        output_dir=output,
        max_samples=int(settings.get("max_samples", 1000)),
        pca_reduced_components=int(settings.get("pca_reduced_components", 8)),
        target_layer=getattr(args, "target_layer", None) or settings.get("target_layer", "layer_00"),
        random_seed=int(settings.get("random_seed", 42)),
    )
    print(f"RQ5 domain-stability outputs saved to: {result}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparse-ffn",
        description="Extract and statistically describe sparse activations in dense OPT FFN layers.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Download/tokenize WikiText-2 using Hugging Face.")
    prepare.add_argument("--overwrite", action="store_true", help="Rebuild cached token blocks.")

    extract = subparsers.add_parser("extract-activations", help="Extract post-ReLU FFN activations.")
    extract.add_argument("--split", choices=["train", "validation", "test"], default="train")
    extract.add_argument("--max-batches", type=int, default=None,
                         help="Batches to process; 0 means the entire split.")
    extract.add_argument("--batch-size", type=int, default=None)
    extract.add_argument("--storage-dtype", choices=["float16", "float32"], default=None)
    extract.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)

    analyze = subparsers.add_parser("analyze-descriptive", help="Create Phase 1 descriptive tables/plots.")
    analyze.add_argument("--split", choices=["train", "validation", "test"], default="train")
    analyze.add_argument("--run-dir", default=None, help="Specific extraction run; default selects latest complete run.")

    run_all = subparsers.add_parser("run-all", help="Prepare data, extract a pilot, and run descriptive analysis.")
    run_all.add_argument("--overwrite-data", action="store_true")
    run_all.add_argument("--split", choices=["train", "validation", "test"], default="train")
    run_all.add_argument("--max-batches", type=int, default=None,
                         help="Batches to process; 0 means the entire split.")
    run_all.add_argument("--batch-size", type=int, default=None)
    run_all.add_argument("--storage-dtype", choices=["float16", "float32"], default=None)
    run_all.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)

    for command_name, help_text in (
        ("analyze-rq1", "Run RQ1 profile analysis of block-level sparsity across layers."),
        ("analyze-rq2", "Run RQ2 neuron joint/marginal/conditional dependence analysis."),
        ("analyze-rq3", "Run RQ3 covariance PCA and exploratory factor analysis."),
        ("analyze-rq4", "Run RQ4 comparison of activation and weight effective dimensions."),
        ("analyze-rq5", "Run RQ5 cross-domain stability analysis (Hotelling T2, MANOVA, Box M, LDA)."),
        ("analyze-rq1-rq3", "Run the first three research-question analyses."),
        ("analyze-all", "Run all five research-question analyses (RQ1 to RQ5)."),
    ):
        command = subparsers.add_parser(command_name, help=help_text)
        command.add_argument("--split", choices=["train", "validation", "test"], default="train")
        command.add_argument("--run-dir", default=None, help="Specific completed activation run.")
        if command_name in ("analyze-rq4", "analyze-all"):
            command.add_argument("--weights-path", default=None, help="Path to pre-saved FFN weights (.npz).")
        if command_name in ("analyze-rq5", "analyze-all"):
            command.add_argument("--target-layer", default=None, help="Target layer for domain stability.")

    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    # Run commands from the repository root even when the config path is absolute.
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    import os

    os.chdir(project_root)

    if args.command == "prepare-data":
        _prepare(args, config)
    elif args.command == "extract-activations":
        _extract(args, config)
    elif args.command == "analyze-descriptive":
        _analyze(args, config)
    elif args.command == "run-all":
        from argparse import Namespace

        _prepare(Namespace(overwrite=args.overwrite_data), config)
        run_dir = _extract(args, config)
        _analyze(Namespace(split=args.split, run_dir=str(run_dir)), config)
    elif args.command == "analyze-rq1":
        _run_rq1(args, config)
    elif args.command == "analyze-rq2":
        _run_rq2(args, config)
    elif args.command == "analyze-rq3":
        _run_rq3(args, config)
    elif args.command == "analyze-rq4":
        _run_rq4(args, config)
    elif args.command == "analyze-rq5":
        _run_rq5(args, config)
    elif args.command == "analyze-rq1-rq3":
        _run_rq1(args, config)
        _run_rq2(args, config)
        _run_rq3(args, config)
    elif args.command == "analyze-all":
        _run_rq1(args, config)
        _run_rq2(args, config)
        _run_rq3(args, config)
        _run_rq4(args, config)
        _run_rq5(args, config)
    else:  # pragma: no cover - argparse enforces known commands
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
