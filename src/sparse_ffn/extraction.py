"""Extract post-ReLU FFN activations from a frozen Hugging Face OPT model."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sparse_ffn.config import resolve_path
from sparse_ffn.data.collator import TokenBlockCollator
from sparse_ffn.data.wikitext import load_hf_tokenizer, load_prepared_split
from sparse_ffn.modeling.opt_adapter import get_opt_ffn_layers, layer_metadata, load_opt_causal_lm

logger = logging.getLogger(__name__)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "--").replace(":", "-")


def extract_activations(
    *,
    model_id: str,
    revision: str,
    cache_dir: str | None,
    processed_dir: str | Path,
    output_dir: str | Path,
    split: str = "train",
    batch_size: int = 2,
    num_workers: int = 0,
    max_batches: int = 20,
    storage_dtype: str = "float16",
    device: str = "auto",
) -> Path:
    """Extract each token's post-ReLU FFN vector and save one shard per batch/layer.

    The model is frozen and evaluated only in inference mode. The hook is placed
    on ``fc1`` and applies the OPT layer's configured activation function before
    saving, which captures the intermediate FFN vector immediately before ``fc2``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_batches < 0:
        raise ValueError("max_batches must be zero (all batches) or a positive integer")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be 'float16' or 'float32'")

    torch_dtype = torch.float16 if storage_dtype == "float16" else torch.float32
    token = os.getenv("HF_TOKEN") or None
    model, resolved_device = load_opt_causal_lm(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
        hf_token=token,
    )
    layers = get_opt_ffn_layers(model)
    metadata = layer_metadata(model, layers)
    # Call the base OPT transformer, not the causal-LM wrapper, because the
    # vocabulary projection/logits are not needed for activation statistics.
    forward_model = getattr(model, "model", model)

    tokenizer = load_hf_tokenizer(model_id, revision=revision, cache_dir=cache_dir)
    dataset = load_prepared_split(processed_dir, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=TokenBlockCollator(tokenizer.pad_token_id),
        pin_memory=resolved_device.type == "cuda",
    )

    run_id = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(output_dir) / split / f"{run_id}_{_safe_model_name(model_id)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for layer_info in metadata:
        (run_dir / layer_info["layer_name"]).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "status": "incomplete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "model_type": model.config.model_type,
        "activation_function": model.config.activation_function,
        "dataset_split": split,
        "prepared_dataset_path": str(resolve_path(processed_dir) / split),
        "device": str(resolved_device),
        "storage_dtype": storage_dtype,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "number_of_layers": len(layers),
        "layer_metadata": metadata,
        "tokens_extracted": 0,
        "batches_extracted": 0,
        "note": "Activations are post-ReLU and pre-fc2; padding positions are excluded.",
        "shard_metadata": "Each token row includes block_id and position within that token block.",
    }
    _write_json(run_dir / "manifest.json", manifest)

    try:
        progress = tqdm(loader, desc=f"Extracting {split} activations")
        for batch_index, batch in enumerate(progress):
            if max_batches and batch_index >= max_batches:
                break
            input_ids = batch["input_ids"].to(resolved_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(resolved_device, non_blocking=True)
            batch_size_actual, sequence_length = input_ids.shape
            captured: dict[int, torch.Tensor] = {}
            handles = []

            for layer_index, layer in enumerate(layers):
                def make_hook(index: int, layer_ref: torch.nn.Module):
                    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor):
                        activated = layer_ref.activation_fn(output.detach())
                        shaped = activated.reshape(batch_size_actual, sequence_length, -1)
                        captured[index] = shaped.to(device="cpu", dtype=torch_dtype)
                    return hook

                handles.append(layer.fc1.register_forward_hook(make_hook(layer_index, layer)))

            try:
                with torch.inference_mode():
                    forward_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
            finally:
                for handle in handles:
                    handle.remove()

            valid_mask = attention_mask.detach().to(device="cpu", dtype=torch.bool)
            block_ids = batch["block_ids"].detach().cpu()
            expanded_block_ids = block_ids[:, None].expand(batch_size_actual, sequence_length)
            token_positions = torch.arange(sequence_length).expand(batch_size_actual, sequence_length)
            valid_block_ids = expanded_block_ids[valid_mask].numpy()
            valid_token_positions = token_positions[valid_mask].numpy()
            for layer_index, layer_info in enumerate(metadata):
                if layer_index not in captured:
                    raise RuntimeError(f"No activations were captured for layer {layer_index}.")
                values = captured[layer_index]
                valid_activations = values[valid_mask].contiguous().numpy()
                shard_path = run_dir / layer_info["layer_name"] / f"batch_{batch_index:06d}.npz"
                np.savez_compressed(
                    shard_path,
                    activations=valid_activations,
                    block_ids=valid_block_ids,
                    token_positions=valid_token_positions,
                )
            manifest["tokens_extracted"] += int(valid_mask.sum().item())
            manifest["batches_extracted"] += 1
            progress.set_postfix(tokens=manifest["tokens_extracted"])

        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "manifest.json", manifest)
        raise

    logger.info(
        "Saved %d activation tokens from %d batches across %d layers to %s",
        manifest["tokens_extracted"],
        manifest["batches_extracted"],
        len(layers),
        run_dir,
    )
    return run_dir
