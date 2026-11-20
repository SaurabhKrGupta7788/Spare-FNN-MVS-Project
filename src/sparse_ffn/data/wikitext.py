"""Hugging Face WikiText-2 loading and preparation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sparse_ffn.config import resolve_path
from sparse_ffn.data.token_blocks import make_token_blocks

logger = logging.getLogger(__name__)


def load_hf_tokenizer(
    model_id: str,
    revision: str = "main",
    cache_dir: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load the model tokenizer directly from Hugging Face Hub."""
    token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer for {model_id} has no EOS token; cannot mark text boundaries.")
    if tokenizer.pad_token_id is None:
        # Padding is only used to batch the last, shorter token block.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_wikitext2(
    dataset_id: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    cache_dir: str | None = None,
) -> DatasetDict:
    """Download/load WikiText-2 through ``datasets.load_dataset``.

    The dataset's supplied train/validation/test partitions are retained. We do
    not randomly split the text again, which would risk data leakage.
    """
    token = os.getenv("HF_TOKEN") or None
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        cache_dir=cache_dir,
        token=token,
    )
    required = {"train", "validation", "test"}
    missing = required.difference(dataset.keys())
    if missing:
        raise ValueError(f"Dataset is missing expected WikiText-2 splits: {sorted(missing)}")
    return dataset


def _tokenize_text_batch(
    batch: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, list[list[int]]]:
    texts = batch.get("text", [])
    usable = [text if text is not None else "" for text in texts]
    encoded = tokenizer(usable, add_special_tokens=False, truncation=False)
    sequences: list[list[int]] = []
    for text, ids in zip(usable, encoded["input_ids"], strict=True):
        if text.strip() and ids:
            sequences.append([int(token) for token in ids])
        else:
            sequences.append([])
    # EOS is inserted between records in make_token_blocks, not appended here.
    return {"input_ids": sequences}


def prepare_wikitext2(
    *,
    dataset_id: str,
    dataset_config: str,
    model_id: str,
    revision: str,
    cache_dir: str | None,
    processed_dir: str | Path,
    block_size: int,
    tokenize_batch_size: int = 1000,
    overwrite: bool = False,
) -> dict[str, dict[str, int | str]]:
    """Tokenize each official split and persist fixed-length blocks to disk.

    Uses Hugging Face ``datasets`` for download, tokenization mapping and
    persistence, and ``transformers`` for the model's own tokenizer.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    output_root = resolve_path(processed_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = load_hf_tokenizer(model_id, revision=revision, cache_dir=cache_dir)
    raw = load_wikitext2(dataset_id, dataset_config, cache_dir=cache_dir)
    results: dict[str, dict[str, int | str]] = {}

    for split_name in ("train", "validation", "test"):
        split_output = output_root / split_name
        if split_output.exists() and not overwrite:
            saved = load_from_disk(str(split_output))
            results[split_name] = {
                "status": "already_prepared",
                "blocks": len(saved),
                "tokens": sum(len(row) for row in saved["input_ids"]),
                "path": str(split_output),
            }
            logger.info("Using existing prepared %s split at %s", split_name, split_output)
            continue
        if split_output.exists() and overwrite:
            import shutil

            shutil.rmtree(split_output)

        logger.info("Tokenizing WikiText-2 %s split (%d source rows)", split_name, len(raw[split_name]))
        tokenized = raw[split_name].map(
            lambda batch: _tokenize_text_batch(batch, tokenizer),
            batched=True,
            batch_size=tokenize_batch_size,
            remove_columns=raw[split_name].column_names,
            desc=f"Tokenizing {split_name}",
        )
        token_sequences = tokenized["input_ids"]
        blocks = make_token_blocks(
            token_sequences,
            block_size=block_size,
            eos_token_id=tokenizer.eos_token_id,
        )
        if not blocks:
            raise ValueError(f"No tokens were produced for WikiText-2 split: {split_name}")

        prepared = Dataset.from_dict(
            {
                "block_id": list(range(len(blocks))),
                "input_ids": blocks,
            }
        )
        prepared.save_to_disk(str(split_output))
        token_count = sum(len(block) for block in blocks)
        results[split_name] = {
            "status": "prepared",
            "source_rows": len(raw[split_name]),
            "blocks": len(blocks),
            "tokens": token_count,
            "block_size": block_size,
            "path": str(split_output),
        }
        logger.info(
            "Prepared %s: %d blocks, %d tokens -> %s",
            split_name,
            len(blocks),
            token_count,
            split_output,
        )

    return results


def load_prepared_split(processed_dir: str | Path, split: str) -> Dataset:
    """Load a prepared split from its Hugging Face ``save_to_disk`` directory."""
    path = resolve_path(processed_dir) / split
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {path}. Run `sparse-ffn prepare-data` first."
        )
    return load_from_disk(str(path))
