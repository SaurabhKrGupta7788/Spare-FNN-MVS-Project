"""Padding collator for token blocks of unequal final lengths."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class TokenBlockCollator:
    """Pad token blocks dynamically and return an attention mask."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        max_length = max(len(item["input_ids"]) for item in features)
        if max_length == 0:
            raise ValueError("Encountered an empty token block.")
        input_ids = torch.full(
            (len(features), max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        block_ids = torch.empty((len(features),), dtype=torch.long)
        for row, item in enumerate(features):
            block_ids[row] = int(item.get("block_id", row))
            values = torch.as_tensor(item["input_ids"], dtype=torch.long)
            length = values.numel()
            input_ids[row, :length] = values
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "block_ids": block_ids}
