"""Utilities for joining tokenized text records into fixed-size causal blocks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def make_token_blocks(
    token_sequences: Iterable[Sequence[int]],
    block_size: int,
    eos_token_id: int | None = None,
) -> list[list[int]]:
    """Flatten token sequences and split them into blocks.

    ``eos_token_id`` is inserted between non-empty source records. The final
    incomplete block is retained; the extraction collator pads it and masks the
    padding positions. This avoids silently discarding the tail of each split.
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    blocks: list[list[int]] = []
    buffer: list[int] = []
    first_record = True
    for sequence in token_sequences:
        values = [int(token) for token in sequence]
        if not values:
            continue
        if eos_token_id is not None and not first_record:
            buffer.append(int(eos_token_id))
        buffer.extend(values)
        first_record = False
        while len(buffer) >= block_size:
            blocks.append(buffer[:block_size])
            buffer = buffer[block_size:]

    if buffer:
        blocks.append(buffer)
    return blocks
