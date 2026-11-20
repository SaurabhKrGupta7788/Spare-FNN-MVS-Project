from sparse_ffn.data.token_blocks import make_token_blocks


def test_token_blocks_preserve_tokens_and_insert_eos_between_rows():
    blocks = make_token_blocks([[10, 11], [12], [], [13, 14, 15]], block_size=3, eos_token_id=2)
    flattened = [token for block in blocks for token in block]
    assert flattened == [10, 11, 2, 12, 2, 13, 14, 15]
    assert all(0 < len(block) <= 3 for block in blocks)


def test_token_blocks_keep_final_incomplete_block():
    blocks = make_token_blocks([[1, 2, 3, 4, 5]], block_size=4)
    assert blocks == [[1, 2, 3, 4], [5]]


def test_token_blocks_reject_nonpositive_size():
    try:
        make_token_blocks([[1]], block_size=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")
