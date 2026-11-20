import pytest

torch = pytest.importorskip("torch")
from sparse_ffn.data.collator import TokenBlockCollator



def test_collator_pads_and_builds_attention_mask():
    collator = TokenBlockCollator(pad_token_id=1)
    batch = collator([{"input_ids": [10, 11, 12]}, {"input_ids": [20]}])
    assert batch["input_ids"].tolist() == [[10, 11, 12], [20, 1, 1]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["block_ids"].tolist() == [0, 1]
    assert batch["input_ids"].dtype == torch.long
