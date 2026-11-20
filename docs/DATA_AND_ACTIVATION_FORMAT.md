# Data and activation format

## Prepared token dataset

`data/processed/wikitext2/<split>/` is a Hugging Face `Dataset` saved with `save_to_disk`. It has one column:

- `block_id`: stable integer ID for the token block within its split.
- `input_ids`: a list of token IDs, usually of length `block_size`; the final block may be shorter.

The official WikiText-2 train, validation and test partitions are retained. Non-empty source rows are tokenized with the OPT tokenizer. The EOS ID is inserted between source rows before fixed-size blocks are created.

## Activation extraction

The OPT FFN intermediate computation is:

```text
hidden state -> fc1 -> ReLU -> fc2 -> residual path
                         ^
                         |
                  captured values
```

For each layer and batch, the extractor saves post-ReLU values immediately before `fc2`. Padding positions are removed using `attention_mask`.

Each shard is an NPZ file containing:

- `activations`: NumPy array with shape `[valid_tokens_in_shard, ffn_dim]`.
- `block_ids`: token-level block IDs aligned with the activation rows.
- `token_positions`: token offset within the original block, aligned with activation rows.

The default stored dtype is float16. Exact zeros are represented exactly in float16. The run's `manifest.json` records checkpoint, split, dimensions, data type, number of processed batches and token count.

## Descriptive analysis output

- `layer_summary.csv`: one row per FFN layer.
- `block_sparsity_by_layer.csv`: one row per token block and layer, with block-level mean/standard deviation of token-wise sparsity.
- `neuron_activation_frequency.csv`: one row per neuron and layer.
- `descriptive_report.json`: machine-readable report with interpretation limits.
- `plots/`: histograms and layer-wise comparison plots.
