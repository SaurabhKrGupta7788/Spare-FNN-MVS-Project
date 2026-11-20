# Sparse FFN MVS Project

A research codebase for the project:

**Statistical Characterisation of Sparse Activations and Low-Rank Latent Structures in Dense Feed-Forward Neural Networks**

The repository is deliberately organised as a small research project rather than a single notebook. It uses the Hugging Face ecosystem to obtain WikiText-2, load the model and tokenizer, prepare/cache token blocks, and extract intermediate activations.

## Current implementation status

| Part | Status | What it does |
|---|---|---|
| Hugging Face data loading | Implemented | Downloads WikiText-2 with its official train/validation/test splits |
| Tokenization and preparation | Implemented | Uses the OPT tokenizer and saves token blocks with `datasets.save_to_disk` |
| Model loading | Implemented | Loads frozen `facebook/opt-125m` using `AutoModelForCausalLM` |
| FFN activation extraction | Implemented | Captures post-ReLU, pre-`fc2` activations for every OPT FFN layer |
| Phase 1 descriptive analysis | Implemented | Sparsity, activation-frequency distributions, summary statistics, CSVs and plots |
| RQ1 profile analysis | Implemented | Block-by-layer sparsity profiles, Hotelling's $T^2$ flatness test, Repeated-Measures ANOVA, and Friedman test |
| RQ2 neuron dependence | Implemented | Empirical joint, marginal and conditional probabilities, phi correlation and hierarchical clustering |
| RQ3 latent structure | Implemented | Covariance-based PCA, participation ratio, entropy-based effective rank, reconstruction error, and factor analysis |
| RQ4 activation/weight rank | Implemented | Weight matrix SVD, singular spectra, effective dimensionality, and Spearman/Pearson rank association |
| RQ5 cross-domain stability | Implemented | Multi-domain activation analysis, Hotelling's $T^2$, One-Way MANOVA, Box's M test, and Linear Discriminant Analysis |

All five research questions from the Project Proposal are now fully implemented with executable analysis commands, rigorous multivariate statistical tests, and comprehensive unit test coverage.

## Why OPT-125M?

The default Hugging Face checkpoint is [`facebook/opt-125m`](https://huggingface.co/facebook/opt-125m). Its Hub configuration specifies 12 Transformer layers, 3,072 FFN intermediate units and a ReLU activation. ReLU gives exact zeros, so the first analysis can measure sparsity without selecting an arbitrary near-zero threshold. The implementation validates these assumptions and stops rather than silently applying the OPT extractor to an incompatible architecture.

The code captures the activation **after ReLU and before the second FFN projection (`fc2`)**. This is the intermediate neuron vector that the project is studying. It does not capture the output of the whole FFN residual block.

## 1. Environment setup

Python 3.10 or newer is recommended. Create a clean environment:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch first using the selector for your machine (CPU or the appropriate CUDA build):

<https://pytorch.org/get-started/locally/>

Then install this project:

```bash
pip install -e ".[dev]"
```

The project uses these Hugging Face packages:

- `datasets`: load WikiText-2 and save prepared token blocks.
- `transformers`: load the OPT tokenizer and causal language model.
- `huggingface-hub`: Hub download/cache and optional authentication support.
- `scikit-learn`: PCA and exploratory Gaussian factor analysis for RQ3.

WikiText-2 and OPT-125M are public. A Hugging Face token is not normally needed. If you use a gated/private resource later, set `HF_TOKEN` as an environment variable; do not place tokens in YAML or commit them.

## 2. Run the complete Phase 1 pipeline

From the repository root:

```bash
python -m sparse_ffn --config configs/default.yaml run-all
```

This executes three stages:

1. Download and prepare the official WikiText-2 train/validation/test splits.
2. Extract a small pilot of train-split activations.
3. Generate Phase 1 descriptive tables and plots.

The default is intentionally a pilot (`max_batches: 20`, `batch_size: 2`, `block_size: 128`) to check that the complete pipeline works before creating a large activation archive.

## 3. Run each stage independently

### Stage A — prepare data

```bash
python -m sparse_ffn prepare-data
```

This uses `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")`, then the matching OPT tokenizer. WikiText-2's existing train, validation and test splits are preserved; the project does not randomly re-split them. Each non-empty source text row is tokenized and terminated by the tokenizer's EOS token before blocks are formed. The final short block is retained and padded only when batching, with padding positions excluded from activation statistics.

Prepared splits are saved with Hugging Face `Dataset.save_to_disk` under:

```text
data/processed/wikitext2/
  train/
  validation/
  test/
```

To rebuild them:

```bash
python -m sparse_ffn prepare-data --overwrite
```

### Stage B — extract activations

```bash
python -m sparse_ffn extract-activations --split train
```

Options:

```bash
# Process 100 batches instead of the 20-batch pilot
python -m sparse_ffn extract-activations --split train --max-batches 100

# Process the entire prepared split (can consume substantial disk space)
python -m sparse_ffn extract-activations --split train --max-batches 0

# Change batch size or force CPU
python -m sparse_ffn extract-activations --split train --batch-size 1 --device cpu
```

Each extraction creates a unique run folder:

```text
artifacts/activations/train/<timestamp>_facebook--opt-125m/
  manifest.json
  layer_00/batch_000000.npz
  layer_01/batch_000000.npz
  ...
  layer_11/...
```

Each NPZ shard contains an `activations` matrix with shape `[valid_tokens, 3072]`, plus aligned `block_ids` and `token_positions` arrays. The IDs make later chunk-level inference possible. It stores float16 by default to reduce disk use; set `storage_dtype: float32` in the YAML or pass `--storage-dtype float32` for higher-precision descriptive values. Zeros are preserved exactly in float16.

**Storage note:** all 12 FFN layers are saved. Before compression, the activation archive is approximately `number_of_tokens × 12 × 3,072 × 2 bytes` with float16 storage. For example, 100,000 tokens would be about 7.4 GB before compression. A full WikiText-2 extraction can therefore require tens of gigabytes or more. Start with the default pilot, inspect the outputs, and increase `max_batches` gradually. Use `--max-batches 0` only when sufficient disk space is available.

### Stage C — Phase 1 descriptive analysis

```bash
python -m sparse_ffn analyze-descriptive --split train
```

By default, this selects the latest completed extraction run. To analyse a particular run, pass its folder:

```bash
python -m sparse_ffn analyze-descriptive --split train --run-dir "artifacts/activations/train/<run-folder>"
```

Outputs are written to:

```text
artifacts/analysis/train/<run-folder>/descriptive/
  layer_summary.csv
  block_sparsity_by_layer.csv
  neuron_activation_frequency.csv
  descriptive_report.json
  plots/
    layer_sparsity.png
    layer_00/
      token_sparsity_distribution.png
      neuron_activation_frequency.png
      positive_activation_distribution.png
    ...
```

## 4. What Phase 1 measures

For an activation matrix `H` with `n` token rows and `d` neurons:

- **Zero fraction:** proportion of all neuron-token activations equal to zero.
- **Token-wise sparsity:** for each token, the proportion of its `d` neurons that are zero.
- **Neuron activation frequency:** for each neuron, the proportion of tokens on which it is positive.
- **Activation magnitude:** mean and standard deviation over all activations and over positive activations.
- **Positive-value distribution:** sample-based quantiles, skewness and excess kurtosis.

The main output, `layer_summary.csv`, gives one row per FFN layer. The plots show how sparsity changes across depth and how token-wise sparsity and neuron activity frequencies are distributed within each layer.

The positive activation quantiles and shape statistics are estimated from a bounded random sample to avoid loading a full multi-gigabyte activation matrix into memory. The CSV marks this with `positive_quantiles_are_sampled=True` and reports the sample size.

### Important interpretation limits

Phase 1 is descriptive only. RQ1 adds a block-level profile-flatness test; RQ2 describes pairwise dependence on a selected neuron subset; RQ3 investigates covariance-based dimension reduction and exploratory factor models.

Tokens from the same document are not independent observations. RQ1 therefore uses the text block as its repeated-measure unit. RQ2 and RQ3 are descriptive/model-exploratory analyses and do not treat token rows as independent replicates for formal significance testing.

## 5. Extending the project for RQ1–RQ5

The future modules are under `src/sparse_ffn/analysis/`:

```text
rq1_layer_comparison.py
rq2_neuron_dependence.py
rq3_latent_structure.py
rq4_weight_rank.py
rq5_domain_stability.py
```

### Available Analysis Commands:

```bash
# Phase 1: Descriptive analysis
python -m sparse_ffn analyze-descriptive --split train

# RQ1: Profile flatness, Repeated-Measures ANOVA, and Friedman test
python -m sparse_ffn analyze-rq1 --split train

# RQ2: Neuron marginal, joint, conditional probabilities, phi correlation, clustering
python -m sparse_ffn analyze-rq2 --split train

# RQ3: Covariance PCA, participation ratio, entropy-based effective rank, factor analysis
python -m sparse_ffn analyze-rq3 --split train

# RQ4: Activation vs weight matrix SVD, singular spectra, and rank correlation
python -m sparse_ffn analyze-rq4 --split train

# RQ5: Cross-domain stability: Hotelling's T^2, One-Way MANOVA, Box's M, LDA
python -m sparse_ffn analyze-rq5 --split train

# Run all analyses sequentially (RQ1 through RQ5)
python -m sparse_ffn analyze-all --split train
```


## 6. Repository structure

```text
sparse_ffn_mvs_project/
├── configs/
│   └── default.yaml
├── docs/
│   ├── DATA_AND_ACTIVATION_FORMAT.md
│   └── RESEARCH_PLAN.md
├── src/sparse_ffn/
│   ├── analysis/
│   │   ├── descriptive.py
│   │   ├── _activation_io.py
│   │   ├── rq1_layer_comparison.py
│   │   ├── rq2_neuron_dependence.py
│   │   ├── rq3_latent_structure.py
│   │   ├── rq4_weight_rank.py
│   │   └── rq5_domain_stability.py
│   ├── data/
│   ├── modeling/
│   ├── cli.py
│   ├── config.py
│   └── extraction.py
├── tests/
├── pyproject.toml
└── README.md
```

## 7. Tests

Run the lightweight unit tests (they do not download a model or dataset):

```bash
pytest -q
```

## References

- [WikiText-2 on the Hugging Face Hub](https://huggingface.co/datasets/Salesforce/wikitext) (dataset repository/configuration; the code uses `wikitext`, `wikitext-2-raw-v1`).
- [OPT-125M model card and configuration](https://huggingface.co/facebook/opt-125m).
- [Hugging Face Datasets loading documentation](https://huggingface.co/docs/datasets/loading).
- [Hugging Face Transformers model loading documentation](https://huggingface.co/docs/transformers/models).
- [PyTorch installation selector](https://pytorch.org/get-started/locally/).
