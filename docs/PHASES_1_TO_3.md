# Running Phases 1–3 / RQ1–RQ3

This update leaves the Phase 1 extraction format unchanged and implements the first three statistical research questions.

## Installation update

Activate the existing project environment, then install the new dependency:

```bash
pip install -e "[dev]"
```

The added dependency is `scikit-learn`, used for PCA and Gaussian factor analysis. PyTorch, Transformers and Datasets remain as before.

## Commands

```bash
# Existing Phase 1 descriptive analysis
python -m sparse_ffn analyze-descriptive --split train

# RQ1: block-by-layer profile analysis / flatness test
python -m sparse_ffn analyze-rq1 --split train

# RQ2: marginal, joint, conditional probabilities, phi correlations, clustering
python -m sparse_ffn analyze-rq2 --split train

# RQ3: covariance PCA and exploratory factor analysis
python -m sparse_ffn analyze-rq3 --split train

# Run RQ1–RQ3 sequentially
python -m sparse_ffn analyze-rq1-rq3 --split train
```

Add `--run-dir "artifacts/activations/train/<run-folder>"` to select a particular completed run. Without it, the most recently completed run is selected.

## Default settings

- RQ1: use the Phase 1 `block_sparsity_by_layer.csv`; text blocks are rows and FFN layers are columns. Test profile flatness with within-block contrasts and a one-sample Hotelling T-squared statistic.
- RQ2: maximum 128 neurons per layer; sample at most 50,000 token rows; use neurons with activation frequency between 0.01 and 0.99; cluster by average linkage on distance `1 - phi`.
- RQ3: sample at most 5,000 token rows; compute 64 PCA components for all neurons; fit Gaussian factor analysis on at most 256 prevalence-filtered neurons for 2, 5 and 10 factors with varimax rotation.

These values are configurable in `configs/default.yaml`.

## Output locations

```text
artifacts/analysis/<split>/<run-folder>/
├── descriptive/                 # existing Phase 1 outputs
├── rq1_profile_analysis/
│   ├── profile_layer_summary.csv
│   ├── block_by_layer_sparsity_matrix.csv
│   ├── profile_analysis_report.json
│   └── mean_sparsity_profile.png
├── rq2_neuron_dependence/
│   ├── rq2_neuron_dependence_report.json
│   └── layer_XX/
│       ├── marginal_activation_probabilities.csv
│       ├── joint_conditional_pairwise_probabilities.csv
│       ├── activation_covariance_matrix.csv
│       ├── phi_correlation_matrix.csv
│       ├── sampled_token_row_indices.csv
│       ├── cluster_assignments.csv
│       ├── phi_correlation_heatmap.png
│       └── neuron_dendrogram.png
└── rq3_latent_structure/
    ├── pca_eigenvalues.csv
    ├── pca_layer_summary.csv
    ├── factor_analysis_model_summary.csv
    ├── factor_analysis_assumption_notes.csv
    └── layer_XX/
        ├── pca_components.npz
        ├── pca_scree_and_cumulative_variance.png
        ├── factor_analysis_neuron_selection.csv
        ├── factor_loadings_kXX.csv
        ├── factor_loadings_heatmap_kXX.png
        └── factor_communalities_kXX.csv
```

## Interpretation notes

1. RQ1 tests whether the average sparsity profile is flat across layers. A significant result means at least one layer mean differs; it does not identify all differing pairs.
2. RQ2 reports pairwise empirical dependence and exploratory clusters for a selected neuron subset. It is not an all-neuron mutual-independence test.
3. RQ3 uses covariance-based PCA on centred, unstandardised activations. Gaussian factor analysis is only exploratory because ReLU outputs are zero-inflated and may be non-normal.
4. The uploaded Phase 1 summary ZIP contains descriptive CSVs and plots, not the raw activation NPZ shards. RQ1 can run from the summary table. RQ2 and RQ3 require the original `artifacts/activations/<split>/<run-folder>/layer_XX/batch_*.npz` files.

## Tests

Run:

```bash
pytest -q
```

The unit tests use small synthetic arrays and do not download from Hugging Face.
