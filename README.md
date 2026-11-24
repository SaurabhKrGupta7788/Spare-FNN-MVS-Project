# Statistical Characterisation of Sparse Activations and Low-Rank Latent Structures in Dense Feed-Forward Neural Networks

An empirical multivariate statistical investigation of post-ReLU intermediate representations in Transformer Feed-Forward Networks (`facebook/opt-125m`) across layer depth, parameter capacity, and linguistic domains.

---

## Academic Context & Attribution

| Attribute | Details |
| :--- | :--- |
| **Course** | **CSP 1602: Multivariate Statistical Analysis** (Semester 7, November 2026) |
| **Academic Supervisor** | **Prof. Ashis SenGupta**, Applied Statistics Unit, Indian Statistical Institute (ISI), Kolkata |
| **Investigators** | **Lakshmi Ganapathi Kodi** (`BT23MA005`) & **Saurabh Kumar Gupta** (`BT23MA006`) |
| **Department** | **Department of Mathematics and Computing**, National Institute of Technology Mizoram |

---

## Project Overview

Modern decoder-only Transformers allocate the majority of their parameter capacity and floating-point operations to Feed-Forward Network (FFN) sub-layers. In `facebook/opt-125m`, each of the 12 decoder blocks contains a two-stage FFN:

$$\mathbf{h}_{\text{inter}} = \text{ReLU}(\mathbf{x} \mathbf{W}_{\text{fc1}}^\top + \mathbf{b}_{\text{fc1}}), \quad \mathbf{y} = \mathbf{h}_{\text{inter}} \mathbf{W}_{\text{fc2}}^\top + \mathbf{b}_{\text{fc2}}$$

where $\mathbf{x} \in \mathbb{R}^{768}$, $\mathbf{W}_{\text{fc1}} \in \mathbb{R}^{3{,}072 \times 768}$, and $\mathbf{W}_{\text{fc2}} \in \mathbb{R}^{768 \times 3{,}072}$. Because OPT uses the exact-zero Rectified Linear Unit (ReLU), negative pre-activations map strictly to zero, yielding an empirical zero-inflated semi-continuous distribution ($88.96\%\text{--}98.77\%$ zeros).

This repository provides the Python package, data pipeline, and multivariate statistical test battery to empirically characterise these representations:

```text
Token x in R^768  --->  fc1 (3072 x 768)  --->  ReLU  --->  Captured Feature A in R^3072  --->  fc2 (768 x 3072)
                                                                    |
                                                     [Saved to NPZ Tensors]
```

---

## Research Questions & Empirical Results

The investigation evaluates five core research questions across the 12 layers of OPT-125M evaluated on WikiText-2 and multi-domain text corpora:

| Research Question | Mathematical / Statistical Formulation | Empirical Test Statistics | Key Findings |
| :--- | :--- | :--- | :--- |
| **RQ1: Layer-Wise Sparsity Non-Flatness** | Profile analysis across depth ($n = 40$ blocks, $q = 12$ layers); contrast vectors $\mathbf{C} \in \mathbb{R}^{11 \times 12}$; Hotelling's $T^2$, Repeated-Measures ANOVA, Greenhouse-Geisser ($\hat{\epsilon}$), Friedman rank test. | $T^2 = 602{,}912.24$ ($p = 1.25 \times 10^{-57}$); RM-ANOVA $F(11, 429) = 1{,}838.82$ ($p < 10^{-300}$); Greenhouse-Geisser $\hat{\epsilon} = 0.1253$ ($p = 1.68 \times 10^{-46}$); Friedman $Q = 432.99$ ($p = 5.99 \times 10^{-86}$). | Profile flatness is decisively rejected. The network implements an extreme early bottleneck at Layer 03 ($98.77\%$ zeros) before plateauing at $\approx 91.5\%$ in middle layers. |
| **RQ2: Neuron Co-Activation & Modularity** | Pairwise $\phi$-correlation on prevalence-filtered binary indicators ($m = 128$); average-linkage Hierarchical Agglomerative Clustering (HAC) at distance threshold $d = 0.70$. | Layer 00: $76.82\%$ positive pairs (mean $\phi = +0.0685$, 94 clusters); Layer 06: mean $\phi = 0.0042$, median $\phi = -0.0047$ (122 clusters); Layer 11: mean $\phi = 0.0391$ (94 clusters). | Demonstrates a non-monotonic U-shaped dependence: early functional co-activation modules, near-orthogonal decorrelation in the middle layers, and partial re-coupling at readout. |
| **RQ3: Latent Intrinsic Dimensionality** | Covariance-based PCA (unstandardised); continuous Participation Ratio ($\text{PR}$); Roy-Vetterli Entropy Effective Rank ($\text{erank}$); split-half principal subspace stability ($\bar{\rho}_{10}$). | Layer 00: $\text{erank} = 1.51$ ($\text{PR} = 1.12$, PC1 explains $81.4\%$ variance); Layer 06: $\text{erank} = 45.53$ ($\text{PR} = 29.86$); Subspace angle stability $\bar{\rho}_{10} \ge 0.878$. | Activations undergo severe low-rank collapse at early layers and expand into broad $\approx 45$-dimensional manifolds in middle layers, never utilizing the nominal 3,072 dimensions. |
| **RQ4: Static Weights vs. Dynamic Activations** | SVD of actual OPT-125M weight matrices $\mathbf{W}_{\text{fc1}}$ ($3{,}072 \times 768$) and $\mathbf{W}_{\text{fc2}}$ ($768 \times 3{,}072$); Pearson $r$ and Spearman $\rho$ against activation rank. | Pearson $r = +0.7991$ ($p = 0.0018$); Spearman $\rho = +0.8392$ ($p = 0.00064$); Mean weight $\text{erank}_W = 500.53$ vs. Activation $\text{erank}_A = 26.38$. | Weight capacity and empirical activation rank are monotonically correlated across depth, but weight matrices operate with an order of magnitude higher effective rank. |
| **RQ5: Cross-Domain Representation Invariance** | One-Way MANOVA across 3 domains ($N = 3{,}000$, $K = 8$ PCs); Box's M test; 5-fold cross-validated Linear and Quadratic Discriminant Analysis (LDA / QDA). | Wilks' $\Lambda = 0.6939$ (Rao $F(16, 5980) = 74.94, p = 7.86 \times 10^{-223}$); Box's M $\chi^2(72) = 8{,}133.62$ ($p < 10^{-300}$); 5-fold LDA: $55.83\% \pm 0.96\%$; QDA: $51.57\% \pm 2.70\%$ (chance $33.33\%$). | Rejects manifold invariance across domains ($p < 10^{-200}$). Feed-forward activations retain distinct domain-conditioned geometric signatures across genres. |

---

## Repository Structure

```text
Spare-FNN-MVS-Project/
|-- assets/                               # Empirical plots and visual results (PNG)
|   |-- fig1_mean_sparsity_profile.png    # RQ1: Mean sparsity across network depth
|   |-- fig2a_phi_corr_l00.png            # RQ2: Phi-correlation heatmap (Layer 00)
|   |-- fig2b_phi_corr_l06.png            # RQ2: Phi-correlation heatmap (Layer 06)
|   |-- fig2c_dendrogram_l00.png          # RQ2: Dendrogram clustering (Layer 00)
|   |-- fig3a_pca_scree_l00.png           # RQ3: PCA scree and cumulative variance (Layer 00)
|   |-- fig3b_pca_scree_l06.png           # RQ3: PCA scree and cumulative variance (Layer 06)
|   |-- fig3c_factor_loadings_l00.png     # RQ3: Principal component factor loadings
|   |-- fig4_activation_vs_weight_rank.png# RQ4: Activation vs static weight effective rank
|   |-- fig5_lda_domain_separation.png    # RQ5: LDA canonical projection across 3 domains
|   `-- nit_logo.png                      # NIT Mizoram emblem
|-- configs/
|   `-- default.yaml                      # Model, dataset, and extraction configuration
|-- data/
|   `-- processed/wikitext2/              # Hugging Face Dataset format (train/val/test)
|-- docs/
|   |-- DATA_AND_ACTIVATION_FORMAT.md     # Tensor schemas and block structure
|   `-- RESEARCH_PLAN.md                  # Complete mathematical research plan & hypotheses
|-- reports/                              # Structured JSON outputs with full test statistics
|   |-- descriptive_report.json           # Layer-by-layer zero fractions and tail moments
|   |-- profile_analysis_report.json      # RQ1: Hotelling's T^2, ANOVA, Greenhouse-Geisser
|   |-- rq2_neuron_dependence_report.json # RQ2: Phi-correlation summaries & HAC cluster counts
|   |-- rq3_latent_structure_report.json  # RQ3: PR, Roy-Vetterli erank, subspace stability
|   |-- rq4_weight_rank_report.json       # RQ4: Weight matrix SVD spectra & correlation stats
|   |-- rq5_domain_stability_report.json  # RQ5: MANOVA statistics, Box's M, LDA/QDA scores
|   `-- README.md                         # Detailed JSON schema documentation
|-- scripts/
|   |-- check_environment.py              # Environment and dependency diagnostic
|   `-- run_real_rq4_rq5.py               # Standalone runner for real weight SVD and RQ5
|-- src/sparse_ffn/                       # Core Python package
|   |-- analysis/                         # Multivariate statistical analysis modules
|   |   |-- _activation_io.py             # Memory-mapped NPZ activation reader
|   |   |-- descriptive.py                # Phase 1 summary statistics and distributions
|   |   |-- rq1_layer_comparison.py       # Profile analysis & Greenhouse-Geisser tests
|   |   |-- rq2_neuron_dependence.py      # Phi correlation & agglomerative clustering
|   |   |-- rq3_latent_structure.py       # Covariance PCA & participation ratio
|   |   |-- rq4_weight_rank.py            # Static weight SVD & activation rank correlation
|   |   `-- rq5_domain_stability.py       # MANOVA, Box's M, & cross-domain LDA
|   |-- data/                             # Dataset handling
|   |   |-- collator.py                   # Attention mask padding and batch collator
|   |   |-- token_blocks.py               # Fixed-size block chunker
|   |   `-- wikitext.py                   # WikiText-2 tokenization and loader
|   |-- modeling/                         # PyTorch model integration
|   |   `-- opt_adapter.py                # Forward hook extracting post-ReLU activations
|   |-- cli.py                            # Click-based command-line interface
|   |-- config.py                         # Dataclass configuration loader
|   |-- extraction.py                     # Streaming activation extraction pipeline
|   `-- logging_utils.py                  # Structured logging
|-- tests/                                # Automated unit test suite (19 tests)
|   |-- test_collator.py                  # Tests for batch collation and masking
|   |-- test_descriptive.py               # Tests for sparsity and moment estimators
|   |-- test_rq1_layer_comparison.py      # Tests for Hotelling's T^2 and contrast matrices
|   |-- test_rq2_neuron_dependence.py     # Tests for phi-correlation and contingency tables
|   |-- test_rq3_latent_structure.py      # Tests for PR and Roy-Vetterli erank
|   |-- test_rq4_weight_rank.py           # Tests for weight SVD and correlation metrics
|   |-- test_rq5_domain_stability.py      # Tests for MANOVA, Box's M, and discriminant analysis
|   `-- test_token_blocks.py              # Tests for block chunking invariants
|-- pyproject.toml                        # Build configuration and pytest settings
|-- requirements.txt                      # Locked dependency list
`-- LICENSE                               # MIT License
```

---

## Visual Results Gallery

All empirical plots generated during the research are stored in the [`assets/`](assets/) directory:

| Research Question | Visual Result | Description |
| :--- | :---: | :--- |
| **RQ1: Sparsity Depth Profile** | [`fig1_mean_sparsity_profile.png`](assets/fig1_mean_sparsity_profile.png) | Layer-wise mean zero fraction across 40 blocks of 128 tokens showing the Layer 03 bottleneck ($98.77\%$). |
| **RQ2: Neuron Co-Activation** | [`fig2a_phi_corr_l00.png`](assets/fig2a_phi_corr_l00.png)<br>[`fig2b_phi_corr_l06.png`](assets/fig2b_phi_corr_l06.png) | Pairwise $\phi$-correlation heatmaps comparing early modular coupling (Layer 00) with orthogonal decorrelation (Layer 06). |
| **RQ2: Hierarchical Clustering** | [`fig2c_dendrogram_l00.png`](assets/fig2c_dendrogram_l00.png) | Average-linkage agglomerative clustering dendrogram revealing functional neuron clusters. |
| **RQ3: Latent Dimensionality** | [`fig3a_pca_scree_l00.png`](assets/fig3a_pca_scree_l00.png)<br>[`fig3b_pca_scree_l06.png`](assets/fig3b_pca_scree_l06.png) | Scree plots and cumulative explained variance comparing rank-collapsed Layer 00 with high-capacity Layer 06. |
| **RQ3: Factor Loadings** | [`fig3c_factor_loadings_l00.png`](assets/fig3c_factor_loadings_l00.png) | Principal component factor loadings across intermediate neurons. |
| **RQ4: Weight vs Activation Rank** | [`fig4_activation_vs_weight_rank.png`](assets/fig4_activation_vs_weight_rank.png) | Singular value spectra comparing static parameter capacity ($\text{erank}_W \approx 500$) with empirical activation rank ($\text{erank}_A \approx 26$). |
| **RQ5: Domain Discrimination** | [`fig5_lda_domain_separation.png`](assets/fig5_lda_domain_separation.png) | Canonical Linear Discriminant Analysis projection separating Wikipedia, Scientific text, and News corpora. |

---

## Installation & Setup

### 1. Prerequisites

- Python 3.10 or higher
- Git

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/SaurabhKrGupta7788/Spare-FNN-MVS-Project.git
cd Spare-FNN-MVS-Project

# Create virtual environment
python -m venv .venv

# Activate environment:
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install package in editable mode with development dependencies:
pip install -e ".[dev]"
```

Verify the environment:

```bash
python scripts/check_environment.py
```

---

## Executing the Analysis Pipeline

### Step 1: Token Dataset Preparation

Tokenizes the official WikiText-2 splits using the `facebook/opt-125m` tokenizer into contiguous 128-token blocks:

```bash
python -m sparse_ffn prepare-data
```

### Step 2: Intermediate Activation Extraction

Runs inference through frozen `facebook/opt-125m` and hooks post-ReLU activations across all 12 layers:

```bash
# Pilot extraction (20 blocks of 128 tokens)
python -m sparse_ffn extract-activations --split train --max-batches 20

# Full split extraction
python -m sparse_ffn extract-activations --split train --max-batches 0
```

### Step 3: Run Multivariate Statistical Tests

Each analysis phase can be executed independently via the CLI:

```bash
# Phase 1: Summary statistics, zero fractions, and moments
python -m sparse_ffn analyze-descriptive --split train

# RQ1: Sparsity profile flatness (Hotelling's T^2, ANOVA, Greenhouse-Geisser)
python -m sparse_ffn analyze-rq1 --split train

# RQ2: Neuron co-activation phi-correlations & hierarchical clustering
python -m sparse_ffn analyze-rq2 --split train

# RQ3: Covariance PCA, Participation Ratio, and effective rank
python -m sparse_ffn analyze-rq3 --split train

# RQ4 & RQ5: Real OPT-125M weight SVD & multi-domain corpora evaluation
python scripts/run_real_rq4_rq5.py
```

To run all phases sequentially:

```bash
python -m sparse_ffn analyze-all --split train
```

---

## Automated Test Suite

The test suite covers data collators, token chunkers, moment estimators, and all multivariate hypothesis testing procedures:

```bash
pytest -v
```

All 19 tests pass deterministically without requiring GPU access or live network connections:

```text
tests/test_collator.py::test_collator_pads_and_masks PASSED
tests/test_collator.py::test_collator_uniform_lengths PASSED
tests/test_descriptive.py::test_zero_fraction_known_array PASSED
tests/test_descriptive.py::test_moments_gaussian PASSED
tests/test_descriptive.py::test_descriptive_pipeline PASSED
tests/test_rq1_layer_comparison.py::test_contrast_matrix_shape PASSED
tests/test_rq1_layer_comparison.py::test_hotelling_rejects_nonflat PASSED
tests/test_rq1_layer_comparison.py::test_greenhouse_geisser_epsilon PASSED
tests/test_rq2_neuron_dependence.py::test_contingency_table_exact PASSED
tests/test_rq2_neuron_dependence.py::test_phi_correlation_bounds PASSED
tests/test_rq2_neuron_dependence.py::test_clustering_pipeline PASSED
tests/test_rq3_latent_structure.py::test_participation_ratio_known PASSED
tests/test_rq3_latent_structure.py::test_roy_vetterli_entropy_rank PASSED
tests/test_rq3_latent_structure.py::test_pca_explained_variance PASSED
tests/test_rq4_weight_rank.py::test_svd_spectrum_properties PASSED
tests/test_rq4_weight_rank.py::test_rank_correlation_computation PASSED
tests/test_rq5_domain_stability.py::test_manova_known_separation PASSED
tests/test_rq5_domain_stability.py::test_box_m_test PASSED
tests/test_token_blocks.py::test_token_blocks_length PASSED
```

---

## Machine-Readable Evidence

All empirical outputs are serialized in [`reports/`](reports/):

- [`reports/descriptive_report.json`](reports/descriptive_report.json)
- [`reports/profile_analysis_report.json`](reports/profile_analysis_report.json)
- [`reports/rq2_neuron_dependence_report.json`](reports/rq2_neuron_dependence_report.json)
- [`reports/rq3_latent_structure_report.json`](reports/rq3_latent_structure_report.json)
- [`reports/rq4_weight_rank_report.json`](reports/rq4_weight_rank_report.json)
- [`reports/rq5_domain_stability_report.json`](reports/rq5_domain_stability_report.json)

Refer to [`reports/README.md`](reports/README.md) for the exact JSON schemas and parameter definitions.

---

## License

This software and its experimental outputs are released under the [MIT License](LICENSE).
