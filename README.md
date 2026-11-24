# Statistical Characterisation of Sparse Activations and Low-Rank Latent Structures in Dense Feed-Forward Neural Networks

A multivariate statistical investigation of post-ReLU intermediate representations in Transformer Feed-Forward Networks (`facebook/opt-125m`) across layer depth, parameter capacity, and linguistic domains.

---

## Academic Project Metadata

| Item | Details |
| :--- | :--- |
| **Course** | **CSP 1602: Multivariate Statistical Analysis** |
| **Academic Guidance** | **Prof. Ashis SenGupta**, Applied Statistics Unit, Indian Statistical Institute (ISI), Kolkata |
| **Investigators** | **Lakshmi Ganapathi Kodi** (`BT23MA005`) & **Saurabh Kumar Gupta** (`BT23MA006`) |
| **Department** | **Department of Mathematics and Computing**, National Institute of Technology Mizoram |
| **Term** | Semester 7, November 2026 |

---

## Core Deliverables

- **Final Comprehensive Report (40 pages, PDF):** [`Sparse_FFN_MVS_Project_Report_LaTeX.pdf`](Sparse_FFN_MVS_Project_Report_LaTeX.pdf)
- **Overleaf Report Package (Ready to upload):** [`Overleaf_Pure_MVS_Report_Package.zip`](Overleaf_Pure_MVS_Report_Package.zip)
- **Conference Presentation Slides (24 slides, PDF):** [`Sparse_FFN_Beamer_v2.pdf`](Sparse_FFN_Beamer_v2.pdf)
- **Overleaf Presentation Package (Ready to upload):** [`Sparse_FFN_Beamer_Overleaf.zip`](Sparse_FFN_Beamer_Overleaf.zip)
- **Machine-Readable JSON Evidence:** [`reports/`](reports/)

---

## Executive Summary & Research Questions

Modern Transformer architectures assign a majority of their parameter count and compute budget to dense Feed-Forward Network (FFN) sub-layers. In `facebook/opt-125m`, the intermediate layer expands representations from $d = 768$ to $d_{\text{ff}} = 3{,}072$. Because OPT utilizes exact-zero Rectified Linear Unit (ReLU) activations, negative pre-activations are zeroed out, creating an empirical semi-continuous mixture distribution with high zero inflation ($88.96\%\text{--}98.77\%$).

This study applies classical Multivariate Statistical Analysis (MVS) to intermediate FFN activations across all 12 layers of OPT-125M on the WikiText-2 benchmark and multi-domain corpora:

```
Token x in R^768  --->  fc1 (3072 x 768)  --->  ReLU  --->  Captured Feature Vector A in R^3072  --->  fc2 (768 x 3072)
```

### Empirical Findings Matrix

| Research Question | Mathematical / Statistical Formulation | Key Empirical Evidence | Scientific Conclusion |
| :--- | :--- | :--- | :--- |
| **RQ1: Layer-Wise Sparsity Flatness** | Profile analysis across depth ($n = 40$ blocks, $q = 12$ layers); Hotelling's $T^2$, Repeated-Measures ANOVA, Greenhouse-Geisser ($\hat{\epsilon}$), Friedman test. | $T^2 = 602{,}912.24$ ($p = 1.25 \times 10^{-57}$); Greenhouse-Geisser $F = 1{,}838.82$ ($p = 1.68 \times 10^{-46}$); Friedman $Q = 432.99$ ($p = 5.99 \times 10^{-86}$). | Profile flatness is decisively rejected. The network operates an extreme early bottleneck at Layer 03 ($98.77\%$ zeros) before plateauing at $\approx 92\%$ in middle layers. |
| **RQ2: Neuron Co-Activation & Modularity** | Pairwise $\phi$-correlation on prevalence-filtered binary indicators ($m = 128$); average-linkage Hierarchical Agglomerative Clustering (HAC). | Layer 00: $76.82\%$ positive pairs (mean $\phi = +0.0685$, max $0.64$, 94 clusters); Layer 06: mean $\phi = 0.0042$, median $\phi = -0.0047$ (122 clusters); Layer 11: mean $\phi = 0.0391$ (94 clusters). | Demonstrates a non-monotonic U-shaped dependence trajectory: strong early co-activation modules, near-orthogonal decorrelation in the interior, and partial re-coupling at readout. |
| **RQ3: Latent Intrinsic Dimensionality** | Covariance-based PCA (unstandardised); Participation Ratio ($\text{PR}$); Roy-Vetterli Entropy Effective Rank ($\text{erank}$); split-half subspace stability. | Layer 00: $\text{erank} = 1.51$ ($\text{PR} = 1.12$, PC1 explains $81.4\%$ variance); Layer 06: $\text{erank} = 45.53$ ($\text{PR} = 29.86$); Subspace angle stability $\bar{\rho}_{10} \ge 0.878$. | Activations undergo severe low-rank collapse at early layers and expand into broad 45-dimensional manifolds in middle layers, never utilizing the nominal 3,072 dimensions. |
| **RQ4: Static Weights vs. Dynamic Activations** | SVD of actual OPT-125M parameter matrices $\mathbf{W}_{\text{fc1}}$ ($3072 \times 768$) and $\mathbf{W}_{\text{fc2}}$ ($768 \times 3072$); Pearson $r$ and Spearman $\rho$. | Pearson $r = +0.7991$ ($p = 0.0018$); Spearman $\rho = +0.8392$ ($p = 0.00064$); Mean weight $\text{erank}_W = 500.53$ vs. Activation $\text{erank}_A = 26.38$. | Weight capacity and empirical activation rank are monotonically coupled across depth, but weights operate with an order of magnitude higher effective rank. |
| **RQ5: Cross-Domain Representation Invariance** | One-Way MANOVA ($G = 3$, $K = 8$ PCA subspace, $N = 3{,}000$ tokens); Box's M test; 5-Fold Linear & Quadratic Discriminant Analysis. | Wilks' $\Lambda = 0.6939$ (Rao $F = 74.94, p = 7.86 \times 10^{-223}$); Box's M $\chi^2(72) = 8{,}133.62$ ($p < 10^{-300}$); LDA accuracy $55.83\% \pm 0.96\%$; QDA $51.57\% \pm 2.70\%$. | Activations are domain-sensitive. Rejecting equal centroids and covariances confirms that FFN representations systematically adapt their geometric manifolds to input text genres. |

---

## Repository Structure

```text
Spare-FNN-MVS-Project/
├── assets/                                 # Presentation and report figures (PNG)
│   ├── nit_logo.png                        # National Institute of Technology Mizoram emblem
│   ├── rq1_layer_sparsity.png              # RQ1 sparsity profile across depth
│   ├── rq2_coactivation.png                # RQ2 pairwise correlation heatmap
│   ├── rq2_dendrogram.png                  # RQ2 hierarchical clustering tree
│   ├── rq3_pca.png                         # RQ3 PCA scree plot
│   ├── rq4_weight_activation.png           # RQ4 weight vs activation singular spectra
│   └── rq5_lda.png                         # RQ5 canonical discriminant projection
├── configs/
│   └── default.yaml                        # Configuration for model, data, and extraction
├── docs/
│   ├── DATA_AND_ACTIVATION_FORMAT.md       # Specification of tokenization and NPZ tensors
│   └── RESEARCH_PLAN.md                    # Detailed research methodology and hypotheses
├── report_latex/                           # Complete LaTeX source of the 40-page report
│   ├── figures/                            # High-resolution vector/raster figures
│   ├── main.tex                            # Primary document (clean, 0 overfull hboxes)
│   ├── references.bib                      # Formal bibliographic database
│   └── main.pdf                            # Compiled PDF output
├── reports/                                # Numerical evidence in structured JSON format
│   ├── descriptive_report.json             # Phase 1 summary statistics
│   ├── profile_analysis_report.json        # RQ1 Hotelling's T^2 and ANOVA outputs
│   ├── rq2_neuron_dependence_report.json   # RQ2 phi correlation and HAC metrics
│   ├── rq3_latent_structure_report.json    # RQ3 PCA, PR, and effective rank
│   ├── rq4_weight_rank_report.json         # RQ4 weight SVD and correlation metrics
│   ├── rq5_domain_stability_report.json    # RQ5 MANOVA, Box's M, and LDA results
│   └── README.md                           # Documentation for JSON report schemas
├── scripts/
│   ├── check_environment.py                # Environment and dependency verification
│   └── run_real_rq4_rq5.py                 # Standalone script for real RQ4 and RQ5 execution
├── src/sparse_ffn/                         # Core Python package
│   ├── analysis/                           # Statistical modules (RQ1 to RQ5)
│   │   ├── _activation_io.py               # Memory-mapped NPZ activation reader
│   │   ├── descriptive.py                  # Phase 1 descriptive metrics
│   │   ├── rq1_layer_comparison.py         # Profile analysis & Greenhouse-Geisser
│   │   ├── rq2_neuron_dependence.py        # Phi-correlation & dendrograms
│   │   ├── rq3_latent_structure.py         # Covariance PCA & participation ratio
│   │   ├── rq4_weight_rank.py              # Weight SVD & correlation analysis
│   │   └── rq5_domain_stability.py         # MANOVA, Box's M, & LDA
│   ├── data/                               # Dataset preparation & block batching
│   │   ├── collator.py                     # Non-padding attention mask collator
│   │   ├── token_blocks.py                 # Fixed-size token block chunking
│   │   └── wikitext.py                     # Hugging Face WikiText-2 loader
│   ├── modeling/                           # Model integration
│   │   └── opt_adapter.py                  # PyTorch forward hook for post-ReLU capture
│   ├── cli.py                              # Unified command-line interface
│   ├── config.py                           # Pydantic / dataclass configuration
│   └── extraction.py                       # Batch extraction pipeline
├── tests/                                  # Comprehensive automated test suite (19 tests)
│   ├── test_collator.py
│   ├── test_descriptive.py
│   ├── test_rq1_layer_comparison.py
│   ├── test_rq2_neuron_dependence.py
│   ├── test_rq3_latent_structure.py
│   ├── test_rq4_weight_rank.py
│   ├── test_rq5_domain_stability.py
│   └── test_token_blocks.py
├── Overleaf_Pure_MVS_Report_Package.zip    # Standalone Overleaf ZIP for report
├── Sparse_FFN_Beamer_Overleaf.zip          # Standalone Overleaf ZIP for slides
├── Sparse_FFN_Beamer_v2.pdf                # Compiled Beamer presentation PDF
├── Sparse_FFN_Beamer_v2.tex                # Presentation LaTeX source
├── Sparse_FFN_MVS_Project_Report_LaTeX.pdf # Compiled report PDF
├── pyproject.toml                          # Package definition and tool configs
├── requirements.txt                        # Python dependencies
└── README.md                               # This document
```

---

## Installation & Setup

### 1. Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/SaurabhKrGupta7788/Spare-FNN-MVS-Project.git
cd Spare-FNN-MVS-Project

# Create and activate environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
# Install PyTorch appropriate for your machine:
# Refer to https://pytorch.org/get-started/locally/

# Install sparse_ffn package in editable mode with development tools:
pip install -e ".[dev]"
```

Verify your installation:

```bash
python scripts/check_environment.py
```

---

## Executing the Analysis Pipeline

### Step 1: Prepare Token Datasets

Downloads and processes the official WikiText-2 splits:

```bash
python -m sparse_ffn prepare-data
```

### Step 2: Extract Post-ReLU Activations

Captures intermediate neuron activations from frozen `facebook/opt-125m`:

```bash
# Pilot run (20 batches of 128 tokens)
python -m sparse_ffn extract-activations --split train --max-batches 20

# Full split extraction
python -m sparse_ffn extract-activations --split train --max-batches 0
```

### Step 3: Run Research Question Analyses

Each module can be executed independently:

```bash
# Phase 1: Summary statistics & zero fractions
python -m sparse_ffn analyze-descriptive --split train

# RQ1: Sparsity profile flatness tests
python -m sparse_ffn analyze-rq1 --split train

# RQ2: Neuron co-activation & hierarchical clustering
python -m sparse_ffn analyze-rq2 --split train

# RQ3: Covariance PCA, effective rank, and stability
python -m sparse_ffn analyze-rq3 --split train

# RQ4 & RQ5: Real OPT-125M weight SVD & multi-domain corpora
python scripts/run_real_rq4_rq5.py
```

To execute all pipeline stages sequentially:

```bash
python -m sparse_ffn analyze-all --split train
```

---

## Testing & Verification

The repository includes a comprehensive 19-test automated test suite verifying collator masks, mathematical estimators, and test statistics:

```bash
pytest -v
```

All 19 tests run in under 75 seconds without requiring live model or dataset downloads.

---

## Compiling LaTeX Documents

The report and presentation are pre-compiled in the repository root. To rebuild them locally:

```bash
# Compile Project Report (40 pages)
cd report_latex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Compile Presentation Slides (24 slides)
cd ..
pdflatex -interaction=nonstopmode Sparse_FFN_Beamer_v2.tex
pdflatex -interaction=nonstopmode Sparse_FFN_Beamer_v2.tex
```

---

## License

This project is licensed under the MIT License — see the [`LICENSE`](LICENSE) file for details.
