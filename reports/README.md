# Multivariate Statistical Analysis (MVS) Reports

This directory contains the official structured JSON analysis reports generated from the empirical evaluation of `facebook/opt-125m` (12 layers, $d_{ff} = 3072$, ReLU activation) on the WikiText-2 benchmark and multi-domain corpora.

## Report Index & Key Metrics

| Report File | Research Question | Primary Statistical Test | Key Empirical Values |
| :--- | :--- | :--- | :--- |
| [`descriptive_report.json`](descriptive_report.json) | Baseline Sparsity & Distributions | Descriptive Statistics, Histograms | Layer 0 mean sparsity: 88.96%; Layer 3: 98.77%; Layer 6: 91.56% |
| [`profile_analysis_report.json`](profile_analysis_report.json) | **RQ1**: Layer-wise Sparsity Flatness | Repeated-Measures ANOVA, Greenhouse-Geisser, Hotelling's $T^2$ | $T^2 = 602,912.2$ ($p = 1.25 \times 10^{-57}$), Greenhouse-Geisser $\hat{\epsilon} = 0.125$ ($p_{\text{GG}} = 1.68 \times 10^{-46}$), Friedman $Q = 432.99$ ($p = 5.99 \times 10^{-86}$) |
| [`rq2_neuron_dependence_report.json`](rq2_neuron_dependence_report.json) | **RQ2**: Neuron Co-activation & Modularity | Pairwise $\phi$-Correlation, Ward's HAC | L0 positive pairs: 76.82% (mean $\phi = +0.068$, max $0.64$); L6 median $\phi = -0.009$ (decorrelated) |
| [`rq3_latent_structure_report.json`](rq3_latent_structure_report.json) | **RQ3**: Latent Dimensionality & PR | Continuous Participation Ratio, Roy-Vetterli Spectral Rank | L0 $\text{PR} = 1.51$ (PC1 = 81.4% variance); L6 $\text{PR} = 46.2$; Subspace stability $>90\%$ |
| [`rq4_weight_rank_report.json`](rq4_weight_rank_report.json) | **RQ4**: Static Weights vs Dynamic Activations | SVD Singular Spectra, Rank Mismatch Ratio | Static weight $\text{erank}_W = 593.51$ vs Dynamic activation $\text{erank}_A = 26.10$; Pearson $r = -0.607$ ($p = 0.036$), Spearman $\rho = -0.622$ ($p = 0.031$) |
| [`rq5_domain_stability_report.json`](rq5_domain_stability_report.json) | **RQ5**: Cross-Domain Manifold Invariance | One-Way MANOVA, Box's M, Fisher's LDA | Wilks' $\Lambda = 2.33 \times 10^{-6}$ ($p = 0.0$), Box's M $M = 15,693.7$ ($p = 0.0$), 5-Fold LDA classification accuracy = 100.0% |

All statistical calculations were executed with 64-bit floating point precision using pure NumPy and SciPy routines, and validated against the 18-test automated pytest suite.
