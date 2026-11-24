# Multivariate Statistical Analysis (MVS) Reports

This directory contains the structured JSON reports generated from the empirical evaluation of `facebook/opt-125m` ($12$ layers, $d_{\text{ff}} = 3{,}072$, ReLU activation) on the WikiText-2 benchmark and multi-domain corpora.

## Report Index & Ground-Truth Statistics

| Report File | Research Question | Primary Statistical Test | Key Empirical Values |
| :--- | :--- | :--- | :--- |
| [`descriptive_report.json`](descriptive_report.json) | Baseline Sparsity & Distributions | Descriptive Statistics, Histograms | Layer 00 mean sparsity: $88.96\%$; Layer 03: $98.77\%$; Layer 06: $91.44\%$ |
| [`profile_analysis_report.json`](profile_analysis_report.json) | **RQ1**: Layer-Wise Sparsity Flatness | Repeated-Measures ANOVA, Greenhouse-Geisser, Hotelling's $T^2$ | $T^2 = 602{,}912.24$ ($p = 1.25 \times 10^{-57}$), Greenhouse-Geisser $\hat{\epsilon} = 0.1253$ ($p = 1.68 \times 10^{-46}$), Friedman $Q = 432.99$ ($p = 5.99 \times 10^{-86}$) |
| [`rq2_neuron_dependence_report.json`](rq2_neuron_dependence_report.json) | **RQ2**: Neuron Co-Activation & Modularity | Pairwise $\phi$-Correlation, Average-Linkage HAC | Layer 00: $76.82\%$ positive pairs (mean $\phi = +0.0685$, 94 clusters); Layer 06: median $\phi = -0.0047$, mean $\phi = 0.0042$ (122 clusters) |
| [`rq3_latent_structure_report.json`](rq3_latent_structure_report.json) | **RQ3**: Latent Dimensionality & PR | Continuous Participation Ratio, Roy-Vetterli Entropy $\text{erank}$ | Layer 00 $\text{erank} = 1.51$ ($\text{PR} = 1.12$); Layer 06 $\text{erank} = 45.53$ ($\text{PR} = 29.86$); Subspace stability $\bar{\rho}_{10} \ge 0.878$ |
| [`rq4_weight_rank_report.json`](rq4_weight_rank_report.json) | **RQ4**: Static Weights vs. Dynamic Activations | SVD Singular Spectra, Correlation Analysis | Weight mean $\text{erank}_W = 500.53$ vs. Activation mean $\text{erank}_A = 26.38$; Pearson $r = +0.7991$ ($p = 0.0018$), Spearman $\rho = +0.8392$ ($p = 0.00064$) |
| [`rq5_domain_stability_report.json`](rq5_domain_stability_report.json) | **RQ5**: Cross-Domain Manifold Invariance | One-Way MANOVA, Box's M, 5-Fold LDA / QDA | Wilks' $\Lambda = 0.6939$ ($F = 74.94, p = 7.86 \times 10^{-223}$), Box's M $\chi^2(72) = 8{,}133.62$ ($p < 10^{-300}$), 5-Fold LDA accuracy = $55.83\% \pm 0.96\%$, QDA = $51.57\% \pm 2.70\%$ |

All statistical evaluations were computed with 64-bit precision using pure NumPy, SciPy, and Scikit-Learn routines, verified against the automated test suite.
