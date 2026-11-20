# Research plan and analysis boundaries

Aligned with **Project Proposal: Statistical Characterisation of Sparse Activations and Low-Rank Latent Structures in Dense Feed-Forward Neural Networks**
- Team: Lakshmi Ganapathi Kodi (BT23MA005), Saurabh Kumar Guptha (BT23MA006)
- Course: Multivariate Statistical Analysis (Semester 7)

## Main research objective

Study whether dense FFN intermediate activations show (i) activation sparsity, (ii) pairwise statistical dependence between neurons, and (iii) a lower-dimensional covariance/factor structure across network depth, weight capacity, and text domains.

## Phase 1 — descriptive analysis (implemented)

For every FFN layer, compute zero/positive fractions, token-wise sparsity distributions, neuron activation frequencies, activation magnitude summaries and sampled positive-value distribution summaries.

## RQ1 — profile analysis of layer-wise sparsity (implemented)

- **Sampling unit:** a text block. Every block has one average sparsity value for each FFN layer.
- **Data matrix:** blocks by layers.
- **Multivariate methods:**
  - One-sample Hotelling's $T^2$ test on contrast vectors against the reference layer (Layer 0).
  - Repeated-measures ANOVA with Mauchly's sphericity check and Greenhouse-Geisser adjustment ($\hat{\epsilon}$).
  - Non-parametric Friedman test for repeated measures across layers.
  - Post-hoc pairwise comparisons with Holm-Bonferroni correction.
- **Outputs:** layer mean profile and confidence intervals, ANOVA table, Friedman test statistic, post-hoc p-values, and mean sparsity profile plot.

## RQ2 — joint, marginal and conditional neuron distributions (implemented)

- Convert each selected neuron's post-ReLU activation into a binary active/inactive indicator.
- Apply a configured prevalence filter ($0.01 \le p \le 0.99$) and reproducibly sample a manageable neuron subset (128 neurons).
- Estimate marginal activation probabilities, pairwise joint probabilities, conditional activation probabilities, the binary-indicator covariance matrix and phi correlation matrix.
- Apply hierarchical agglomerative cluster analysis to the selected-neuron phi-correlation matrix using average linkage on correlation distance ($1 - \phi$).
- **Outputs:** probability tables, covariance/correlation matrices, heatmaps, cluster assignments, and dendrograms.

## RQ3 — PCA, effective dimensionality, and exploratory factor analysis (implemented)

- **PCA:** centre but do not variance-standardise the activation variables. Use covariance-based PCA to report eigenvalues, explained-variance ratios, cumulative variance and scree plots.
- **Effective dimensionality:** calculate participation ratio $\text{PR} = (\sum \lambda_i)^2 / \sum \lambda_i^2$ and entropy-based effective rank $\text{erank} = \exp(-\sum p_i \ln p_i)$ for every layer.
- **Validation:** estimate reconstruction error curves as a function of $K$ and evaluate sampling stability via split-half cross-sample subspace cosine similarity.
- **Factor analysis:** fit exploratory Gaussian factor models with selected factor counts ($K \in \{2, 5, 10\}$) and varimax rotation on a prevalence-filtered subset of neurons. Report loadings, communalities, uniquenesses and covariance-reproduction error.

## RQ4 — activation versus weight effective dimension (implemented)

- **Weight SVD:** extract incoming ($\text{fc1}: 3072 \times 768$) and outgoing ($\text{fc2}: 768 \times 3072$) weight matrices for all 12 FFN layers and compute Singular Value Decomposition.
- **Weight effective rank:** compute participation ratio and entropy-based effective rank from singular-value spectra.
- **Cross-modal comparison:** compute Pearson correlation ($r$) and Spearman rank correlation ($\rho$) between activation effective dimensionality and weight effective dimensionality across depth.
- **Outputs:** `weight_svd_summary.csv`, `activation_vs_weight_dimensionality.csv`, `activation_vs_weight_rank.png`, and report.

## RQ5 — cross-domain stability (implemented)

- **Domains:** assess activations across 3 text domains (Wikipedia, Scientific articles, News articles).
- **Dimension reduction:** project activations to a reduced set of principal components ($K = 8$) to guarantee non-singular sample covariance matrices ($N \gg K$).
- **Multivariate tests:**
  - Construct sample mean vectors and sample covariance matrices for each domain.
  - Two-sample Hotelling's $T^2$ tests for pairwise domain comparisons.
  - One-Way MANOVA across all 3 domains reporting Wilks' Lambda, Pillai's trace, Hotelling-Lawley trace, Roy's root, and Rao's $F$-test.
  - Test of equality of covariance matrices using Box's M test and Frobenius distance.
  - Linear Discriminant Analysis (LDA) with 5-fold cross-validated domain classification accuracy and 2D canonical discriminant projection.
- **Outputs:** `domain_mean_vectors.csv`, `domain_covariance_matrices.csv`, `pairwise_hotelling_t2.csv`, `manova_summary.csv`, `box_m_test.csv`, `lda_classification_report.csv`, `lda_domain_separation.png`, and report.
