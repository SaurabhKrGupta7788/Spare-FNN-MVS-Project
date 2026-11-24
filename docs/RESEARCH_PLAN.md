# Research Plan and Analysis Methodology

**Project:** Statistical Characterisation of Sparse Activations and Low-Rank Latent Structures in Dense Feed-Forward Neural Networks  
**Course:** CSP 1602: Multivariate Statistical Analysis (Semester 7, November 2026)  
**Authors:** Lakshmi Ganapathi Kodi (BT23MA005) & Saurabh Kumar Gupta (BT23MA006)  
**Supervisor:** Prof. Ashis SenGupta, Applied Statistics Unit, Indian Statistical Institute, Kolkata  
**Department:** Department of Mathematics and Computing, National Institute of Technology Mizoram  

---

## 1. Research Objectives

This investigation characterises the internal representation geometry of dense Transformer Feed-Forward Networks (FFNs) using classical multivariate mathematical statistics. Studying the intermediate post-ReLU activation space of `facebook/opt-125m` ($12$ layers, $d_{\text{ff}} = 3{,}072$), we evaluate five core research questions:

1. **RQ1 (Layer-Wise Sparsity Non-Flatness):** Does activation sparsity exhibit significant profile non-flatness across network depth?
2. **RQ2 (Neuron Co-Activation & Modularity):** Do individual neurons fire independently, or do they form correlated functional modules?
3. **RQ3 (Latent Intrinsic Dimensionality):** How many orthogonal principal axes capture the majority of activation variance across layers?
4. **RQ4 (Static Weights vs. Dynamic Activations):** Does static weight SVD capacity determine or decouple from empirical activation rank?
5. **RQ5 (Cross-Domain Representation Invariance):** Are latent activation manifolds invariant across text domains, or do they encode domain-specific signatures?

---

## 2. Experimental Pipeline and Sampling Scheme

- **Base Architecture:** `facebook/opt-125m` (12 decoder layers, 12 attention heads, $d = 768$, $d_{\text{ff}} = 3{,}072$, exact-zero ReLU activations).
- **Benchmark Corpus:** WikiText-2 raw split ($M = 5{,}120$ evaluation tokens partitioned into $n = 40$ non-overlapping blocks of $B = 128$ tokens).
- **Multi-Domain Corpora (RQ5):** $N = 3{,}000$ tokens ($1{,}000$ each from Wikipedia, ML-ArXiv scientific abstracts, and AG-News).
- **Activation Capture Point:** Captured post-ReLU and pre-`fc2` (the intermediate expansion dimension $\mathbb{R}^{3{,}072}$). Non-padding tokens retained; stored as float16 tensors.

---

## 3. Multivariate Methodology by Research Question

### Phase 1: Descriptive Statistics and Distributional Taxonomy
- For each layer $\ell \in \{0, \dots, 11\}$: zero fraction $\hat{\zeta}^{(\ell)}$, active neuron count, positive-tail mean, standard deviation, skewness, and excess kurtosis.
- Establishes that post-ReLU activations follow a zero-inflated semi-continuous mixture distribution with point mass at zero and heavy positive tails.

### RQ1: Profile Analysis of Layer-Wise Sparsity
- **Sampling Unit:** Independent text block ($n = 40$ blocks, $q = 12$ layers).
- **Hypothesis:** $\mathcal{H}_0: \boldsymbol{\mu}_{\text{sparse}} = \zeta_0 \mathbf{1}_{12}$.
- **Statistical Tests:**
  - One-sample Hotelling's $T^2$ test on contrast vectors $\mathbf{C} \in \mathbb{R}^{11 \times 12}$ ($T^2 = 602{,}912.24$, $F(11, 29) = 40{,}756.31$, $p = 1.25 \times 10^{-57}$).
  - Repeated-Measures ANOVA ($F(11, 429) = 1{,}838.82$, $p < 10^{-300}$).
  - Greenhouse-Geisser sphericity correction ($\hat{\epsilon} = 0.1253$, $F^* = 1{,}838.82$, $p = 1.68 \times 10^{-46}$).
  - Non-parametric Friedman rank test ($Q = 432.99$, $\text{df} = 11$, $p = 5.99 \times 10^{-86}$).

### RQ2: Binary Co-Activation Dependence and Hierarchical Clustering
- **Coordinate Selection:** Prevalence filter $0.01 \le \hat{\pi}_j \le 0.99$ to remove inactive dead units; $m = 128$ sampled units.
- **Metrics:** Pairwise joint probability $\pi_{jk}$, marginal probabilities, and Pearson $\phi$-correlation coefficients:
  $$\phi_{jk} = \frac{\pi_{jk} - \pi_j \pi_k}{\sqrt{\pi_j(1-\pi_j)\pi_k(1-\pi_k)}}$$
- **Hierarchical Clustering:** Average-linkage agglomerative clustering on distance metric $d_{jk} = 1 - \phi_{jk}$ at cutoff $d = 0.70$.
- **Key Finding:** Non-monotonic U-shaped dependence: early coupling ($\bar{\phi} = +0.0685$, 94 clusters in Layer 00), near-orthogonal interior ($\bar{\phi} \approx 0.004$, 122 clusters in Layer 06), and output re-coupling ($\bar{\phi} = 0.0391$, 94 clusters in Layer 11).

### RQ3: Covariance PCA, Effective Rank, and Subspace Stability
- **Unstandardised Covariance PCA:** Principal component decomposition on centered activation matrix $\mathbf{S}^{(\ell)} = \frac{1}{N-1} \mathbf{X}_c^\top \mathbf{X}_c$.
- **Continuous Dimensionality Metrics:**
  - Participation Ratio: $\text{PR} = (\sum \lambda_i)^2 / \sum \lambda_i^2$
  - Roy-Vetterli Entropy Effective Rank: $\text{erank} = \exp\left(-\sum p_i \ln p_i\right)$, where $p_i = \lambda_i / \sum \lambda_k$.
- **Subspace Stability:** Split-half cross-validation evaluating the mean cosine of principal angles $\bar{\rho}_{10} \ge 0.878$.
- **Key Finding:** Severe rank collapse at early bottleneck (Layer 00 $\text{erank} = 1.51$), expanding to broad middle-layer feature manifolds (Layer 05–08 $\text{erank} \approx 45\text{--}46$), then compressing at output (Layer 11 $\text{erank} = 10.81$).

### RQ4: Static Weight Singular Spectra vs. Dynamic Activation Geometry
- **Weight Matrix SVD:** SVD on actual OPT-125M weight matrices $\mathbf{W}_{\text{fc1}} \in \mathbb{R}^{3072 \times 768}$ and $\mathbf{W}_{\text{fc2}} \in \mathbb{R}^{768 \times 3072}$.
- **Association Metrics:** Pearson correlation $r = 0.7991$ ($p = 0.0018$) and Spearman rank correlation $\rho = 0.8392$ ($p = 0.00064$).
- **Key Finding:** While static weight spectral capacity correlates with activation rank across depth, weight matrices maintain an order of magnitude higher effective rank ($\text{mean weight erank} \approx 500.5$ vs. $\text{activation erank} \approx 26.4$). Static parameter capacity is an upper bound, not a surrogate for dynamic rank.

### RQ5: Cross-Domain Manifold Invariance and Discriminant Analysis
- **Domain Corpora:** 3 populations ($n_g = 1{,}000$ per domain) projected onto $K = 8$ PCA subspace.
- **Multivariate Tests:**
  - One-Way MANOVA: Wilks' $\Lambda = 0.6939$ (Rao $F(16, 5980) = 74.94, p = 7.86 \times 10^{-223}$).
  - Pairwise Hotelling's Two-Sample $T^2$: All domain pairs reject equal centroids ($p < 10^{-50}$).
  - Box's M Test: $\chi^2(72) = 8{,}133.62$ ($p < 10^{-300}$), decisively rejecting covariance equality.
  - Supervised Classification (5-Fold Cross-Validation): Linear Discriminant Analysis ($55.83\% \pm 0.96\%$) and Quadratic Discriminant Analysis ($51.57\% \pm 2.70\%$) against chance baseline ($33.33\%$).
- **Key Finding:** FFN intermediate activations are domain-sensitive, establishing that generic pre-trained representations adapt their geometric distributions to incoming linguistic domains.
