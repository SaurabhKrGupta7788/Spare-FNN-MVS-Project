# Forensic Audit & Pure-MVS Revision Plan
**Project:** Statistical Characterisation of Sparse Activations and Low-Rank Latent Structures in Dense Feed-Forward Neural Networks (OPT-125M)  
**Target Codebase:** `Spare-FNN-MVS-Project` (`src/sparse_ffn`, `report_latex/main.tex`, `reports/`)  
**Audit Date:** November 2026  
**Auditor:** Senior Statistical & Software Engineering Forensic Review

---

## Executive Summary & Urgent Findings

This forensic audit evaluates the 31-page LaTeX project report (`report_latex/main.tex`), Python source code (`src/sparse_ffn/`), and serialized JSON execution outputs (`reports/`). 

### Urgent Finding 1: RQ4 Weight Matrix SVD Verification
* **Did RQ4 use actual `facebook/opt-125m` weights?**
  **NO.** In `configs/default.yaml`, `rq4.weights_path` is set to `null`. As a consequence, `src/sparse_ffn/analysis/rq4_weight_rank.py` (lines 76–99) falls back to synthesizing synthetic weight matrices in memory.
* **Why does Figure 4.7 / Table B.1 display unusually smooth straight-line singular values?**
  The synthetic generator explicitly enforces an exact exponential spectrum on random orthogonal bases via:
  ```python
  s_in = np.exp(-np.linspace(0, 1.5 * depth_factor, hidden_dim)) * 0.02 * np.sqrt(hidden_dim)
  s_out = np.exp(-np.linspace(0, 1.2 * depth_factor, hidden_dim)) * 0.02 * np.sqrt(ffn_dim)
  ```
  On a semilogarithmic plot ($\log s$ vs. index), an exponential decay is algebraically an **exact straight line** ($-\text{slope} \cdot k + C$). Furthermore, `depth_factor` is defined as a linear ramp across layers ($0.85 + 0.3 \cdot \frac{i}{11}$), causing the reported effective ranks in Table B.1 (`tab:app_rq4_layer_data`) to drop with artificial mechanical regularity ($\approx 7.8$ drop per layer for `fc1` and $\approx 6.1$ drop for `fc2`). **Real OPT-125M model weights were never downloaded or decomposed in RQ4.**

---

### Urgent Finding 2: RQ5 Domain Generation Verification
* **How were the three RQ5 domains (Wikipedia, Scientific, News) generated?**
  They were **artificially synthesized in memory from a single 1,000-token slice of WikiText-2 activations** in Layer 00 (`src/sparse_ffn/analysis/rq5_domain_stability.py`, lines 44–77):
  1. **Wikipedia:** An unperturbed copy of the base WikiText-2 activations (`base_activations.copy()`).
  2. **Scientific:** Formed by adding synthetic uniform scaling and random Gaussian shift plus a Bernoulli-activated spike to the same WikiText-2 activations:
     $$\mathbf{A}_{\text{sci}} = \max\left( \mathbf{A}_{\text{base}} \odot \mathbf{s}_{\text{sci}} + \boldsymbol{\mu}_{\text{sci}} + 0.15 \cdot \mathbf{m}_{\text{sci}}, \, 0 \right)$$
     where $\mathbf{s}_{\text{sci}} \sim \mathcal{U}(0.8, 1.3)$, $\boldsymbol{\mu}_{\text{sci}} \sim \mathcal{N}(0, 0.05)$, and $\mathbf{m}_{\text{sci}} \sim \text{Bernoulli}(0.08)$.
  3. **News:** Formed by applying another synthetic affine perturbation:
     $$\mathbf{A}_{\text{news}} = \max\left( \mathbf{A}_{\text{base}} \odot \mathbf{s}_{\text{news}} + \boldsymbol{\mu}_{\text{news}}, \, 0 \right)$$
     where $\mathbf{s}_{\text{news}} \sim \mathcal{U}(0.9, 1.15)$ and $\boldsymbol{\mu}_{\text{news}} \sim \mathcal{N}(-0.01, 0.04)$.
* **Consequence:** No real scientific (e.g., arXiv/PubMed) or news (e.g., CNN/DailyMail) corpora were ever processed through the language model. Because artificial persistent feature-wise offsets were applied across all tokens, the three clusters became artificially separated with virtually zero overlap. This fully explains why MANOVA yielded an astronomically implausible $F$-statistic of $244,389.3$ ($\Lambda = 2.33 \times 10^{-6}$) and LDA achieved an unrealistic $100.0\% \pm 0.0\%$ cross-validated accuracy.

---

## 1. Mathematical & Statistical Audit

```
+---------------------------------------------------------------------------------------------------+
| Issue                              | Observed in Code / Report              | Required MVS Standard       |
+------------------------------------+----------------------------------------+-----------------------------+
| 1.1 Assumption of Normality        | Zero tests for multivariate normality  | Mardia's test / Chi-Square  |
|                                    | on activations with >88% zeros         | Q-Q plots of Mahalanobis D2 |
| 1.2 LDA with Unequal Covariances   | LDA applied despite Box's M rejecting  | QDA or Regularized DA       |
|                                    | covariance equality (p = 0.0)          | when Sigma_1 != Sigma_2     |
| 1.3 RQ4 Correlation Overclaim      | Pearson r = -0.607 (p = 0.036),        | Report non-significance     |
|                                    | but Spearman rho = -0.476 (p = 0.118)  | for n = 12 rank correlation |
| 1.4 Profile Analysis Formulation   | Only tested flatness; omitted parallel | Full 3-stage Profile        |
|                                    | and level hypotheses                   | Analysis decomposition      |
| 1.5 Box's M Non-Normality Bias     | Reported chi2(72) = 15,635.1           | Highlight kurtosis inflation|
|                                    | without noting kurtosis inflation      | or use permutation tests    |
+---------------------------------------------------------------------------------------------------+
```

### Detailed Statistical Discrepancies
1. **Severe Violation of Multivariate Normality:**
   All classical multivariate test statistics employed—Hotelling's $T^2$, One-Way MANOVA (Wilks' $\Lambda$, Pillai's trace, Hotelling-Lawley trace, Roy's root), and Box's M test—derive their exact null distributions under the assumption that observation vectors follow a continuous multivariate normal distribution $\mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. However, post-ReLU activations contain 88.9% to 98.8% exact zeros. This extreme zero-inflation and positive skewness (sample skewness reaches $54.7$ in Layer 0) heavily violates normality. The report never acknowledges this limitation or computes Mardia's multivariate skewness and kurtosis.
2. **Methodological Inconsistency: LDA vs. Box's M Test:**
   Linear Discriminant Analysis (LDA) assumes homoscedasticity ($\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2 = \dots = \boldsymbol{\Sigma}_G$). In Table 4.4, Box's M test decisively rejects covariance equality ($M = 15,693.7, \chi^2(72) = 15,635.1, p < 10^{-100}$). Applying standard linear pooled discriminant analysis when group covariances are proven unequal is theoretically contradictory; Quadratic Discriminant Analysis (QDA) or an explicit discussion of quadratic decision boundaries is mandatory.
3. **Small-Sample Rank Correlation (RQ4):**
   With only $N = 12$ layers, Pearson's correlation ($r = -0.607, p = 0.036$) is easily swayed by endpoints (Layers 0 and 2). The non-parametric Spearman rank correlation ($\rho = -0.476, p = 0.118$) fails to reach statistical significance at $\alpha = 0.05$. Asserting a confirmed "decoupling between weights and activations" based on an insignificant rank correlation on synthetic weights is scientifically invalid.

---

## 2. Typesetting & LaTeX Deficiencies (`report_latex/main.tex`)

1. **Title Page Title / Course Inconsistency:**
   - The subtitle displays: `CSP 1602: Software Engineering Lab / Multivariate Statistical Analysis`. This conflates two separate courses.
   - An empty figure placeholder is left on the title page:
     ```latex
     \begin{figure}[h]
         \centering
         % Department Details
     \end{figure}
     ```
     This creates unwanted vertical gaps and compilation warnings.
2. **Hardcoded Equation Suffixes:**
   Equations 2.12–2.22, 2.25, 2.27, etc., include hardcoded text tags inside the math environment, such as:
   ```latex
   \text{erank} = \exp(H) \quad \text{(RQ3)}
   ```
   This duplicates equation numbers and clashes with standard academic journal style.
3. **Overleaf & Path Portability:**
   `main.tex` specifies `\graphicspath{{figures/}}`. In the root repository and `.zip` archives, several asset images reside in `assets/` while others are duplicated in `report_latex/figures/` and `report_figures/`.
4. **Table B.1 Numbering & Labels:**
   In Appendix B, the table is labeled `\label{tab:app_rq4_layer_data}`, while text in Section 4 references Table B.1 informally. Explicit cross-referencing must use standard LaTeX `\ref{tab:...}` with consistent caption numbering.

---

## 3. Data Integrity & Codebase Artifacts

1. **Leakage of Local Workstation Paths in Committed Reports:**
   The JSON files in `reports/` contain absolute local file system paths from two distinct machines:
   - `descriptive_report.json`:  
     `C:\Users\GANAPATHI\Desktop\NIT\Academics\SEM7\Multivariate stastical analysis\MVS_Sparse_FFN_Starter_Project\...`
   - `profile_analysis_report.json`, `rq2_neuron_dependence_report.json`, `rq3_latent_structure_report.json`, `rq4_weight_rank_report.json`:  
     `C:\Users\100ra\Desktop\sparse_ffn_mvs_project\...`  
   All paths in reports must be relative to repository root (`artifacts/...`).
2. **Sample Size Disparity:**
   - Total extracted tokens: 5,120 tokens across 40 blocks.
   - For an intermediate dimension of $p = 3,072$, computing raw sample covariances without PCA projection or regularized shrinkage creates near-singular matrices ($N \approx 1.6 \times p$).
   - The report needs explicit reporting of condition numbers and shrinkage parameters ($\lambda_{\text{shrink}}$).

---

## 4. Pure-MVS Syllabus Alignment (CSP 1602 / MVS Curriculum)

To ensure this project satisfies rigorous Multivariate Statistical Analysis examination standards, the following classical curriculum components must be formalized:

```
+--------------------------------------------------------------------------------------------------+
| Curriculum Module               | Current Status in Repo            | Planned Addition           |
+---------------------------------+-----------------------------------+----------------------------+
| 1. Profile Analysis             | Flatness only (1-sample Hotelling)| Parallelism, Levels, and   |
|                                 |                                   | Flatness decomposition     |
| 2. Sampling Adequacy for PCA    | Omitted                           | Kaiser-Meyer-Olkin (KMO) & |
|                                 |                                   | Bartlett's Sphericity Test |
| 3. Multivariate Normality Check | Omitted                           | Mardia's b1p, b2p &        |
|                                 |                                   | Mahalanobis Chi2 Q-Q plot  |
| 4. Covariance Testing           | Box's M without robustness check  | Permutation Box's M &      |
|                                 |                                   | Frobenius Norm distance    |
| 5. Discriminant Analysis        | 2D LDA with 100% artificial acc.  | QDA comparison +           |
|                                 |                                   | Wilks' Lambda step-down    |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. De-AI Prose & Academic Tone Overhaul

The prose in `main.tex` and presentation files contains tell-tale LLM boilerplate phrases that should be revised to formal statistical reporting:
* **Current:** *"decisively reject profile flatness across network depth, identifying a dramatic early-intermediate sparsity spike"*  
  **Revised:** *"The hypothesis of uniform mean sparsity across depth is rejected at the $\alpha = 0.01$ level using Hotelling's $T^2$ contrast test ($F(11, 29) = 40,756.3, p < 10^{-15}$). Mean activation sparsity exhibits non-monotonic behavior, reaching an empirical maximum of $98.77\%$ at Layer 3."*
* **Current:** *"proving empirical activation collapse is driven by dynamic token representations rather than parameter bottlenecks"*  
  **Revised:** *"indicating that the effective dimensionality of activations is substantially lower than the algebraic rank of the parameter matrices, consistent with non-linear subspace compression."*
* **Current:** *"achieve 100% cross-validated domain classification accuracy, confirming that FFN intermediate activations encode domain-specific style and semantics."*  
  **Revised:** *"Under the preliminary synthetic domain perturbation protocol, cross-validated discriminant accuracy reaches ceiling performance; evaluation on natural multi-corpus benchmarks is required to establish generalization."*

---

## 6. Actionable Step-by-Step Revision Roadmap

### Step 1: Real Weight Extraction for RQ4
* Replace synthetic generation in `rq4_weight_rank.py` with direct extraction from `facebook/opt-125m` via Hugging Face `transformers` or PyTorch state dict:
  ```python
  from transformers import OPTForCausalLM
  model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
  fc1_weights = [layer.fc1.weight.detach().cpu().numpy() for layer in model.model.decoder.layers]
  fc2_weights = [layer.fc2.weight.detach().cpu().numpy() for layer in model.model.decoder.layers]
  ```
* Recompute true SVD spectra, participation ratios, and genuine Pearson/Spearman correlations.

### Step 2: Real Multi-Domain Token Extraction for RQ5
* Ingest genuine samples from three standard public text domains:
  1. **Wikipedia:** `wikitext-2-raw-v1`
  2. **Scientific:** `scientific_papers` (arXiv or PubMed abstracts)
  3. **News:** `cnn_dailymail` or `cc_news`
* Pass all three through frozen OPT-125M forward hooks to record true empirical activations.
* Re-run MANOVA, Box's M, LDA, and QDA on real domain activations.

### Step 3: Implement Pure-MVS Diagnostic Suite
* Add Bartlett's test of sphericity and KMO index to Phase 3 PCA pipeline (`src/sparse_ffn/analysis/rq3_latent_structure.py`).
* Add Mardia's multivariate skewness/kurtosis calculation and Mahalanobis distance Q-Q plot export.
* Add full 3-hypothesis profile analysis decomposition (flatness, parallelism across splits, levels).

### Step 4: Clean and Recompile LaTeX (`main.tex`)
* Fix title page formatting, clean empty figure blocks, and embed NIT Mizoram crest cleanly.
* Strip hardcoded `\text{(RQ*)}` tags from equation environments.
* Update all tables (Table 4.1 through Table B.1) with genuine non-synthetic numbers.
* Remove AI buzzwords and frame findings with scientific precision and acknowledged boundary conditions.
