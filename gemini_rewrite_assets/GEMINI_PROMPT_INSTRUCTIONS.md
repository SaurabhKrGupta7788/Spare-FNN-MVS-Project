# Master Gemini Prompt: Comprehensive Pure-MVS LaTeX Report Rewrite

> **Instructions for the User:**  
> Copy and paste the entire block below into Gemini along with the attached files (`main.tex`, `references.bib`, `ground_truth_statistics.json`, `mvs_report_audit_and_revision_plan.md`). Alternatively, upload `Gemini_Rewrite_Pack.zip` and send this prompt!

---

```markdown
You are a distinguished Professor of Multivariate Statistical Analysis (MVS) and an expert LaTeX typesetter reviewing and revising a 31-page academic project report for the Department of Computer Science & Engineering and Department of Mathematics at the National Institute of Technology Mizoram.

### Context & Task Overview
I have uploaded the current project codebase files:
1. `main.tex`: The current 31-page LaTeX project report.
2. `references.bib`: The BibTeX bibliography file.
3. `ground_truth_statistics.json`: The complete, verified, empirical ground-truth numbers from Phase 1 through RQ5.
4. `mvs_report_audit_and_revision_plan.md`: The complete forensic audit detailing mathematical bugs, typesetting defects, data integrity checks, and required syllabus additions.

Your task is to completely rewrite and polish `main.tex` from start to finish. The output must be a pristine, 100% complete, fully compilable LaTeX document that builds cleanly on Overleaf (or via pdflatex) without errors, missing references, or placeholder omissions.

---

### Critical Requirements & Corrections

#### 1. Title Page & Administrative Metadata
- Course Title: Change strictly to `CSP 1602: Multivariate Statistical Analysis` (remove `/ Software Engineering Lab`).
- Guidance Details: Under the guidance of `Dr. C Lalengmawia`, Assistant Professor, Department of CSE, National Institute of Technology Mizoram.
- Authors: `Lakshmi Ganapathi Kodi` (BT23MA005) & `Saurabh Kumar Guptha` (BT23MA006), Semester 7, November 2026.
- Institutional Emblem: Properly embed `\includegraphics[width=2.5cm]{nit_logo.png}` inside a clean centered block on the title page (fix the empty figure environment).

#### 2. Pure-MVS Syllabus Alignment & Mathematical Rigor
- Multivariate Normality: Add formal mathematical commentary acknowledging the zero-inflated, non-negative nature of post-ReLU activations ($88.9\%\text{--}98.8\%$ zeros). Explicitly discuss departures from multivariate normality, Mardia's kurtosis implications, and why PCA (covariance-based) remains mathematically valid as an orthogonal projection without normality assumptions.
- Profile Analysis (RQ1): Present the complete 3-stage Profile Analysis framework (Flatness, Parallelism, and Levels) rather than confusing it with simple univariate RM-ANOVA. Detail contrast matrix $\mathbf{C}$ ($11 \times 12$), Hotelling's $T^2$ ($T^2 = 602,912.2, F(11, 29) = 40,756.3$), Greenhouse-Geisser sphericity correction ($\hat{\epsilon} = 0.1253, F = 1,838.82$), and Friedman test ($Q = 432.99, df = 11$).
- Sampling Adequacy (RQ3): Integrate theoretical foundations for Kaiser-Meyer-Olkin (KMO) Measure of Sampling Adequacy and Bartlett's Test of Sphericity before PCA/Factor Analysis.
- Box's M Test vs. LDA (RQ5): Address the fundamental MVS dilemma: Box's M test decisively rejects covariance equality ($\chi^2(72) = 8,133.62, p < 10^{-100}$). Provide both Linear Discriminant Analysis ($55.83\%$) and Quadratic Discriminant Analysis ($51.57\%$), explaining why QDA provides a theoretically appropriate quadratic boundary under heteroscedasticity.

#### 3. Real Empirical Data Integration (Never Use Synthetic Numbers!)
All numbers must strictly match `ground_truth_statistics.json`:
- RQ4 (Real OPT-125M Weights):
  - Actual weights from frozen `facebook/opt-125m` were decomposed via SVD (NOT synthetic random matrices!).
  - Mean weight entropy effective rank = $500.53$, Participation Ratio = $281.52$.
  - Pearson correlation: $r = +0.799$ ($p = 0.0018$, Statistically Significant).
  - Spearman rank correlation: $\rho = +0.839$ ($p = 0.00064$, Highly Significant Positive Rank Correlation).
  - Crucial Scientific Interpretation: Activations and weights are STRONGLY POSITIVELY COUPLED across depth. Early-layer parameter bottlenecks (Layer 0: $\text{erank}_W = 371.04$, Layer 2: $\text{erank}_W = 428.70$) directly reinforce the severe empirical activation collapse.
  - Table B.1 (`tab:app_rq4_layer_data`): Use the exact 12-layer numbers from `ground_truth_statistics.json`.
- RQ5 (Genuine Multi-Domain Datasets):
  - Ingested 1,000 real tokens per domain ($N = 3,000$ total) across Wikipedia (`wikitext-2-raw-v1`), Scientific (`ML-ArXiv-Papers`), and News (`ag_news`).
  - One-Way MANOVA: Wilks' $\Lambda = 0.6939$, Rao's $F(16, 5980) = 74.94, p = 7.86 \times 10^{-223}$ (Reject Equal Means).
  - Pillai's trace = $0.3208$, Hotelling-Lawley = $0.4201$, Roy's root = $0.3619$.
  - Box's M: $M = 8,164.09, \chi^2(72) = 8,133.62, p < 10^{-100}$.
  - Pairwise Hotelling's $T^2$: Wiki vs Sci ($T^2 = 801.16, F = 99.79$), Wiki vs News ($T^2 = 624.99, F = 77.85$), Sci vs News ($T^2 = 283.17, F = 35.27$).
  - 5-Fold CV LDA Accuracy: $55.83\% \pm 0.96\%$ (well above $33.33\%$ chance baseline).
  - 5-Fold CV QDA Accuracy: $51.57\% \pm 2.70\%$.

#### 4. Typesetting & Formatting Hygiene
- Remove all hardcoded equation tags like `\quad \text{(RQ1)}` or `\quad \text{(RQ3)}` from inside `\begin{equation} ... \end{equation}`. Use standard LaTeX `\label{eq:...}` and `\eqref{eq:...}`.
- Fix all tabular widths to avoid overfull `\hbox` warnings. Use appropriate `tabularx` column specifiers (`X`, `c`, `r`, `l`) with clear booktabs lines (`\toprule`, `\midrule`, `\bottomrule`).
- Ensure all figures use exact relative filenames available in the `figures/` folder:
  - `nit_logo.png`
  - `fig_layer_sparsity_bars.png`
  - `fig1_mean_sparsity_profile.png`
  - `fig2a_phi_corr_l00.png`
  - `fig2b_phi_corr_l06.png`
  - `fig2c_dendrogram_l00.png`
  - `fig3a_pca_scree_l00.png`
  - `fig3b_pca_scree_l06.png`
  - `fig3c_factor_loadings_l00.png`
  - `fig4_activation_vs_weight_rank.png`
  - `fig5_lda_domain_separation.png`
- Fix fancyhdr `\headheight` warning by setting `\setlength{\headheight}{14.5pt}` in the preamble.
- Ensure proper cross-referencing between text and Table B.1 (`\ref{tab:app_rq4_layer_data}`).

#### 5. Academic Tone & De-AI Overhaul
- Strip all repetitive AI boilerplate, hyperbolic superlatives ("staggering", "decisively proving", "dramatic spike", "revolutionizes").
- Adopt the dignified, rigorous, precise tone of classical multivariate literature (Anderson, Johnson & Wichern, C.R. Rao).
- Discuss statistical limitations objectively: finite sample sizes ($N = 5,120$ tokens), pilot text-block boundaries, conditioning of high-dimensional covariance matrices, and the trade-offs of linear vs. quadratic discriminant surfaces.

---

### Deliverable
Output the complete, full-text LaTeX code for `main.tex`, starting from `\documentclass[12pt,a4paper,oneside]{report}` through `\end{document}`. Do not abbreviate or truncate chapters.
```
