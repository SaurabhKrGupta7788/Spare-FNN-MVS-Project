"""Run real RQ4 (true OPT-125M weights SVD) and real RQ5 (genuine multi-domain text activations)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f, chi2, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
ASSETS_DIR = REPO_ROOT / "assets"
REPORT_FIGS_DIR = REPO_ROOT / "report_latex" / "figures"


# ==============================================================================
# Helper Functions for MVS
# ==============================================================================
def compute_spectral_pr(s: np.ndarray) -> float:
    s2 = np.maximum(np.asarray(s, dtype=np.float64), 0) ** 2
    sum_s2 = float(np.sum(s2))
    sum_s4 = float(np.sum(s2**2))
    return float(sum_s2**2 / max(sum_s4, 1e-15))


def compute_spectral_erank(s: np.ndarray) -> float:
    s2 = np.maximum(np.asarray(s, dtype=np.float64), 0) ** 2
    total = np.sum(s2)
    if total <= 0:
        return 0.0
    p = s2 / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def hotelling_two_sample_t2(x1: np.ndarray, x2: np.ndarray) -> dict[str, Any]:
    n1, p = x1.shape
    n2, p2 = x2.shape
    diff = np.mean(x1, axis=0) - np.mean(x2, axis=0)
    s1 = np.cov(x1, rowvar=False, ddof=1)
    s2 = np.cov(x2, rowvar=False, ddof=1)
    s_pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    s_inv_diff = np.linalg.solve(s_pooled, diff)
    t2 = float((n1 * n2 / (n1 + n2)) * np.dot(diff, s_inv_diff))
    df1 = p
    df2 = n1 + n2 - p - 1
    f_stat = float(((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * t2)
    p_val = float(f.sf(f_stat, df1, df2))
    return {
        "hotelling_t2": t2,
        "f_statistic": f_stat,
        "df1": int(df1),
        "df2": int(df2),
        "p_value": p_val,
        "decision": "reject_equal_means" if p_val < 0.05 else "do_not_reject",
    }


def compute_manova(domain_data: dict[str, np.ndarray]) -> dict[str, Any]:
    groups = list(domain_data.keys())
    g = len(groups)
    p = domain_data[groups[0]].shape[1]
    n_total = sum(len(d) for d in domain_data.values())
    grand_mean = np.zeros(p, dtype=np.float64)
    for data in domain_data.values():
        grand_mean += np.sum(data, axis=0)
    grand_mean /= n_total

    h_mat = np.zeros((p, p), dtype=np.float64)
    e_mat = np.zeros((p, p), dtype=np.float64)
    for data in domain_data.values():
        n_k = len(data)
        mean_k = np.mean(data, axis=0)
        diff = (mean_k - grand_mean).reshape(-1, 1)
        h_mat += n_k * (diff @ diff.T)
        dev = data - mean_k
        e_mat += dev.T @ dev

    try:
        e_inv_h = np.linalg.solve(e_mat, h_mat)
        eigvals = np.sort(np.maximum(np.real(np.linalg.eigvals(e_inv_h)), 0.0))[::-1]
    except np.linalg.LinAlgError:
        e_reg = e_mat + 1e-6 * np.trace(e_mat) / p * np.eye(p)
        e_inv_h = np.linalg.solve(e_reg, h_mat)
        eigvals = np.sort(np.maximum(np.real(np.linalg.eigvals(e_inv_h)), 0.0))[::-1]

    s = min(p, g - 1)
    non_zero = eigvals[:s]
    wilks = float(np.prod(1.0 / (1.0 + non_zero)))
    pillai = float(np.sum(non_zero / (1.0 + non_zero)))
    hotelling_lawley = float(np.sum(non_zero))
    roys = float(non_zero[0] if len(non_zero) > 0 else 0.0)

    w = n_total - 1 - (p + g) / 2.0
    term = p**2 * (g - 1) ** 2 - 4
    denom = p**2 + (g - 1) ** 2 - 5
    t_val = np.sqrt(term / denom) if (term > 0 and denom > 0) else 1.0
    df1 = p * (g - 1)
    df2 = w * t_val - (p * (g - 1) - 2) / 2.0
    if 0 < wilks < 1:
        f_approx = float(((1.0 - wilks ** (1.0 / t_val)) / (wilks ** (1.0 / t_val))) * (df2 / df1))
        p_val = float(f.sf(f_approx, df1, df2))
    else:
        f_approx, p_val = 0.0, 1.0

    return {
        "test": "One-Way MANOVA",
        "n_total": int(n_total),
        "n_groups": int(g),
        "n_features": int(p),
        "wilks_lambda": wilks,
        "rao_f_statistic": f_approx,
        "df1": float(df1),
        "df2": float(df2),
        "p_value": p_val,
        "pillai_trace": pillai,
        "hotelling_lawley_trace": hotelling_lawley,
        "roys_largest_root": roys,
        "decision": "reject_equal_domain_means" if p_val < 0.05 else "do_not_reject",
    }


def box_m_test(domain_data: dict[str, np.ndarray]) -> dict[str, Any]:
    groups = list(domain_data.keys())
    g = len(groups)
    p = domain_data[groups[0]].shape[1]
    covs = [np.cov(domain_data[k], rowvar=False, ddof=1) for k in groups]
    dfs = [len(domain_data[k]) - 1 for k in groups]
    total_df = sum(dfs)
    s_pooled = sum(df_k * s_k for df_k, s_k in zip(dfs, covs)) / total_df

    sign_pool, logdet_pool = np.linalg.slogdet(s_pooled)
    logdet_sum = sum(df_k * np.linalg.slogdet(s_k)[1] for df_k, s_k in zip(dfs, covs))
    m_stat = float(total_df * logdet_pool - logdet_sum)
    c_term = (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    c_factor = 1.0 - c_term * (sum(1.0 / df_k for df_k in dfs) - (1.0 / total_df))
    chi2_stat = float(m_stat * c_factor)
    chi2_df = 0.5 * p * (p + 1) * (g - 1)
    p_val = float(chi2.sf(chi2_stat, chi2_df))
    return {
        "box_m_statistic": m_stat,
        "chi2_statistic": chi2_stat,
        "degrees_of_freedom": float(chi2_df),
        "p_value": p_val,
        "decision": "reject_equal_covariances" if p_val < 0.05 else "do_not_reject",
    }


# ==============================================================================
# 1. Real RQ4: Actual OPT-125M Weights SVD
# ==============================================================================
def run_real_rq4(model: AutoModelForCausalLM) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logger.info("Starting Real RQ4: Extracting and decomposing actual facebook/opt-125m weights...")
    layers = model.model.decoder.layers
    n_layers = len(layers)

    layer_rows = []
    spectrum_rows = []
    weights_dict = {}

    for i, layer in enumerate(layers):
        name = f"layer_{i:02d}"
        w_in = layer.fc1.weight.detach().cpu().numpy().astype(np.float32)   # (3072, 768)
        w_out = layer.fc2.weight.detach().cpu().numpy().astype(np.float32)  # (768, 3072)
        weights_dict[f"{name}_fc1"] = w_in
        weights_dict[f"{name}_fc2"] = w_out

        s_in = np.linalg.svd(w_in, compute_uv=False)
        s_out = np.linalg.svd(w_out, compute_uv=False)

        pr_in = compute_spectral_pr(s_in)
        erank_in = compute_spectral_erank(s_in)
        pr_out = compute_spectral_pr(s_out)
        erank_out = compute_spectral_erank(s_out)

        layer_rows.append({
            "layer": name,
            "fc1_rank_full": int(min(w_in.shape)),
            "fc1_participation_ratio": pr_in,
            "fc1_entropy_effective_rank": erank_in,
            "fc2_rank_full": int(min(w_out.shape)),
            "fc2_participation_ratio": pr_out,
            "fc2_entropy_effective_rank": erank_out,
            "mean_weight_entropy_effective_rank": (erank_in + erank_out) / 2.0,
            "mean_weight_participation_ratio": (pr_in + pr_out) / 2.0,
        })

        for rank_idx, (v_in, v_out) in enumerate(zip(s_in[:128], s_out[:128]), start=1):
            spectrum_rows.append({
                "layer": name,
                "singular_vector_index": rank_idx,
                "fc1_singular_value": float(v_in),
                "fc1_energy_fraction": float(v_in**2 / np.sum(s_in**2)),
                "fc2_singular_value": float(v_out),
                "fc2_energy_fraction": float(v_out**2 / np.sum(s_out**2)),
            })

    weight_summary_df = pd.DataFrame(layer_rows)
    weight_spectra_df = pd.DataFrame(spectrum_rows)

    # Save real weights
    weights_save_path = ARTIFACTS_DIR / "analysis" / "train" / "opt_125m_ffn_weights_real.npz"
    weights_save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(weights_save_path, **weights_dict)

    # Load RQ3 activation effective dimensionality
    # Look for existing RQ3 summary in artifacts or reports
    rq3_json_path = REPORTS_DIR / "rq3_latent_structure_report.json"
    if rq3_json_path.is_file():
        rq3_data = json.loads(rq3_json_path.read_text(encoding="utf-8"))
        rq3_layers = rq3_data["layers"]
        act_erank = [l["entropy_effective_rank"] for l in rq3_layers]
        act_pr = [l["participation_ratio"] for l in rq3_layers]
    else:
        raise FileNotFoundError("RQ3 report not found")

    weight_summary_df["activation_entropy_erank"] = act_erank
    weight_summary_df["activation_participation_ratio"] = act_pr

    w_erank = weight_summary_df["mean_weight_entropy_effective_rank"].to_numpy()
    w_pr = weight_summary_df["mean_weight_participation_ratio"].to_numpy()

    p_corr_erank, p_val_erank = pearsonr(act_erank, w_erank)
    s_corr_erank, s_val_erank = spearmanr(act_erank, w_erank)
    p_corr_pr, p_val_pr = pearsonr(act_pr, w_pr)
    s_corr_pr, s_val_pr = spearmanr(act_pr, w_pr)

    rq4_summary = {
        "n_layers": n_layers,
        "pearson_correlation_entropy_effective_rank": float(p_corr_erank),
        "pearson_p_value_entropy_effective_rank": float(p_val_erank),
        "spearman_correlation_entropy_effective_rank": float(s_corr_erank),
        "spearman_p_value_entropy_effective_rank": float(s_val_erank),
        "pearson_correlation_participation_ratio": float(p_corr_pr),
        "pearson_p_value_participation_ratio": float(p_val_pr),
        "spearman_correlation_participation_ratio": float(s_corr_pr),
        "spearman_p_value_participation_ratio": float(s_val_pr),
        "mean_activation_entropy_erank": float(np.mean(act_erank)),
        "mean_weight_entropy_erank": float(np.mean(w_erank)),
        "mean_activation_participation_ratio": float(np.mean(act_pr)),
        "mean_weight_participation_ratio": float(np.mean(w_pr)),
        "is_real_weights": True,
    }

    # Plot real SVD spectrum and comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    layer_0_spec = weight_spectra_df[weight_spectra_df["layer"] == "layer_00"]
    layer_last_spec = weight_spectra_df[weight_spectra_df["layer"] == "layer_11"]
    axes[0].plot(layer_0_spec["singular_vector_index"], layer_0_spec["fc1_singular_value"], label="Layer 00 fc1", color="#1f77b4")
    axes[0].plot(layer_0_spec["singular_vector_index"], layer_0_spec["fc2_singular_value"], label="Layer 00 fc2", linestyle="--", color="#1f77b4")
    axes[0].plot(layer_last_spec["singular_vector_index"], layer_last_spec["fc1_singular_value"], label="Layer 11 fc1", color="#ff7f0e")
    axes[0].plot(layer_last_spec["singular_vector_index"], layer_last_spec["fc2_singular_value"], label="Layer 11 fc2", linestyle="--", color="#ff7f0e")
    axes[0].set_title("Real OPT-125M Weight SVD Spectra")
    axes[0].set_xlabel("Singular Vector Index")
    axes[0].set_ylabel("Singular Value (Log Scale)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    x_layers = np.arange(n_layers)
    axes[1].plot(x_layers, act_erank, marker="o", linewidth=1.8, label="Activation erank", color="#2ca02c")
    axes[1].plot(x_layers, w_erank, marker="s", linewidth=1.8, label="Real Weight erank", color="#d62728")
    axes[1].set_xticks(x_layers, weight_summary_df["layer"], rotation=45, ha="right")
    axes[1].set_title("Effective Rank across Depth")
    axes[1].set_xlabel("FFN Layer")
    axes[1].set_ylabel("Entropy Effective Rank")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].scatter(w_erank, act_erank, color="#9467bd", s=50, edgecolors="black", zorder=3)
    if np.std(w_erank) > 1e-6:
        m, b = np.polyfit(w_erank, act_erank, 1)
        x_line = np.linspace(min(w_erank), max(w_erank), 50)
        axes[2].plot(x_line, m * x_line + b, color="#8c564b", linestyle="--", label=f"Fit (r={p_corr_erank:.2f}, rho={s_corr_erank:.2f})")
    for idx, row in weight_summary_df.iterrows():
        axes[2].annotate(row["layer"].replace("layer_", "L"), (row["mean_weight_entropy_effective_rank"], row["activation_entropy_erank"]), fontsize=8, alpha=0.8)
    axes[2].set_title("Activation erank vs Real Weight erank")
    axes[2].set_xlabel("Weight Effective Rank")
    axes[2].set_ylabel("Activation Effective Rank")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "rq4_weight_activation.png", dpi=180)
    fig.savefig(REPORT_FIGS_DIR / "fig4_activation_vs_weight_rank.png", dpi=180)
    plt.close(fig)

    # Save real JSON report
    report_content = {
        "phase": "rq4_weight_rank",
        "description": "Comparison of empirical activation dimensionality and REAL OPT-125M weight matrix rank.",
        "weights_saved_to": str(weights_save_path.resolve()),
        "correlation_metrics": rq4_summary,
        "layer_data": weight_summary_df.to_dict(orient="records"),
    }
    (REPORTS_DIR / "rq4_weight_rank_report.json").write_text(json.dumps(report_content, indent=2), encoding="utf-8")
    logger.info("Real RQ4 completed successfully! Pearson r=%.3f (p=%.4f), Spearman rho=%.3f (p=%.4f)",
                p_corr_erank, p_val_erank, s_corr_erank, s_val_erank)
    return weight_summary_df, weight_spectra_df, rq4_summary


# ==============================================================================
# 2. Real RQ5: Genuine Multi-Domain Activation Extraction and MVS
# ==============================================================================
def extract_domain_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    target_layer_idx: int = 0,
    target_tokens: int = 1000,
) -> np.ndarray:
    """Extract intermediate post-ReLU activations from OPT for a list of texts."""
    collected_acts = []
    layer = model.model.decoder.layers[target_layer_idx]

    # Hook to capture post-ReLU
    current_act = []
    def hook_fn(module, input, output):
        # input[0] to fc2 is post-ReLU activations
        # or fc1 output passed through activation_fn
        pass

    # We can explicitly compute: act = relu(fc1(x))
    # We register forward hook on layer.fc1
    hook_storage = []
    def fc1_hook(mod, inp, out):
        # out is pre-activation: shape (batch, seq, 3072)
        act = torch.relu(out)
        hook_storage.append(act.detach())

    handle = layer.fc1.register_forward_hook(fc1_hook)

    with torch.no_grad():
        for text in texts:
            if not text or len(text.strip()) == 0:
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            hook_storage.clear()
            _ = model.model.decoder(input_ids=input_ids, attention_mask=attention_mask)
            if hook_storage:
                raw_out = hook_storage[0]
                act = raw_out if raw_out.ndim == 2 else raw_out[0]  # shape (seq_len, 3072)
                mask = attention_mask[0].bool()
                valid_act = act[mask].cpu().numpy().astype(np.float32)
                collected_acts.append(valid_act)
                if sum(len(a) for a in collected_acts) >= target_tokens:
                    break

    handle.remove()
    all_acts = np.vstack(collected_acts)
    return all_acts[:target_tokens]


def run_real_rq5(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device) -> dict[str, Any]:
    logger.info("Starting Real RQ5: Extracting activations on genuine Wikipedia, Scientific, and News text...")

    # Domain 1: Wikipedia (from wikitext)
    ds_wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [row["text"] for row in ds_wiki if len(row["text"].strip()) > 50][:100]

    # Domain 2: Scientific (from ML-ArXiv-Papers)
    ds_sci = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    sci_texts = [f"{t}. {a}" for t, a in zip(ds_sci[:100]["title"], ds_sci[:100]["abstract"])]

    # Domain 3: News (from ag_news)
    ds_news = load_dataset("ag_news", split="test")
    news_texts = ds_news[:100]["text"]

    n_target = 1000
    acts_wiki = extract_domain_activations(model, tokenizer, wiki_texts, device, target_layer_idx=0, target_tokens=n_target)
    acts_sci = extract_domain_activations(model, tokenizer, sci_texts, device, target_layer_idx=0, target_tokens=n_target)
    acts_news = extract_domain_activations(model, tokenizer, news_texts, device, target_layer_idx=0, target_tokens=n_target)

    logger.info("Extracted shapes: Wiki=%s, Sci=%s, News=%s", acts_wiki.shape, acts_sci.shape, acts_news.shape)

    domain_data_raw = {
        "wikipedia": acts_wiki,
        "scientific": acts_sci,
        "news": acts_news,
    }

    # PCA Dimension Reduction (K=8 as per proposal Step 8)
    domain_names = ("wikipedia", "scientific", "news")
    pooled_raw = np.vstack([domain_data_raw[k] for k in domain_names])
    pca = PCA(n_components=8, random_state=42)
    pca.fit(pooled_raw)

    reduced_domain_data = {k: pca.transform(domain_data_raw[k]) for k in domain_names}

    # MANOVA
    manova_res = compute_manova(reduced_domain_data)
    # Box's M
    box_m_res = box_m_test(reduced_domain_data)

    # Pairwise Hotelling T^2
    pairwise_results = []
    for i in range(len(domain_names)):
        for j in range(i + 1, len(domain_names)):
            d1, d2 = domain_names[i], domain_names[j]
            res = hotelling_two_sample_t2(reduced_domain_data[d1], reduced_domain_data[d2])
            pairwise_results.append({
                "domain_1": d1,
                "domain_2": d2,
                **res
            })

    # LDA & QDA
    all_x = np.vstack([reduced_domain_data[k] for k in domain_names])
    all_y = np.concatenate([[idx] * len(reduced_domain_data[k]) for idx, k in enumerate(domain_names)])

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_coords = lda.fit_transform(all_x, all_y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lda_scores = cross_val_score(lda, all_x, all_y, cv=cv, scoring="accuracy")

    qda = QuadraticDiscriminantAnalysis()
    qda_scores = cross_val_score(qda, all_x, all_y, cv=cv, scoring="accuracy")

    lda_report = {
        "mean_cross_validated_accuracy": float(np.mean(lda_scores)),
        "cv_accuracy_std": float(np.std(lda_scores)),
        "qda_cross_validated_accuracy": float(np.mean(qda_scores)),
        "qda_accuracy_std": float(np.std(qda_scores)),
        "explained_variance_ratio": lda.explained_variance_ratio_.tolist(),
        "baseline_chance_accuracy": 0.3333333333333333,
    }

    # Plot genuine LDA domain separation
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    for idx, name in enumerate(domain_names):
        mask = (all_y == idx)
        ax.scatter(
            lda_coords[mask, 0],
            lda_coords[mask, 1],
            c=colors[idx],
            marker=markers[idx],
            alpha=0.45,
            edgecolors="none",
            label=f"{name.capitalize()} (n={np.sum(mask)})",
        )
        centroid = np.mean(lda_coords[mask], axis=0)
        ax.scatter(centroid[0], centroid[1], c=colors[idx], marker="X", s=130, edgecolors="black", linewidth=1.5)

    ax.set_title("Genuine Cross-Domain LDA Separation in Latent FFN Activation Space (Layer 00)")
    ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}% between-domain variance)")
    ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}% between-domain variance)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    fig.savefig(ASSETS_DIR / "rq5_lda.png", dpi=180)
    fig.savefig(REPORT_FIGS_DIR / "fig5_lda_domain_separation.png", dpi=180)
    plt.close(fig)

    # Save real JSON report
    rq5_report_content = {
        "phase": "rq5_domain_stability",
        "description": "Evaluation of mean vector and covariance stability across GENUINE text domains (Wikipedia, ML-ArXiv, AG-News).",
        "target_layer": "layer_00",
        "is_real_data": True,
        "n_samples_per_domain": {name: int(len(reduced_domain_data[name])) for name in domain_names},
        "pca_reduced_components": 8,
        "manova": manova_res,
        "pairwise_hotelling_t2": pairwise_results,
        "box_m_test": box_m_res,
        "lda_classification": lda_report,
    }
    (REPORTS_DIR / "rq5_domain_stability_report.json").write_text(json.dumps(rq5_report_content, indent=2), encoding="utf-8")
    logger.info("Real RQ5 completed successfully! Wilks Lambda=%.5f, Rao F=%.2f, LDA Acc=%.3f, QDA Acc=%.3f",
                manova_res["wilks_lambda"], manova_res["rao_f_statistic"], lda_report["mean_cross_validated_accuracy"], lda_report["qda_cross_validated_accuracy"])
    return rq5_report_content


# ==============================================================================
# Main Execution Flow
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model_id = "facebook/opt-125m"
    logger.info("Loading model %s...", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 1. Real RQ4
    rq4_report_path = REPORTS_DIR / "rq4_weight_rank_report.json"
    if rq4_report_path.is_file() and json.loads(rq4_report_path.read_text(encoding="utf-8")).get("correlation_metrics", {}).get("is_real_weights"):
        logger.info("Real RQ4 report already generated. Skipping re-computation.")
        rq4_metrics = json.loads(rq4_report_path.read_text(encoding="utf-8"))["correlation_metrics"]
    else:
        w_summary, w_spectra, rq4_metrics = run_real_rq4(model)

    # 2. Real RQ5
    rq5_metrics = run_real_rq5(model, tokenizer, device)

    print("\n" + "="*80)
    print("REAL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("RQ4 Metrics:", json.dumps(rq4_metrics, indent=2))
    print("RQ5 Metrics Summary:")
    print(f"  Wilks Lambda: {rq5_metrics['manova']['wilks_lambda']:.6e}")
    print(f"  Rao F: {rq5_metrics['manova']['rao_f_statistic']:.2f} (df1={rq5_metrics['manova']['df1']}, df2={rq5_metrics['manova']['df2']})")
    print(f"  Box M Chi2: {rq5_metrics['box_m_test']['chi2_statistic']:.2f} (df={rq5_metrics['box_m_test']['degrees_of_freedom']})")
    print(f"  LDA Accuracy: {rq5_metrics['lda_classification']['mean_cross_validated_accuracy']*100:.2f}%")
    print(f"  QDA Accuracy: {rq5_metrics['lda_classification']['qda_cross_validated_accuracy']*100:.2f}%")
    print("="*80)
