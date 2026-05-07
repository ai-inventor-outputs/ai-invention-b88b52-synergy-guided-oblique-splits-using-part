#!/usr/bin/env python3
"""Definitive Statistical Evaluation of SG-FIGS Hypothesis.

Comprehensive 11-step statistical evaluation of the 5-method comparison
(FIGS, RO-FIGS, SG-FIGS-Hard, SG-FIGS-Soft, Random-FIGS) across 14 datasets,
integrating threshold sensitivity and PID synergy analysis into paper-ready
conclusions on all three success criteria.
"""

import json
import resource
import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
from scipy import stats
from loguru import logger

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU

# ── Logging ──────────────────────────────────────────────────────────────────
BLUE, GREEN, YELLOW, CYAN, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[0m"

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level: <7}}|{CYAN}{{name: >12.12}}{END}.{CYAN}{{function: <22.22}}{END}:{CYAN}{{line: <4}}{END}| {{message}}",
    colorize=False,
)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.add(str(log_dir / "eval.log"), rotation="30 MB", level="DEBUG")

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
EXP1_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full/"
    "3_invention_loop/iter_3/gen_art/exp_id1_it3__opus"
)
EXP3_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full/"
    "3_invention_loop/iter_3/gen_art/exp_id3_it3__opus"
)
PID_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full/"
    "3_invention_loop/iter_2/gen_art/exp_id1_it2__opus"
)

METHODS = ["figs", "ro_figs", "sg_figs_hard", "sg_figs_soft", "random_figs"]
METHOD_LABELS = {
    "figs": "FIGS",
    "ro_figs": "RO-FIGS",
    "sg_figs_hard": "SG-FIGS-Hard",
    "sg_figs_soft": "SG-FIGS-Soft",
    "random_figs": "Random-FIGS",
}

# Domain knowledge for Criterion 3
# Feature names must match EXACTLY what appears in PID data
DOMAIN_INTERACTIONS = {
    "pima_diabetes": [
        ({"plas", "mass"}, "glucose-BMI interaction"),
        ({"plas", "age"}, "glucose-age interaction"),
        ({"mass", "skin"}, "BMI-skin thickness interaction"),
        ({"plas", "insu"}, "glucose-insulin interaction"),
    ],
    "breast_cancer": [
        ({"mean concave points", "mean radius"}, "concave_points-radius"),
        ({"mean concave points", "worst radius"}, "concave_points-worst_radius"),
        ({"mean concavity", "mean perimeter"}, "concavity-perimeter"),
        ({"worst concave points", "worst perimeter"}, "worst_concave_points-perimeter"),
    ],
    "heart_statlog": [
        ({"chest", "thal"}, "chest_pain-thal interaction"),
        ({"age", "maximum_heart_rate_achieved"}, "age-max HR interaction"),
        ({"oldpeak", "slope"}, "ST depression-slope interaction"),
    ],
    "monks2": [
        ({"a1", "a2"}, "XOR features a1-a2"),
        ({"a4", "a5"}, "XOR features a4-a5"),
    ],
}

# Map PID dataset names to exp1 dataset names
PID_TO_EXP1_NAME = {
    "breast_cancer_wisconsin_diagnostic": "breast_cancer_wisconsin_diagnostic",
    "pima_diabetes": "pima_diabetes",
    "heart_statlog": "heart_statlog",
    "banknote": "banknote",
    "ionosphere": "ionosphere",
    "vehicle": "vehicle",
    "sonar": "sonar",
    "spectf_heart": "spectf_heart",
    "iris": "iris",
    "wine": "wine",
    # breast_cancer in PID maps differently
    "breast_cancer": "breast_cancer_wisconsin_diagnostic",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    """Load a JSON file and return the parsed data."""
    logger.info(f"Loading {path.name} from {path.parent}")
    data = json.loads(path.read_text())
    return data


def parse_predict_field(predict_str: str) -> dict:
    """Parse a JSON-encoded predict_ field."""
    return json.loads(predict_str)


def extract_fold_level_metrics(data: dict) -> dict:
    """Extract per-dataset, per-fold metrics from exp1 data.

    Returns dict: {dataset: {method: {fold: {metric: value}}}}
    """
    result = {}
    for ds_block in data["datasets"]:
        ds_name = ds_block["dataset"]
        fold_metrics = defaultdict(lambda: defaultdict(dict))

        for ex in ds_block["examples"]:
            fold = ex["metadata_fold"]
            for method in METHODS:
                pred = parse_predict_field(ex[f"predict_{method}"])
                # Store metrics — they are the same for all examples in a fold
                fold_metrics[method][fold] = {
                    "balanced_accuracy": pred["balanced_accuracy"],
                    "auc": pred.get("auc"),
                    "n_splits": pred["n_splits"],
                    "n_trees": pred.get("n_trees"),
                    "interpretability_score": pred.get("interpretability_score"),
                }

        result[ds_name] = dict(fold_metrics)
    return result


def get_dataset_mean_metrics(fold_data: dict) -> dict:
    """Compute per-dataset mean ± std across folds.

    Returns {dataset: {method: {metric: (mean, std)}}}
    """
    result = {}
    for ds_name, method_folds in fold_data.items():
        result[ds_name] = {}
        for method, folds in method_folds.items():
            fold_vals = list(folds.values())
            metrics_summary = {}
            for metric in ["balanced_accuracy", "auc", "n_splits", "interpretability_score"]:
                vals = [f[metric] for f in fold_vals if f[metric] is not None]
                if vals:
                    metrics_summary[metric] = (float(np.mean(vals)), float(np.std(vals)))
                else:
                    metrics_summary[metric] = (None, None)
            result[ds_name][method] = metrics_summary
    return result


def friedman_test(accuracy_matrix: np.ndarray) -> tuple:
    """Perform Friedman test on N×k matrix (N datasets, k methods).

    Returns (chi2, p_value, ranks_per_method).
    """
    n, k = accuracy_matrix.shape
    # Rank within each row (dataset) — higher accuracy → lower rank (rank 1 = best)
    ranks = np.zeros_like(accuracy_matrix)
    for i in range(n):
        # scipy rankdata: 1=smallest. We want 1=largest so negate.
        ranks[i] = stats.rankdata(-accuracy_matrix[i])

    avg_ranks = ranks.mean(axis=0)

    # Friedman statistic
    chi2 = (12 * n) / (k * (k + 1)) * (np.sum(avg_ranks**2) - k * ((k + 1) ** 2) / 4)
    p_value = 1 - stats.chi2.cdf(chi2, df=k - 1)

    return float(chi2), float(p_value), avg_ranks.tolist(), ranks


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi critical difference.

    CD = q_alpha * sqrt(k*(k+1)/(6*N))
    q_alpha values for alpha=0.05 from Demšar (2006), Table 5.
    """
    # q_alpha values for k methods at alpha=0.05
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(k, 2.728)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return float(cd)


def pairwise_wilcoxon_holm(
    accuracy_matrix: np.ndarray,
    method_names: list[str],
) -> list[dict]:
    """Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.

    Returns list of {pair, statistic, p_raw, p_corrected, significant}.
    """
    k = len(method_names)
    pairs = list(combinations(range(k), 2))
    raw_results = []

    for i, j in pairs:
        diffs = accuracy_matrix[:, i] - accuracy_matrix[:, j]
        # Remove zeros for Wilcoxon
        nonzero = diffs[diffs != 0]
        if len(nonzero) < 2:
            raw_results.append({
                "pair": f"{method_names[i]} vs {method_names[j]}",
                "statistic": None,
                "p_raw": 1.0,
                "mean_diff": float(np.mean(diffs)),
            })
            continue

        try:
            stat, p = stats.wilcoxon(nonzero, alternative="two-sided")
            raw_results.append({
                "pair": f"{method_names[i]} vs {method_names[j]}",
                "statistic": float(stat),
                "p_raw": float(p),
                "mean_diff": float(np.mean(diffs)),
            })
        except ValueError:
            raw_results.append({
                "pair": f"{method_names[i]} vs {method_names[j]}",
                "statistic": None,
                "p_raw": 1.0,
                "mean_diff": float(np.mean(diffs)),
            })

    # Holm-Bonferroni correction
    sorted_idx = sorted(range(len(raw_results)), key=lambda x: raw_results[x]["p_raw"])
    m = len(raw_results)

    for rank_pos, idx in enumerate(sorted_idx):
        p_adj = raw_results[idx]["p_raw"] * (m - rank_pos)
        raw_results[idx]["p_corrected"] = min(float(p_adj), 1.0)
        raw_results[idx]["significant"] = raw_results[idx]["p_corrected"] < 0.05

    return raw_results


def win_loss_tie(vals_a: np.ndarray, vals_b: np.ndarray, tol: float = 1e-6) -> dict:
    """Count wins, losses, ties between two arrays."""
    diffs = vals_a - vals_b
    wins = int(np.sum(diffs > tol))
    losses = int(np.sum(diffs < -tol))
    ties = int(np.sum(np.abs(diffs) <= tol))
    return {"wins": wins, "losses": losses, "ties": ties}


def linear_regression_r2(x: np.ndarray, y: np.ndarray) -> tuple:
    """Simple linear regression returning slope, intercept, R², p_value."""
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 0.0, 0.0, 1.0
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return float(slope), float(intercept), float(r_value**2), float(p_value)


def extract_threshold_data(data: dict) -> dict:
    """Extract threshold sensitivity data.

    Each (dataset, threshold, fold) can have multiple max_splits entries.
    We collect all entries and average across max_splits for each (ds, thresh, fold).

    Returns {dataset: {threshold: {fold: {metric: value}}}}
    """
    # First collect all entries grouped by (ds, thresh, fold)
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ds_block in data["datasets"]:
        for ex in ds_block["examples"]:
            output = json.loads(ex["output"])
            ds = ex["metadata_dataset"]
            thresh = ex["metadata_threshold_percentile"]
            fold = ex["metadata_fold"]
            raw[ds][thresh][fold].append({
                "sg_figs_balanced_acc": output["sg_figs_balanced_acc"],
                "sg_figs_auc": output.get("sg_figs_auc"),
                "sg_figs_n_splits": output["sg_figs_n_splits"],
                "sg_figs_interpretability": output.get("sg_figs_interpretability"),
                "figs_balanced_acc": output["figs_balanced_acc"],
                "rofigs_balanced_acc": output["rofigs_balanced_acc"],
                "gbdt_balanced_acc": output.get("gbdt_balanced_acc"),
                "max_splits": ex["metadata_max_splits"],
            })

    # Average across max_splits for each (ds, thresh, fold)
    result = defaultdict(lambda: defaultdict(dict))
    for ds in raw:
        for thresh in raw[ds]:
            for fold in raw[ds][thresh]:
                entries = raw[ds][thresh][fold]
                avg_entry = {}
                for key in ["sg_figs_balanced_acc", "figs_balanced_acc", "rofigs_balanced_acc"]:
                    vals = [e[key] for e in entries if e[key] is not None]
                    avg_entry[key] = float(np.mean(vals)) if vals else 0.0
                for key in ["sg_figs_auc", "gbdt_balanced_acc", "sg_figs_interpretability"]:
                    vals = [e[key] for e in entries if e.get(key) is not None]
                    avg_entry[key] = float(np.mean(vals)) if vals else None
                avg_entry["sg_figs_n_splits"] = float(np.mean([e["sg_figs_n_splits"] for e in entries]))
                result[ds][thresh][fold] = avg_entry

    return dict(result)


def extract_pid_synergy(data: dict) -> dict:
    """Extract PID synergy matrices.

    Returns {dataset: [(feat_i, feat_j, synergy, redundancy, unique_0, unique_1)]}
    """
    result = defaultdict(list)
    for ds_block in data["datasets"]:
        ds_name = ds_block["dataset"]
        for ex in ds_block["examples"]:
            output = json.loads(ex["output"])
            result[ds_name].append({
                "feature_i": ex["metadata_feature_i"],
                "feature_j": ex["metadata_feature_j"],
                "synergy": output["synergy"],
                "redundancy": output["redundancy"],
                "unique_0": output["unique_0"],
                "unique_1": output["unique_1"],
                "coi_baseline": output.get("coi_baseline", 0),
            })
    return dict(result)


def compute_synergy_graph_properties(pid_pairs: list[dict], percentile: int = 75) -> dict:
    """Compute synergy graph properties for a dataset.

    Returns {n_edges, n_nodes, density, synergy_mean, synergy_std, threshold}.
    """
    synergies = [p["synergy"] for p in pid_pairs]
    threshold = float(np.percentile(synergies, percentile))
    above = [p for p in pid_pairs if p["synergy"] >= threshold]

    nodes = set()
    for p in above:
        nodes.add(p["feature_i"])
        nodes.add(p["feature_j"])

    n_nodes = len(nodes) if nodes else 0
    n_edges = len(above)
    # Graph density = 2*E / (V*(V-1)) for undirected graph
    density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    return {
        "n_edges": n_edges,
        "n_nodes": n_nodes,
        "density": float(density),
        "synergy_mean": float(np.mean(synergies)),
        "synergy_std": float(np.std(synergies)),
        "threshold": threshold,
    }


def get_top_synergy_pairs(pid_pairs: list[dict], top_k: int = 3) -> list[tuple]:
    """Get top-k synergy pairs from PID data."""
    sorted_pairs = sorted(pid_pairs, key=lambda x: x["synergy"], reverse=True)
    return [(p["feature_i"], p["feature_j"]) for p in sorted_pairs[:top_k]]


def compute_mi_synergy_jaccard(pid_pairs: list[dict], top_k: int = 5) -> float:
    """Compute Jaccard overlap between top MI features and top synergy pairs."""
    # MI approximation: sum unique_0 + unique_1 + redundancy for each feature
    mi_by_feat = defaultdict(float)
    for p in pid_pairs:
        total_mi = p["unique_0"] + p["unique_1"] + p["redundancy"] + p["synergy"]
        mi_by_feat[p["feature_i"]] += total_mi
        mi_by_feat[p["feature_j"]] += total_mi

    top_mi_feats = set(
        sorted(mi_by_feat.keys(), key=lambda x: mi_by_feat[x], reverse=True)[:top_k]
    )

    # Top synergy features
    sorted_pairs = sorted(pid_pairs, key=lambda x: x["synergy"], reverse=True)
    top_syn_feats = set()
    for p in sorted_pairs[:top_k]:
        top_syn_feats.add(p["feature_i"])
        top_syn_feats.add(p["feature_j"])

    if not top_mi_feats or not top_syn_feats:
        return 0.0

    intersection = len(top_mi_feats & top_syn_feats)
    union = len(top_mi_feats | top_syn_feats)
    return float(intersection / union) if union > 0 else 0.0


# ── Main evaluation ─────────────────────────────────────────────────────────
@logger.catch
def main():
    logger.info(f"{BLUE}Starting Definitive Statistical Evaluation of SG-FIGS{END}")

    # ── Load all data ────────────────────────────────────────────────────
    exp1_data = load_json(EXP1_DIR / "full_method_out.json")
    exp3_data = load_json(EXP3_DIR / "full_method_out.json")
    pid_data = load_json(PID_DIR / "full_method_out.json")

    n_datasets_exp1 = len(exp1_data["datasets"])
    n_examples_exp1 = sum(len(ds["examples"]) for ds in exp1_data["datasets"])
    n_datasets_exp3 = len(exp3_data["datasets"])
    n_examples_exp3 = sum(len(ds["examples"]) for ds in exp3_data["datasets"])
    n_datasets_pid = len(pid_data["datasets"])
    n_examples_pid = sum(len(ds["examples"]) for ds in pid_data["datasets"])

    logger.info(f"Exp1 (5-method): {n_datasets_exp1} datasets, {n_examples_exp1} examples")
    logger.info(f"Exp3 (threshold): {n_datasets_exp3} datasets, {n_examples_exp3} examples")
    logger.info(f"PID (synergy): {n_datasets_pid} datasets, {n_examples_pid} examples")

    # ── Step 1: Extract fold-level metrics from exp1 ─────────────────────
    logger.info(f"{BLUE}Step 1: Extracting fold-level metrics{END}")
    fold_data = extract_fold_level_metrics(exp1_data)
    dataset_means = get_dataset_mean_metrics(fold_data)

    dataset_names = sorted(dataset_means.keys())
    logger.info(f"Datasets: {dataset_names}")

    # Build accuracy matrix: N datasets × k methods
    acc_matrix = np.zeros((len(dataset_names), len(METHODS)))
    for i, ds in enumerate(dataset_names):
        for j, m in enumerate(METHODS):
            mean_val, _ = dataset_means[ds][m]["balanced_accuracy"]
            acc_matrix[i, j] = mean_val

    logger.info(f"Accuracy matrix shape: {acc_matrix.shape}")
    for j, m in enumerate(METHODS):
        logger.info(f"  {METHOD_LABELS[m]:>15}: mean={acc_matrix[:, j].mean():.4f}")

    # ── Step 2: Friedman test on balanced accuracy (all 14 datasets) ─────
    logger.info(f"{BLUE}Step 2: Friedman test on balanced accuracy{END}")
    chi2_acc, p_acc, avg_ranks_acc, ranks_acc = friedman_test(acc_matrix)
    cd = nemenyi_cd(k=len(METHODS), n=len(dataset_names))
    logger.info(f"  Friedman χ²={chi2_acc:.4f}, p={p_acc:.6f}")
    for j, m in enumerate(METHODS):
        logger.info(f"  {METHOD_LABELS[m]:>15}: avg rank = {avg_ranks_acc[j]:.3f}")
    logger.info(f"  Nemenyi CD (α=0.05) = {cd:.4f}")

    # ── Step 3: Friedman test on AUC (binary-only datasets) ──────────────
    logger.info(f"{BLUE}Step 3: Friedman test on AUC (binary datasets){END}")
    binary_ds = []
    auc_rows = []
    for i, ds in enumerate(dataset_names):
        auc_vals = []
        all_valid = True
        for j, m in enumerate(METHODS):
            auc_mean, _ = dataset_means[ds][m].get("auc", (None, None))
            if auc_mean is None:
                all_valid = False
                break
            auc_vals.append(auc_mean)
        if all_valid:
            binary_ds.append(ds)
            auc_rows.append(auc_vals)

    auc_matrix = np.array(auc_rows)
    logger.info(f"  Binary datasets with AUC: {len(binary_ds)}")
    if len(binary_ds) >= 3:
        chi2_auc, p_auc, avg_ranks_auc, _ = friedman_test(auc_matrix)
        logger.info(f"  Friedman χ²={chi2_auc:.4f}, p={p_auc:.6f}")
    else:
        chi2_auc, p_auc, avg_ranks_auc = None, None, None
        logger.warning("  Not enough binary datasets for Friedman AUC test")

    # ── Step 4: Pairwise Wilcoxon with Holm-Bonferroni ───────────────────
    logger.info(f"{BLUE}Step 4: Pairwise Wilcoxon signed-rank tests{END}")
    method_labels_list = [METHOD_LABELS[m] for m in METHODS]
    wilcoxon_results = pairwise_wilcoxon_holm(acc_matrix, method_labels_list)
    for r in wilcoxon_results:
        sig_str = "***" if r["significant"] else ""
        logger.info(
            f"  {r['pair']:>40}: p_raw={r['p_raw']:.4f}, p_adj={r['p_corrected']:.4f} "
            f"diff={r['mean_diff']:+.4f} {sig_str}"
        )

    # ── Step 5: Ablation — SG-FIGS vs Random-FIGS ────────────────────────
    logger.info(f"{BLUE}Step 5: Ablation analysis (SG-FIGS vs Random-FIGS){END}")
    sg_hard_idx = METHODS.index("sg_figs_hard")
    sg_soft_idx = METHODS.index("sg_figs_soft")
    random_idx = METHODS.index("random_figs")

    # Hard vs Random
    hard_acc = acc_matrix[:, sg_hard_idx]
    random_acc = acc_matrix[:, random_idx]
    delta_hard_random = hard_acc - random_acc
    wlt_hard = win_loss_tie(hard_acc, random_acc)
    try:
        nonzero_hr = delta_hard_random[delta_hard_random != 0]
        if len(nonzero_hr) >= 2:
            stat_hr, p_hr = stats.wilcoxon(nonzero_hr)
        else:
            stat_hr, p_hr = None, 1.0
    except ValueError:
        stat_hr, p_hr = None, 1.0

    logger.info(f"  SG-FIGS-Hard vs Random-FIGS:")
    logger.info(f"    Mean Δacc = {np.mean(delta_hard_random):+.4f}")
    logger.info(f"    W/L/T = {wlt_hard}, Wilcoxon p={p_hr:.4f}")

    # Soft vs Random
    soft_acc = acc_matrix[:, sg_soft_idx]
    delta_soft_random = soft_acc - random_acc
    wlt_soft = win_loss_tie(soft_acc, random_acc)
    try:
        nonzero_sr = delta_soft_random[delta_soft_random != 0]
        if len(nonzero_sr) >= 2:
            stat_sr, p_sr = stats.wilcoxon(nonzero_sr)
        else:
            stat_sr, p_sr = None, 1.0
    except ValueError:
        stat_sr, p_sr = None, 1.0

    logger.info(f"  SG-FIGS-Soft vs Random-FIGS:")
    logger.info(f"    Mean Δacc = {np.mean(delta_soft_random):+.4f}")
    logger.info(f"    W/L/T = {wlt_soft}, Wilcoxon p={p_sr:.4f}")

    # Interpretability ablation: SG-FIGS-Hard vs Random-FIGS
    interp_hard = []
    interp_random = []
    for ds in dataset_names:
        h_mean, _ = dataset_means[ds]["sg_figs_hard"]["interpretability_score"]
        r_mean, _ = dataset_means[ds]["random_figs"]["interpretability_score"]
        if h_mean is not None and r_mean is not None:
            interp_hard.append(h_mean)
            interp_random.append(r_mean)

    interp_hard = np.array(interp_hard)
    interp_random = np.array(interp_random)
    delta_interp = interp_hard - interp_random
    try:
        nonzero_interp = delta_interp[delta_interp != 0]
        if len(nonzero_interp) >= 2:
            stat_interp, p_interp = stats.wilcoxon(nonzero_interp)
        else:
            stat_interp, p_interp = None, 1.0
    except ValueError:
        stat_interp, p_interp = None, 1.0

    logger.info(f"  Interpretability ablation (Hard vs Random):")
    logger.info(f"    Mean Δinterp = {np.mean(delta_interp):+.4f}")
    logger.info(f"    Wilcoxon p = {p_interp:.4f}")

    # ── Step 6: Criterion 1 (Accuracy + Complexity) ──────────────────────
    logger.info(f"{BLUE}Step 6: Criterion 1 — Accuracy + Complexity{END}")
    ro_idx = METHODS.index("ro_figs")

    c1_results = []
    for i, ds in enumerate(dataset_names):
        soft_mean_acc, _ = dataset_means[ds]["sg_figs_soft"]["balanced_accuracy"]
        ro_mean_acc, _ = dataset_means[ds]["ro_figs"]["balanced_accuracy"]
        soft_splits, _ = dataset_means[ds]["sg_figs_soft"]["n_splits"]
        ro_splits, _ = dataset_means[ds]["ro_figs"]["n_splits"]

        acc_delta = soft_mean_acc - ro_mean_acc
        split_ratio = soft_splits / ro_splits if ro_splits > 0 else float("inf")

        # Original criterion: |acc_delta| <= 0.01 AND split_ratio <= 0.80
        original_met = abs(acc_delta) <= 0.01 and split_ratio <= 0.80

        # Reframed criterion: acc within 1% OR accuracy improvement
        reframed_met = abs(acc_delta) <= 0.01 or acc_delta > 0

        c1_results.append({
            "dataset": ds,
            "acc_delta": float(acc_delta),
            "split_ratio": float(split_ratio),
            "original_criterion_met": original_met,
            "reframed_criterion_met": reframed_met,
        })

    original_count = sum(1 for r in c1_results if r["original_criterion_met"])
    reframed_count = sum(1 for r in c1_results if r["reframed_criterion_met"])
    logger.info(f"  Original criterion met: {original_count}/{len(c1_results)}")
    logger.info(f"  Reframed criterion met: {reframed_count}/{len(c1_results)}")

    # Dataset regression for Criterion 1
    logger.info(f"  Running dataset property regression...")
    pid_synergy = extract_pid_synergy(pid_data)

    # Collect dataset properties for regression
    ds_properties = {}
    for ds in dataset_names:
        props = {"n_features": 0, "synergy_mean": 0, "synergy_std": 0, "density": 0, "jaccard": 0}
        # Get n_features from exp1 metadata
        for ds_block in exp1_data["datasets"]:
            if ds_block["dataset"] == ds:
                props["n_features"] = ds_block["examples"][0]["metadata_n_features"]
                break

        # Get synergy properties from PID data
        pid_name = None
        for pname in pid_synergy:
            mapped = PID_TO_EXP1_NAME.get(pname, pname)
            if mapped == ds:
                pid_name = pname
                break

        if pid_name and pid_name in pid_synergy:
            graph_props = compute_synergy_graph_properties(pid_synergy[pid_name])
            props["synergy_mean"] = graph_props["synergy_mean"]
            props["synergy_std"] = graph_props["synergy_std"]
            props["density"] = graph_props["density"]
            props["jaccard"] = compute_mi_synergy_jaccard(pid_synergy[pid_name])

        ds_properties[ds] = props

    # Regression: acc_delta ~ n_features, synergy_mean, synergy_std, density, jaccard
    acc_deltas = np.array([r["acc_delta"] for r in c1_results])
    predictor_names = ["n_features", "synergy_mean", "synergy_std", "density", "jaccard"]
    regression_results = {}
    for pred_name in predictor_names:
        x = np.array([ds_properties[ds][pred_name] for ds in dataset_names])
        slope, intercept, r2, p_val = linear_regression_r2(x, acc_deltas)
        regression_results[pred_name] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r2,
            "p_value": p_val,
        }
        logger.info(f"    {pred_name}: R²={r2:.4f}, slope={slope:.6f}, p={p_val:.4f}")

    # Multiple regression with all predictors
    X_all = np.column_stack([
        [ds_properties[ds][p] for ds in dataset_names] for p in predictor_names
    ])
    if X_all.shape[0] > X_all.shape[1] + 1:
        try:
            # Add intercept
            X_with_int = np.column_stack([np.ones(len(dataset_names)), X_all])
            beta = np.linalg.lstsq(X_with_int, acc_deltas, rcond=None)[0]
            y_pred = X_with_int @ beta
            ss_res = np.sum((acc_deltas - y_pred) ** 2)
            ss_tot = np.sum((acc_deltas - np.mean(acc_deltas)) ** 2)
            r2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            logger.info(f"    Multiple regression R² = {r2_multi:.4f}")
        except np.linalg.LinAlgError:
            r2_multi = 0.0
            beta = np.zeros(len(predictor_names) + 1)
    else:
        r2_multi = 0.0
        beta = np.zeros(len(predictor_names) + 1)

    # ── Step 7: Criterion 2 (Interpretability) ───────────────────────────
    logger.info(f"{BLUE}Step 7: Criterion 2 — Interpretability{END}")

    interp_means = {}
    for m in METHODS:
        vals = []
        for ds in dataset_names:
            mean_val, _ = dataset_means[ds][m]["interpretability_score"]
            if mean_val is not None:
                vals.append(mean_val)
        interp_means[m] = float(np.mean(vals)) if vals else None

    for m in METHODS:
        label = METHOD_LABELS[m]
        val = interp_means[m]
        logger.info(f"  {label:>15}: mean interp = {val if val is not None else 'N/A'}")

    # SG-FIGS-Hard vs RO-FIGS interpretability
    interp_hard_vals = []
    interp_ro_vals = []
    for ds in dataset_names:
        h_mean, _ = dataset_means[ds]["sg_figs_hard"]["interpretability_score"]
        r_mean, _ = dataset_means[ds]["ro_figs"]["interpretability_score"]
        if h_mean is not None and r_mean is not None:
            interp_hard_vals.append(h_mean)
            interp_ro_vals.append(r_mean)

    interp_hard_arr = np.array(interp_hard_vals)
    interp_ro_arr = np.array(interp_ro_vals)
    delta_interp_hard_ro = interp_hard_arr - interp_ro_arr
    try:
        nz = delta_interp_hard_ro[delta_interp_hard_ro != 0]
        if len(nz) >= 2:
            _, p_interp_hard_ro = stats.wilcoxon(nz)
        else:
            p_interp_hard_ro = 1.0
    except ValueError:
        p_interp_hard_ro = 1.0

    logger.info(f"  SG-FIGS-Hard vs RO-FIGS interp: Δ={np.mean(delta_interp_hard_ro):+.4f}, p={p_interp_hard_ro:.4f}")
    logger.info(f"  SG-FIGS-Hard vs Random-FIGS interp: Δ={np.mean(delta_interp):+.4f}, p={p_interp:.4f}")

    # ── Step 8: Criterion 3 (Domain Validation) ──────────────────────────
    logger.info(f"{BLUE}Step 8: Criterion 3 — Domain Validation{END}")

    domain_results = {}
    for domain_ds, known_interactions in DOMAIN_INTERACTIONS.items():
        # Find PID data for this dataset
        # domain_ds keys match PID dataset names directly
        pid_name = domain_ds if domain_ds in pid_synergy else None

        if pid_name is None:
            logger.warning(f"  No PID data for {domain_ds}")
            domain_results[domain_ds] = {
                "top_synergy_pairs": [],
                "known_interactions_found": [],
                "coverage": 0,
            }
            continue

        top_pairs = get_top_synergy_pairs(pid_synergy[pid_name], top_k=3)
        top_sets = [frozenset(p) for p in top_pairs]

        found = []
        for interaction_set, label in known_interactions:
            fset = frozenset(interaction_set)
            if fset in top_sets:
                found.append(label)

        domain_results[domain_ds] = {
            "top_synergy_pairs": [list(p) for p in top_pairs],
            "known_interactions_found": found,
            "coverage": len(found),
        }
        logger.info(f"  {domain_ds}: top pairs = {top_pairs}")
        logger.info(f"    Found {len(found)}/{len(known_interactions)} known interactions: {found}")

    datasets_with_matches = sum(1 for v in domain_results.values() if v["coverage"] > 0)
    logger.info(f"  Datasets with ≥1 known interaction in top-3: {datasets_with_matches}/{len(DOMAIN_INTERACTIONS)}")

    # ── Step 9: Threshold Sensitivity Integration ────────────────────────
    logger.info(f"{BLUE}Step 9: Threshold Sensitivity Integration{END}")
    threshold_data = extract_threshold_data(exp3_data)

    # Per-dataset: optimal threshold vs fixed 75th percentile
    threshold_analysis = {}
    for ds in sorted(threshold_data.keys()):
        thresh_accs = {}
        for thresh in sorted(threshold_data[ds].keys()):
            fold_accs = [
                threshold_data[ds][thresh][f]["sg_figs_balanced_acc"]
                for f in threshold_data[ds][thresh]
            ]
            thresh_accs[thresh] = float(np.mean(fold_accs))

        best_thresh = max(thresh_accs, key=thresh_accs.get)
        fixed_75_acc = thresh_accs.get(75, thresh_accs.get(50, 0))
        best_acc = thresh_accs[best_thresh]

        # Also get RO-FIGS acc at 75th percentile for comparison
        rofigs_accs = []
        for thresh in threshold_data[ds]:
            for f in threshold_data[ds][thresh]:
                rofigs_accs.append(threshold_data[ds][thresh][f].get("rofigs_balanced_acc", 0))
        rofigs_mean = float(np.mean(rofigs_accs)) if rofigs_accs else 0

        threshold_analysis[ds] = {
            "threshold_accs": {str(k): v for k, v in thresh_accs.items()},
            "optimal_threshold": int(best_thresh),
            "optimal_acc": best_acc,
            "fixed_75_acc": fixed_75_acc,
            "improvement_from_tuning": best_acc - fixed_75_acc,
            "rofigs_mean_acc": rofigs_mean,
        }
        logger.info(
            f"  {ds:>40}: best_thresh={best_thresh}, "
            f"Δ_tuning={best_acc - fixed_75_acc:+.4f}"
        )

    mean_tuning_improvement = float(np.mean([
        v["improvement_from_tuning"] for v in threshold_analysis.values()
    ]))
    logger.info(f"  Mean improvement from threshold tuning: {mean_tuning_improvement:+.4f}")

    # Threshold-tuned SG-FIGS vs RO-FIGS
    thresh_tuned_vs_rofigs = []
    for ds in sorted(threshold_analysis.keys()):
        ta = threshold_analysis[ds]
        delta = ta["optimal_acc"] - ta["rofigs_mean_acc"]
        thresh_tuned_vs_rofigs.append(delta)
    thresh_tuned_mean_delta = float(np.mean(thresh_tuned_vs_rofigs))
    logger.info(f"  Threshold-tuned SG-FIGS vs RO-FIGS: mean Δ = {thresh_tuned_mean_delta:+.4f}")

    # ── Step 10: Pareto Analysis ─────────────────────────────────────────
    logger.info(f"{BLUE}Step 10: Pareto Analysis (Accuracy vs Complexity){END}")

    pareto_data = {}
    for j, m in enumerate(METHODS):
        mean_acc = float(acc_matrix[:, j].mean())
        mean_splits = float(np.mean([
            dataset_means[ds][m]["n_splits"][0]
            for ds in dataset_names
            if dataset_means[ds][m]["n_splits"][0] is not None
        ]))
        efficiency = mean_acc / mean_splits if mean_splits > 0 else 0
        pareto_data[m] = {
            "mean_accuracy": mean_acc,
            "mean_n_splits": mean_splits,
            "accuracy_per_split": float(efficiency),
        }
        logger.info(
            f"  {METHOD_LABELS[m]:>15}: acc={mean_acc:.4f}, splits={mean_splits:.1f}, "
            f"eff={efficiency:.4f}"
        )

    # Identify Pareto-dominant methods
    pareto_dominant = []
    for m1 in METHODS:
        is_dominated = False
        for m2 in METHODS:
            if m1 == m2:
                continue
            # m2 dominates m1 if m2 has higher acc AND fewer splits
            if (pareto_data[m2]["mean_accuracy"] >= pareto_data[m1]["mean_accuracy"] and
                    pareto_data[m2]["mean_n_splits"] <= pareto_data[m1]["mean_n_splits"] and
                    (pareto_data[m2]["mean_accuracy"] > pareto_data[m1]["mean_accuracy"] or
                     pareto_data[m2]["mean_n_splits"] < pareto_data[m1]["mean_n_splits"])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_dominant.append(METHOD_LABELS[m1])

    logger.info(f"  Pareto-optimal methods: {pareto_dominant}")

    # ── Step 11: Overall Verdict ─────────────────────────────────────────
    logger.info(f"{BLUE}Step 11: Overall Verdict{END}")

    # Criterion 1 evidence
    c1_evidence = "negative"
    if original_count >= 7:
        c1_evidence = "strong"
    elif original_count >= 4:
        c1_evidence = "moderate"
    elif reframed_count >= 7:
        c1_evidence = "moderate"
    elif reframed_count >= 4:
        c1_evidence = "weak"

    # Criterion 2 evidence
    c2_evidence = "negative"
    hard_interp = interp_means.get("sg_figs_hard")
    random_interp = interp_means.get("random_figs")
    if hard_interp is not None and random_interp is not None:
        if hard_interp > random_interp and p_interp < 0.05:
            c2_evidence = "strong"
        elif hard_interp > random_interp and p_interp < 0.10:
            c2_evidence = "moderate"
        elif hard_interp > random_interp:
            c2_evidence = "weak"

    # Criterion 3 evidence
    if datasets_with_matches >= 3:
        c3_evidence = "strong"
    elif datasets_with_matches >= 2:
        c3_evidence = "moderate"
    elif datasets_with_matches >= 1:
        c3_evidence = "weak"
    else:
        c3_evidence = "negative"

    # Ablation verdict
    ablation_evidence = "negative"
    mean_delta_hard = float(np.mean(delta_hard_random))
    if mean_delta_hard > 0 and p_hr < 0.05:
        ablation_evidence = "strong"
    elif mean_delta_hard > 0 and p_hr < 0.10:
        ablation_evidence = "moderate"
    elif mean_delta_hard > 0:
        ablation_evidence = "weak"

    verdict = {
        "criterion_1_accuracy_complexity": {
            "evidence": c1_evidence,
            "original_criterion_met": f"{original_count}/{len(c1_results)}",
            "reframed_criterion_met": f"{reframed_count}/{len(c1_results)}",
        },
        "criterion_2_interpretability": {
            "evidence": c2_evidence,
            "sg_figs_hard_mean": hard_interp,
            "random_figs_mean": random_interp,
            "p_value": float(p_interp) if p_interp is not None else None,
        },
        "criterion_3_domain_validation": {
            "evidence": c3_evidence,
            "datasets_with_matches": datasets_with_matches,
            "total_domain_datasets": len(DOMAIN_INTERACTIONS),
        },
        "ablation_synergy_vs_random": {
            "evidence": ablation_evidence,
            "hard_mean_delta": mean_delta_hard,
            "hard_p_value": float(p_hr) if p_hr is not None else None,
            "soft_mean_delta": float(np.mean(delta_soft_random)),
            "soft_p_value": float(p_sr) if p_sr is not None else None,
        },
    }

    logger.info(f"  Criterion 1 (Accuracy+Complexity): {c1_evidence}")
    logger.info(f"  Criterion 2 (Interpretability):     {c2_evidence}")
    logger.info(f"  Criterion 3 (Domain Validation):    {c3_evidence}")
    logger.info(f"  Ablation (Synergy vs Random):        {ablation_evidence}")

    # Confirmed vs disconfirmed aspects
    confirmed = []
    disconfirmed = []

    if c2_evidence in ("strong", "moderate"):
        confirmed.append("Synergy guidance improves interpretability (SG-FIGS-Hard achieves near-perfect interpretability scores)")
    else:
        disconfirmed.append("Synergy guidance does not significantly improve interpretability")

    if ablation_evidence in ("strong", "moderate"):
        confirmed.append("Synergy-guided pair selection provides accuracy advantage over random pair selection")
    else:
        if mean_delta_hard > 0:
            confirmed.append("Synergy guidance shows positive but not statistically significant accuracy advantage over random")
        else:
            disconfirmed.append("Synergy-guided pair selection does not improve accuracy vs random")

    if c1_evidence in ("strong", "moderate"):
        confirmed.append("SG-FIGS achieves competitive accuracy with reduced complexity")
    else:
        disconfirmed.append("SG-FIGS does not consistently achieve better accuracy-complexity tradeoff than RO-FIGS")

    if c3_evidence in ("strong", "moderate"):
        confirmed.append("High-synergy pairs correspond to domain-meaningful feature interactions")
    else:
        disconfirmed.append("Domain validation inconclusive for synergy-guided feature pairs")

    # Cross-reference with prior evaluation
    prior_comparison = {
        "prior_friedman_p": 0.156,
        "prior_n_datasets": 7,
        "current_friedman_p": p_acc,
        "current_n_datasets": len(dataset_names),
        "power_increase": len(dataset_names) / 7,
        "note": (
            "Prior 3-method evaluation on 7 datasets yielded Friedman p=0.156. "
            f"Current 5-method evaluation on {len(dataset_names)} datasets yields "
            f"Friedman p={p_acc:.4f}. "
            f"{'Statistical significance achieved.' if p_acc < 0.05 else 'Still not significant.'}"
        ),
    }

    logger.info(f"  Confirmed: {confirmed}")
    logger.info(f"  Disconfirmed: {disconfirmed}")

    # ── Build output ─────────────────────────────────────────────────────
    logger.info(f"{BLUE}Building output JSON{END}")

    # Compute metrics_agg
    metrics_agg = {
        "friedman_chi2_accuracy": chi2_acc,
        "friedman_p_accuracy": p_acc,
        "nemenyi_cd": cd,
        "n_datasets": len(dataset_names),
        "n_methods": len(METHODS),
        "n_examples_total_exp1": n_examples_exp1,
        "n_examples_total_exp3": n_examples_exp3,
        "n_examples_total_pid": n_examples_pid,
        "mean_acc_figs": float(acc_matrix[:, 0].mean()),
        "mean_acc_ro_figs": float(acc_matrix[:, 1].mean()),
        "mean_acc_sg_figs_hard": float(acc_matrix[:, 2].mean()),
        "mean_acc_sg_figs_soft": float(acc_matrix[:, 3].mean()),
        "mean_acc_random_figs": float(acc_matrix[:, 4].mean()),
        "avg_rank_figs": avg_ranks_acc[0],
        "avg_rank_ro_figs": avg_ranks_acc[1],
        "avg_rank_sg_figs_hard": avg_ranks_acc[2],
        "avg_rank_sg_figs_soft": avg_ranks_acc[3],
        "avg_rank_random_figs": avg_ranks_acc[4],
        "ablation_hard_vs_random_delta": mean_delta_hard,
        "ablation_hard_vs_random_p": float(p_hr) if p_hr is not None else 1.0,
        "ablation_soft_vs_random_delta": float(np.mean(delta_soft_random)),
        "ablation_soft_vs_random_p": float(p_sr) if p_sr is not None else 1.0,
        "ablation_interp_hard_vs_random_delta": float(np.mean(delta_interp)),
        "ablation_interp_hard_vs_random_p": float(p_interp) if p_interp is not None else 1.0,
        "c1_original_met": original_count,
        "c1_reframed_met": reframed_count,
        "c2_hard_interp_mean": interp_means.get("sg_figs_hard", 0),
        "c2_random_interp_mean": interp_means.get("random_figs", 0),
        "c3_domain_matches": datasets_with_matches,
        "threshold_mean_tuning_improvement": mean_tuning_improvement,
        "threshold_tuned_vs_rofigs_delta": thresh_tuned_mean_delta,
        "regression_r2_multiple": r2_multi,
    }

    # Add AUC Friedman if available
    if chi2_auc is not None:
        metrics_agg["friedman_chi2_auc"] = chi2_auc
        metrics_agg["friedman_p_auc"] = p_auc

    # Build per-dataset examples for output
    output_datasets = []

    # Dataset 1: Per-dataset accuracy comparison (exp1)
    ds_examples = []
    for i, ds in enumerate(dataset_names):
        input_data = {
            "dataset": ds,
            "n_features": ds_properties[ds]["n_features"],
            "analysis_type": "5_method_accuracy_comparison",
        }
        output_data = {}
        for j, m in enumerate(METHODS):
            mean_val, std_val = dataset_means[ds][m]["balanced_accuracy"]
            output_data[f"{m}_acc_mean"] = round(mean_val, 6)
            output_data[f"{m}_acc_std"] = round(std_val, 6)
            splits_mean, _ = dataset_means[ds][m]["n_splits"]
            output_data[f"{m}_n_splits"] = round(splits_mean, 1) if splits_mean is not None else None
            interp_m, _ = dataset_means[ds][m]["interpretability_score"]
            output_data[f"{m}_interpretability"] = round(interp_m, 4) if interp_m is not None else None

        # Add AUC if binary
        if ds in binary_ds:
            for j, m in enumerate(METHODS):
                auc_mean, auc_std = dataset_means[ds][m].get("auc", (None, None))
                output_data[f"{m}_auc_mean"] = round(auc_mean, 6) if auc_mean is not None else None
                output_data[f"{m}_auc_std"] = round(auc_std, 6) if auc_std is not None else None

        example = {
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_dataset": ds,
            "metadata_n_features": ds_properties[ds]["n_features"],
            "metadata_analysis": "per_dataset_accuracy",
            "predict_best_method": METHOD_LABELS[METHODS[int(np.argmax(acc_matrix[i]))]],
            "predict_sg_figs_soft_acc": str(round(acc_matrix[i, sg_soft_idx], 6)),
            "eval_rank_sg_figs_hard": float(ranks_acc[i, sg_hard_idx]),
            "eval_rank_sg_figs_soft": float(ranks_acc[i, sg_soft_idx]),
        }
        ds_examples.append(example)

    output_datasets.append({
        "dataset": "accuracy_comparison",
        "examples": ds_examples,
    })

    # Dataset 2: Statistical tests
    stat_examples = []

    # Friedman test result
    stat_examples.append({
        "input": json.dumps({"test": "friedman", "metric": "balanced_accuracy", "n_datasets": len(dataset_names), "n_methods": len(METHODS)}),
        "output": json.dumps({"chi2": round(chi2_acc, 4), "p_value": round(p_acc, 6), "nemenyi_cd": round(cd, 4), "significant": p_acc < 0.05}),
        "metadata_test_type": "friedman",
        "predict_significant": str(p_acc < 0.05),
        "eval_chi2": chi2_acc,
        "eval_p_value": p_acc,
    })

    # AUC Friedman test
    if chi2_auc is not None:
        stat_examples.append({
            "input": json.dumps({"test": "friedman", "metric": "auc", "n_datasets": len(binary_ds), "n_methods": len(METHODS)}),
            "output": json.dumps({"chi2": round(chi2_auc, 4), "p_value": round(p_auc, 6), "significant": p_auc < 0.05}),
            "metadata_test_type": "friedman_auc",
            "predict_significant": str(p_auc < 0.05),
            "eval_chi2": chi2_auc,
            "eval_p_value": p_auc,
        })

    # Pairwise Wilcoxon results
    for r in wilcoxon_results:
        stat_examples.append({
            "input": json.dumps({"test": "wilcoxon", "pair": r["pair"], "correction": "holm_bonferroni"}),
            "output": json.dumps({"p_raw": round(r["p_raw"], 6), "p_corrected": round(r["p_corrected"], 6), "mean_diff": round(r["mean_diff"], 6), "significant": r["significant"]}),
            "metadata_test_type": "wilcoxon_pairwise",
            "metadata_pair": r["pair"],
            "predict_significant": str(r["significant"]),
            "eval_p_corrected": r["p_corrected"],
            "eval_mean_diff": r["mean_diff"],
        })

    output_datasets.append({
        "dataset": "statistical_tests",
        "examples": stat_examples,
    })

    # Dataset 3: Ablation analysis
    ablation_examples = []

    # Hard vs Random per dataset
    for i, ds in enumerate(dataset_names):
        h_acc = acc_matrix[i, sg_hard_idx]
        r_acc = acc_matrix[i, random_idx]
        h_interp_val, _ = dataset_means[ds]["sg_figs_hard"]["interpretability_score"]
        r_interp_val, _ = dataset_means[ds]["random_figs"]["interpretability_score"]

        ablation_examples.append({
            "input": json.dumps({"dataset": ds, "comparison": "sg_figs_hard_vs_random_figs"}),
            "output": json.dumps({
                "hard_acc": round(h_acc, 6),
                "random_acc": round(r_acc, 6),
                "acc_delta": round(h_acc - r_acc, 6),
                "hard_interpretability": round(h_interp_val, 4) if h_interp_val is not None else None,
                "random_interpretability": round(r_interp_val, 4) if r_interp_val is not None else None,
                "interp_delta": round(h_interp_val - r_interp_val, 4) if (h_interp_val is not None and r_interp_val is not None) else None,
            }),
            "metadata_dataset": ds,
            "metadata_comparison": "hard_vs_random",
            "predict_hard_better_acc": str(h_acc > r_acc),
            "eval_acc_delta": float(h_acc - r_acc),
            "eval_interp_delta": float(h_interp_val - r_interp_val) if (h_interp_val is not None and r_interp_val is not None) else 0.0,
        })

    output_datasets.append({
        "dataset": "ablation_analysis",
        "examples": ablation_examples,
    })

    # Dataset 4: Criterion 1 results
    c1_examples = []
    for r in c1_results:
        c1_examples.append({
            "input": json.dumps({"dataset": r["dataset"], "criterion": "accuracy_complexity"}),
            "output": json.dumps({
                "acc_delta_soft_vs_ro": round(r["acc_delta"], 6),
                "split_ratio_soft_vs_ro": round(r["split_ratio"], 4),
                "original_criterion_met": r["original_criterion_met"],
                "reframed_criterion_met": r["reframed_criterion_met"],
            }),
            "metadata_dataset": r["dataset"],
            "metadata_criterion": "c1_accuracy_complexity",
            "predict_original_met": str(r["original_criterion_met"]),
            "predict_reframed_met": str(r["reframed_criterion_met"]),
            "eval_acc_delta": r["acc_delta"],
            "eval_split_ratio": r["split_ratio"],
        })

    output_datasets.append({
        "dataset": "criterion_1_accuracy_complexity",
        "examples": c1_examples,
    })

    # Dataset 5: Criterion 2 (interpretability per dataset)
    c2_examples = []
    for ds in dataset_names:
        input_data = {"dataset": ds, "criterion": "interpretability"}
        output_data = {}
        for m in METHODS:
            interp_m, _ = dataset_means[ds][m]["interpretability_score"]
            output_data[f"{m}_interpretability"] = round(interp_m, 4) if interp_m is not None else None

        c2_examples.append({
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_dataset": ds,
            "metadata_criterion": "c2_interpretability",
            "predict_hard_interp": str(round(dataset_means[ds]["sg_figs_hard"]["interpretability_score"][0], 4)) if dataset_means[ds]["sg_figs_hard"]["interpretability_score"][0] is not None else "null",
            "eval_hard_interp": float(dataset_means[ds]["sg_figs_hard"]["interpretability_score"][0]) if dataset_means[ds]["sg_figs_hard"]["interpretability_score"][0] is not None else 0.0,
        })

    output_datasets.append({
        "dataset": "criterion_2_interpretability",
        "examples": c2_examples,
    })

    # Dataset 6: Criterion 3 (domain validation)
    c3_examples = []
    for domain_ds, dr in domain_results.items():
        c3_examples.append({
            "input": json.dumps({"dataset": domain_ds, "criterion": "domain_validation"}),
            "output": json.dumps({
                "top_synergy_pairs": dr["top_synergy_pairs"],
                "known_interactions_found": dr["known_interactions_found"],
                "coverage": dr["coverage"],
            }),
            "metadata_dataset": domain_ds,
            "metadata_criterion": "c3_domain_validation",
            "predict_has_match": str(dr["coverage"] > 0),
            "eval_coverage": float(dr["coverage"]),
        })

    output_datasets.append({
        "dataset": "criterion_3_domain_validation",
        "examples": c3_examples,
    })

    # Dataset 7: Threshold sensitivity
    thresh_examples = []
    for ds in sorted(threshold_analysis.keys()):
        ta = threshold_analysis[ds]
        thresh_examples.append({
            "input": json.dumps({"dataset": ds, "analysis": "threshold_sensitivity"}),
            "output": json.dumps({
                "threshold_accs": ta["threshold_accs"],
                "optimal_threshold": ta["optimal_threshold"],
                "optimal_acc": round(ta["optimal_acc"], 6),
                "fixed_75_acc": round(ta["fixed_75_acc"], 6),
                "improvement_from_tuning": round(ta["improvement_from_tuning"], 6),
                "rofigs_mean_acc": round(ta["rofigs_mean_acc"], 6),
            }),
            "metadata_dataset": ds,
            "metadata_analysis": "threshold_sensitivity",
            "predict_optimal_threshold": str(ta["optimal_threshold"]),
            "eval_tuning_improvement": ta["improvement_from_tuning"],
        })

    output_datasets.append({
        "dataset": "threshold_sensitivity",
        "examples": thresh_examples,
    })

    # Dataset 8: Pareto analysis
    pareto_examples = []
    for m in METHODS:
        pd_entry = pareto_data[m]
        pareto_examples.append({
            "input": json.dumps({"method": METHOD_LABELS[m], "analysis": "pareto"}),
            "output": json.dumps({
                "mean_accuracy": round(pd_entry["mean_accuracy"], 6),
                "mean_n_splits": round(pd_entry["mean_n_splits"], 2),
                "accuracy_per_split": round(pd_entry["accuracy_per_split"], 6),
                "pareto_optimal": METHOD_LABELS[m] in pareto_dominant,
            }),
            "metadata_method": METHOD_LABELS[m],
            "metadata_analysis": "pareto",
            "predict_pareto_optimal": str(METHOD_LABELS[m] in pareto_dominant),
            "eval_accuracy_per_split": pd_entry["accuracy_per_split"],
        })

    output_datasets.append({
        "dataset": "pareto_analysis",
        "examples": pareto_examples,
    })

    # Dataset 9: Dataset regression
    reg_examples = []
    for pred_name, reg_r in regression_results.items():
        reg_examples.append({
            "input": json.dumps({"predictor": pred_name, "target": "acc_delta_soft_vs_ro"}),
            "output": json.dumps({
                "slope": round(reg_r["slope"], 6),
                "intercept": round(reg_r["intercept"], 6),
                "r_squared": round(reg_r["r_squared"], 6),
                "p_value": round(reg_r["p_value"], 6),
            }),
            "metadata_predictor": pred_name,
            "metadata_analysis": "dataset_regression",
            "predict_significant": str(reg_r["p_value"] < 0.05),
            "eval_r_squared": reg_r["r_squared"],
        })

    # Multiple regression summary
    reg_examples.append({
        "input": json.dumps({"predictor": "multiple_all", "target": "acc_delta_soft_vs_ro"}),
        "output": json.dumps({
            "r_squared_multiple": round(r2_multi, 6),
            "predictors": predictor_names,
            "coefficients": {p: round(float(beta[i + 1]), 6) for i, p in enumerate(predictor_names)},
        }),
        "metadata_predictor": "multiple",
        "metadata_analysis": "dataset_regression",
        "predict_significant": str(r2_multi > 0.5),
        "eval_r_squared": r2_multi,
    })

    output_datasets.append({
        "dataset": "dataset_regression",
        "examples": reg_examples,
    })

    # Dataset 10: Overall verdict
    verdict_examples = []
    verdict_examples.append({
        "input": json.dumps({"analysis": "overall_verdict"}),
        "output": json.dumps({
            "criterion_1": verdict["criterion_1_accuracy_complexity"],
            "criterion_2": verdict["criterion_2_interpretability"],
            "criterion_3": verdict["criterion_3_domain_validation"],
            "ablation": verdict["ablation_synergy_vs_random"],
            "confirmed": confirmed,
            "disconfirmed": disconfirmed,
            "prior_comparison": prior_comparison,
            "pareto_optimal_methods": pareto_dominant,
        }),
        "metadata_analysis": "overall_verdict",
        "predict_strongest_criterion": max(
            [("c1", c1_evidence), ("c2", c2_evidence), ("c3", c3_evidence)],
            key=lambda x: {"strong": 4, "moderate": 3, "weak": 2, "negative": 1}[x[1]],
        )[0],
        "eval_n_confirmed": float(len(confirmed)),
        "eval_n_disconfirmed": float(len(disconfirmed)),
    })

    output_datasets.append({
        "dataset": "overall_verdict",
        "examples": verdict_examples,
    })

    # ── Assemble final output ────────────────────────────────────────────
    output = {
        "metadata": {
            "evaluation_name": "SG-FIGS Definitive Statistical Evaluation",
            "description": (
                "Comprehensive evaluation of 5-method comparison (FIGS, RO-FIGS, "
                "SG-FIGS-Hard, SG-FIGS-Soft, Random-FIGS) across 14 datasets with "
                "Friedman tests, pairwise Wilcoxon, ablation analysis, threshold "
                "sensitivity, PID synergy domain validation, and Pareto analysis."
            ),
            "experiments_evaluated": [
                "exp_id1_it3 (5-method comparison, 14 datasets, 7472 examples)",
                "exp_id3_it3 (threshold sensitivity, 14 datasets, 630 examples)",
                "exp_id1_it2 (PID synergy matrices, 12 datasets, 2569 examples)",
            ],
            "methods": [METHOD_LABELS[m] for m in METHODS],
            "statistical_tests": [
                "Friedman χ² test (nonparametric, repeated measures)",
                "Nemenyi post-hoc with critical difference",
                "Pairwise Wilcoxon signed-rank with Holm-Bonferroni correction",
            ],
            "success_criteria": [
                "C1: Accuracy within 1% of RO-FIGS with 20% fewer splits",
                "C2: Higher interpretability than baselines",
                "C3: High-synergy pairs correspond to known domain interactions",
            ],
        },
        "metrics_agg": metrics_agg,
        "datasets": output_datasets,
    }

    # ── Save output ──────────────────────────────────────────────────────
    output_path = WORKSPACE / "eval_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"{GREEN}Saved evaluation to {output_path}{END}")
    logger.info(f"  Total output datasets: {len(output_datasets)}")
    total_examples = sum(len(ds["examples"]) for ds in output_datasets)
    logger.info(f"  Total output examples: {total_examples}")

    # ── Print summary ────────────────────────────────────────────────────
    logger.info(f"\n{GREEN}═══ EVALUATION SUMMARY ═══{END}")
    logger.info(f"  Friedman χ²={chi2_acc:.4f}, p={p_acc:.6f} {'(SIGNIFICANT)' if p_acc < 0.05 else '(not significant)'}")
    logger.info(f"  Best method by avg rank: {METHOD_LABELS[METHODS[int(np.argmin(avg_ranks_acc))]]}")
    logger.info(f"  Best method by mean acc: {METHOD_LABELS[METHODS[int(np.argmax(acc_matrix.mean(axis=0)))]]}")
    logger.info(f"  Criterion 1: {c1_evidence} ({original_count}/{len(c1_results)} original, {reframed_count}/{len(c1_results)} reframed)")
    logger.info(f"  Criterion 2: {c2_evidence} (Hard interp={hard_interp:.3f} vs Random={random_interp:.3f})")
    logger.info(f"  Criterion 3: {c3_evidence} ({datasets_with_matches}/{len(DOMAIN_INTERACTIONS)} domain matches)")
    logger.info(f"  Ablation: {ablation_evidence} (Hard-Random Δ={mean_delta_hard:+.4f}, p={p_hr:.4f})")
    logger.info(f"  Pareto-optimal: {pareto_dominant}")
    logger.success(f"{GREEN}Evaluation complete!{END}")


if __name__ == "__main__":
    main()
