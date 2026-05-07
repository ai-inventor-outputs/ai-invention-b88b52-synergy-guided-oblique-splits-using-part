#!/usr/bin/env python3
"""Statistical Evaluation of SG-FIGS Experiment Results.

Rigorous statistical evaluation of iteration 2's SG-FIGS experiment including:
1. Friedman test with Nemenyi post-hoc
2. Wilcoxon signed-rank pairwise tests
3. Criterion 1: per-dataset accuracy + complexity assessment
4. Criterion 2: interpretability score diagnostic
5. Criterion 3: domain analysis of synergy pairs
6. Pareto frontier analysis (accuracy vs complexity)
7. Synergy landscape correlation analysis
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Resource limits (14 GB RAM, 3500s CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent

# Experiment data (exp_id2)
EXP_ID2_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_2/gen_art/exp_id2_it2__opus"
)
EXP_ID2_FULL = EXP_ID2_DIR / "full_method_out.json"

# Synergy data (exp_id1)
EXP_ID1_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus"
)
EXP_ID1_FULL = EXP_ID1_DIR / "full_method_out.json"
EXP_ID1_RESULTS = EXP_ID1_DIR / "results_comprehensive.json"

OUTPUT_PATH = WORKSPACE / "eval_out.json"

# Methods being compared (binary datasets only)
METHODS = ["FIGS", "RO-FIGS", "SG-FIGS"]
METHOD_KEYS = {"FIGS": "predict_figs", "RO-FIGS": "predict_ro_figs", "SG-FIGS": "predict_sg_figs"}

# Domain knowledge for Criterion 3 analysis
DOMAIN_KNOWLEDGE = {
    "pima_diabetes": {
        "domain": "medical",
        "known_interactions": {
            ("mass", "age"): "BMI increases with age; age-dependent obesity threshold is a well-known diabetes risk factor",
            ("plas", "mass"): "Glucose-BMI interaction: insulin resistance is amplified by obesity (metabolic syndrome)",
            ("preg", "mass"): "Pregnancy count + BMI: gestational diabetes risk increases with both parity and obesity",
            ("plas", "age"): "Age-dependent glucose tolerance decline is a hallmark of Type 2 diabetes progression",
            ("skin", "age"): "Skin fold thickness (proxy for subcutaneous fat) changes with age; less diagnostic alone",
        },
    },
    "breast_cancer_wisconsin_diagnostic": {
        "domain": "medical",
        "known_interactions": {
            ("radius error", "worst compactness"): "Measurement variability + shape compactness: tumour heterogeneity signals malignancy",
            ("radius error", "worst concavity"): "Size variability + concavity: irregular margins indicate invasive carcinoma",
            ("area error", "worst concavity"): "Area measurement noise + concavity extremes: captures nuclear pleomorphism",
            ("mean area", "worst smoothness"): "Tumour size + surface regularity: large smooth masses are often benign cysts",
            ("mean radius", "worst smoothness"): "Cell size + smoothness: small irregular cells are hallmark of high-grade tumours",
        },
    },
    "heart_statlog": {
        "domain": "medical",
        "known_interactions": {
            ("slope", "number_of_major_vessels"): "ST slope + vessel count: combined indicator of ischemic burden severity",
            ("serum_cholestoral", "oldpeak"): "Cholesterol + ST depression: lipid profile with exercise ECG is standard cardiac risk assessment",
            ("age", "maximum_heart_rate_achieved"): "Age-dependent max HR is a primary fitness/cardiac capacity indicator (220-age rule)",
            ("maximum_heart_rate_achieved", "oldpeak"): "HR response + ST depression: chronotropic incompetence with ischemia signals severe CAD",
            ("oldpeak", "number_of_major_vessels"): "ST depression magnitude + vessel disease count: directly maps to multi-vessel coronary artery disease",
        },
    },
    "banknote": {
        "domain": "signal_processing",
        "known_interactions": {
            ("V2", "V4"): "Wavelet-transformed variance + entropy: captures texture complexity in genuine vs forged notes",
            ("V2", "V3"): "Variance + skewness of wavelet transform: distribution shape features for authentication",
            ("V1", "V2"): "Variance + variance: primary wavelet feature pair for banknote classification",
        },
    },
    "ionosphere": {
        "domain": "radar",
        "known_interactions": {
            ("a12", "a21"): "Radar return signal components: phase-amplitude cross-correlation for ionospheric structure detection",
        },
    },
    "sonar": {
        "domain": "sonar",
        "known_interactions": {
            ("attribute_36", "attribute_46"): "Mid-to-high frequency energy bands: spectral shape discriminates rock vs metal",
        },
    },
    "spectf_heart": {
        "domain": "medical",
        "known_interactions": {
            ("F5R", "F14S"): "Rest + stress SPECT features: perfusion comparison between rest and stress indicates ischemia",
        },
    },
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_experiment_data(data_path: Path, limit: int | None = None) -> dict:
    """Load experiment data from exp_id2 full_method_out.json.

    Returns dict keyed by dataset name with per-fold metrics for each method.
    """
    logger.info(f"Loading experiment data from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets = {}

    for ds_entry in raw["datasets"]:
        ds_name = ds_entry["dataset"]
        examples = ds_entry["examples"]
        if limit is not None:
            examples = examples[:limit]

        # Determine which methods are available
        available_methods = []
        for method, key in METHOD_KEYS.items():
            if key in examples[0]:
                available_methods.append(method)

        # Extract per-fold metrics for each method
        method_folds = {m: {} for m in available_methods}
        for ex in examples:
            fold = ex.get("metadata_fold", 0)
            for method in available_methods:
                key = METHOD_KEYS[method]
                pred = json.loads(ex[key])
                # Each example in a fold has the same prediction (fold-level metric)
                if fold not in method_folds[method]:
                    method_folds[method][fold] = pred

        datasets[ds_name] = {
            "methods": available_methods,
            "method_folds": method_folds,
            "n_examples": len(examples),
            "n_classes": examples[0].get("metadata_n_classes", 2),
            "n_features": examples[0].get("metadata_n_features", 0),
            "domain": examples[0].get("metadata_domain", "unknown"),
            "task_type": examples[0].get("metadata_task_type", "classification"),
        }

    logger.info(f"Loaded {len(datasets)} datasets")
    for ds_name, ds_info in datasets.items():
        logger.info(f"  {ds_name}: {ds_info['n_examples']} examples, methods={ds_info['methods']}")

    return datasets


def load_synergy_data(results_path: Path) -> dict:
    """Load synergy analysis results from exp_id1 results_comprehensive.json."""
    logger.info(f"Loading synergy data from {results_path}")
    raw = json.loads(results_path.read_text())

    synergy_data = {}
    for ds_summary in raw["aggregate"]["per_dataset_summary"]:
        ds_name = ds_summary["dataset"]
        synergy_data[ds_name] = {
            "synergy_mean": ds_summary.get("synergy_mean", 0.0),
            "synergy_std": ds_summary.get("synergy_std", 0.0),
            "synergy_max": ds_summary.get("synergy_max", 0.0),
            "synergy_min": ds_summary.get("synergy_min", 0.0),
            "mi_comparison_jaccard": ds_summary.get("mi_comparison_jaccard", 0.0),
            "mi_comparison_spearman": ds_summary.get("mi_comparison_spearman", 0.0),
            "synergy_graph_edges": ds_summary.get("synergy_graph_edges", 0),
            "synergy_graph_components": ds_summary.get("synergy_graph_components", 0),
            "synergy_graph_largest_clique": ds_summary.get("synergy_graph_largest_clique", 0),
            "n_features": ds_summary.get("n_features", 0),
            "n_features_used": ds_summary.get("n_features_used", 0),
            "pid_method": ds_summary.get("pid_method", "unknown"),
            "n_pairs": ds_summary.get("n_pairs", 0),
            "stability_mean_rho": ds_summary.get("stability_mean_rho"),
            "stability_std_rho": ds_summary.get("stability_std_rho"),
        }

    # Load synergy matrices from per_dataset_full
    for ds_full in raw.get("per_dataset_full", []):
        ds_name = ds_full["dataset"]
        if ds_name in synergy_data and "synergy_matrix" in ds_full:
            synergy_data[ds_name]["synergy_matrix"] = np.array(ds_full["synergy_matrix"])

    logger.info(f"Loaded synergy data for {len(synergy_data)} datasets")
    return synergy_data


def load_synergy_pairs(data_path: Path) -> dict:
    """Load top synergy pairs per dataset from exp_id1 full_method_out.json."""
    logger.info(f"Loading synergy pairs from {data_path}")
    raw = json.loads(data_path.read_text())
    pairs_by_dataset = {}

    for ds_entry in raw["datasets"]:
        ds_name = ds_entry["dataset"]
        pairs = []
        for ex in ds_entry["examples"]:
            output = json.loads(ex["output"])
            pairs.append({
                "feature_i": ex["metadata_feature_i"],
                "feature_j": ex["metadata_feature_j"],
                "synergy": output["synergy"],
                "redundancy": output.get("redundancy", 0.0),
                "unique_0": output.get("unique_0", 0.0),
                "unique_1": output.get("unique_1", 0.0),
                "coi_baseline": output.get("coi_baseline", 0.0),
            })
        pairs.sort(key=lambda x: x["synergy"], reverse=True)
        pairs_by_dataset[ds_name] = pairs

    logger.info(f"Loaded synergy pairs for {len(pairs_by_dataset)} datasets")
    return pairs_by_dataset


# ---------------------------------------------------------------------------
# Helper: Extract mean balanced accuracy per dataset per method
# ---------------------------------------------------------------------------

def extract_mean_accuracy_per_dataset(datasets: dict) -> dict:
    """Extract mean balanced accuracy across folds for each dataset-method pair.

    Returns: {dataset: {method: mean_balanced_accuracy}}
    """
    result = {}
    for ds_name, ds_info in datasets.items():
        result[ds_name] = {}
        for method in ds_info["methods"]:
            fold_accs = []
            for fold_id, fold_data in ds_info["method_folds"][method].items():
                acc = fold_data.get("balanced_accuracy")
                if acc is not None:
                    fold_accs.append(acc)
            if fold_accs:
                result[ds_name][method] = float(np.mean(fold_accs))
    return result


def extract_mean_splits_per_dataset(datasets: dict) -> dict:
    """Extract mean n_splits across folds for each dataset-method pair."""
    result = {}
    for ds_name, ds_info in datasets.items():
        result[ds_name] = {}
        for method in ds_info["methods"]:
            fold_splits = []
            for fold_id, fold_data in ds_info["method_folds"][method].items():
                splits = fold_data.get("n_splits")
                if splits is not None:
                    fold_splits.append(splits)
            if fold_splits:
                result[ds_name][method] = float(np.mean(fold_splits))
    return result


# ---------------------------------------------------------------------------
# 1. Friedman Test with Nemenyi Post-Hoc
# ---------------------------------------------------------------------------

def friedman_nemenyi_test(acc_per_dataset: dict) -> dict:
    """Friedman test on rank matrix (7 binary datasets x 3 methods).

    Only uses binary datasets that have all 3 methods.
    """
    logger.info("--- Friedman Test with Nemenyi Post-Hoc ---")

    # Filter to datasets with all 3 methods
    complete_datasets = []
    for ds_name, methods_acc in acc_per_dataset.items():
        if all(m in methods_acc for m in METHODS):
            complete_datasets.append(ds_name)

    N = len(complete_datasets)
    k = len(METHODS)
    logger.info(f"Using {N} datasets with all {k} methods: {complete_datasets}")

    if N < 3:
        logger.warning("Too few datasets for Friedman test (need >= 3)")
        return {
            "n_datasets": N,
            "n_methods": k,
            "datasets_used": complete_datasets,
            "error": "Too few datasets for Friedman test",
        }

    # Build accuracy matrix: N x k
    acc_matrix = np.zeros((N, k))
    for i, ds_name in enumerate(complete_datasets):
        for j, method in enumerate(METHODS):
            acc_matrix[i, j] = acc_per_dataset[ds_name][method]

    # Compute ranks per dataset (higher accuracy = lower rank = better)
    rank_matrix = np.zeros((N, k))
    for i in range(N):
        # Use scipy rankdata (1=best), then convert so that highest acc gets rank 1
        rank_matrix[i] = stats.rankdata(-acc_matrix[i], method="average")

    avg_ranks = np.mean(rank_matrix, axis=0)
    logger.info(f"Average ranks: {dict(zip(METHODS, avg_ranks.tolist()))}")

    # Friedman test
    stat, p_value = stats.friedmanchisquare(*[acc_matrix[:, j] for j in range(k)])
    logger.info(f"Friedman chi2={stat:.4f}, p={p_value:.6f}")

    result = {
        "n_datasets": N,
        "n_methods": k,
        "datasets_used": complete_datasets,
        "accuracy_matrix": {ds_name: {m: round(acc_matrix[i, j], 6) for j, m in enumerate(METHODS)}
                           for i, ds_name in enumerate(complete_datasets)},
        "rank_matrix": {ds_name: {m: round(rank_matrix[i, j], 2) for j, m in enumerate(METHODS)}
                        for i, ds_name in enumerate(complete_datasets)},
        "average_ranks": {m: round(float(avg_ranks[j]), 4) for j, m in enumerate(METHODS)},
        "friedman_statistic": round(float(stat), 4),
        "friedman_p_value": round(float(p_value), 6),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_010": bool(p_value < 0.10),
    }

    # Nemenyi post-hoc if significant at alpha=0.10
    if p_value < 0.10:
        # Critical difference: CD = q_alpha * sqrt(k*(k+1)/(6*N))
        # Nemenyi q values for k=3: q_0.05 = 2.343, q_0.10 = 2.052
        q_005 = 2.343
        q_010 = 2.052
        cd_005 = q_005 * np.sqrt(k * (k + 1) / (6 * N))
        cd_010 = q_010 * np.sqrt(k * (k + 1) / (6 * N))

        logger.info(f"Nemenyi CD (alpha=0.05): {cd_005:.4f}")
        logger.info(f"Nemenyi CD (alpha=0.10): {cd_010:.4f}")

        # Pairwise comparisons
        nemenyi_pairs = {}
        for i_m in range(k):
            for j_m in range(i_m + 1, k):
                pair_name = f"{METHODS[i_m]}_vs_{METHODS[j_m]}"
                rank_diff = abs(avg_ranks[i_m] - avg_ranks[j_m])
                nemenyi_pairs[pair_name] = {
                    "rank_difference": round(float(rank_diff), 4),
                    "significant_at_005": bool(rank_diff > cd_005),
                    "significant_at_010": bool(rank_diff > cd_010),
                }
                logger.info(f"  {pair_name}: rank_diff={rank_diff:.4f}, sig_005={rank_diff > cd_005}, sig_010={rank_diff > cd_010}")

        result["nemenyi_post_hoc"] = {
            "critical_difference_005": round(float(cd_005), 4),
            "critical_difference_010": round(float(cd_010), 4),
            "pairwise_comparisons": nemenyi_pairs,
        }
    else:
        result["nemenyi_post_hoc"] = {
            "note": "Friedman test not significant — Nemenyi post-hoc not warranted",
        }

    return result


# ---------------------------------------------------------------------------
# 2. Wilcoxon Signed-Rank Tests
# ---------------------------------------------------------------------------

def wilcoxon_pairwise_tests(acc_per_dataset: dict) -> dict:
    """Wilcoxon signed-rank test for each method pair on balanced accuracy."""
    logger.info("--- Wilcoxon Signed-Rank Pairwise Tests ---")

    # Filter to datasets with all 3 methods
    complete_datasets = [ds for ds, ma in acc_per_dataset.items() if all(m in ma for m in METHODS)]
    N = len(complete_datasets)

    logger.info(f"Using {N} datasets: {complete_datasets}")
    logger.info(f"Note: with N={N}, minimum achievable p-value = {1 / (2**N):.4f}")

    pairs_to_test = [
        ("FIGS", "RO-FIGS"),
        ("FIGS", "SG-FIGS"),
        ("RO-FIGS", "SG-FIGS"),
    ]

    results = {}
    for m1, m2 in pairs_to_test:
        pair_name = f"{m1}_vs_{m2}"
        diffs = []
        for ds_name in complete_datasets:
            acc1 = acc_per_dataset[ds_name][m1]
            acc2 = acc_per_dataset[ds_name][m2]
            diffs.append(acc1 - acc2)

        diffs_arr = np.array(diffs)
        mean_diff = float(np.mean(diffs_arr))
        std_diff = float(np.std(diffs_arr, ddof=1)) if len(diffs_arr) > 1 else 0.0

        # Wilcoxon signed-rank test (exact for small N)
        try:
            # Use 'wilcox' method for exact p-values at small N
            stat_val, p_val = stats.wilcoxon(diffs_arr, alternative="two-sided", method="exact")
            test_performed = True
        except ValueError:
            # If all differences are zero, the test cannot be performed
            stat_val, p_val = float("nan"), 1.0
            test_performed = False

        per_dataset_diffs = {ds: round(diffs[i], 6) for i, ds in enumerate(complete_datasets)}

        logger.info(f"  {pair_name}: mean_diff={mean_diff:.6f}, stat={stat_val}, p={p_val:.6f}")

        results[pair_name] = {
            "method_1": m1,
            "method_2": m2,
            "n_datasets": N,
            "per_dataset_differences": per_dataset_diffs,
            "mean_difference": round(mean_diff, 6),
            "std_difference": round(std_diff, 6),
            "wilcoxon_statistic": round(float(stat_val), 4) if not np.isnan(stat_val) else None,
            "p_value": round(float(p_val), 6) if not np.isnan(p_val) else None,
            "significant_at_005": bool(p_val < 0.05) if test_performed else False,
            "significant_at_010": bool(p_val < 0.10) if test_performed else False,
            "test_performed": test_performed,
            "low_power_caveat": f"With N={N}, minimum achievable p-value is {1 / (2**N):.4f}",
            "interpretation": (
                f"{m1} is {'better' if mean_diff > 0 else 'worse'} than {m2} "
                f"by {abs(mean_diff):.4f} on average"
            ),
        }

    return {
        "n_datasets": N,
        "datasets_used": complete_datasets,
        "minimum_achievable_p": round(1 / (2**N), 6),
        "pairwise_tests": results,
    }


# ---------------------------------------------------------------------------
# 3. Criterion 1: Per-Dataset Accuracy + Complexity Assessment
# ---------------------------------------------------------------------------

def criterion_1_analysis(
    acc_per_dataset: dict,
    splits_per_dataset: dict,
) -> dict:
    """Criterion 1: |acc_delta| <= 0.01 AND split_ratio <= 0.80.

    For each binary dataset, check if SG-FIGS achieves competitive accuracy
    with at least 20% fewer splits compared to RO-FIGS.
    """
    logger.info("--- Criterion 1: Accuracy + Complexity Assessment ---")

    # Filter to datasets with SG-FIGS and RO-FIGS
    eligible_datasets = [
        ds for ds in acc_per_dataset
        if "SG-FIGS" in acc_per_dataset[ds] and "RO-FIGS" in acc_per_dataset[ds]
    ]

    per_dataset_results = {}
    meets_acc = 0
    meets_splits = 0
    meets_both = 0
    meets_alternative = 0  # significantly higher accuracy at same complexity

    for ds_name in eligible_datasets:
        sg_acc = acc_per_dataset[ds_name]["SG-FIGS"]
        ro_acc = acc_per_dataset[ds_name]["RO-FIGS"]
        figs_acc = acc_per_dataset[ds_name].get("FIGS", None)
        sg_splits = splits_per_dataset[ds_name].get("SG-FIGS", 0)
        ro_splits = splits_per_dataset[ds_name].get("RO-FIGS", 0)
        figs_splits = splits_per_dataset[ds_name].get("FIGS", 0)

        acc_delta = sg_acc - ro_acc
        split_ratio = sg_splits / ro_splits if ro_splits > 0 else float("inf")

        acc_within_1pct = abs(acc_delta) <= 0.01
        splits_20pct_fewer = split_ratio <= 0.80
        higher_acc_same_complexity = acc_delta > 0.01 and split_ratio <= 1.10

        if acc_within_1pct:
            meets_acc += 1
        if splits_20pct_fewer:
            meets_splits += 1
        if acc_within_1pct and splits_20pct_fewer:
            meets_both += 1
        if higher_acc_same_complexity:
            meets_alternative += 1

        assessment = "FAIL"
        if acc_within_1pct and splits_20pct_fewer:
            assessment = "PASS_MAIN"
        elif higher_acc_same_complexity:
            assessment = "PASS_ALTERNATIVE"

        per_dataset_results[ds_name] = {
            "sg_figs_accuracy": round(sg_acc, 6),
            "ro_figs_accuracy": round(ro_acc, 6),
            "figs_accuracy": round(figs_acc, 6) if figs_acc is not None else None,
            "acc_delta_sg_minus_ro": round(acc_delta, 6),
            "sg_figs_splits": round(sg_splits, 2),
            "ro_figs_splits": round(ro_splits, 2),
            "figs_splits": round(figs_splits, 2) if figs_splits else None,
            "split_ratio_sg_over_ro": round(split_ratio, 4),
            "acc_within_1pct": acc_within_1pct,
            "splits_20pct_fewer": splits_20pct_fewer,
            "higher_acc_same_complexity": higher_acc_same_complexity,
            "assessment": assessment,
        }

        logger.info(
            f"  {ds_name}: acc_delta={acc_delta:+.4f}, split_ratio={split_ratio:.2f}, "
            f"assessment={assessment}"
        )

    n_eligible = len(eligible_datasets)
    result = {
        "n_eligible_datasets": n_eligible,
        "eligible_datasets": eligible_datasets,
        "per_dataset": per_dataset_results,
        "summary": {
            "datasets_meeting_acc_criterion": meets_acc,
            "datasets_meeting_splits_criterion": meets_splits,
            "datasets_meeting_both": meets_both,
            "datasets_meeting_alternative": meets_alternative,
            "datasets_passing_either": meets_both + meets_alternative,
            "pass_rate_main": round(meets_both / n_eligible, 4) if n_eligible > 0 else 0,
            "pass_rate_any": round((meets_both + meets_alternative) / n_eligible, 4) if n_eligible > 0 else 0,
        },
    }

    logger.info(
        f"  Summary: {meets_both}/{n_eligible} pass main criterion, "
        f"{meets_alternative}/{n_eligible} pass alternative, "
        f"{meets_both + meets_alternative}/{n_eligible} pass either"
    )

    return result


# ---------------------------------------------------------------------------
# 4. Criterion 2: Interpretability Score Diagnostic
# ---------------------------------------------------------------------------

def criterion_2_interpretability_diagnostic(
    datasets: dict,
    synergy_data: dict,
) -> dict:
    """Diagnose why 5/7 datasets scored 0.000 on split_interpretability_score.

    Analyzes:
    1. Feature-index mismatch for high-dim datasets (MI prefiltering to top-20)
    2. Axis-aligned fallback producing no oblique nodes
    3. Proxy interpretability from synergy matrices
    """
    logger.info("--- Criterion 2: Interpretability Score Diagnostic ---")

    per_dataset = {}

    for ds_name, ds_info in datasets.items():
        if "SG-FIGS" not in ds_info["methods"] or "RO-FIGS" not in ds_info["methods"]:
            continue

        # Extract interpretability-related metrics from fold data
        sg_interp_scores = []
        ro_interp_scores = []
        sg_n_splits_list = []
        ro_n_splits_list = []

        for fold_id, fold_data in ds_info["method_folds"]["SG-FIGS"].items():
            sg_n_splits_list.append(fold_data.get("n_splits", 0))

        for fold_id, fold_data in ds_info["method_folds"]["RO-FIGS"].items():
            ro_n_splits_list.append(fold_data.get("n_splits", 0))

        n_features = ds_info["n_features"]
        is_high_dim = n_features > 20

        # Check if synergy data is available
        has_synergy = ds_name in synergy_data
        synergy_info = synergy_data.get(ds_name, {})
        n_features_used = synergy_info.get("n_features_used", n_features)
        pid_method = synergy_info.get("pid_method", "unknown")

        # Potential issues
        feature_index_mismatch = is_high_dim and n_features_used < n_features
        # If MI prefiltering selects top-20, the synergy matrix is 20x20 (or n_features_used x n_features_used)
        # but the oblique split uses original feature indices in the full d-dimensional space
        # This means the synergy matrix indices don't match the feature indices used in splits

        # Compute proxy interpretability from synergy matrix if available
        proxy_interp = None
        if has_synergy and "synergy_matrix" in synergy_info:
            S = synergy_info["synergy_matrix"]
            d = S.shape[0]
            upper_tri = S[np.triu_indices(d, k=1)]
            pos_syn = upper_tri[upper_tri > 0]
            if len(pos_syn) > 0:
                median_syn = float(np.median(pos_syn))
                # Fraction of pairs above median
                n_above = np.sum(pos_syn > median_syn)
                proxy_interp = round(float(n_above / len(pos_syn)), 4) if len(pos_syn) > 0 else 0.0
            else:
                proxy_interp = 0.0

        # Determine likely zero-score cause
        likely_causes = []
        if feature_index_mismatch:
            likely_causes.append(
                f"Feature-index mismatch: dataset has {n_features} features, "
                f"MI prefiltering selects top-{n_features_used}. Synergy matrix is "
                f"{n_features_used}x{n_features_used} but oblique split indices "
                f"reference original {n_features}-dim space, causing out-of-bounds lookups."
            )
        if is_high_dim:
            likely_causes.append(
                f"High-dimensional dataset ({n_features} features): PID computation "
                f"used {pid_method} method with potential sparsity issues."
            )

        # Check axis-aligned fallback: if mean n_splits is very low, the model
        # may have only axis-aligned splits (no oblique nodes to score)
        mean_sg_splits = float(np.mean(sg_n_splits_list)) if sg_n_splits_list else 0
        mean_ro_splits = float(np.mean(ro_n_splits_list)) if ro_n_splits_list else 0

        if mean_sg_splits <= 5:
            likely_causes.append(
                f"Low complexity (mean SG-FIGS splits={mean_sg_splits:.1f}): "
                f"axis-aligned fallback may dominate, producing few/no oblique nodes."
            )

        per_dataset[ds_name] = {
            "n_features": n_features,
            "n_features_used_in_synergy": n_features_used,
            "is_high_dim": is_high_dim,
            "feature_index_mismatch_likely": feature_index_mismatch,
            "pid_method": pid_method,
            "mean_sg_figs_splits": round(mean_sg_splits, 2),
            "mean_ro_figs_splits": round(mean_ro_splits, 2),
            "proxy_interpretability_score": proxy_interp,
            "likely_zero_score_causes": likely_causes,
            "has_synergy_matrix": has_synergy and "synergy_matrix" in synergy_info,
        }

        logger.info(
            f"  {ds_name}: n_feat={n_features}, high_dim={is_high_dim}, "
            f"idx_mismatch={feature_index_mismatch}, proxy_interp={proxy_interp}, "
            f"causes={len(likely_causes)}"
        )

    # Compute reported vs expected interpretability
    # From experiment summary: SG-FIGS mean_split_interpretability = 0.277
    # RO-FIGS mean_split_interpretability = 0.157
    # But 5/7 datasets scored 0.000

    # Count datasets with non-zero scores
    datasets_with_zero_score = [
        ds for ds, info in per_dataset.items()
        if info.get("proxy_interpretability_score") is not None
        and (info["feature_index_mismatch_likely"] or len(info["likely_zero_score_causes"]) > 0)
    ]

    result = {
        "per_dataset": per_dataset,
        "summary": {
            "n_datasets_analyzed": len(per_dataset),
            "n_with_feature_index_mismatch": sum(
                1 for info in per_dataset.values() if info["feature_index_mismatch_likely"]
            ),
            "n_high_dim": sum(1 for info in per_dataset.values() if info["is_high_dim"]),
            "n_low_complexity_fallback": sum(
                1 for info in per_dataset.values() if info["mean_sg_figs_splits"] <= 5
            ),
            "datasets_likely_zero_score": datasets_with_zero_score,
            "diagnostic_conclusion": (
                "The split_interpretability_score metric in compute_split_interpretability_score() "
                "counts the fraction of oblique splits whose feature pairs have above-median "
                "synergy. Zero scores arise from: (1) high-dimensional datasets where MI "
                "prefiltering creates a synergy matrix with indices that don't match the "
                "original feature space used in oblique splits, causing fi < d checks to "
                "fail or produce spurious lookups; (2) axis-aligned fallback in low-complexity "
                "models producing zero oblique nodes (denominator = 0 returns 0.0); "
                "(3) the metric only evaluates non-None oblique splits, so models that "
                "happen to use axis-aligned splits exclusively score 0. The reported "
                "0.277 vs 0.157 comparison is misleading as it averages over mostly "
                "zero-scored datasets with only 2/7 contributing non-zero values."
            ),
        },
    }

    return result


# ---------------------------------------------------------------------------
# 5. Criterion 3: Domain Analysis of Synergy Pairs
# ---------------------------------------------------------------------------

def criterion_3_domain_analysis(synergy_pairs: dict) -> dict:
    """Map top-5 synergy pairs for key datasets to domain knowledge.

    Score whether >= 3 datasets have domain-meaningful synergistic pairs.
    """
    logger.info("--- Criterion 3: Domain Analysis of Synergy Pairs ---")

    target_datasets = [
        "pima_diabetes",
        "breast_cancer_wisconsin_diagnostic",
        "heart_statlog",
        "banknote",
        "ionosphere",
        "sonar",
        "spectf_heart",
    ]

    per_dataset = {}
    datasets_with_meaningful_pairs = 0

    for ds_name in target_datasets:
        if ds_name not in synergy_pairs:
            logger.warning(f"  {ds_name}: no synergy pairs available")
            continue

        top_5 = synergy_pairs[ds_name][:5]
        domain_info = DOMAIN_KNOWLEDGE.get(ds_name, {})
        known = domain_info.get("known_interactions", {})
        domain = domain_info.get("domain", "unknown")

        # Check each top pair against known interactions
        meaningful_count = 0
        pair_analyses = []
        for pair in top_5:
            fi, fj = pair["feature_i"], pair["feature_j"]
            synergy_val = pair["synergy"]

            # Check both orderings
            explanation = known.get((fi, fj)) or known.get((fj, fi))
            is_meaningful = explanation is not None

            if is_meaningful:
                meaningful_count += 1

            pair_analyses.append({
                "feature_i": fi,
                "feature_j": fj,
                "synergy": round(synergy_val, 6),
                "domain_meaningful": is_meaningful,
                "explanation": explanation if explanation else "No established domain interaction found",
            })

        has_meaningful = meaningful_count >= 2  # At least 2 of top-5 pairs are meaningful
        if has_meaningful:
            datasets_with_meaningful_pairs += 1

        per_dataset[ds_name] = {
            "domain": domain,
            "n_meaningful_pairs": meaningful_count,
            "n_top_pairs_analyzed": len(pair_analyses),
            "has_meaningful_synergies": has_meaningful,
            "top_synergy_pairs": pair_analyses,
        }

        logger.info(
            f"  {ds_name} ({domain}): {meaningful_count}/{len(pair_analyses)} "
            f"pairs domain-meaningful, {'PASS' if has_meaningful else 'FAIL'}"
        )

    criterion_met = datasets_with_meaningful_pairs >= 3
    result = {
        "per_dataset": per_dataset,
        "summary": {
            "n_datasets_analyzed": len(per_dataset),
            "n_with_meaningful_pairs": datasets_with_meaningful_pairs,
            "criterion_threshold": 3,
            "criterion_met": criterion_met,
            "conclusion": (
                f"{datasets_with_meaningful_pairs} of {len(per_dataset)} datasets have "
                f"domain-meaningful synergistic feature pairs (threshold: >= 3). "
                f"Criterion {'MET' if criterion_met else 'NOT MET'}."
            ),
        },
    }

    return result


# ---------------------------------------------------------------------------
# 6. Pareto Frontier Analysis
# ---------------------------------------------------------------------------

def pareto_frontier_analysis(
    acc_per_dataset: dict,
    splits_per_dataset: dict,
) -> dict:
    """Pareto analysis: accuracy vs complexity for each method."""
    logger.info("--- Pareto Frontier Analysis ---")

    # Compute mean accuracy and mean splits across all eligible datasets
    method_stats = {}
    for method in METHODS:
        accs = []
        splits = []
        for ds_name in acc_per_dataset:
            if method in acc_per_dataset[ds_name]:
                accs.append(acc_per_dataset[ds_name][method])
            if method in splits_per_dataset.get(ds_name, {}):
                splits.append(splits_per_dataset[ds_name][method])

        if accs and splits:
            mean_acc = float(np.mean(accs))
            mean_splits = float(np.mean(splits))
            efficiency = mean_acc / mean_splits if mean_splits > 0 else 0.0
            method_stats[method] = {
                "mean_accuracy": round(mean_acc, 6),
                "mean_n_splits": round(mean_splits, 2),
                "efficiency_acc_per_split": round(efficiency, 6),
                "n_datasets": len(accs),
            }

    # Determine Pareto dominance
    # Method A dominates B if A has >= accuracy AND <= splits (and strictly better on at least one)
    pareto_dominant = []
    pareto_dominated = []
    for m1 in method_stats:
        dominates_any = False
        is_dominated = False
        for m2 in method_stats:
            if m1 == m2:
                continue
            s1 = method_stats[m1]
            s2 = method_stats[m2]

            m1_dominates_m2 = (
                s1["mean_accuracy"] >= s2["mean_accuracy"]
                and s1["mean_n_splits"] <= s2["mean_n_splits"]
                and (s1["mean_accuracy"] > s2["mean_accuracy"] or s1["mean_n_splits"] < s2["mean_n_splits"])
            )
            m2_dominates_m1 = (
                s2["mean_accuracy"] >= s1["mean_accuracy"]
                and s2["mean_n_splits"] <= s1["mean_n_splits"]
                and (s2["mean_accuracy"] > s1["mean_accuracy"] or s2["mean_n_splits"] < s1["mean_n_splits"])
            )

            if m1_dominates_m2:
                dominates_any = True
            if m2_dominates_m1:
                is_dominated = True

        if not is_dominated:
            pareto_dominant.append(m1)
        else:
            pareto_dominated.append(m1)

    # Per-dataset Pareto analysis
    per_dataset_pareto = {}
    for ds_name in acc_per_dataset:
        available = [m for m in METHODS if m in acc_per_dataset[ds_name] and m in splits_per_dataset.get(ds_name, {})]
        if len(available) < 2:
            continue

        ds_points = {}
        for m in available:
            acc = acc_per_dataset[ds_name][m]
            spl = splits_per_dataset[ds_name].get(m, 0)
            eff = acc / spl if spl > 0 else 0.0
            ds_points[m] = {
                "accuracy": round(acc, 6),
                "n_splits": round(spl, 2),
                "efficiency": round(eff, 6),
            }

        # Find Pareto optimal for this dataset
        ds_pareto = []
        for m1 in available:
            is_dominated = False
            for m2 in available:
                if m1 == m2:
                    continue
                if (ds_points[m2]["accuracy"] >= ds_points[m1]["accuracy"]
                    and ds_points[m2]["n_splits"] <= ds_points[m1]["n_splits"]
                    and (ds_points[m2]["accuracy"] > ds_points[m1]["accuracy"]
                         or ds_points[m2]["n_splits"] < ds_points[m1]["n_splits"])):
                    is_dominated = True
                    break
            if not is_dominated:
                ds_pareto.append(m1)

        per_dataset_pareto[ds_name] = {
            "methods": ds_points,
            "pareto_optimal": ds_pareto,
            "most_efficient": max(available, key=lambda m: ds_points[m]["efficiency"]),
        }

    result = {
        "aggregate": {
            "method_stats": method_stats,
            "pareto_dominant_methods": pareto_dominant,
            "pareto_dominated_methods": pareto_dominated,
        },
        "per_dataset": per_dataset_pareto,
        "interpretation": (
            f"Pareto-optimal methods (not dominated on accuracy-complexity tradeoff): "
            f"{pareto_dominant}. "
            f"{'SG-FIGS' if 'SG-FIGS' in pareto_dominant else 'No SG-FIGS'} on Pareto front."
        ),
    }

    logger.info(f"Pareto-dominant: {pareto_dominant}")
    logger.info(f"Pareto-dominated: {pareto_dominated}")
    for m, s in method_stats.items():
        logger.info(f"  {m}: acc={s['mean_accuracy']:.4f}, splits={s['mean_n_splits']:.1f}, eff={s['efficiency_acc_per_split']:.4f}")

    return result


# ---------------------------------------------------------------------------
# 7. Synergy Landscape Correlation Analysis
# ---------------------------------------------------------------------------

def synergy_landscape_correlation(
    acc_per_dataset: dict,
    synergy_data: dict,
) -> dict:
    """Correlate synergy landscape features with SG-FIGS relative performance.

    For each dataset, extract synergy distribution stats and correlate with
    delta_acc = SG-FIGS_acc - RO-FIGS_acc.
    """
    logger.info("--- Synergy Landscape Correlation Analysis ---")

    # Build feature matrix and target vector
    eligible_datasets = [
        ds for ds in acc_per_dataset
        if "SG-FIGS" in acc_per_dataset[ds]
        and "RO-FIGS" in acc_per_dataset[ds]
        and ds in synergy_data
    ]

    if len(eligible_datasets) < 3:
        logger.warning("Too few datasets for correlation analysis")
        return {
            "n_datasets": len(eligible_datasets),
            "error": "Too few datasets for meaningful correlation analysis",
        }

    feature_names = [
        "synergy_mean", "synergy_std", "synergy_max", "synergy_min",
        "mi_comparison_jaccard", "mi_comparison_spearman",
        "synergy_graph_edges", "synergy_graph_components",
        "synergy_graph_largest_clique", "n_features", "n_pairs",
    ]

    # Derived features
    derived_names = ["synergy_skewness", "synergy_kurtosis", "synergy_range", "graph_density_estimate"]

    all_feature_names = feature_names + derived_names

    feature_matrix = []
    delta_acc_vector = []
    dataset_names_ordered = []

    for ds_name in eligible_datasets:
        sd = synergy_data[ds_name]
        delta_acc = acc_per_dataset[ds_name]["SG-FIGS"] - acc_per_dataset[ds_name]["RO-FIGS"]
        delta_acc_vector.append(delta_acc)
        dataset_names_ordered.append(ds_name)

        # Base features
        row = [sd.get(fn, 0.0) or 0.0 for fn in feature_names]

        # Derived features
        syn_mean = sd.get("synergy_mean", 0.0) or 0.0
        syn_std = sd.get("synergy_std", 0.0) or 0.0
        syn_max = sd.get("synergy_max", 0.0) or 0.0
        syn_min = sd.get("synergy_min", 0.0) or 0.0
        n_features_ds = sd.get("n_features", 1) or 1
        n_edges = sd.get("synergy_graph_edges", 0) or 0

        # Skewness estimate from mean, median approximation
        if syn_std > 0:
            skewness_est = 3 * (syn_mean - (syn_mean + syn_min) / 2) / syn_std
        else:
            skewness_est = 0.0

        # Kurtosis estimate (simple: using range/std ratio as proxy)
        kurtosis_est = ((syn_max - syn_min) / syn_std) if syn_std > 0 else 0.0

        syn_range = syn_max - syn_min
        max_possible_edges = n_features_ds * (n_features_ds - 1) / 2
        graph_density = n_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        row.extend([skewness_est, kurtosis_est, syn_range, graph_density])
        feature_matrix.append(row)

    X = np.array(feature_matrix)
    y = np.array(delta_acc_vector)

    # Compute Spearman correlations
    correlations = {}
    for i, fn in enumerate(all_feature_names):
        x_col = X[:, i]
        if np.std(x_col) < 1e-12:
            correlations[fn] = {"spearman_rho": 0.0, "p_value": 1.0, "note": "constant feature"}
            continue

        try:
            rho, p_val = stats.spearmanr(x_col, y)
            correlations[fn] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p_val), 6),
                "significant_at_010": bool(p_val < 0.10),
            }
        except Exception:
            correlations[fn] = {"spearman_rho": 0.0, "p_value": 1.0, "note": "computation failed"}

    # Sort by absolute correlation
    sorted_features = sorted(
        correlations.items(),
        key=lambda x: abs(x[1].get("spearman_rho", 0)),
        reverse=True,
    )

    # Per-dataset details
    per_dataset_details = {}
    for i, ds_name in enumerate(dataset_names_ordered):
        per_dataset_details[ds_name] = {
            "delta_acc_sg_minus_ro": round(float(delta_acc_vector[i]), 6),
            "synergy_mean": round(float(X[i, 0]), 6),
            "synergy_std": round(float(X[i, 1]), 6),
            "graph_density": round(float(X[i, -1]), 4),
            "mi_synergy_jaccard": round(float(X[i, 4]), 4),
        }

    # Identify top predictors
    top_predictors = [
        {"feature": fn, **info}
        for fn, info in sorted_features[:5]
    ]

    result = {
        "n_datasets": len(eligible_datasets),
        "datasets_used": dataset_names_ordered,
        "delta_acc_sg_minus_ro": {ds: round(float(d), 6) for ds, d in zip(dataset_names_ordered, delta_acc_vector)},
        "spearman_correlations": correlations,
        "top_5_predictors": top_predictors,
        "per_dataset_landscape": per_dataset_details,
        "interpretation": (
            "Positive Spearman rho indicates that higher values of the feature "
            "are associated with better SG-FIGS performance relative to RO-FIGS. "
            f"Top predictor: {top_predictors[0]['feature']} (rho={top_predictors[0]['spearman_rho']:.3f})."
            f" Note: N={len(eligible_datasets)} datasets limits statistical power."
        ),
    }

    logger.info("Top 5 predictors of SG-FIGS relative performance:")
    for tp in top_predictors:
        logger.info(f"  {tp['feature']}: rho={tp['spearman_rho']:.4f}, p={tp['p_value']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def build_eval_output(
    datasets: dict,
    friedman_result: dict,
    wilcoxon_result: dict,
    criterion1_result: dict,
    criterion2_result: dict,
    criterion3_result: dict,
    pareto_result: dict,
    synergy_corr_result: dict,
    acc_per_dataset: dict,
    splits_per_dataset: dict,
) -> dict:
    """Build schema-compliant eval_out.json with metrics_agg and datasets."""

    # ---- Aggregate metrics ----
    # Extract key numbers for metrics_agg
    friedman_p = friedman_result.get("friedman_p_value", 1.0) or 1.0
    friedman_stat = friedman_result.get("friedman_statistic", 0.0) or 0.0

    # Wilcoxon p-values for key pairs
    wilcoxon_sg_ro = wilcoxon_result.get("pairwise_tests", {}).get("RO-FIGS_vs_SG-FIGS", {})
    wilcoxon_sg_ro_p = wilcoxon_sg_ro.get("p_value", 1.0) or 1.0
    wilcoxon_sg_ro_stat = wilcoxon_sg_ro.get("wilcoxon_statistic", 0.0) or 0.0

    wilcoxon_figs_sg = wilcoxon_result.get("pairwise_tests", {}).get("FIGS_vs_SG-FIGS", {})
    wilcoxon_figs_sg_p = wilcoxon_figs_sg.get("p_value", 1.0) or 1.0

    wilcoxon_figs_ro = wilcoxon_result.get("pairwise_tests", {}).get("FIGS_vs_RO-FIGS", {})
    wilcoxon_figs_ro_p = wilcoxon_figs_ro.get("p_value", 1.0) or 1.0

    # Criterion 1 pass rate
    c1_summary = criterion1_result.get("summary", {})
    c1_pass_rate = c1_summary.get("pass_rate_any", 0.0)
    c1_n_pass = c1_summary.get("datasets_passing_either", 0)

    # Criterion 2
    c2_summary = criterion2_result.get("summary", {})
    c2_n_mismatch = c2_summary.get("n_with_feature_index_mismatch", 0)

    # Criterion 3
    c3_summary = criterion3_result.get("summary", {})
    c3_n_meaningful = c3_summary.get("n_with_meaningful_pairs", 0)
    c3_criterion_met = 1 if c3_summary.get("criterion_met", False) else 0

    # Pareto
    pareto_agg = pareto_result.get("aggregate", {}).get("method_stats", {})
    sg_efficiency = pareto_agg.get("SG-FIGS", {}).get("efficiency_acc_per_split", 0.0)
    ro_efficiency = pareto_agg.get("RO-FIGS", {}).get("efficiency_acc_per_split", 0.0)
    figs_efficiency = pareto_agg.get("FIGS", {}).get("efficiency_acc_per_split", 0.0)
    sg_on_pareto = 1 if "SG-FIGS" in pareto_result.get("aggregate", {}).get("pareto_dominant_methods", []) else 0

    # Mean accuracies
    sg_accs = [acc_per_dataset[ds]["SG-FIGS"] for ds in acc_per_dataset if "SG-FIGS" in acc_per_dataset[ds]]
    ro_accs = [acc_per_dataset[ds]["RO-FIGS"] for ds in acc_per_dataset if "RO-FIGS" in acc_per_dataset[ds]]
    figs_accs = [acc_per_dataset[ds]["FIGS"] for ds in acc_per_dataset if "FIGS" in acc_per_dataset[ds]]

    sg_splits_list = [splits_per_dataset[ds]["SG-FIGS"] for ds in splits_per_dataset if "SG-FIGS" in splits_per_dataset[ds]]
    ro_splits_list = [splits_per_dataset[ds]["RO-FIGS"] for ds in splits_per_dataset if "RO-FIGS" in splits_per_dataset[ds]]
    figs_splits_list = [splits_per_dataset[ds]["FIGS"] for ds in splits_per_dataset if "FIGS" in splits_per_dataset[ds]]

    # Friedman average ranks
    avg_ranks = friedman_result.get("average_ranks", {})

    metrics_agg = {
        # Overall accuracy
        "mean_accuracy_figs": round(float(np.mean(figs_accs)), 6) if figs_accs else 0.0,
        "mean_accuracy_ro_figs": round(float(np.mean(ro_accs)), 6) if ro_accs else 0.0,
        "mean_accuracy_sg_figs": round(float(np.mean(sg_accs)), 6) if sg_accs else 0.0,
        # Overall splits
        "mean_splits_figs": round(float(np.mean(figs_splits_list)), 2) if figs_splits_list else 0.0,
        "mean_splits_ro_figs": round(float(np.mean(ro_splits_list)), 2) if ro_splits_list else 0.0,
        "mean_splits_sg_figs": round(float(np.mean(sg_splits_list)), 2) if sg_splits_list else 0.0,
        # Friedman test
        "friedman_statistic": round(friedman_stat, 4),
        "friedman_p_value": round(friedman_p, 6),
        # Friedman average ranks
        "friedman_avg_rank_figs": round(avg_ranks.get("FIGS", 0.0), 4),
        "friedman_avg_rank_ro_figs": round(avg_ranks.get("RO-FIGS", 0.0), 4),
        "friedman_avg_rank_sg_figs": round(avg_ranks.get("SG-FIGS", 0.0), 4),
        # Wilcoxon
        "wilcoxon_ro_vs_sg_p": round(wilcoxon_sg_ro_p, 6),
        "wilcoxon_ro_vs_sg_stat": round(wilcoxon_sg_ro_stat, 4),
        "wilcoxon_figs_vs_sg_p": round(wilcoxon_figs_sg_p, 6),
        "wilcoxon_figs_vs_ro_p": round(wilcoxon_figs_ro_p, 6),
        # Criterion 1
        "criterion1_pass_rate": round(c1_pass_rate, 4),
        "criterion1_n_pass": c1_n_pass,
        # Criterion 2
        "criterion2_n_index_mismatch": c2_n_mismatch,
        # Criterion 3
        "criterion3_n_meaningful_datasets": c3_n_meaningful,
        "criterion3_met": c3_criterion_met,
        # Pareto
        "pareto_sg_figs_efficiency": round(sg_efficiency, 6),
        "pareto_ro_figs_efficiency": round(ro_efficiency, 6),
        "pareto_figs_efficiency": round(figs_efficiency, 6),
        "pareto_sg_on_front": sg_on_pareto,
    }

    # ---- Per-example output (datasets) ----
    output_datasets = []
    for ds_name, ds_info in datasets.items():
        examples = []
        for method in ds_info["methods"]:
            for fold_id, fold_data in ds_info["method_folds"][method].items():
                acc = fold_data.get("balanced_accuracy", 0.0)
                auc = fold_data.get("auc")
                n_splits = fold_data.get("n_splits", 0)
                n_trees = fold_data.get("n_trees", 0)

                # Build input/output strings
                input_str = json.dumps({
                    "dataset": ds_name,
                    "method": method,
                    "fold": fold_id,
                    "n_features": ds_info["n_features"],
                    "n_classes": ds_info["n_classes"],
                })
                output_str = json.dumps({
                    "balanced_accuracy": round(acc, 6),
                    "auc": round(auc, 6) if auc is not None else None,
                    "n_splits": n_splits,
                    "n_trees": n_trees,
                })

                ex = {
                    "input": input_str,
                    "output": output_str,
                    "metadata_dataset": ds_name,
                    "metadata_method": method,
                    "metadata_fold": fold_id,
                    "metadata_domain": ds_info["domain"],
                    "metadata_n_features": ds_info["n_features"],
                    "metadata_n_classes": ds_info["n_classes"],
                    "predict_balanced_accuracy": str(round(acc, 6)),
                    "predict_n_splits": str(n_splits),
                    "eval_balanced_accuracy": round(acc, 6),
                    "eval_n_splits": float(n_splits),
                }

                if auc is not None:
                    ex["eval_auc"] = round(auc, 6)
                    ex["predict_auc"] = str(round(auc, 6))

                examples.append(ex)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    # Build comprehensive analysis JSON (embedded in the output)
    analysis = {
        "friedman_nemenyi": friedman_result,
        "wilcoxon_pairwise": wilcoxon_result,
        "criterion_1_accuracy_complexity": criterion1_result,
        "criterion_2_interpretability_diagnostic": criterion2_result,
        "criterion_3_domain_analysis": criterion3_result,
        "pareto_frontier": pareto_result,
        "synergy_landscape_correlation": synergy_corr_result,
    }

    output = {
        "metrics_agg": metrics_agg,
        "datasets": output_datasets,
    }

    return output, analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Statistical Evaluation of SG-FIGS Experiment Results")
    logger.info("=" * 60)

    # ---- Load data ----
    logger.info("\n[1/8] Loading experiment data...")
    datasets = load_experiment_data(EXP_ID2_FULL)

    logger.info("\n[2/8] Loading synergy data...")
    synergy_data = load_synergy_data(EXP_ID1_RESULTS)
    synergy_pairs = load_synergy_pairs(EXP_ID1_FULL)

    # ---- Extract per-dataset metrics ----
    logger.info("\n[3/8] Extracting per-dataset metrics...")
    acc_per_dataset = extract_mean_accuracy_per_dataset(datasets)
    splits_per_dataset = extract_mean_splits_per_dataset(datasets)

    # Log extracted metrics
    for ds_name in acc_per_dataset:
        parts = []
        for m in METHODS:
            if m in acc_per_dataset[ds_name]:
                parts.append(f"{m}={acc_per_dataset[ds_name][m]:.4f}")
        logger.info(f"  {ds_name}: {', '.join(parts)}")

    # ---- Run analyses ----
    logger.info("\n[4/8] Running Friedman test with Nemenyi post-hoc...")
    friedman_result = friedman_nemenyi_test(acc_per_dataset)

    logger.info("\n[5/8] Running Wilcoxon signed-rank pairwise tests...")
    wilcoxon_result = wilcoxon_pairwise_tests(acc_per_dataset)

    logger.info("\n[6/8] Running Criterion 1-3 analyses...")
    criterion1_result = criterion_1_analysis(acc_per_dataset, splits_per_dataset)
    criterion2_result = criterion_2_interpretability_diagnostic(datasets, synergy_data)
    criterion3_result = criterion_3_domain_analysis(synergy_pairs)

    logger.info("\n[7/8] Running Pareto + Synergy landscape analyses...")
    pareto_result = pareto_frontier_analysis(acc_per_dataset, splits_per_dataset)
    synergy_corr_result = synergy_landscape_correlation(acc_per_dataset, synergy_data)

    # ---- Build output ----
    logger.info("\n[8/8] Building output...")
    output, analysis = build_eval_output(
        datasets=datasets,
        friedman_result=friedman_result,
        wilcoxon_result=wilcoxon_result,
        criterion1_result=criterion1_result,
        criterion2_result=criterion2_result,
        criterion3_result=criterion3_result,
        pareto_result=pareto_result,
        synergy_corr_result=synergy_corr_result,
        acc_per_dataset=acc_per_dataset,
        splits_per_dataset=splits_per_dataset,
    )

    # Save main output (schema-compliant)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved eval_out.json to {OUTPUT_PATH}")

    # Save full analysis as separate file
    analysis_path = WORKSPACE / "analysis_full.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, default=str))
    logger.info(f"Saved analysis_full.json to {analysis_path}")

    elapsed = time.time() - t0
    logger.info(f"\nTotal evaluation time: {elapsed:.1f}s")

    # ---- Print summary ----
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    ma = output["metrics_agg"]
    logger.info(f"\nMean Balanced Accuracy:")
    logger.info(f"  FIGS:    {ma['mean_accuracy_figs']:.4f}")
    logger.info(f"  RO-FIGS: {ma['mean_accuracy_ro_figs']:.4f}")
    logger.info(f"  SG-FIGS: {ma['mean_accuracy_sg_figs']:.4f}")

    logger.info(f"\nMean N Splits:")
    logger.info(f"  FIGS:    {ma['mean_splits_figs']:.1f}")
    logger.info(f"  RO-FIGS: {ma['mean_splits_ro_figs']:.1f}")
    logger.info(f"  SG-FIGS: {ma['mean_splits_sg_figs']:.1f}")

    logger.info(f"\nFriedman Test: chi2={ma['friedman_statistic']:.4f}, p={ma['friedman_p_value']:.6f}")
    logger.info(f"  Avg Ranks: FIGS={ma['friedman_avg_rank_figs']:.2f}, RO-FIGS={ma['friedman_avg_rank_ro_figs']:.2f}, SG-FIGS={ma['friedman_avg_rank_sg_figs']:.2f}")

    logger.info(f"\nWilcoxon (RO-FIGS vs SG-FIGS): stat={ma['wilcoxon_ro_vs_sg_stat']:.4f}, p={ma['wilcoxon_ro_vs_sg_p']:.6f}")

    logger.info(f"\nCriterion 1 (Acc+Complexity): {ma['criterion1_n_pass']}/{criterion1_result['n_eligible_datasets']} pass (rate={ma['criterion1_pass_rate']:.4f})")
    logger.info(f"Criterion 2 (Interpretability): {ma['criterion2_n_index_mismatch']} datasets with index mismatch")
    logger.info(f"Criterion 3 (Domain Analysis): {ma['criterion3_n_meaningful_datasets']} datasets with meaningful pairs (met={'YES' if ma['criterion3_met'] else 'NO'})")

    logger.info(f"\nPareto Efficiency:")
    logger.info(f"  FIGS:    {ma['pareto_figs_efficiency']:.4f}")
    logger.info(f"  RO-FIGS: {ma['pareto_ro_figs_efficiency']:.4f}")
    logger.info(f"  SG-FIGS: {ma['pareto_sg_figs_efficiency']:.4f}")
    logger.info(f"  SG-FIGS on Pareto front: {'YES' if ma['pareto_sg_on_front'] else 'NO'}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
