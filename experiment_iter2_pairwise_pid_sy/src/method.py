#!/usr/bin/env python3
"""Pairwise PID Synergy Matrices on Benchmark Datasets.

Computes pairwise PID synergy matrices on all benchmark datasets from two
dependency workspaces, measures timing, compares synergy vs MI rankings,
assesses stability, and constructs synergy graphs to validate assumptions
1-3 of the SG-FIGS hypothesis.

Method: PID_BROJA (small datasets) with PID_MMI fallback (large datasets).
Baseline: Co-Information (interaction information) as synergy proxy.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import time
import resource
import numpy as np
from collections import Counter
from itertools import combinations
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))  # ~58 min CPU time

# ── Logging ──────────────────────────────────────────────────────────────────
BLUE, GREEN, YELLOW, CYAN, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[0m"
WORKSPACE = Path(__file__).parent.resolve()
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ── Dependency paths ─────────────────────────────────────────────────────────
DATA_ID2_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
)
DATA_ID3_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus/full_data_out.json"
)

# ── Configuration ────────────────────────────────────────────────────────────
N_BINS = 5               # Discretization bins
BROJA_PAIR_LIMIT = 100   # Max pairs for BROJA (else fallback to MMI)
STABILITY_SUBSAMPLES = 5  # Number of subsamples for stability analysis
STABILITY_FRACTION = 0.8  # Subsample fraction
SYNERGY_THRESHOLD_PCTL = 75  # Percentile for synergy graph threshold
MAX_FEATURES_LARGE = 30   # Subsample features for datasets with >30 features


def load_all_datasets() -> dict[str, dict[str, Any]]:
    """Load and de-duplicate datasets from both dependency workspaces.

    Returns dict: dataset_name -> {X, y, feature_names, n_classes, n_samples, source}
    """
    datasets: dict[str, dict[str, Any]] = {}

    for path, source_id in [(DATA_ID2_PATH, "data_id2"), (DATA_ID3_PATH, "data_id3")]:
        logger.info(f"{BLUE}Loading datasets from {source_id}{END}")
        raw = json.loads(path.read_text())

        for ds_entry in raw["datasets"]:
            ds_name = ds_entry["dataset"]
            # Prefer data_id2 version (loaded first) for de-duplication
            if ds_name in datasets:
                logger.info(f"  Skipping duplicate: {ds_name} (already from {datasets[ds_name]['source']})")
                continue

            examples = ds_entry["examples"]
            if len(examples) == 0:
                logger.warning(f"  Empty dataset: {ds_name}, skipping")
                continue

            # Parse features
            first_input = json.loads(examples[0]["input"])
            feature_names = list(first_input.keys())
            n_features = len(feature_names)

            # Build X and y arrays
            X_rows = []
            y_vals = []
            for ex in examples:
                inp = json.loads(ex["input"])
                row = [float(inp[fn]) for fn in feature_names]
                X_rows.append(row)
                y_vals.append(str(ex["output"]))

            X = np.array(X_rows, dtype=np.float64)
            # Encode y as integers
            unique_classes = sorted(set(y_vals))
            class_map = {c: i for i, c in enumerate(unique_classes)}
            y = np.array([class_map[v] for v in y_vals], dtype=np.int32)

            n_classes = int(examples[0].get("metadata_n_classes", len(unique_classes)))
            n_samples = X.shape[0]

            datasets[ds_name] = {
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "n_classes": n_classes,
                "n_samples": n_samples,
                "n_features": n_features,
                "source": source_id,
                "class_labels": unique_classes,
            }
            n_pairs = n_features * (n_features - 1) // 2
            logger.info(
                f"  {ds_name:45s} | {n_samples:5d} samples | "
                f"{n_features:3d} features | {n_classes} classes | {n_pairs} pairs"
            )

    logger.info(f"{GREEN}Loaded {len(datasets)} unique datasets{END}")
    return datasets


def discretize(X: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile-based discretization of continuous features."""
    from sklearn.preprocessing import KBinsDiscretizer

    # Handle constant columns: KBinsDiscretizer fails on them
    X_disc = np.zeros_like(X, dtype=np.int32)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        n_unique = len(np.unique(col_data))
        if n_unique <= 1:
            X_disc[:, col] = 0
        else:
            actual_bins = min(n_bins, n_unique)
            disc = KBinsDiscretizer(
                n_bins=actual_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
            )
            X_disc[:, col] = disc.fit_transform(col_data.reshape(-1, 1)).ravel().astype(np.int32)
    return X_disc


def build_trivariate_dist(xi: np.ndarray, xj: np.ndarray, y: np.ndarray):
    """Build a dit.Distribution from three discrete arrays."""
    import dit

    counts = Counter(zip(xi.tolist(), xj.tolist(), y.tolist()))
    total = sum(counts.values())
    outcomes = []
    probs = []
    for (a, b, c), count in counts.items():
        outcomes.append(f"{a}{b}{c}")
        probs.append(count / total)

    # Need to handle multi-digit values by using tuples
    # If any value >= 10, use tuple-based outcomes
    max_val = max(
        max(xi), max(xj), max(y)
    )
    if max_val >= 10:
        outcomes_tuples = []
        probs_clean = []
        for (a, b, c), count in counts.items():
            outcomes_tuples.append((a, b, c))
            probs_clean.append(count / total)
        d = dit.Distribution(outcomes_tuples, probs_clean)
    else:
        d = dit.Distribution(outcomes, probs)
    return d


def compute_pid_synergy_broja(xi: np.ndarray, xj: np.ndarray, y: np.ndarray) -> float:
    """Compute PID synergy using BROJA measure."""
    from dit.pid import PID_BROJA
    d = build_trivariate_dist(xi, xj, y)
    result = PID_BROJA(d)
    return float(result.get_pi(((0, 1),)))


def compute_pid_synergy_mmi(xi: np.ndarray, xj: np.ndarray, y: np.ndarray) -> float:
    """Compute PID synergy using MMI measure (fast fallback)."""
    from dit.pid import PID_MMI
    d = build_trivariate_dist(xi, xj, y)
    result = PID_MMI(d)
    return float(result.get_pi(((0, 1),)))


def compute_full_pid(xi: np.ndarray, xj: np.ndarray, y: np.ndarray, use_broja: bool = True) -> dict:
    """Compute full PID decomposition and return all atoms."""
    if use_broja:
        from dit.pid import PID_BROJA as PID_Cls
    else:
        from dit.pid import PID_MMI as PID_Cls

    d = build_trivariate_dist(xi, xj, y)
    result = PID_Cls(d)

    return {
        "synergy": float(result.get_pi(((0, 1),))),
        "unique_0": float(result.get_pi(((0,),))),
        "unique_1": float(result.get_pi(((1,),))),
        "redundancy": float(result.get_pi(((0,), (1,)))),
    }


def compute_co_information(xi: np.ndarray, xj: np.ndarray, y: np.ndarray) -> float:
    """Compute co-information (interaction information) as synergy proxy.

    CoI(Fi, Fj; Y) = MI(Fi;Y) + MI(Fj;Y) - MI({Fi,Fj};Y)
    Negative CoI indicates synergy. We return -CoI so positive = synergy.
    """
    from sklearn.feature_selection import mutual_info_classif

    n = len(y)
    # MI(Fi; Y)
    mi_i = mutual_info_classif(
        xi.reshape(-1, 1), y, discrete_features=True, random_state=42
    )[0]
    # MI(Fj; Y)
    mi_j = mutual_info_classif(
        xj.reshape(-1, 1), y, discrete_features=True, random_state=42
    )[0]
    # MI({Fi,Fj}; Y)
    X_pair = np.column_stack([xi, xj])
    mi_pair = mutual_info_classif(
        X_pair, y, discrete_features=True, random_state=42
    )[0]

    co_info = mi_i + mi_j - mi_pair
    # Negative co-info = synergy, so return -co_info
    return -co_info


def compute_synergy_matrix(
    X_disc: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    dataset_name: str,
    feature_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Compute pairwise PID synergy matrix for a dataset.

    Returns dict with synergy_matrix, timing info, pid_method used, etc.
    """
    if feature_indices is None:
        feature_indices = list(range(X_disc.shape[1]))

    n_feat = len(feature_indices)
    n_pairs = n_feat * (n_feat - 1) // 2
    synergy_matrix = np.zeros((n_feat, n_feat), dtype=np.float64)
    coi_matrix = np.zeros((n_feat, n_feat), dtype=np.float64)  # Baseline
    pid_details = {}

    # Decide PID method based on pair count
    use_broja = n_pairs <= BROJA_PAIR_LIMIT

    # Time estimation: compute first 3 pairs
    pairs = list(combinations(range(n_feat), 2))
    test_pairs = pairs[:3]
    test_times = []

    for (li, lj) in test_pairs:
        gi, gj = feature_indices[li], feature_indices[lj]
        t0 = time.time()
        try:
            if use_broja:
                compute_pid_synergy_broja(X_disc[:, gi], X_disc[:, gj], y)
            else:
                compute_pid_synergy_mmi(X_disc[:, gi], X_disc[:, gj], y)
        except Exception:
            pass
        test_times.append(time.time() - t0)

    avg_time = np.mean(test_times) if test_times else 1.0
    estimated_total = avg_time * n_pairs
    pid_method = "BROJA" if use_broja else "MMI"

    # If BROJA estimate > 600s, switch to MMI
    if use_broja and estimated_total > 600:
        logger.warning(
            f"{YELLOW}{dataset_name}: BROJA estimated {estimated_total:.0f}s "
            f"for {n_pairs} pairs, switching to MMI{END}"
        )
        use_broja = False
        pid_method = "MMI (fallback from BROJA)"

    logger.info(
        f"  {dataset_name}: {n_pairs} pairs, method={pid_method}, "
        f"est_time={avg_time*n_pairs:.1f}s"
    )

    t_start = time.time()
    completed = 0
    errors = 0
    log_interval = max(1, n_pairs // 10)

    for idx, (li, lj) in enumerate(pairs):
        gi, gj = feature_indices[li], feature_indices[lj]
        xi = X_disc[:, gi]
        xj = X_disc[:, gj]

        # PID synergy
        try:
            if use_broja:
                pid = compute_full_pid(xi, xj, y, use_broja=True)
            else:
                pid = compute_full_pid(xi, xj, y, use_broja=False)

            synergy_matrix[li, lj] = pid["synergy"]
            synergy_matrix[lj, li] = pid["synergy"]

            pair_key = f"{feature_names[gi]}__x__{feature_names[gj]}"
            pid_details[pair_key] = pid
            completed += 1
        except Exception as e:
            logger.debug(f"  PID error on pair ({gi},{gj}): {e}")
            errors += 1

        # Co-Information baseline
        try:
            coi_val = compute_co_information(xi, xj, y)
            coi_matrix[li, lj] = coi_val
            coi_matrix[lj, li] = coi_val
        except Exception as e:
            logger.debug(f"  CoI error on pair ({gi},{gj}): {e}")

        if (idx + 1) % log_interval == 0 or idx == n_pairs - 1:
            elapsed = time.time() - t_start
            logger.debug(
                f"  {dataset_name}: {idx+1}/{n_pairs} pairs done "
                f"({elapsed:.1f}s elapsed)"
            )

    total_time = time.time() - t_start
    logger.info(
        f"  {GREEN}{dataset_name}: {completed}/{n_pairs} pairs computed "
        f"in {total_time:.1f}s ({errors} errors){END}"
    )

    return {
        "synergy_matrix": synergy_matrix.tolist(),
        "coi_matrix": coi_matrix.tolist(),
        "pid_method": pid_method,
        "n_pairs": n_pairs,
        "completed_pairs": completed,
        "errors": errors,
        "total_time_s": round(total_time, 2),
        "avg_time_per_pair_s": round(total_time / max(completed, 1), 4),
        "pid_details": pid_details,
        "feature_indices_used": feature_indices,
    }


def compute_mi_ranking(X_disc: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute MI between each feature and the target."""
    from sklearn.feature_selection import mutual_info_classif

    mi_values = mutual_info_classif(
        X_disc, y, discrete_features=True, random_state=42
    )
    return mi_values


def compare_synergy_vs_mi(
    synergy_matrix: np.ndarray,
    mi_values: np.ndarray,
    feature_names: list[str],
    feature_indices: list[int],
    top_k: int = 10,
) -> dict:
    """Compare top synergy pairs vs pairs formed from top MI features."""
    from scipy.stats import spearmanr

    n_feat = len(feature_indices)
    pairs = list(combinations(range(n_feat), 2))

    # Get synergy values for all pairs
    synergy_vals = []
    for (i, j) in pairs:
        synergy_vals.append(synergy_matrix[i, j])
    synergy_vals = np.array(synergy_vals)

    # Top-k synergy pairs
    k = min(top_k, len(pairs))
    top_synergy_idx = np.argsort(synergy_vals)[-k:]
    top_synergy_pairs = set()
    for idx in top_synergy_idx:
        i, j = pairs[idx]
        gi, gj = feature_indices[i], feature_indices[j]
        top_synergy_pairs.add((gi, gj))

    # Top-k MI features → pairs among them
    mi_sub = mi_values[feature_indices]
    top_mi_feat = np.argsort(mi_sub)[-k:]
    top_mi_pairs = set()
    for i, j in combinations(sorted(top_mi_feat), 2):
        gi, gj = feature_indices[i], feature_indices[j]
        top_mi_pairs.add((gi, gj))

    # Jaccard overlap
    intersection = top_synergy_pairs & top_mi_pairs
    union = top_synergy_pairs | top_mi_pairs
    jaccard = len(intersection) / max(len(union), 1)

    # Spearman correlation between synergy rank and MI rank of features
    # For each feature, get its max synergy with any other feature
    max_synergy_per_feat = np.max(synergy_matrix, axis=1)
    # Correlate with MI values for these features
    mi_sub_arr = mi_values[feature_indices]
    if len(mi_sub_arr) > 2:
        rho, pval = spearmanr(max_synergy_per_feat, mi_sub_arr)
    else:
        rho, pval = 0.0, 1.0

    # Top synergy pairs with names
    top_synergy_named = []
    for idx in reversed(np.argsort(synergy_vals)[-min(5, len(synergy_vals)):]):
        i, j = pairs[idx]
        gi, gj = feature_indices[i], feature_indices[j]
        top_synergy_named.append({
            "feature_i": feature_names[gi],
            "feature_j": feature_names[gj],
            "synergy": round(float(synergy_vals[idx]), 6),
        })

    return {
        "jaccard_overlap": round(jaccard, 4),
        "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else 0.0,
        "spearman_pval": round(float(pval), 6) if not np.isnan(pval) else 1.0,
        "top_synergy_pairs": top_synergy_named,
        "n_top_synergy": len(top_synergy_pairs),
        "n_top_mi_pairs": len(top_mi_pairs),
        "n_overlap": len(intersection),
    }


def stability_analysis(
    X_disc: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    dataset_name: str,
    feature_indices: list[int],
    n_subsamples: int = STABILITY_SUBSAMPLES,
    subsample_frac: float = STABILITY_FRACTION,
) -> dict:
    """Compute synergy matrices on random subsamples and assess stability."""
    from scipy.stats import spearmanr

    n_feat = len(feature_indices)
    n_pairs = n_feat * (n_feat - 1) // 2
    n_samples = X_disc.shape[0]
    subsample_size = int(n_samples * subsample_frac)

    rng = np.random.RandomState(42)
    upper_triangles = []

    logger.info(
        f"  {BLUE}Stability analysis for {dataset_name}: "
        f"{n_subsamples} subsamples of {subsample_size}/{n_samples}{END}"
    )

    for s in range(n_subsamples):
        indices = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X_disc[indices]
        y_sub = y[indices]

        # Compute synergy matrix on subsample (always use MMI for speed)
        syn_mat = np.zeros((n_feat, n_feat))
        pairs = list(combinations(range(n_feat), 2))
        for (li, lj) in pairs:
            gi, gj = feature_indices[li], feature_indices[lj]
            try:
                syn = compute_pid_synergy_mmi(X_sub[:, gi], X_sub[:, gj], y_sub)
                syn_mat[li, lj] = syn
                syn_mat[lj, li] = syn
            except Exception:
                pass

        # Extract upper triangle
        ut = []
        for (li, lj) in pairs:
            ut.append(syn_mat[li, lj])
        upper_triangles.append(ut)

        logger.debug(f"  Subsample {s+1}/{n_subsamples} done")

    # Pairwise Spearman correlations between subsamples
    correlations = []
    for i in range(n_subsamples):
        for j in range(i + 1, n_subsamples):
            if len(upper_triangles[i]) > 2:
                rho, _ = spearmanr(upper_triangles[i], upper_triangles[j])
                if not np.isnan(rho):
                    correlations.append(rho)

    mean_corr = float(np.mean(correlations)) if correlations else 0.0
    std_corr = float(np.std(correlations)) if correlations else 0.0

    logger.info(
        f"  {GREEN}Stability {dataset_name}: "
        f"mean_rho={mean_corr:.4f} +/- {std_corr:.4f}{END}"
    )

    return {
        "n_subsamples": n_subsamples,
        "subsample_fraction": subsample_frac,
        "subsample_size": subsample_size,
        "n_pairs": n_pairs,
        "mean_spearman": round(mean_corr, 4),
        "std_spearman": round(std_corr, 4),
        "all_correlations": [round(c, 4) for c in correlations],
    }


def build_synergy_graph(
    synergy_matrix: np.ndarray,
    feature_names: list[str],
    feature_indices: list[int],
    threshold_percentile: int = SYNERGY_THRESHOLD_PCTL,
) -> dict:
    """Construct synergy graph from thresholded synergy matrix."""
    import networkx as nx

    n_feat = len(feature_indices)
    pairs = list(combinations(range(n_feat), 2))

    # Get all synergy values
    syn_vals = [synergy_matrix[i, j] for (i, j) in pairs]
    if not syn_vals or max(syn_vals) == 0:
        return {
            "threshold": 0.0,
            "n_edges": 0,
            "n_nodes": n_feat,
            "n_components": n_feat,
            "largest_clique_size": 0,
            "top_5_edges": [],
        }

    threshold = float(np.percentile(syn_vals, threshold_percentile))

    G = nx.Graph()
    for idx in range(n_feat):
        gi = feature_indices[idx]
        G.add_node(feature_names[gi])

    for (li, lj) in pairs:
        if synergy_matrix[li, lj] >= threshold:
            gi, gj = feature_indices[li], feature_indices[lj]
            G.add_edge(
                feature_names[gi],
                feature_names[gj],
                weight=synergy_matrix[li, lj],
            )

    n_components = nx.number_connected_components(G)

    # Largest clique
    try:
        cliques = list(nx.find_cliques(G))
        largest_clique_size = max(len(c) for c in cliques) if cliques else 0
    except Exception:
        largest_clique_size = 0

    # Top-5 edges by synergy
    edges_sorted = sorted(
        G.edges(data=True), key=lambda e: e[2].get("weight", 0), reverse=True
    )
    top_5 = [
        {
            "feature_i": e[0],
            "feature_j": e[1],
            "synergy": round(e[2]["weight"], 6),
        }
        for e in edges_sorted[:5]
    ]

    return {
        "threshold": round(threshold, 6),
        "n_edges": G.number_of_edges(),
        "n_nodes": G.number_of_nodes(),
        "n_components": n_components,
        "largest_clique_size": largest_clique_size,
        "top_5_edges": top_5,
    }


def xor_validation() -> dict:
    """Validate PID computation on XOR distribution."""
    import dit
    from dit.pid import PID_BROJA, PID_MMI

    logger.info(f"{BLUE}Phase 2: XOR Validation{END}")

    # XOR: Y = X1 XOR X2
    xor_dist = dit.Distribution(
        ["000", "001", "010", "011", "100", "101", "110", "111"],
        [1 / 4, 0, 0, 1 / 4, 0, 1 / 4, 1 / 4, 0],
    )
    r_broja = PID_BROJA(xor_dist)
    broja_syn = r_broja.get_pi(((0, 1),))
    broja_red = r_broja.get_pi(((0,), (1,)))
    broja_u0 = r_broja.get_pi(((0,),))
    broja_u1 = r_broja.get_pi(((1,),))

    r_mmi = PID_MMI(xor_dist)
    mmi_syn = r_mmi.get_pi(((0, 1),))
    mmi_red = r_mmi.get_pi(((0,), (1,)))

    # AND: Y = X1 AND X2 (redundancy test)
    and_dist = dit.Distribution(
        ["000", "010", "100", "111"],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    )
    r_and = PID_BROJA(and_dist)
    and_syn = r_and.get_pi(((0, 1),))
    and_red = r_and.get_pi(((0,), (1,)))

    # Conservation check: Synergy + Redundancy + Unique0 + Unique1 = MI
    total_pi = broja_syn + broja_red + broja_u0 + broja_u1

    results = {
        "xor_broja_synergy": round(float(broja_syn), 6),
        "xor_broja_redundancy": round(float(broja_red), 6),
        "xor_broja_unique_0": round(float(broja_u0), 6),
        "xor_broja_unique_1": round(float(broja_u1), 6),
        "xor_mmi_synergy": round(float(mmi_syn), 6),
        "xor_mmi_redundancy": round(float(mmi_red), 6),
        "xor_conservation_total": round(float(total_pi), 6),
        "and_broja_synergy": round(float(and_syn), 6),
        "and_broja_redundancy": round(float(and_red), 6),
        "xor_synergy_pass": bool(abs(broja_syn - 1.0) < 0.01),
        "xor_redundancy_pass": bool(abs(broja_red) < 0.01),
        "conservation_pass": bool(abs(total_pi - 1.0) < 0.05),
    }

    logger.info(
        f"  XOR synergy (BROJA): {broja_syn:.4f} (expect ~1.0) "
        f"{'PASS' if results['xor_synergy_pass'] else 'FAIL'}"
    )
    logger.info(
        f"  XOR redundancy: {broja_red:.4f} (expect ~0.0) "
        f"{'PASS' if results['xor_redundancy_pass'] else 'FAIL'}"
    )
    logger.info(
        f"  Conservation: {total_pi:.4f} (expect ~1.0) "
        f"{'PASS' if results['conservation_pass'] else 'FAIL'}"
    )
    logger.info(
        f"  AND synergy: {and_syn:.4f}, AND redundancy: {and_red:.4f}"
    )

    return results


def process_dataset(
    ds_name: str,
    ds_info: dict,
    do_stability: bool = False,
) -> dict:
    """Process a single dataset: discretize, compute synergy, MI, graph."""
    X = ds_info["X"]
    y = ds_info["y"]
    feature_names = ds_info["feature_names"]
    n_features = ds_info["n_features"]

    # Subsample features for very large datasets
    feature_indices = list(range(n_features))
    if n_features > MAX_FEATURES_LARGE:
        logger.info(
            f"  {YELLOW}{ds_name}: {n_features} features > {MAX_FEATURES_LARGE}, "
            f"subsampling to top {MAX_FEATURES_LARGE} by MI{END}"
        )
        mi_all = compute_mi_ranking(
            discretize(X, n_bins=N_BINS), y
        )
        top_feat_idx = np.argsort(mi_all)[-MAX_FEATURES_LARGE:]
        feature_indices = sorted(top_feat_idx.tolist())

    # Phase 3: Discretize
    X_disc = discretize(X, n_bins=N_BINS)

    # Phase 4: Synergy matrix
    synergy_result = compute_synergy_matrix(
        X_disc=X_disc,
        y=y,
        feature_names=feature_names,
        dataset_name=ds_name,
        feature_indices=feature_indices,
    )

    # Phase 5: MI comparison
    mi_values = compute_mi_ranking(X_disc, y)
    mi_comparison = compare_synergy_vs_mi(
        synergy_matrix=np.array(synergy_result["synergy_matrix"]),
        mi_values=mi_values,
        feature_names=feature_names,
        feature_indices=feature_indices,
    )

    # Phase 7: Synergy graph
    graph_result = build_synergy_graph(
        synergy_matrix=np.array(synergy_result["synergy_matrix"]),
        feature_names=feature_names,
        feature_indices=feature_indices,
    )

    # Phase 6: Stability (if requested)
    stability_result = None
    if do_stability:
        stability_result = stability_analysis(
            X_disc=X_disc,
            y=y,
            feature_names=feature_names,
            dataset_name=ds_name,
            feature_indices=feature_indices,
        )

    # MI values for features used
    mi_dict = {}
    for idx in feature_indices:
        mi_dict[feature_names[idx]] = round(float(mi_values[idx]), 6)

    return {
        "dataset": ds_name,
        "n_samples": ds_info["n_samples"],
        "n_features": n_features,
        "n_features_used": len(feature_indices),
        "n_classes": ds_info["n_classes"],
        "source": ds_info["source"],
        "synergy": synergy_result,
        "mi_values": mi_dict,
        "mi_comparison": mi_comparison,
        "synergy_graph": graph_result,
        "stability": stability_result,
    }


def build_output_examples(all_results: list[dict]) -> list[dict]:
    """Build output in exp_gen_sol_out schema format.

    Schema: {"datasets": [{"dataset": name, "examples": [...]}]}
    Each example: {"input": str, "output": str, "metadata_*": ..., "predict_*": str}
    """
    output_datasets = []

    for ds_result in all_results:
        ds_name = ds_result["dataset"]
        examples = []

        synergy_data = ds_result["synergy"]
        pid_details = synergy_data["pid_details"]

        for pair_key, pid in pid_details.items():
            parts = pair_key.split("__x__")
            feat_i = parts[0] if len(parts) == 2 else pair_key
            feat_j = parts[1] if len(parts) == 2 else ""

            input_str = json.dumps({
                "feature_i": feat_i,
                "feature_j": feat_j,
                "dataset": ds_name,
                "n_samples": ds_result["n_samples"],
                "n_classes": ds_result["n_classes"],
            })

            output_str = json.dumps({
                "synergy": round(pid["synergy"], 6),
                "unique_0": round(pid["unique_0"], 6),
                "unique_1": round(pid["unique_1"], 6),
                "redundancy": round(pid["redundancy"], 6),
            })

            # Co-information baseline for this pair
            coi_matrix = np.array(synergy_data["coi_matrix"])
            feat_indices = synergy_data["feature_indices_used"]
            all_feat_names = list(ds_result["mi_values"].keys())

            # Find local indices
            try:
                li = next(
                    k for k, gi in enumerate(feat_indices)
                    if all_feat_names[k] == feat_i or (gi < len(all_feat_names) and False)
                )
            except StopIteration:
                li = 0
            try:
                lj = next(
                    k for k, gj in enumerate(feat_indices)
                    if all_feat_names[k] == feat_j or (gj < len(all_feat_names) and False)
                )
            except StopIteration:
                lj = 0

            example = {
                "input": input_str,
                "output": output_str,
                "metadata_pid_method": synergy_data["pid_method"],
                "metadata_dataset": ds_name,
                "metadata_feature_i": feat_i,
                "metadata_feature_j": feat_j,
                "predict_synergy_value": str(round(pid["synergy"], 6)),
                "predict_baseline_coi": str(round(
                    coi_matrix[li, lj] if li < coi_matrix.shape[0] and lj < coi_matrix.shape[1] else 0.0,
                    6
                )),
            }
            examples.append(example)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    return output_datasets


def build_output_examples_v2(all_results: list[dict]) -> list[dict]:
    """Build output in exp_gen_sol_out schema format - cleaner version.

    Each pair becomes one example. Input describes the pair, output has PID results.
    """
    output_datasets = []

    for ds_result in all_results:
        ds_name = ds_result["dataset"]
        examples = []
        synergy_data = ds_result["synergy"]
        pid_details = synergy_data["pid_details"]
        coi_matrix = np.array(synergy_data["coi_matrix"])
        feat_indices = synergy_data["feature_indices_used"]

        # Map pair keys to local indices
        pairs_list = list(combinations(range(len(feat_indices)), 2))
        all_feature_names_global = ds_result.get("_all_feature_names", [])

        pair_idx = 0
        for pair_key, pid in pid_details.items():
            parts = pair_key.split("__x__")
            feat_i = parts[0] if len(parts) == 2 else pair_key
            feat_j = parts[1] if len(parts) == 2 else ""

            # Find the pair in combinations to get CoI
            coi_val = 0.0
            if pair_idx < len(pairs_list):
                li, lj = pairs_list[pair_idx]
                if li < coi_matrix.shape[0] and lj < coi_matrix.shape[1]:
                    coi_val = coi_matrix[li, lj]
            pair_idx += 1

            input_str = json.dumps({
                "feature_i": feat_i,
                "feature_j": feat_j,
                "dataset": ds_name,
                "n_samples": ds_result["n_samples"],
                "n_classes": ds_result["n_classes"],
                "n_features": ds_result["n_features"],
            })

            output_str = json.dumps({
                "synergy": round(pid["synergy"], 6),
                "unique_0": round(pid["unique_0"], 6),
                "unique_1": round(pid["unique_1"], 6),
                "redundancy": round(pid["redundancy"], 6),
                "coi_baseline": round(coi_val, 6),
            })

            example = {
                "input": input_str,
                "output": output_str,
                "metadata_pid_method": synergy_data["pid_method"],
                "metadata_dataset": ds_name,
                "metadata_feature_i": feat_i,
                "metadata_feature_j": feat_j,
                "metadata_n_samples": ds_result["n_samples"],
                "metadata_n_classes": ds_result["n_classes"],
                "predict_synergy_value": str(round(pid["synergy"], 6)),
                "predict_baseline_coi": str(round(coi_val, 6)),
            }
            examples.append(example)

        if not examples:
            # Ensure at least one example per dataset
            examples.append({
                "input": json.dumps({"dataset": ds_name, "status": "no_pairs_computed"}),
                "output": json.dumps({"synergy": 0.0, "note": "no pairs computed"}),
                "metadata_dataset": ds_name,
                "predict_synergy_value": "0.0",
                "predict_baseline_coi": "0.0",
            })

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    return output_datasets


@logger.catch
def main():
    t_global_start = time.time()
    logger.info(f"{GREEN}{'='*60}{END}")
    logger.info(f"{GREEN}Pairwise PID Synergy Matrix Experiment{END}")
    logger.info(f"{GREEN}{'='*60}{END}")

    # ── Phase 1: Load all datasets ───────────────────────────────────────
    logger.info(f"{BLUE}Phase 1: Loading datasets{END}")
    datasets = load_all_datasets()

    # ── Phase 2: XOR Validation ──────────────────────────────────────────
    xor_results = xor_validation()

    # ── Sort datasets by number of pairs (ascending) ─────────────────────
    def sort_key(item):
        name, info = item
        n = info["n_features"]
        if n > MAX_FEATURES_LARGE:
            n = MAX_FEATURES_LARGE
        return n * (n - 1) // 2

    sorted_datasets = sorted(datasets.items(), key=sort_key)
    logger.info(f"\n{BLUE}Processing order:{END}")
    for name, info in sorted_datasets:
        n = min(info["n_features"], MAX_FEATURES_LARGE)
        logger.info(f"  {name}: {n*(n-1)//2} pairs")

    # ── Datasets for stability analysis ──────────────────────────────────
    stability_datasets = {"pima_diabetes", "breast_cancer_wisconsin_diagnostic", "breast_cancer"}

    # ── Phase 3-7: Process each dataset ──────────────────────────────────
    all_results = []
    for ds_name, ds_info in sorted_datasets:
        logger.info(f"\n{CYAN}{'='*50}{END}")
        logger.info(f"{CYAN}Processing: {ds_name}{END}")
        logger.info(f"{CYAN}{'='*50}{END}")

        do_stability = ds_name in stability_datasets
        try:
            result = process_dataset(
                ds_name=ds_name,
                ds_info=ds_info,
                do_stability=do_stability,
            )
            all_results.append(result)
        except Exception:
            logger.exception(f"Failed processing {ds_name}")
            continue

        elapsed = time.time() - t_global_start
        logger.info(f"  Total elapsed: {elapsed:.1f}s")
        if elapsed > 3000:  # 50 min safety
            logger.warning(f"{YELLOW}Approaching time limit, stopping{END}")
            break

    # ── Phase 8: Build output ────────────────────────────────────────────
    logger.info(f"\n{BLUE}Phase 8: Building output{END}")

    # Build exp_gen_sol_out format
    output_datasets = build_output_examples_v2(all_results)

    method_out = {"datasets": output_datasets}
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2))

    total_examples = sum(len(ds["examples"]) for ds in output_datasets)
    logger.info(f"  Wrote {total_examples} examples across {len(output_datasets)} datasets")

    # ── Comprehensive results ────────────────────────────────────────────
    # Aggregate summary
    aggregate = {
        "total_datasets_processed": len(all_results),
        "total_pairs_computed": sum(r["synergy"]["completed_pairs"] for r in all_results),
        "total_errors": sum(r["synergy"]["errors"] for r in all_results),
        "total_time_s": round(time.time() - t_global_start, 2),
        "xor_validation": xor_results,
        "per_dataset_summary": [],
    }

    for r in all_results:
        syn_mat = np.array(r["synergy"]["synergy_matrix"])
        upper_vals = []
        n = syn_mat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                upper_vals.append(syn_mat[i, j])

        ds_summary = {
            "dataset": r["dataset"],
            "n_samples": r["n_samples"],
            "n_features": r["n_features"],
            "n_features_used": r["n_features_used"],
            "n_classes": r["n_classes"],
            "pid_method": r["synergy"]["pid_method"],
            "n_pairs": r["synergy"]["n_pairs"],
            "completed_pairs": r["synergy"]["completed_pairs"],
            "total_time_s": r["synergy"]["total_time_s"],
            "synergy_mean": round(float(np.mean(upper_vals)), 6) if upper_vals else 0.0,
            "synergy_std": round(float(np.std(upper_vals)), 6) if upper_vals else 0.0,
            "synergy_max": round(float(np.max(upper_vals)), 6) if upper_vals else 0.0,
            "synergy_min": round(float(np.min(upper_vals)), 6) if upper_vals else 0.0,
            "mi_comparison_jaccard": r["mi_comparison"]["jaccard_overlap"],
            "mi_comparison_spearman": r["mi_comparison"]["spearman_rho"],
            "synergy_graph_edges": r["synergy_graph"]["n_edges"],
            "synergy_graph_components": r["synergy_graph"]["n_components"],
            "synergy_graph_largest_clique": r["synergy_graph"]["largest_clique_size"],
        }
        if r["stability"] is not None:
            ds_summary["stability_mean_rho"] = r["stability"]["mean_spearman"]
            ds_summary["stability_std_rho"] = r["stability"]["std_spearman"]

        aggregate["per_dataset_summary"].append(ds_summary)

    comprehensive = {
        "experiment": "Pairwise PID Synergy Matrices on Benchmark Datasets",
        "method": "PID_BROJA (small) / PID_MMI (large) with Co-Information baseline",
        "n_bins": N_BINS,
        "broja_pair_limit": BROJA_PAIR_LIMIT,
        "aggregate": aggregate,
        "per_dataset_full": [],
    }

    for r in all_results:
        ds_full = {
            "dataset": r["dataset"],
            "n_samples": r["n_samples"],
            "n_features": r["n_features"],
            "n_features_used": r["n_features_used"],
            "n_classes": r["n_classes"],
            "source": r["source"],
            "pid_method": r["synergy"]["pid_method"],
            "synergy_matrix": r["synergy"]["synergy_matrix"],
            "coi_matrix": r["synergy"]["coi_matrix"],
            "mi_values": r["mi_values"],
            "mi_comparison": r["mi_comparison"],
            "synergy_graph": r["synergy_graph"],
            "timing": {
                "total_time_s": r["synergy"]["total_time_s"],
                "avg_time_per_pair_s": r["synergy"]["avg_time_per_pair_s"],
                "n_pairs": r["synergy"]["n_pairs"],
                "completed_pairs": r["synergy"]["completed_pairs"],
            },
        }
        if r["stability"] is not None:
            ds_full["stability"] = r["stability"]
        comprehensive["per_dataset_full"].append(ds_full)

    comp_path = WORKSPACE / "results_comprehensive.json"
    comp_path.write_text(json.dumps(comprehensive, indent=2, cls=NumpyEncoder))
    logger.info(f"  Wrote comprehensive results to {comp_path}")

    total_time = time.time() - t_global_start
    logger.info(f"\n{GREEN}{'='*60}{END}")
    logger.info(f"{GREEN}DONE in {total_time:.1f}s{END}")
    logger.info(f"{GREEN}Datasets: {len(all_results)}, Pairs: {aggregate['total_pairs_computed']}{END}")
    logger.info(f"{GREEN}Output: {out_path}{END}")
    logger.info(f"{GREEN}{'='*60}{END}")


if __name__ == "__main__":
    main()
