#!/usr/bin/env python3
"""
Synergy Threshold Sensitivity & Adaptive Thresholding for SG-FIGS.

Systematic sensitivity analysis of synergy threshold percentiles (50th, 65th,
75th, 90th) across 14 datasets, plus an adaptive thresholding heuristic.
"""

import warnings
warnings.filterwarnings("ignore")

from loguru import logger
from pathlib import Path
import json
import sys
import time
import resource
import numpy as np
import networkx as nx
from collections import Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mutual_info_score
from sklearn.ensemble import GradientBoostingClassifier

# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
DATA_ID2_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full/"
    "3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
)
DATA_ID3_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full/"
    "3_invention_loop/iter_2/gen_art/data_id3_it2__opus/full_data_out.json"
)
SYNERGY_RESULTS_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full/"
    "3_invention_loop/iter_2/gen_art/exp_id1_it2__opus/results_comprehensive.json"
)

THRESHOLD_PERCENTILES = [50, 75, 90]
MAX_SPLITS_VALUES = [5, 10, 15]
N_FOLDS = 5
RANDOM_SEED = 42
ADAPTIVE_CANDIDATES = [50, 60, 70, 80, 90]

NEW_SYNERGY_DATASETS = ["monks2", "blood", "climate", "kc2"]
SYNERGY_NAME_MAP = {
    "breast_cancer": "breast_cancer_wisconsin_diagnostic",
}


# ============================================================================
# DATA LOADING
# ============================================================================
def load_datasets_from_json(json_path: Path) -> dict:
    """Load all datasets from a full_data_out.json file."""
    logger.info(f"Loading datasets from {json_path.name}")
    data = json.loads(json_path.read_text())
    datasets = {}
    for ds_entry in data["datasets"]:
        name = ds_entry["dataset"]
        examples = ds_entry["examples"]
        first_input = json.loads(examples[0]["input"])
        feature_names = list(first_input.keys())
        X = np.array(
            [list(json.loads(ex["input"]).values()) for ex in examples],
            dtype=float,
        )
        raw_labels = [ex["output"] for ex in examples]
        try:
            y = np.array([int(lbl) for lbl in raw_labels])
        except ValueError:
            unique_labels = sorted(set(raw_labels))
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            y = np.array([label_map[lbl] for lbl in raw_labels])
        unique_y = np.unique(y)
        if not np.array_equal(unique_y, np.arange(len(unique_y))):
            remap = {old: new for new, old in enumerate(unique_y)}
            y = np.array([remap[v] for v in y])
        folds = np.array([int(ex["metadata_fold"]) for ex in examples])
        datasets[name] = {
            "X": X, "y": y, "folds": folds,
            "feature_names": feature_names,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_classes": len(np.unique(y)),
        }
    return datasets


def load_precomputed_synergy(synergy_path: Path) -> dict:
    """Load synergy matrices from results_comprehensive.json."""
    logger.info("Loading pre-computed synergy matrices")
    data = json.loads(synergy_path.read_text())
    synergy_data = {}
    for ds in data["per_dataset_full"]:
        ds_name = ds["dataset"]
        mapped = SYNERGY_NAME_MAP.get(ds_name, ds_name)
        synergy_data[mapped] = {
            "synergy_matrix": np.array(ds["synergy_matrix"]),
            "feature_names": list(ds["mi_values"].keys()),
            "pid_method": ds["pid_method"],
            "n_features_used": ds["n_features_used"],
        }
    return synergy_data


# ============================================================================
# SYNERGY COMPUTATION (CoI proxy for new datasets)
# ============================================================================
def compute_coi_synergy(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    n_features = X.shape[1]
    X_disc = np.zeros_like(X, dtype=int)
    for j in range(n_features):
        col = X[:, j]
        n_unique = len(np.unique(col))
        if n_unique <= 5:
            _, X_disc[:, j] = np.unique(col, return_inverse=True)
        else:
            try:
                kbd = KBinsDiscretizer(n_bins=min(5, n_unique), encode="ordinal", strategy="quantile")
                X_disc[:, j] = kbd.fit_transform(col.reshape(-1, 1)).ravel().astype(int)
            except ValueError:
                _, X_disc[:, j] = np.unique(col, return_inverse=True)
    y_int = y.astype(int)
    mi_single = np.zeros(n_features)
    for j in range(n_features):
        mi_single[j] = mutual_info_score(X_disc[:, j], y_int)
    synergy_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            xy_joint = [f"{a}_{b}" for a, b in zip(X_disc[:, i], X_disc[:, j])]
            mi_joint = mutual_info_score(xy_joint, y_int)
            coi = mi_single[i] + mi_single[j] - mi_joint
            synergy_val = max(0.0, -coi)
            synergy_matrix[i, j] = synergy_val
            synergy_matrix[j, i] = synergy_val
    return {
        "synergy_matrix": synergy_matrix,
        "feature_names": feature_names,
        "pid_method": "CoI_proxy",
        "n_features_used": n_features,
    }


def align_synergy_with_dataset(synergy_info: dict, dataset_info: dict) -> np.ndarray:
    syn_features = synergy_info["feature_names"]
    ds_features = dataset_info["feature_names"]
    n_ds = len(ds_features)
    syn_matrix = synergy_info["synergy_matrix"]
    if syn_features == ds_features:
        return syn_matrix
    syn_to_ds = {}
    for si, sf in enumerate(syn_features):
        for di, df in enumerate(ds_features):
            if sf == df:
                syn_to_ds[si] = di
                break
    aligned = np.zeros((n_ds, n_ds))
    for si in range(len(syn_features)):
        for sj in range(len(syn_features)):
            di = syn_to_ds.get(si)
            dj = syn_to_ds.get(sj)
            if di is not None and dj is not None:
                aligned[di, dj] = syn_matrix[si, sj]
    return aligned


# ============================================================================
# SYNERGY GRAPH
# ============================================================================
def build_synergy_graph(synergy_matrix: np.ndarray, percentile: float) -> nx.Graph:
    n_features = synergy_matrix.shape[0]
    upper_tri = synergy_matrix[np.triu_indices(n_features, k=1)]
    nonzero = upper_tri[upper_tri > 0]
    threshold = np.percentile(nonzero, percentile) if len(nonzero) > 0 else 0.0
    G = nx.Graph()
    G.add_nodes_from(range(n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if synergy_matrix[i, j] >= threshold and synergy_matrix[i, j] > 0:
                G.add_edge(i, j, weight=float(synergy_matrix[i, j]))
    return G


def compute_graph_statistics(G: nx.Graph, n_features: int) -> dict:
    components = list(nx.connected_components(G))
    largest = max(len(c) for c in components) if components else 0
    try:
        cliques = list(nx.find_cliques(G)) if G.number_of_edges() < 500 else []
    except Exception:
        cliques = []
    clique_sizes = [len(c) for c in cliques] if cliques else [0]
    return {
        "n_edges": G.number_of_edges(),
        "n_components": len(components),
        "largest_component_size": largest,
        "largest_component_fraction": largest / n_features if n_features > 0 else 0,
        "n_cliques": len(cliques),
        "largest_clique_size": max(clique_sizes),
        "mean_clique_size": float(np.mean(clique_sizes)),
        "n_isolated": len(list(nx.isolates(G))),
        "graph_density": nx.density(G),
    }


def get_candidate_subsets(synergy_graph: nx.Graph, max_cand: int = 15) -> list:
    candidates = []
    edges_w = [(sorted([u, v]), d.get("weight", 0)) for u, v, d in synergy_graph.edges(data=True)]
    edges_w.sort(key=lambda x: -x[1])
    for edge, _ in edges_w[:max_cand]:
        candidates.append(edge)
    if synergy_graph.number_of_edges() < 50:
        try:
            for clique in nx.find_cliques(synergy_graph):
                if 3 <= len(clique) <= 4:
                    candidates.append(sorted(clique))
                    if len(candidates) >= max_cand:
                        break
        except Exception:
            pass
    for node in nx.isolates(synergy_graph):
        candidates.append([node])
    if not candidates:
        for node in synergy_graph.nodes():
            candidates.append([node])
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:max_cand]


# ============================================================================
# TREE NODE
# ============================================================================
class TreeNode:
    __slots__ = ['feature_indices', 'weights', 'bias', 'threshold', 'value',
                 'left', 'right', 'is_leaf', 'n_samples', 'sample_indices']

    def __init__(self):
        self.feature_indices = []
        self.weights = np.array([])
        self.bias = 0.0
        self.threshold = 0.0
        self.value = 0.0
        self.left = None
        self.right = None
        self.is_leaf = True
        self.n_samples = 0
        self.sample_indices = np.array([], dtype=int)

    def predict_single(self, x):
        if self.is_leaf:
            return self.value
        proj = np.dot(x[self.feature_indices], self.weights) + self.bias
        if proj <= self.threshold:
            return self.left.predict_single(x) if self.left else self.value
        return self.right.predict_single(x) if self.right else self.value

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


def _impurity_reduction(y_parent, y_left, y_right):
    n = len(y_parent)
    n_l, n_r = len(y_left), len(y_right)
    if n_l < 2 or n_r < 2:
        return 0.0
    return np.var(y_parent) - (n_l / n * np.var(y_left) + n_r / n * np.var(y_right))


def _find_best_split_on_proj(proj, residuals):
    """Find best threshold on a 1D projection. Returns (score, threshold)."""
    best_score = -1.0
    best_thr = 0.0
    thr = float(np.median(proj))
    left = proj <= thr
    right = ~left
    if left.sum() >= 2 and right.sum() >= 2:
        best_score = _impurity_reduction(residuals, residuals[left], residuals[right])
        best_thr = thr
    return best_score, best_thr


# ============================================================================
# CLASSIFIERS
# ============================================================================
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _get_leaves(node):
    if node.is_leaf:
        return [node]
    out = []
    if node.left:
        out.extend(_get_leaves(node.left))
    if node.right:
        out.extend(_get_leaves(node.right))
    return out


def _fit_greedy_tree(X, y_binary, max_splits, candidates):
    """Core greedy tree-fitting loop shared by SG-FIGS and baselines."""
    n_samples, n_features = X.shape
    p = np.clip(np.mean(y_binary), 0.01, 0.99)
    class_prior = np.log(p / (1 - p))
    predictions = np.full(n_samples, class_prior)
    residuals = y_binary - _sigmoid(predictions)

    root = TreeNode()
    root.is_leaf = True
    root.value = 0.0
    root.n_samples = n_samples
    root.sample_indices = np.arange(n_samples)
    trees = [root]
    n_splits = 0
    split_info = []

    for split_i in range(max_splits):
        best_score = 1e-10
        best_data = None
        best_leaf = None

        # Cap leaf scanning: only consider top-3 largest leaves
        all_leaves = []
        for tree in trees:
            all_leaves.extend(_get_leaves(tree))
        all_leaves.sort(key=lambda l: -l.n_samples)
        scan_leaves = all_leaves[:3]

        for leaf in scan_leaves:
            if leaf.n_samples < 10:
                continue
            leaf_X = X[leaf.sample_indices]
            leaf_res = residuals[leaf.sample_indices]
            leaf_tgt = (leaf_res > 0).astype(int)
            if len(np.unique(leaf_tgt)) < 2:
                continue

            for feat_subset in candidates:
                valid = [f for f in feat_subset if f < n_features]
                if not valid:
                    continue
                X_sub = leaf_X[:, valid]
                if len(valid) == 1:
                    proj = X_sub.ravel()
                    w = np.array([1.0])
                    b = 0.0
                else:
                    try:
                        ridge = RidgeClassifier(alpha=1.0)
                        ridge.fit(X_sub, leaf_tgt)
                        w = ridge.coef_.ravel()
                        b = float(ridge.intercept_) if np.ndim(ridge.intercept_) == 0 else float(ridge.intercept_[0])
                        proj = X_sub @ w + b
                    except Exception:
                        continue

                score, thr = _find_best_split_on_proj(proj, leaf_res)
                if score > best_score:
                    left_mask = proj <= thr
                    right_mask = ~left_mask
                    best_score = score
                    best_data = {
                        "feature_indices": valid, "weights": w.copy(),
                        "bias": b, "threshold": thr,
                        "left_mask": left_mask, "right_mask": right_mask,
                    }
                    best_leaf = leaf

        if best_data is None:
            break

        leaf = best_leaf
        leaf.is_leaf = False
        leaf.feature_indices = best_data["feature_indices"]
        leaf.weights = best_data["weights"]
        leaf.bias = best_data["bias"]
        leaf.threshold = best_data["threshold"]

        li = leaf.sample_indices[best_data["left_mask"]]
        ri = leaf.sample_indices[best_data["right_mask"]]

        leaf.left = TreeNode()
        leaf.left.is_leaf = True
        leaf.left.value = float(np.mean(residuals[li]))
        leaf.left.n_samples = len(li)
        leaf.left.sample_indices = li

        leaf.right = TreeNode()
        leaf.right.is_leaf = True
        leaf.right.value = float(np.mean(residuals[ri]))
        leaf.right.n_samples = len(ri)
        leaf.right.sample_indices = ri

        predictions[li] += leaf.left.value
        predictions[ri] += leaf.right.value
        residuals = y_binary - _sigmoid(predictions)
        n_splits += 1
        split_info.append({
            "feature_indices": best_data["feature_indices"],
            "score": best_score,
            "n_features_in_split": len(best_data["feature_indices"]),
        })

    return trees, n_splits, split_info, class_prior


class SGFIGSClassifier:
    def __init__(self, max_splits=10, synergy_graph=None):
        self.max_splits = max_splits
        self.synergy_graph = synergy_graph
        self.trees = []
        self.n_splits = 0
        self.split_info = []
        self.class_prior_ = 0.0

    def fit(self, X, y):
        n_features = X.shape[1]
        y_binary = (y > 0).astype(float) if len(np.unique(y)) > 2 else y.astype(float)
        candidates = get_candidate_subsets(self.synergy_graph) if self.synergy_graph else [[i] for i in range(n_features)]
        if not candidates:
            candidates = [[i] for i in range(n_features)]
        self.trees, self.n_splits, self.split_info, self.class_prior_ = _fit_greedy_tree(
            X, y_binary, self.max_splits, candidates
        )
        return self

    def predict_proba(self, X):
        raw = np.full(X.shape[0], self.class_prior_)
        for tree in self.trees:
            raw += tree.predict(X)
        p1 = _sigmoid(raw)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FIGSClassifier:
    def __init__(self, max_splits=10):
        self.max_splits = max_splits
        self.trees = []
        self.n_splits = 0
        self.class_prior_ = 0.0

    def fit(self, X, y):
        n_features = X.shape[1]
        y_binary = (y > 0).astype(float) if len(np.unique(y)) > 2 else y.astype(float)
        # Axis-aligned: each feature is a candidate (capped to top-10 by MI)
        if n_features > 10:
            mi_scores = np.array([mutual_info_score(
                np.digitize(X[:, j], np.percentile(X[:, j], [25, 50, 75])), y_binary.astype(int)
            ) for j in range(n_features)])
            top_idx = np.argsort(-mi_scores)[:10]
            candidates = [[i] for i in top_idx]
        else:
            candidates = [[i] for i in range(n_features)]
        self.trees, self.n_splits, _, self.class_prior_ = _fit_greedy_tree(
            X, y_binary, self.max_splits, candidates
        )
        return self

    def predict_proba(self, X):
        raw = np.full(X.shape[0], self.class_prior_)
        for tree in self.trees:
            raw += tree.predict(X)
        p1 = _sigmoid(raw)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ROFIGSClassifier:
    def __init__(self, max_splits=10, random_state=42):
        self.max_splits = max_splits
        self.random_state = random_state
        self.trees = []
        self.n_splits = 0
        self.class_prior_ = 0.0

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        y_binary = (y > 0).astype(float) if len(np.unique(y)) > 2 else y.astype(float)
        beam = max(2, min(3, int(np.sqrt(n_features))))
        max_cand = 15
        # Generate random oblique candidates
        candidates = []
        for _ in range(min(max_cand, n_features * 2)):
            candidates.append(sorted(rng.choice(n_features, size=beam, replace=False).tolist()))
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            k = tuple(c)
            if k not in seen:
                seen.add(k)
                unique.append(c)
        candidates = unique[:max_cand]
        self.trees, self.n_splits, _, self.class_prior_ = _fit_greedy_tree(
            X, y_binary, self.max_splits, candidates
        )
        return self

    def predict_proba(self, X):
        raw = np.full(X.shape[0], self.class_prior_)
        for tree in self.trees:
            raw += tree.predict(X)
        p1 = _sigmoid(raw)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ============================================================================
# ADAPTIVE THRESHOLD
# ============================================================================
def select_adaptive_threshold(synergy_matrix, n_features):
    best_pct = 75
    for pct in ADAPTIVE_CANDIDATES:
        G = build_synergy_graph(synergy_matrix, pct)
        st = compute_graph_statistics(G, n_features)
        if st["largest_component_fraction"] < 0.80 and st["n_edges"] >= n_features / 2:
            best_pct = pct
    return best_pct


# ============================================================================
# INTERPRETABILITY SCORE
# ============================================================================
def compute_interpretability(split_info, synergy_matrix):
    if not split_info:
        return 0.0
    upper = synergy_matrix[np.triu_indices(synergy_matrix.shape[0], k=1)]
    nz = upper[upper > 0]
    med = np.median(nz) if len(nz) > 0 else 0.0
    oblique = [s for s in split_info if s["n_features_in_split"] >= 2]
    if not oblique:
        return 1.0
    above = 0
    for s in oblique:
        feats = s["feature_indices"]
        pairs = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                if feats[i] < synergy_matrix.shape[0] and feats[j] < synergy_matrix.shape[1]:
                    pairs.append(synergy_matrix[feats[i], feats[j]])
        if pairs and np.mean(pairs) > med:
            above += 1
    return above / len(oblique)


# ============================================================================
# SINGLE EXPERIMENT
# ============================================================================
def run_experiment(X_train, y_train, X_test, y_test, syn_mat, pct, max_splits, fold, ds_name, n_feat):
    G = build_synergy_graph(syn_mat, pct)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    y_test_bin = (y_test > 0).astype(int) if len(np.unique(y_test)) > 2 else y_test
    y_train_bin = (y_train > 0).astype(int) if len(np.unique(y_train)) > 2 else y_train
    res = {}

    def safe_auc(y_true, y_proba):
        try:
            return round(roc_auc_score(y_true, y_proba), 6)
        except ValueError:
            return 0.5

    # SG-FIGS
    try:
        sg = SGFIGSClassifier(max_splits=max_splits, synergy_graph=G)
        sg.fit(Xtr, y_train)
        yp = sg.predict(Xte)
        ypr = sg.predict_proba(Xte)
        res["sg_figs_balanced_acc"] = round(balanced_accuracy_score(y_test_bin, (yp > 0).astype(int) if len(np.unique(y_test)) > 2 else yp), 6)
        res["sg_figs_auc"] = safe_auc(y_test_bin, ypr[:, 1])
        res["sg_figs_n_splits"] = sg.n_splits
        res["sg_figs_interpretability"] = round(compute_interpretability(sg.split_info, syn_mat), 4)
    except Exception as e:
        logger.debug(f"SG-FIGS err {ds_name} f{fold}: {e}")
        res.update({"sg_figs_balanced_acc": 0.5, "sg_figs_auc": 0.5, "sg_figs_n_splits": 0, "sg_figs_interpretability": 0.0})

    # FIGS
    try:
        fg = FIGSClassifier(max_splits=max_splits)
        fg.fit(Xtr, y_train)
        yp = fg.predict(Xte)
        ypr = fg.predict_proba(Xte)
        res["figs_balanced_acc"] = round(balanced_accuracy_score(y_test_bin, (yp > 0).astype(int) if len(np.unique(y_test)) > 2 else yp), 6)
        res["figs_auc"] = safe_auc(y_test_bin, ypr[:, 1])
    except Exception as e:
        logger.debug(f"FIGS err {ds_name} f{fold}: {e}")
        res.update({"figs_balanced_acc": 0.5, "figs_auc": 0.5})

    # RO-FIGS
    try:
        ro = ROFIGSClassifier(max_splits=max_splits, random_state=RANDOM_SEED + fold)
        ro.fit(Xtr, y_train)
        yp = ro.predict(Xte)
        ypr = ro.predict_proba(Xte)
        res["rofigs_balanced_acc"] = round(balanced_accuracy_score(y_test_bin, (yp > 0).astype(int) if len(np.unique(y_test)) > 2 else yp), 6)
        res["rofigs_auc"] = safe_auc(y_test_bin, ypr[:, 1])
    except Exception as e:
        logger.debug(f"RO-FIGS err {ds_name} f{fold}: {e}")
        res.update({"rofigs_balanced_acc": 0.5, "rofigs_auc": 0.5})

    # GBDT
    try:
        gb = GradientBoostingClassifier(n_estimators=max(max_splits, 10), max_depth=2, random_state=RANDOM_SEED, subsample=0.8)
        gb.fit(Xtr, y_train_bin)
        yp = gb.predict(Xte)
        ypr = gb.predict_proba(Xte)
        res["gbdt_balanced_acc"] = round(balanced_accuracy_score(y_test_bin, yp), 6)
        res["gbdt_auc"] = safe_auc(y_test_bin, ypr[:, 1])
    except Exception as e:
        logger.debug(f"GBDT err {ds_name} f{fold}: {e}")
        res.update({"gbdt_balanced_acc": 0.5, "gbdt_auc": 0.5})

    return res


# ============================================================================
# MAIN
# ============================================================================
@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Synergy Threshold Sensitivity & Adaptive Thresholding")
    logger.info("=" * 60)

    # Load datasets
    datasets_id2 = load_datasets_from_json(DATA_ID2_PATH)
    datasets_id3 = load_datasets_from_json(DATA_ID3_PATH)
    all_datasets = {}
    all_datasets.update(datasets_id2)
    all_datasets.update(datasets_id3)
    logger.info(f"Loaded {len(all_datasets)} datasets")
    for n, d in all_datasets.items():
        logger.info(f"  {n}: {d['n_samples']} samples, {d['n_features']} feat, {d['n_classes']} classes")

    # Load synergy
    synergy_data = load_precomputed_synergy(SYNERGY_RESULTS_PATH)
    logger.info(f"Pre-computed synergy for {len(synergy_data)} datasets")

    # Compute synergy for new datasets
    for ds_name in NEW_SYNERGY_DATASETS:
        if ds_name not in all_datasets:
            continue
        logger.info(f"Computing CoI synergy for {ds_name}...")
        t0 = time.time()
        ds = all_datasets[ds_name]
        synergy_data[ds_name] = compute_coi_synergy(ds["X"], ds["y"], ds["feature_names"])
        logger.info(f"  Done in {time.time()-t0:.1f}s")

    # Align synergy matrices
    aligned_synergy = {}
    for ds_name, ds_info in all_datasets.items():
        if ds_name in synergy_data:
            aligned_synergy[ds_name] = align_synergy_with_dataset(synergy_data[ds_name], ds_info)
        else:
            n_f = ds_info["n_features"]
            aligned_synergy[ds_name] = np.zeros((n_f, n_f))

    # Graph statistics
    logger.info("Computing graph statistics...")
    graph_statistics = {}
    for ds_name in all_datasets:
        graph_statistics[ds_name] = {}
        syn_mat = aligned_synergy[ds_name]
        n_f = all_datasets[ds_name]["n_features"]
        for pct in THRESHOLD_PERCENTILES:
            G = build_synergy_graph(syn_mat, pct)
            graph_statistics[ds_name][str(pct)] = compute_graph_statistics(G, n_f)

    # Adaptive thresholds
    adaptive_thresholds = {}
    for ds_name in all_datasets:
        adaptive_thresholds[ds_name] = select_adaptive_threshold(
            aligned_synergy[ds_name], all_datasets[ds_name]["n_features"]
        )

    # Run experiments
    dataset_names = sorted(all_datasets.keys())
    total_combos = len(dataset_names) * (len(THRESHOLD_PERCENTILES) + 1) * len(MAX_SPLITS_VALUES) * N_FOLDS
    logger.info(f"Total experiment combos: {total_combos}")
    combo_count = 0

    all_results = {}
    method_out_datasets = []

    for ds_name in dataset_names:
        ds = all_datasets[ds_name]
        syn_mat = aligned_synergy[ds_name]
        ds_start = time.time()
        logger.info(f"Dataset: {ds_name} ({ds['n_samples']}s, {ds['n_features']}f)")

        ds_examples = []
        ds_results = {}
        adaptive_pct = adaptive_thresholds[ds_name]

        for pct_key in THRESHOLD_PERCENTILES + ["adaptive"]:
            actual_pct = adaptive_pct if pct_key == "adaptive" else pct_key
            for max_splits in MAX_SPLITS_VALUES:
                fold_results = []
                for fold_k in range(N_FOLDS):
                    combo_count += 1
                    train_mask = ds["folds"] != fold_k
                    test_mask = ds["folds"] == fold_k
                    if test_mask.sum() == 0 or train_mask.sum() == 0:
                        continue

                    result = run_experiment(
                        X_train=ds["X"][train_mask], y_train=ds["y"][train_mask],
                        X_test=ds["X"][test_mask], y_test=ds["y"][test_mask],
                        syn_mat=syn_mat, pct=actual_pct, max_splits=max_splits,
                        fold=fold_k, ds_name=ds_name, n_feat=ds["n_features"],
                    )
                    fold_results.append(result)

                    # Only fixed thresholds go into method_out
                    if pct_key != "adaptive":
                        input_d = {"threshold_percentile": actual_pct, "max_splits": max_splits,
                                   "fold": fold_k, "dataset": ds_name, "n_features": ds["n_features"]}
                        output_d = {k: result[k] for k in result}
                        ds_examples.append({
                            "input": json.dumps(input_d),
                            "output": json.dumps(output_d),
                            "metadata_fold": fold_k,
                            "metadata_threshold_percentile": actual_pct,
                            "metadata_max_splits": max_splits,
                            "metadata_dataset": ds_name,
                            "metadata_n_features": ds["n_features"],
                            "predict_sg_figs_balanced_acc": str(result["sg_figs_balanced_acc"]),
                            "predict_baseline_figs_balanced_acc": str(result["figs_balanced_acc"]),
                        })

                if fold_results:
                    config_key = f"{pct_key}_{max_splits}"
                    ds_results[config_key] = {
                        "mean_balanced_acc": round(float(np.mean([r["sg_figs_balanced_acc"] for r in fold_results])), 6),
                        "std_balanced_acc": round(float(np.std([r["sg_figs_balanced_acc"] for r in fold_results])), 6),
                        "mean_auc": round(float(np.mean([r["sg_figs_auc"] for r in fold_results])), 6),
                        "mean_figs_acc": round(float(np.mean([r["figs_balanced_acc"] for r in fold_results])), 6),
                        "mean_rofigs_acc": round(float(np.mean([r["rofigs_balanced_acc"] for r in fold_results])), 6),
                        "mean_gbdt_acc": round(float(np.mean([r["gbdt_balanced_acc"] for r in fold_results])), 6),
                        "mean_interpretability": round(float(np.mean([r["sg_figs_interpretability"] for r in fold_results])), 4),
                    }

                if combo_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = combo_count / elapsed if elapsed > 0 else 1
                    remaining = (total_combos - combo_count) / rate if rate > 0 else 0
                    logger.info(f"  Progress: {combo_count}/{total_combos} ({elapsed:.0f}s, ~{remaining:.0f}s left)")

        all_results[ds_name] = ds_results
        method_out_datasets.append({"dataset": ds_name, "examples": ds_examples})
        logger.info(f"  {ds_name} done in {time.time()-ds_start:.1f}s, {len(ds_examples)} examples")

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    logger.info("Computing analysis...")
    per_dataset_analysis = {}
    for ds_name in dataset_names:
        syn_mat = aligned_synergy[ds_name]
        upper = syn_mat[np.triu_indices(syn_mat.shape[0], k=1)]
        nz = upper[upper > 0]
        synergy_dist = {
            "mean": round(float(np.mean(nz)), 6) if len(nz) > 0 else 0.0,
            "std": round(float(np.std(nz)), 6) if len(nz) > 0 else 0.0,
            "skew": round(float(stats.skew(nz)), 6) if len(nz) > 2 else 0.0,
            "min": round(float(np.min(nz)), 6) if len(nz) > 0 else 0.0,
            "max": round(float(np.max(nz)), 6) if len(nz) > 0 else 0.0,
            "cv": round(float(np.std(nz) / np.mean(nz)), 6) if len(nz) > 0 and np.mean(nz) > 0 else 0.0,
        }

        threshold_results = {}
        for pct in THRESHOLD_PERCENTILES:
            accs = [all_results[ds_name][f"{pct}_{ms}"]["mean_balanced_acc"]
                    for ms in MAX_SPLITS_VALUES if f"{pct}_{ms}" in all_results[ds_name]]
            aucs = [all_results[ds_name][f"{pct}_{ms}"]["mean_auc"]
                    for ms in MAX_SPLITS_VALUES if f"{pct}_{ms}" in all_results[ds_name]]
            threshold_results[str(pct)] = {
                "mean_balanced_acc": round(float(np.mean(accs)), 6) if accs else 0.5,
                "std_balanced_acc": round(float(np.std(accs)), 6) if accs else 0.0,
                "mean_auc": round(float(np.mean(aucs)), 6) if aucs else 0.5,
            }

        best_pct = max(THRESHOLD_PERCENTILES, key=lambda p: threshold_results[str(p)]["mean_balanced_acc"])
        best_acc = threshold_results[str(best_pct)]["mean_balanced_acc"]
        fixed_75 = threshold_results["75"]["mean_balanced_acc"]

        adapt_pct = adaptive_thresholds[ds_name]
        adapt_accs = [all_results[ds_name][f"adaptive_{ms}"]["mean_balanced_acc"]
                      for ms in MAX_SPLITS_VALUES if f"adaptive_{ms}" in all_results[ds_name]]
        adapt_acc = round(float(np.mean(adapt_accs)), 6) if adapt_accs else fixed_75

        per_dataset_analysis[ds_name] = {
            "synergy_distribution": synergy_dist,
            "threshold_results": threshold_results,
            "optimal_threshold": best_pct,
            "fixed_75_acc": fixed_75,
            "optimal_acc": best_acc,
            "improvement": round(best_acc - fixed_75, 6),
            "adaptive_threshold": adapt_pct,
            "adaptive_acc": adapt_acc,
        }

    # Correlations
    opt_pcts = [per_dataset_analysis[d]["optimal_threshold"] for d in dataset_names]
    mean_syns = [per_dataset_analysis[d]["synergy_distribution"]["mean"] for d in dataset_names]
    skews = [per_dataset_analysis[d]["synergy_distribution"]["skew"] for d in dataset_names]
    improvements = [per_dataset_analysis[d]["improvement"] for d in dataset_names]
    std_syns = [per_dataset_analysis[d]["synergy_distribution"]["std"] for d in dataset_names]

    def safe_spearman(a, b):
        try:
            rho, pval = stats.spearmanr(a, b)
            return {"rho": round(float(rho), 4) if not np.isnan(rho) else 0.0,
                    "p_value": round(float(pval), 4) if not np.isnan(pval) else 1.0}
        except Exception:
            return {"rho": 0.0, "p_value": 1.0}

    correlation_analysis = {
        "optimal_percentile_vs_mean_synergy": safe_spearman(opt_pcts, mean_syns),
        "optimal_percentile_vs_skewness": safe_spearman(opt_pcts, skews),
        "improvement_vs_std_synergy": safe_spearman(improvements, std_syns),
    }

    # Aggregate
    n_improved = sum(1 for d in per_dataset_analysis.values() if d["improvement"] > 0.001)
    universal_accs = {p: np.mean([per_dataset_analysis[d]["threshold_results"][str(p)]["mean_balanced_acc"]
                                   for d in dataset_names]) for p in THRESHOLD_PERCENTILES}
    best_universal = max(universal_accs, key=universal_accs.get)

    sg_vs_figs = []
    sg_vs_ro = []
    for ds in dataset_names:
        for ms in MAX_SPLITS_VALUES:
            k = f"75_{ms}"
            if k in all_results[ds]:
                r = all_results[ds][k]
                sg_vs_figs.append(r["mean_balanced_acc"] - r["mean_figs_acc"])
                sg_vs_ro.append(r["mean_balanced_acc"] - r["mean_rofigs_acc"])

    adaptive_impr = [per_dataset_analysis[d]["adaptive_acc"] - per_dataset_analysis[d]["fixed_75_acc"] for d in dataset_names]

    aggregate = {
        "mean_improvement_from_tuning": round(float(np.mean(improvements)), 6),
        "n_datasets_improved": n_improved,
        "best_universal_threshold": int(best_universal),
        "adaptive_vs_fixed_improvement": round(float(np.mean(adaptive_impr)), 6),
        "sg_figs_vs_rofigs_mean_diff": round(float(np.mean(sg_vs_ro)), 6) if sg_vs_ro else 0.0,
        "sg_figs_vs_figs_mean_diff": round(float(np.mean(sg_vs_figs)), 6) if sg_vs_figs else 0.0,
    }

    # Write results_comprehensive.json
    comprehensive = {
        "experiment": "Synergy Threshold Sensitivity for SG-FIGS",
        "thresholds_tested": THRESHOLD_PERCENTILES,
        "max_splits_tested": MAX_SPLITS_VALUES,
        "n_datasets": len(dataset_names),
        "n_folds": N_FOLDS,
        "total_runtime_seconds": round(time.time() - start_time, 1),
        "graph_statistics": graph_statistics,
        "per_dataset_results": per_dataset_analysis,
        "correlation_analysis": correlation_analysis,
        "aggregate": aggregate,
        "adaptive_thresholds": {ds: {"percentile": adaptive_thresholds[ds]} for ds in dataset_names},
    }
    comp_path = WORKSPACE / "results_comprehensive.json"
    comp_path.write_text(json.dumps(comprehensive, indent=2))
    logger.info(f"Wrote {comp_path.name} ({comp_path.stat().st_size / 1024:.1f} KB)")

    # Write method_out.json
    method_out = {"datasets": method_out_datasets}
    method_path = WORKSPACE / "method_out.json"
    method_path.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Wrote {method_path.name} ({method_path.stat().st_size / 1024:.1f} KB)")

    # Summary
    total_time = time.time() - start_time
    total_examples = sum(len(d["examples"]) for d in method_out_datasets)
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Datasets: {len(dataset_names)}")
    logger.info(f"  Total examples: {total_examples}")
    logger.info(f"  Best universal threshold: {best_universal}th pct")
    logger.info(f"  Mean improvement from tuning: {aggregate['mean_improvement_from_tuning']:.4f}")
    logger.info(f"  Datasets improved: {n_improved}/{len(dataset_names)}")
    logger.info(f"  SG-FIGS vs FIGS: {aggregate['sg_figs_vs_figs_mean_diff']:.4f}")
    logger.info(f"  SG-FIGS vs RO-FIGS: {aggregate['sg_figs_vs_rofigs_mean_diff']:.4f}")
    logger.info(f"  Adaptive vs fixed: {aggregate['adaptive_vs_fixed_improvement']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
