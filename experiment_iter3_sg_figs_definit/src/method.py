#!/usr/bin/env python3
"""SG-FIGS Definitive 5-Method Comparison Experiment.

Compares FIGS, RO-FIGS, SG-FIGS-Hard, SG-FIGS-Soft, and Random-FIGS
across 14 tabular classification benchmarks with pre-computed synergy
matrices, fair complexity matching, corrected interpretability scores,
and random-subset ablation.
"""

import json
import random
import resource
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

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

DATA_PATH_ID2 = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
)
DATA_PATH_ID3 = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_2/gen_art/data_id3_it2__opus/full_data_out.json"
)
SYNERGY_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus/results_comprehensive.json"
)

OUTPUT_PATH = WORKSPACE / "method_out.json"

TIER_ORDER = [
    ["monks2", "banknote", "iris"],
    ["blood", "pima_diabetes", "wine", "heart_statlog"],
    ["vehicle", "climate", "kc2"],
    ["breast_cancer_wisconsin_diagnostic", "ionosphere", "spectf_heart", "sonar"],
]

METHOD_NAMES = ["FIGS", "RO-FIGS", "SG-FIGS-Hard", "SG-FIGS-Soft", "Random-FIGS"]
MAX_SPLITS_GRID = [5, 10, 15, 25]
N_FOLDS = 5
RANDOM_SEED = 42
TIME_LIMIT_SECONDS = 3400  # ~56 min hard limit


# =====================================================================
# 1. Data Loading
# =====================================================================

def load_dataset(dataset_dict: dict) -> dict:
    """Parse one dataset dict from JSON into X, y, feature_names, folds."""
    examples = dataset_dict["examples"]
    first_input = json.loads(examples[0]["input"])
    feature_names = list(first_input.keys())
    n_features = len(feature_names)

    X = np.zeros((len(examples), n_features), dtype=np.float64)
    folds = np.zeros(len(examples), dtype=int)

    # Collect raw labels first to handle string labels
    raw_labels = [ex["output"] for ex in examples]
    unique_labels = sorted(set(raw_labels))

    # Build label mapping: try int conversion first, fall back to alphabetical
    label_map = {}
    try:
        for lbl in unique_labels:
            label_map[lbl] = int(lbl)
    except ValueError:
        # String labels (e.g., "yes"/"no") — map alphabetically to 0, 1, ...
        for idx, lbl in enumerate(unique_labels):
            label_map[lbl] = idx

    # Remap to 0-based contiguous integers
    mapped_values = sorted(set(label_map.values()))
    remap = {v: i for i, v in enumerate(mapped_values)}

    y = np.zeros(len(examples), dtype=int)
    for i, ex in enumerate(examples):
        feat_dict = json.loads(ex["input"])
        X[i] = [feat_dict[fname] for fname in feature_names]
        y[i] = remap[label_map[ex["output"]]]
        folds[i] = int(ex["metadata_fold"])

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "folds": folds,
        "n_classes": int(examples[0]["metadata_n_classes"]),
        "n_features": n_features,
        "domain": examples[0].get("metadata_domain", "unknown"),
        "label_map": label_map,
    }


def load_all_datasets() -> tuple[dict, list]:
    """Load all 14 datasets from both data files. Returns (datasets_dict, raw_entries)."""
    logger.info("Loading datasets from two data files")
    datasets = {}
    raw_entries = []

    for data_path in [DATA_PATH_ID2, DATA_PATH_ID3]:
        logger.info(f"  Loading {data_path.name}")
        data = json.loads(data_path.read_text())
        for ds in data["datasets"]:
            name = ds["dataset"]
            datasets[name] = load_dataset(ds)
            raw_entries.append(ds)
            info = datasets[name]
            logger.info(
                f"    {name}: {info['X'].shape[0]} samples, "
                f"{info['n_features']} features, {info['n_classes']} classes"
            )

    logger.info(f"Loaded {len(datasets)} datasets total")
    return datasets, raw_entries


# =====================================================================
# 2. Pre-computed Synergy Loading
# =====================================================================

def load_precomputed_synergy(json_path: Path) -> dict:
    """Load pre-computed synergy matrices from results_comprehensive.json."""
    logger.info(f"Loading pre-computed synergy from {json_path.name}")
    data = json.loads(json_path.read_text())
    synergy_db = {}

    for entry in data["per_dataset_full"]:
        ds_name = entry["dataset"]
        # Skip diabetes_binarized (not in our data files)
        if ds_name == "diabetes_binarized":
            continue
        # Normalize breast_cancer → breast_cancer_wisconsin_diagnostic
        if ds_name == "breast_cancer":
            continue  # duplicate of breast_cancer_wisconsin_diagnostic

        S = np.array(entry["synergy_matrix"])
        mi_feature_names = list(entry.get("mi_values", {}).keys())
        sg = entry.get("synergy_graph", {})

        synergy_db[ds_name] = {
            "synergy_matrix": S,
            "n_features_used": entry["n_features_used"],
            "mi_feature_names": mi_feature_names,
            "threshold": sg.get("threshold", 0.0),
            "n_edges": sg.get("n_edges", 0),
            "largest_clique_size": sg.get("largest_clique_size", 2),
        }
        logger.info(
            f"  {ds_name}: {S.shape[0]}x{S.shape[1]} synergy, "
            f"{sg.get('n_edges', 0)} edges, clique_size={sg.get('largest_clique_size', 2)}"
        )

    logger.info(f"Loaded synergy for {len(synergy_db)} datasets")
    return synergy_db


def map_synergy_to_full_features(
    synergy_entry: dict, dataset_feature_names: list
) -> tuple[np.ndarray, list]:
    """Map synergy matrix indices to original feature indices.

    Returns (full_synergy_matrix, index_map).
    """
    mi_feature_names = synergy_entry["mi_feature_names"]
    d_full = len(dataset_feature_names)
    S_small = synergy_entry["synergy_matrix"]

    # Build mapping from synergy index -> original feature index
    index_map = []
    for fname in mi_feature_names:
        if fname in dataset_feature_names:
            index_map.append(dataset_feature_names.index(fname))
        else:
            index_map.append(-1)

    # Build full-size synergy matrix (mostly zeros for unmapped features)
    S_full = np.zeros((d_full, d_full))
    for i_small, i_full in enumerate(index_map):
        for j_small, j_full in enumerate(index_map):
            if i_full >= 0 and j_full >= 0 and i_small < S_small.shape[0] and j_small < S_small.shape[1]:
                S_full[i_full, j_full] = S_small[i_small, j_small]

    return S_full, index_map


# =====================================================================
# 3. PID Synergy Computation (for new datasets)
# =====================================================================

def discretize_features(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Discretize continuous features using quantile binning."""
    # Check if features are already categorical (few unique values)
    max_unique = max(len(np.unique(X[:, j])) for j in range(X.shape[1]))
    if max_unique <= n_bins:
        return X.astype(int)
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    X_disc = disc.fit_transform(X).astype(int)
    return X_disc


def compute_pairwise_synergy(
    xi_disc: np.ndarray, xj_disc: np.ndarray, y_disc: np.ndarray
) -> float:
    """Compute PID synergy between features i, j w.r.t. target y."""
    import dit
    from dit.pid import PID_BROJA, PID_WB

    if len(np.unique(xi_disc)) <= 1 or len(np.unique(xj_disc)) <= 1:
        return 0.0

    triples = list(zip(xi_disc.astype(int), xj_disc.astype(int), y_disc.astype(int)))
    counts = Counter(triples)
    total = len(triples)

    max_label = max(int(np.max(xi_disc)), int(np.max(xj_disc)), int(np.max(y_disc)))
    if max_label >= 10:
        outcomes = [f"{a} {b} {c}" for (a, b, c) in counts.keys()]
    else:
        outcomes = [f"{a}{b}{c}" for (a, b, c) in counts.keys()]
    pmf = [v / total for v in counts.values()]

    try:
        d = dit.Distribution(outcomes, pmf)
        n_joint_states = len(counts)
        if n_joint_states > 80:
            result = PID_WB(d)
        else:
            result = PID_BROJA(d)
        synergy = float(result[((0, 1),)])
    except Exception as e:
        logger.debug(f"PID computation failed: {e}")
        synergy = 0.0

    return max(synergy, 0.0)


def build_synergy_matrix_fresh(
    X: np.ndarray, y: np.ndarray, n_bins: int = 5, max_time: float = 300.0
) -> np.ndarray:
    """Compute full pairwise synergy matrix with time budget."""
    X_disc = discretize_features(X, n_bins=n_bins)
    y_disc = y.astype(int)
    d = X_disc.shape[1]
    S = np.zeros((d, d))
    total_pairs = d * (d - 1) // 2
    computed = 0
    t0 = time.time()

    # For high-dim, prefilter by MI
    if d > 20:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_disc, y_disc, discrete_features=True, random_state=42)
        top_features = np.argsort(mi_scores)[-20:]
        pairs = [(i, j) for idx_i, i in enumerate(sorted(top_features))
                 for j in sorted(top_features)[idx_i + 1:]]
        total_pairs = len(pairs)
        logger.info(f"  High-dim ({d} features): synergy on top-20 by MI ({total_pairs} pairs)")
    else:
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    for i, j in pairs:
        elapsed = time.time() - t0
        if elapsed > max_time:
            logger.warning(f"  Synergy time limit ({max_time:.0f}s) after {computed}/{total_pairs}")
            break
        S[i, j] = compute_pairwise_synergy(X_disc[:, i], X_disc[:, j], y_disc)
        S[j, i] = S[i, j]
        computed += 1
        if computed % 30 == 0:
            logger.info(f"  Synergy: {computed}/{total_pairs} ({time.time()-t0:.1f}s)")

    return S


# =====================================================================
# 4. Synergy Graph & Subset Extraction
# =====================================================================

def extract_synergy_subsets(
    S: np.ndarray, threshold_percentile: int = 75
) -> tuple[list, float, list]:
    """Extract feature subsets from synergy matrix.

    Returns (subsets, threshold, clique_sizes).
    """
    import networkx as nx

    d = S.shape[0]
    vals = S[np.triu_indices(d, k=1)]
    pos_vals = vals[vals > 0]

    if len(pos_vals) == 0:
        return [], 0.0, [2]

    # Progressively lower threshold if graph has no edges
    for pct in [threshold_percentile, 50, 25, 0]:
        tau = float(np.percentile(pos_vals, pct)) if pct > 0 else 0.0
        G = nx.Graph()
        G.add_nodes_from(range(d))
        for i in range(d):
            for j in range(i + 1, d):
                if S[i, j] > tau:
                    G.add_edge(i, j, weight=S[i, j])
        if G.number_of_edges() > 0:
            break

    if G.number_of_edges() == 0:
        return [], 0.0, [2]

    subsets = []
    clique_sizes = []
    for clique in nx.find_cliques(G):
        if 2 <= len(clique) <= 5:
            subsets.append(sorted(clique))
            clique_sizes.append(len(clique))

    # Also add edges as size-2 subsets
    for u, v in G.edges():
        pair = sorted([u, v])
        if pair not in subsets:
            subsets.append(pair)
            clique_sizes.append(2)

    if not clique_sizes:
        clique_sizes = [2]

    return subsets, tau, clique_sizes


# =====================================================================
# 5. Oblique FIGS Node
# =====================================================================

class ObliqueFIGSNode:
    """Node supporting both axis-aligned and oblique splits."""

    __slots__ = [
        "feature", "features", "weights", "threshold", "value",
        "idxs", "is_root", "impurity_reduction", "tree_num",
        "left", "right", "depth", "is_oblique", "n_samples",
    ]

    def __init__(
        self,
        feature=None,
        features=None,
        weights=None,
        threshold=None,
        value=None,
        idxs=None,
        is_root=False,
        impurity_reduction=None,
        tree_num=None,
        left=None,
        right=None,
        depth=0,
        is_oblique=False,
        n_samples=0,
    ):
        self.feature = feature
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.value = value
        self.idxs = idxs
        self.is_root = is_root
        self.impurity_reduction = impurity_reduction
        self.tree_num = tree_num
        self.left = left
        self.right = right
        self.depth = depth
        self.is_oblique = is_oblique
        self.n_samples = n_samples


# =====================================================================
# 6. Oblique Split Primitive (Ridge-based)
# =====================================================================

def fit_oblique_split_ridge(
    X: np.ndarray,
    y_residuals: np.ndarray,
    feature_indices: list,
) -> dict | None:
    """Fit oblique split using Ridge regression + 1D stump."""
    X_sub = X[:, feature_indices]

    if X_sub.shape[0] < 5:
        return None

    col_std = np.std(X_sub, axis=0)
    non_const = col_std > 1e-12
    if not np.any(non_const):
        return None

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_sub, y_residuals)
    weights = ridge.coef_.flatten()

    projections = X_sub @ weights
    if np.std(projections) < 1e-12:
        return None

    stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=2)
    stump.fit(projections.reshape(-1, 1), y_residuals)

    tree = stump.tree_
    if tree.feature[0] == -2 or tree.n_node_samples.shape[0] < 3:
        return None

    threshold = tree.threshold[0]
    left_mask = projections <= threshold

    if np.sum(left_mask) < 1 or np.sum(~left_mask) < 1:
        return None

    return {
        "features": np.array(feature_indices),
        "weights": weights,
        "threshold": threshold,
        "left_mask": left_mask,
        "value_left": np.mean(y_residuals[left_mask]),
        "value_right": np.mean(y_residuals[~left_mask]),
    }


# =====================================================================
# 7. BaseFIGSOblique — FIGS greedy loop with oblique support
# =====================================================================

class BaseFIGSOblique:
    """FIGS greedy tree-sum with oblique split support."""

    def __init__(
        self,
        max_splits: int = 25,
        max_trees: int | None = None,
        max_depth: int = 6,
        min_impurity_decrease: float = 0.0,
        num_repetitions: int = 3,
        random_state: int | None = None,
    ):
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.num_repetitions = num_repetitions
        self.random_state = random_state
        self.trees_ = []
        self.complexity_ = 0
        self.scaler_ = None
        self.n_features_ = 0
        self.feature_names_ = None

    def _get_feature_subsets_for_split(self, X: np.ndarray, rng: random.Random) -> list:
        raise NotImplementedError

    @staticmethod
    def _weighted_mse(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.var(y) * len(y))

    def _best_split_for_node(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        idxs: np.ndarray,
        rng: random.Random,
    ) -> dict | None:
        idx_arr = np.where(idxs)[0]
        if len(idx_arr) < 5:
            return None

        X_node = X[idx_arr]
        y_node = residuals[idx_arr]
        parent_mse = self._weighted_mse(y_node)

        best = None
        best_gain = self.min_impurity_decrease

        # Axis-aligned stump
        stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=2)
        stump.fit(X_node, y_node)
        t = stump.tree_
        if t.feature[0] >= 0 and t.n_node_samples.shape[0] >= 3:
            left_sub = X_node[:, t.feature[0]] <= t.threshold[0]
            if 2 <= np.sum(left_sub) <= len(idx_arr) - 2:
                gain = parent_mse - (
                    self._weighted_mse(y_node[left_sub])
                    + self._weighted_mse(y_node[~left_sub])
                )
                if gain > best_gain:
                    best_gain = gain
                    full_left = np.zeros(len(X), dtype=bool)
                    full_left[idx_arr[left_sub]] = True
                    best = {
                        "is_oblique": False,
                        "feature": int(t.feature[0]),
                        "threshold": float(t.threshold[0]),
                        "gain": gain,
                        "left_mask": full_left,
                        "val_left": float(np.mean(y_node[left_sub])),
                        "val_right": float(np.mean(y_node[~left_sub])),
                        "n_left": int(np.sum(left_sub)),
                        "n_right": int(np.sum(~left_sub)),
                    }

        # Oblique splits
        for _ in range(self.num_repetitions):
            subsets = self._get_feature_subsets_for_split(X, rng)
            for feat_idx in subsets:
                if len(feat_idx) < 2:
                    continue
                obl = fit_oblique_split_ridge(X_node, y_node, feat_idx)
                if obl is None:
                    continue
                sub_left = obl["left_mask"]
                if np.sum(sub_left) < 2 or np.sum(~sub_left) < 2:
                    continue
                gain = parent_mse - (
                    self._weighted_mse(y_node[sub_left])
                    + self._weighted_mse(y_node[~sub_left])
                )
                if gain > best_gain:
                    best_gain = gain
                    full_left = np.zeros(len(X), dtype=bool)
                    full_left[idx_arr[sub_left]] = True
                    best = {
                        "is_oblique": True,
                        "features": obl["features"],
                        "weights": obl["weights"],
                        "threshold": obl["threshold"],
                        "gain": gain,
                        "left_mask": full_left,
                        "val_left": float(np.mean(y_node[sub_left])),
                        "val_right": float(np.mean(y_node[~sub_left])),
                        "n_left": int(np.sum(sub_left)),
                        "n_right": int(np.sum(~sub_left)),
                    }

        return best

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list | None = None):
        """Main FIGS greedy loop."""
        rng = random.Random(self.random_state)
        np.random.seed(self.random_state if self.random_state else 42)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.feature_names_ = feature_names

        self.scaler_ = MinMaxScaler()
        X_s = self.scaler_.fit_transform(X)

        y_target = y.astype(float)

        all_idxs = np.ones(n_samples, dtype=bool)
        root_leaf = ObliqueFIGSNode(
            value=float(np.mean(y_target)),
            idxs=all_idxs,
            is_root=True,
            tree_num=0,
            depth=0,
            n_samples=n_samples,
        )
        self.trees_ = [root_leaf]
        leaves = [(0, root_leaf, None, None)]
        total_splits = 0

        while total_splits < self.max_splits and leaves:
            predictions = self._compute_predictions(X_s)
            residuals = y_target - predictions

            scored = []
            for tree_idx, leaf, parent, side in leaves:
                if leaf.depth >= self.max_depth:
                    continue
                split_info = self._best_split_for_node(X_s, residuals, leaf.idxs, rng)
                if split_info is not None:
                    scored.append((split_info["gain"], tree_idx, leaf, parent, side, split_info))

            if not scored:
                # Limit new tree creation to prevent infinite loops
                max_trees_limit = self.max_trees if self.max_trees else min(self.max_splits, 10)
                if len(self.trees_) < max_trees_limit:
                    new_tree_idx = len(self.trees_)
                    new_root = ObliqueFIGSNode(
                        value=float(np.mean(residuals)),
                        idxs=all_idxs,
                        is_root=True,
                        tree_num=new_tree_idx,
                        depth=0,
                        n_samples=n_samples,
                    )
                    self.trees_.append(new_root)
                    leaves.append((new_tree_idx, new_root, None, None))
                    continue
                else:
                    break

            scored.sort(key=lambda x: x[0], reverse=True)
            best_gain, tree_idx, leaf, parent, side, info = scored[0]

            node = ObliqueFIGSNode(
                idxs=leaf.idxs,
                is_root=leaf.is_root,
                tree_num=tree_idx,
                depth=leaf.depth,
                impurity_reduction=best_gain,
                is_oblique=info["is_oblique"],
                n_samples=leaf.n_samples,
            )
            if info["is_oblique"]:
                node.features = info["features"]
                node.weights = info["weights"]
            else:
                node.feature = info["feature"]
            node.threshold = info["threshold"]

            left_idxs = info["left_mask"]
            right_idxs = leaf.idxs & ~left_idxs

            left_leaf = ObliqueFIGSNode(
                value=info["val_left"],
                idxs=left_idxs,
                tree_num=tree_idx,
                depth=leaf.depth + 1,
                n_samples=info["n_left"],
            )
            right_leaf = ObliqueFIGSNode(
                value=info["val_right"],
                idxs=right_idxs,
                tree_num=tree_idx,
                depth=leaf.depth + 1,
                n_samples=info["n_right"],
            )
            node.left = left_leaf
            node.right = right_leaf

            if parent is None:
                self.trees_[tree_idx] = node
            else:
                if side == "left":
                    parent.left = node
                else:
                    parent.right = node

            leaves = [(ti, lf, p, s) for (ti, lf, p, s) in leaves if lf is not leaf]
            leaves.append((tree_idx, left_leaf, node, "left"))
            leaves.append((tree_idx, right_leaf, node, "right"))
            total_splits += 1

        # Final pass: update leaf values
        for t_idx, tree in enumerate(self.trees_):
            other_preds = np.zeros(n_samples)
            for j, other_tree in enumerate(self.trees_):
                if j != t_idx:
                    other_preds += self._predict_tree_vectorized(other_tree, X_s)
            residuals_for_tree = y_target - other_preds
            self._update_leaf_values(tree, residuals_for_tree)

        self.complexity_ = total_splits
        return self

    def _update_leaf_values(self, node: ObliqueFIGSNode, residuals: np.ndarray):
        if node is None:
            return
        if node.left is None and node.right is None:
            if node.idxs is not None and np.any(node.idxs):
                node.value = float(np.mean(residuals[node.idxs]))
            return
        self._update_leaf_values(node.left, residuals)
        self._update_leaf_values(node.right, residuals)

    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree_vectorized(tree, X)
        return preds

    def _predict_tree_vectorized(self, root: ObliqueFIGSNode, X: np.ndarray) -> np.ndarray:
        """Vectorized tree prediction using index partitioning."""
        preds = np.zeros(X.shape[0])
        stack = [(root, np.arange(X.shape[0]))]
        while stack:
            node, indices = stack.pop()
            if indices.size == 0 or node is None:
                continue
            if node.left is None and node.right is None:
                preds[indices] = float(node.value) if node.value is not None else 0.0
                continue

            if node.is_oblique and node.features is not None and node.weights is not None:
                proj = X[np.ix_(indices, node.features)] @ node.weights
                go_left = proj <= node.threshold
            elif node.feature is not None:
                go_left = X[indices, node.feature] <= node.threshold
            else:
                preds[indices] = float(node.value) if node.value is not None else 0.0
                continue

            stack.append((node.left, indices[go_left]))
            stack.append((node.right, indices[~go_left]))
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        preds = self._compute_predictions(X_scaled)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        preds = self._compute_predictions(X_scaled)
        probs = np.clip(preds, 0.0, 1.0)
        return np.vstack((1 - probs, probs)).T


# =====================================================================
# 8. Five Method Subclasses
# =====================================================================

class ROFIGSClassifier(BaseFIGSOblique):
    """RO-FIGS: Random Oblique feature subsets."""

    def __init__(self, subset_size: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.subset_size = subset_size

    def _get_feature_subsets_for_split(self, X: np.ndarray, rng: random.Random) -> list:
        d = X.shape[1]
        k = min(self.subset_size, d)
        indices = list(range(d))
        return [sorted(rng.sample(indices, k))]


class SGFIGSHardClassifier(BaseFIGSOblique):
    """SG-FIGS-Hard: Synergy-constrained oblique splits from cliques/edges."""

    def __init__(self, synergy_subsets: list | None = None, synergy_matrix: np.ndarray | None = None, **kwargs):
        super().__init__(**kwargs)
        self.synergy_subsets = synergy_subsets or []
        self.synergy_matrix = synergy_matrix

    def _get_feature_subsets_for_split(self, X: np.ndarray, rng: random.Random) -> list:
        if not self.synergy_subsets:
            d = X.shape[1]
            i, j = rng.sample(range(d), 2)
            return [[i, j]]
        return [rng.choice(self.synergy_subsets)]


class SGFIGSSoftClassifier(BaseFIGSOblique):
    """SG-FIGS-Soft: Synergy-weighted probabilistic feature sampling."""

    def __init__(self, synergy_matrix: np.ndarray | None = None, subset_size: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.synergy_matrix = synergy_matrix
        self.subset_size = subset_size

    def _get_feature_subsets_for_split(self, X: np.ndarray, rng: random.Random) -> list:
        d = X.shape[1]
        S = self.synergy_matrix
        k = min(self.subset_size, d)

        if S is None or k < 2:
            return [sorted(rng.sample(range(d), k))]

        # Pick seed feature uniformly
        seed = rng.randint(0, d - 1)
        chosen = [seed]
        remaining = set(range(d)) - {seed}

        while len(chosen) < k and remaining:
            candidates = list(remaining)
            scores = np.array([
                sum(S[c, ch] for ch in chosen) / len(chosen) for c in candidates
            ])
            scores = scores + 1e-8
            probs = scores / scores.sum()
            idx = rng.choices(range(len(candidates)), weights=probs.tolist(), k=1)[0]
            chosen.append(candidates[idx])
            remaining.discard(candidates[idx])

        return [sorted(chosen)]


class RandomFIGSAblation(BaseFIGSOblique):
    """Random-FIGS: Ablation control with matched clique sizes."""

    def __init__(self, clique_sizes: list | None = None, **kwargs):
        super().__init__(**kwargs)
        self.clique_sizes = clique_sizes or [2]

    def _get_feature_subsets_for_split(self, X: np.ndarray, rng: random.Random) -> list:
        d = X.shape[1]
        k = rng.choice(self.clique_sizes)
        k = min(k, d)
        k = max(k, 2)
        return [sorted(rng.sample(range(d), k))]


# =====================================================================
# 9. Multi-Class Wrapper (One-vs-Rest)
# =====================================================================

class MultiClassObliqueWrapper:
    """One-vs-Rest wrapper for oblique FIGS methods on multi-class problems."""

    def __init__(self, base_cls, **kwargs):
        self.base_cls = base_cls
        self.kwargs = kwargs
        self.models = {}
        self.classes_ = None
        self.scaler_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list | None = None):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            y_bin = (y == c).astype(float)
            model = self.base_cls(**self.kwargs)
            model.fit(X, y_bin, feature_names=feature_names)
            self.models[c] = model
        # Use first model's scaler for convenience
        self.scaler_ = self.models[self.classes_[0]].scaler_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([
            self.models[c]._compute_predictions(self.models[c].scaler_.transform(X))
            for c in self.classes_
        ])
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([
            self.models[c]._compute_predictions(self.models[c].scaler_.transform(X))
            for c in self.classes_
        ])
        # Simple softmax-like normalization
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    @property
    def trees_(self):
        return [t for m in self.models.values() for t in m.trees_]

    @property
    def complexity_(self):
        return sum(m.complexity_ for m in self.models.values())


# =====================================================================
# 10. FIGS Baseline Wrapper (imodels)
# =====================================================================

class FIGSBaselineWrapper:
    """Wrapper around imodels FIGSClassifier."""

    def __init__(self, max_splits: int = 25):
        from imodels import FIGSClassifier
        self.model = FIGSClassifier(max_rules=max_splits)
        self.max_splits = max_splits
        self.trees_ = []
        self.complexity_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list | None = None):
        self.model.fit(X, y, feature_names=feature_names)
        self.trees_ = self.model.trees_ if hasattr(self.model, "trees_") else []
        self.complexity_ = self.model.complexity_ if hasattr(self.model, "complexity_") else 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# =====================================================================
# 11. Evaluation Metrics
# =====================================================================

def collect_oblique_nodes(node, result: list):
    """Recursively collect oblique internal nodes from ObliqueFIGSNode trees."""
    if node is None:
        return
    if not isinstance(node, ObliqueFIGSNode):
        return
    if node.is_oblique and (node.left is not None or node.right is not None):
        result.append(node)
    if hasattr(node, "left"):
        collect_oblique_nodes(node.left, result)
    if hasattr(node, "right"):
        collect_oblique_nodes(node.right, result)


def count_total_nodes(node) -> int:
    """Count total internal nodes (splits) in a tree."""
    if node is None:
        return 0
    if not hasattr(node, "left") or not hasattr(node, "right"):
        return 0
    if node.left is None and node.right is None:
        return 0
    left_count = count_total_nodes(node.left) if node.left is not None else 0
    right_count = count_total_nodes(node.right) if node.right is not None else 0
    return 1 + left_count + right_count


def count_trees_and_splits(model) -> tuple[int, int]:
    """Count trees and total splits."""
    if isinstance(model, MultiClassObliqueWrapper):
        n_trees = sum(len(m.trees_) for m in model.models.values())
        n_splits = sum(m.complexity_ for m in model.models.values())
        return n_trees, n_splits
    if hasattr(model, "trees_"):
        n_trees = len(model.trees_)
        n_splits = sum(count_total_nodes(t) for t in model.trees_)
    elif hasattr(model, "model") and hasattr(model.model, "trees_"):
        n_trees = len(model.model.trees_)
        n_splits = model.model.complexity_ if hasattr(model.model, "complexity_") else 0
    else:
        n_trees = 0
        n_splits = 0
    return n_trees, n_splits


def compute_avg_features_per_split(model) -> float:
    """Average number of active features per oblique split."""
    trees = []
    if isinstance(model, MultiClassObliqueWrapper):
        trees = model.trees_
    elif hasattr(model, "trees_"):
        trees = model.trees_
    else:
        return 1.0

    oblique_nodes = []
    for tree_root in trees:
        if isinstance(tree_root, ObliqueFIGSNode):
            collect_oblique_nodes(tree_root, oblique_nodes)

    if not oblique_nodes:
        return 1.0

    counts = []
    for node in oblique_nodes:
        if node.features is not None:
            n_nonzero = int(np.sum(np.abs(node.weights) > 1e-10)) if node.weights is not None else len(node.features)
            counts.append(n_nonzero)
        else:
            counts.append(1)
    return float(np.mean(counts)) if counts else 1.0


def compute_split_interpretability_score(
    model, synergy_matrix: np.ndarray
) -> float:
    """Fraction of oblique splits whose features have above-median synergy.

    KEY FIX: Use ALL feature pairs for median (not just positive ones).
    """
    d = synergy_matrix.shape[0]
    all_synergies = synergy_matrix[np.triu_indices(d, k=1)]
    # Use ALL pairs for median, not just positive ones
    median_synergy = float(np.median(all_synergies))

    trees = []
    if isinstance(model, MultiClassObliqueWrapper):
        trees = model.trees_
    elif hasattr(model, "trees_"):
        trees = model.trees_
    else:
        return float("nan")

    oblique_splits = []
    for tree_root in trees:
        if isinstance(tree_root, ObliqueFIGSNode):
            collect_oblique_nodes(tree_root, oblique_splits)

    if not oblique_splits:
        return float("nan")

    above_median = 0
    total_scored = 0
    for node in oblique_splits:
        feats = node.features
        if feats is None or len(feats) < 2:
            continue

        # Only count features with non-zero weights
        if node.weights is not None:
            active = [feats[i] for i in range(len(feats)) if abs(node.weights[i]) > 1e-10]
        else:
            active = list(feats)

        if len(active) < 2:
            continue

        pair_syns = []
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                fi, fj = int(active[ii]), int(active[jj])
                if fi < d and fj < d:
                    pair_syns.append(synergy_matrix[fi, fj])

        if pair_syns:
            mean_syn = np.mean(pair_syns)
            if mean_syn > median_synergy:
                above_median += 1
            total_scored += 1

    return above_median / total_scored if total_scored > 0 else float("nan")


# =====================================================================
# 12. Model Factory
# =====================================================================

def create_model(
    method_name: str,
    max_splits: int,
    synergy_subsets: list | None,
    synergy_matrix: np.ndarray | None,
    clique_sizes: list | None,
    median_clique_size: int,
    is_binary: bool,
):
    """Create a model by method name, with multi-class wrapping if needed."""
    oblique_kwargs = {
        "max_splits": max_splits,
        "random_state": RANDOM_SEED,
        "num_repetitions": 1,
        "max_depth": 6,
    }

    if method_name == "FIGS":
        return FIGSBaselineWrapper(max_splits=max_splits)

    elif method_name == "RO-FIGS":
        cls = ROFIGSClassifier
        kwargs = {**oblique_kwargs, "subset_size": median_clique_size}
        if is_binary:
            return cls(**kwargs)
        return MultiClassObliqueWrapper(base_cls=cls, **kwargs)

    elif method_name == "SG-FIGS-Hard":
        cls = SGFIGSHardClassifier
        kwargs = {
            **oblique_kwargs,
            "synergy_subsets": synergy_subsets,
            "synergy_matrix": synergy_matrix,
        }
        if is_binary:
            return cls(**kwargs)
        return MultiClassObliqueWrapper(base_cls=cls, **kwargs)

    elif method_name == "SG-FIGS-Soft":
        cls = SGFIGSSoftClassifier
        kwargs = {
            **oblique_kwargs,
            "synergy_matrix": synergy_matrix,
            "subset_size": median_clique_size,
        }
        if is_binary:
            return cls(**kwargs)
        return MultiClassObliqueWrapper(base_cls=cls, **kwargs)

    elif method_name == "Random-FIGS":
        cls = RandomFIGSAblation
        kwargs = {**oblique_kwargs, "clique_sizes": clique_sizes}
        if is_binary:
            return cls(**kwargs)
        return MultiClassObliqueWrapper(base_cls=cls, **kwargs)

    else:
        raise ValueError(f"Unknown method: {method_name}")


# =====================================================================
# 13. Per-Dataset Evaluation
# =====================================================================

def evaluate_method_on_dataset(
    method_name: str,
    ds: dict,
    synergy_matrix: np.ndarray,
    synergy_subsets: list,
    clique_sizes: list,
    median_clique_size: int,
) -> dict:
    """Evaluate a method on a dataset with hyperparameter tuning + 5-fold CV."""
    X, y = ds["X"], ds["y"]
    folds = ds["folds"]
    is_binary = ds["n_classes"] == 2
    feature_names = ds["feature_names"]

    best_config = None
    best_val_score = -1.0

    # Use smaller grid for multi-class (OvR multiplies work by n_classes)
    is_binary = ds["n_classes"] == 2
    tuning_grid = MAX_SPLITS_GRID if is_binary else [5, 10, 15]

    # Hyperparameter tuning on fold 0
    for max_splits in tuning_grid:
        try:
            train_idx = folds != 0
            val_idx = folds == 0
            if np.sum(train_idx) < 5 or np.sum(val_idx) < 2:
                continue

            model = create_model(
                method_name=method_name,
                max_splits=max_splits,
                synergy_subsets=synergy_subsets,
                synergy_matrix=synergy_matrix,
                clique_sizes=clique_sizes,
                median_clique_size=median_clique_size,
                is_binary=is_binary,
            )
            model.fit(X[train_idx], y[train_idx], feature_names=feature_names)
            y_pred = model.predict(X[val_idx])
            val_score = balanced_accuracy_score(y[val_idx], y_pred)

            if val_score > best_val_score:
                best_val_score = val_score
                best_config = max_splits
        except Exception:
            logger.debug(f"  {method_name} ms={max_splits} tuning failed")
            continue

    if best_config is None:
        best_config = 10

    logger.info(f"    {method_name}: best_ms={best_config} (val={best_val_score:.4f})")

    # 5-fold evaluation
    fold_results = []
    for fold_id in range(N_FOLDS):
        try:
            test_idx = folds == fold_id
            train_idx = ~test_idx
            if np.sum(train_idx) < 5 or np.sum(test_idx) < 2:
                continue

            model = create_model(
                method_name=method_name,
                max_splits=best_config,
                synergy_subsets=synergy_subsets,
                synergy_matrix=synergy_matrix,
                clique_sizes=clique_sizes,
                median_clique_size=median_clique_size,
                is_binary=is_binary,
            )
            model.fit(X[train_idx], y[train_idx], feature_names=feature_names)
            y_pred = model.predict(X[test_idx])

            bal_acc = balanced_accuracy_score(y[test_idx], y_pred)

            auc = None
            if is_binary:
                try:
                    y_proba = model.predict_proba(X[test_idx])
                    if y_proba.shape[1] >= 2:
                        auc = roc_auc_score(y[test_idx], y_proba[:, 1])
                except Exception:
                    pass

            n_trees, n_splits = count_trees_and_splits(model)
            avg_feat = compute_avg_features_per_split(model) if method_name != "FIGS" else 1.0
            interp = compute_split_interpretability_score(model, synergy_matrix)

            fold_results.append({
                "fold": fold_id,
                "balanced_accuracy": round(bal_acc, 6),
                "auc": round(auc, 6) if auc is not None else None,
                "n_splits": n_splits,
                "n_trees": n_trees,
                "avg_features_per_split": round(avg_feat, 4),
                "interpretability_score": round(interp, 4) if not np.isnan(interp) else None,
            })
        except Exception:
            logger.exception(f"  {method_name} fold {fold_id} failed")
            continue

    if not fold_results:
        return {"error": "all folds failed"}

    accs = [r["balanced_accuracy"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results if r["auc"] is not None]
    splits = [r["n_splits"] for r in fold_results]
    interps = [r["interpretability_score"] for r in fold_results if r["interpretability_score"] is not None]

    return {
        "best_max_splits": best_config,
        "fold_results": fold_results,
        "mean_balanced_accuracy": round(float(np.mean(accs)), 6),
        "std_balanced_accuracy": round(float(np.std(accs)), 6),
        "mean_auc": round(float(np.mean(aucs)), 6) if aucs else None,
        "mean_n_splits": round(float(np.mean(splits)), 2),
        "mean_avg_features_per_split": round(float(np.mean([r["avg_features_per_split"] for r in fold_results])), 4),
        "mean_interpretability": round(float(np.mean(interps)), 4) if interps else None,
    }


# =====================================================================
# 14. Domain Validation
# =====================================================================

def collect_domain_analysis(model, feature_names: list) -> list:
    """Collect oblique split details for domain validation."""
    trees = []
    if isinstance(model, MultiClassObliqueWrapper):
        trees = model.trees_
    elif hasattr(model, "trees_"):
        trees = model.trees_

    splits_info = []
    for tree_root in trees:
        if not isinstance(tree_root, ObliqueFIGSNode):
            continue
        oblique_nodes = []
        collect_oblique_nodes(tree_root, oblique_nodes)
        for node in oblique_nodes:
            if node.features is not None and node.weights is not None:
                feat_details = [
                    {
                        "feature": feature_names[f] if f < len(feature_names) else f"f{f}",
                        "weight": round(float(w), 4),
                        "index": int(f),
                    }
                    for f, w in zip(node.features, node.weights)
                    if abs(w) > 1e-10
                ]
                if feat_details:
                    splits_info.append({
                        "features": feat_details,
                        "threshold": round(float(node.threshold), 4),
                        "depth": node.depth,
                    })
    return splits_info


# =====================================================================
# 15. Main Experiment
# =====================================================================

def run_experiment(
    datasets: dict,
    synergy_db: dict,
) -> tuple[dict, dict, dict]:
    """Run the full 5-method comparison across all datasets."""
    all_results = {}
    all_synergy_info = {}
    domain_analysis = {}
    total_start = time.time()

    for tier_idx, tier_datasets in enumerate(TIER_ORDER):
        logger.info(f"\n{'='*60}")
        logger.info(f"TIER {tier_idx + 1}: {tier_datasets}")
        logger.info(f"{'='*60}")

        for ds_name in tier_datasets:
            if ds_name not in datasets:
                logger.warning(f"Dataset {ds_name} not found, skipping")
                continue

            ds = datasets[ds_name]
            X, y = ds["X"], ds["y"]
            is_binary = ds["n_classes"] == 2
            feature_names = ds["feature_names"]

            logger.info(
                f"\n--- {ds_name}: {X.shape[0]} samples, "
                f"{X.shape[1]} features, {ds['n_classes']} classes ---"
            )

            ds_start = time.time()

            # Get or compute synergy matrix
            if ds_name in synergy_db:
                logger.info(f"  Using pre-computed synergy")
                se = synergy_db[ds_name]
                # Map synergy indices to original feature indices if needed
                if se["n_features_used"] < ds["n_features"]:
                    synergy_matrix, _ = map_synergy_to_full_features(se, feature_names)
                else:
                    synergy_matrix = se["synergy_matrix"]
            else:
                logger.info(f"  Computing synergy from scratch...")
                train_mask = ds["folds"] != 0
                synergy_matrix = build_synergy_matrix_fresh(
                    X[train_mask], y[train_mask], n_bins=5, max_time=300.0
                )
                # Expand to full feature size if computed on subset
                if synergy_matrix.shape[0] < X.shape[1]:
                    S_full = np.zeros((X.shape[1], X.shape[1]))
                    n = synergy_matrix.shape[0]
                    S_full[:n, :n] = synergy_matrix
                    synergy_matrix = S_full
                logger.info(f"  Synergy computed in {time.time()-ds_start:.1f}s")

            # Extract synergy subsets and clique sizes
            synergy_subsets, tau, clique_sizes = extract_synergy_subsets(
                synergy_matrix, threshold_percentile=75
            )
            median_clique_size = int(np.median(clique_sizes)) if clique_sizes else 2
            median_clique_size = max(median_clique_size, 2)

            logger.info(
                f"  Synergy: {len(synergy_subsets)} subsets, "
                f"median_clique={median_clique_size}, threshold={tau:.4f}"
            )

            all_synergy_info[ds_name] = {
                "synergy_matrix": synergy_matrix.tolist(),
                "threshold": float(tau),
                "n_subsets": len(synergy_subsets),
                "median_clique_size": median_clique_size,
                "clique_sizes": clique_sizes,
            }

            # Evaluate all 5 methods
            ds_results = {}
            for method_name in METHOD_NAMES:
                method_start = time.time()
                logger.info(f"  Evaluating {method_name}...")

                try:
                    method_result = evaluate_method_on_dataset(
                        method_name=method_name,
                        ds=ds,
                        synergy_matrix=synergy_matrix,
                        synergy_subsets=synergy_subsets,
                        clique_sizes=clique_sizes,
                        median_clique_size=median_clique_size,
                    )
                    method_time = time.time() - method_start

                    if "error" not in method_result:
                        logger.info(
                            f"    acc={method_result['mean_balanced_accuracy']:.4f} "
                            f"±{method_result['std_balanced_accuracy']:.4f}, "
                            f"splits={method_result['mean_n_splits']:.1f}, "
                            f"interp={method_result.get('mean_interpretability', 'N/A')}, "
                            f"time={method_time:.1f}s"
                        )
                    else:
                        logger.warning(f"    FAILED: {method_result['error']}")

                    ds_results[method_name] = method_result
                except Exception:
                    logger.exception(f"  {method_name} completely failed")
                    ds_results[method_name] = {"error": "exception"}

            all_results[ds_name] = ds_results

            # Domain validation for key datasets
            domain_datasets = [
                "pima_diabetes", "breast_cancer_wisconsin_diagnostic",
                "heart_statlog", "monks2",
            ]
            if ds_name in domain_datasets:
                # Train one final SG-FIGS-Hard model on full data for domain analysis
                try:
                    sg_model = create_model(
                        method_name="SG-FIGS-Hard",
                        max_splits=15,
                        synergy_subsets=synergy_subsets,
                        synergy_matrix=synergy_matrix,
                        clique_sizes=clique_sizes,
                        median_clique_size=median_clique_size,
                        is_binary=is_binary,
                    )
                    sg_model.fit(X, y, feature_names=feature_names)
                    domain_analysis[ds_name] = collect_domain_analysis(sg_model, feature_names)
                    logger.info(f"  Domain analysis: {len(domain_analysis[ds_name])} oblique splits")
                except Exception:
                    logger.exception(f"  Domain analysis failed for {ds_name}")

            ds_time = time.time() - ds_start
            logger.info(f"  {ds_name} complete in {ds_time:.1f}s")

            # Time check
            total_elapsed = time.time() - total_start
            if total_elapsed > TIME_LIMIT_SECONDS:
                logger.warning(f"Time limit reached ({total_elapsed/60:.1f} min). Stopping.")
                break

        total_elapsed = time.time() - total_start
        if total_elapsed > TIME_LIMIT_SECONDS:
            logger.warning(f"Time limit reached in outer loop. Skipping remaining tiers.")
            break

        # Tier summary
        logger.info(f"\n--- Tier {tier_idx + 1} Summary ---")
        for dn in tier_datasets:
            if dn in all_results:
                parts = []
                for mn in METHOD_NAMES:
                    mr = all_results[dn].get(mn, {})
                    acc = mr.get("mean_balanced_accuracy")
                    if acc is not None:
                        parts.append(f"{mn}={acc:.4f}")
                logger.info(f"  {dn}: {', '.join(parts)}")

    total_time = time.time() - total_start
    logger.info(f"\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return all_results, all_synergy_info, domain_analysis


# =====================================================================
# 16. Success Criteria & Aggregate Analysis
# =====================================================================

def evaluate_success_criteria(results: dict) -> dict:
    """Evaluate success criteria for SG-FIGS."""
    sg_accs, ro_accs, rand_accs = [], [], []
    sg_splits, ro_splits = [], []
    sg_interps, ro_interps, rand_interps = [], [], []

    for ds_results in results.values():
        for method, key_list in [
            ("SG-FIGS-Hard", sg_accs),
            ("RO-FIGS", ro_accs),
            ("Random-FIGS", rand_accs),
        ]:
            mr = ds_results.get(method, {})
            acc = mr.get("mean_balanced_accuracy")
            if acc is not None:
                key_list.append(acc)

        for method, key_list in [
            ("SG-FIGS-Hard", sg_splits),
            ("RO-FIGS", ro_splits),
        ]:
            mr = ds_results.get(method, {})
            ns = mr.get("mean_n_splits")
            if ns is not None:
                key_list.append(ns)

        for method, key_list in [
            ("SG-FIGS-Hard", sg_interps),
            ("RO-FIGS", ro_interps),
            ("Random-FIGS", rand_interps),
        ]:
            mr = ds_results.get(method, {})
            interp = mr.get("mean_interpretability")
            if interp is not None:
                key_list.append(interp)

    # Criterion 1: SG-FIGS accuracy within 1% of RO-FIGS with 20%+ fewer splits
    acc_diff = float(np.mean(sg_accs) - np.mean(ro_accs)) if sg_accs and ro_accs else None
    split_reduction = (1.0 - np.mean(sg_splits) / np.mean(ro_splits)) if sg_splits and ro_splits else None
    criterion_1 = None
    if acc_diff is not None and split_reduction is not None:
        criterion_1 = (abs(acc_diff) < 0.01 and split_reduction >= 0.20) or (acc_diff > 0.01)

    # Criterion 2: SG-FIGS interpretability > RO-FIGS
    criterion_2 = None
    if sg_interps and ro_interps:
        criterion_2 = float(np.mean(sg_interps)) > float(np.mean(ro_interps))

    # Ablation: SG-FIGS-Hard vs Random-FIGS
    ablation_acc = float(np.mean(sg_accs) - np.mean(rand_accs)) if sg_accs and rand_accs else None
    ablation_interp = float(np.mean(sg_interps) - np.mean(rand_interps)) if sg_interps and rand_interps else None

    return {
        "criterion_1_competitive_accuracy_fewer_splits": criterion_1,
        "criterion_1_acc_diff": round(acc_diff, 6) if acc_diff is not None else None,
        "criterion_1_split_reduction": round(split_reduction, 4) if split_reduction is not None else None,
        "criterion_2_higher_interpretability": criterion_2,
        "criterion_2_sg_interp": round(float(np.mean(sg_interps)), 4) if sg_interps else None,
        "criterion_2_ro_interp": round(float(np.mean(ro_interps)), 4) if ro_interps else None,
        "ablation_sg_vs_random_acc": round(ablation_acc, 6) if ablation_acc is not None else None,
        "ablation_sg_vs_random_interp": round(ablation_interp, 4) if ablation_interp is not None else None,
    }


def build_aggregate_comparison(results: dict) -> dict:
    """Build per-method aggregate statistics."""
    agg = {}
    for method in METHOD_NAMES:
        accs, aucs, splits, interps = [], [], [], []
        for ds_results in results.values():
            mr = ds_results.get(method, {})
            if "mean_balanced_accuracy" in mr:
                accs.append(mr["mean_balanced_accuracy"])
            if mr.get("mean_auc") is not None:
                aucs.append(mr["mean_auc"])
            if "mean_n_splits" in mr:
                splits.append(mr["mean_n_splits"])
            if mr.get("mean_interpretability") is not None:
                interps.append(mr["mean_interpretability"])

        agg[method] = {
            "mean_balanced_accuracy": round(float(np.mean(accs)), 6) if accs else None,
            "std_balanced_accuracy": round(float(np.std(accs)), 6) if accs else None,
            "mean_auc": round(float(np.mean(aucs)), 6) if aucs else None,
            "mean_n_splits": round(float(np.mean(splits)), 2) if splits else None,
            "mean_interpretability": round(float(np.mean(interps)), 4) if interps else None,
            "n_datasets_evaluated": len(accs),
        }
    return agg


# =====================================================================
# 17. Output Formatting
# =====================================================================

def format_schema_output(
    results: dict, raw_entries: list
) -> dict:
    """Format output per exp_gen_sol_out schema: {datasets: [...]}."""
    output_datasets = []

    for ds_entry in raw_entries:
        ds_name = ds_entry["dataset"]
        ds_results = results.get(ds_name, {})

        output_examples = []
        for ex in ds_entry["examples"]:
            out_ex = {
                "input": ex["input"],
                "output": ex["output"],
            }
            for key, val in ex.items():
                if key.startswith("metadata_"):
                    out_ex[key] = val

            fold_id = ex.get("metadata_fold", 0)

            for method_name in METHOD_NAMES:
                safe_name = method_name.replace("-", "_").lower()
                mr = ds_results.get(method_name, {})
                if "mean_balanced_accuracy" in mr:
                    fold_res = None
                    for fr in mr.get("fold_results", []):
                        if fr["fold"] == fold_id:
                            fold_res = fr
                            break

                    if fold_res:
                        out_ex[f"predict_{safe_name}"] = json.dumps({
                            "balanced_accuracy": fold_res["balanced_accuracy"],
                            "auc": fold_res["auc"],
                            "n_splits": fold_res["n_splits"],
                            "n_trees": fold_res["n_trees"],
                            "interpretability_score": fold_res.get("interpretability_score"),
                        })
                    else:
                        out_ex[f"predict_{safe_name}"] = json.dumps({
                            "mean_balanced_accuracy": mr["mean_balanced_accuracy"],
                            "mean_auc": mr.get("mean_auc"),
                        })
                else:
                    out_ex[f"predict_{safe_name}"] = json.dumps({"error": mr.get("error", "not_evaluated")})

            output_examples.append(out_ex)

        output_datasets.append({
            "dataset": ds_name,
            "examples": output_examples,
        })

    return {"datasets": output_datasets}


def make_serializable(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj


# =====================================================================
# 18. Main Entry Point
# =====================================================================

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("SG-FIGS Definitive 5-Method Comparison Experiment")
    logger.info("=" * 60)

    # Load all datasets
    datasets, raw_entries = load_all_datasets()

    # Load pre-computed synergy
    synergy_db = load_precomputed_synergy(SYNERGY_PATH)

    # Run experiment
    results, synergy_info, domain_analysis = run_experiment(datasets, synergy_db)

    # Build aggregate comparison
    agg = build_aggregate_comparison(results)
    success = evaluate_success_criteria(results)

    # Format schema-compliant output
    schema_output = format_schema_output(results, raw_entries)

    # Save schema-compliant output
    serializable_output = make_serializable(schema_output)
    OUTPUT_PATH.write_text(json.dumps(serializable_output, indent=2))
    logger.info(f"Schema output saved to {OUTPUT_PATH}")

    # Save comprehensive results
    comprehensive = {
        "experiment": "SG-FIGS Definitive 5-Method Comparison",
        "methods": METHOD_NAMES,
        "n_datasets": len(results),
        "datasets_evaluated": list(results.keys()),
        "per_dataset_results": make_serializable(results),
        "aggregate_comparison": make_serializable(agg),
        "success_criteria": make_serializable(success),
        "domain_analysis": make_serializable(domain_analysis),
        "synergy_stats": make_serializable({
            ds: {k: v for k, v in info.items() if k != "synergy_matrix"}
            for ds, info in synergy_info.items()
        }),
    }
    comp_path = WORKSPACE / "results_comprehensive.json"
    comp_path.write_text(json.dumps(comprehensive, indent=2))
    logger.info(f"Comprehensive results saved to {comp_path}")

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")

    header = f"{'Dataset':<35} {'FIGS':>7} {'RO':>7} {'SG-H':>7} {'SG-S':>7} {'Rand':>7} {'Interp':>7}"
    logger.info(header)
    logger.info("-" * len(header))

    for ds_name in results:
        dr = results[ds_name]
        vals = []
        for mn in METHOD_NAMES:
            acc = dr.get(mn, {}).get("mean_balanced_accuracy")
            vals.append(f"{acc:.4f}" if acc is not None else "  N/A ")
        interp = dr.get("SG-FIGS-Hard", {}).get("mean_interpretability")
        interp_str = f"{interp:.4f}" if interp is not None else "  N/A "
        logger.info(f"{ds_name:<35} {vals[0]:>7} {vals[1]:>7} {vals[2]:>7} {vals[3]:>7} {vals[4]:>7} {interp_str:>7}")

    logger.info("-" * len(header))
    for mn in METHOD_NAMES:
        avg = agg[mn].get("mean_balanced_accuracy")
        avg_str = f"{avg:.4f}" if avg is not None else "N/A"
        interp = agg[mn].get("mean_interpretability")
        interp_str = f"{interp:.4f}" if interp is not None else "N/A"
        logger.info(f"  {mn}: avg_acc={avg_str}, avg_interp={interp_str}, n={agg[mn]['n_datasets_evaluated']}")

    logger.info(f"\nSuccess Criteria:")
    for k, v in success.items():
        logger.info(f"  {k}: {v}")

    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
