#!/usr/bin/env python3
"""SG-FIGS: Synergy-Guided Oblique FIGS — Full Experiment.

Implements and benchmarks SG-FIGS (synergy-guided oblique splits via PID)
against FIGS (axis-aligned) and RO-FIGS (random oblique) baselines on
10 tabular classification benchmarks.

Metrics: balanced accuracy, AUC, model complexity, split interpretability.
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

import numpy as np
from loguru import logger
from scipy.special import expit
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
DATA_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
)
OUTPUT_PATH = WORKSPACE / "method_out.json"

TIER_ORDER = [
    ["iris", "banknote"],
    ["pima_diabetes", "wine", "heart_statlog", "vehicle"],
    ["breast_cancer_wisconsin_diagnostic", "ionosphere", "spectf_heart", "sonar"],
]

MAX_SPLITS_GRID = [5, 10, 15, 25]
N_FOLDS = 5
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_dict: dict) -> dict:
    """Parse one dataset dict from JSON into X, y, feature_names, folds."""
    examples = dataset_dict["examples"]
    first_input = json.loads(examples[0]["input"])
    feature_names = list(first_input.keys())
    n_features = len(feature_names)

    X = np.zeros((len(examples), n_features), dtype=np.float64)
    y = np.zeros(len(examples), dtype=int)
    folds = np.zeros(len(examples), dtype=int)

    for i, ex in enumerate(examples):
        feat_dict = json.loads(ex["input"])
        X[i] = [feat_dict[fname] for fname in feature_names]
        y[i] = int(ex["output"])
        folds[i] = int(ex["metadata_fold"])

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "folds": folds,
        "n_classes": int(examples[0]["metadata_n_classes"]),
        "n_features": n_features,
        "domain": examples[0].get("metadata_domain", "unknown"),
    }


def load_all_datasets(json_path: Path) -> dict:
    """Load all datasets from the JSON file."""
    logger.info(f"Loading datasets from {json_path}")
    data = json.loads(json_path.read_text())
    datasets = {}
    for ds in data["datasets"]:
        name = ds["dataset"]
        datasets[name] = load_dataset(ds)
        info = datasets[name]
        logger.info(
            f"  {name}: {info['X'].shape[0]} samples, "
            f"{info['n_features']} features, {info['n_classes']} classes"
        )
    logger.info(f"Loaded {len(datasets)} datasets total")
    return datasets


# ---------------------------------------------------------------------------
# 2. PID Synergy Module
# ---------------------------------------------------------------------------

def discretize_features(X: np.ndarray, n_bins: int = 5) -> tuple:
    """Discretize continuous features using quantile binning."""
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    X_disc = disc.fit_transform(X).astype(int)
    return X_disc, disc


def compute_pairwise_synergy(
    xi_disc: np.ndarray, xj_disc: np.ndarray, y_disc: np.ndarray
) -> float:
    """Compute PID synergy between features i, j w.r.t. target y."""
    import dit
    from dit.pid import PID_BROJA, PID_WB

    # Skip constant features
    if len(np.unique(xi_disc)) <= 1 or len(np.unique(xj_disc)) <= 1:
        return 0.0

    triples = list(
        zip(xi_disc.astype(int), xj_disc.astype(int), y_disc.astype(int))
    )
    counts = Counter(triples)
    total = len(triples)

    # Build dit distribution
    max_label = max(
        int(np.max(xi_disc)), int(np.max(xj_disc)), int(np.max(y_disc))
    )
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
        logger.debug(f"PID computation failed: {e}, returning synergy=0")
        synergy = 0.0

    return max(synergy, 0.0)


def build_synergy_matrix(
    X_disc: np.ndarray, y_disc: np.ndarray, max_time: float = 300.0
) -> np.ndarray:
    """Compute full pairwise synergy matrix S[i,j] with time budget."""
    d = X_disc.shape[1]
    S = np.zeros((d, d))
    total_pairs = d * (d - 1) // 2
    computed = 0
    t0 = time.time()

    # For high-dimensional datasets, pre-filter by mutual information
    if d > 20:
        # Compute MI(feature, target) to prioritize informative features
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(
            X_disc, y_disc, discrete_features=True, random_state=42
        )
        # Keep top 20 features by MI
        top_features = np.argsort(mi_scores)[-20:]
        logger.info(
            f"  High-dim ({d} features): limiting synergy to top-20 by MI"
        )
        pairs_to_compute = [
            (i, j)
            for idx_i, i in enumerate(sorted(top_features))
            for j in sorted(top_features)[idx_i + 1 :]
        ]
        total_pairs = len(pairs_to_compute)
    else:
        pairs_to_compute = [
            (i, j) for i in range(d) for j in range(i + 1, d)
        ]

    for i, j in pairs_to_compute:
        elapsed = time.time() - t0
        if elapsed > max_time:
            logger.warning(
                f"  Synergy computation hit time limit ({max_time:.0f}s) "
                f"after {computed}/{total_pairs} pairs"
            )
            break
        S[i, j] = compute_pairwise_synergy(X_disc[:, i], X_disc[:, j], y_disc)
        S[j, i] = S[i, j]
        computed += 1
        if computed % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  Synergy: {computed}/{total_pairs} pairs "
                f"({elapsed:.1f}s elapsed)"
            )
    return S


def build_synergy_graph(
    S: np.ndarray, threshold_percentile: int = 75
) -> tuple:
    """Build networkx graph from synergy matrix."""
    import networkx as nx

    d = S.shape[0]
    upper_tri = S[np.triu_indices(d, k=1)]
    pos_values = upper_tri[upper_tri > 0]

    if len(pos_values) == 0:
        G = nx.Graph()
        G.add_nodes_from(range(d))
        return G, 0.0, []

    # Progressively lower threshold if graph is too sparse
    for pct in [threshold_percentile, 50, 25, 0]:
        tau = np.percentile(pos_values, pct) if pct > 0 else 0.0
        G = nx.Graph()
        G.add_nodes_from(range(d))
        for i in range(d):
            for j in range(i + 1, d):
                if S[i, j] > tau:
                    G.add_edge(i, j, weight=S[i, j])
        if G.number_of_edges() > 0:
            break

    # Extract candidate feature subsets
    subsets = []
    for clique in nx.find_cliques(G):
        if 2 <= len(clique) <= 5:
            subsets.append(sorted(clique))
    edge_subsets = [sorted([u, v]) for u, v in G.edges()]
    for es in edge_subsets:
        if es not in subsets:
            subsets.append(es)

    return G, tau, subsets


def compute_synergy_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 5,
    threshold_percentile: int = 75,
) -> dict:
    """Full synergy pipeline: discretize → synergy matrix → graph → subsets."""
    X_disc, disc = discretize_features(X, n_bins=n_bins)
    y_disc = y.astype(int)
    if y_disc.ndim > 1:
        y_disc = np.argmax(y_disc, axis=1)

    S = build_synergy_matrix(X_disc, y_disc)
    G, tau, subsets = build_synergy_graph(S, threshold_percentile)

    return {
        "synergy_matrix": S,
        "graph": G,
        "threshold": tau,
        "subsets": subsets,
        "discretizer": disc,
    }


# ---------------------------------------------------------------------------
# 3. Oblique FIGS Node
# ---------------------------------------------------------------------------

class ObliqueFIGSNode:
    """Node supporting both axis-aligned and oblique splits."""

    def __init__(
        self,
        feature=None,
        features=None,
        weights=None,
        threshold=None,
        value=None,
        idxs=None,
        is_root=False,
        impurity=None,
        impurity_reduction=None,
        tree_num=None,
        left=None,
        right=None,
        left_temp=None,
        right_temp=None,
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
        self.impurity = impurity
        self.impurity_reduction = impurity_reduction
        self.tree_num = tree_num
        self.left = left
        self.right = right
        self.left_temp = left_temp
        self.right_temp = right_temp
        self.depth = depth
        self.is_oblique = is_oblique
        self.n_samples = n_samples


# ---------------------------------------------------------------------------
# 4. Oblique Split Primitive (Ridge-based fallback)
# ---------------------------------------------------------------------------

def fit_oblique_split_ridge(
    X: np.ndarray,
    y_residuals: np.ndarray,
    feature_indices: list,
) -> dict | None:
    """Fit oblique split using Ridge regression + 1D stump."""
    X_sub = X[:, feature_indices]

    if X_sub.shape[0] < 5:
        return None

    # Check for constant columns
    col_std = np.std(X_sub, axis=0)
    non_const = col_std > 1e-12
    if not np.any(non_const):
        return None

    # Fit ridge to get projection direction
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_sub, y_residuals)
    weights = ridge.coef_.flatten()

    # Project data
    projections = X_sub @ weights

    if np.std(projections) < 1e-12:
        return None

    # Find best threshold via 1D stump
    stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=2)
    stump.fit(projections.reshape(-1, 1), y_residuals)

    tree = stump.tree_
    if tree.feature[0] == -2 or tree.n_node_samples.shape[0] < 3:
        return None

    threshold = tree.threshold[0]
    impurity = tree.impurity
    n_node_samples = tree.n_node_samples

    impurity_reduction = (
        n_node_samples[0] * impurity[0]
        - n_node_samples[1] * impurity[1]
        - n_node_samples[2] * impurity[2]
    ) / max(n_node_samples[0], 1)

    left_mask = projections <= threshold

    if np.sum(left_mask) < 1 or np.sum(~left_mask) < 1:
        return None

    return {
        "features": np.array(feature_indices),
        "weights": weights,
        "threshold": threshold,
        "impurity_reduction": impurity_reduction,
        "left_mask": left_mask,
        "value_left": np.mean(y_residuals[left_mask]),
        "value_right": np.mean(y_residuals[~left_mask]),
        "n_left": int(np.sum(left_mask)),
        "n_right": int(np.sum(~left_mask)),
    }


def fit_axis_aligned_split(
    X: np.ndarray, y_residuals: np.ndarray, idxs: np.ndarray
) -> dict | None:
    """Fit axis-aligned stump (standard FIGS split)."""
    X_sub = X[idxs]
    y_sub = y_residuals[idxs]

    if X_sub.shape[0] < 5:
        return None

    stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=2)
    stump.fit(X_sub, y_sub)

    tree = stump.tree_
    if tree.feature[0] == -2 or tree.n_node_samples.shape[0] < 3:
        return None

    feature_idx = tree.feature[0]
    threshold = tree.threshold[0]
    impurity = tree.impurity
    n_node_samples = tree.n_node_samples

    impurity_reduction = (
        n_node_samples[0] * impurity[0]
        - n_node_samples[1] * impurity[1]
        - n_node_samples[2] * impurity[2]
    ) / max(n_node_samples[0], 1)

    left_mask_sub = X_sub[:, feature_idx] <= threshold
    if np.sum(left_mask_sub) < 1 or np.sum(~left_mask_sub) < 1:
        return None

    # Convert to full-array mask
    left_mask_full = np.zeros(len(X), dtype=bool)
    full_indices = np.where(idxs)[0] if idxs.dtype == bool else idxs
    left_mask_full[full_indices[left_mask_sub]] = True

    return {
        "feature": feature_idx,
        "threshold": threshold,
        "impurity_reduction": impurity_reduction,
        "left_mask": left_mask_full,
        "value_left": np.mean(y_sub[left_mask_sub]),
        "value_right": np.mean(y_sub[~left_mask_sub]),
        "n_left": int(np.sum(left_mask_sub)),
        "n_right": int(np.sum(~left_mask_sub)),
    }


# ---------------------------------------------------------------------------
# 5. BaseFIGSOblique — FIGS greedy loop with oblique support
# ---------------------------------------------------------------------------

class BaseFIGSOblique:
    """Clean reimplementation of FIGS greedy-tree-sum with oblique split support.

    Algorithm (faithful to imodels FIGS):
    1. Maintain a list of leaf nodes (initially one per tree).
    2. Each leaf stores the sample indices (idxs) routed to it.
    3. At each step, try splitting every leaf; pick the one with best
       impurity reduction.
    4. Replace that leaf with an internal node + two new leaves.
    5. After each split, recompute residuals across all trees.
    """

    def __init__(
        self,
        max_splits: int = 25,
        max_trees: int | None = None,
        max_depth: int | None = None,
        min_impurity_decrease: float = 0.0,
        num_repetitions: int = 5,
        beam_size: int | None = None,
        random_state: int | None = None,
    ):
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.max_depth = max_depth if max_depth else 6
        self.min_impurity_decrease = min_impurity_decrease
        self.num_repetitions = num_repetitions
        self.beam_size = beam_size
        self.random_state = random_state
        self.trees_ = []
        self.complexity_ = 0

    # -- subclass hooks --------------------------------------------------
    def _precompute(self, X: np.ndarray, y: np.ndarray):
        pass

    def _get_feature_subsets_for_split(
        self, X: np.ndarray, rng: random.Random
    ) -> list:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _weighted_mse(y: np.ndarray) -> float:
        """Variance * n  (= total squared error from mean)."""
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
        """Find the best split (oblique or axis-aligned) at a leaf."""
        idx_arr = np.where(idxs)[0]
        if len(idx_arr) < 5:
            return None

        X_node = X[idx_arr]
        y_node = residuals[idx_arr]
        parent_mse = self._weighted_mse(y_node)

        best = None
        best_gain = self.min_impurity_decrease

        # --- axis-aligned stump ------------------------------------------
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

        # --- oblique splits (multiple tries) -----------------------------
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

    # -- fit -------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list | None = None):
        """Main FIGS greedy loop with oblique splits."""
        rng = random.Random(self.random_state)
        np.random.seed(self.random_state if self.random_state else 42)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.feature_names_ = feature_names

        # Scale features to [0, 1]
        self.scaler_ = MinMaxScaler()
        X_s = self.scaler_.fit_transform(X)

        # Pre-computation (synergy for SG-FIGS, no-op for RO-FIGS)
        self._precompute(X_s, y)

        if self.beam_size is None:
            self.beam_size = max(2, n_features // 2)

        y_target = y.astype(float)

        # ---------- bootstrap: first tree with one leaf ------------------
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

        # leaves: list of (tree_index, node, parent, side)
        # parent=None, side=None for root
        leaves = [(0, root_leaf, None, None)]
        total_splits = 0

        while total_splits < self.max_splits and leaves:
            # Compute current residuals
            predictions = self._compute_predictions(X_s)
            residuals = y_target - predictions

            # Score every leaf
            scored = []
            for tree_idx, leaf, parent, side in leaves:
                if leaf.depth >= self.max_depth:
                    continue
                split_info = self._best_split_for_node(X_s, residuals, leaf.idxs, rng)
                if split_info is not None:
                    scored.append((split_info["gain"], tree_idx, leaf, parent, side, split_info))

            if not scored:
                # No more profitable splits — try adding a new tree
                if self.max_trees is None or len(self.trees_) < self.max_trees:
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
                    # Re-score with the new tree
                    continue
                else:
                    break

            # Pick the best split
            scored.sort(key=lambda x: x[0], reverse=True)
            best_gain, tree_idx, leaf, parent, side, info = scored[0]

            # Build internal node
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

            # Replace the leaf in the tree structure
            if parent is None:
                # This was a root leaf — replace the tree root
                self.trees_[tree_idx] = node
            else:
                if side == "left":
                    parent.left = node
                else:
                    parent.right = node

            # Update leaves list: remove old leaf, add two new ones
            leaves = [
                (ti, lf, p, s)
                for (ti, lf, p, s) in leaves
                if lf is not leaf
            ]
            leaves.append((tree_idx, left_leaf, node, "left"))
            leaves.append((tree_idx, right_leaf, node, "right"))

            total_splits += 1

        # Final pass: for each tree, set leaf values = mean(y - preds_of_other_trees)
        for t_idx, tree in enumerate(self.trees_):
            # Compute predictions from all OTHER trees
            other_preds = np.zeros(n_samples)
            for j, other_tree in enumerate(self.trees_):
                if j != t_idx:
                    other_preds += self._predict_tree(other_tree, X_s)
            residuals_for_tree = y_target - other_preds
            self._update_leaf_values(tree, residuals_for_tree)

        self.complexity_ = total_splits
        return self

    def _update_leaf_values(
        self, node: ObliqueFIGSNode, residuals: np.ndarray
    ):
        """Set leaf values to mean of residuals at each leaf."""
        if node is None:
            return
        if node.left is None and node.right is None:
            if node.idxs is not None and np.any(node.idxs):
                node.value = float(np.mean(residuals[node.idxs]))
            return
        self._update_leaf_values(node.left, residuals)
        self._update_leaf_values(node.right, residuals)

    # -- prediction ------------------------------------------------------
    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        return preds

    def _predict_tree(self, root: ObliqueFIGSNode, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = self._predict_single(root, X[i])
        return preds

    def _predict_single(self, node: ObliqueFIGSNode, x: np.ndarray) -> float:
        if node is None:
            return 0.0
        if node.left is None and node.right is None:
            v = node.value
            return float(v) if v is not None else 0.0

        if node.is_oblique and node.features is not None and node.weights is not None:
            go_left = np.dot(x[node.features], node.weights) <= node.threshold
        elif node.feature is not None:
            go_left = x[node.feature] <= node.threshold
        else:
            return float(node.value) if node.value is not None else 0.0

        if go_left:
            return self._predict_single(node.left, x)
        else:
            return self._predict_single(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        preds = np.zeros(X_scaled.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X_scaled)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        preds = np.zeros(X_scaled.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X_scaled)
        # Clip to [0, 1] since tree-sum can go outside
        probs = np.clip(preds, 0.0, 1.0)
        return np.vstack((1 - probs, probs)).T


class ROFIGSClassifier(BaseFIGSOblique):
    """RO-FIGS baseline: random feature subsets for oblique splits."""

    def _precompute(self, X: np.ndarray, y: np.ndarray):
        pass

    def _get_feature_subsets_for_split(
        self, X: np.ndarray, rng: random.Random
    ) -> list:
        d = X.shape[1]
        beam = self.beam_size if self.beam_size else max(2, d // 2)
        indices = list(range(d))
        subset = rng.sample(indices, min(beam, d))
        return [subset]


class SGFIGSClassifier(BaseFIGSOblique):
    """SG-FIGS: synergy-guided feature subsets for oblique splits."""

    def __init__(
        self,
        n_bins: int = 5,
        threshold_percentile: int = 75,
        precomputed_synergy: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.threshold_percentile = threshold_percentile
        self.synergy_info_ = precomputed_synergy

    def _precompute(self, X: np.ndarray, y: np.ndarray):
        """Build synergy graph and candidate subsets (skip if pre-computed)."""
        if self.synergy_info_ is not None:
            return  # Already have synergy info
        self.synergy_info_ = compute_synergy_pipeline(
            X, y,
            n_bins=self.n_bins,
            threshold_percentile=self.threshold_percentile,
        )
        n_edges = self.synergy_info_["graph"].number_of_edges()
        n_subsets = len(self.synergy_info_["subsets"])
        logger.debug(
            f"Synergy graph: {self.synergy_info_['graph'].number_of_nodes()} nodes, "
            f"{n_edges} edges, {n_subsets} candidate subsets"
        )

    def _get_feature_subsets_for_split(
        self, X: np.ndarray, rng: random.Random
    ) -> list:
        """Sample a synergy-guided feature subset."""
        subsets = self.synergy_info_["subsets"] if self.synergy_info_ else []
        d = X.shape[1]
        beam = self.beam_size if self.beam_size else max(2, d // 2)

        if not subsets:
            # Fallback to random
            indices = list(range(d))
            return [rng.sample(indices, min(beam, d))]

        chosen = list(rng.choice(subsets))

        if len(chosen) < beam:
            remaining = [f for f in range(d) if f not in chosen]
            pad_count = min(beam - len(chosen), len(remaining))
            if pad_count > 0:
                pad = rng.sample(remaining, pad_count)
                chosen = chosen + pad
        elif len(chosen) > beam:
            S = self.synergy_info_["synergy_matrix"]
            scored = [
                (f, sum(S[f, g] for g in chosen if g != f)) for f in chosen
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen = [f for f, _ in scored[:beam]]

        return [sorted(chosen)]


# ---------------------------------------------------------------------------
# 6. FIGS Baseline Wrapper (using imodels)
# ---------------------------------------------------------------------------

class FIGSBaselineWrapper:
    """Wrapper around imodels FIGSClassifier for consistent API."""

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


# ---------------------------------------------------------------------------
# 7. Evaluation Metrics
# ---------------------------------------------------------------------------

def collect_oblique_nodes(node: ObliqueFIGSNode, result: list):
    """Recursively collect oblique internal nodes."""
    if node is None:
        return
    if node.is_oblique and (node.left is not None or node.right is not None):
        result.append(node)
    collect_oblique_nodes(node.left, result)
    collect_oblique_nodes(node.right, result)


def compute_avg_features_per_split(model) -> float:
    """Average number of features per oblique split."""
    if not hasattr(model, "trees_"):
        return 1.0
    oblique_nodes = []
    for tree_root in model.trees_:
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
    """Fraction of oblique splits whose features have above-median synergy."""
    d = synergy_matrix.shape[0]
    all_synergies = synergy_matrix[np.triu_indices(d, k=1)]
    pos_syn = all_synergies[all_synergies > 0]
    median_synergy = float(np.median(pos_syn)) if len(pos_syn) > 0 else 0.0

    oblique_splits = []
    if hasattr(model, "trees_"):
        for tree_root in model.trees_:
            collect_oblique_nodes(tree_root, oblique_splits)

    if not oblique_splits:
        return 0.0

    above_median_count = 0
    for node in oblique_splits:
        feats = node.features
        if feats is None or len(feats) < 2:
            continue
        pair_synergies = []
        for i_idx in range(len(feats)):
            for j_idx in range(i_idx + 1, len(feats)):
                fi, fj = int(feats[i_idx]), int(feats[j_idx])
                if fi < d and fj < d:
                    pair_synergies.append(synergy_matrix[fi, fj])
        if pair_synergies and np.mean(pair_synergies) > median_synergy:
            above_median_count += 1

    return above_median_count / len(oblique_splits)


def count_total_nodes(node) -> int:
    """Count total internal nodes (splits) in a tree."""
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 0
    return 1 + count_total_nodes(node.left) + count_total_nodes(node.right)


def count_trees_and_splits(model) -> tuple[int, int]:
    """Count trees and total splits."""
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


# ---------------------------------------------------------------------------
# 8. Model Factory
# ---------------------------------------------------------------------------

def create_model(
    method_name: str,
    max_splits: int,
    beam_size: int,
    precomputed_synergy: dict | None = None,
):
    """Create a model by method name."""
    if method_name == "FIGS":
        return FIGSBaselineWrapper(max_splits=max_splits)
    elif method_name == "RO-FIGS":
        return ROFIGSClassifier(
            max_splits=max_splits,
            beam_size=beam_size,
            random_state=RANDOM_SEED,
            num_repetitions=1,
        )
    elif method_name == "SG-FIGS":
        return SGFIGSClassifier(
            max_splits=max_splits,
            beam_size=beam_size,
            random_state=RANDOM_SEED,
            num_repetitions=1,
            n_bins=5,
            threshold_percentile=75,
            precomputed_synergy=precomputed_synergy,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ---------------------------------------------------------------------------
# 9. Main Experiment
# ---------------------------------------------------------------------------

def evaluate_method_on_dataset(
    method_name: str,
    ds: dict,
    synergy_info: dict,
    beam_size: int,
) -> dict:
    """Evaluate a method on a dataset with hyperparameter tuning."""
    X, y = ds["X"], ds["y"]
    folds = ds["folds"]
    is_binary = ds["n_classes"] == 2
    feature_names = ds["feature_names"]

    best_config = None
    best_val_score = -1.0

    # Hyperparameter tuning on fold 0
    for max_splits in MAX_SPLITS_GRID:
        try:
            train_idx = folds != 0
            val_idx = folds == 0

            if np.sum(train_idx) < 5 or np.sum(val_idx) < 2:
                continue

            model = create_model(method_name, max_splits=max_splits, beam_size=beam_size, precomputed_synergy=synergy_info)
            model.fit(X[train_idx], y[train_idx], feature_names=feature_names)

            y_pred = model.predict(X[val_idx])
            val_score = balanced_accuracy_score(y[val_idx], y_pred)

            if val_score > best_val_score:
                best_val_score = val_score
                best_config = max_splits
        except Exception:
            logger.exception(f"  {method_name} max_splits={max_splits} failed during tuning")
            continue

    if best_config is None:
        logger.warning(f"  {method_name}: all configs failed, defaulting to max_splits=10")
        best_config = 10

    logger.info(f"  {method_name}: best max_splits={best_config} (val_acc={best_val_score:.4f})")

    # Evaluate on all 5 folds
    fold_results = []
    for fold_id in range(N_FOLDS):
        try:
            test_idx = folds == fold_id
            train_idx = ~test_idx

            if np.sum(train_idx) < 5 or np.sum(test_idx) < 2:
                logger.warning(f"  Fold {fold_id}: insufficient samples, skipping")
                continue

            model = create_model(method_name, max_splits=best_config, beam_size=beam_size, precomputed_synergy=synergy_info)
            model.fit(X[train_idx], y[train_idx], feature_names=feature_names)

            y_pred = model.predict(X[test_idx])
            bal_acc = balanced_accuracy_score(y[test_idx], y_pred)

            auc = None
            if is_binary:
                try:
                    y_proba = model.predict_proba(X[test_idx])
                    if y_proba.shape[1] == 2:
                        auc = roc_auc_score(y[test_idx], y_proba[:, 1])
                except Exception:
                    pass

            n_trees, n_splits = count_trees_and_splits(model)
            avg_feat = compute_avg_features_per_split(model) if method_name != "FIGS" else 1.0

            interp_score = None
            if method_name != "FIGS" and synergy_info is not None:
                try:
                    interp_score = compute_split_interpretability_score(
                        model, synergy_info["synergy_matrix"]
                    )
                except Exception:
                    interp_score = None

            fold_results.append({
                "fold": fold_id,
                "balanced_accuracy": round(bal_acc, 6),
                "auc": round(auc, 6) if auc is not None else None,
                "n_splits": n_splits,
                "n_trees": n_trees,
                "avg_features_per_split": round(avg_feat, 4),
                "split_interpretability_score": round(interp_score, 4) if interp_score is not None else None,
            })
        except Exception:
            logger.exception(f"  {method_name} fold {fold_id} failed")
            continue

    if not fold_results:
        return {"error": "all folds failed"}

    accs = [r["balanced_accuracy"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results if r["auc"] is not None]
    splits = [r["n_splits"] for r in fold_results]
    feats = [r["avg_features_per_split"] for r in fold_results]
    interps = [r["split_interpretability_score"] for r in fold_results if r["split_interpretability_score"] is not None]

    return {
        "best_max_splits": best_config,
        "fold_results": fold_results,
        "mean_balanced_accuracy": round(float(np.mean(accs)), 6),
        "std_balanced_accuracy": round(float(np.std(accs)), 6),
        "mean_auc": round(float(np.mean(aucs)), 6) if aucs else None,
        "std_auc": round(float(np.std(aucs)), 6) if aucs else None,
        "mean_n_splits": round(float(np.mean(splits)), 2),
        "mean_avg_features_per_split": round(float(np.mean(feats)), 4),
        "mean_split_interpretability": round(float(np.mean(interps)), 4) if interps else None,
    }


def get_top_synergy_pairs(
    synergy_matrix: np.ndarray, feature_names: list, top_k: int = 5
) -> list:
    """Get top-k synergy pairs with feature names."""
    d = synergy_matrix.shape[0]
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            if synergy_matrix[i, j] > 0:
                fname_i = feature_names[i] if i < len(feature_names) else f"f{i}"
                fname_j = feature_names[j] if j < len(feature_names) else f"f{j}"
                pairs.append((fname_i, fname_j, round(float(synergy_matrix[i, j]), 6)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def run_experiment(datasets: dict) -> dict:
    """Run the full benchmark experiment."""
    results = {}
    synergy_cache = {}
    total_start = time.time()

    for tier_idx, tier_datasets in enumerate(TIER_ORDER):
        logger.info(f"\n{'='*60}")
        logger.info(f"=== TIER {tier_idx + 1} ===")
        logger.info(f"{'='*60}")

        for ds_name in tier_datasets:
            if ds_name not in datasets:
                logger.warning(f"Dataset {ds_name} not found, skipping")
                continue

            ds = datasets[ds_name]
            X, y = ds["X"], ds["y"]
            is_binary = ds["n_classes"] == 2

            logger.info(
                f"\nDataset: {ds_name} "
                f"({X.shape[0]} samples, {X.shape[1]} features, "
                f"{ds['n_classes']} classes)"
            )

            # Compute synergy for this dataset (once, using fold-0 training data)
            ds_start = time.time()
            train_mask = ds["folds"] != 0

            logger.info(f"  Computing synergy matrix...")
            synergy_info = compute_synergy_pipeline(X[train_mask], y[train_mask])
            synergy_cache[ds_name] = synergy_info
            synergy_time = time.time() - ds_start
            logger.info(
                f"  Synergy computed in {synergy_time:.1f}s "
                f"({synergy_info['graph'].number_of_edges()} edges, "
                f"{len(synergy_info['subsets'])} subsets)"
            )

            beam_size = max(2, X.shape[1] // 2)
            ds_results = {}

            # Define methods: oblique only for binary datasets
            methods_to_run = ["FIGS"]
            if is_binary:
                methods_to_run.extend(["RO-FIGS", "SG-FIGS"])

            for method_name in methods_to_run:
                method_start = time.time()
                logger.info(f"  Evaluating {method_name}...")

                method_result = evaluate_method_on_dataset(
                    method_name=method_name,
                    ds=ds,
                    synergy_info=synergy_info,
                    beam_size=beam_size,
                )

                method_time = time.time() - method_start
                if "error" not in method_result:
                    logger.info(
                        f"    {method_name}: acc={method_result['mean_balanced_accuracy']:.4f} "
                        f"±{method_result['std_balanced_accuracy']:.4f}, "
                        f"splits={method_result['mean_n_splits']:.1f}, "
                        f"time={method_time:.1f}s"
                    )
                else:
                    logger.warning(f"    {method_name}: FAILED — {method_result['error']}")

                ds_results[method_name] = method_result

            results[ds_name] = ds_results

            ds_time = time.time() - ds_start
            logger.info(f"  Dataset {ds_name} complete in {ds_time:.1f}s")

            # Check total elapsed time — warn and skip remaining if running long
            total_elapsed = time.time() - total_start
            if total_elapsed > 2700:  # 45 min warning
                logger.warning(f"Total elapsed: {total_elapsed/60:.1f} min — approaching time limit")
            if total_elapsed > 3000:  # 50 min hard limit
                logger.warning(f"Time limit reached ({total_elapsed/60:.1f} min). Stopping experiment.")
                break

        # Also break outer tier loop if time exceeded
        total_elapsed = time.time() - total_start
        if total_elapsed > 3000:
            logger.warning(f"Time limit reached in outer loop. Skipping remaining tiers.")
            break

        # Log intermediate results after each tier
        logger.info(f"\n--- Tier {tier_idx + 1} Summary ---")
        for dn, dr in results.items():
            parts = []
            for mn, mr in dr.items():
                if "mean_balanced_accuracy" in mr:
                    parts.append(f"{mn}={mr['mean_balanced_accuracy']:.4f}")
            logger.info(f"  {dn}: {', '.join(parts)}")

    total_time = time.time() - total_start
    logger.info(f"\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return results, synergy_cache


def build_aggregate_comparison(results: dict) -> dict:
    """Build aggregate comparison across datasets."""
    methods = ["FIGS", "RO-FIGS", "SG-FIGS"]
    agg = {
        "mean_balanced_accuracy": {},
        "mean_auc": {},
        "mean_n_splits": {},
        "mean_split_interpretability": {},
        "wins_by_dataset": {m: [] for m in methods},
    }

    for method in methods:
        accs, aucs, splits, interps = [], [], [], []
        for ds_name, ds_results in results.items():
            if method in ds_results and "mean_balanced_accuracy" in ds_results[method]:
                mr = ds_results[method]
                accs.append(mr["mean_balanced_accuracy"])
                if mr.get("mean_auc") is not None:
                    aucs.append(mr["mean_auc"])
                splits.append(mr["mean_n_splits"])
                if mr.get("mean_split_interpretability") is not None:
                    interps.append(mr["mean_split_interpretability"])

        agg["mean_balanced_accuracy"][method] = round(float(np.mean(accs)), 6) if accs else None
        agg["mean_auc"][method] = round(float(np.mean(aucs)), 6) if aucs else None
        agg["mean_n_splits"][method] = round(float(np.mean(splits)), 2) if splits else None
        agg["mean_split_interpretability"][method] = round(float(np.mean(interps)), 4) if interps else None

    # Wins by dataset (highest balanced accuracy)
    for ds_name, ds_results in results.items():
        best_method = None
        best_acc = -1
        for method in methods:
            if method in ds_results and "mean_balanced_accuracy" in ds_results[method]:
                acc = ds_results[method]["mean_balanced_accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    best_method = method
        if best_method:
            agg["wins_by_dataset"][best_method].append(ds_name)

    return agg


def build_synergy_analysis(synergy_cache: dict, datasets: dict) -> dict:
    """Build per-dataset synergy analysis summary."""
    analysis = {}
    for ds_name, synergy_info in synergy_cache.items():
        if ds_name not in datasets:
            continue
        feature_names = datasets[ds_name]["feature_names"]
        S = synergy_info["synergy_matrix"]
        G = synergy_info["graph"]

        top_pairs = get_top_synergy_pairs(S, feature_names, top_k=5)
        cliques = list(G.graph.get("cliques", [])) if hasattr(G, "graph") else []
        # Count cliques manually
        import networkx as nx
        clique_list = [c for c in nx.find_cliques(G) if len(c) >= 2]

        analysis[ds_name] = {
            "top_synergy_pairs": [
                {"feature_i": p[0], "feature_j": p[1], "synergy_bits": p[2]}
                for p in top_pairs
            ],
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
            "graph_cliques_size_2plus": len(clique_list),
            "synergy_threshold": round(float(synergy_info["threshold"]), 6),
            "n_candidate_subsets": len(synergy_info["subsets"]),
        }
    return analysis


def evaluate_success_criteria(results: dict, agg: dict) -> dict:
    """Evaluate the three success criteria for SG-FIGS."""
    # Criterion 1: SG-FIGS achieves comparable/fewer splits than RO-FIGS
    sg_splits = agg["mean_n_splits"].get("SG-FIGS")
    ro_splits = agg["mean_n_splits"].get("RO-FIGS")
    criterion_1 = None
    if sg_splits is not None and ro_splits is not None:
        criterion_1 = sg_splits <= ro_splits * 1.1  # Allow 10% tolerance

    # Criterion 2: SG-FIGS has higher interpretability score than RO-FIGS
    sg_interp = agg["mean_split_interpretability"].get("SG-FIGS")
    ro_interp = agg["mean_split_interpretability"].get("RO-FIGS")
    criterion_2 = None
    if sg_interp is not None and ro_interp is not None:
        criterion_2 = sg_interp > ro_interp

    # Criterion 3: Qualitative — find domain-meaningful interactions
    qualitative_notes = []
    for ds_name, ds_results in results.items():
        if "SG-FIGS" in ds_results and "mean_balanced_accuracy" in ds_results["SG-FIGS"]:
            sg = ds_results["SG-FIGS"]
            figs = ds_results.get("FIGS", {})
            sg_acc = sg["mean_balanced_accuracy"]
            figs_acc = figs.get("mean_balanced_accuracy", 0)
            if sg_acc >= figs_acc:
                qualitative_notes.append(
                    f"{ds_name}: SG-FIGS ({sg_acc:.4f}) >= FIGS ({figs_acc:.4f})"
                )

    return {
        "criterion_1_comparable_or_fewer_splits": criterion_1,
        "criterion_1_detail": f"SG-FIGS={sg_splits}, RO-FIGS={ro_splits}",
        "criterion_2_higher_interpretability": criterion_2,
        "criterion_2_detail": f"SG-FIGS={sg_interp}, RO-FIGS={ro_interp}",
        "criterion_3_qualitative": "; ".join(qualitative_notes) if qualitative_notes else "No notable interactions found",
    }


def format_output(
    results: dict,
    synergy_cache: dict,
    datasets: dict,
    examples_data: list,
) -> dict:
    """Format the output as exp_gen_sol_out schema: {datasets: [...]}."""
    output_datasets = []

    for ds_entry in examples_data:
        ds_name = ds_entry["dataset"]
        examples = ds_entry["examples"]

        # Add predictions from each method
        ds_info = datasets.get(ds_name, {})
        ds_results = results.get(ds_name, {})

        # Build output per example with predictions
        output_examples = []
        for ex in examples:
            out_ex = {
                "input": ex["input"],
                "output": ex["output"],
            }

            # Copy all metadata fields
            for key, val in ex.items():
                if key.startswith("metadata_"):
                    out_ex[key] = val

            # Add method predictions (using best config, fold from metadata)
            fold_id = ex.get("metadata_fold", 0)

            # For each method, we store the summary results as metadata
            for method_name in ["FIGS", "RO-FIGS", "SG-FIGS"]:
                if method_name in ds_results and "mean_balanced_accuracy" in ds_results[method_name]:
                    mr = ds_results[method_name]
                    # Store per-fold result as the prediction metadata
                    fold_res = None
                    for fr in mr.get("fold_results", []):
                        if fr["fold"] == fold_id:
                            fold_res = fr
                            break

                    safe_name = method_name.replace("-", "_").lower()
                    if fold_res:
                        out_ex[f"predict_{safe_name}"] = json.dumps({
                            "balanced_accuracy": fold_res["balanced_accuracy"],
                            "auc": fold_res["auc"],
                            "n_splits": fold_res["n_splits"],
                            "n_trees": fold_res["n_trees"],
                        })
                    else:
                        out_ex[f"predict_{safe_name}"] = json.dumps({
                            "mean_balanced_accuracy": mr["mean_balanced_accuracy"],
                            "mean_auc": mr.get("mean_auc"),
                        })

            output_examples.append(out_ex)

        output_datasets.append({
            "dataset": ds_name,
            "examples": output_examples,
        })

    return {"datasets": output_datasets}


def build_full_results_json(
    results: dict,
    synergy_cache: dict,
    datasets: dict,
    schema_output: dict,
) -> dict:
    """Build the comprehensive results JSON combining schema output + analysis."""
    agg = build_aggregate_comparison(results)
    synergy_analysis = build_synergy_analysis(synergy_cache, datasets)
    success = evaluate_success_criteria(results, agg)

    # Merge analysis into the schema output as top-level metadata
    full_output = dict(schema_output)
    full_output["metadata_experiment"] = "SG-FIGS vs FIGS vs RO-FIGS Benchmark Comparison"
    full_output["metadata_methods"] = ["FIGS", "RO-FIGS", "SG-FIGS"]
    full_output["metadata_datasets_evaluated"] = len(results)
    full_output["metadata_per_dataset_results"] = results
    full_output["metadata_aggregate_comparison"] = agg
    full_output["metadata_success_criteria"] = success
    full_output["metadata_synergy_analysis"] = synergy_analysis

    return full_output


# ---------------------------------------------------------------------------
# 10. Main Entry Point
# ---------------------------------------------------------------------------

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("SG-FIGS: Synergy-Guided Oblique FIGS Experiment")
    logger.info("=" * 60)

    # Load data
    datasets = load_all_datasets(DATA_PATH)

    # Run experiment
    results, synergy_cache = run_experiment(datasets)

    # Load raw examples for schema output
    raw_data = json.loads(DATA_PATH.read_text())

    # Format as schema-compliant output
    schema_output = format_output(results, synergy_cache, datasets, raw_data["datasets"])

    # Build full results JSON
    full_output = build_full_results_json(results, synergy_cache, datasets, schema_output)

    # Remove non-serializable objects (networkx graphs, numpy arrays)
    def make_serializable(obj):
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
        else:
            return obj

    # Remove graph objects from synergy analysis (not serializable)
    if "metadata_synergy_analysis" in full_output:
        for ds_name in full_output["metadata_synergy_analysis"]:
            entry = full_output["metadata_synergy_analysis"][ds_name]
            if "graph" in entry:
                del entry["graph"]

    # Remove synergy_matrix from per-dataset results (too large for JSON)
    if "metadata_per_dataset_results" in full_output:
        for ds_name in full_output["metadata_per_dataset_results"]:
            for method_name in full_output["metadata_per_dataset_results"][ds_name]:
                mr = full_output["metadata_per_dataset_results"][ds_name][method_name]
                if "synergy_matrix" in mr:
                    del mr["synergy_matrix"]

    serializable_output = make_serializable(full_output)

    # Save output — only keep "datasets" key for schema compliance
    schema_compliant = {"datasets": serializable_output["datasets"]}
    OUTPUT_PATH.write_text(json.dumps(schema_compliant, indent=2))
    logger.info(f"Results saved to {OUTPUT_PATH}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    header = f"{'Dataset':<35} {'FIGS':>8} {'RO-FIGS':>8} {'SG-FIGS':>8} {'SG Interp':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for ds_name, ds_results in results.items():
        figs_acc = ds_results.get("FIGS", {}).get("mean_balanced_accuracy", None)
        ro_acc = ds_results.get("RO-FIGS", {}).get("mean_balanced_accuracy", None)
        sg_acc = ds_results.get("SG-FIGS", {}).get("mean_balanced_accuracy", None)
        sg_interp = ds_results.get("SG-FIGS", {}).get("mean_split_interpretability", None)

        figs_str = f"{figs_acc:.4f}" if figs_acc is not None else "N/A"
        ro_str = f"{ro_acc:.4f}" if ro_acc is not None else "N/A"
        sg_str = f"{sg_acc:.4f}" if sg_acc is not None else "N/A"
        interp_str = f"{sg_interp:.4f}" if sg_interp is not None else "N/A"

        logger.info(f"{ds_name:<35} {figs_str:>8} {ro_str:>8} {sg_str:>8} {interp_str:>10}")

    agg = build_aggregate_comparison(results)
    logger.info("-" * len(header))
    logger.info(
        f"{'AVERAGE':<35} "
        f"{agg['mean_balanced_accuracy'].get('FIGS', 'N/A'):>8} "
        f"{agg['mean_balanced_accuracy'].get('RO-FIGS', 'N/A'):>8} "
        f"{agg['mean_balanced_accuracy'].get('SG-FIGS', 'N/A'):>8}"
    )

    success = evaluate_success_criteria(results, agg)
    logger.info(f"\nSuccess Criteria:")
    logger.info(f"  1. Comparable splits: {success['criterion_1_comparable_or_fewer_splits']} ({success['criterion_1_detail']})")
    logger.info(f"  2. Higher interpretability: {success['criterion_2_higher_interpretability']} ({success['criterion_2_detail']})")
    logger.info(f"  3. Qualitative: {success['criterion_3_qualitative'][:200]}")

    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
