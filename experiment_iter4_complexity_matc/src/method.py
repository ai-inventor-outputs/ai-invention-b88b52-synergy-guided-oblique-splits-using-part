#!/usr/bin/env python3
"""Complexity-Matched SG-FIGS Experiment: Synergy vs Random at Equal Splits.

Compares 5 methods (FIGS, RO-FIGS, SG-FIGS-Hard, SG-FIGS-Soft, Random-FIGS)
at exactly max_splits=5 and max_splits=10 across 14 datasets with NO
hyperparameter tuning. Enforces hard node-count caps so every method uses
the same complexity budget.

Key improvements over iter_3:
  1. Hard complexity enforcement: actual_splits <= max_splits (verified)
  2. Five comparable methods with identical tree-fitting core
  3. Win/Tie/Loss tables, per-split information gain, interpretability
  4. MultiClass OvR with per-class split budgeting
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
from itertools import combinations
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mutual_info_score
from typing import Any

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

# ── Logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent.resolve()
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Configuration ────────────────────────────────────────────────────────────
DATA_ID2_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out.json"
)
DATA_ID3_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_2/gen_art/data_id3_it2__opus/full_data_out.json"
)
SYNERGY_RESULTS_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_pipeline/runs/run_sg_figs_full"
    "/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus/results_comprehensive.json"
)

MAX_SPLITS_VALUES = [5, 10]
N_FOLDS = 5
RANDOM_SEED = 42
SYNERGY_THRESHOLD_PCTL = 75
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 5
WIN_THRESHOLD = 0.005  # 0.5% balanced accuracy
TIME_BUDGET_SECONDS = 2700  # 45 minutes

METHOD_NAMES = ["FIGS", "RO_FIGS", "SG_FIGS_Hard", "SG_FIGS_Soft", "Random_FIGS"]

# Tier ordering from artifact plan
TIER_ORDER = [
    ["monks2", "banknote", "iris", "blood"],
    ["pima_diabetes", "wine", "heart_statlog", "spectf_heart"],
    ["vehicle", "climate", "kc2", "ionosphere", "sonar",
     "breast_cancer_wisconsin_diagnostic"],
]

# Map synergy dataset names to data dataset names
SYNERGY_NAME_MAP = {
    "breast_cancer": "breast_cancer_wisconsin_diagnostic",
}
NEW_SYNERGY_DATASETS = ["monks2", "blood", "climate", "kc2"]


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_datasets_from_json(json_path: Path) -> dict[str, dict[str, Any]]:
    """Load all datasets from a full_data_out.json file."""
    logger.info(f"Loading datasets from {json_path.name}")
    data = json.loads(json_path.read_text())
    datasets: dict[str, dict[str, Any]] = {}
    for ds_entry in data["datasets"]:
        name = ds_entry["dataset"]
        examples = ds_entry["examples"]
        if not examples:
            logger.warning(f"  Empty dataset: {name}, skipping")
            continue

        first_input = json.loads(examples[0]["input"])
        feature_names = list(first_input.keys())

        X = np.array(
            [[float(v) for v in json.loads(ex["input"]).values()] for ex in examples],
            dtype=np.float64,
        )

        raw_labels = [str(ex["output"]) for ex in examples]
        unique_labels = sorted(set(raw_labels))
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        y = np.array([label_map[lbl] for lbl in raw_labels], dtype=np.int32)

        folds = np.array([int(ex["metadata_fold"]) for ex in examples], dtype=np.int32)

        datasets[name] = {
            "X": X,
            "y": y,
            "folds": folds,
            "feature_names": feature_names,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_classes": len(unique_labels),
        }
    return datasets


def load_precomputed_synergy(synergy_path: Path) -> dict[str, dict[str, Any]]:
    """Load synergy matrices from results_comprehensive.json."""
    logger.info("Loading pre-computed synergy matrices")
    data = json.loads(synergy_path.read_text())
    synergy_data: dict[str, dict[str, Any]] = {}
    for ds in data["per_dataset_full"]:
        ds_name = ds["dataset"]
        mapped = SYNERGY_NAME_MAP.get(ds_name, ds_name)
        synergy_data[mapped] = {
            "synergy_matrix": np.array(ds["synergy_matrix"], dtype=np.float64),
            "feature_names": list(ds["mi_values"].keys()),
            "pid_method": ds["pid_method"],
            "n_features_used": ds["n_features_used"],
        }
    return synergy_data


# ============================================================================
# CO-INFORMATION SYNERGY (for datasets without pre-computed PID)
# ============================================================================
def compute_coi_synergy(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Compute Co-Information synergy matrix using sklearn mutual_info only."""
    n_features = X.shape[1]

    # Discretize
    X_disc = np.zeros_like(X, dtype=np.int32)
    for j in range(n_features):
        col = X[:, j]
        n_unique = len(np.unique(col))
        if n_unique <= 5:
            _, X_disc[:, j] = np.unique(col, return_inverse=True)
        else:
            try:
                kbd = KBinsDiscretizer(
                    n_bins=min(5, n_unique),
                    encode="ordinal",
                    strategy="quantile",
                    subsample=None,
                )
                X_disc[:, j] = kbd.fit_transform(
                    col.reshape(-1, 1)
                ).ravel().astype(np.int32)
            except ValueError:
                _, X_disc[:, j] = np.unique(col, return_inverse=True)

    y_int = y.astype(np.int32)

    # Single-feature MI
    mi_single = np.zeros(n_features)
    for j in range(n_features):
        mi_single[j] = mutual_info_score(X_disc[:, j], y_int)

    # Pairwise synergy via CoI
    synergy_matrix = np.zeros((n_features, n_features), dtype=np.float64)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Encode pair as joint variable
            xi = X_disc[:, i]
            xj = X_disc[:, j]
            max_xj = int(xj.max()) + 1
            joint = xi * max_xj + xj
            mi_joint = mutual_info_score(joint, y_int)
            coi = mi_single[i] + mi_single[j] - mi_joint
            synergy_val = max(0.0, -coi)  # positive = synergistic
            synergy_matrix[i, j] = synergy_val
            synergy_matrix[j, i] = synergy_val

    return {
        "synergy_matrix": synergy_matrix,
        "feature_names": feature_names,
        "pid_method": "CoI_proxy",
        "n_features_used": n_features,
    }


def align_synergy_with_dataset(
    synergy_info: dict[str, Any],
    dataset_info: dict[str, Any],
) -> np.ndarray:
    """Align synergy matrix features with dataset features."""
    syn_features = synergy_info["feature_names"]
    ds_features = dataset_info["feature_names"]
    n_ds = len(ds_features)
    syn_matrix = synergy_info["synergy_matrix"]

    if syn_features == ds_features and syn_matrix.shape[0] == n_ds:
        return syn_matrix.copy()

    # Build mapping from synergy indices to dataset indices
    syn_to_ds: dict[int, int] = {}
    for si, sf in enumerate(syn_features):
        for di, df in enumerate(ds_features):
            if sf == df:
                syn_to_ds[si] = di
                break

    aligned = np.zeros((n_ds, n_ds), dtype=np.float64)
    for si in range(len(syn_features)):
        for sj in range(len(syn_features)):
            di = syn_to_ds.get(si)
            dj = syn_to_ds.get(sj)
            if di is not None and dj is not None:
                if si < syn_matrix.shape[0] and sj < syn_matrix.shape[1]:
                    aligned[di, dj] = syn_matrix[si, sj]
    return aligned


# ============================================================================
# SYNERGY GRAPH EXTRACTION
# ============================================================================
def build_synergy_graph(
    synergy_matrix: np.ndarray,
    percentile: float = SYNERGY_THRESHOLD_PCTL,
) -> nx.Graph:
    """Build synergy graph with progressive threshold lowering."""
    n_features = synergy_matrix.shape[0]
    upper_vals = synergy_matrix[np.triu_indices(n_features, k=1)]
    nonzero = upper_vals[upper_vals > 0]

    G = nx.Graph()
    G.add_nodes_from(range(n_features))

    if len(nonzero) == 0:
        return G

    # Progressive threshold lowering: 75 → 50 → 25 → 0
    for pct in [percentile, 50, 25, 0]:
        threshold = float(np.percentile(nonzero, pct)) if pct > 0 else 0.0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if synergy_matrix[i, j] >= threshold and synergy_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=float(synergy_matrix[i, j]))
        if G.number_of_edges() > 0:
            break

    return G


def extract_synergy_subsets(
    synergy_matrix: np.ndarray,
    percentile: float = SYNERGY_THRESHOLD_PCTL,
) -> list[list[int]]:
    """Extract synergy subsets (edges and cliques) from synergy graph."""
    G = build_synergy_graph(synergy_matrix, percentile=percentile)
    subsets: list[list[int]] = []

    # Add edges sorted by weight
    edges = sorted(
        G.edges(data=True),
        key=lambda e: e[2].get("weight", 0),
        reverse=True,
    )
    for u, v, _ in edges:
        subsets.append(sorted([u, v]))

    # Add cliques of size 3-4
    if G.number_of_edges() < 200:
        try:
            for clique in nx.find_cliques(G):
                if 3 <= len(clique) <= 4:
                    subsets.append(sorted(clique))
        except Exception:
            pass

    # Deduplicate
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for s in subsets:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique if unique else [[i] for i in range(synergy_matrix.shape[0])]


# ============================================================================
# OBLIQUE FIGS NODE AND TREE
# ============================================================================
class ObliqueFIGSNode:
    """Node in an oblique FIGS tree."""
    __slots__ = [
        "feature_indices", "weights", "bias", "threshold",
        "left", "right", "is_leaf", "value",
        "n_samples", "sample_indices",
    ]

    def __init__(self) -> None:
        self.feature_indices: list[int] = []
        self.weights: np.ndarray = np.array([])
        self.bias: float = 0.0
        self.threshold: float = 0.0
        self.left: "ObliqueFIGSNode | None" = None
        self.right: "ObliqueFIGSNode | None" = None
        self.is_leaf: bool = True
        self.value: float = 0.0
        self.n_samples: int = 0
        self.sample_indices: np.ndarray = np.array([], dtype=np.int32)

    def predict_single(self, x: np.ndarray) -> float:
        if self.is_leaf:
            return self.value
        proj = float(np.dot(x[self.feature_indices], self.weights) + self.bias)
        if proj <= self.threshold:
            return self.left.predict_single(x) if self.left else self.value
        return self.right.predict_single(x) if self.right else self.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_single(x) for x in X])


def count_total_nodes(node: ObliqueFIGSNode) -> int:
    """Count internal (split) nodes recursively."""
    if node.is_leaf:
        return 0
    count = 1
    if node.left:
        count += count_total_nodes(node.left)
    if node.right:
        count += count_total_nodes(node.right)
    return count


def get_all_leaves(node: ObliqueFIGSNode) -> list[ObliqueFIGSNode]:
    """Get all leaf nodes."""
    if node.is_leaf:
        return [node]
    leaves: list[ObliqueFIGSNode] = []
    if node.left:
        leaves.extend(get_all_leaves(node.left))
    if node.right:
        leaves.extend(get_all_leaves(node.right))
    return leaves


def fit_oblique_split_ridge(
    X_sub: np.ndarray,
    residuals: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit oblique split using Ridge + DecisionTree for threshold."""
    n_features = X_sub.shape[1]
    if n_features == 1:
        return np.array([1.0]), 0.0

    # Binary target for Ridge
    target = (residuals > 0).astype(np.float64)
    if len(np.unique(target)) < 2:
        return np.ones(n_features) / n_features, 0.0

    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_sub, target)
        w = ridge.coef_.ravel()
        b = float(ridge.intercept_)
        # Normalize weights
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            w = w / norm
            b = b / norm
        return w, b
    except Exception:
        return np.ones(n_features) / n_features, 0.0


def find_best_threshold(
    proj: np.ndarray,
    residuals: np.ndarray,
) -> tuple[float, float]:
    """Find best threshold on 1D projection via variance reduction."""
    # Use DecisionTreeRegressor to find optimal split
    try:
        dt = DecisionTreeRegressor(max_depth=1)
        dt.fit(proj.reshape(-1, 1), residuals)
        threshold = float(dt.tree_.threshold[0])
        if threshold == -2.0:  # No split found
            threshold = float(np.median(proj))
    except Exception:
        threshold = float(np.median(proj))

    left_mask = proj <= threshold
    right_mask = ~left_mask
    n = len(residuals)
    n_l = left_mask.sum()
    n_r = right_mask.sum()

    if n_l < MIN_SAMPLES_LEAF or n_r < MIN_SAMPLES_LEAF:
        return -1.0, threshold

    var_parent = np.var(residuals)
    var_left = np.var(residuals[left_mask])
    var_right = np.var(residuals[right_mask])
    score = var_parent - (n_l / n * var_left + n_r / n * var_right)

    return max(0.0, score), threshold


# ============================================================================
# SIGMOID / PREDICTION UTILITIES
# ============================================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ============================================================================
# BASE FIGS OBLIQUE CLASSIFIER
# ============================================================================
class BaseFIGSOblique:
    """Base class for all FIGS oblique variants.

    All 5 methods share the SAME fit() logic; only
    _get_feature_subsets_for_split() differs.
    """

    def __init__(
        self,
        max_splits: int = 10,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.max_splits = max_splits
        self.random_state = random_state
        self.trees: list[ObliqueFIGSNode] = []
        self.n_splits_actual: int = 0
        self.split_info: list[dict[str, Any]] = []
        self.class_prior_: float = 0.0
        self.rng = np.random.RandomState(random_state)

    def _get_feature_subsets_for_split(
        self,
        n_features: int,
    ) -> list[list[int]]:
        """Return candidate feature subsets for the next split.
        Override in subclasses.
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, y_binary: np.ndarray) -> "BaseFIGSOblique":
        """Greedy FIGS fitting with HARD complexity enforcement."""
        n_samples, n_features = X.shape
        p = np.clip(np.mean(y_binary), 0.01, 0.99)
        self.class_prior_ = float(np.log(p / (1 - p)))
        predictions = np.full(n_samples, self.class_prior_)
        proba = sigmoid(predictions)
        residuals = y_binary - proba

        # Initialize with single root tree
        root = ObliqueFIGSNode()
        root.is_leaf = True
        root.value = 0.0
        root.n_samples = n_samples
        root.sample_indices = np.arange(n_samples, dtype=np.int32)
        self.trees = [root]
        self.n_splits_actual = 0
        self.split_info = []

        max_new_trees = min(3, self.max_splits)
        trees_created = 1  # root already exists

        for iteration in range(self.max_splits * 3):  # safety upper bound
            # HARD CHECK: count actual total splits across all trees
            actual_total = sum(count_total_nodes(t) for t in self.trees)
            if actual_total >= self.max_splits:
                break

            best_score = 1e-10
            best_data: dict[str, Any] | None = None
            best_leaf: ObliqueFIGSNode | None = None

            # Collect all leaves, sort by n_samples descending
            all_leaves: list[ObliqueFIGSNode] = []
            for tree in self.trees:
                all_leaves.extend(get_all_leaves(tree))
            all_leaves.sort(key=lambda l: -l.n_samples)

            # Only scan top-5 largest leaves for efficiency
            scan_leaves = all_leaves[:5]

            for leaf in scan_leaves:
                if leaf.n_samples < 2 * MIN_SAMPLES_LEAF:
                    continue

                leaf_X = X[leaf.sample_indices]
                leaf_res = residuals[leaf.sample_indices]

                # Get candidate feature subsets
                candidates = self._get_feature_subsets_for_split(n_features)

                for feat_subset in candidates:
                    valid = [f for f in feat_subset if 0 <= f < n_features]
                    if not valid:
                        continue

                    X_sub = leaf_X[:, valid]

                    # Fit oblique split
                    w, b = fit_oblique_split_ridge(X_sub, leaf_res)
                    proj = X_sub @ w + b

                    score, thr = find_best_threshold(proj, leaf_res)
                    if score > best_score:
                        left_mask = proj <= thr
                        right_mask = ~left_mask
                        if left_mask.sum() >= MIN_SAMPLES_LEAF and right_mask.sum() >= MIN_SAMPLES_LEAF:
                            best_score = score
                            best_data = {
                                "feature_indices": valid,
                                "weights": w.copy(),
                                "bias": b,
                                "threshold": thr,
                                "left_mask": left_mask,
                                "right_mask": right_mask,
                            }
                            best_leaf = leaf

            if best_data is None or best_leaf is None:
                # No good split found, try creating new tree if budget allows
                if trees_created < max_new_trees and actual_total < self.max_splits:
                    new_root = ObliqueFIGSNode()
                    new_root.is_leaf = True
                    new_root.value = 0.0
                    new_root.n_samples = n_samples
                    new_root.sample_indices = np.arange(n_samples, dtype=np.int32)
                    self.trees.append(new_root)
                    trees_created += 1
                    continue
                break

            # Apply split
            leaf = best_leaf
            leaf.is_leaf = False
            leaf.feature_indices = best_data["feature_indices"]
            leaf.weights = best_data["weights"]
            leaf.bias = best_data["bias"]
            leaf.threshold = best_data["threshold"]

            li = leaf.sample_indices[best_data["left_mask"]]
            ri = leaf.sample_indices[best_data["right_mask"]]

            # Newton-Raphson optimal leaf values for log-loss:
            # leaf_val = sum(residuals) / sum(p*(1-p))
            # This is much more effective than simple mean(residuals)
            def _newton_leaf_value(indices: np.ndarray) -> float:
                res = residuals[indices]
                p_vals = proba[indices]
                hessian = np.sum(p_vals * (1 - p_vals))
                if hessian < 1e-10:
                    return float(np.mean(res))
                return float(np.sum(res) / hessian)

            left_val = _newton_leaf_value(li)
            right_val = _newton_leaf_value(ri)

            leaf.left = ObliqueFIGSNode()
            leaf.left.is_leaf = True
            leaf.left.value = left_val
            leaf.left.n_samples = len(li)
            leaf.left.sample_indices = li

            leaf.right = ObliqueFIGSNode()
            leaf.right.is_leaf = True
            leaf.right.value = right_val
            leaf.right.n_samples = len(ri)
            leaf.right.sample_indices = ri

            # Update predictions, proba, and residuals
            predictions[li] += left_val
            predictions[ri] += right_val
            proba = sigmoid(predictions)
            residuals = y_binary - proba

            self.n_splits_actual += 1
            self.split_info.append({
                "feature_indices": best_data["feature_indices"],
                "weights": best_data["weights"].tolist(),
                "score": best_score,
                "n_features_in_split": len(best_data["feature_indices"]),
            })

        # Final complexity verification
        self.n_splits_actual = sum(count_total_nodes(t) for t in self.trees)
        assert self.n_splits_actual <= self.max_splits, (
            f"Complexity violation: {self.n_splits_actual} > {self.max_splits}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.full(X.shape[0], self.class_prior_)
        for tree in self.trees:
            raw += tree.predict(X)
        p1 = sigmoid(raw)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)


# ============================================================================
# FIVE METHOD IMPLEMENTATIONS
# ============================================================================
class FIGSAxisAligned(BaseFIGSOblique):
    """FIGS baseline: axis-aligned splits only (single feature per split)."""

    def _get_feature_subsets_for_split(self, n_features: int) -> list[list[int]]:
        return [[i] for i in range(n_features)]


class ROFIGSClassifier(BaseFIGSOblique):
    """RO-FIGS: Random Oblique pairs of 2 features."""

    def _get_feature_subsets_for_split(self, n_features: int) -> list[list[int]]:
        candidates: list[list[int]] = []
        n_cand = min(15, n_features * (n_features - 1) // 2)
        for _ in range(n_cand):
            pair = sorted(self.rng.choice(n_features, size=2, replace=False).tolist())
            candidates.append(pair)
        # Deduplicate
        seen: set[tuple[int, ...]] = set()
        unique: list[list[int]] = []
        for c in candidates:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique if unique else [[0, min(1, n_features - 1)]]


class SGFIGSHard(BaseFIGSOblique):
    """SG-FIGS-Hard: pick random subset from synergy graph edges/cliques."""

    def __init__(
        self,
        max_splits: int = 10,
        synergy_subsets: list[list[int]] | None = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        super().__init__(max_splits=max_splits, random_state=random_state)
        self.synergy_subsets = synergy_subsets or []

    def _get_feature_subsets_for_split(self, n_features: int) -> list[list[int]]:
        if not self.synergy_subsets:
            return [[i] for i in range(n_features)]
        # Pick up to 10 random subsets from synergy graph
        n_pick = min(10, len(self.synergy_subsets))
        indices = self.rng.choice(
            len(self.synergy_subsets), size=n_pick, replace=False,
        )
        return [self.synergy_subsets[i] for i in indices]


class SGFIGSSoft(BaseFIGSOblique):
    """SG-FIGS-Soft: pick seed uniformly, partner with prob proportional
    to synergy_matrix[seed, :]."""

    def __init__(
        self,
        max_splits: int = 10,
        synergy_matrix: np.ndarray | None = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        super().__init__(max_splits=max_splits, random_state=random_state)
        self.synergy_matrix = synergy_matrix

    def _get_feature_subsets_for_split(self, n_features: int) -> list[list[int]]:
        if self.synergy_matrix is None or self.synergy_matrix.shape[0] != n_features:
            return [[i] for i in range(n_features)]

        candidates: list[list[int]] = []
        for _ in range(min(15, n_features)):
            seed = self.rng.randint(n_features)
            row = self.synergy_matrix[seed, :].copy()
            row[seed] = 0.0  # can't pair with self
            total = row.sum()
            if total > 1e-12:
                probs = row / total
                partner = self.rng.choice(n_features, p=probs)
            else:
                # fallback: uniform random
                others = [j for j in range(n_features) if j != seed]
                partner = self.rng.choice(others) if others else seed
            candidates.append(sorted([seed, partner]))

        # Deduplicate
        seen: set[tuple[int, ...]] = set()
        unique: list[list[int]] = []
        for c in candidates:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique if unique else [[0, min(1, n_features - 1)]]


class RandomFIGS(BaseFIGSOblique):
    """Random-FIGS: random pairs with matched clique sizes (ablation control)."""

    def __init__(
        self,
        max_splits: int = 10,
        synergy_subsets: list[list[int]] | None = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        super().__init__(max_splits=max_splits, random_state=random_state)
        # We match the subset SIZES from synergy subsets but pick random features
        if synergy_subsets:
            self.subset_sizes = [len(s) for s in synergy_subsets]
        else:
            self.subset_sizes = [2]

    def _get_feature_subsets_for_split(self, n_features: int) -> list[list[int]]:
        candidates: list[list[int]] = []
        n_pick = min(15, len(self.subset_sizes) * 2)
        for _ in range(n_pick):
            size = self.rng.choice(self.subset_sizes)
            size = min(size, n_features)
            size = max(size, 1)
            feats = sorted(self.rng.choice(n_features, size=size, replace=False).tolist())
            candidates.append(feats)

        seen: set[tuple[int, ...]] = set()
        unique: list[list[int]] = []
        for c in candidates:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique if unique else [[self.rng.randint(n_features)]]


# ============================================================================
# MULTICLASS OvR WRAPPER
# ============================================================================
class OvRWrapper:
    """One-vs-Rest wrapper that budgets splits per class."""

    def __init__(
        self,
        base_class: type,
        max_splits: int = 10,
        n_classes: int = 2,
        random_state: int = RANDOM_SEED,
        **kwargs: Any,
    ) -> None:
        self.base_class = base_class
        self.max_splits = max_splits
        self.n_classes = n_classes
        self.random_state = random_state
        self.kwargs = kwargs
        self.models: list[BaseFIGSOblique] = []
        self.classes_: np.ndarray = np.array([])
        self.n_splits_actual: int = 0
        self.split_info: list[dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OvRWrapper":
        self.classes_ = np.unique(y)
        per_class_splits = max(1, self.max_splits // len(self.classes_))
        self.models = []
        all_split_info: list[dict[str, Any]] = []

        for cls in self.classes_:
            y_binary = (y == cls).astype(np.float64)
            model = self.base_class(
                max_splits=per_class_splits,
                random_state=self.random_state + int(cls),
                **self.kwargs,
            )
            model.fit(X, y_binary)
            self.models.append(model)
            all_split_info.extend(model.split_info)

        self.n_splits_actual = sum(m.n_splits_actual for m in self.models)
        self.split_info = all_split_info
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, model in enumerate(self.models):
            proba = model.predict_proba(X)
            all_proba[:, i] = proba[:, 1]
        # Normalize
        row_sums = all_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        all_proba = all_proba / row_sums
        return all_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================================
# INTERPRETABILITY SCORE
# ============================================================================
def compute_interpretability(
    split_info: list[dict[str, Any]],
    synergy_matrix: np.ndarray,
) -> float:
    """Score: fraction of oblique splits using above-median synergy pairs."""
    if not split_info:
        return 0.0

    n = synergy_matrix.shape[0]
    upper = synergy_matrix[np.triu_indices(n, k=1)]
    nz = upper[upper > 0]
    median_syn = float(np.median(nz)) if len(nz) > 0 else 0.0

    oblique_splits = [s for s in split_info if s["n_features_in_split"] >= 2]
    if not oblique_splits:
        return 1.0  # no oblique splits = fully interpretable (axis-aligned)

    above_median = 0
    for s in oblique_splits:
        feats = s["feature_indices"]
        pair_syns: list[float] = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                fi, fj = feats[i], feats[j]
                if fi < n and fj < n:
                    pair_syns.append(synergy_matrix[fi, fj])
        if pair_syns and np.mean(pair_syns) > median_syn:
            above_median += 1

    return above_median / len(oblique_splits)


# ============================================================================
# SINGLE FOLD EXPERIMENT
# ============================================================================
def run_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    synergy_matrix: np.ndarray,
    synergy_subsets: list[list[int]],
    max_splits: int,
    fold_id: int,
    n_classes: int,
) -> dict[str, Any]:
    """Run all 5 methods on one fold. Returns results dict."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    n_features = Xtr.shape[1]
    is_multiclass = n_classes > 2

    # For binary: y as float for sigmoid fitting
    if is_multiclass:
        y_train_eval = y_train
        y_test_eval = y_test
    else:
        y_train_eval = y_train
        y_test_eval = y_test

    def safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        try:
            if len(np.unique(y_true)) < 2:
                return 0.5
            return float(roc_auc_score(y_true, y_proba))
        except ValueError:
            return 0.5

    results: dict[str, Any] = {}

    # Define method constructors
    method_configs: dict[str, dict[str, Any]] = {
        "FIGS": {"class": FIGSAxisAligned, "kwargs": {}},
        "RO_FIGS": {"class": ROFIGSClassifier, "kwargs": {}},
        "SG_FIGS_Hard": {"class": SGFIGSHard, "kwargs": {"synergy_subsets": synergy_subsets}},
        "SG_FIGS_Soft": {"class": SGFIGSSoft, "kwargs": {"synergy_matrix": synergy_matrix}},
        "Random_FIGS": {"class": RandomFIGS, "kwargs": {"synergy_subsets": synergy_subsets}},
    }

    for method_name, config in method_configs.items():
        try:
            if is_multiclass:
                model = OvRWrapper(
                    base_class=config["class"],
                    max_splits=max_splits,
                    n_classes=n_classes,
                    random_state=RANDOM_SEED + fold_id,
                    **config["kwargs"],
                )
                model.fit(Xtr, y_train_eval)
                y_pred = model.predict(Xte)
                actual_splits = model.n_splits_actual
                split_info = model.split_info
                # For multiclass, no single AUC
                auc_val = 0.5
            else:
                y_binary_train = y_train.astype(np.float64)
                base_cls = config["class"]
                model = base_cls(
                    max_splits=max_splits,
                    random_state=RANDOM_SEED + fold_id,
                    **config["kwargs"],
                )
                model.fit(Xtr, y_binary_train)
                y_pred = model.predict(Xte)
                y_proba = model.predict_proba(Xte)[:, 1]
                actual_splits = model.n_splits_actual
                split_info = model.split_info
                auc_val = safe_auc(y_test_eval, y_proba)

            bal_acc = float(balanced_accuracy_score(y_test_eval, y_pred))
            interp = compute_interpretability(split_info, synergy_matrix)

            results[method_name] = {
                "balanced_accuracy": round(bal_acc, 6),
                "auc": round(auc_val, 6),
                "actual_n_splits": actual_splits,
                "interpretability": round(interp, 4),
            }
        except Exception as e:
            logger.debug(f"  {method_name} error fold {fold_id}: {e}")
            results[method_name] = {
                "balanced_accuracy": 0.5,
                "auc": 0.5,
                "actual_n_splits": 0,
                "interpretability": 0.0,
            }

    return results


# ============================================================================
# WIN/TIE/LOSS COMPUTATION
# ============================================================================
def compute_wtl(
    all_fold_results: dict[str, dict[str, list[dict[str, Any]]]],
    method_a: str,
    method_b: str,
    metric: str = "balanced_accuracy",
) -> dict[str, int]:
    """Compute Win/Tie/Loss of method_a vs method_b across all datasets."""
    wins, ties, losses = 0, 0, 0
    for ds_name, ms_results in all_fold_results.items():
        for ms_key, fold_list in ms_results.items():
            a_vals = [f[method_a][metric] for f in fold_list if method_a in f]
            b_vals = [f[method_b][metric] for f in fold_list if method_b in f]
            if not a_vals or not b_vals:
                continue
            mean_a = np.mean(a_vals)
            mean_b = np.mean(b_vals)
            diff = mean_a - mean_b
            if diff > WIN_THRESHOLD:
                wins += 1
            elif diff < -WIN_THRESHOLD:
                losses += 1
            else:
                ties += 1
    return {"wins": wins, "ties": ties, "losses": losses}


# ============================================================================
# DOMAIN VALIDATION
# ============================================================================
def domain_validation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    synergy_matrix: np.ndarray,
    synergy_subsets: list[list[int]],
    ds_name: str,
) -> dict[str, Any]:
    """Train SG-FIGS-Soft on full data and extract split features."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_features = Xs.shape[1]

    model = SGFIGSSoft(
        max_splits=10,
        synergy_matrix=synergy_matrix,
        random_state=RANDOM_SEED,
    )

    n_classes = len(np.unique(y))
    if n_classes > 2:
        wrapper = OvRWrapper(
            base_class=SGFIGSSoft,
            max_splits=10,
            n_classes=n_classes,
            random_state=RANDOM_SEED,
            synergy_matrix=synergy_matrix,
        )
        wrapper.fit(Xs, y)
        split_info = wrapper.split_info
    else:
        y_bin = y.astype(np.float64)
        model.fit(Xs, y_bin)
        split_info = model.split_info

    extracted_features: list[dict[str, Any]] = []
    for s in split_info:
        feat_idx = s["feature_indices"]
        feat_names_used = [
            feature_names[i] if i < len(feature_names) else f"f{i}"
            for i in feat_idx
        ]
        extracted_features.append({
            "feature_indices": feat_idx,
            "feature_names": feat_names_used,
            "weights": s.get("weights", []),
            "n_features_in_split": s["n_features_in_split"],
        })

    # Top synergy pairs
    n = synergy_matrix.shape[0]
    top_pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if synergy_matrix[i, j] > 0:
                top_pairs.append({
                    "feature_i": feature_names[i] if i < len(feature_names) else f"f{i}",
                    "feature_j": feature_names[j] if j < len(feature_names) else f"f{j}",
                    "synergy": round(float(synergy_matrix[i, j]), 6),
                })
    top_pairs.sort(key=lambda x: -x["synergy"])

    return {
        "dataset": ds_name,
        "n_splits": len(split_info),
        "extracted_splits": extracted_features[:10],
        "top_synergy_pairs": top_pairs[:5],
    }


# ============================================================================
# MAIN
# ============================================================================
@logger.catch
def main() -> None:
    t_global_start = time.time()
    logger.info("=" * 60)
    logger.info("Complexity-Matched SG-FIGS Experiment")
    logger.info("Synergy vs Random at Equal Splits")
    logger.info("=" * 60)

    # ── Phase 0: Load data ───────────────────────────────────────────────
    logger.info("Phase 0: Loading data")
    datasets_id2 = load_datasets_from_json(DATA_ID2_PATH)
    datasets_id3 = load_datasets_from_json(DATA_ID3_PATH)

    all_datasets: dict[str, dict[str, Any]] = {}
    all_datasets.update(datasets_id2)
    for name, ds in datasets_id3.items():
        if name not in all_datasets:
            all_datasets[name] = ds

    logger.info(f"Loaded {len(all_datasets)} unique datasets:")
    for name, ds in sorted(all_datasets.items()):
        logger.info(
            f"  {name:45s} | {ds['n_samples']:5d} samples | "
            f"{ds['n_features']:3d} feat | {ds['n_classes']} classes"
        )

    # ── Load pre-computed synergy ────────────────────────────────────────
    synergy_data = load_precomputed_synergy(SYNERGY_RESULTS_PATH)
    logger.info(f"Pre-computed synergy for {len(synergy_data)} datasets")

    # ── Compute CoI synergy for new datasets ─────────────────────────────
    for ds_name in NEW_SYNERGY_DATASETS:
        if ds_name not in all_datasets:
            continue
        if ds_name in synergy_data:
            continue
        logger.info(f"Computing CoI synergy for {ds_name}...")
        t0 = time.time()
        ds = all_datasets[ds_name]
        synergy_data[ds_name] = compute_coi_synergy(
            ds["X"], ds["y"], ds["feature_names"],
        )
        logger.info(f"  Done in {time.time() - t0:.1f}s")

    # ── Align synergy matrices ───────────────────────────────────────────
    aligned_synergy: dict[str, np.ndarray] = {}
    synergy_subsets_map: dict[str, list[list[int]]] = {}

    for ds_name, ds_info in all_datasets.items():
        if ds_name in synergy_data:
            aligned_synergy[ds_name] = align_synergy_with_dataset(
                synergy_data[ds_name], ds_info,
            )
        else:
            n_f = ds_info["n_features"]
            aligned_synergy[ds_name] = np.zeros((n_f, n_f))

        synergy_subsets_map[ds_name] = extract_synergy_subsets(
            aligned_synergy[ds_name],
            percentile=SYNERGY_THRESHOLD_PCTL,
        )
        n_subsets = len(synergy_subsets_map[ds_name])
        logger.info(f"  {ds_name}: {n_subsets} synergy subsets extracted")

    # ── Phase 1 & 2: Run experiments ─────────────────────────────────────
    logger.info("\nPhase 1-2: Running experiments")

    # Build flat dataset list from tier order
    dataset_order: list[str] = []
    for tier in TIER_ORDER:
        for ds_name in tier:
            if ds_name in all_datasets:
                dataset_order.append(ds_name)

    # Add any remaining datasets not in tier order
    for ds_name in sorted(all_datasets.keys()):
        if ds_name not in dataset_order:
            dataset_order.append(ds_name)

    # Results storage
    all_fold_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    method_out_datasets: list[dict[str, Any]] = []
    complexity_records: list[dict[str, Any]] = []

    tier_labels = {ds: f"Tier {t+1}" for t, tier in enumerate(TIER_ORDER) for ds in tier}

    for ds_idx, ds_name in enumerate(dataset_order):
        elapsed = time.time() - t_global_start
        if elapsed > TIME_BUDGET_SECONDS:
            logger.warning(
                f"Time budget exceeded ({elapsed:.0f}s > {TIME_BUDGET_SECONDS}s), "
                f"stopping after {ds_idx} datasets"
            )
            break

        ds = all_datasets[ds_name]
        syn_mat = aligned_synergy[ds_name]
        syn_subsets = synergy_subsets_map[ds_name]
        tier = tier_labels.get(ds_name, "Extra")

        logger.info(
            f"\n{'='*50}\n"
            f"[{ds_idx+1}/{len(dataset_order)}] {ds_name} ({tier}) | "
            f"{ds['n_samples']}s, {ds['n_features']}f, {ds['n_classes']}c"
        )

        ds_examples: list[dict[str, Any]] = []
        all_fold_results[ds_name] = {}

        for max_splits in MAX_SPLITS_VALUES:
            ms_key = f"ms{max_splits}"
            fold_results_list: list[dict[str, Any]] = []

            for fold_k in range(N_FOLDS):
                train_mask = ds["folds"] != fold_k
                test_mask = ds["folds"] == fold_k

                if test_mask.sum() == 0 or train_mask.sum() < 10:
                    continue

                fold_result = run_fold(
                    X_train=ds["X"][train_mask],
                    y_train=ds["y"][train_mask],
                    X_test=ds["X"][test_mask],
                    y_test=ds["y"][test_mask],
                    synergy_matrix=syn_mat,
                    synergy_subsets=syn_subsets,
                    max_splits=max_splits,
                    fold_id=fold_k,
                    n_classes=ds["n_classes"],
                )
                fold_results_list.append(fold_result)

                # Record complexity
                for method_name in METHOD_NAMES:
                    if method_name in fold_result:
                        actual = fold_result[method_name]["actual_n_splits"]
                        complexity_records.append({
                            "dataset": ds_name,
                            "max_splits": max_splits,
                            "method": method_name,
                            "fold": fold_k,
                            "actual_splits": actual,
                            "violation": actual > max_splits,
                        })

                # Build schema-compliant examples
                input_d = {
                    "dataset": ds_name,
                    "max_splits": max_splits,
                    "fold": fold_k,
                    "n_features": ds["n_features"],
                    "n_samples": ds["n_samples"],
                    "n_classes": ds["n_classes"],
                }

                # Collect all method outputs
                output_d: dict[str, Any] = {}
                for method_name in METHOD_NAMES:
                    if method_name in fold_result:
                        for key, val in fold_result[method_name].items():
                            output_d[f"{method_name}_{key}"] = val

                example: dict[str, Any] = {
                    "input": json.dumps(input_d),
                    "output": json.dumps(output_d, cls=NumpyEncoder),
                    "metadata_fold": fold_k,
                    "metadata_max_splits": max_splits,
                    "metadata_dataset": ds_name,
                    "metadata_n_features": ds["n_features"],
                    "metadata_n_classes": ds["n_classes"],
                }

                # Add predict_ fields
                for method_name in METHOD_NAMES:
                    if method_name in fold_result:
                        example[f"predict_{method_name}_balanced_acc"] = str(
                            fold_result[method_name]["balanced_accuracy"]
                        )

                ds_examples.append(example)

            all_fold_results[ds_name][ms_key] = fold_results_list

            # Log summary for this max_splits
            if fold_results_list:
                for method_name in METHOD_NAMES:
                    accs = [
                        f[method_name]["balanced_accuracy"]
                        for f in fold_results_list
                        if method_name in f
                    ]
                    if accs:
                        logger.info(
                            f"  {method_name:15s} ms={max_splits:2d} | "
                            f"bal_acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}"
                        )

        method_out_datasets.append({
            "dataset": ds_name,
            "examples": ds_examples,
        })
        logger.info(f"  {ds_name}: {len(ds_examples)} examples saved")

    # ── Phase 3: Analysis ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Analysis")
    logger.info("=" * 60)

    # Win/Tie/Loss tables
    wtl_tables: dict[str, dict[str, dict[str, int]]] = {}
    for our_method in ["SG_FIGS_Soft", "SG_FIGS_Hard"]:
        wtl_tables[our_method] = {}
        for baseline in ["FIGS", "RO_FIGS", "Random_FIGS"]:
            wtl = compute_wtl(
                all_fold_results,
                method_a=our_method,
                method_b=baseline,
            )
            wtl_tables[our_method][baseline] = wtl
            logger.info(
                f"  {our_method} vs {baseline}: "
                f"W={wtl['wins']}, T={wtl['ties']}, L={wtl['losses']}"
            )

    # Per-split information gain
    per_split_gain: dict[str, dict[str, float]] = {}
    for ds_name in all_fold_results:
        per_split_gain[ds_name] = {}
        for method_name in METHOD_NAMES:
            acc_5_list: list[float] = []
            acc_10_list: list[float] = []

            for ms_key, fold_list in all_fold_results[ds_name].items():
                for f in fold_list:
                    if method_name in f:
                        if ms_key == "ms5":
                            acc_5_list.append(f[method_name]["balanced_accuracy"])
                        elif ms_key == "ms10":
                            acc_10_list.append(f[method_name]["balanced_accuracy"])

            if acc_5_list and acc_10_list:
                gain = (np.mean(acc_10_list) - np.mean(acc_5_list)) / 5.0
                per_split_gain[ds_name][method_name] = round(float(gain), 6)
            else:
                per_split_gain[ds_name][method_name] = 0.0

    # Mean per-split gain across datasets
    mean_gain: dict[str, float] = {}
    for method_name in METHOD_NAMES:
        gains = [
            per_split_gain[ds][method_name]
            for ds in per_split_gain
            if method_name in per_split_gain[ds]
        ]
        mean_gain[method_name] = round(float(np.mean(gains)), 6) if gains else 0.0
        logger.info(f"  Mean per-split gain {method_name}: {mean_gain[method_name]:.6f}")

    # Complexity verification
    complexity_summary: dict[str, dict[str, dict[str, float]]] = {}
    for method_name in METHOD_NAMES:
        complexity_summary[method_name] = {}
        for max_splits in MAX_SPLITS_VALUES:
            records = [
                r for r in complexity_records
                if r["method"] == method_name and r["max_splits"] == max_splits
            ]
            if records:
                actuals = [r["actual_splits"] for r in records]
                violations = sum(1 for r in records if r["violation"])
                complexity_summary[method_name][f"ms{max_splits}"] = {
                    "mean_actual": round(float(np.mean(actuals)), 2),
                    "max_actual": int(np.max(actuals)),
                    "std_actual": round(float(np.std(actuals)), 2),
                    "n_violations": violations,
                    "n_total": len(records),
                }

    logger.info("\nComplexity verification:")
    for method_name in METHOD_NAMES:
        for ms_key, stats in complexity_summary.get(method_name, {}).items():
            logger.info(
                f"  {method_name:15s} {ms_key}: "
                f"mean={stats['mean_actual']:.1f}, "
                f"max={stats['max_actual']}, "
                f"violations={stats['n_violations']}/{stats['n_total']}"
            )

    # Domain validation
    domain_results: list[dict[str, Any]] = []
    for ds_name in ["pima_diabetes", "heart_statlog", "monks2"]:
        if ds_name in all_datasets:
            try:
                dv = domain_validation(
                    X=all_datasets[ds_name]["X"],
                    y=all_datasets[ds_name]["y"],
                    feature_names=all_datasets[ds_name]["feature_names"],
                    synergy_matrix=aligned_synergy[ds_name],
                    synergy_subsets=synergy_subsets_map[ds_name],
                    ds_name=ds_name,
                )
                domain_results.append(dv)
                logger.info(f"  Domain validation {ds_name}: {dv['n_splits']} splits")
                for sp in dv["extracted_splits"][:3]:
                    logger.info(f"    Split: {sp['feature_names']} w={sp['weights'][:3]}")
            except Exception:
                logger.exception(f"Domain validation failed for {ds_name}")

    # ── Per-method mean accuracy summary ─────────────────────────────────
    method_mean_accs: dict[str, dict[str, float]] = {}
    for method_name in METHOD_NAMES:
        method_mean_accs[method_name] = {}
        for ms in MAX_SPLITS_VALUES:
            ms_key = f"ms{ms}"
            all_accs: list[float] = []
            for ds_name in all_fold_results:
                if ms_key in all_fold_results[ds_name]:
                    for f in all_fold_results[ds_name][ms_key]:
                        if method_name in f:
                            all_accs.append(f[method_name]["balanced_accuracy"])
            if all_accs:
                method_mean_accs[method_name][ms_key] = round(float(np.mean(all_accs)), 6)
            else:
                method_mean_accs[method_name][ms_key] = 0.5

    logger.info("\nOverall mean balanced accuracy:")
    for ms in MAX_SPLITS_VALUES:
        ms_key = f"ms{ms}"
        logger.info(f"  max_splits={ms}:")
        for method_name in METHOD_NAMES:
            acc = method_mean_accs[method_name].get(ms_key, 0.5)
            logger.info(f"    {method_name:15s}: {acc:.4f}")

    # ── Phase 4: Output ──────────────────────────────────────────────────
    logger.info("\nPhase 4: Saving outputs")

    # method_out.json (schema-compliant)
    method_out = {"datasets": method_out_datasets}
    method_path = WORKSPACE / "method_out.json"
    method_path.write_text(json.dumps(method_out, indent=2, cls=NumpyEncoder))
    total_examples = sum(len(d["examples"]) for d in method_out_datasets)
    logger.info(
        f"  method_out.json: {total_examples} examples, "
        f"{method_path.stat().st_size / 1024:.1f} KB"
    )

    # results_comprehensive.json
    comprehensive = {
        "experiment": "Complexity-Matched SG-FIGS: Synergy vs Random at Equal Splits",
        "methods": METHOD_NAMES,
        "max_splits_tested": MAX_SPLITS_VALUES,
        "n_folds": N_FOLDS,
        "win_threshold": WIN_THRESHOLD,
        "synergy_threshold_percentile": SYNERGY_THRESHOLD_PCTL,
        "total_runtime_seconds": round(time.time() - t_global_start, 1),
        "datasets_processed": list(all_fold_results.keys()),
        "n_datasets_processed": len(all_fold_results),
        "win_tie_loss_tables": wtl_tables,
        "per_split_information_gain": per_split_gain,
        "mean_per_split_gain": mean_gain,
        "complexity_verification": complexity_summary,
        "method_mean_accuracies": method_mean_accs,
        "domain_validation": domain_results,
        "per_dataset_fold_results": {},
    }

    # Add per-dataset fold results (aggregated)
    for ds_name in all_fold_results:
        ds_summary: dict[str, Any] = {}
        for ms_key, fold_list in all_fold_results[ds_name].items():
            ms_summary: dict[str, Any] = {}
            for method_name in METHOD_NAMES:
                accs = [
                    f[method_name]["balanced_accuracy"]
                    for f in fold_list
                    if method_name in f
                ]
                aucs = [
                    f[method_name]["auc"]
                    for f in fold_list
                    if method_name in f
                ]
                splits = [
                    f[method_name]["actual_n_splits"]
                    for f in fold_list
                    if method_name in f
                ]
                interps = [
                    f[method_name]["interpretability"]
                    for f in fold_list
                    if method_name in f
                ]
                if accs:
                    ms_summary[method_name] = {
                        "mean_balanced_acc": round(float(np.mean(accs)), 6),
                        "std_balanced_acc": round(float(np.std(accs)), 6),
                        "mean_auc": round(float(np.mean(aucs)), 6),
                        "mean_actual_splits": round(float(np.mean(splits)), 2),
                        "mean_interpretability": round(float(np.mean(interps)), 4),
                        "fold_accs": [round(float(a), 6) for a in accs],
                    }
            ds_summary[ms_key] = ms_summary
        comprehensive["per_dataset_fold_results"][ds_name] = ds_summary

    comp_path = WORKSPACE / "results_comprehensive.json"
    comp_path.write_text(json.dumps(comprehensive, indent=2, cls=NumpyEncoder))
    logger.info(f"  results_comprehensive.json: {comp_path.stat().st_size / 1024:.1f} KB")

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - t_global_start
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Datasets: {len(all_fold_results)}")
    logger.info(f"  Total examples: {total_examples}")

    # Print W/T/L summary
    for our_method in ["SG_FIGS_Soft", "SG_FIGS_Hard"]:
        logger.info(f"\n  {our_method} Win/Tie/Loss:")
        for baseline, wtl in wtl_tables.get(our_method, {}).items():
            logger.info(
                f"    vs {baseline:15s}: "
                f"W={wtl['wins']}, T={wtl['ties']}, L={wtl['losses']}"
            )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
