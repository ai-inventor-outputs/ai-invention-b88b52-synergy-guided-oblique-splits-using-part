# SG-FIGS Spec

## Summary

Comprehensive technical specification for implementing Synergy-Guided Oblique FIGS (SG-FIGS), covering: dit PID API with exact synergy extraction keys ((0,1),), FIGS source code modification points (Node extension, _construct_node_with_stump replacement, predict modification), complete RO-FIGS source code analysis with SPyCT/ODT oblique split construction and L1/2 regularization, synergy graph construction algorithm (discretize, pairwise PID, threshold, networkx cliques), and evaluation protocol with 22 OpenML datasets. Saved to resources/sg_figs_technical_specification.md (626 lines, 17KB).

## Research Findings

# Technical Specification for Synergy-Guided Oblique FIGS (SG-FIGS)

## 1. PID Library Selection and API Specification

### 1.1 Recommended Library: dit

The `dit` library (discrete information theory) is the recommended choice for computing pairwise Partial Information Decomposition (PID) synergy [1, 2]. The library implements the Williams & Beer framework with multiple PID measures including `PID_WB`, `PID_BROJA`, `PID_CCS`, `PID_GK`, and others [3].

**Why dit over alternatives:**
- **pidpy**: Only supports binary features (0/1) for arbitrary numbers of variables; non-binary integer variables are limited to triplets only [4]. This makes pidpy unsuitable for discretized continuous features with >2 bins.
- **PIDF** (from Westphal et al., AISTATS'25): Uses a custom neural mutual information estimation (MINE) approach rather than classical PID [5, 6]. While scalable, it does not provide the exact PID decomposition needed for synergy graph construction.
- **JIDT** (Java): Requires jpype bridge; useful as fallback if dit is too slow for >50 features [2].

### 1.2 Exact API for Synergy Computation

**Installation:** `uv pip install dit`

**Core API pattern for bivariate PID:**

```python
import dit
from dit.pid import PID_WB  # or PID_BROJA

# Create distribution from tuple outcomes (feature_i, feature_j, target)
d = dit.Distribution(['000', '011', '102', '113'], [1/4]*4)

# Compute bivariate PID with two source variables and one target
result = PID_WB(d)  # or PID_BROJA(d)
```

**Node key format in the bivariate lattice** (tuples of tuples) [3, 7]:
- `{0:1}` = `((0, 1),)` = **Synergy** (joint-only info)
- `{0}` = `((0,),)` = Unique info from X_0
- `{1}` = `((1,),)` = Unique info from X_1
- `{0}{1}` = `((0,), (1,))` = Redundancy (shared info)

**Programmatic synergy extraction** [7]:

```python
# Method 1: Dictionary-style access via __getitem__
synergy = result[((0, 1),)]  # calls get_pi internally

# Method 2: Explicit get_pi method
synergy = result.get_pi(((0, 1),))

# Method 3: Get redundancy
redundancy = result.get_red(((0,), (1,)))
```

The `get_pi` method performs Moebius inversion: `pi(node) = red(node) - sum(pi(descendant) for descendant in lattice.descendants(node))` [7].

**Constructing distributions from empirical data** [8]:

```python
from collections import Counter
import dit
from dit.pid import PID_BROJA

def compute_pairwise_synergy(xi_discrete, xj_discrete, y_discrete):
    triples = list(zip(xi_discrete, xj_discrete, y_discrete))
    counts = Counter(triples)
    total = len(triples)
    outcomes = [str(a) + str(b) + str(c) for (a, b, c) in counts.keys()]
    pmf = [v / total for v in counts.values()]
    d = dit.Distribution(outcomes, pmf)
    result = PID_BROJA(d)
    return result.get_pi(((0, 1),))  # synergy value in bits
```

### 1.3 PID Measure Choice: PID_WB vs PID_BROJA

**PID_WB (Williams & Beer I_min):** Known to over-attribute information as redundant in some cases. For the concatenation distribution, I_min incorrectly assigns 1 bit redundancy and 1 bit synergy instead of 2 bits unique information [3].

**PID_BROJA (Bertschinger-Rauh-Olbrich-Jost-Ay):** Defines unique information via constrained optimization over distributions with fixed input-output marginals. Generally considered more principled for bivariate cases, but can produce unintuitive results on some distributions (e.g., reduced or) [3, 9].

**Recommendation:** Use PID_BROJA as primary (more principled synergy estimates) with PID_WB as sensitivity analysis. Both are available in dit via `from dit.pid import PID_BROJA, PID_WB` [3].

### 1.4 Validation Distributions

- **XOR distribution** (`bivariates['synergy']`): Should yield synergy ~= 1.0 bit, redundancy ~= 0.0 [3]
- **Redundant distribution** (`bivariates['redundant']`): Should yield synergy ~= 0.0, redundancy ~= 1.0 [3]
- **Unique information** (`bivariates['cat']`): Should yield unique_0 ~= 1.0, unique_1 ~= 1.0 [3]

## 2. Discretization Pipeline

### 2.1 sklearn KBinsDiscretizer

```python
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_discrete = discretizer.fit_transform(X).astype(int)
```

**Bin count analysis** [10]:
- B=5: 5x5xn_classes = 50-125 joint states per pair -> fast (~0.01-0.05s per pair)
- B=8: 8x8xn_classes = 128-512 joint states -> moderate (~0.05-0.2s per pair)
- B=10: 10x10xn_classes = 200-1000 joint states -> upper limit (~0.1-0.5s per pair)

### 2.2 Computational Budget for Pairwise Synergy

Number of pairs = d(d-1)/2:
- d=10 -> 45 pairs x ~0.05s = ~2s
- d=30 -> 435 pairs x ~0.05s = ~22s
- d=50 -> 1,225 pairs x ~0.05s = ~61s
- d=100 -> 4,950 pairs x ~0.05s = ~4 min

These estimates assume B=5, binary classification. PID_BROJA may be ~2-5x slower than PID_WB due to optimization [3].

## 3. FIGS Source Code Analysis and Modification Points

### 3.1 Original FIGS Node Class (imodels/tree/figs.py) [11]

```python
class Node:
    def __init__(self, feature: int = None, threshold: int = None,
                 value=None, value_sklearn=None, idxs=None,
                 is_root: bool = False, left=None,
                 impurity: float = None, impurity_reduction: float = None,
                 tree_num: int = None, node_id: int = None,
                 right=None, depth=None):
        self.feature = feature          # Single feature index
        self.threshold = threshold      # Single threshold value
        self.value = value
        self.impurity_reduction = impurity_reduction
        self.left = left; self.right = right
        self.is_root = is_root; self.idxs = idxs
```

### 3.2 Required Node Extension for Oblique Splits

Based on the RO-FIGS implementation [13], the Node class must be extended:

```python
class Node:
    def __init__(self, features=None,     # List of feature indices
                 weights=None,            # Weight vector for linear combination
                 threshold: int = None,   # Threshold for oblique hyperplane
                 value=None, idxs=None, left=None, right=None,
                 impurity: float = None, impurity_reduction: float = None,
                 is_root: bool = False, tree_num: int = None):
        self.features = features    # np.array of feature indices
        self.weights = weights      # np.array of weights
        self.threshold = threshold  # scalar threshold
```

### 3.3 Four Key Modification Points

**Point 1: Replace _construct_node_with_stump with oblique split constructor** [11, 13]

The original method fits `DecisionTreeRegressor(max_depth=1)` for axis-aligned splits. Replace with oblique split using SPyCT ODT:
```python
stump = odt.Model(max_features=beam_size,
                  splitting_features=splitting_features,
                  random_state=self.random_state)
stump.fit(X[idxs], y_regr[idxs])
```

**Point 2: Modify predict to handle oblique nodes** [13]
Original: `x[root.feature] <= root.threshold`
Replace: `np.dot(x[root.features], root.weights) <= root.threshold`

**Point 3: Add synergy pre-computation at start of fit()**
Insert before the main loop:
```python
self.synergy_graph_ = self._build_synergy_graph(X, y)
self.synergy_subsets_ = self._extract_feature_subsets(self.synergy_graph_)
```

**Point 4: Modify feature subset selection in main loop**
Replace `random.sample(range(d), beam_size)` with synergy-guided selection.

### 3.4 FIGS Fit Loop Structure [11]

1. Initialize: Construct first stump on full dataset
2. Main while loop: While potential_splits not empty and not finished:
   a. Pop node with max impurity_reduction
   b. Check stopping conditions (min_impurity_decrease, max_trees, max_depth)
   c. If root: start new tree, add placeholder new root to potential_splits
   d. Add children to potential_splits
   e. Recompute residuals: For each tree, subtract predictions of ALL other trees
   f. Recompute all potential splits: Re-fit stumps on updated residuals
   g. Sort potential_splits by impurity_reduction

## 4. RO-FIGS Algorithm Details

### 4.1 Complete Algorithm (from arxiv 2504.06927) [14]

Algorithm 1: RO-FIGS
Input: X (features), y (outcomes), beam_size, min_imp_dec, max_splits

trees = []
while (max_imp_dec > min_imp_dec OR first_iteration) AND total_splits < max_splits:
    for repetition in range(r=5):
        feat = select_random(beam_size)
        for tree in all_trees:
            y_res = y - predict(all_trees except tree)
            for leaf in tree:
                phi = compute_linear_combination(X, y_res, feat)
                potential_splits.append(define_oblique_split(phi))
        best_split = argmax(impurity_decrease, potential_splits)
        if impurity_decrease(best_split) > min_imp_dec:
            break
    trees.insert(best_split)
Return: trees

### 4.2 SPyCT/ODT Integration [13, 15, 16]

The RO-FIGS ODT module wraps SPyCT's GradSplitter:

```python
model = odt.Model(
    splitter='grad',       # Gradient-based split optimization
    max_depth=1,           # Single split (stump)
    num_trees=1,
    max_features=beam_size,
    max_iter=100,          # Gradient descent iterations
    lr=0.1,                # Learning rate
    C=10,                  # Regularization (strength = 1/C)
    tol=1e-2,
    eps=1e-8,
    adam_beta1=0.9,
    adam_beta2=0.999,
    min_examples_to_split=5,
    min_impurity_decrease=0.05,
    splitting_features=splitting_features
)
```

**How splitting_features works in _grow_tree** [15]:
When splitting_features is provided, the code copies the list, appends a bias column index, converts to numpy array, and sorts. This directly restricts which features the GradSplitter considers during optimization.

### 4.3 L_1/2 Regularization [14]

The split optimization objective is:

min_{w,b} ||w||_{1/2} + C * g(w,b)

where:
- ||w||_{1/2} = (sum_{i=1}^{k} sqrt(|w_i|))^2 -- the L_1/2 quasi-norm
- C -- regularization strength (in SPyCT: regularization = 1/C)
- g(w,b) -- fitness function minimizing weighted variance impurity on both sides of the hyperplane
- k -- number of features (= beam_size)

The L_1/2 norm induces weight sparsity, meaning the actual number of non-zero feature weights per split is typically much less than beam_size [14].

### 4.4 RO-FIGS Node Structure [13]

The RO-FIGS Node stores:
- features: numpy array of feature indices used in the split
- weights: numpy array of learned weights
- threshold: scalar threshold for the hyperplane
- impurity_reduction: impurity reduction from this split
- value: prediction value (for leaves, or aggregate at split)

Prediction for a single point [13]:
```python
if isinstance(root.features, int):
    left = x[root.features] * root.weights <= root.threshold
else:
    projection = sum(x[root.features[i]] * root.weights[i] for i in range(beam_size))
    left = projection <= root.threshold
```

### 4.5 Hyperparameter Grid [14]

RO-FIGS uses grid search over:
- beam_size: {d/2, d} where d = number of features
- min_imp_dec: constant search space (see Table V appendix)

Best configuration trained on joint train+validation data; evaluated on held-out test set. 30 tuning iterations via hyperopt for baselines; grid search for RO-FIGS [14].

### 4.6 odt_info Dictionary [15]

Populated after split construction:
- features: numpy array of feature indices (excluding bias)
- weights: numpy array from splitter.weights_bias (excluding bias)
- threshold: splitter.threshold - splitter.weights_bias[-1]
- n_samples_left: int
- n_samples_right: int
- error: bool (True if split failed)
- one_node_only: bool (True if only root node)

## 5. Synergy Graph Construction Algorithm

### 5.1 Complete Algorithm

```python
import networkx as nx
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter
import dit
from dit.pid import PID_BROJA

def build_synergy_graph(X, y, n_bins=5, threshold_percentile=75):
    n_samples, d = X.shape
    disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X_disc = disc.fit_transform(X).astype(int)
    y_disc = y.astype(int)

    synergy_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            synergy_matrix[i, j] = compute_pairwise_synergy(
                X_disc[:, i], X_disc[:, j], y_disc)
            synergy_matrix[j, i] = synergy_matrix[i, j]

    positive_synergies = synergy_matrix[synergy_matrix > 0]
    tau = np.percentile(positive_synergies, threshold_percentile) if len(positive_synergies) > 0 else 0.0

    G = nx.Graph()
    G.add_nodes_from(range(d))
    for i in range(d):
        for j in range(i+1, d):
            if synergy_matrix[i, j] > tau:
                G.add_edge(i, j, weight=synergy_matrix[i, j])

    feature_subsets = []
    for clique in nx.find_cliques(G):
        if 2 <= len(clique) <= 5:
            feature_subsets.append(sorted(clique))
    for u, v in G.edges():
        pair = sorted([u, v])
        if pair not in feature_subsets:
            feature_subsets.append(pair)

    return G, synergy_matrix, feature_subsets
```

### 5.2 Threshold Strategies

Three approaches:
1. Percentile-based (recommended): tau = 75th percentile of positive synergy values. Adaptive to data distribution.
2. Fixed absolute: tau = 0.01 bits. Simple but not adaptive.
3. Permutation test (expensive): Shuffle target 100 times, compute null synergy distribution, use 95th percentile as threshold. ~100x slower.

### 5.3 Clique Extraction [17]

NetworkX find_cliques(G) returns all maximal cliques using the Bron-Kerbosch algorithm. Filter cliques by size: 2-5 suitable for oblique splits, >5 decompose into overlapping subsets of size 5.

## 6. Complete SG-FIGS Algorithm Design

### 6.1 Class Hierarchy

```python
class SGFIGSClassifier(ROFIGSClassifier):
    def __init__(self, beam_size=None, min_impurity_decrease=0.0,
                 max_splits=75, max_trees=None, num_repetitions=5,
                 n_bins=5, threshold_percentile=75,
                 fallback_to_random=True, random_state=None):
        super().__init__(beam_size=beam_size, ...)
        self.n_bins = n_bins
        self.threshold_percentile = threshold_percentile
        self.fallback_to_random = fallback_to_random
```

### 6.2 Key Algorithmic Difference

The ONLY algorithmic difference between SG-FIGS and RO-FIGS is in feature subset selection:

```python
# RO-FIGS (line 4 of Algorithm 1):
splitting_features = random.sample(range(d), beam_size)

# SG-FIGS:
splitting_features = self._select_synergy_guided_subset()
```

Where _select_synergy_guided_subset: (1) Samples a synergy clique/edge from the pre-computed graph, (2) If clique size < beam_size: pads with random features, (3) If clique size > beam_size: selects top-synergy subset, (4) Fallback: random selection if no synergy subsets available.

### 6.3 Axis-Aligned Fallback

SG-FIGS should maintain axis-aligned splits as fallback by evaluating BOTH oblique (from synergy subset) and axis-aligned (original FIGS stump) splits, then choosing the one with higher impurity_reduction.

## 7. Evaluation Protocol

### 7.1 Datasets [14]

RO-FIGS uses 22 binary classification datasets from OpenML: blood, diabetes, breast-w, ilpd, monks2, climate, kc2, pc1, kc1, heart, tictactoe, wdbc, churn, pc3, biodeg, credit, spambase, credit-g, friedman, usps, bioresponse, speeddating.

Four priority datasets with known domain interactions:
- diabetes (8 features): BMI-glucose, age-insulin
- monks2 (6 features): XOR structure (pure synergy)
- heart (13 features): chest_pain-exercise_angina
- breast-w (9 features): texture-concavity

### 7.2 Metrics [14]

- Primary: Balanced accuracy on test split (mean +/- std across folds)
- Model complexity: Number of splits, number of trees
- Per-split complexity: Average features per split with non-zero weights
- SG-FIGS specific: Fraction of oblique splits using above-median synergy pairs

### 7.3 Statistical Testing [14]

- 10-fold CV with OpenML-provided train/test splits
- 80%/10%/10% train/val/test
- Corrected Friedman test followed by Bonferroni-Dunn post-hoc test (p < 0.05)
- Critical difference diagram for visual comparison

### 7.4 Baselines [14]

From RO-FIGS paper: DT, MT, OT, ODT, RF, ETC, CatBoost, FIGS, Ens-ODT, MLP, RO-FIGS.

For SG-FIGS, critical comparisons: (1) FIGS (axis-aligned), (2) RO-FIGS (random oblique, identical baseline), (3) SG-FIGS (proposed), (4) Ablation: random graph instead of synergy graph.

## 8. RO-FIGS Baseline Implementation

### 8.1 Official Code [13, 15]

Available at https://github.com/um-k/rofigs.

Key dependencies:
- spyct (install: pip install git+https://gitlab.com/TStepi/spyct.git) [16]
- numpy, scipy, scikit-learn, joblib
- Requires C compiler (gcc) for spyct compilation

### 8.2 Critical Implementation Notes

1. Data must be min-max scaled to the zero-one range before passing to RO-FIGS [14]
2. Categorical features must be one-hot encoded (E-ohe performs best) [14]
3. SPyCT adds a bias column automatically in fit() [15]
4. Worst-case complexity: O(i*r*m^2*n^2*d) where i=gradient iterations (100), r=repetitions (5), m=splits, n=samples, d=beam_size features [14]

## 9. Critical Findings and Caveats

### 9.1 PID_WB Over-estimates Synergy

The I_min measure (PID_WB) has a well-documented shortcoming: for distributions where inputs provide orthogonal information about the output, it misclassifies unique information as a mix of redundancy and synergy [3, 9]. PID_BROJA addresses this but is computationally more expensive.

### 9.2 Discretization Sensitivity

PID results are sensitive to the number of bins. Too few bins (B=2-3) loses information; too many bins (B>10) creates sparse contingency tables with unreliable probability estimates. Rule of thumb: ensure n_samples >> n_bins^3 for reliable PID estimates [10].

### 9.3 SPyCT Required for Fair Comparison

RO-FIGS uses SPyCT's gradient-based splitter because it directly optimizes impurity reduction with L_1/2 regularization. Using sklearn's RidgeClassifier as fallback provides L_2 regularization only (no sparsity). For fair comparison, SPyCT must be used [14, 16].

### 9.4 Synergy Graph is One-Time Cost

The synergy graph construction is a one-time cost per fit() call. For d=50 features with B=5 bins, expect ~1-2 minutes. This is amortized over the entire FIGS training loop. The synergy computation does not need to be repeated during the greedy loop.

## Sources

[1] [dit: Python package for discrete information theory (GitHub)](https://github.com/dit/dit) — Official repository for the dit library, confirming it as the primary Python package for information-theoretic computations including PID.

[2] [dit documentation — dit 1.2.3](https://dit.readthedocs.io/en/latest/) — Official documentation homepage for dit, confirming library capabilities and version.

[3] [Partial Information Decomposition — dit 1.2.3 documentation](https://dit.readthedocs.io/en/latest/measures/pid.html) — Core PID documentation with all PID measures (PID_WB, PID_BROJA, PID_CCS, etc.), lattice atom keys ({0:1} for synergy, {0}{1} for redundancy), code examples, and known criticisms of each measure.

[4] [pidpy: Python package for computing partial information decomposition](https://github.com/pietromarchesi/pidpy) — pidpy repository showing binary-only feature support and triplet limitation for non-binary variables, making it unsuitable for discretized continuous features.

[5] [Partial Information Decomposition for Data Interpretability and Feature Selection (AISTATS 2025)](https://arxiv.org/abs/2405.19212) — PIDF paper using neural mutual information estimation (MINE) for PID-based feature selection, relevant as alternative approach but using custom rather than classical PID.

[6] [PIDF Repository (GitHub)](https://github.com/c-s-westphal/PIDF) — Implementation code for PIDF paper, showing custom neural-estimation approach rather than using dit or pidpy libraries.

[7] [dit PID source code — BasePID class](https://raw.githubusercontent.com/dit/dit/master/dit/pid/pid.py) — Complete BasePID implementation showing get_pi() method with Moebius inversion, __getitem__ for dictionary-style access, and tuple-of-tuples node key format.

[8] [Numpy-based Distribution — dit 1.2.3 documentation](https://dit.readthedocs.io/en/latest/distributions/npdist.html) — Documentation for creating dit Distribution objects from numpy arrays via from_ndarray() and from dictionary constructors.

[9] [A Measure of Synergy Based on Union Information (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10969115/) — Academic comparison of PID measures showing PID_WB and PID_BROJA trade-offs, criticisms of each, and that no universal consensus exists on the best measure.

[10] [KBinsDiscretizer — scikit-learn 1.8.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) — Official sklearn documentation for KBinsDiscretizer with quantile strategy for equal-frequency binning of continuous features.

[11] [FIGS source code — imodels repository](https://raw.githubusercontent.com/csinva/imodels/master/imodels/tree/figs.py) — Complete FIGS implementation with Node class, _construct_node_with_stump method, fit() greedy loop with potential_splits and residual updates, and predict() tree traversal.

[13] [RO-FIGS main source code (rofigs.py)](https://raw.githubusercontent.com/um-k/rofigs/main/src/rofigs.py) — Complete ROFIGS implementation showing Node class with features/weights/threshold, _construct_oblique_split, fit() with num_repetitions loop, and _predict_tree with dot-product oblique prediction.

[14] [RO-FIGS: Efficient and Expressive Tree-Based Ensembles for Tabular Data (IEEE 2025)](https://arxiv.org/html/2504.06927) — Full RO-FIGS paper with Algorithm 1 pseudocode, L_1/2 regularization formula, 22 OpenML datasets, balanced accuracy results, corrected Friedman test, beam_size/min_imp_dec grid search, and computational complexity analysis.

[15] [RO-FIGS ODT module source code (odt.py)](https://raw.githubusercontent.com/um-k/rofigs/main/src/odt.py) — Complete oblique decision tree wrapper around SPyCT's GradSplitter, showing splitting_features restriction, odt_info dict population, and GradSplitter initialization with all hyperparameters.

[16] [SPyCT: Python implementation of Oblique Predictive Clustering Trees (GitHub)](https://github.com/knowledge-technologies/spyct) — Official SPyCT repository with Model class API, splitter options (grad/svm), regularization parameter C, learning rate, and installation instructions.

[17] [find_cliques — NetworkX documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html) — NetworkX maximal clique enumeration using Bron-Kerbosch algorithm, returning iterator over all maximal cliques.

## Follow-up Questions

- What is the empirical impact of different PID measures (PID_WB vs PID_BROJA vs PID_CCS) on the resulting synergy graph topology and downstream SG-FIGS performance — does the measure choice significantly affect which feature subsets are selected?
- How should SG-FIGS handle the cold-start problem where the synergy graph is computed on the full dataset but the FIGS greedy loop operates on residuals — should synergy be recomputed on residuals at each iteration, or is the initial synergy graph sufficient?
- Can the computational cost of pairwise PID be reduced for high-dimensional datasets (d>100) using approximate methods like mutual information pre-screening to prune obviously non-synergistic pairs before running full PID?

---
*Generated by AI Inventor Pipeline*
