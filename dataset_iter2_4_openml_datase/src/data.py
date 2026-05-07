# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0.0",
#     "scikit-learn>=1.3.0",
#     "numpy>=1.26.0,<2.3.0",
# ]
# ///
"""
Build full_data_out.json from 4 OpenML binary classification datasets
selected for evaluating SG-FIGS (Synergy-Guided FIGS).

Datasets (from RO-FIGS paper, arxiv 2504.06927):
  1. monks2 (OpenML 334): 601 instances, 6 nominal features, XOR synergy benchmark
  2. blood  (OpenML 1464): 748 instances, 4 numeric features, blood transfusion
  3. climate (OpenML 1467): 540 instances, 20 numeric features, simulation crashes
  4. kc2    (OpenML 1063): 522 instances, 21 numeric features, software defects

Output conforms to exp_sel_data_out.json schema:
  { "datasets": [ { "dataset": "name", "examples": [...] } ] }
Each example:
  { "input": "<json-string>", "output": "<label>",
    "metadata_fold": int, "metadata_feature_names": [...],
    "metadata_task_type": "classification", "metadata_n_classes": 2,
    "metadata_row_index": int, "metadata_n_features": int,
    "metadata_domain": "..." }

Also performs monks2 XOR synergy verification via mutual information analysis.
"""
import json
import resource
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold

# ── Resource limits ──────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

# ── Paths ────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"

# ── Dataset configuration (4 selected datasets) ─────────────────────
DATASETS_CONFIG = [
    {
        "csv": "monks2.csv",
        "name": "monks2",
        "domain": "artificial_XOR_benchmark",
        "openml_id": 334,
        "description": "MONK's Problem 2: exactly-two-of-six categorical XOR synergy benchmark",
    },
    {
        "csv": "blood.csv",
        "name": "blood",
        "domain": "healthcare_blood_transfusion",
        "openml_id": 1464,
        "description": "Blood Transfusion Service Center: predict blood donation",
    },
    {
        "csv": "climate.csv",
        "name": "climate",
        "domain": "climate_science",
        "openml_id": 1467,
        "description": "Climate Model Simulation Crashes: predict simulation outcome",
    },
    {
        "csv": "kc2.csv",
        "name": "kc2",
        "domain": "software_engineering",
        "openml_id": 1063,
        "description": "KC2 NASA software defect prediction",
    },
]


def verify_monks2_xor_synergy(csv_path: Path) -> None:
    """Verify XOR interaction structure in monks2 via mutual information analysis.

    The monks2 target is: EXACTLY TWO of {attr_i == 1} are true.
    Individual features should have low MI with target, but feature pairs
    should show higher MI, confirming XOR-like interaction structure.
    """
    print("\n  ── monks2 XOR Synergy Verification ──")
    df = pd.read_csv(csv_path)
    feature_names = [f"attr{i}" for i in range(1, 7)]
    X = df[feature_names].values.astype(int)
    y = df["target"].values.astype(int)

    # Individual feature MI
    print("  Individual feature MI with target:")
    individual_mi = mutual_info_classif(
        X, y, discrete_features=True, random_state=42
    )
    for fname, mi_val in zip(feature_names, individual_mi):
        print(f"    {fname}: MI = {mi_val:.4f}")
    avg_individual = np.mean(individual_mi)

    # Pairwise feature MI (create joint categorical features)
    print("\n  Pairwise feature MI (joint categories) with target:")
    pair_mis = []
    for i, j in combinations(range(6), 2):
        # Create joint categorical feature by combining the two features
        joint = X[:, i] * 10 + X[:, j]  # unique combo encoding
        joint_mi = mutual_info_classif(
            joint.reshape(-1, 1), y, discrete_features=True, random_state=42
        )[0]
        pair_mis.append(joint_mi)
        fname_i, fname_j = feature_names[i], feature_names[j]
        print(f"    ({fname_i}, {fname_j}): MI = {joint_mi:.4f}")

    avg_pair = np.mean(pair_mis)
    max_pair = np.max(pair_mis)
    print(f"\n  Summary:")
    print(f"    Avg individual MI: {avg_individual:.4f}")
    print(f"    Avg pairwise MI:   {avg_pair:.4f}")
    print(f"    Max pairwise MI:   {max_pair:.4f}")
    if avg_pair > avg_individual:
        print("    ✓ Pairwise MI > Individual MI → confirms XOR interaction structure")
    else:
        print("    ⚠ Unexpected: pairwise MI not higher than individual MI")
    print("  ── End XOR Verification ──\n")


def load_and_process_dataset(config: dict) -> dict:
    """Load a CSV dataset, assign folds, and build schema-compliant examples."""
    csv_path = DATA_DIR / config["csv"]
    meta_path = DATA_DIR / f"{config['name']}_meta.json"

    # Load data
    df = pd.read_csv(csv_path)
    with open(meta_path) as f:
        meta = json.load(f)

    feature_names: list[str] = meta["feature_names"]
    n_classes: int = meta["n_classes"]
    n_features: int = meta["n_features"]

    # Separate features and target
    X = df[feature_names].copy()
    y = df["target"].astype(str)

    # Handle missing values
    for col in X.columns:
        if X[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"  [{config['name']}] Imputed NaN in numeric column '{col}' with median={median_val}")
        else:
            if X[col].isna().any():
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                print(f"  [{config['name']}] Imputed NaN in categorical column '{col}' with mode={mode_val}")

    # Assign 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_assignments = np.zeros(len(X), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(kf.split(X)):
        fold_assignments[val_idx] = fold_idx

    # Run monks2 synergy verification if applicable
    if config["name"] == "monks2":
        verify_monks2_xor_synergy(csv_path)

    # Build examples
    examples = []
    for i in range(len(X)):
        row_dict = {}
        for col in feature_names:
            val = X.iloc[i][col]
            # Convert numpy types to native Python types for JSON
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, (np.bool_,)):
                val = bool(val)
            row_dict[col] = val

        example = {
            "input": json.dumps(row_dict),
            "output": str(y.iloc[i]),
            "metadata_fold": int(fold_assignments[i]),
            "metadata_feature_names": feature_names,
            "metadata_task_type": "classification",
            "metadata_n_classes": n_classes,
            "metadata_row_index": i,
            "metadata_n_features": n_features,
            "metadata_domain": config["domain"],
        }
        examples.append(example)

    print(f"  [{config['name']}] {len(examples)} examples, {n_features} features, {n_classes} classes")
    print(f"    Class distribution: {dict(y.value_counts())}")
    return {"dataset": config["name"], "examples": examples}


def main() -> None:
    """Process 4 selected datasets and write full_data_out.json."""
    print("=" * 60)
    print("Building full_data_out.json from 4 OpenML datasets")
    print("  monks2 (334), blood (1464), climate (1467), kc2 (1063)")
    print("=" * 60)

    datasets_out = []
    for config in DATASETS_CONFIG:
        print(f"\nProcessing: {config['name']} (OpenML {config['openml_id']})")
        ds = load_and_process_dataset(config)
        datasets_out.append(ds)

    output = {"datasets": datasets_out}

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    file_size = OUTPUT_PATH.stat().st_size
    total_examples = sum(len(d["examples"]) for d in datasets_out)
    print(f"\n{'=' * 60}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"  Total datasets: {len(datasets_out)}")
    print(f"  Total examples: {total_examples}")
    print(f"  File size: {file_size / 1024:.1f} KB")
    print("  Datasets included:")
    for ds in datasets_out:
        print(f"    - {ds['dataset']}: {len(ds['examples'])} examples")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
