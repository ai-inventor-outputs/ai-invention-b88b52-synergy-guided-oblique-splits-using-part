# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.3.0",
#     "numpy>=1.26.0,<2.3.0",
# ]
# ///
"""
data.py - Load 4 sklearn built-in datasets and convert to exp_sel_data_out schema.

Loads 4 sklearn datasets for PID synergy analysis:
  1. breast_cancer: 569 × 30 features (binary, medical domain, timing scalability)
  2. wine: 178 × 13 features (3-class, chemical domain, interpretable interactions)
  3. iris: 150 × 4 features (3-class, classic benchmark, baseline)
  4. diabetes_binarized: 442 × 10 features (binary, medical domain, binarized regression)

Each sample → JSON example with:
  - input: JSON string of feature name→value pairs
  - output: target label as string
  - metadata_* fields: fold, feature_names, task_type, n_classes, row_index, n_features, n_samples

Output format:
{
  "datasets": [
    {"dataset": "iris", "examples": [{"input": "...", "output": "...", ...}, ...]},
    ...
  ]
}
"""
import json
import resource
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.model_selection import KFold

# Resource limits: 14GB RAM, 1 hour CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

WORKSPACE = Path(__file__).parent
OUTPUT_PATH = WORKSPACE / "full_data_out.json"


def make_fold_assignments(n_samples: int, n_folds: int = 5, random_state: int = 42) -> list[int]:
    """Assign each sample to a fold using KFold."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = [0] * n_samples
    for fold_idx, (_, test_idx) in enumerate(kf.split(range(n_samples))):
        for idx in test_idx:
            folds[idx] = fold_idx
    return folds


def process_dataset(
    name: str,
    data: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    target_names: list[str] | None,
    task_type: str,
    n_classes: int,
) -> dict:
    """Convert a sklearn dataset to schema-compliant format."""
    n_samples, n_features = data.shape
    folds = make_fold_assignments(n_samples)

    examples = []
    for i in range(n_samples):
        # Build input as JSON string of feature name→value pairs
        feature_dict = {}
        for j, fname in enumerate(feature_names):
            val = float(data[i, j])
            # Round to avoid excessive decimal places
            feature_dict[fname] = round(val, 6)

        input_str = json.dumps(feature_dict)

        # Build output as string label
        if target_names is not None and task_type == "classification":
            output_str = str(target_names[int(target[i])])
        else:
            output_str = str(target[i])

        example = {
            "input": input_str,
            "output": output_str,
            "metadata_fold": folds[i],
            "metadata_feature_names": feature_names,
            "metadata_task_type": task_type,
            "metadata_n_classes": n_classes,
            "metadata_row_index": i,
            "metadata_n_features": n_features,
            "metadata_n_samples": n_samples,
        }
        examples.append(example)

    return {"dataset": name, "examples": examples}


def load_all_datasets() -> list[dict]:
    """Load 4 sklearn datasets for PID synergy analysis."""
    results = []

    # 1. Breast Cancer Wisconsin (Diagnostic) - 569 × 30, binary
    bc = load_breast_cancer()
    results.append(process_dataset(
        name="breast_cancer",
        data=bc.data,
        target=bc.target,
        feature_names=list(bc.feature_names),
        target_names=["malignant", "benign"],
        task_type="classification",
        n_classes=2,
    ))
    print(f"  breast_cancer: {len(results[-1]['examples'])} examples")

    # 2. Wine - 178 × 13, 3-class
    wine = load_wine()
    results.append(process_dataset(
        name="wine",
        data=wine.data,
        target=wine.target,
        feature_names=list(wine.feature_names),
        target_names=["class_0", "class_1", "class_2"],
        task_type="classification",
        n_classes=3,
    ))
    print(f"  wine: {len(results[-1]['examples'])} examples")

    # 3. Iris - 150 × 4, 3-class
    iris = load_iris()
    results.append(process_dataset(
        name="iris",
        data=iris.data,
        target=iris.target,
        feature_names=list(iris.feature_names),
        target_names=["setosa", "versicolor", "virginica"],
        task_type="classification",
        n_classes=3,
    ))
    print(f"  iris: {len(results[-1]['examples'])} examples")

    # 4. Diabetes (binarized at median) - 442 × 10, binary
    diab = load_diabetes()
    median_target = np.median(diab.target)
    binarized_target = (diab.target > median_target).astype(int)
    results.append(process_dataset(
        name="diabetes_binarized",
        data=diab.data,
        target=binarized_target,
        feature_names=list(diab.feature_names),
        target_names=["below_median", "above_median"],
        task_type="classification",
        n_classes=2,
    ))
    print(f"  diabetes_binarized: {len(results[-1]['examples'])} examples")

    return results


def main() -> None:
    print("Loading 4 sklearn datasets for PID synergy analysis...")
    datasets = load_all_datasets()

    output = {"datasets": datasets}

    total_examples = sum(len(d["examples"]) for d in datasets)
    print(f"\nTotal: {len(datasets)} datasets, {total_examples} examples")

    print(f"Writing to {OUTPUT_PATH}...")
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Done. File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
