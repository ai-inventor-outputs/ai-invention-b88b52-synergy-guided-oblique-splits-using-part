# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow>=12.0",
#     "numpy>=1.24",
# ]
# ///
"""
Convert 15 tabular classification datasets from parquet to exp_sel_data_out.json schema.

Each row in each dataset becomes a separate example with:
  - input: JSON string of feature values
  - output: target label as string
  - metadata_fold: fold assignment (5-fold CV)
  - metadata_feature_names: list of feature names
  - metadata_task_type: "classification"
  - metadata_n_classes: number of classes
  - metadata_row_index: original row index
  - metadata_n_features: number of features
  - metadata_domain: dataset domain
"""

import json
import resource
from pathlib import Path

import numpy as np
import pandas as pd

# Resource limits: 14GB RAM, 1 hour CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"

# Top 10 datasets chosen for SG-FIGS evaluation:
# - All have named (non-anonymous) features for interpretability
# - Cover 4-60 features, 150-1372 samples, diverse domains
# - Known feature interactions documented for PID analysis
# Excluded: digits (pixel_N names), australian_credit (A1,A2 anonymous),
#   haberman (OHE distorted), german_credit (OHE explosion), glass (small+imbalanced)
DATASET_CONFIGS = [
    "breast_cancer_wisconsin_diagnostic",  # 569 x 30, medical
    "wine",                                 # 178 x 13, food science
    "pima_diabetes",                        # 768 x 8, medical
    "heart_statlog",                        # 270 x 13, medical
    "ionosphere",                           # 351 x 34, signal processing
    "vehicle",                              # 846 x 18, computer vision
    "sonar",                                # 208 x 60, signal processing
    "banknote",                             # 1372 x 4, image processing
    "spectf_heart",                         # 349 x 44, medical
    "iris",                                 # 150 x 4, botany
]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super().default(obj)


def process_dataset(snake_name: str) -> dict:
    """Process a single dataset into the schema format."""
    parquet_path = DATASETS_DIR / f"{snake_name}.parquet"
    meta_path = DATASETS_DIR / f"{snake_name}_meta.json"

    # Load data
    df = pd.read_parquet(parquet_path)
    with open(meta_path) as f:
        meta = json.load(f)

    # Separate features and target
    feature_cols = [c for c in df.columns if c != "target"]
    feature_names = list(feature_cols)
    n_classes = meta["n_classes"]
    domain = meta.get("domain", "unknown")
    n_features = len(feature_cols)

    # Assign 5-fold CV
    n_samples = len(df)
    rng = np.random.RandomState(42)
    fold_assignments = rng.randint(0, 5, size=n_samples).tolist()

    examples = []
    for idx in range(n_samples):
        row = df.iloc[idx]

        # Build input: JSON string of feature name -> value pairs
        feature_dict = {}
        for col in feature_cols:
            val = row[col]
            # Convert numpy types to Python native types
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, (np.bool_,)):
                val = bool(val)
            feature_dict[col] = val

        input_str = json.dumps(feature_dict)

        # Build output: target label as string (integer class label)
        target_val = row["target"]
        if isinstance(target_val, (np.integer,)):
            target_val = int(target_val)
        elif isinstance(target_val, (np.floating, float)):
            target_val = int(target_val)
        output_str = str(target_val)

        example = {
            "input": input_str,
            "output": output_str,
            "metadata_fold": fold_assignments[idx],
            "metadata_feature_names": feature_names,
            "metadata_task_type": "classification",
            "metadata_n_classes": n_classes,
            "metadata_row_index": idx,
            "metadata_n_features": n_features,
            "metadata_domain": domain,
        }
        examples.append(example)

    print(f"  {snake_name}: {len(examples)} examples, {n_features} features, {n_classes} classes")
    return {"dataset": snake_name, "examples": examples}


def main():
    print("=" * 60)
    print("Converting datasets to exp_sel_data_out.json format")
    print("=" * 60)

    datasets_output = []
    total_examples = 0

    for name in DATASET_CONFIGS:
        parquet_path = DATASETS_DIR / f"{name}.parquet"
        if not parquet_path.exists():
            print(f"  SKIP: {name} - parquet file not found")
            continue
        dataset_entry = process_dataset(name)
        datasets_output.append(dataset_entry)
        total_examples += len(dataset_entry["examples"])

    result = {"datasets": datasets_output}

    print(f"\nTotal: {len(datasets_output)} datasets, {total_examples} examples")
    print(f"Writing to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    file_size = OUTPUT_FILE.stat().st_size
    print(f"Output file size: {file_size / 1024 / 1024:.2f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
