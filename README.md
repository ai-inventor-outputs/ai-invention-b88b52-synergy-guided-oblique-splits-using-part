# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

## Research Paper

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://github.com/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/paper/paper.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/tree/main/paper) [![Figures](https://img.shields.io/badge/Figures-5-blue)](https://github.com/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/tree/main/figures)

## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
| `dataset_iter1_tabular_classif` | Tabular Classification Benchmarks for SG-FIGS Eval... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/dataset_iter1_tabular_classif/demo/data_code_demo.ipynb) |
| `dataset_iter1_pid_synergy_mat` | PID Synergy Matrices, Timing, MI Comparison & Stab... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/dataset_iter1_pid_synergy_mat/demo/data_code_demo.ipynb) |
| `dataset_iter2_4_openml_datase` | 4 OpenML Datasets for SG-FIGS (monks2, blood, clim... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/dataset_iter2_4_openml_datase/demo/data_code_demo.ipynb) |
| `experiment_iter2_pairwise_pid_sy` | Pairwise PID Synergy Matrices on Benchmark Dataset... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter2_pairwise_pid_sy/demo/method_code_demo.ipynb) |
| `experiment_iter2_sg_figs_full_ex` | SG-FIGS: Full Experiment Implementation and Benchm... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter2_sg_figs_full_ex/demo/method_code_demo.ipynb) |
| `evaluation_iter3_statistical_eva` | Statistical Evaluation of SG-FIGS Experiment Resul... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/evaluation_iter3_statistical_eva/demo/eval_code_demo.ipynb) |
| `experiment_iter3_sg_figs_definit` | SG-FIGS Definitive Comparison Experiment... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter3_sg_figs_definit/demo/method_code_demo.ipynb) |
| `experiment_iter3_synergy_thresho` | Synergy Threshold Sensitivity & Adaptive Threshold... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter3_synergy_thresho/demo/method_code_demo.ipynb) |
| `evaluation_iter4_final_integrate` | Final Integrated Research Synthesis for SG-FIGS... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/evaluation_iter4_final_integrate/demo/eval_code_demo.ipynb) |
| `evaluation_iter4_definitive_stat` | Definitive Statistical Evaluation of SG-FIGS Hypot... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/evaluation_iter4_definitive_stat/demo/eval_code_demo.ipynb) |
| `experiment_iter4_complexity_matc` | Complexity-Matched SG-FIGS Experiment: Synergy vs ... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/experiment_iter4_complexity_matc/demo/method_code_demo.ipynb) |

### Research & Documentation

| Folder | Description | View Research |
|--------|-------------|---------------|
| `research_iter1_sg_figs_spec` | SG-FIGS Spec... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/blob/main/research_iter1_sg_figs_spec/demo/research_demo.md) |

## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper/                       # LaTeX paper and PDF
├── figures/                     # Visualizations
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part.git
cd ai-invention-b88b52-synergy-guided-oblique-splits-using-part

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook exp_001/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
