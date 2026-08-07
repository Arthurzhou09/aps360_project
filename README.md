# aps360_project — Sequence to Fitness Prediction

Arthur Zhou

## Background

TEM-1 β-lactamase is a bacterial antibiotic-resistance enzyme; predicting how mutations
affect its fitness helps forecast resistance evolution and guide antibiotic design. Deep
mutational scanning (DMS) provides large-scale sequence-to-fitness datasets, but
experimentally measuring all higher-order variants is infeasible. This
project trains graph neural networks on single-mutant DMS data — combining protein
structure, amino-acid properties, and dynamic-coupling features to predict fitness
effects, and evaluates how well that generalizes to unseen double mutants.

## Repo layout

- `src/data/` — data loading, feature engineering, and raw-to-processed pipelines (see `src/data/README.md`).
- `src/model/` — model architectures: GNN building blocks, the main GNN, and the baseline MLP.
- `src/train/` — training/inference entry points:
  - `base/` — baseline MLP.
  - `gnn/` — supervised GNN, plus self-supervised (masked-residue) pretraining.
  - `evolutionary/` — PSSM-based evolutionary baseline.
- `configs/` — JSON configs (hyperparameters, data/logging settings) for training runs.
- `experiments/` — Jupyter notebooks for data exploration and result visualization.
- `manuscript/` — final report and course rubric PDFs.

## Results

Single-mutant test split (structure-position holdout), Spearman ρ:
- Baseline MLP: 0.41
- GNN: 0.63
- Evolutionary PSSM baseline (zero-shot, no training): 0.21

Held-out double mutants (unseen residue positions), GNN: Pearson r 0.70, Spearman ρ 0.73 —
better generalization than on the single-mutant test split.

Self-supervised (masked-residue) GNN pretraining on homolog/MSA data was also explored as
a way to avoid needing fitness labels; see `manuscript/` for why this direction was not
pursued further.
