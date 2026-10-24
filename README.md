# Deep Learning Lab 1

**Master MBD | FST Tanger | Pr. Elaachak Lotfi**

PyTorch DNN/MLP experiments for regression and multiclass classification.

**Status:** Repository scaffold only. Training, tuning, and evaluation remain to be implemented. Start with [Issue #1](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/1).

## Datasets and tasks

| Part | Dataset | Planned task |
|---|---|---|
| Regression | [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse) | Predict a selected stock's next-day closing price using information available through today |
| Classification | [Predictive maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) | Predict the multiclass failure type |

Record exact filenames, stock/date coverage, targets, features, and class labels in the notebooks. Dataset files are not committed.

## Notebooks

- [Regression](notebooks/01_regression.ipynb)
- [Classification](notebooks/02_classification.ipynb)

Each notebook will run independently from top to bottom. The current notebooks contain section placeholders only.

## Setup and execution

1. Import the appropriate notebook into Kaggle and attach its dataset.
2. Install the packages in [requirements.txt](requirements.txt); record and pin tested versions during Issue #1.
3. Configure dataset paths, seeds, and GPU/CPU device selection.
4. After implementation, run each notebook from a fresh session with **Run All**.

Kaggle notebook links and measured runtime: _to add_.

## Evaluation plan

- Regression: chronological partitions, time-aware CV, persistence baseline, MAE/RMSE/R². Use R²/epoch in place of classification accuracy and explain the substitution.
- Classification: stratified partitions, training-only balancing, accuracy, macro/per-class recall and F1, class support, and confusion matrices.
- Fit preprocessing and resampling only within training partitions, including CV folds.
- Use sklearn GridSearchCV with skorch. Select models using validation/CV only.
- Freeze model configurations before producing the required final train/test curves and metrics.
- Compare regularization variants with controlled settings.

## Roadmap

See [the roadmap](ROADMAP.md), [issues](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues), and [milestones](https://github.com/soulaimane-22/Deep-Learning-Lab1/milestones).

## Results

_To complete after experiments. Export small result tables and search configurations to `reports/results/` and plots to `assets/plots/`._

## Synthesis

[Personal synthesis](reports/synthesis.md) — to complete after the experiments.
