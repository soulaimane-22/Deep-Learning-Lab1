# Deep Learning Lab 1

**Master MBD | FST Tanger | Pr. Elaachak Lotfi**

PyTorch DNN/MLP experiments for regression and multiclass classification.

**Status:** Issue #1 setup is complete: both downloaded Kaggle notebooks contain successful environment and dataset checks, and the regression notebook includes the successful synthetic PyTorch/GridSearchCV smoke test. Lab modeling, tuning, and evaluation remain to be implemented. Next: [Issue #2](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/2).

## Datasets and tasks

| Part | Dataset | Planned task |
|---|---|---|
| Regression | [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse) | Predict a selected stock's next-day closing price using information available through today |
| Classification | [Predictive maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) | Predict the multiclass failure type |

Record exact filenames, stock/date coverage, targets, features, and class labels in the notebooks. Dataset files are not committed.

## Notebooks

- [Regression](notebooks/01_regression.ipynb)
- [Classification](notebooks/02_classification.ipynb)

Each notebook is intended to run independently from top to bottom. Environment and dataset checks have saved outputs; later lab sections remain placeholders.

## Setup and execution

1. Import the appropriate notebook into Kaggle and attach its dataset.
2. Use the versions recorded in [requirements.txt](requirements.txt). The exported notebooks currently install skorch and imbalanced-learn without pins; compare their printed versions with this file when rerunning.
3. Configure dataset paths, seeds, and GPU/CPU device selection.
4. After implementation, run each notebook from a fresh session with **Run All**.

Kaggle notebook links and measured runtime: _to add_.

Verified setup: Python 3.13.15, Tesla T4, PyTorch 2.11.0+cu128, seed 42, and automatic CPU fallback. The regression CSV has 851,264 rows and 7 columns; the maintenance CSV has 10,000 rows, 10 columns, and six failure-type classes. These are setup checks, not model evaluation results. Notebook code was executed in Kaggle; local review inspected saved code and outputs only.

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
