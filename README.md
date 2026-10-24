# Deep Learning Lab 1

**Master MBD | FST Tanger | Pr. Elaachak Lotfi**

PyTorch DNN/MLP experiments for regression and multiclass classification.

**Status:** Issues #1 and #2 are complete: Kaggle setup, dataset inspection, AAPL EDA, and chronological split checks. Next: [Issue #3 — Regression baseline](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/3).

## Datasets and tasks

| Part | Dataset | Planned task |
|---|---|---|
| Regression | [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse) | Predict a selected stock's next-day closing price using information available through today |
| Classification | [Predictive maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) | Predict the multiclass failure type |

Record exact filenames, stock/date coverage, targets, features, and class labels in the notebooks. Dataset files are not committed.

## Notebooks

- [Regression](notebooks/01_regression.ipynb)
- [Classification](notebooks/02_classification.ipynb)

Each notebook is intended to run independently from top to bottom. Environment checks and regression EDA have saved outputs; modeling sections remain placeholders.

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

## Regression EDA

AAPL has 1,762 observations spanning 2010–2016, with no missing observed trading dates or zero-volume rows. The next-day prediction setup yields 1,257 training samples (2010–2014), 251 validation samples (2015), and 251 test samples (2016). Two boundary-crossing examples are excluded. Saved assertions verify disjoint partitions and target alignment.

Training-only analysis shows strongly correlated OHLC prices, right-skewed volume, and occasional large daily price changes. Extreme observations are retained. Scaling must be fitted on training data, and the future MLP will be compared with persistence. Correlation alone does not establish forecasting performance.

![Training price and volume](assets/plots/aapl_training_price_volume.png)

Additional plots: [distributions and boxplots](assets/plots/aapl_training_distributions_boxplots.png), [correlations](assets/plots/aapl_training_correlations.png), and [next-day changes](assets/plots/aapl_training_next_day_changes.png).

Split records: [dates and counts](reports/results/aapl_split_summary.csv), [excluded examples](reports/results/aapl_excluded_boundary_rows.csv), and [export provenance](reports/results/README.md).
