# Lab 1 Roadmap

Planned submission: **October 28, 2026**, based on the project plan; the lab PDF does not specify a deadline. Dates use Africa/Casablanca time.

Finish both end-to-end baselines before extensive tuning. Keep October 28 for final review, push, and submission.

| Date | Work | Completion evidence |
|---|---|---|
| Oct 24 | Setup, task definitions, dataset inspection, start EDA | Both datasets load; targets and splits documented |
| Oct 25 | Finish EDA and both baselines; compare class balancing | Both notebooks execute through validation metrics |
| Oct 26 | Small GridSearchCV runs for both parts; controlled regularization experiments | Saved searches and validation comparisons |
| Oct 27 | Finish comparisons, freeze settings, final test reporting, README/synthesis, fresh-session reruns | Complete reproducible submission |
| Oct 28 | Final review, push, submit | Repository and submission links verified |

| Milestone | Due | Issues |
|---|---|---|
| Both baselines and balancing complete | Oct 25 | #1, #2, #3, #6, #7, #8 |
| Tuning, evaluation, and submission ready | Oct 27 | #4, #5, #9, #10, #11, #12 |

If time becomes tight, reduce grid sizes and optional experiments first. Preserve the required GridSearchCV runs, balancing, multiple regularization comparisons, train/test curves, metrics, README, and synthesis.

## Issue Links

| Plan issue | GitHub issue |
|---|---|
| 1 | [Repo & Environment Setup](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/1) |
| 2 | [Problem Definition & EDA: NYSE](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/2) |
| 3 | [Baseline DNN/MLP for Regression](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/3) |
| 4 | [Hyperparameter Tuning: Regression](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/4) |
| 5 | [Regularization & Final Regression Evaluation](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/5) |
| 6 | [Preprocessing & EDA: Maintenance Dataset](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/6) |
| 7 | [Data Augmentation / Class Balancing](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/7) |
| 8 | [Baseline DNN/MLP for Classification](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/8) |
| 9 | [Hyperparameter Tuning: Classification](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/9) |
| 10 | [Regularization & Final Classification Metrics](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/10) |
| 11 | [README Report & Reproducibility Check](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/11) |
| 12 | [Synthesis Report](https://github.com/soulaimane-22/Deep-Learning-Lab1/issues/12) |

## Experimental Setup

### Part 1: Regression

- Proposed task: predict one selected stock's next trading-day closing price using information available through today's close. Document the stock, dataset file, date range, features, and limitations of the single-stock scope.
- Sort by date before creating the next-day target and lagged features. Exclude rows without a valid target and never include future information in inputs.
- Reserve the latest period as the final test set. Within development data, use chronological training/validation partitions and time-aware cross-validation. Record dates and ensure training targets do not cross partition boundaries.
- Use `TimeSeriesSplit` for GridSearchCV, with a gap appropriate to the forecasting horizon (for example, `gap=1` for a one-step target).
- Include a persistence baseline: tomorrow's predicted close equals today's close.
- Report MAE, RMSE, and R²; calculate price errors in original units. The lab requests an accuracy curve, but regression has no standard classification accuracy: use a clearly labeled R²/epoch curve and explain the substitution.

### Part 2: Multiclass classification

- Inspect the schema and explicitly identify the failure-type target, its exact column name, classes, and label mapping. Do not accidentally substitute a binary failure target.
- Remove identifiers such as UDI and Product ID if present, plus other label/outcome columns that reveal failure status or type.
- Use stratified development/test and training/validation splits. Check that minority-class counts support the split proportions and CV fold count.
- One-hot encode nominal input categories and encode target labels as integer class indices. Fit numerical scaling on training data only.
- Start balancing with random oversampling. SMOTE is optional: mixed numerical/categorical inputs require deliberate handling, such as SMOTENC, rather than interpolation of category codes.
- Select models primarily by validation/CV macro-F1. Report accuracy, macro/per-class recall (sensitivity), macro/per-class F1, class support, and confusion matrices.

### Shared evaluation and reproducibility rules

1. Reserve the final test set before model development. Use development-data EDA to guide modeling choices.
2. Fit scalers, imputers, and encoders only on training partitions. During CV, fit them separately within each training fold using a pipeline.
3. Oversample only training data, inside an imbalanced-learn pipeline during CV. Keep validation and test data at their original distributions.
4. Use sklearn `GridSearchCV` with skorch wrappers for PyTorch models. A manual search loop does not replace the explicit lab requirement.
5. Start with small grids covering learning rate, optimizer, epochs, and architecture. Record configurations × folds before running; use `n_jobs=1` when sharing one GPU.
6. Use validation/CV scores for model choices and early stopping, never test scores.
7. For the required train/test epoch curves, freeze configurations and selection rules first. Then evaluate saved epoch checkpoints on test data for reporting without changing the model based on these curves. If checkpoint storage is impractical, use and document a fixed reporting rerun.
8. Compute evaluation curves and metrics in evaluation mode without gradients. Classification training metrics should use the original, unresampled training partition.
9. Compare regularization variants with the same splits, architecture, preprocessing, balancing policy, and training budget. Initially change one technique at a time.
10. Record seeds, Python/package versions, device, dataset paths, split definitions, and selected settings. Seeds improve repeatability but do not guarantee identical GPU results.
