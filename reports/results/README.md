# EDA output provenance

The split CSVs were extracted from the saved HTML table outputs of the Kaggle regression notebook. The excluded-row prices retain the displayed six-decimal precision. The four PNGs in `assets/plots/` were extracted from the notebook's embedded image outputs, rather than the separate 160-DPI savefig downloads. No lab computation was rerun locally.

## Baseline outputs

The baseline CSVs, configuration JSON, architecture PNG, and learning-curve PNG were downloaded directly from Kaggle. Local checks verified 100 history rows, 251 validation predictions, metric agreement with the comparison, and no saved notebook errors. No model training was run locally.

## Tuning outputs

Issue #4 outputs were downloaded directly from Kaggle. Local review checked 16 candidates and three folds, selected-score consistency, chronological target boundaries, 251 validation predictions against reported metrics, and the final row of the 50-epoch reporting history. The corrected notebook has no saved errors or target-broadcast warnings. No training was run locally.

## Regularization and final evaluation outputs

Issue #5 exports were downloaded from Kaggle. Local verification recomputed all eight final metric rows from 251 saved test predictions, checked five training histories and 250 test-epoch rows against selected checkpoints, and confirmed that the frozen plan selects batch normalization using validation. Early stopping restores epoch 48 after 50 epochs. No model training was run locally.

## Maintenance preprocessing, baseline, and oversampling

Classification tables and plots were downloaded from Kaggle. The oversampled confusion-matrix PNG was recovered from the notebook's embedded saved image because its separate download was missing. Local review checked notebook syntax and saved errors, 10,000 unique split assignments, both sets of 2,000 validation predictions against confusion matrices and per-class reports, all four comparison rows against the final history records, and 50 evaluation points per run. Recomputed validation macro-F1 values are 0.5007548376 (unresampled) and 0.4827429213 (oversampled). Configurations confirm no test evaluation. No model training was run locally; a fresh-session Kaggle rerun remains part of the final reproducibility review.

## Classification tuning and final test exports

Issues #9–#10 exports were downloaded from Kaggle. Local checks recomputed metrics and per-class reports/confusion matrices from all nine models' 2,000 test predictions and the tuned model's 2,000 validation predictions. Checks also covered 16 search candidates, six fold-support records, the 100-epoch reporting history, and 461 checkpoint records against selected-epoch metrics. All eight new PNGs were present and the notebook had no saved errors. Stale notebook headings were restored or completed without changing experiment code. No model training was run locally. Per-class reports on original training rows remain to be exported for the final Issue #10 checklist item.
