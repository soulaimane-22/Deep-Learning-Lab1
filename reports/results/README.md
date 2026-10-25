# EDA output provenance

The split CSVs were extracted from the saved HTML table outputs of the Kaggle regression notebook. The excluded-row prices retain the displayed six-decimal precision. The four PNGs in `assets/plots/` were extracted from the notebook's embedded image outputs, rather than the separate 160-DPI savefig downloads. No lab computation was rerun locally.

## Baseline outputs

The baseline CSVs, configuration JSON, architecture PNG, and learning-curve PNG were downloaded directly from Kaggle. Local checks verified 100 history rows, 251 validation predictions, metric agreement with the comparison, and no saved notebook errors. No model training was run locally.

## Tuning outputs

Issue #4 outputs were downloaded directly from Kaggle. Local review checked 16 candidates and three folds, selected-score consistency, chronological target boundaries, 251 validation predictions against reported metrics, and the final row of the 50-epoch reporting history. The corrected notebook has no saved errors or target-broadcast warnings. No training was run locally.
