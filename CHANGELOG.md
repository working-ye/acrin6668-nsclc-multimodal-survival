# Changelog

All notable changes to the public workflow are documented here.

## [0.1.0] - 2026-08-06

- Added strict development-only model construction for CP, CPR, CPRD, and RD.
- Added fold-contained feature selection, imputation, scaling, and RSF tuning.
- Added explicit patient and optional image-fingerprint overlap guards.
- Separated outcome-blind prediction from censoring-aware evaluation.
- Added Harrell/Uno C-index, IPCW time-dependent AUC, Brier/IBS, calibration,
  frozen-cutoff KM analysis, paired bootstrap comparisons, and SCI-style figures.
- Added a pre-outcome prediction-manifest trust anchor and complete build/model/
  prediction/configuration hash verification.
- Added exact day-1,095 event handling, explicit metric status fields, fixed-schema
  empty outputs, and complete CP-only/RD-only workflow tests.
- Added synthetic end-to-end tests, environment locks, data-governance notes,
  model card, citation metadata, and continuous integration.
