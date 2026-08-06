# Strict held-out validation policy

A result may be labeled **strict held-out** by this software only when all of the
following are true:

1. The validation cohort was fixed before model fitting.
2. No validation patient, scan, ROI, patch, augmentation, or duplicate image is
   present in development data.
3. Validation outcomes were not used to select predictors, CT region, encoder,
   algorithm, hyperparameters, probability calibration, cutoff, subgroup, time
   point, figure, or preferred result.
4. Feature selection, imputation, scaling, tuning, and threshold selection used
   development data only.
5. The frozen model is applied without fitting, recalibration, threshold
   optimization, risk-direction reversal, or model selection in validation.
6. Every evaluated model uses the same prespecified patient set when models are
   compared.
7. Censoring-aware metrics and uncertainty intervals are reported, together
   with sample size, event count, early-censoring count, and missingness.
8. Input, configuration, code, environment, and artifact hashes are archived;
   the prediction-manifest SHA-256 is recorded before validation outcomes are
   opened and is supplied as the evaluator's external trust anchor.

The program checks the rules that can be verified computationally. It cannot
prove that investigators never inspected a cohort previously. A dataset used
during earlier model or reporting decisions is not restored to pristine
confirmatory status merely by excluding its rows from a later fit. Such results
should be described as exploratory, internal, or transportability analyses as
appropriate.

## External cohorts

An institution or archive name alone does not establish confirmatory external
validation. Differences in treatment, CT acquisition time, endpoint definition,
follow-up, event prevalence, or patient selection must be reported. For an
imaging-only cohort, use the frozen RD model (`--models RD`) and do not fill
missing clinical/PET variables from other patients.

## Hard failures

Prediction stops for duplicate IDs, any development/validation ID overlap,
available image-fingerprint overlap, absent frozen features, an entirely missing
frozen validation feature, excessive prespecified feature missingness,
non-finite transformed values, configuration/artifact hash mismatch, or a
modified risk cutoff. Evaluation stops when prediction and outcome patient sets
differ, prediction-manifest/file hashes fail, survival probabilities increase
over time, or risk groups do not match the frozen cutoff.

## Confidential artifacts

The local model bundle stores development IDs and outcomes to enforce overlap
checks and supply the IPCW censoring reference. It is not a de-identified public
artifact and must not be committed or attached to a release. If public model
weights are later required, replace this design with an approved privacy-safe
audit service and complete a separate disclosure review.
