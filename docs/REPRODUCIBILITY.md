# Reproducibility and release procedure

## Environment

`environment.yml` and `requirements.txt` record the supported Python 3.9
environment. `pyproject.toml` defines compatible package ranges. Every run
records the versions imported at runtime.

## Determinism

- Cross-validation seed: `20260629`.
- CP seed: `21207110`.
- CPR seed: `21153296`.
- CPRD seed: `20285464`.
- RD seed: `20330496`.
- Validation bootstrap seed: `20260727`.

Parallel tree fitting is deterministic for the locked scikit-survival version
and fixed random seeds, subject to platform and numerical-library differences.

## Archive for a manuscript result

Archive the exact input-table hashes, `configs/paper.yaml`, training and
validation manifests, model registry, feature rankings, selected variables,
tuning results, aggregate tables, figures, the Git commit, and the tagged
release. Keep all patient-level material and fitted bundles in approved secure
storage rather than the public repository.

## Release checklist

1. Run `python -m pytest` and `ruff check src tests scripts`.
2. Run the synthetic smoke test in a clean environment.
3. Confirm `git status` contains no data, artifacts, outputs, credentials, or
   generated bundles.
4. Run a secret scanner and inspect files larger than 5 MB.
5. Confirm CP/CPR/CPRD/RD terminology matches the manuscript.
6. Verify that manuscript numbers come from the tagged release and its recorded
   manifests.
7. Update `CITATION.cff` with the final author list and archive DOI when ready.
