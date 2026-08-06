# Contributing

Contributions that improve correctness, reproducibility, tests, documentation,
or portability are welcome.

1. Open an issue describing the scientific or software problem.
2. Create a focused branch and add tests for behavioral changes.
3. Run `python -m pytest` and `ruff check src tests scripts`.
4. Do not include patient data, model artifacts, predictions, credentials,
   institution-specific paths, or unpublished manuscript results.
5. Document any change to outcome handling, feature selection, preprocessing,
   tuning, censoring, or metrics in `CHANGELOG.md`.

Changes that alter statistical estimates require a new locked run, updated
manifest, independent review, and a new tagged release. They must not silently
replace a manuscript-linked version.
