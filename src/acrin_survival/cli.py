"""Command-line interface for model training and TCIA internal validation."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Sequence

from .provenance import sha256_file
from .synthetic import write_synthetic_data
from .workflows import build_models, evaluate_predictions, predict_models


def _models(value: str | None) -> Sequence[str] | None:
    if value is None:
        return None
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acrin-survival",
        description="Train multimodal survival models and run TCIA internal validation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Train frozen models from the training table.")
    build.add_argument("--training-csv", required=True, type=Path)
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)

    predict = subparsers.add_parser(
        "predict", help="Apply frozen models to a TCIA feature table without loading outcomes."
    )
    predict.add_argument("--features-csv", required=True, type=Path)
    predict.add_argument("--artifacts-dir", required=True, type=Path)
    predict.add_argument("--config", required=True, type=Path)
    predict.add_argument("--output-dir", required=True, type=Path)
    predict.add_argument("--models", help="Comma-separated subset, for example RD or CP,CPR.")

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate frozen predictions for TCIA internal validation."
    )
    evaluate.add_argument("--predictions-csv", required=True, type=Path)
    evaluate.add_argument("--outcomes-csv", required=True, type=Path)
    evaluate.add_argument("--artifacts-dir", required=True, type=Path)
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--cohort-name", required=True)
    evaluate.add_argument(
        "--expected-prediction-manifest-sha256",
        required=True,
        help="SHA-256 archived immediately after prediction and before outcomes are opened.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Run prediction and TCIA internal validation in one workflow.",
    )
    validate.add_argument("--validation-csv", required=True, type=Path,
                          help="TCIA internal validation table.")
    validate.add_argument("--artifacts-dir", required=True, type=Path)
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--output-dir", required=True, type=Path)
    validate.add_argument("--cohort-name", required=True)
    validate.add_argument("--models", help="Comma-separated model subset.")

    synthetic = subparsers.add_parser("make-synthetic", help="Create fake data for a smoke test.")
    synthetic.add_argument("--output-dir", required=True, type=Path)
    synthetic.add_argument("--n-training", type=int, default=120)
    synthetic.add_argument("--n-validation", type=int, default=60)
    synthetic.add_argument("--seed", type=int, default=20260629)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        build_models(args.training_csv, args.config, args.output_dir)
    elif args.command == "predict":
        prediction_path = predict_models(
            args.features_csv,
            args.artifacts_dir,
            args.config,
            args.output_dir,
            _models(args.models),
        )
        prediction_manifest = prediction_path.parent / "prediction_manifest.json"
        print(f"prediction_manifest_sha256={sha256_file(prediction_manifest)}")
    elif args.command == "evaluate":
        evaluate_predictions(
            args.predictions_csv,
            args.outcomes_csv,
            args.artifacts_dir,
            args.config,
            args.output_dir,
            args.cohort_name,
            expected_prediction_manifest_sha256=args.expected_prediction_manifest_sha256,
        )
    elif args.command == "validate":
        with tempfile.TemporaryDirectory(prefix="acrin_survival_predictions_") as temporary:
            prediction_path = predict_models(
                args.validation_csv,
                args.artifacts_dir,
                args.config,
                Path(temporary),
                _models(args.models),
            )
            evaluate_predictions(
                prediction_path,
                args.validation_csv,
                args.artifacts_dir,
                args.config,
                args.output_dir,
                args.cohort_name,
                expected_prediction_manifest_sha256=sha256_file(
                    prediction_path.parent / "prediction_manifest.json"
                ),
            )
    elif args.command == "make-synthetic":
        training, validation = write_synthetic_data(
            args.output_dir, args.n_training, args.n_validation, args.seed
        )
        print(training)
        print(validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
