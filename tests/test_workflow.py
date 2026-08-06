from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from acrin_survival.audit import LeakageError
from acrin_survival.cli import main
from acrin_survival.config import ConfigurationError, load_config
from acrin_survival.metrics import dynamic_auc_table, uno_cindex
from acrin_survival.modeling import load_bundle
from acrin_survival.synthetic import write_synthetic_data
from acrin_survival.workflows import build_models, evaluate_predictions, predict_models


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fast_config(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "configs" / "demo.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["training"]["hyperparameter_grid"]["n_estimators"] = [10]
    config["training"]["hyperparameter_grid"]["min_samples_leaf"] = [3]
    config["validation"]["bootstrap_repeats"] = 5
    target = tmp_path / "test_config.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def test_strict_end_to_end_and_frozen_artifacts(tmp_path: Path) -> None:
    development, validation = write_synthetic_data(
        tmp_path / "data", n_development=72, n_validation=36, seed=20260629
    )
    config = _fast_config(tmp_path)
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    build_models(development, config, artifacts)

    model_paths = {
        name: artifacts / "models" / name / "model.joblib"
        for name in ("CP", "CPR", "CPRD", "RD")
    }
    hashes_before = {name: _sha256(path) for name, path in model_paths.items()}
    predictions = predict_models(validation, artifacts, config, tmp_path / "predictions")
    prediction_manifest = predictions.parent / "prediction_manifest.json"
    prediction_manifest_hash = _sha256(prediction_manifest)
    evaluate_predictions(
        predictions,
        validation,
        artifacts,
        config,
        results,
        "synthetic_heldout",
        expected_prediction_manifest_sha256=prediction_manifest_hash,
    )
    hashes_after = {name: _sha256(path) for name, path in model_paths.items()}
    assert hashes_after == hashes_before

    metrics = pd.read_csv(results / "table" / "model_metrics.csv")
    assert metrics["model_name"].tolist() == ["CP", "CPR", "CPRD", "RD"]
    assert metrics["n_used"].eq(36).all()
    assert (results / "table" / "analysis_manifest.json").is_file()
    assert (results / "fig" / "km_risk_groups.pdf").is_file()

    original_predictions = predictions.read_bytes()
    tampered = pd.read_csv(predictions)
    tampered.loc[0, "risk_score"] = tampered.loc[0, "risk_score"] + 1.0
    tampered.to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="Prediction-file hash"):
        evaluate_predictions(
            predictions,
            validation,
            artifacts,
            config,
            tmp_path / "tampered_results",
            "synthetic_heldout",
            expected_prediction_manifest_sha256=prediction_manifest_hash,
        )
    predictions.write_bytes(original_predictions)

    original_manifest = prediction_manifest.read_bytes()
    rewritten_predictions = pd.read_csv(predictions)
    rewritten_predictions.loc[0, "risk_score"] += 2.0
    rewritten_predictions.to_csv(predictions, index=False)
    rewritten_manifest = json.loads(original_manifest)
    rewritten_manifest["predictions_file_sha256"] = _sha256(predictions)
    prediction_manifest.write_text(
        json.dumps(rewritten_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="trusted pre-outcome anchor"):
        evaluate_predictions(
            predictions,
            validation,
            artifacts,
            config,
            tmp_path / "rewritten_predictions_results",
            "synthetic_heldout",
            expected_prediction_manifest_sha256=prediction_manifest_hash,
        )
    predictions.write_bytes(original_predictions)
    prediction_manifest.write_bytes(original_manifest)

    build_manifest = artifacts / "table" / "analysis_manifest.json"
    original_build_manifest = build_manifest.read_bytes()
    rewritten_build = json.loads(original_build_manifest)
    rewritten_build["development_n"] += 1
    build_manifest.write_text(
        json.dumps(rewritten_build, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="build-manifest hash mismatch"):
        evaluate_predictions(
            predictions,
            validation,
            artifacts,
            config,
            tmp_path / "rewritten_build_results",
            "synthetic_heldout",
            expected_prediction_manifest_sha256=prediction_manifest_hash,
        )
    build_manifest.write_bytes(original_build_manifest)

    cp_path = model_paths["CP"]
    original_bundle = cp_path.read_bytes()
    bundle = load_bundle(cp_path)
    bundle.training_risk_cutoff += 1.0
    joblib.dump(bundle, cp_path, compress=3)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        predict_models(validation, artifacts, config, tmp_path / "tampered_artifact")
    cp_path.write_bytes(original_bundle)

    wrapper_results = tmp_path / "wrapper_results"
    assert (
        main(
            [
                "validate",
                "--validation-csv",
                str(validation),
                "--artifacts-dir",
                str(artifacts),
                "--config",
                str(config),
                "--output-dir",
                str(wrapper_results),
                "--cohort-name",
                "synthetic_wrapper",
            ]
        )
        == 0
    )
    assert not list(wrapper_results.rglob("predictions.csv"))


def test_predictions_do_not_depend_on_outcome_values(tmp_path: Path) -> None:
    development, validation = write_synthetic_data(
        tmp_path / "data", n_development=72, n_validation=36, seed=22
    )
    config = _fast_config(tmp_path)
    artifacts = tmp_path / "artifacts"
    build_models(development, config, artifacts)
    first = predict_models(validation, artifacts, config, tmp_path / "first")

    changed = pd.read_csv(validation)
    changed["overall_survival_days"] = changed["overall_survival_days"] + 5000
    changed["death_event_overall"] = 1 - changed["death_event_overall"]
    changed_path = tmp_path / "changed_outcomes.csv"
    changed.to_csv(changed_path, index=False)
    second = predict_models(changed_path, artifacts, config, tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()


def test_prediction_rejects_patient_overlap(tmp_path: Path) -> None:
    development, validation = write_synthetic_data(
        tmp_path / "data", n_development=72, n_validation=36, seed=33
    )
    config = _fast_config(tmp_path)
    artifacts = tmp_path / "artifacts"
    build_models(development, config, artifacts)

    development_frame = pd.read_csv(development)
    validation_frame = pd.read_csv(validation)
    validation_frame.loc[0, "patient_id"] = development_frame.loc[0, "patient_id"]
    overlap_path = tmp_path / "overlap.csv"
    validation_frame.to_csv(overlap_path, index=False)
    with pytest.raises(LeakageError):
        predict_models(overlap_path, artifacts, config, tmp_path / "predictions")


def test_prediction_and_evaluation_functions_contain_no_fit_calls() -> None:
    for function in (predict_models, evaluate_predictions):
        source = inspect.getsource(function)
        assert ".fit(" not in source
        assert ".fit_transform(" not in source


def test_missing_horizon_auc_is_rejected(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "demo.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["validation"]["auc_times_days"] = [365, 730]
    target = tmp_path / "missing_horizon.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must include"):
        load_config(target)


def test_brier_interval_must_end_at_locked_horizon(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "demo.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["validation"]["brier_end_days"] = 730
    target = tmp_path / "short_brier_interval.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must equal horizon_days"):
        load_config(target)


def test_model_configuration_must_not_be_empty(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "demo.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["models"] = []
    target = tmp_path / "empty_models.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="non-empty list"):
        load_config(target)


def test_insufficient_followup_does_not_relabel_36_month_auc() -> None:
    development = pd.DataFrame(
        {
            "time_days": [200.0, 500.0, 1095.0],
            "metric_time_days": [200.0, 500.0, np.nextafter(1095.0, np.inf)],
            "event": [1, 1, 0],
        }
    )
    validation = pd.DataFrame(
        {
            "time_days": [100.0, 400.0, 800.0],
            "metric_time_days": [100.0, 400.0, 800.0],
            "event": [1, 1, 0],
        }
    )
    from acrin_survival.data import make_survival_outcome

    table = dynamic_auc_table(
        make_survival_outcome(development),
        validation,
        np.array([0.9, 0.5, 0.1]),
        [1095],
        repeats=0,
        seed=1,
    )
    assert table.loc[0, "status"] == "insufficient_followup"
    assert pd.isna(table.loc[0, "evaluation_time_days"])
    assert pd.isna(table.loc[0, "cumulative_dynamic_auc"])


def test_uno_cindex_includes_events_on_exact_horizon() -> None:
    horizon = 1095.0
    horizon_control_time = np.nextafter(horizon, np.inf)
    development = np.array(
        [
            (True, 100.0),
            (True, horizon),
            (False, horizon_control_time),
            (False, horizon_control_time),
        ],
        dtype=[("event", "?"), ("time", "<f8")],
    )
    validation = pd.DataFrame(
        {
            "time_days": [200.0, horizon, horizon, horizon],
            "metric_time_days": [200.0, horizon, horizon_control_time, horizon_control_time],
            "event": [1, 1, 0, 0],
        }
    )
    value, tau, status = uno_cindex(
        development, validation, np.array([4.0, 1.0, 3.0, 2.0]), horizon
    )
    assert status == "ok"
    assert tau == horizon_control_time
    assert value == pytest.approx(0.6)

    short = validation.copy()
    short["time_days"] = [200.0, 400.0, 600.0, 800.0]
    short["metric_time_days"] = short["time_days"]
    missing, missing_tau, missing_status = uno_cindex(
        development, short, np.array([4.0, 1.0, 3.0, 2.0]), horizon
    )
    assert pd.isna(missing)
    assert pd.isna(missing_tau)
    assert missing_status == "insufficient_followup"


def test_single_model_configuration_predicts_that_model_by_default(tmp_path: Path) -> None:
    development, validation = write_synthetic_data(
        tmp_path / "data", n_development=72, n_validation=36, seed=44
    )
    config_path = _fast_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"] = ["RD"]
    single_config = tmp_path / "single.yaml"
    single_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    build_models(development, single_config, artifacts)
    predictions = predict_models(
        validation, artifacts, single_config, tmp_path / "predictions"
    )
    assert pd.read_csv(predictions)["model_name"].unique().tolist() == ["RD"]
    single_results = tmp_path / "single_results"
    evaluate_predictions(
        predictions,
        validation,
        artifacts,
        single_config,
        single_results,
        "synthetic_rd_only",
        expected_prediction_manifest_sha256=_sha256(
            predictions.parent / "prediction_manifest.json"
        ),
    )
    comparisons = pd.read_csv(single_results / "table" / "model_comparisons.csv")
    assert comparisons.empty
    assert "reference_model" in comparisons.columns


def test_cp_only_build_writes_readable_empty_feature_rankings(tmp_path: Path) -> None:
    development, _ = write_synthetic_data(
        tmp_path / "data", n_development=72, n_validation=36, seed=55
    )
    config_path = _fast_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"] = ["CP"]
    cp_config = tmp_path / "cp_only.yaml"
    cp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    artifacts = tmp_path / "cp_artifacts"
    build_models(development, cp_config, artifacts)
    rankings = pd.read_csv(artifacts / "table" / "feature_rankings.csv")
    assert rankings.empty
    assert "feature" in rankings.columns
