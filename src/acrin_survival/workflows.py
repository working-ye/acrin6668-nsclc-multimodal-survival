"""Publication-facing build, prediction, and evaluation workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .audit import (
    assert_disjoint_identity_values,
    assert_disjoint_ids,
    environment_snapshot,
    git_commit,
    normalize_patient_id,
    patient_id_digest,
    sha256_file,
    write_json,
)
from .config import load_config
from .data import (
    CohortSchema,
    numeric_feature_frame,
    read_cohort,
    read_feature_table,
    read_outcome_table,
    validate_feature_families,
)
from .metrics import (
    bootstrap_harrell_cindex,
    brier_table,
    calibration_table,
    dynamic_auc_table,
    harrell_cindex,
    paired_cindex_comparisons,
    risk_group_summary,
    uno_cindex,
)
from .modeling import (
    FrozenModelBundle,
    load_bundle,
    predict_risk,
    predict_survival_probabilities,
    save_bundle,
    train_model,
)
from .plotting import plot_calibration, plot_km_risk_groups, plot_time_dependent_auc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(no rows)_"
    columns = list(frame.columns)
    header = "| " + " | ".join(map(str, columns)) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepare_clean_output(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use a new directory for each locked run."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_json(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {source}")
    return payload


def _brier_time_grid(config: dict) -> list[int]:
    validation = config["validation"]
    start = int(validation["brier_start_days"])
    end = int(validation["brier_end_days"])
    step = int(validation["brier_step_days"])
    values = list(range(start, end + 1, step))
    if values[-1] != end:
        values.append(end)
    return values


def build_models(
    development_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Build all requested models without accepting a validation-data argument."""

    config = load_config(config_path)
    schema = CohortSchema.from_config(config)
    development_source = Path(development_csv).resolve()
    config_source = Path(config_path).resolve()
    output = _prepare_clean_output(output_dir)
    table_dir = output / "table"
    model_dir = output / "models"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    development = read_cohort(development_source, schema)
    if float(development["metric_time_days"].max()) <= float(schema.horizon_days):
        raise ValueError(
            "Development follow-up does not support the configured horizon: at least one "
            "event-free patient must be observed through the horizon."
        )
    top_k = int(config["feature_selection"]["top_k_per_imaging_modality"])
    if any(model != "CP" for model in config["models"]):
        validate_feature_families(development, schema, top_k)
    development_hash = sha256_file(development_source)
    config_hash = sha256_file(config_source)

    results = {}
    registry_rows = []
    tuning_frames = []
    ranking_frames = []
    input_frames = []
    selected_rows = []
    for model_name in config["models"]:
        result = train_model(
            development,
            schema,
            config,
            model_name,
            development_hash,
            config_hash,
        )
        results[model_name] = result
        bundle_path = model_dir / model_name / "model.joblib"
        save_bundle(result.bundle, bundle_path)
        bundle_hash = sha256_file(bundle_path)
        tuning_frames.append(result.tuning)
        input_frames.append(result.input_summary)
        for ranking in (result.radiomics_ranking, result.deep_ranking):
            if not ranking.empty:
                ranking_frames.append(ranking.assign(model_name=model_name))
        radiomics = set(result.bundle.selected_radiomics)
        deep = set(result.bundle.selected_deep_features)
        for position, feature in enumerate(result.bundle.feature_columns, start=1):
            if feature in radiomics:
                modality = "radiomics"
            elif feature in deep:
                modality = "deep_feature"
            else:
                modality = "clinical_pet_prespecified"
            selected_rows.append(
                {
                    "model_name": model_name,
                    "display_name": result.bundle.display_name,
                    "input_position": position,
                    "modality": modality,
                    "feature": feature,
                    "selection_source": (
                        "development_only"
                        if modality != "clinical_pet_prespecified"
                        else "protocol_configuration"
                    ),
                }
            )
        registry_rows.append(
            {
                "model_name": model_name,
                "display_name": result.bundle.display_name,
                "algorithm": "random_survival_forest",
                "n_development": len(development),
                "event_count_development": int(development["event"].sum()),
                "input_count": len(result.bundle.feature_columns),
                "radiomics_count": len(result.bundle.selected_radiomics),
                "deep_feature_count": len(result.bundle.selected_deep_features),
                "random_seed": result.bundle.random_seed,
                "hyperparameters": json.dumps(result.bundle.hyperparameters, sort_keys=True),
                "training_risk_cutoff": result.bundle.training_risk_cutoff,
                "artifact": str(bundle_path.relative_to(output)).replace("\\", "/"),
                "artifact_sha256": bundle_hash,
            }
        )

    radiomics_sets = {
        tuple(result.bundle.selected_radiomics)
        for result in results.values()
        if result.bundle.selected_radiomics
    }
    deep_sets = {
        tuple(result.bundle.selected_deep_features)
        for result in results.values()
        if result.bundle.selected_deep_features
    }
    if len(radiomics_sets) > 1 or len(deep_sets) > 1:
        raise RuntimeError(
            "Final imaging selections differ across model families. The public model definitions "
            "require one frozen Top-k set per imaging modality."
        )

    registry = pd.DataFrame(registry_rows)
    selected = pd.DataFrame(selected_rows)
    tuning = pd.concat(tuning_frames, ignore_index=True, sort=False)
    rankings = (
        pd.concat(ranking_frames, ignore_index=True, sort=False)
        if ranking_frames
        else pd.DataFrame(
            columns=[
                "modality",
                "feature",
                "missing_fraction",
                "variance",
                "univariable_cindex",
                "status",
                "rank_score",
                "selected",
                "selection_rank",
                "model_name",
            ]
        )
    )
    inputs = pd.concat(input_frames, ignore_index=True, sort=False)
    _write_csv(registry, table_dir / "model_registry.csv")
    _write_csv(selected, table_dir / "selected_variables.csv")
    _write_csv(tuning, table_dir / "hyperparameter_search.csv")
    _write_csv(rankings, table_dir / "feature_rankings.csv")
    _write_csv(inputs, table_dir / "input_variable_summary.csv")

    manifest = {
        "workflow": "strict_heldout_model_building",
        "software_version": __version__,
        "created_at_utc": _utc_now(),
        "study": config["study"],
        "models": config["models"],
        "development_n": len(development),
        "development_event_count": int(development["event"].sum()),
        "development_patient_digest_sha256": patient_id_digest(
            development[schema.id_column]
        ),
        "development_file_sha256": development_hash,
        "config_file_sha256": config_hash,
        "git_commit": git_commit(_repo_root()),
        "environment": environment_snapshot(),
        "leakage_controls": {
            "validation_data_argument_accepted_by_builder": False,
            "feature_selection_scope": "each internal-CV training fold; then full development",
            "imputation_scope": "each internal-CV training fold; then full development",
            "scaling_scope": "each internal-CV training fold; then full development",
            "hyperparameter_selection_scope": "development-only cross-validation",
            "risk_cutoff_source": "full-development median risk",
            "patient_level_artifacts_publishable": False,
        },
        "artifact_sha256": dict(zip(registry["model_name"], registry["artifact_sha256"])),
    }
    write_json(manifest, table_dir / "analysis_manifest.json")
    summary = [
        "# Model-building summary",
        "",
        f"- Development cohort: n={len(development)}; 36-month events={int(development['event'].sum())}.",
        "- Independent validation data were not accepted by or read during this workflow.",
        "- Imaging feature selection, imputation, scaling, and RSF fitting were repeated inside each CV training fold.",
        "- Final preprocessing, selected features, models, and risk thresholds were frozen from development data only.",
        "- The three clinical/PET inputs are protocol-configured predictors; they are not selected by this script.",
        "- Model bundles contain private development identifiers/outcomes for leakage and IPCW auditing and must not be published.",
        "",
        _markdown_table(registry),
        "",
    ]
    (table_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return output


def _verified_bundles(
    artifacts_dir: str | Path,
    config_hash: str,
    model_names: Sequence[str],
) -> tuple[Dict[str, FrozenModelBundle], dict]:
    root = Path(artifacts_dir).resolve()
    build_manifest = _read_json(root / "table" / "analysis_manifest.json")
    if build_manifest.get("config_file_sha256") != config_hash:
        raise ValueError("Build-manifest configuration hash does not match the requested config.")
    expected_hashes = build_manifest.get("artifact_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Build manifest does not contain artifact SHA-256 values.")
    bundles = {}
    for name in model_names:
        path = root / "models" / name / "model.joblib"
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = expected_hashes.get(name)
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(f"Frozen artifact hash mismatch for {name}; refusing to deserialize.")
        bundle = load_bundle(path)
        if bundle.model_name != name:
            raise ValueError(f"Artifact/model mismatch: expected={name}, found={bundle.model_name}")
        if bundle.config_file_sha256 != config_hash:
            raise ValueError(f"Bundle configuration hash mismatch for {name}.")
        bundles[name] = bundle
    return bundles, build_manifest


def predict_models(
    feature_csv: str | Path,
    artifacts_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    model_names: Sequence[str] | None = None,
) -> Path:
    """Generate frozen predictions while loading no outcome values."""

    config = load_config(config_path)
    config_hash = sha256_file(config_path)
    requested_models = list(model_names) if model_names else list(config["models"])
    bundles, build_manifest = _verified_bundles(
        artifacts_dir, config_hash, requested_models
    )
    first = next(iter(bundles.values()))
    schema = first.schema
    for bundle in bundles.values():
        if bundle.schema != schema:
            raise ValueError("Model artifacts do not share one frozen cohort schema.")
        if bundle.config_file_sha256 != config_hash:
            raise ValueError("Configuration hash does not match the frozen model artifacts.")

    required_features = sorted(
        {feature for bundle in bundles.values() for feature in bundle.feature_columns}
    )
    feature_source = Path(feature_csv).resolve()
    features = read_feature_table(feature_source, schema, required_features)
    validation_ids = features[schema.id_column].astype(str).tolist()
    validation_identity = {
        column: features[column].tolist()
        for column in schema.identity_columns
        if column in features.columns
    }

    rows = []
    auc_times = [int(value) for value in config["validation"]["auc_times_days"]]
    brier_times = _brier_time_grid(config)
    prediction_times = sorted(set([*auc_times, *brier_times]))
    artifact_hashes = {}
    missingness_rows = []
    missingness_limit = float(
        config["validation"].get(
            "maximum_missing_fraction_per_feature",
            config["feature_selection"]["maximum_missing_fraction"],
        )
    )
    for model_name, bundle in bundles.items():
        assert_disjoint_ids(
            bundle.training_patient_ids,
            validation_ids,
            development_label="frozen development",
            validation_label="prediction cohort",
        )
        assert_disjoint_identity_values(bundle.training_identity_values, validation_identity)
        numeric = numeric_feature_frame(features, bundle.feature_columns, schema)
        missingness = numeric.isna().mean()
        for feature_name, fraction in missingness.items():
            missingness_rows.append(
                {
                    "model_name": model_name,
                    "feature": feature_name,
                    "validation_missing_fraction": float(fraction),
                }
            )
        excessive = missingness[missingness > missingness_limit]
        if not excessive.empty:
            raise ValueError(
                f"Validation missingness exceeds {missingness_limit:.1%} for {model_name}: "
                f"{excessive.to_dict()}"
            )
        risk = predict_risk(bundle, features)
        survival = predict_survival_probabilities(bundle, features, prediction_times)
        groups = np.where(risk >= bundle.training_risk_cutoff, "high", "low")
        for index, patient_id in enumerate(validation_ids):
            row = {
                "patient_id": patient_id,
                "model_name": model_name,
                "risk_score": float(risk[index]),
                "risk_group": groups[index],
                "risk_cutoff_development_median": bundle.training_risk_cutoff,
            }
            for time_index, time_days in enumerate(prediction_times):
                row[f"survival_probability_{time_days}d"] = float(survival[index, time_index])
            rows.append(row)
        artifact_hashes[model_name] = sha256_file(
            Path(artifacts_dir).resolve() / "models" / model_name / "model.joblib"
        )

    output = _prepare_clean_output(output_dir)
    predictions = pd.DataFrame(rows)
    prediction_path = output / "predictions.csv"
    _write_csv(predictions, prediction_path)
    missingness_table = pd.DataFrame(missingness_rows)
    _write_csv(missingness_table, output / "validation_missingness.csv")
    manifest = {
        "workflow": "frozen_prediction_without_outcomes",
        "software_version": __version__,
        "created_at_utc": _utc_now(),
        "models": list(bundles),
        "prediction_n": len(validation_ids),
        "prediction_patient_digest_sha256": patient_id_digest(validation_ids),
        "feature_file_sha256": sha256_file(feature_source),
        "predictions_file_sha256": sha256_file(prediction_path),
        "config_file_sha256": config_hash,
        "artifact_sha256": artifact_hashes,
        "build_manifest_sha256": sha256_file(
            Path(artifacts_dir).resolve() / "table" / "analysis_manifest.json"
        ),
        "build_manifest_workflow": build_manifest.get("workflow"),
        "prediction_times_days": prediction_times,
        "maximum_validation_missing_fraction": float(
            missingness_table["validation_missing_fraction"].max()
        ),
        "maximum_allowed_missing_fraction_per_feature": missingness_limit,
        "overlap_count": 0,
        "outcome_columns_loaded": False,
        "fit_or_fit_transform_called": False,
        "patient_level_output_confidential": True,
    }
    write_json(manifest, output / "prediction_manifest.json")
    return prediction_path


def evaluate_predictions(
    predictions_csv: str | Path,
    outcomes_csv: str | Path,
    artifacts_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    cohort_name: str,
    *,
    expected_prediction_manifest_sha256: str,
) -> Path:
    """Evaluate predictions anchored by a hash recorded before outcomes are opened."""

    config = load_config(config_path)
    config_hash = sha256_file(config_path)
    predictions_source = Path(predictions_csv).resolve()
    prediction_manifest_path = predictions_source.parent / "prediction_manifest.json"
    actual_prediction_manifest_hash = sha256_file(prediction_manifest_path)
    trusted_prediction_manifest_hash = expected_prediction_manifest_sha256.strip().lower()
    if actual_prediction_manifest_hash != trusted_prediction_manifest_hash:
        raise ValueError(
            "Prediction-manifest hash does not match the trusted pre-outcome anchor."
        )
    prediction_manifest = _read_json(prediction_manifest_path)
    if prediction_manifest.get("predictions_file_sha256") != sha256_file(predictions_source):
        raise ValueError("Prediction-file hash does not match prediction_manifest.json.")
    if prediction_manifest.get("config_file_sha256") != config_hash:
        raise ValueError("Prediction-manifest configuration hash mismatch.")
    predictions = pd.read_csv(predictions_source, dtype={"patient_id": "string"})
    required_prediction_columns = {
        "patient_id",
        "model_name",
        "risk_score",
        "risk_group",
        "risk_cutoff_development_median",
    }
    missing = sorted(required_prediction_columns.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    predictions["patient_id"] = predictions["patient_id"].map(normalize_patient_id)
    if predictions.duplicated(["model_name", "patient_id"]).any():
        raise ValueError("Prediction file contains duplicate model/patient rows.")

    file_models = list(dict.fromkeys(predictions["model_name"].astype(str).tolist()))
    if not file_models:
        raise ValueError("No configured model names are present in predictions.csv.")
    manifest_models = prediction_manifest.get("models")
    if not isinstance(manifest_models, list) or set(manifest_models) != set(file_models):
        raise ValueError(
            f"Prediction-manifest model list mismatch: manifest={manifest_models}, file={file_models}"
        )
    unknown_models = sorted(set(manifest_models).difference(config["models"]))
    if unknown_models:
        raise ValueError(f"Predictions contain models absent from the locked config: {unknown_models}")
    model_names = list(manifest_models)
    build_manifest_path = Path(artifacts_dir).resolve() / "table" / "analysis_manifest.json"
    if prediction_manifest.get("build_manifest_sha256") != sha256_file(build_manifest_path):
        raise ValueError("Prediction-manifest build-manifest hash mismatch.")
    bundles, _ = _verified_bundles(artifacts_dir, config_hash, model_names)
    current_artifact_hashes = {
        model_name: sha256_file(
            Path(artifacts_dir).resolve() / "models" / model_name / "model.joblib"
        )
        for model_name in model_names
    }
    if prediction_manifest.get("artifact_sha256") != current_artifact_hashes:
        raise ValueError("Prediction-manifest artifact hashes do not match current bundles.")
    schema = next(iter(bundles.values())).schema
    outcomes_source = Path(outcomes_csv).resolve()
    outcomes = read_outcome_table(outcomes_source, schema)
    outcome_ids = outcomes[schema.id_column].astype(str).tolist()
    if prediction_manifest.get("prediction_n") != len(outcome_ids):
        raise ValueError("Prediction-manifest sample size does not match the outcome cohort.")
    if prediction_manifest.get("prediction_patient_digest_sha256") != patient_id_digest(
        outcome_ids
    ):
        raise ValueError("Prediction-manifest patient digest does not match the outcome cohort.")
    outcome_set = set(outcome_ids)
    for model_name in model_names:
        model_set = set(
            predictions.loc[predictions["model_name"].eq(model_name), "patient_id"].astype(str)
        )
        if model_set != outcome_set:
            raise ValueError(
                f"Prediction/outcome patient sets differ for {model_name}: "
                f"prediction_n={len(model_set)}, outcome_n={len(outcome_set)}."
            )
        assert_disjoint_ids(
            bundles[model_name].training_patient_ids,
            outcome_ids,
            development_label="frozen development",
            validation_label=cohort_name,
        )

    repeats = int(config["validation"]["bootstrap_repeats"])
    bootstrap_seed = int(config["validation"]["bootstrap_seed"])
    auc_times = [int(value) for value in config["validation"]["auc_times_days"]]
    brier_times = _brier_time_grid(config)
    horizon = int(config["study"]["horizon_days"])
    bins = int(config["validation"]["calibration_bins"])
    minimum_bin_size = int(config["validation"]["calibration_min_bin_size"])
    minimum_bootstrap_fraction = float(
        config["validation"]["minimum_bootstrap_valid_fraction"]
    )
    minimum_valid_bootstrap = int(np.ceil(repeats * minimum_bootstrap_fraction))

    metric_rows = []
    auc_frames = []
    brier_frames = []
    calibration_frames = []
    group_frames = []
    risk_frames = []
    paired_risks: Dict[str, np.ndarray] = {}

    order = pd.DataFrame({"patient_id": outcome_ids, "_order": np.arange(len(outcome_ids))})
    for model_index, model_name in enumerate(model_names):
        bundle = bundles[model_name]
        model_predictions = predictions[predictions["model_name"].eq(model_name)].copy()
        merged = outcomes.rename(columns={schema.id_column: "patient_id"}).merge(
            model_predictions, on="patient_id", how="inner", validate="one_to_one"
        )
        merged = merged.merge(order, on="patient_id", how="left", validate="one_to_one")
        merged = merged.sort_values("_order").reset_index(drop=True)
        risk = pd.to_numeric(merged["risk_score"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(risk).all():
            raise ValueError(f"Non-finite risk scores for {model_name}.")
        cutoffs = pd.to_numeric(
            merged["risk_cutoff_development_median"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.allclose(cutoffs, bundle.training_risk_cutoff, rtol=0, atol=1e-12):
            raise ValueError(f"Prediction cutoff differs from frozen development cutoff: {model_name}")
        expected_groups = np.where(risk >= bundle.training_risk_cutoff, "high", "low")
        if not np.array_equal(expected_groups, merged["risk_group"].astype(str).to_numpy()):
            raise ValueError(f"Risk groups were not generated from the frozen cutoff: {model_name}")
        paired_risks[model_name] = risk

        harrell = harrell_cindex(merged, risk)
        uno, uno_tau, uno_status = uno_cindex(
            bundle.development_outcome, merged, risk, horizon
        )
        ci_low, ci_high, valid_bootstrap = bootstrap_harrell_cindex(
            merged, risk, repeats, bootstrap_seed + model_index
        )
        cindex_ci_status = "ok"
        if valid_bootstrap < minimum_valid_bootstrap:
            ci_low = float("nan")
            ci_high = float("nan")
            cindex_ci_status = "insufficient_valid_bootstrap"
        auc = dynamic_auc_table(
            bundle.development_outcome,
            merged,
            risk,
            auc_times,
            repeats,
            bootstrap_seed + 1000 * (model_index + 1),
        ).assign(model_name=model_name, cohort=cohort_name)
        insufficient_auc_bootstrap = (
            auc["status"].eq("ok")
            & auc["bootstrap_valid_repeats"].lt(minimum_valid_bootstrap)
        )
        auc.loc[insufficient_auc_bootstrap, ["ci95_low", "ci95_high"]] = np.nan
        auc.loc[insufficient_auc_bootstrap, "status"] = "insufficient_valid_bootstrap"
        auc_frames.append(auc)

        probability_columns = [f"survival_probability_{time}d" for time in brier_times]
        missing_probability = sorted(set(probability_columns).difference(merged.columns))
        if missing_probability:
            raise ValueError(f"Missing frozen survival probabilities: {missing_probability}")
        survival = merged[probability_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        if not np.isfinite(survival).all() or ((survival < 0) | (survival > 1)).any():
            raise ValueError(f"Invalid survival probabilities for {model_name}.")
        if np.any(np.diff(survival, axis=1) > 1e-10):
            raise ValueError(f"Survival probabilities increase over time for {model_name}.")
        brier, integrated, ibs_status = brier_table(
            bundle.development_outcome, merged, survival, brier_times
        )
        if not brier.empty:
            brier_frames.append(brier.assign(model_name=model_name, cohort=cohort_name))

        horizon_index = brier_times.index(horizon)
        calibration = calibration_table(
            merged, survival[:, horizon_index], horizon, bins, minimum_bin_size
        ).assign(model_name=model_name, cohort=cohort_name)
        if not calibration.empty:
            calibration_frames.append(calibration)

        groups, logrank_p = risk_group_summary(merged, risk, bundle.training_risk_cutoff)
        group_frames.append(groups.assign(model_name=model_name, cohort=cohort_name))
        risk_output = merged[
            [
                "patient_id",
                schema.time_column,
                schema.event_column,
                "time_days",
                "event",
                "risk_score",
                "risk_group",
            ]
        ].copy()
        risk_frames.append(risk_output.assign(model_name=model_name, cohort=cohort_name))

        auc_lookup = auc.set_index("requested_time_days")["cumulative_dynamic_auc"].to_dict()
        metric_rows.append(
            {
                "model_name": model_name,
                "cohort": cohort_name,
                "n_used": len(merged),
                "event_count": int(merged["event"].sum()),
                "concordance_index": harrell,
                "cindex_ci95_low": ci_low,
                "cindex_ci95_high": ci_high,
                "cindex_bootstrap_valid_repeats": valid_bootstrap,
                "cindex_ci_status": cindex_ci_status,
                "uno_concordance_index": uno,
                "uno_tau_days": uno_tau,
                "uno_status": uno_status,
                "auc_12m": auc_lookup.get(365, float("nan")),
                "auc_24m": auc_lookup.get(730, float("nan")),
                "auc_36m": auc_lookup.get(1095, float("nan")),
                "integrated_brier_score": integrated,
                "ibs_status": ibs_status,
                "ibs_start_days": brier_times[0],
                "ibs_end_days": brier_times[-1],
                "ibs_time_points": len(brier_times),
                "logrank_p_high_vs_low": logrank_p,
                "early_censored_before_horizon": int(
                    ((merged[schema.event_column] == 0) & (merged[schema.time_column] < horizon)).sum()
                ),
                "partial_aic": float("nan"),
                "lr_test_p": float("nan"),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    auc_all = pd.concat(auc_frames, ignore_index=True) if auc_frames else pd.DataFrame()
    brier_all = pd.concat(brier_frames, ignore_index=True) if brier_frames else pd.DataFrame()
    calibration_all = (
        pd.concat(calibration_frames, ignore_index=True)
        if calibration_frames
        else pd.DataFrame(
            columns=[
                "calibration_bin",
                "n",
                "event_count",
                "n_at_risk_at_horizon",
                "mean_predicted_risk",
                "km_observed_risk",
                "km_observed_risk_ci95_low",
                "km_observed_risk_ci95_high",
                "status",
                "model_name",
                "cohort",
            ]
        )
    )
    groups_all = pd.concat(group_frames, ignore_index=True)
    risks_all = pd.concat(risk_frames, ignore_index=True)
    comparisons = paired_cindex_comparisons(
        outcomes, paired_risks, repeats, bootstrap_seed + 99999
    )
    if not comparisons.empty:
        insufficient = comparisons["bootstrap_valid_repeats"].lt(minimum_valid_bootstrap)
        comparisons.loc[insufficient, ["ci95_low", "ci95_high"]] = np.nan
        comparisons["ci_status"] = np.where(
            insufficient, "insufficient_valid_bootstrap", "ok"
        )
        comparisons["cohort"] = cohort_name
    else:
        comparisons["ci_status"] = pd.Series(dtype="string")
        comparisons["cohort"] = pd.Series(dtype="string")

    output = _prepare_clean_output(output_dir)
    table_dir = output / "table"
    fig_dir = output / "fig"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics, table_dir / "model_metrics.csv")
    _write_csv(groups_all, table_dir / "risk_group_summary.csv")
    _write_csv(auc_all, table_dir / "time_dependent_auc.csv")
    _write_csv(brier_all, table_dir / "brier_scores.csv")
    _write_csv(calibration_all, table_dir / "calibration.csv")
    _write_csv(comparisons, table_dir / "model_comparisons.csv")
    if bool(config["validation"].get("export_patient_level", False)):
        _write_csv(risks_all, table_dir / "risk_scores.csv")

    plot_km_risk_groups(risks_all, fig_dir / "km_risk_groups")
    plot_time_dependent_auc(auc_all, fig_dir / "time_dependent_auc")
    plot_calibration(calibration_all, fig_dir / "calibration")

    manifest = {
        "workflow": "strict_heldout_evaluation_of_frozen_predictions",
        "software_version": __version__,
        "created_at_utc": _utc_now(),
        "cohort": cohort_name,
        "models": model_names,
        "n": len(outcomes),
        "event_count_36m": int(outcomes["event"].sum()),
        "early_censored_before_36m": int(
            ((outcomes[schema.event_column] == 0) & (outcomes[schema.time_column] < horizon)).sum()
        ),
        "validation_patient_digest_sha256": patient_id_digest(outcome_ids),
        "predictions_file_sha256": sha256_file(predictions_source),
        "outcomes_file_sha256": sha256_file(outcomes_source),
        "config_file_sha256": config_hash,
        "prediction_manifest_sha256": actual_prediction_manifest_hash,
        "artifact_sha256": current_artifact_hashes,
        "overlap_count": 0,
        "risk_cutoff_source": "frozen development median",
        "feature_selection_or_model_fitting_during_evaluation": False,
        "censoring_aware_primary_metrics": [
            "Harrell C-index",
            "Uno C-index",
            "IPCW cumulative/dynamic AUC",
            "IPCW Brier score",
            "integrated Brier score",
            "Kaplan-Meier calibration",
        ],
        "naive_binary_roc_or_hosmer_lemeshow_used": False,
        "patient_level_results_exported": bool(
            config["validation"].get("export_patient_level", False)
        ),
        "ibs_interval_days": [brier_times[0], brier_times[-1]],
        "ibs_time_points": len(brier_times),
        "bootstrap_confidence_intervals": {
            "type": "conditional pointwise percentile",
            "development_pipeline_refitted_in_bootstrap": False,
            "minimum_valid_fraction": minimum_bootstrap_fraction,
        },
        "environment": environment_snapshot(),
        "git_commit": git_commit(_repo_root()),
    }
    write_json(manifest, table_dir / "analysis_manifest.json")
    summary = [
        f"# Strict held-out evaluation: {cohort_name}",
        "",
        f"- Cohort: n={len(outcomes)}; 36-month events={int(outcomes['event'].sum())}.",
        "- Development/validation patient overlap: 0 (hard-stop guard passed).",
        "- Prediction and outcome evaluation were separated; no fit or fit_transform operation was called.",
        "- All risk groups use the frozen development-cohort median cutoff.",
        "- Primary discrimination and accuracy metrics account for censoring; naive binary ROC and Hosmer–Lemeshow tests are intentionally omitted.",
        "",
        _markdown_table(metrics),
        "",
    ]
    (table_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return output
