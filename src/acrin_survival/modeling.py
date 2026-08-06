"""Fold-contained preprocessing, RSF tuning, and frozen model bundles."""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest

from .audit import patient_id_digest
from .data import (
    CohortSchema,
    candidate_feature_columns,
    make_survival_outcome,
    numeric_feature_frame,
)
from .selection import FeatureSelectionResult, select_top_features

MODEL_DISPLAY_NAMES = {
    "CP": "clinical–PET model",
    "CPR": "clinical–PET–radiomics model",
    "CPRD": "clinical–PET–radiomics–deep feature model",
    "RD": "CT-based radiomics–deep feature model",
}


@dataclass
class FrozenModelBundle:
    """Everything needed for prediction, with no validation-derived state."""

    format_version: str
    model_name: str
    display_name: str
    schema: CohortSchema
    feature_columns: List[str]
    selected_radiomics: List[str]
    selected_deep_features: List[str]
    hyperparameters: Dict[str, Any]
    imputer: SimpleImputer
    scaler: StandardScaler
    estimator: RandomSurvivalForest
    training_risk_cutoff: float
    training_patient_ids: Tuple[str, ...]
    training_patient_digest: str
    training_identity_values: Dict[str, Tuple[str, ...]]
    development_outcome: np.ndarray
    development_file_sha256: str
    config_file_sha256: str
    random_seed: int


@dataclass
class ModelTrainingResult:
    bundle: FrozenModelBundle
    tuning: pd.DataFrame
    radiomics_ranking: pd.DataFrame
    deep_ranking: pd.DataFrame
    input_summary: pd.DataFrame


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int(base_seed) + int(zlib.crc32(payload) % 1_000_000)


def _adaptive_cv_splits(frame: pd.DataFrame, n_splits: int, seed: int):
    if len(frame) < n_splits:
        raise ValueError(f"Development n={len(frame)} is smaller than CV folds={n_splits}.")
    event = frame["event"].astype(int)
    ranks = frame["time_days"].rank(method="first")
    bins = pd.qcut(ranks, q=min(4, len(frame)), labels=False, duplicates="drop")
    strata = event.astype(str) + "_" + bins.astype(str)
    counts = strata.value_counts()
    if not counts.empty and int(counts.min()) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(frame, strata))
    event_counts = event.value_counts()
    if len(event_counts) > 1 and int(event_counts.min()) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(frame, event))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame))


def _needed_modalities(model_name: str) -> tuple[bool, bool]:
    return model_name in {"CPR", "CPRD", "RD"}, model_name in {"CPRD", "RD"}


def model_features(
    model_name: str,
    schema: CohortSchema,
    radiomics: Sequence[str],
    deep_features: Sequence[str],
) -> List[str]:
    if model_name == "CP":
        return list(schema.clinical_pet_features)
    if model_name == "CPR":
        return [*schema.clinical_pet_features, *radiomics]
    if model_name == "CPRD":
        return [*schema.clinical_pet_features, *radiomics, *deep_features]
    if model_name == "RD":
        return [*radiomics, *deep_features]
    raise ValueError(f"Unsupported model: {model_name}")


def _select_modalities(
    training: pd.DataFrame,
    schema: CohortSchema,
    config: Dict[str, Any],
    model_name: str,
) -> tuple[FeatureSelectionResult | None, FeatureSelectionResult | None]:
    needs_radiomics, needs_deep = _needed_modalities(model_name)
    radiomics_result = None
    deep_result = None
    if needs_radiomics:
        candidates = candidate_feature_columns(training, schema.radiomics_prefixes)
        radiomics_result = select_top_features(
            training, candidates, schema, config["feature_selection"], "radiomics"
        )
    if needs_deep:
        candidates = candidate_feature_columns(training, schema.deep_feature_prefixes)
        deep_result = select_top_features(
            training, candidates, schema, config["feature_selection"], "deep_feature"
        )
    return radiomics_result, deep_result


def _fit_transformer(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    schema: CohortSchema,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    training_numeric = numeric_feature_frame(training, features, schema)
    validation_numeric = numeric_feature_frame(validation, features, schema)
    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    scaler = StandardScaler()
    x_training = scaler.fit_transform(imputer.fit_transform(training_numeric))
    x_validation = scaler.transform(imputer.transform(validation_numeric))
    if x_training.shape[1] != len(features):
        raise ValueError("A frozen model input was entirely missing in a CV training fold.")
    return x_training, x_validation, imputer, scaler


def _build_estimator(params: Dict[str, Any], seed: int, n_jobs: int) -> RandomSurvivalForest:
    return RandomSurvivalForest(
        n_estimators=int(params["n_estimators"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        max_depth=params.get("max_depth"),
        random_state=int(seed),
        n_jobs=int(n_jobs),
        low_memory=False,
    )


def _safe_score(estimator: RandomSurvivalForest, x: np.ndarray, frame: pd.DataFrame) -> float:
    risk = np.asarray(estimator.predict(x), dtype=float)
    if not np.isfinite(risk).all() or np.std(risk) <= 1e-12:
        return float("nan")
    outcome = make_survival_outcome(frame)
    try:
        from sksurv.metrics import concordance_index_censored

        return float(concordance_index_censored(outcome["event"], outcome["time"], risk)[0])
    except (ValueError, ZeroDivisionError):
        return float("nan")


def tune_hyperparameters(
    development: pd.DataFrame,
    schema: CohortSchema,
    config: Dict[str, Any],
    model_name: str,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    """Tune with feature selection, imputation, and scaling repeated inside each CV fold."""

    training_config = config["training"]
    cv_seed = int(training_config["cross_validation_seed"])
    model_seed = stable_seed(cv_seed, model_name, "cv")
    splits = _adaptive_cv_splits(
        development, int(training_config["cross_validation_folds"]), model_seed
    )

    prepared_folds = []
    for fold_index, (training_index, validation_index) in enumerate(splits):
        fold_training = development.iloc[training_index].copy()
        fold_validation = development.iloc[validation_index].copy()
        radiomics_result, deep_result = _select_modalities(
            fold_training, schema, config, model_name
        )
        radiomics = radiomics_result.selected if radiomics_result else []
        deep_features = deep_result.selected if deep_result else []
        features = model_features(model_name, schema, radiomics, deep_features)
        x_training, x_validation, _, _ = _fit_transformer(
            fold_training, fold_validation, features, schema
        )
        prepared_folds.append(
            {
                "fold": fold_index,
                "training": fold_training,
                "validation": fold_validation,
                "x_training": x_training,
                "x_validation": x_validation,
                "features": features,
            }
        )

    rows = []
    grid = list(ParameterGrid(training_config["hyperparameter_grid"]))
    for params in grid:
        scores = []
        for prepared in prepared_folds:
            seed = stable_seed(model_seed, "fold", str(prepared["fold"]))
            estimator = _build_estimator(params, seed, int(training_config["n_jobs"]))
            estimator.fit(
                prepared["x_training"], make_survival_outcome(prepared["training"])
            )
            score = _safe_score(
                estimator, prepared["x_validation"], prepared["validation"]
            )
            scores.append(score)
        finite = np.asarray([score for score in scores if np.isfinite(score)], dtype=float)
        complete = finite.size == len(prepared_folds)
        rows.append(
            {
                "model_name": model_name,
                "parameters": json.dumps(params, sort_keys=True),
                "mean_cv_cindex": float(finite.mean()) if complete else float("nan"),
                "sd_cv_cindex": (
                    float(finite.std(ddof=1)) if complete and finite.size > 1 else float("nan")
                ),
                "valid_folds": int(finite.size),
                "cv_fold_feature_sets": json.dumps(
                    [prepared["features"] for prepared in prepared_folds],
                    ensure_ascii=False,
                ),
                **{f"fold_{index + 1}_cindex": value for index, value in enumerate(scores)},
            }
        )
    tuning = pd.DataFrame(rows)
    valid = tuning.dropna(subset=["mean_cv_cindex"])
    if valid.empty:
        raise RuntimeError(f"Every hyperparameter candidate failed for {model_name}.")
    best = valid.sort_values(
        ["mean_cv_cindex", "sd_cv_cindex", "parameters"],
        ascending=[False, True, True],
    ).iloc[0]
    return json.loads(best["parameters"]), tuning


def train_model(
    development: pd.DataFrame,
    schema: CohortSchema,
    config: Dict[str, Any],
    model_name: str,
    development_sha256: str,
    config_sha256: str,
) -> ModelTrainingResult:
    best_params, tuning = tune_hyperparameters(development, schema, config, model_name)
    radiomics_result, deep_result = _select_modalities(development, schema, config, model_name)
    selected_radiomics = radiomics_result.selected if radiomics_result else []
    selected_deep = deep_result.selected if deep_result else []
    features = model_features(model_name, schema, selected_radiomics, selected_deep)
    numeric = numeric_feature_frame(development, features, schema)
    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    scaler = StandardScaler()
    x_development = scaler.fit_transform(imputer.fit_transform(numeric))
    if x_development.shape[1] != len(features):
        raise ValueError("A final model input was entirely missing in the development cohort.")

    model_seed = int(config["training"]["model_random_seeds"][model_name])
    estimator = _build_estimator(best_params, model_seed, int(config["training"]["n_jobs"]))
    estimator.fit(x_development, make_survival_outcome(development))
    development_risk = np.asarray(estimator.predict(x_development), dtype=float)
    cutoff = float(np.median(development_risk))

    ids = tuple(development[schema.id_column].astype(str).tolist())
    identity_values = {
        column: tuple(development[column].dropna().astype(str).tolist())
        for column in schema.identity_columns
        if column in development.columns
    }
    bundle = FrozenModelBundle(
        format_version="1.0",
        model_name=model_name,
        display_name=MODEL_DISPLAY_NAMES[model_name],
        schema=schema,
        feature_columns=features,
        selected_radiomics=selected_radiomics,
        selected_deep_features=selected_deep,
        hyperparameters=best_params,
        imputer=imputer,
        scaler=scaler,
        estimator=estimator,
        training_risk_cutoff=cutoff,
        training_patient_ids=ids,
        training_patient_digest=patient_id_digest(ids),
        training_identity_values=identity_values,
        development_outcome=make_survival_outcome(development),
        development_file_sha256=development_sha256,
        config_file_sha256=config_sha256,
        random_seed=model_seed,
    )

    input_rows = []
    for feature in features:
        input_rows.append(
            {
                "model_name": model_name,
                "variable": feature,
                "development_missing_fraction": float(numeric[feature].isna().mean()),
                "development_variance": float(numeric[feature].var(skipna=True)),
            }
        )
    empty_ranking = pd.DataFrame(
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
        ]
    )
    return ModelTrainingResult(
        bundle=bundle,
        tuning=tuning,
        radiomics_ranking=(
            radiomics_result.ranking if radiomics_result is not None else empty_ranking.copy()
        ),
        deep_ranking=(deep_result.ranking if deep_result is not None else empty_ranking.copy()),
        input_summary=pd.DataFrame(input_rows),
    )


def transform_with_bundle(bundle: FrozenModelBundle, frame: pd.DataFrame) -> np.ndarray:
    numeric = numeric_feature_frame(frame, bundle.feature_columns, bundle.schema)
    entirely_missing = numeric.columns[numeric.isna().all()].tolist()
    if entirely_missing:
        raise ValueError(f"Validation features are entirely missing: {entirely_missing}")
    transformed = bundle.scaler.transform(bundle.imputer.transform(numeric))
    if not np.isfinite(transformed).all():
        raise ValueError("Validation transformation produced non-finite values.")
    return transformed


def predict_risk(bundle: FrozenModelBundle, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(bundle.estimator.predict(transform_with_bundle(bundle, frame)), dtype=float)


def predict_survival_probabilities(
    bundle: FrozenModelBundle, frame: pd.DataFrame, times: Sequence[float]
) -> np.ndarray:
    functions = bundle.estimator.predict_survival_function(
        transform_with_bundle(bundle, frame), return_array=False
    )
    requested = np.asarray(times, dtype=float)
    rows = []
    for function in functions:
        upper = float(function.x[-1])
        unsupported = (requested > upper) & ~np.isclose(requested, upper)
        if np.any(unsupported):
            raise ValueError(
                f"Requested survival time exceeds the frozen model support: "
                f"requested_max={requested.max()}, model_max={upper}."
            )
        evaluation_times = np.where(requested >= upper, np.nextafter(upper, 0.0), requested)
        rows.append(function(evaluation_times))
    return np.vstack(rows)


def save_bundle(bundle: FrozenModelBundle, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=3)


def load_bundle(path: str | Path) -> FrozenModelBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, FrozenModelBundle):
        raise TypeError(f"Unexpected artifact type in {path!s}.")
    if bundle.format_version != "1.0":
        raise ValueError(f"Unsupported model-bundle format: {bundle.format_version}")
    return bundle
