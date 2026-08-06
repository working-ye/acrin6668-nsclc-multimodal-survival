"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

ALLOWED_MODELS = ("CP", "CPR", "CPRD", "RD")


class ConfigurationError(ValueError):
    """Raised when a workflow configuration is incomplete or inconsistent."""


def _require(mapping: Dict[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing required configuration key: {location}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration and enforce the public workflow contract."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping.")

    config = deepcopy(raw)
    for section in ("study", "schema", "models", "feature_selection", "training", "validation"):
        _require(config, section, "config")

    schema = config["schema"]
    if not isinstance(schema, dict):
        raise ConfigurationError("config.schema must be a mapping.")
    for key in (
        "id_column",
        "time_column",
        "event_column",
        "clinical_pet_features",
        "radiomics_prefixes",
        "deep_feature_prefixes",
    ):
        _require(schema, key, "config.schema")

    if not isinstance(config["models"], list) or not config["models"]:
        raise ConfigurationError("config.models must be a non-empty list.")
    models = [str(value).upper() for value in config["models"]]
    if len(models) != len(set(models)):
        raise ConfigurationError("config.models contains duplicate model names.")
    unknown = sorted(set(models).difference(ALLOWED_MODELS))
    if unknown:
        raise ConfigurationError(f"Unsupported models: {unknown}; allowed={ALLOWED_MODELS}")
    config["models"] = models

    study = config["study"]
    horizon = int(_require(study, "horizon_days", "config.study"))
    if horizon <= 0:
        raise ConfigurationError("config.study.horizon_days must be positive.")

    selection = config["feature_selection"]
    top_k = int(_require(selection, "top_k_per_imaging_modality", "config.feature_selection"))
    if top_k < 1:
        raise ConfigurationError("top_k_per_imaging_modality must be at least 1.")
    missing = float(_require(selection, "maximum_missing_fraction", "config.feature_selection"))
    correlation = float(
        _require(selection, "maximum_absolute_correlation", "config.feature_selection")
    )
    if not 0 <= missing < 1:
        raise ConfigurationError("maximum_missing_fraction must be in [0, 1).")
    if not 0 < correlation <= 1:
        raise ConfigurationError("maximum_absolute_correlation must be in (0, 1].")

    training = config["training"]
    _require(training, "cross_validation_seed", "config.training")
    model_seeds = _require(training, "model_random_seeds", "config.training")
    if not isinstance(model_seeds, dict):
        raise ConfigurationError("config.training.model_random_seeds must be a mapping.")
    missing_seeds = sorted(set(models).difference(model_seeds))
    if missing_seeds:
        raise ConfigurationError(f"Missing explicit model random seeds: {missing_seeds}")
    folds = int(_require(training, "cross_validation_folds", "config.training"))
    if folds < 2:
        raise ConfigurationError("cross_validation_folds must be at least 2.")
    grid = _require(training, "hyperparameter_grid", "config.training")
    for key in ("n_estimators", "min_samples_leaf", "max_features", "max_depth"):
        values = _require(grid, key, "config.training.hyperparameter_grid")
        if not isinstance(values, list) or not values:
            raise ConfigurationError(f"Hyperparameter grid entry {key!r} must be a non-empty list.")

    validation = config["validation"]
    if int(_require(validation, "bootstrap_repeats", "config.validation")) < 0:
        raise ConfigurationError("bootstrap_repeats cannot be negative.")
    auc_times = [int(value) for value in _require(validation, "auc_times_days", "config.validation")]
    if any(value <= 0 or value > horizon for value in auc_times):
        raise ConfigurationError("Every AUC time must be positive and no later than horizon_days.")
    if auc_times != sorted(set(auc_times)):
        raise ConfigurationError("auc_times_days must be unique and strictly increasing.")
    if horizon not in auc_times:
        raise ConfigurationError("auc_times_days must include the configured analysis horizon.")
    brier_start = int(_require(validation, "brier_start_days", "config.validation"))
    brier_end = int(_require(validation, "brier_end_days", "config.validation"))
    brier_step = int(_require(validation, "brier_step_days", "config.validation"))
    if not 0 < brier_start < brier_end <= horizon:
        raise ConfigurationError(
            "Brier interval must satisfy 0 < brier_start_days < brier_end_days <= horizon_days."
        )
    if brier_end != horizon:
        raise ConfigurationError(
            "brier_end_days must equal horizon_days so horizon calibration and IBS use one locked endpoint."
        )
    if brier_step <= 0:
        raise ConfigurationError("brier_step_days must be positive.")
    missing_limit = float(
        _require(
            validation,
            "maximum_missing_fraction_per_feature",
            "config.validation",
        )
    )
    if not 0 <= missing_limit < 1:
        raise ConfigurationError(
            "maximum_missing_fraction_per_feature must be in [0, 1)."
        )
    minimum_bootstrap = float(
        _require(
            validation,
            "minimum_bootstrap_valid_fraction",
            "config.validation",
        )
    )
    if not 0 < minimum_bootstrap <= 1:
        raise ConfigurationError("minimum_bootstrap_valid_fraction must be in (0, 1].")
    config["validation"]["auc_times_days"] = auc_times
    config["_config_path"] = str(config_path)
    return config
