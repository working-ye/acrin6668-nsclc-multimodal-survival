"""Cohort schema enforcement and numeric feature preparation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .provenance import normalize_patient_id


class DataSchemaError(ValueError):
    """Raised when a cohort table violates the declared analysis schema."""


@dataclass(frozen=True)
class CohortSchema:
    id_column: str
    time_column: str
    event_column: str
    horizon_days: int
    clinical_pet_features: tuple[str, ...]
    radiomics_prefixes: tuple[str, ...]
    deep_feature_prefixes: tuple[str, ...]
    feature_encodings: Dict[str, Dict[str, Any]]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CohortSchema:
        schema = config["schema"]
        return cls(
            id_column=str(schema["id_column"]),
            time_column=str(schema["time_column"]),
            event_column=str(schema["event_column"]),
            horizon_days=int(config["study"]["horizon_days"]),
            clinical_pet_features=tuple(map(str, schema["clinical_pet_features"])),
            radiomics_prefixes=tuple(map(str, schema["radiomics_prefixes"])),
            deep_feature_prefixes=tuple(map(str, schema["deep_feature_prefixes"])),
            feature_encodings=dict(schema.get("feature_encodings", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_column": self.id_column,
            "time_column": self.time_column,
            "event_column": self.event_column,
            "horizon_days": self.horizon_days,
            "clinical_pet_features": list(self.clinical_pet_features),
            "radiomics_prefixes": list(self.radiomics_prefixes),
            "deep_feature_prefixes": list(self.deep_feature_prefixes),
            "feature_encodings": self.feature_encodings,
        }


def read_cohort(
    path: str | Path,
    schema: CohortSchema,
    *,
    required_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read a wide feature table and fail fast on malformed outcomes or identifiers."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, dtype={schema.id_column: "string"}, low_memory=False)
    required = {
        schema.id_column,
        schema.time_column,
        schema.event_column,
    }
    required.update(required_features if required_features is not None else schema.clinical_pet_features)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataSchemaError(f"Required columns are missing from {source.name}: {missing}")

    frame = frame.copy()
    frame[schema.id_column] = frame[schema.id_column].map(normalize_patient_id)
    if frame[schema.id_column].eq("").any():
        rows = frame.index[frame[schema.id_column].eq("")].tolist()[:5]
        raise DataSchemaError(f"Blank patient identifiers at rows {rows}.")
    duplicated = frame.loc[frame[schema.id_column].duplicated(keep=False), schema.id_column]
    if not duplicated.empty:
        values = sorted(duplicated.unique().tolist())[:5]
        raise DataSchemaError(f"Duplicate patient identifiers: {values}")

    raw_time = pd.to_numeric(frame[schema.time_column], errors="coerce")
    raw_event = pd.to_numeric(frame[schema.event_column], errors="coerce")
    if raw_time.isna().any() or not np.isfinite(raw_time.to_numpy(dtype=float)).all() or (raw_time <= 0).any():
        raise DataSchemaError("Survival times must be finite and strictly positive.")
    if raw_event.isna().any() or not raw_event.isin([0, 1]).all():
        raise DataSchemaError("Event indicators must contain only 0 and 1.")

    frame[schema.time_column] = raw_time.astype(float)
    frame[schema.event_column] = raw_event.astype(int)
    frame["time_days"] = np.minimum(raw_time, float(schema.horizon_days))
    frame["event"] = (
        raw_event.eq(1) & raw_time.le(float(schema.horizon_days))
    ).astype(int)
    frame["metric_time_days"] = frame["time_days"].astype(float)
    horizon_controls = frame["event"].eq(0) & raw_time.ge(float(schema.horizon_days))
    frame.loc[horizon_controls, "metric_time_days"] = np.nextafter(
        float(schema.horizon_days), np.inf
    )
    return frame


def read_feature_table(
    path: str | Path,
    schema: CohortSchema,
    required_features: Iterable[str],
) -> pd.DataFrame:
    """Read prediction-only features without requiring or inspecting outcomes."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    header = pd.read_csv(source, nrows=0)
    required = {schema.id_column, *required_features}
    missing = sorted(required.difference(header.columns))
    if missing:
        raise DataSchemaError(f"Prediction columns are missing from {source.name}: {missing}")
    use_columns = sorted(required)
    frame = pd.read_csv(
        source,
        usecols=use_columns,
        dtype={schema.id_column: "string"},
        low_memory=False,
    )
    frame = frame.copy()
    frame[schema.id_column] = frame[schema.id_column].map(normalize_patient_id)
    if frame[schema.id_column].eq("").any():
        raise DataSchemaError("Prediction table contains blank patient identifiers.")
    duplicated = frame.loc[frame[schema.id_column].duplicated(keep=False), schema.id_column]
    if not duplicated.empty:
        raise DataSchemaError(
            f"Duplicate prediction identifiers: {sorted(duplicated.unique().tolist())[:5]}"
        )
    return frame


def read_outcome_table(path: str | Path, schema: CohortSchema) -> pd.DataFrame:
    """Read only identifiers and outcomes for the second-stage evaluator."""

    source = Path(path).resolve()
    required = [schema.id_column, schema.time_column, schema.event_column]
    header = pd.read_csv(source, nrows=0)
    missing = sorted(set(required).difference(header.columns))
    if missing:
        raise DataSchemaError(f"Outcome columns are missing from {source.name}: {missing}")
    frame = pd.read_csv(
        source,
        usecols=required,
        dtype={schema.id_column: "string"},
        low_memory=False,
    )
    frame[schema.id_column] = frame[schema.id_column].map(normalize_patient_id)
    if frame[schema.id_column].eq("").any():
        raise DataSchemaError("Outcome table contains blank patient identifiers.")
    if frame[schema.id_column].duplicated().any():
        raise DataSchemaError("Outcome table contains duplicate patient identifiers.")
    raw_time = pd.to_numeric(frame[schema.time_column], errors="coerce")
    raw_event = pd.to_numeric(frame[schema.event_column], errors="coerce")
    if raw_time.isna().any() or not np.isfinite(raw_time.to_numpy(dtype=float)).all() or (raw_time <= 0).any():
        raise DataSchemaError("Outcome times must be finite and strictly positive.")
    if raw_event.isna().any() or not raw_event.isin([0, 1]).all():
        raise DataSchemaError("Outcome event indicators must contain only 0 and 1.")
    frame[schema.time_column] = raw_time.astype(float)
    frame[schema.event_column] = raw_event.astype(int)
    frame["time_days"] = np.minimum(raw_time, float(schema.horizon_days))
    frame["event"] = (
        raw_event.eq(1) & raw_time.le(float(schema.horizon_days))
    ).astype(int)
    frame["metric_time_days"] = frame["time_days"].astype(float)
    horizon_controls = frame["event"].eq(0) & raw_time.ge(float(schema.horizon_days))
    frame.loc[horizon_controls, "metric_time_days"] = np.nextafter(
        float(schema.horizon_days), np.inf
    )
    return frame


def make_survival_outcome(frame: pd.DataFrame) -> np.ndarray:
    time_column = "metric_time_days" if "metric_time_days" in frame.columns else "time_days"
    return np.array(
        [(bool(event), float(time)) for event, time in zip(frame["event"], frame[time_column])],
        dtype=[("event", "?"), ("time", "<f8")],
    )


def candidate_feature_columns(frame: pd.DataFrame, prefixes: Iterable[str]) -> List[str]:
    prefixes_tuple = tuple(prefixes)
    return sorted(column for column in frame.columns if column.startswith(prefixes_tuple))


def _encode_series(series: pd.Series, encoding: Dict[str, Any] | None) -> pd.Series:
    if not encoding:
        return pd.to_numeric(series, errors="coerce")
    kind = str(encoding.get("type", "numeric"))
    if kind == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if kind == "leading_integer":
        extracted = series.astype(str).str.extract(r"^\s*([+-]?\d+)", expand=False)
        return pd.to_numeric(extracted, errors="coerce")
    if kind == "binary_map":
        mapping = {str(key).strip().lower(): value for key, value in encoding["values"].items()}
        normalized = series.astype(str).str.strip().str.lower()
        return pd.to_numeric(normalized.map(mapping), errors="coerce")
    if kind == "regex":
        pattern = re.compile(str(encoding["pattern"]))
        extracted = series.astype(str).map(
            lambda value: pattern.search(value).group(1) if pattern.search(value) else np.nan
        )
        return pd.to_numeric(extracted, errors="coerce")
    raise DataSchemaError(f"Unsupported feature encoding type: {kind}")


def numeric_feature_frame(
    frame: pd.DataFrame,
    features: Iterable[str],
    schema: CohortSchema,
) -> pd.DataFrame:
    """Coerce only frozen model inputs; missing required validation features are fatal."""

    requested = list(features)
    missing = sorted(set(requested).difference(frame.columns))
    if missing:
        raise DataSchemaError(f"Frozen model features are absent: {missing}")
    output = pd.DataFrame(index=frame.index)
    for feature in requested:
        encoded = _encode_series(frame[feature], schema.feature_encodings.get(feature))
        output[feature] = encoded.replace([np.inf, -np.inf], np.nan).astype(float)
    return output


def validate_feature_families(frame: pd.DataFrame, schema: CohortSchema, top_k: int) -> None:
    radiomics = candidate_feature_columns(frame, schema.radiomics_prefixes)
    deep = candidate_feature_columns(frame, schema.deep_feature_prefixes)
    if len(radiomics) < top_k:
        raise DataSchemaError(
            f"Only {len(radiomics)} radiomics columns were found; at least {top_k} are required."
        )
    if len(deep) < top_k:
        raise DataSchemaError(
            f"Only {len(deep)} deep-feature columns were found; at least {top_k} are required."
        )
