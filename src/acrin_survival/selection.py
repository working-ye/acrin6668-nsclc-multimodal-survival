"""Outcome-aware feature screening confined to development data or CV training folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from .data import CohortSchema, make_survival_outcome, numeric_feature_frame


@dataclass
class FeatureSelectionResult:
    modality: str
    selected: List[str]
    ranking: pd.DataFrame


def _safe_cindex(frame: pd.DataFrame, risk: np.ndarray) -> float:
    values = np.asarray(risk, dtype=float)
    if (
        len(frame) < 2
        or int(frame["event"].sum()) == 0
        or not np.isfinite(values).all()
        or np.std(values) <= 1e-12
    ):
        return float("nan")
    outcome = make_survival_outcome(frame)
    try:
        return float(concordance_index_censored(outcome["event"], outcome["time"], values)[0])
    except (ValueError, ZeroDivisionError):
        return float("nan")


def select_top_features(
    training: pd.DataFrame,
    candidates: Iterable[str],
    schema: CohortSchema,
    settings: Dict[str, Any],
    modality: str,
) -> FeatureSelectionResult:
    """Rank by univariable C-index and remove highly correlated features.

    This function must only receive a development dataset or an internal-CV
    training fold. The independent validation dataset is never an argument.
    """

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError(f"No candidate features supplied for modality={modality}.")
    numeric = numeric_feature_frame(training, candidate_list, schema)
    missing_limit = float(settings["maximum_missing_fraction"])
    variance_floor = float(settings["minimum_variance"])
    correlation_limit = float(settings["maximum_absolute_correlation"])
    top_k = int(settings["top_k_per_imaging_modality"])

    rows = []
    usable: List[str] = []
    for feature in candidate_list:
        values = numeric[feature]
        missing_fraction = float(values.isna().mean())
        variance = float(values.var(skipna=True)) if values.notna().sum() > 1 else 0.0
        status = "candidate"
        if missing_fraction > missing_limit:
            status = "drop_missing"
        elif not np.isfinite(variance) or variance <= variance_floor:
            status = "drop_variance"

        cindex = float("nan")
        if status == "candidate":
            median = values.median()
            if not np.isfinite(median):
                status = "drop_no_finite_median"
            else:
                filled = values.fillna(median).to_numpy(dtype=float)
                forward = _safe_cindex(training, filled)
                reverse = _safe_cindex(training, -filled)
                finite = [value for value in (forward, reverse) if np.isfinite(value)]
                cindex = max(finite) if finite else float("nan")
                if not np.isfinite(cindex):
                    status = "drop_no_comparable_pairs"
                else:
                    usable.append(feature)

        rows.append(
            {
                "modality": modality,
                "feature": feature,
                "missing_fraction": missing_fraction,
                "variance": variance,
                "univariable_cindex": cindex,
                "status": status,
            }
        )

    ranking = pd.DataFrame(rows)
    ranking["rank_score"] = ranking["univariable_cindex"] - 0.5
    ranking = ranking.sort_values(
        ["rank_score", "feature"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    selected: List[str] = []
    medians = numeric[usable].median() if usable else pd.Series(dtype=float)
    for feature in ranking.loc[ranking["status"].eq("candidate"), "feature"]:
        current = numeric[feature].fillna(medians[feature]).to_numpy(dtype=float)
        too_correlated = False
        for kept in selected:
            other = numeric[kept].fillna(medians[kept]).to_numpy(dtype=float)
            correlation = np.corrcoef(current, other)[0, 1]
            if np.isfinite(correlation) and abs(correlation) >= correlation_limit:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(feature)
        if len(selected) == top_k:
            break

    if len(selected) != top_k:
        raise ValueError(
            f"Feature selection retained {len(selected)} {modality} features; "
            f"the configuration requires exactly {top_k}."
        )
    ranking["selected"] = ranking["feature"].isin(selected)
    ranking["selection_rank"] = ranking["feature"].map(
        {feature: index + 1 for index, feature in enumerate(selected)}
    )
    return FeatureSelectionResult(modality=modality, selected=selected, ranking=ranking)
