"""Censoring-aware validation metrics for frozen survival models."""

from __future__ import annotations

import itertools
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from .data import make_survival_outcome


def harrell_cindex(frame: pd.DataFrame, risk: np.ndarray) -> float:
    risk = np.asarray(risk, dtype=float)
    if (
        len(frame) < 2
        or int(frame["event"].sum()) == 0
        or not np.isfinite(risk).all()
        or np.std(risk) <= 1e-12
    ):
        return float("nan")
    outcome = make_survival_outcome(frame)
    try:
        return float(concordance_index_censored(outcome["event"], outcome["time"], risk)[0])
    except (ValueError, ZeroDivisionError):
        return float("nan")


def uno_cindex(
    development_outcome: np.ndarray,
    validation: pd.DataFrame,
    risk: np.ndarray,
    horizon_days: float,
) -> tuple[float, float, str]:
    validation_outcome = make_survival_outcome(validation)
    support_upper = min(
        float(np.max(development_outcome["time"])),
        float(np.max(validation_outcome["time"])),
    )
    tau = float(np.nextafter(float(horizon_days), np.inf))
    if support_upper < tau:
        return float("nan"), float("nan"), "insufficient_followup"
    try:
        value = float(
            concordance_index_ipcw(
                development_outcome, validation_outcome, np.asarray(risk), tau=tau
            )[0]
        )
    except (ValueError, ZeroDivisionError):
        return float("nan"), tau, "calculation_failed"
    if not np.isfinite(value):
        return float("nan"), tau, "calculation_failed"
    return value, tau, "ok"


def bootstrap_harrell_cindex(
    frame: pd.DataFrame, risk: np.ndarray, repeats: int, seed: int
) -> tuple[float, float, int]:
    if repeats <= 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    values = []
    risk = np.asarray(risk, dtype=float)
    for _ in range(repeats):
        indices = rng.integers(0, len(frame), len(frame))
        value = harrell_cindex(frame.iloc[indices], risk[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan"), 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def _adjust_evaluation_time(
    requested: float, development_outcome: np.ndarray, validation_outcome: np.ndarray
) -> float | None:
    lower = max(
        float(np.min(development_outcome["time"])),
        float(np.min(validation_outcome["time"])),
    )
    upper = min(
        float(np.max(development_outcome["time"])),
        float(np.max(validation_outcome["time"])),
    )
    requested = float(requested)
    if requested > upper and not np.isclose(requested, upper):
        return None
    adjusted = np.nextafter(upper, 0.0) if requested >= upper else requested
    if adjusted <= lower:
        return None
    return adjusted


def dynamic_auc_table(
    development_outcome: np.ndarray,
    validation: pd.DataFrame,
    risk: np.ndarray,
    requested_times: Sequence[int],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    validation_outcome = make_survival_outcome(validation)
    risk = np.asarray(risk, dtype=float)
    censoring_km = KaplanMeierFitter().fit(
        development_outcome["time"],
        event_observed=~development_outcome["event"],
    )
    rows = []
    for offset, requested in enumerate(requested_times):
        evaluation_time = _adjust_evaluation_time(
            requested, development_outcome, validation_outcome
        )
        auc = float("nan")
        bootstrap_values = []
        cases = int(
            (validation_outcome["event"] & (validation_outcome["time"] <= requested)).sum()
        )
        controls = int((validation_outcome["time"] > requested).sum())
        censored_before = int(
            ((~validation_outcome["event"]) & (validation_outcome["time"] <= requested)).sum()
        )
        number_at_risk = int((validation["time_days"] >= float(requested)).sum())
        censoring_survival = float(censoring_km.predict(float(requested)))
        status = "ok"
        if evaluation_time is None:
            status = "insufficient_followup"
        elif cases == 0 or controls == 0:
            status = "insufficient_cases_or_controls"
        elif not np.isfinite(censoring_survival) or censoring_survival <= 0:
            status = "censoring_positivity_failure"

        if status == "ok":
            try:
                auc_values, _ = cumulative_dynamic_auc(
                    development_outcome,
                    validation_outcome,
                    risk,
                    np.asarray([evaluation_time]),
                )
                auc = float(auc_values[0])
            except (ValueError, ZeroDivisionError):
                status = "calculation_failed"
            if not np.isfinite(auc):
                status = "calculation_failed"

            if status == "ok" and repeats > 0:
                rng = np.random.default_rng(seed + offset)
                for _ in range(repeats):
                    indices = rng.integers(0, len(validation), len(validation))
                    try:
                        values, _ = cumulative_dynamic_auc(
                            development_outcome,
                            validation_outcome[indices],
                            risk[indices],
                            np.asarray([evaluation_time]),
                        )
                        if np.isfinite(values[0]):
                            bootstrap_values.append(float(values[0]))
                    except (ValueError, ZeroDivisionError):
                        continue
        rows.append(
            {
                "requested_time_days": int(requested),
                "evaluation_time_days": evaluation_time,
                "status": status,
                "case_count": cases,
                "control_count": controls,
                "censored_before_time": censored_before,
                "number_at_risk": number_at_risk,
                "development_censoring_survival": censoring_survival,
                "cumulative_dynamic_auc": auc,
                "ci95_low": (
                    float(np.quantile(bootstrap_values, 0.025))
                    if bootstrap_values
                    else float("nan")
                ),
                "ci95_high": (
                    float(np.quantile(bootstrap_values, 0.975))
                    if bootstrap_values
                    else float("nan")
                ),
                "bootstrap_valid_repeats": len(bootstrap_values),
            }
        )
    return pd.DataFrame(rows)


def brier_table(
    development_outcome: np.ndarray,
    validation: pd.DataFrame,
    survival_probabilities: np.ndarray,
    requested_times: Sequence[int],
) -> tuple[pd.DataFrame, float, str]:
    validation_outcome = make_survival_outcome(validation)
    adjusted_times = []
    columns = []
    status_rows = []
    for index, requested in enumerate(requested_times):
        adjusted = _adjust_evaluation_time(requested, development_outcome, validation_outcome)
        if adjusted is not None:
            adjusted_times.append(adjusted)
            columns.append(index)
        status_rows.append(
            {
                "requested_time_days": int(requested),
                "evaluation_time_days": adjusted,
                "status": "ok" if adjusted is not None else "insufficient_followup",
                "brier_score": float("nan"),
            }
        )
    if not adjusted_times:
        return pd.DataFrame(status_rows), float("nan"), "insufficient_followup"

    estimates = np.asarray(survival_probabilities, dtype=float)[:, columns]
    try:
        _, scores = brier_score(
            development_outcome,
            validation_outcome,
            estimates,
            np.asarray(adjusted_times),
        )
    except (ValueError, ZeroDivisionError):
        scores = np.repeat(float("nan"), len(adjusted_times))

    for score_index, column_index in enumerate(columns):
        score = float(scores[score_index])
        status_rows[column_index]["brier_score"] = score
        if not np.isfinite(score):
            status_rows[column_index]["status"] = "calculation_failed"

    integrated = float("nan")
    integrated_status = "insufficient_followup"
    if len(adjusted_times) == len(requested_times) and len(adjusted_times) >= 2:
        integrated_status = "calculation_failed"
        try:
            integrated = float(
                integrated_brier_score(
                    development_outcome,
                    validation_outcome,
                    estimates,
                    np.asarray(adjusted_times),
                )
            )
        except (ValueError, ZeroDivisionError):
            pass
        if np.isfinite(integrated):
            integrated_status = "ok"
    elif len(adjusted_times) == len(requested_times):
        integrated_status = "insufficient_time_grid"
    return pd.DataFrame(status_rows), integrated, integrated_status


def risk_group_summary(
    validation: pd.DataFrame, risk: np.ndarray, cutoff: float
) -> tuple[pd.DataFrame, float]:
    work = validation[["time_days", "event"]].copy()
    work["risk_score"] = np.asarray(risk, dtype=float)
    work["risk_group"] = np.where(work["risk_score"] >= float(cutoff), "high", "low")
    low = work[work["risk_group"].eq("low")]
    high = work[work["risk_group"].eq("high")]
    logrank_p = float("nan")
    if not low.empty and not high.empty and int(work["event"].sum()) > 0:
        try:
            logrank_p = float(
                logrank_test(
                    low["time_days"],
                    high["time_days"],
                    event_observed_A=low["event"],
                    event_observed_B=high["event"],
                ).p_value
            )
        except (ValueError, ZeroDivisionError):
            pass

    rows = []
    for group in ("low", "high"):
        subset = work[work["risk_group"].eq(group)]
        median = float("nan")
        if not subset.empty:
            fitted = KaplanMeierFitter().fit(
                subset["time_days"], event_observed=subset["event"]
            )
            value = fitted.median_survival_time_
            median = float(value) if np.isfinite(value) else float("nan")
        rows.append(
            {
                "risk_group": group,
                "n": len(subset),
                "event_count": int(subset["event"].sum()),
                "median_os_days": median,
                "logrank_p_high_vs_low": logrank_p,
            }
        )
    return pd.DataFrame(rows), logrank_p


def calibration_table(
    validation: pd.DataFrame,
    predicted_survival: np.ndarray,
    horizon_days: int,
    bins: int,
    minimum_bin_size: int,
) -> pd.DataFrame:
    columns = [
        "calibration_bin",
        "n",
        "event_count",
        "n_at_risk_at_horizon",
        "mean_predicted_risk",
        "km_observed_risk",
        "km_observed_risk_ci95_low",
        "km_observed_risk_ci95_high",
        "status",
    ]
    predicted_risk = 1.0 - np.asarray(predicted_survival, dtype=float)
    work = validation[["time_days", "event"]].copy()
    if float(work["time_days"].max()) < float(horizon_days):
        return pd.DataFrame(columns=columns)
    work["predicted_risk"] = np.clip(predicted_risk, 0.0, 1.0)
    unique = int(work["predicted_risk"].nunique())
    q = min(int(bins), unique, max(1, len(work) // max(1, int(minimum_bin_size))))
    if q < 2:
        return pd.DataFrame(columns=columns)
    work["calibration_bin"] = pd.qcut(
        work["predicted_risk"], q=q, labels=False, duplicates="drop"
    )
    if int(work["calibration_bin"].nunique(dropna=True)) < 2:
        return pd.DataFrame(columns=columns)
    rows = []
    for bin_index, subset in work.groupby("calibration_bin", sort=True):
        status = "ok"
        observed_risk = float("nan")
        observed_ci_low = float("nan")
        observed_ci_high = float("nan")
        n_at_risk = int((subset["time_days"] >= float(horizon_days)).sum())
        if len(subset) < int(minimum_bin_size):
            status = "insufficient_bin_size"
        elif float(subset["time_days"].max()) < float(horizon_days):
            status = "insufficient_followup"
        else:
            km = KaplanMeierFitter().fit(
                subset["time_days"], event_observed=subset["event"]
            )
            observed_risk = 1.0 - float(km.predict(float(horizon_days)))
            confidence = km.confidence_interval_survival_function_
            eligible = confidence.loc[confidence.index <= float(horizon_days)]
            if not eligible.empty:
                survival_low, survival_high = eligible.iloc[-1].tolist()
                observed_ci_low = 1.0 - float(survival_high)
                observed_ci_high = 1.0 - float(survival_low)
        rows.append(
            {
                "calibration_bin": int(bin_index) + 1,
                "n": len(subset),
                "event_count": int(subset["event"].sum()),
                "n_at_risk_at_horizon": n_at_risk,
                "mean_predicted_risk": float(subset["predicted_risk"].mean()),
                "km_observed_risk": observed_risk,
                "km_observed_risk_ci95_low": observed_ci_low,
                "km_observed_risk_ci95_high": observed_ci_high,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def paired_cindex_comparisons(
    validation: pd.DataFrame,
    risks: Dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Use one shared patient-bootstrap index matrix for all model contrasts."""

    names = list(risks)
    point = {name: harrell_cindex(validation, risks[name]) for name in names}
    rng = np.random.default_rng(seed)
    indices_matrix = (
        rng.integers(0, len(validation), size=(repeats, len(validation)))
        if repeats > 0
        else np.empty((0, len(validation)), dtype=int)
    )
    bootstrap = {name: [] for name in names}
    for indices in indices_matrix:
        sample = validation.iloc[indices]
        for name in names:
            bootstrap[name].append(harrell_cindex(sample, np.asarray(risks[name])[indices]))

    columns = [
        "reference_model",
        "comparison_model",
        "delta_harrell_cindex",
        "ci95_low",
        "ci95_high",
        "bootstrap_valid_repeats",
    ]
    rows = []
    for reference, comparison in itertools.combinations(names, 2):
        differences = np.asarray(bootstrap[comparison]) - np.asarray(bootstrap[reference])
        finite = differences[np.isfinite(differences)]
        rows.append(
            {
                "reference_model": reference,
                "comparison_model": comparison,
                "delta_harrell_cindex": point[comparison] - point[reference],
                "ci95_low": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
                "ci95_high": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
                "bootstrap_valid_repeats": int(finite.size),
            }
        )
    return pd.DataFrame(rows, columns=columns)
