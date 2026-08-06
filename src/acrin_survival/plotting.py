"""Journal-oriented validation figures using the project color standard."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter

MODEL_COLORS: Dict[str, str] = {
    "CP": "#4575B4",
    "CPR": "#91BFDB",
    "CPRD": "#FC8D59",
    "RD": "#D73027",
}
RISK_COLORS = {"low": "#4575B4", "high": "#D73027"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    try:
        fig.savefig(
            stem.with_suffix(".tiff"),
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except (OSError, ValueError, TypeError):
        pass
    plt.close(fig)


def plot_km_risk_groups(risk_data: pd.DataFrame, output_stem: Path) -> None:
    _style()
    models = list(dict.fromkeys(risk_data["model_name"].tolist()))
    columns = 2
    rows = math.ceil(len(models) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7.2, 3.1 * rows), squeeze=False)
    for index, model_name in enumerate(models):
        axis = axes.flat[index]
        subset = risk_data[risk_data["model_name"].eq(model_name)]
        for group in ("low", "high"):
            part = subset[subset["risk_group"].eq(group)]
            if part.empty:
                continue
            KaplanMeierFitter().fit(
                part["time_days"] / (365.25 / 12.0),
                event_observed=part["event"],
                label=f"{group.title()} risk (n={len(part)})",
            ).plot_survival_function(
                ax=axis,
                ci_show=False,
                color=RISK_COLORS[group],
                linewidth=1.5,
                censor_styles={"ms": 3, "marker": "+"},
            )
        axis.set_title(f"{model_name} model")
        axis.set_xlabel("Time (months)")
        axis.set_ylabel("Overall survival probability")
        axis.set_xlim(0, 36)
        axis.set_ylim(0, 1.02)
        axis.grid(False)
    for index in range(len(models), rows * columns):
        axes.flat[index].axis("off")
    fig.tight_layout()
    _save(fig, output_stem)


def plot_time_dependent_auc(auc_table: pd.DataFrame, output_stem: Path) -> None:
    if auc_table.empty:
        return
    _style()
    fig, axis = plt.subplots(figsize=(4.8, 3.6))
    for model_name, subset in auc_table.groupby("model_name", sort=False):
        subset = subset.dropna(subset=["cumulative_dynamic_auc"])
        if subset.empty:
            continue
        axis.plot(
            subset["requested_time_days"] / (365.25 / 12.0),
            subset["cumulative_dynamic_auc"],
            color=MODEL_COLORS.get(model_name, "#333333"),
            linewidth=1.5,
            marker="o",
            markersize=3.5,
            label=f"{model_name} model",
        )
    axis.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Time (months)")
    axis.set_ylabel("Cumulative/dynamic AUC")
    axis.set_xlim(11, 36.5)
    axis.set_ylim(0.0, 1.0)
    axis.grid(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output_stem)


def plot_calibration(calibration: pd.DataFrame, output_stem: Path) -> None:
    if calibration.empty:
        return
    _style()
    fig, axis = plt.subplots(figsize=(4.2, 4.0))
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.9, label="Ideal")
    for model_name, subset in calibration.groupby("model_name", sort=False):
        subset = subset[
            subset["status"].eq("ok")
            & subset["mean_predicted_risk"].notna()
            & subset["km_observed_risk"].notna()
        ]
        if subset.empty:
            continue
        axis.plot(
            subset["mean_predicted_risk"],
            subset["km_observed_risk"],
            color=MODEL_COLORS.get(model_name, "#333333"),
            linewidth=1.4,
            marker="o",
            markersize=3.5,
            label=f"{model_name} model",
        )
    axis.set_xlabel("Predicted 36-month event risk")
    axis.set_ylabel("Kaplan–Meier observed event risk")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output_stem)
