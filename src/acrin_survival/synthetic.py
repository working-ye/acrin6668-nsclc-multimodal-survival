"""Synthetic, non-clinical data for smoke tests and examples."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def make_synthetic_cohort(prefix: str, n: int, seed: int) -> pd.DataFrame:
    """Generate fake multimodal survival features with no relationship to real patients."""

    rng = np.random.default_rng(seed)
    mtv = rng.lognormal(mean=3.2, sigma=0.65, size=n)
    zubrod = rng.choice([0, 1, 2], size=n, p=[0.28, 0.56, 0.16])
    metastasis = rng.binomial(1, 0.22, size=n)

    radiomics = rng.normal(size=(n, 16))
    deep = rng.normal(size=(n, 14))
    latent = (
        0.24 * np.log1p(mtv)
        + 0.35 * zubrod
        + 0.48 * metastasis
        + 0.38 * radiomics[:, 0]
        - 0.30 * radiomics[:, 3]
        + 0.42 * deep[:, 1]
        - 0.27 * deep[:, 5]
    )
    event_time = rng.exponential(scale=780.0 * np.exp(-latent + latent.mean())) + 20.0
    censor_time = rng.uniform(420.0, 1600.0, size=n)
    observed_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    ids = [f"SYN-{prefix.upper()}-{index + 1:04d}" for index in range(n)]
    frame = pd.DataFrame(
        {
            "patient_id": ids,
            "overall_survival_days": observed_time,
            "death_event_overall": event,
            "le_pre_pet_total_metabolic_tumor_volume": mtv,
            "baseline_zubrod_performance_status": zubrod,
            "ss_metastatic_disease_present_on_pet_before_rt": np.where(
                metastasis == 1, "yes", "no"
            ),
            "study_instance_uid": [f"2.25.{seed}{index:05d}1" for index in range(n)],
            "series_instance_uid": [f"2.25.{seed}{index:05d}2" for index in range(n)],
            "image_sha256": [_fingerprint(f"{prefix}-image-{index}") for index in range(n)],
            "roi_sha256": [_fingerprint(f"{prefix}-roi-{index}") for index in range(n)],
        }
    )
    for index in range(8):
        frame[f"rad_tumor_feature_{index + 1:03d}"] = radiomics[:, index]
    for index in range(8, 16):
        frame[f"rad_peri5_feature_{index - 7:03d}"] = radiomics[:, index]
    for index in range(14):
        frame[f"dl_swin_feature_{index + 1:04d}"] = deep[:, index]

    # Reproducible, non-informative missingness below the configured threshold.
    for column in ("rad_tumor_feature_006", "dl_swin_feature_0012"):
        missing = rng.choice(n, size=max(1, n // 20), replace=False)
        frame.loc[missing, column] = np.nan
    return frame


def write_synthetic_data(
    output_dir: str | Path,
    n_development: int = 120,
    n_validation: int = 60,
    seed: int = 20260629,
) -> Tuple[Path, Path]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    development = make_synthetic_cohort("DEV", n_development, seed)
    validation = make_synthetic_cohort("VAL", n_validation, seed + 1)
    development_path = output / "development.csv"
    validation_path = output / "heldout_validation.csv"
    development.to_csv(development_path, index=False, encoding="utf-8")
    validation.to_csv(validation_path, index=False, encoding="utf-8")
    return development_path, validation_path
