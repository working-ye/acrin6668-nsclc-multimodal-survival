from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acrin_survival.config import load_config
from acrin_survival.data import CohortSchema, DataSchemaError, read_cohort, read_feature_table
from acrin_survival.synthetic import make_synthetic_cohort


def test_horizon_censoring_and_prediction_read_are_separated(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "demo.yaml"
    schema = CohortSchema.from_config(load_config(config_path))
    frame = make_synthetic_cohort("TEST", 12, 11)
    frame.loc[0, "overall_survival_days"] = 1200.0
    frame.loc[0, "death_event_overall"] = 1
    frame.loc[1, "overall_survival_days"] = 800.0
    frame.loc[1, "death_event_overall"] = 0
    source = tmp_path / "cohort.csv"
    frame.to_csv(source, index=False)

    cohort = read_cohort(source, schema)
    assert cohort.loc[0, "time_days"] == 1095.0
    assert cohort.loc[0, "event"] == 0
    assert cohort.loc[1, "time_days"] == 800.0
    assert cohort.loc[1, "event"] == 0
    assert cohort.loc[0, "metric_time_days"] > 1095.0

    required_features = [
        *schema.clinical_pet_features,
        "rad_tumor_feature_001",
        "dl_swin_feature_0001",
    ]
    prediction_input = read_feature_table(source, schema, required_features)
    assert schema.time_column not in prediction_input.columns
    assert schema.event_column not in prediction_input.columns


def test_infinite_survival_time_is_rejected(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "demo.yaml"
    schema = CohortSchema.from_config(load_config(config_path))
    frame = make_synthetic_cohort("INF", 12, 12)
    frame.loc[0, "overall_survival_days"] = np.inf
    source = tmp_path / "infinite.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(DataSchemaError, match="finite"):
        read_cohort(source, schema)
