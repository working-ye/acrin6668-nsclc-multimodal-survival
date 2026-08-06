from __future__ import annotations

import pytest

from acrin_survival.audit import (
    LeakageError,
    assert_disjoint_identity_values,
    assert_disjoint_ids,
    normalize_patient_id,
)


def test_identifier_normalization_is_leakage_conservative() -> None:
    assert normalize_patient_id(" 001.0 ") == "1"
    assert normalize_patient_id("001.00") == "1"
    assert normalize_patient_id("1e0") == "1"
    assert normalize_patient_id("+001") == "1"
    assert normalize_patient_id("TCIA-ABC ") == "tcia-abc"


def test_one_overlapping_patient_stops_validation() -> None:
    with pytest.raises(LeakageError, match="overlap"):
        assert_disjoint_ids(["001", "TCIA-ABC"], ["1", "tcia-other"])


def test_image_fingerprint_collision_stops_validation() -> None:
    development = {"image_sha256": ["abc", "def"]}
    validation = {"image_sha256": ["XYZ", " ABC "]}
    with pytest.raises(LeakageError, match="identity collision"):
        assert_disjoint_identity_values(development, validation)
