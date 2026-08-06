"""Integrity, provenance, and leakage-control utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence


class LeakageError(RuntimeError):
    """Raised when development and validation patients are not strictly disjoint."""


def normalize_patient_id(value: Any) -> str:
    """Canonicalize identifiers conservatively for leakage detection."""

    if value is None:
        return ""
    text = str(value).strip().casefold()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        numeric = Decimal(text)
        if numeric.is_finite():
            if numeric == numeric.to_integral_value():
                return str(int(numeric))
            return format(numeric.normalize(), "f")
    except InvalidOperation:
        pass
    return text


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def patient_id_digest(ids: Iterable[Any]) -> str:
    """Return a deterministic set digest without writing raw identifiers to manifests."""

    normalized = sorted({normalize_patient_id(value) for value in ids})
    payload = "\n".join(value for value in normalized if value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_disjoint_ids(
    development_ids: Sequence[Any],
    validation_ids: Sequence[Any],
    *,
    development_label: str = "development",
    validation_label: str = "validation",
) -> None:
    """Abort before scoring if any patient appears in both datasets."""

    development = {normalize_patient_id(value) for value in development_ids}
    validation = {normalize_patient_id(value) for value in validation_ids}
    development.discard("")
    validation.discard("")
    overlap = sorted(development.intersection(validation))
    if overlap:
        preview = ", ".join(overlap[:5])
        suffix = " ..." if len(overlap) > 5 else ""
        raise LeakageError(
            f"Patient overlap detected between {development_label} and {validation_label}: "
            f"n={len(overlap)} ({preview}{suffix}). Validation was stopped before prediction."
        )


def assert_disjoint_identity_values(
    development: dict[str, Sequence[Any]], validation: dict[str, Sequence[Any]]
) -> None:
    """Detect renamed-patient leakage using optional image/series fingerprints."""

    for column in sorted(set(development).intersection(validation)):
        development_values = {
            str(value).strip().casefold()
            for value in development[column]
            if str(value).strip().casefold() not in {"", "nan", "none", "<na>"}
        }
        validation_values = {
            str(value).strip().casefold()
            for value in validation[column]
            if str(value).strip().casefold() not in {"", "nan", "none", "<na>"}
        }
        overlap = development_values.intersection(validation_values)
        if overlap:
            raise LeakageError(
                f"Cross-split identity collision in {column!r}: n={len(overlap)}. "
                "Validation was stopped before prediction."
            )


def git_commit(repo_root: str | Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in (
        "joblib",
        "lifelines",
        "matplotlib",
        "numpy",
        "pandas",
        "Pillow",
        "PyYAML",
        "scikit-learn",
        "scikit-survival",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "pid": os.getpid(),
    }


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
