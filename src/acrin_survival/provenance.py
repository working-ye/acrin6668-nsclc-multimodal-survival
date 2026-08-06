"""File provenance and run-environment utilities."""

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
from typing import Any


def normalize_patient_id(value: Any) -> str:
    """Canonicalize identifiers for prediction/outcome row matching."""

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
