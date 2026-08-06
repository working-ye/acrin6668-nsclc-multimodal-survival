#!/usr/bin/env python
"""Fail if a patient or image identity crosses the development/validation split."""

from __future__ import annotations

import sys

from acrin_survival.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["audit-split", *sys.argv[1:]]))
