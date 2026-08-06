#!/usr/bin/env python
"""Evaluate frozen predictions for TCIA internal validation."""

from __future__ import annotations

import sys

from acrin_survival.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
