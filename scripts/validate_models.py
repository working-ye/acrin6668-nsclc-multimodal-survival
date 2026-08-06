#!/usr/bin/env python
"""Run the separated prediction/evaluation validation workflow."""

from __future__ import annotations

import sys

from acrin_survival.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *sys.argv[1:]]))
