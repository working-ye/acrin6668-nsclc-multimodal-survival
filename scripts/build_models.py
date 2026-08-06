#!/usr/bin/env python
"""Build frozen CP/CPR/CPRD/RD models from development data only."""

from __future__ import annotations

import sys

from acrin_survival.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build", *sys.argv[1:]]))
