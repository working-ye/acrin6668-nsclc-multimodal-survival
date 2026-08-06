#!/usr/bin/env python
"""Apply frozen models without loading survival outcomes."""

from __future__ import annotations

import sys

from acrin_survival.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["predict", *sys.argv[1:]]))
