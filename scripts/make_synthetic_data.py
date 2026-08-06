#!/usr/bin/env python
"""Generate fake, non-clinical inputs for the reproducibility smoke test."""

from __future__ import annotations

import sys

from acrin_survival.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["make-synthetic", *sys.argv[1:]]))
