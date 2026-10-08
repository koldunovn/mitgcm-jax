#!/usr/bin/env python3
"""Shim: the implementation moved to mitjax/make_rundir.py (docs plan 20261006 Task 5, R4 of docs/PORTABILITY.md:
nothing in mitjax/ loads reference/ at run time). `python reference/make_rundir.py ...` and loading this file by path
keep working: the module is loaded by path (stdlib only, no jax) and its names are re-exported here. Monkeypatch the
implementation module (mitjax.make_rundir), not this one."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_mjx_make_rundir_impl",
                                               Path(__file__).resolve().parents[1] / "mitjax" / "make_rundir.py")
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
globals().update({k: v for k, v in vars(_impl).items() if not (k.startswith("__") and k.endswith("__"))})

if __name__ == "__main__":
    sys.exit(main())  # noqa: F821 (re-exported)
