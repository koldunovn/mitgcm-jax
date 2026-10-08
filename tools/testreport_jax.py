"""Shim: the implementation moved to mitjax/testreport_jax.py (docs plan 20261006 Task 5, R7 of docs/PORTABILITY.md:
the digit comparison is part of the package). `python tools/testreport_jax.py ...`, `from tools import testreport_jax`
and loading this file by path keep working: the module is loaded by path (stdlib only, no jax) and its names are
re-exported here. Monkeypatch the implementation module (mitjax.testreport_jax), not this one."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_mjx_testreport_jax_impl",
                                               Path(__file__).resolve().parents[1] / "mitjax" / "testreport_jax.py")
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
globals().update({k: v for k, v in vars(_impl).items() if not (k.startswith("__") and k.endswith("__"))})

if __name__ == "__main__":
    sys.exit(main())  # noqa: F821 (re-exported)
