"""mitjax.xla_flags.flag_set (lane SHARDGRAD, session 2): the gate set by default and for "gate"; "gpu" (the gate set
without --xla_disable_hlo_passes=algsimp) is refused since 2026-10-08 with the reason it was retired (XLA compiled a
wrong 4-GPU program with algsimp on: jobs 27980864, 27981958); anything else refused. Pure Python (seconds; one
test: the tier-1 test budget is 100)."""

import pytest

from mitjax import xla_flags as X


def test_flag_sets(monkeypatch):
    monkeypatch.delenv("MJX_XLA_FLAG_SET", raising=False)
    assert X.flag_set() == X.GATE_FLAGS
    monkeypatch.setenv("MJX_XLA_FLAG_SET", "gate")
    assert X.flag_set() == X.GATE_FLAGS
    monkeypatch.setenv("MJX_XLA_FLAG_SET", "gpu")
    with pytest.raises(ValueError, match="retired"):
        X.flag_set()
    with pytest.raises(ValueError, match="retired"):
        X.set_api_xla_flags()                       # the API refuses it too (it reads flag_set)
    monkeypatch.setenv("MJX_XLA_FLAG_SET", "fast")
    with pytest.raises(ValueError):
        X.flag_set()
