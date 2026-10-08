"""mitjax.xla_flags.flag_set (lane SHARDGRAD, session 2): the gate set by default; under MJX_XLA_FLAG_SET=gpu (tier-2
GPU runs) the gate set without --xla_disable_hlo_passes=algsimp; anything else refused. Pure Python (seconds; one
test: the tier-1 test budget is 100)."""

import pytest

from mitjax import xla_flags as X


def test_flag_sets(monkeypatch):
    monkeypatch.delenv("MJX_XLA_FLAG_SET", raising=False)
    assert X.flag_set() == X.GATE_FLAGS
    monkeypatch.setenv("MJX_XLA_FLAG_SET", "gate")
    assert X.flag_set() == X.GATE_FLAGS
    monkeypatch.setenv("MJX_XLA_FLAG_SET", "gpu")
    fs = X.flag_set()
    assert set(fs) == set(X.GATE_FLAGS) - {"--xla_disable_hlo_passes=algsimp"}
    assert X.gate_xla_flags("", fs) == " ".join(fs)
    monkeypatch.setenv("MJX_XLA_FLAG_SET", "fast")
    with pytest.raises(ValueError):
        X.flag_set()
