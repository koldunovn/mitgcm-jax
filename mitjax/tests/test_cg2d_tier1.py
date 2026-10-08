"""Tier-1 gate of the pressure solver (M1 sub-lane CG2D): the barotropic gyre's INI_CG2D operator and CG2D, bitwise
against the oracle, and the implicit rule's forward value bitwise equal to the literal solve (the extended gates are in
test_cg2d.py)."""

import jax

from mitjax.tests import cg2d_gate as G


def test_cg2d_barotropic_gyre_operator_replay_and_rule_bitwise():
    v = ("tutorial_barotropic_gyre", "input")
    c, _ = G.run_ini_cg2d(*v)
    assert float(c.cg2dNorm) == G.stdout_cg2dnorm(*v)[1]
    ds, _, _ = G.oracle(*v)
    it = ds.iterations()[0]
    res = G.compare_operator(c, *v, it, "C01_cg2d_inputs")
    assert not {k: r for k, r in res.items() if any(r[1:])}, res
    lit = G.replay_cg2d(*v, it, solve="literal")
    res = G.compare_solution(*v, it, lit)
    assert not {k: r for k, r in res.items() if any(r[1:])}, res
    rule = G.replay_cg2d(*v, it, solve="rule")
    for a, b in zip(jax.tree.leaves(lit), jax.tree.leaves(rule)):
        assert G.same_bits(a, b)
