"""R2 tutorial_baroclinic_gyre: the variant's one tier-1 smoke test (plan Task 13; helpers in r2_gate.py): the whole
10-step run at P=1 with tracer stepping; CG2D's Sum(rhs) line and SOLVE_FOR_PRESSURE's lines (incl. testreport's
`PS` = cg2d_init_res) of every step identical to the oracle STDOUT, the 11 %MON blocks identical character for
character, tools/testreport_jax.py digits vs results/ >= the yardstick (the oracle's own digits) and vs the oracle
STDOUT >= 13."""

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()


def test_whole_run_checklist_and_monitor():
    from mitjax.tests import cg2d_gate as cgg
    from mitjax.tests import monitor_gate as mg
    from mitjax.tests import r2_gate as r2
    m = r2.model()
    o, sums, mons, records, blocks, _ = r2.whole_run(m)
    osums, omons = cgg.stdout_solver_lines(m.exp, m.inp)
    assert sums == osums
    assert mons == omons                        # testreport's PS = cg2d_init_res, every step
    assert len(blocks) == 11
    for tsn, ours in blocks.items():
        assert not mg.compare(ours, o.block_lines(tsn)), tsn
    rows = r2.testreport_digits(m, o, records)
    assert len(rows) >= 17
    for name, ours, yard, vs_oracle in rows:
        assert ours >= yard, (name, ours, yard)
        assert vs_oracle >= 13, (name, vs_oracle)
