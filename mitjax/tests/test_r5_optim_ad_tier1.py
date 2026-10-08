"""R5 tutorial_global_oce_optim/input_ad: the variant's one tier-1 smoke test (plan Task 16; helpers in r5_gate.py):
step 1 of the code_ad forward, teacher-forced from the oracle's S00_begin dump, through THERMODYNAMICS -- every
dumped field of S02 ... S05 bitwise (periodic forcing preload, control, freezing, JMD95Z, IVDC, GM/Redi tensor and
fluxes, TEMP/SALT_INTEGRATE). Becomes the whole-run smoke (global fc, %MON) once the CD scheme is in the step."""

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()


def test_r5_step1_front_bitwise():
    from mitjax.tests import r5_gate as r5
    m = r5.model("zero")
    _, probes = m.run_front(0, m.front_fn(r5.UNTIL))
    r = r5.compare_front(m, 0, probes)
    assert not r5.missing(m, 0, r), r5.missing(m, 0, r)
    assert not r5.bad(r), r5.bad(r)
