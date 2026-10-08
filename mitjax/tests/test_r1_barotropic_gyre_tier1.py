"""R1 tutorial_barotropic_gyre: the variant's one tier-1 smoke test (plan Task 12), through the run path of
`python -m mitjax run` (mitjax/drivers/run.py `run`): the 10-step run writes <out>/rundir/output.txt and the
end-of-run pickup. Every record of output.txt occurs in the oracle STDOUT; per kind and in order they equal the
oracle's records: CG2D's Sum(rhs) lines, SOLVE_FOR_PRESSURE's cg2d_* lines (testreport's PS = cg2d_init_res), the
%MON records of the 11 MONITOR blocks and their Begin/End banners, %CHECKPOINT; tools/testreport_jax.py digits vs
results/ >= the yardstick (the oracle's own digits) and vs the oracle STDOUT = the full digits (16; 22 for zero
series); the pickup.ckptA files (data and meta) equal the oracle's byte for byte."""

import filecmp
import importlib.util as iu
from pathlib import Path

from mitjax import paths

KINDS = {"sum": lambda r: r.startswith(" cg2d: Sum(rhs)"), "cg2d": lambda r: "      cg2d_" in r,
         "mon": lambda r: "%MON " in r, "banner": lambda r: "MONITOR dynamic field statistics" in r,
         "ckpt": lambda r: "%CHECKPOINT" in r}


def test_run_cli_checklist_monitor_pickup():
    from mitjax.drivers.run import run
    from mitjax.tests import monitor_gate as mg
    from mitjax.tests.test_r1_driver import EXP, EXP_DIR, INP, out_dir
    out = run(EXP_DIR, INP, out_dir("tier1_cli"))
    o = mg.oracle(EXP, INP)
    ours = Path(out).read_text().rstrip("\n").split("\n")
    theirs = set(o.raw)
    assert all(r in theirs for r in ours), [r for r in ours if r not in theirs][:5]
    # the oracle's records from the banner before the first dynamic MONITOR block on (INI_PARMS / CONFIG_SUMMARY /
    # the grid statistics %MON XC_max ... of INITIALISE_FIXED come before it and are not ported)
    k0 = next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    for kind, sel in KINDS.items():
        a, b = [r for r in ours if sel(r)], [r for r in o.raw[k0:] if sel(r)]
        assert a and a == b, (kind, len(a), len(b))
    assert sum("%MON time_tsnumber" in r for r in ours) == 11
    spec = iu.spec_from_file_location("_trj", paths.REPO / "tools" / "testreport_jax.py")
    trj = iu.module_from_spec(spec)
    spec.loader.exec_module(trj)
    ours_res = trj.compare(str(out), EXP, INP)
    yard = trj.compare(str(o.stdout_path), EXP, INP)
    vs_oracle = trj.compare(str(out), EXP, INP, reference=str(o.stdout_path))
    for a, y, s in zip(ours_res.run.variables, yard.run.variables, vs_oracle.run.variables):
        assert a.name == y.name == s.name
        assert a.digits >= y.digits, (a.name, a.digits, y.digits)
        assert s.digits >= 16, (s.name, s.digits)
    for suf in ("data", "meta"):
        f = f"pickup.ckptA.001.001.{suf}"
        assert filecmp.cmp(Path(out).parent / f, o.stdout_path.parent / f, shallow=False), f
