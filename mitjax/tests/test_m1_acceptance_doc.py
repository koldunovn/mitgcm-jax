"""docs/M1_ACCEPTANCE.md cannot drift from the runs it reports (plan Task 18, GO lane session 6).

For every row of the acceptance table the test re-derives, from the files the acceptance job wrote (`$MJX_RUNS/
m1_acceptance/<job>-<exp>-<input>/run/cli/rundir`, scripts/m1_acceptance.py), the digits columns with
tools/testreport_jax.py (ours vs results/, the yardstick = our gfortran oracle's STDOUT vs results/, ours vs the
oracle), the %MON verdict (every %MON record and MONITOR banner equal to the oracle STDOUT's, and the number of
blocks), the pickup verdict (every pickup file of the oracle run directory byte for byte, no extra file) and the P=N
cell (the job's result.json: no differing carry leaf, equal per-step outputs). Text only, seconds. Negative controls
(measured 2026-10-02): seven planted cell errors (a digit in each of the three digits columns and in the optim row,
a %MON block count, a pickup file count, a P=N step count) each fail their row.
The variant without a forward results/ file (tutorial_global_oce_optim/input_ad: results/ holds only output_adm.txt)
is compared with the forward check list against the oracle only; its adjoint line is test_r5_adjoint.py's.
"""

import filecmp
import importlib.util as iu
import json
import re

import pytest

from mitjax import paths

DOC = paths.REPO / "docs" / "M1_ACCEPTANCE.md"
BEGIN, END = "<!-- m1-table -->", "<!-- /m1-table -->"


def _trj():
    spec = iu.spec_from_file_location("_trj", paths.REPO / "tools" / "testreport_jax.py")
    trj = iu.module_from_spec(spec)
    spec.loader.exec_module(trj)
    return trj


def _rows():
    text = DOC.read_text()
    body = text[text.index(BEGIN) + len(BEGIN):text.index(END)]
    rows = []
    for ln in body.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != 10 or cells[0] in ("variant", "") or set(cells[0]) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def _digits(cell):
    return None if cell == "n/a" else [99 if d == "--" else int(d) for d in cell.split()]


def _oracle_stdout(exp, inp):
    """monitor_gate.Oracle(exp, inp).stdout_path without loading the dump set."""
    from mitjax.tests.monitor_gate import rr
    return rr.run_top(rr.find_run(exp, inp, "jdon")) / "rundir" / "output.txt"


def _mon_diffs(ours, oracle):
    """advect_gate.run_verdict's %MON selection (every %MON record and MONITOR banner, in order) on the files."""
    a = ours.read_bytes().decode("latin-1").split("\n")
    b = oracle.read_bytes().decode("latin-1").split("\n")
    k0 = next(n for n, r in enumerate(b) if "Begin MONITOR dynamic field statistics" in r) - 1
    n = 0
    for sel in (lambda r: "%MON " in r, lambda r: "MONITOR dynamic field statistics" in r):
        x, y = [r for r in a if sel(r)], [r for r in b[k0:] if sel(r)]
        n += sum(p != q for p, q in zip(x, y)) + abs(len(x) - len(y)) + (0 if x else 1)
    return n, sum("%MON time_tsnumber" in r for r in a)


ROWS = _rows()


def test_table_present():
    assert len(ROWS) == 9, [r[0] for r in ROWS]


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_row_rederived(row):
    variant, job, names, dres, dyard, dorc, mon, pk, pn, _ = row
    exp, inp = variant.split("/")
    jdir = paths.RUNS / "m1_acceptance" / f"{job}-{exp}-{inp}"
    rundir = jdir / "run" / "cli" / "rundir"
    ours, orc = rundir / "output.txt", _oracle_stdout(exp, inp)
    trj = _trj()
    if dres == "n/a":                       # no forward results/ file: the forward check list vs the oracle only
        assert dyard == "n/a"
        so = trj.compare(str(ours), exp, inp, kind=trj.KIND_FWD, reference=str(orc))
        got = {"names": [v.name for v in so.run.variables], "orc": [v.digits for v in so.run.variables]}
        assert got == {"names": names.split(), "orc": _digits(dorc)}
    else:
        r = trj.compare(str(ours), exp, inp)
        y = trj.compare(str(orc), exp, inp)
        so = trj.compare(str(ours), exp, inp, reference=str(orc))
        got = [[v.name for v in t.run.variables] for t in (r, y, so)]
        assert got == [names.split()] * 3
        assert [[v.digits for v in t.run.variables] for t in (r, y, so)] == [_digits(dres), _digits(dyard),
                                                                            _digits(dorc)]
    nd, nblocks = _mon_diffs(ours, orc)
    assert mon == f"{'yes' if nd == 0 else 'NO'} ({nblocks} blocks)"
    theirs = sorted(p.name for p in orc.parent.iterdir() if p.name.startswith("pickup"))
    same = all((rundir / f).exists() and filecmp.cmp(rundir / f, orc.parent / f, shallow=False) for f in theirs)
    extra = [p.name for p in rundir.iterdir() if p.name.startswith("pickup") and p.name not in theirs]
    assert pk == f"{'yes' if same and not extra else 'NO'} ({len(theirs)} files)"
    if pn.startswith("n/a"):
        from mitjax.tests import monitor_gate as mg
        sz = mg.experiment(exp, inp).cfg.size
        assert sz.nSx * sz.nSy * sz.nPx * sz.nPy == 1                  # one tile: nothing to shard
    else:
        m = re.fullmatch(r"P=(\d+) (yes|NO), (\d+) steps \(job (\d+)\)", pn)
        assert m, pn
        res = json.loads(next((paths.RUNS / "m1_acceptance").glob(f"{m[4]}-{exp}-{inp}")).joinpath(
            "run", "result.json").read_text())["pn"]
        ok = res["differing_leaves"] == 0 and res["outputs_equal"]
        assert (res["P"], "yes" if ok else "NO", res["steps"]) == (int(m[1]), m[2], int(m[3]))
