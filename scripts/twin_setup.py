"""Tier-3 twin set-up (plan Task 18; lane B): a shadow MITgcm tree under $MJX_RUNS/twin/MITgcm with the twin variants of
global_ocean.90x40x15, for `python -m mitjax run` and reference/make_rundir.py (--upstream). Never writes into the
upstream clone; refuses an existing shadow tree (nothing is overwritten or removed).

    python scripts/twin_setup.py

The tree: every top-level entry of the clone (and .git, for the commit check) and every other experiment as a symlink;
verification/global_ocean.90x40x15/ a real directory with its entries as symlinks plus
  input.twin/   data (input/data with nTimeSteps=3600 = 10 model years, monitorFreq=2592000 = 30 days,
                dumpFreq=0), data.pkg (useDiagnostics=.FALSE.: output only), data.exch2.mpi -> ../input/ (testreport's
                MPI linkdata takes *.mpi from the first directory only; the retile build reads data.exch2);
  input.twin3/  data, data.pkg -> ../input.twin/, pickup.0000036000 = input's with Theta(i=46,j=21,k=1) one ulp up.
Jobs (docs/M1_ACCEPTANCE.md section 6): F1 = the plain binary on input.twin, F2 = the retile binary on input.twin
(make_rundir --mpi), F3 = the plain binary on input.twin3, JAX = `python -m mitjax run <tree>/verification/
global_ocean.90x40x15 --variant input.twin`; scripts/twin_compare.py compares them.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitjax import paths  # noqa: E402

EXP = "global_ocean.90x40x15"


def main():
    up = paths.UPSTREAM
    top = paths.RUNS / "twin" / "MITgcm"
    if top.exists():
        raise SystemExit(f"{top} exists (nothing is overwritten)")
    (top / "verification").mkdir(parents=True)
    for e in sorted(os.listdir(up)):
        if e != "verification" and e not in (".github", ".gitignore", ".readthedocs.yml"):
            (top / e).symlink_to(up / e)
    for v in sorted(os.listdir(up / "verification")):
        if v != EXP:
            (top / "verification" / v).symlink_to(up / "verification" / v)
    src, exp = up / "verification" / EXP, top / "verification" / EXP
    exp.mkdir()
    for v in sorted(os.listdir(src)):
        (exp / v).symlink_to(src / v)
    t = exp / "input.twin"
    t.mkdir()
    data = (src / "input" / "data").read_text()
    for a, b in ((" nTimeSteps=10,\n", " nTimeSteps=3600,\n"), (" dumpFreq=   311040000.,\n", ""),
                 (" dumpFreq=   864000.,\n", " dumpFreq=   0.,\n"), (" monitorFreq=1.,\n", " monitorFreq=2592000.,\n")):
        if data.count(a) != 1:
            raise SystemExit(f"input/data: {a.strip()!r} not found once")
        data = data.replace(a, b)
    (t / "data").write_text("# tier-3 twin (lane B, plan Task 18): input/data with nTimeSteps=3600 (10 years), "
                            "monitorFreq=30 days, dumpFreq=0\n" + data)
    pkg = (src / "input" / "data.pkg").read_text()
    if pkg.count(" useDiagnostics=.TRUE.,\n") != 1:
        raise SystemExit("input/data.pkg: useDiagnostics=.TRUE. not found once")
    (t / "data.pkg").write_text("# tier-3 twin (lane B): diagnostics output off (output only; lane A's diagoff runs)\n"
                                + pkg.replace(" useDiagnostics=.TRUE.,\n", " useDiagnostics=.FALSE.,\n"))
    (t / "data.exch2.mpi").symlink_to("../input/data.exch2.mpi")
    t3 = exp / "input.twin3"
    t3.mkdir()
    (t3 / "data").symlink_to("../input.twin/data")
    (t3 / "data.pkg").symlink_to("../input.twin/data.pkg")
    a = np.fromfile(src / "input" / "pickup.0000036000", ">f8").reshape(138, 40, 90)
    Nr, j, i = 15, 20, 45                     # Theta is the third Nr-record block (pickup meta fldList)
    old = a[2 * Nr, j, i]
    a[2 * Nr, j, i] = np.nextafter(old, np.inf)
    a.astype(">f8").tofile(t3 / "pickup.0000036000")
    (t3 / "README.twin3").write_text(f"tier-3 twin F3 (lane B): input/pickup.0000036000 with Theta(i=46,j=21,k=1) one "
                                     f"ulp up ({old!r} -> {a[2 * Nr, j, i]!r}); data, data.pkg as input.twin\n")
    print(top)


if __name__ == "__main__":
    main()
