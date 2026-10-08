"""Planted control files for the ctrl/cost gates of tutorial_global_oce_optim/input_ad (lane ctrl, plan Task 16 prep).

    python mitjax/tests/ctrl_xx_files.py SOURCE_TOP OUT_DIR [--seed 20261001] [--amp 50]

Writes `xx_qnet.0000000000.<bi>.<bj>.{data,meta}` (the first-guess control file CTRL_MAP_INI_GENTIM2D reads,
ctrl_map_ini_gentim2d.F:166-168, 223-225 at 63cdc0b) with seeded uniform values in [-amp, amp) W/m^2 at every point
of every tile. The layout is the one the forward-only build itself writes at initialisation (CTRL_INIT_CTRLVAR ->
CTRL_SET_GLOBFLD_XY, ctrl_init_ctrlvar.F:143-151, globalFiles=.FALSE.): one file per tile, sNx*sNy big-endian float64
(ctrlprec=64: CTRL_SET_PREC_32 undefined in code_ad/CTRL_OPTIONS.h, ctrl_readparms.F:229), one record, no halo.
The meta files are byte copies of the source run's own `xx_qnet.0000000000.*.meta` (they hold no file name), and the
data files must have the source's size. Used with `make_rundir.py --copy` and `--set data.ctrl:CTRL_NML:doInitXX=
.FALSE.` (otherwise CTRL_INIT_CTRLVAR overwrites the file with zeros: doInitXX defaults to .TRUE.,
ctrl_readparms.F:192, ctrl_init_ctrlvar.F:150). OUT_DIR must not exist; nothing is overwritten. Writes SHA256SUMS.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

SNX, SNY, NSX, NSY = 45, 20, 2, 2      # verification/tutorial_global_oce_optim/code_ad/SIZE.h


def planted_values(seed, amp):
    """[bj, bi, j, i] float64 values (bi, bj 0-based) in [-amp, amp)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-amp, amp, size=(NSY, NSX, SNY, SNX))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source_top", help="oracle run (holds rundir/xx_qnet.0000000000.*.meta)")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--amp", type=float, default=50.0)
    a = ap.parse_args(argv)
    src = Path(a.source_top) / "rundir"
    out = Path(a.out_dir)
    if out.exists():
        sys.exit(f"{out} exists (nothing is overwritten)")
    out.mkdir(parents=True)
    vals = planted_values(a.seed, a.amp)
    sums = []
    for bj in range(1, NSY + 1):
        for bi in range(1, NSX + 1):
            stem = f"xx_qnet.0000000000.{bi:03d}.{bj:03d}"
            want = (src / f"{stem}.data").stat().st_size
            data = vals[bj - 1, bi - 1].astype(">f8").tobytes()
            if len(data) != want:
                sys.exit(f"{stem}.data: {len(data)} bytes, the source has {want}")
            (out / f"{stem}.data").write_bytes(data)
            (out / f"{stem}.meta").write_bytes((src / f"{stem}.meta").read_bytes())
            for name in (f"{stem}.data", f"{stem}.meta"):
                sums.append(f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (out / "PROVENANCE.txt").write_text(f"seed {a.seed} amp {a.amp} source {src}\n")
    print("\n".join(sums))


if __name__ == "__main__":
    main()
