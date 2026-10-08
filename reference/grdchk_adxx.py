#!/usr/bin/env python3
"""Adjoint control files for the gradient check of the forward-only tutorial_global_oce_optim/code_ad build
(plan Task 3b item 4, option (i), Nikolay 2026-10-01); since lane A session 6 also for the M2 code_ad variants
(global_ocean.90x40x15/input_ad*: genarr3d/genarr2d controls through CTRL_SET_GLOBFLD_XYZ/_XY, the same routine for
xx and adxx, ctrl_init_ctrlvar.F:131-151; useSingleCpuIO=.TRUE. there, so ONE global file; tutorial_tracer_adjsens:
xx_ptr1, genarr3d, tiled).

    grdchk_adxx.py SOURCE_TOP OUT_DIR [--name xx_qnet] [--random SEED]

Why: a forward-only build (no ALLOW_ADJOINT_RUN) runs THE_MAIN_LOOP and the cost, then GRDCHK_MAIN
(the_model_main.F:733-737) stops in GRDCHK_GETADXX (grdchk_main.F:274), which reads the adjoint gradient file
`ad<name>.<optimcycle as I10.10>` (grdchk_getadxx.F:80-81, yadmark='ad' ctrl_readparms.F:443) that only an
adjoint build writes. That value is used only in the printed ratio (grdchk_main.F:434-437, 492-496, 514); the
finite-difference gradient comes from the +-grdchk_eps forward runs (THE_MAIN_LOOP at grdchk_main.F:359, 400).

What an adjoint build writes, from the Fortran at 63cdc0b:
  * CTRL_INIT_CTRLVAR (ctrl_init_ctrlvar.F:143-151) calls CTRL_SET_GLOBFLD_XY for fname(2) = 'ad'//name//'.'//
    optimcycle (ctrl_set_fname.F; only #if ALLOW_ADJOINT_RUN or ALLOW_TANGENTLINEAR_RUN) and, in every build with
    doInitXX and optimcycle = 0, for fname(1) = name//'.'//optimcycle -- the SAME routine with the SAME varRecs and
    ctrlprec. CTRL_SET_GLOBFLD_XY (ctrl_set_globfld_xy.F) writes varRecs records of zeros with WRITE_REC_3D_RL,
    i.e. MDS_WRITE_FIELD with globalFile = globalFiles (default .FALSE.: one file per tile `.<iG>.<jG>.data`, each
    with its `.meta`), precision ctrlprec, big-endian (-fconvert=big-endian), no halo.
  * The adjoint sweep then adds the gradient into it: ADACTIVE_READ_XY -> ACTIVE_READ_3D_RL in REVERSE_SIMULATION
    (active_file_ad.F:105-113, active_file_control.F:143-171) reads the file and writes it back with
    MDS_WRITE_FIELD(..., globalFile=.FALSE., ..., jRec=-|iRec|): tiled, same precision, same records, and no new meta
    file (jrecord < 0, mdsio_write_field.F:163-164), so the meta written at initialisation stays.
So the file an adjoint build leaves has exactly the layout (tiles, records, precision, endianness, meta) of the
zero file `<name>.<optimcycle>.*` that the forward-only run itself writes at initialisation through the same routine.
This script writes `ad<name>.<optimcycle>.<iG>.<jG>.{data,meta}` with that layout: the data bytes are generated here
(zeros, or with --random seeded uniform values in [-1e-5, 1e-5) at every point, packed big-endian in the source's
precision) and the zero version is asserted byte-equal to the forward-only run's own `<name>` files; the meta files are
byte copies of the source's meta files (they hold no file name). Also writes SHA256SUMS and PROVENANCE.txt.
OUT_DIR must not exist (nothing is overwritten). Stdlib only.

Planted first-guess controls (lane A session 8; the GOADK gates S02 + control = S03 for the global_ocean.90x40x15
code_ad family, as job27829159-ctrlxx for optim): `--prefix ''` writes `<name>.<cycle>.*` itself (the file
CTRL_MAP_INI_GENARR reads with doInitXX=.FALSE.) with the same layout, `--random SEED --amp A` seeded uniform values in
[-A, A) at every point (land included: CTRL_MAP_GENARR* adds the control times the mask).
"""

import argparse
import hashlib
import random
import re
import struct
import sys
from pathlib import Path


def meta_info(text):
    prec = re.search(r"dataprec = \[ '(float32|float64)' \]", text)
    nrec = re.search(r"nrecords = \[\s*(\d+) \]", text)
    if not prec or not nrec:
        raise SystemExit(f"unexpected meta file:\n{text}")
    return prec.group(1), int(nrec.group(1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source_top", help="run directory of the forward-only build (holds <name>.<cycle>.*.data)")
    ap.add_argument("out_dir")
    ap.add_argument("--name", default="xx_qnet")
    ap.add_argument("--cycle", type=int, default=0, help="optimcycle (default 0)")
    ap.add_argument("--random", type=int, metavar="SEED", help="random values instead of zeros")
    ap.add_argument("--amp", type=float, default=1e-5, help="random values in [-amp, amp) (default 1e-5)")
    ap.add_argument("--subdir", default="", help="directory of the control files inside rundir (data.ctrl ctrlDir, "
                    "e.g. ctrl_variables for lab_sea/input_ad*; default: rundir itself)")
    ap.add_argument("--prefix", default="ad", help="prefix of the written files (default 'ad'; '' = the control "
                    "file itself)")
    a = ap.parse_args(argv)
    src = Path(a.source_top) / "rundir" / a.subdir if a.subdir else Path(a.source_top) / "rundir"
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists (nothing is overwritten)")
    stem = f"{a.name}.{a.cycle:010d}"
    datas = sorted(src.glob(f"{stem}.[0-9][0-9][0-9].[0-9][0-9][0-9].data"))
    glob_file = src / f"{stem}.data"
    if datas and glob_file.exists():
        raise SystemExit(f"both tiled files {stem}.<iG>.<jG>.data and a global file in {src}")
    if not datas:
        # useSingleCpuIO=.TRUE. (global_ocean.90x40x15/input_ad*): MDS_WRITE_FIELD writes ONE global file for both
        # the initial control and the adjoint build's adxx file (mdsio_write_field.F, the useSingleCPUIO branch),
        # and the forward-only run reads the global ad<name>.<cycle>.data (its MDS_READ_FIELD message)
        if not glob_file.exists():
            raise SystemExit(f"expected tiled files {stem}.<iG>.<jG>.data or one global file {stem}.data in {src}")
        datas = [glob_file]
    rng = random.Random(a.random) if a.random is not None else None
    out.mkdir(parents=True)
    sums, prov = [], [f"source {src.resolve()}", f"name {a.name} cycle {a.cycle}",
                      "values " + ("zeros" if rng is None else f"random.Random({a.random}).uniform({-a.amp!r}, "
                                                             f"{a.amp!r})"), f"prefix {a.prefix!r}"]
    for d in datas:
        tile = d.name[len(stem) + 1:-len(".data")]          # '' for the global file
        meta = d.with_name(d.name[:-len(".data")] + ".meta")
        mtext = meta.read_text()
        prec, nrec = meta_info(mtext)
        size = 8 if prec == "float64" else 4
        raw = d.read_bytes()
        n = len(raw) // size
        if n * size != len(raw) or n % nrec:
            raise SystemExit(f"{d}: {len(raw)} bytes do not hold {nrec} records of {prec}")
        fmt = ">" + ("d" if size == 8 else "f") * n
        zeros = struct.pack(fmt, *([0.0] * n))
        if raw != zeros:
            raise SystemExit(f"{d} is not all zero (+0.0): not the initial control file")
        body = zeros if rng is None else struct.pack(fmt, *(rng.uniform(-a.amp, a.amp) for _ in range(n)))
        for suffix, content in ((".data", body), (".meta", mtext.encode())):
            target = out / (f"{a.prefix}{stem}.{tile}{suffix}" if tile else f"{a.prefix}{stem}{suffix}")
            target.write_bytes(content)
            sums.append(f"{hashlib.sha256(content).hexdigest()}  {target.name}")
        prov.append(f"{'tile ' + tile if tile else 'global file'}: {nrec} record(s) of {n // nrec} {prec} big-endian;"
                    f" meta copied from {meta.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (out / "PROVENANCE.txt").write_text("\n".join(prov) + "\n")
    print("\n".join(prov + sums))
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
