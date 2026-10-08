#!/usr/bin/env python3
"""Outputs of the cube replay harness (plan Task 22; reference/replay_cube/code/the_main_loop.F), numpy only.

Files in the run directory (big-endian stream, the build's -fconvert=big-endian):
    cube_grid.bin  int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, nRec; nRec x (CHARACTER*16 name,
                   float64 field (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nSx, nSy))
    cube_exch.bin  the same header, float64 cbase, then nRec records as above: 'z<iz><kind>_u' / '_v' with iz = 0
                   the index-coded probe (comp*cbase + flat + 1, flat over (tile, j, i) of the padded arrays, tile =
                   W2_myTileList(bi,bj)), iz = 1..4 signed zeros (u,v) = (+0,+0), (-0,-0), (-0,+0), (+0,-0)
    cube_fill.bin  as cube_exch.bin, records 'z<iz>TR<d>s<s>_u', 'z<iz>UVRSs<s>_u/_v', 'z<iz>UVRLs<s>_u/_v',
                   'z<iz>AG<d>s<s>_u/_v' (s = 1: withSigns .TRUE., 2: .FALSE.; AG d = 1: fill4dirX .TRUE.)
A Fortran array A(i,j,bi,bj) is returned as [tile, j, i] (tile = (bj-1)*nSx + bi - 1, bi fastest).
"""

from pathlib import Path

import numpy as np

MAGIC, VERSION = 20261002, 1


def read(path, with_cbase):
    raw = Path(path).read_bytes()
    hdr = np.frombuffer(raw, ">i4", 9)
    if hdr[0] != MAGIC or hdr[1] != VERSION:
        raise ValueError(f"{path}: not a cube replay file (header {hdr[:2]})")
    sNx, sNy, OLx, OLy, nSx, nSy, nRec = (int(x) for x in hdr[2:])
    off = 36
    cbase = None
    if with_cbase:
        cbase = float(np.frombuffer(raw, ">f8", 1, off)[0])
        off += 8
    nx, ny = sNx + 2 * OLx, sNy + 2 * OLy
    n = nx * ny * nSx * nSy
    out = {}
    for _ in range(nRec):
        name = raw[off:off + 16].decode().strip()
        off += 16
        a = np.frombuffer(raw, ">f8", n, off).astype(np.float64)
        off += 8 * n
        out[name] = a.reshape(nSy * nSx, ny, nx)
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return {"sNx": sNx, "sNy": sNy, "OLx": OLx, "OLy": OLy, "nSx": nSx, "nSy": nSy, "cbase": cbase, "fields": out}
