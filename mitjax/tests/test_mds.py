"""mitjax/io/mds.py (plan Task 7b): MDS reader against the oracle, write/read round trips, meta files of both widths.

Oracle: tutorial_baroclinic_gyre reads `bathy.bin` (data: bathyFile; readBinaryPrec not set, so precFloat32 =
set_defaults.F:355) into R_low on 2 x 2 tiles of 31 x 31, and R_low is dumped at G00_geometry (lane A's probe run
27826873). INI_DEPTHS keeps the file values there (all points equal, measured); on global_ocean.90x40x15 (32-bit,
36 tiles under exch2) INI_DEPTHS/INI_MASKS_ETC change 194 of 3600 points (hFac rounding), so it is checked by count.
"""

import numpy as np
import pytest

from mitjax import paths
from mitjax.io import mds
from mitjax.io.dump import DumpSet


def _r_low(exp, layout, olx, oly):
    run = paths.REFERENCE_RUNS / exp / "input" / "job27826873-jdon"
    ds = DumpSet(run / "dumps")
    rl = ds.field(ds.iterations()[0], "G00_geometry", "R_low")
    return run / "rundir", rl[:, :, oly:oly + layout.sNy, olx:olx + layout.sNx]


def test_mds_reader(tmp_path):
    """(1) bathy.bin of the baroclinic gyre == the oracle's R_low on every point, bitwise; global_ocean: 3406 of 3600
    points equal (measured; the rest are adjusted by the model). Controls on the same file: little-endian bytes, 64-bit,
    and swapped tile offsets each fail. (2) Global and tiled files, 32 and 64 bit, 2 records x 3 levels: read ==
    written (64) and == float64(float32(x)) (32), -0 and NaN preserved, irecord selects the record, tiled == global.
    (3) .meta: nrecords written as I10 (master) and as I5 (c66g) both read back; reading master's line with I5 gives
    12 for 1234567 (the width rule bites); dimList and dataprec parsed."""
    lay = mds.MdsLayout(31, 31, 2, 2)
    rundir, r_low = _r_low("tutorial_baroclinic_gyre", lay, 2, 2)
    got = mds.read_field(rundir / "bathy.bin", 32, lay)
    assert np.array_equal(got, r_low)
    raw = (rundir / "bathy.bin").read_bytes()
    little = np.frombuffer(raw, "<f4").astype(np.float64).reshape(1, 62, 62)
    assert not np.array_equal(little[:, :31, :31], r_low[0])
    with pytest.raises(ValueError, match="beyond the end"):
        mds.read_field(rundir / "bathy.bin", 64, lay)
    swapped = got[[0, 2, 1, 3]]                                             # tiles (2,1) and (1,2) exchanged
    assert not np.array_equal(swapped, r_low)
    lay_go = mds.MdsLayout(10, 10, 9, 4)
    rundir_go, r_low_go = _r_low("global_ocean.90x40x15", lay_go, 3, 3)
    assert int(np.sum(mds.read_field(rundir_go / "bathymetry.bin", 32, lay_go) == r_low_go)) == 3406

    rng = np.random.default_rng(0)
    small = mds.MdsLayout(5, 4, 3, 2)
    vals = rng.standard_normal((2, 3, small.Ny, small.Nx)) * 1e3
    vals[0, 0, 0, :3] = (-0.0, np.nan, 1e-300)
    for prec in (32, 64):
        for tiled in (False, True):
            name = tmp_path / f"f{prec}{'t' if tiled else 'g'}"
            mds.write_field(name, vals, prec, small, tiled=tiled)
            want = vals if prec == 64 else vals.astype(np.float32).astype(np.float64)
            for rec in (1, 2):
                r = mds.read_field(name, prec, small, nNz=3, irecord=rec)
                w = np.stack([want[rec - 1][:, (bj - 1) * 4:bj * 4, (bi - 1) * 5:bi * 5] for bi, bj in small.tiles()])
                assert np.array_equal(r.view(np.int64), w.view(np.int64)), (prec, tiled, rec)
            meta = mds.read_meta(name.with_suffix(".meta") if not tiled else mds.tile_file(name, 1, 1).with_suffix(".meta"))
            assert meta["filePrec"] == prec and meta["nRecords"] == 2 and meta["nDims"] == 3
            assert meta["dimList"][0][0] == small.Nx and meta["dimList"][2] == [3, 1, 3]
        with pytest.raises(FileExistsError):
            mds.write_field(tmp_path / f"f{prec}g", vals, prec, small)
    placed = mds.into_tiles(mds.read_field(tmp_path / "f64g", 64, small, nNz=3), 2, 1)
    assert placed.shape == (6, 3, 6, 9) and np.all(placed[:, :, 0, :] == 0)

    for width, nrec in ((10, 1234567), (5, 12345)):
        p = tmp_path / f"w{width}.meta"
        mds.write_meta(p, [[90, 1, 90], [40, 1, 40]], 64, nrec, timeStepNumber=72, nrecords_width=width)
        m = mds.read_meta(p)
        assert (m["nRecords"], m["fileIter"], m["filePrec"]) == (nrec, 72, 64), (width, m)
        assert m["dimList"] == [[90, 1, 90], [40, 1, 40]]
    line = [ln for ln in (tmp_path / "w10.meta").read_text().splitlines() if "nrecords" in ln][0]
    assert len(line) >= 25 and mds._int(line[14:], 5) == 12 and mds._int(line[14:], 10) == 1234567
