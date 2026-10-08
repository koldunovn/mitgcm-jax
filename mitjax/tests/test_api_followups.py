"""Lane APIF (docs plan 20261006 S5): the API follow-ups that run the model, against the driver path each wraps and the
Fortran's own files (oracle runs under $MJX_REFERENCE_RUNS; the Fortran comparisons skip without them).

* sharded forward (contract 2): exp.run(out, devices=P) == exp.run(out) bit for bit -- output.txt and pickups byte for
  byte, every field of Run.fields -- on global_ocean.90x40x15/input_ad (exch1, 4 tiles, P = 3: two tiles per device
  and padding) and adjustment.cs-32x32x1/input (pkg/exch2 cube, 48 tiles, P = 4).
  Control: one ulp planted into tile 4's theta on its device (drivers/run.sharded_step wrapped) -> differs.
* Run.global_field (contract 4) vs the Fortran's own output at the same iteration: T.0000000010 (90x40x15, exch1
  [k, Ny, Nx]) and Eta.0000000024 (cube, [face, j, i] from the 192 x 32 global file, faces side by side) within one
  float32 ulp of the file's value (the files are float32). Control: the tiles in another order -> fails.
* exp.grid() (contract 5): every GRID.h array the Model's own grid holds, bit for bit (the driver path), and the grid
  files WRITE_GRID writes (XC, YC, XG, YG, RAC, RAW, RAS, RAZ, DXC, DYC, DXG, DYG, DXF, DYF, DXV, DYU, AngleCS,
  AngleSN, Depth, hFacC/W/S, RC, RF, DRC, DRF) within one float32 ulp; no run directory is made for a plain variant.
  Control: the tiles in another order -> fails.
* Run.fields' sea ice (contract 3): 1D_ocean_ice_column/input: AREA, HEFF, HSNOW, UICE, VICE present, bit for bit
  the driver's carry[4]["seaice"], AREA and HEFF within one float32 ulp of the Fortran's AREA/HEFF.0000000010.
  Control: AREA and HEFF swapped -> fails.
* LSR default (contract 10) on global_ocean.cs32x15/input_ad.seaice (no SEAICE_LSR_ADJOINT_ITER): lsr_derivative=None
  gives the Arrays of lsr_derivative="sweeps" in both modes (same program), the build's own choice is "forward_only".
  Control: the default removed -> "forward_only".
About 30 min on a CPU node (6 forward runs, 4 Model set-ups).
"""

import dataclasses

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax import paths  # noqa: E402

GO90 = ("global_ocean.90x40x15", "input_ad", 3, "theta", "T.0000000010")
CUBE = ("adjustment.cs-32x32x1", "input", 4, "etaN", "Eta.0000000024")
CASES = [GO90, CUBE]
GRID_FILES = {"XC": "xC", "YC": "yC", "XG": "xG", "YG": "yG", "RAC": "rA", "RAW": "rAw", "RAS": "rAs", "RAZ": "rAz",
              "DXC": "dxC", "DYC": "dyC", "DXG": "dxG", "DYG": "dyG", "DXF": "dxF", "DYF": "dyF", "DXV": "dxV",
              "DYU": "dyU", "AngleCS": "angleCosC", "AngleSN": "angleSinC", "hFacC": "hFacC", "hFacW": "hFacW",
              "hFacS": "hFacS", "RC": "rC", "RF": "rF", "DRC": "drC", "DRF": "drF"}     # write_grid.F:81-131


def _out(tag):
    from mitjax.tests import advect_gate as ag
    return ag.out_dir(f"apif-{tag}")


def _exp(name):
    return paths.UPSTREAM / "verification" / name


def _release():
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


def _oracle(exp, variant):
    """The oracle run directory of exp/variant (jaxdump off), or skip."""
    try:
        root = paths.REFERENCE_RUNS
    except Exception:                                                    # noqa: BLE001 (unset without the oracle)
        root = None
    if root is None or not root.exists():
        pytest.skip("no MJX_REFERENCE_RUNS (oracle)")
    hits = sorted((root / exp / variant).glob("job*-jdoff/rundir"))
    if not hits:
        pytest.skip(f"no oracle run of {exp}/{variant}")
    return hits[0]


def _fortran(rundir, name, tm):
    """A global file the Fortran wrote (MDS, big-endian), as [(k,) Ny, Nx] (exch1) or [face, (k,) j, i] (exch2:
    the faces side by side in x, the 6 x 32 x 32 cube's 192 x 32 layout)."""
    from mitjax.io import mds
    meta = mds.read_meta(rundir / f"{name}.meta")
    dims = [d[0] for d in meta["dimList"]]
    a = np.fromfile(rundir / f"{name}.data", dtype=">f4" if meta["filePrec"] == 32 else ">f8")
    a = a.reshape(dims[::-1]).astype(np.float64)
    if sum(d > 1 for d in dims) <= 1 and name.upper() == name:          # WRITE_GLVEC_RS (RC, RF, DRC, DRF): a vector
        a = a.ravel()
    if a.ndim == 1 or tm.kind == "exch1":
        return a, meta["filePrec"]
    nf, ny, nx = tm.shape
    assert a.shape[-2:] == (ny, nf * nx), a.shape
    return np.stack([a[..., f * nx:(f + 1) * nx] for f in range(nf)]), meta["filePrec"]


def _within_ulp(ours, ref, prec):
    """|ours rounded to the file's precision - file| <= one ulp of the file's value at every point."""
    o = np.asarray(ours, np.float32 if prec == 32 else np.float64)
    r = np.asarray(ref, o.dtype)
    if o.shape != r.shape and np.squeeze(o).shape == np.squeeze(r).shape:  # Nr = 1: the file has no level axis
        o, r = np.squeeze(o), np.squeeze(r)
    if o.shape != r.shape:
        return False, f"shape {o.shape} vs {r.shape}"
    bad = np.abs(o.astype(np.float64) - r.astype(np.float64)) > np.spacing(np.abs(r)).astype(np.float64)
    bad &= ~(np.isnan(o) & np.isnan(r))
    return not bad.any(), f"{int(bad.sum())} of {r.size} points off by more than one ulp; " \
        f"{int(np.sum(o == r))} equal"


def _tree_equal(a, b):
    """The same pytree structure (static options included) and every leaf bit for bit."""
    import jax
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    return ta == tb and len(la) == len(lb) and all(_bits_equal(x, y) for x, y in zip(la, lb))


def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def runs(request):
    import mitjax
    name, variant, P, field, _ = request.param
    exp = mitjax.load(_exp(name), variant=variant)
    r1 = exp.run(_out(f"{name}-p1"))
    _release()
    rP = exp.run(_out(f"{name}-p{P}"), devices=P)
    _release()
    return dict(case=request.param, exp=exp, r1=r1, rP=rP)


def _same_runs(r1, rP):
    """(ok, info): output.txt and pickups byte for byte, every field bit for bit."""
    info = []
    ok = r1.output.read_bytes() == rP.output.read_bytes()
    info.append(f"output.txt equal {ok}")
    pk1 = sorted(p.name for p in r1.rundir.iterdir() if p.name.startswith("pickup"))
    pkP = sorted(p.name for p in rP.rundir.iterdir() if p.name.startswith("pickup"))
    same_pk = pk1 == pkP and all((r1.rundir / n).read_bytes() == (rP.rundir / n).read_bytes() for n in pk1)
    ok &= same_pk
    info.append(f"pickups {pk1} equal {same_pk}")
    diff = sorted(n for n in r1.fields if n not in rP.fields or not _bits_equal(r1.fields[n], rP.fields[n]))
    ok &= not diff and set(r1.fields) == set(rP.fields)
    info.append(f"fields differing {diff}")
    return ok, "; ".join(info)


def test_sharded_forward_bitwise(runs):
    r1, rP = runs["r1"], runs["rP"]
    assert r1.devices == 1 and rP.devices == runs["case"][2]
    ok, info = _same_runs(r1, rP)
    print(runs["case"][:3], info)
    assert ok, info


def test_sharded_forward_planted_ulp(monkeypatch):
    """Control: one ulp in tile 4's theta (an interior wet point) on its device -> the P = 3 run differs."""
    import jax.numpy as jnp

    import mitjax
    from mitjax.drivers import run as R
    name, variant, P, _, _ = GO90
    orig = R.sharded_step

    def planted(m, sh, carry):
        st = carry[0]
        th = np.array(st.theta.data)
        sz = m.cfg.size
        # the first wet interior point of tile 4's top level (its centre is land in 90x40x15: gate 27950211)
        k = 0
        wet = np.argwhere(th[3, k, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] != 0)
        assert len(wet), "tile 4 has no wet point at k = 1"
        j, i = int(wet[0][0]) + sz.OLy, int(wet[0][1]) + sz.OLx
        th[3, k, j, i] = np.nextafter(th[3, k, j, i], np.inf)
        st = st.replace(theta=type(st.theta)(jnp.asarray(th), st.theta.name, tiled=st.theta.tiled,
                                             _dims=st.theta.dims))
        return orig(m, sh, (st,) + tuple(carry[1:]))
    monkeypatch.setattr(R, "sharded_step", planted)
    exp = mitjax.load(_exp(name), variant=variant)
    r1 = exp.run(_out(f"{name}-ctl-p1"))
    rP = exp.run(_out(f"{name}-ctl-p{P}"), devices=P)
    _release()
    ok, info = _same_runs(r1, rP)
    print("planted ulp:", info)
    assert not ok


def _assert_global(tm_field, ref, prec, label):
    ok, info = _within_ulp(tm_field, ref, prec)
    print(label, info)
    return ok


def _wrong_order(tm):
    """The planted control: the tiles placed in the reverse order."""
    return dataclasses.replace(tm, origins=tm.origins[::-1])


def test_global_field_vs_fortran(runs):
    name, variant, _, field, fname = runs["case"]
    rd = _oracle(name, variant)
    r1 = runs["r1"]
    tm = r1.tilemap
    ref, prec = _fortran(rd, fname, tm)
    ours = r1.global_field(field)
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    assert _assert_global(ours, ref, prec, f"{name} {field} vs {fname}")
    assert not _assert_global(_wrong_order(tm).global_field(r1.fields[field]), ref, prec, "planted tile order")


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def grids(request):
    import mitjax
    name, variant = request.param[:2]
    exp = mitjax.load(_exp(name), variant=variant)
    before = sorted(p.name for p in (paths.RUNS).iterdir()) if paths.RUNS.exists() else []
    g = exp.grid()
    after = sorted(p.name for p in (paths.RUNS).iterdir()) if paths.RUNS.exists() else []
    m = exp.model(_out(f"{name}-grid-model"))
    _release()
    return dict(case=request.param, exp=exp, g=g, model_grid=m.grid, runs_unchanged=before == after)


def test_grid_bitwise_vs_model(grids):
    from mitjax.farray import FArray
    g, mg = grids["g"], grids["model_grid"]
    names = sorted(n for n in mg.names() if isinstance(getattr(mg, n), FArray))
    missing = sorted(set(names) - set(g))
    diff = sorted(n for n in g if n in names and not _bits_equal(g[n], np.asarray(getattr(mg, n).data)))
    print(grids["case"][0], len(g), "grid arrays;", "missing", missing, "differ", diff)
    assert grids["runs_unchanged"]                       # no run directory made under $MJX_RUNS
    assert {"xC", "yC", "xG", "yG", "dxC", "dyC", "dxG", "dyG", "rA", "rC", "rF", "drC", "drF", "hFacC", "hFacW",
            "hFacS", "maskC", "R_low", "Ro_surf"} <= set(g), sorted(g)
    assert not diff, diff                # (missing: what the Model adds after INI_CORI, e.g. INI_LINEAR_PHISURF)


def test_grid_vs_fortran_files(grids):
    name, variant = grids["case"][:2]
    rd = _oracle(name, variant)
    g = grids["g"]
    tm = g.tilemap
    checked, bad, planted_bad = [], [], []
    for fname, n in GRID_FILES.items():
        if not (rd / f"{fname}.data").exists() or n not in g:
            continue
        ref, prec = _fortran(rd, fname, tm)
        ours = g.global_field(n)
        if ours.ndim == 1:
            ours = ours[:ref.size]
        ok, info = _within_ulp(ours, ref, prec)
        checked.append(fname)
        if not ok:
            bad.append((fname, info))
        if ours.ndim > 1 and not _within_ulp(_wrong_order(tm).global_field(g[n]), ref, prec)[0]:
            planted_bad.append(fname)
    depth = g.global_field("Ro_surf") - g.global_field("R_low")                     # write_grid.F:57-75
    ref, prec = _fortran(rd, "Depth", tm)
    ok, info = _within_ulp(depth, ref, prec)
    checked.append("Depth")
    if not ok:
        bad.append(("Depth", info))
    print(name, "checked", checked, "bad", bad, "planted order fails for", planted_bad)
    assert {"XC", "YC", "RAC", "hFacC", "DRF"} <= set(checked), checked
    assert not bad, bad
    assert {"XC", "YC", "RAC", "hFacC"} <= set(planted_bad), planted_bad


# -------------------------------------------------------------------------------------------- sea-ice fields

SEAICE = ("AREA", "HEFF", "HSNOW", "UICE", "VICE")


def test_seaice_fields():
    import mitjax
    from mitjax.drivers.run import forward
    exp = mitjax.load(_exp("1D_ocean_ice_column"), variant="input")
    r = exp.run(_out("col-seaice"))
    res = forward(exp.model(_out("col-seaice-driver")))
    _release()
    assert set(SEAICE) <= set(r.fields), sorted(r.fields)
    for n in SEAICE:
        assert _bits_equal(r.fields[n], np.asarray(res.carry[4]["seaice"][n].data)), n
    rd = _oracle("1D_ocean_ice_column", "input")
    tm = r.tilemap
    oks, planted = [], []
    for n, other in (("AREA", "HEFF"), ("HEFF", "AREA")):
        from mitjax.io import mds
        p = rd / f"{n}.0000000010.001.001"
        meta = mds.read_meta(p.with_name(p.name + ".meta"))
        ref = np.fromfile(p.with_name(p.name + ".data"), ">f4" if meta["filePrec"] == 32 else ">f8")
        ref = ref.reshape(tm.shape).astype(np.float64)
        oks.append(_within_ulp(r.global_field(n), ref, meta["filePrec"]))
        planted.append(_within_ulp(r.global_field(other), ref, meta["filePrec"])[0])
        print(n, float(r.global_field(n).ravel()[0]), float(ref.ravel()[0]), oks[-1][1])
    assert all(o for o, _ in oks), oks
    assert not any(planted), planted                                   # control: AREA and HEFF swapped


# ------------------------------------------------------------------------------------------------ LSR default

def test_lsr_default_cs32_seaice(monkeypatch):
    import mitjax
    from mitjax import api
    from mitjax.ad.modes import lsr_derivative
    exp = mitjax.load(_exp("global_ocean.cs32x15"), variant="input_ad.seaice")
    m = exp.model(_out("cs32-seaice-lsr"))
    sp0 = m.arrays.pkc["sp"]
    assert lsr_derivative(m.cfg, sp0) == "forward_only"                  # SEAICE_LSR_ADJOINT_ITER undefined
    for mode in ("run", "exact"):
        a_def = exp._mode_arrays(m, m.arrays, mode, None)
        a_sw = exp._mode_arrays(m, m.arrays, mode, "sweeps")
        assert a_def.pkc["sp"].mjx_lsr_derivative == "sweeps"
        assert lsr_derivative(m.cfg, a_def.pkc["sp"]) == "sweeps"
        assert _tree_equal(a_def, a_sw), mode                         # the same Arrays: the same program
    monkeypatch.setattr(api, "_lsr_default_applies", lambda m, sp, choice: False)        # planted: no default
    assert lsr_derivative(m.cfg, exp._mode_arrays(m, m.arrays, "exact", None).pkc["sp"]) == "forward_only"
    _release()
