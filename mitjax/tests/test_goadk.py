"""GOADK gates (M2 sub-lane GOADK, plan Task 24): leaf kernels of the global_ocean.90x40x15/input_ad family.

1. GM advective (bolus) form, dump gates on the four code_ad variants (input_ad, .kapgm, .kapredi, .bottomdrag; lane
   A's dumps-on runs, job 27832229): GMREDI_CALC_TENSOR with GMREDI_CALC_PSI_BOLUS, GMREDI_SLOPE_PSI ('dm95'), the
   GM_READ_K3D_REDI / GM_READ_K3D_GM fields and GM_ExtraDiag (Kuz, Kvz) at every dumped iteration (P05); the bolus
   velocity of GMREDI_RESIDUAL_FLOW (T01) at iterations 0 and 1 (its recip_hFacW/S are those UPDATE_SURF_DR(.TRUE.)
   sets after the momentum step, forward_step.F:852, i.e. the G00 grid of the next dumped iteration: iteration 2 has
   no next dump). Every point of every tile (halos, land): element equality, both finite, equal bit patterns.
2. Negative controls, each measured to bite in this experiment (counts asserted > 0).
3. Gradients through GMREDI_CALC_TENSOR (bolus path included): finite on every lane; FD h-sweep at smooth points.
All under the gate XLA flags (conftest.py), REAL parameters traced (jit arguments inside `Gmredi`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import goadk_gate as G

VARIANTS = G.VARIANTS


def _ulp(x):
    return np.float64(np.nextafter(np.float64(x), np.inf))


# ------------------------------------------------------------------------------------------------ 1. dump gates

@pytest.mark.parametrize("inp", VARIANTS)
def test_calc_tensor_bolus_bitwise_vs_dumps(inp):
    ds = G.oracle(inp)
    its = ds.iterations()
    assert its == [0, 1, 2]
    gm = G.setup_gm(inp)
    assert gm.GM_AdvForm and gm.GM_ExtraDiag and float(gm.GM_skewflx) == 0.0
    for it in its:
        ours, ref = G.tensor_case(inp, it)
        res = G.compare_fields(ours, ref)
        assert set(res) == set(G.TENSOR)
        assert not G.failures(res), (it, G.failures(res))
        # the bolus stream functions and the extra-diagonal terms are live (nonzero) in the oracle
        for n in ("GM_PsiX", "GM_PsiY", "Kuz", "Kvz"):
            assert np.count_nonzero(ref[n]) > 1000, (n, it)


@pytest.mark.parametrize("inp", VARIANTS)
def test_residual_flow_bolus_bitwise_vs_dumps(inp):
    for it in (0, 1):
        ours, ref = G.residual_case(inp, it, grid_it=it + 1)
        res = G.compare_fields(ours, ref)
        assert not G.failures(res), (it, G.failures(res))


# ------------------------------------------------------------------------------------------- 2. negative controls

def _tensor_bites(inp, **kw):
    return sum(G.n_differing(G.compare_fields(*G.tensor_case(inp, it, **kw))) for it in (0, 1, 2))


def test_tensor_negative_controls_bite():
    inp = "input_ad"
    counts = {}
    counts["GM_inpK3dGM x(1+2^-52)"] = _tensor_bites(inp, gm_override=lambda g: g.replace(
        GM_inpK3dGM=G.with_data(g.GM_inpK3dGM, g.GM_inpK3dGM.data * (1.0 + 2.0**-52))))
    counts["GM_inpK3dRedi x(1+2^-52)"] = _tensor_bites(inp, gm_override=lambda g: g.replace(
        GM_inpK3dRedi=G.with_data(g.GM_inpK3dRedi, g.GM_inpK3dRedi.data * (1.0 + 2.0**-52))))
    counts["GM_Scrit +1ulp"] = _tensor_bites(inp, gm_override=lambda g: g.replace(GM_Scrit=_ulp(g.GM_Scrit)))
    counts["GM_ExtraDiag off"] = _tensor_bites(inp, gm_override=lambda g: g.replace(GM_ExtraDiag=False))
    counts["GM_AdvForm off"] = _tensor_bites(inp, gm_override=lambda g: g.replace(GM_AdvForm=False))
    counts["GM_Small_Number x1e12"] = _tensor_bites(inp, gm_override=lambda g: g.replace(
        GM_Small_Number=np.float64(1e12) * g.GM_Small_Number))
    print("tensor negative controls (differing points over the 3 dumped iterations):", counts)
    for name, n in counts.items():
        assert n > 0, (name, counts)


def test_residual_flow_negative_controls_bite():
    inp = "input_ad"
    counts = {}
    # recip_hFacW/S of the step start (G00 of the same iteration) instead of UPDATE_SURF_DR(.TRUE.)'s
    counts["grid of the step start (it 1)"] = G.n_differing(G.compare_fields(*G.residual_case(inp, 1, grid_it=1)))
    counts["GM_PsiX x(1+2^-52)"] = G.n_differing(G.compare_fields(*G.residual_case(
        inp, 0, grid_it=1, gm_override=lambda g: g.replace(GM_PsiX=G.with_data(g.GM_PsiX,
            g.GM_PsiX.data * (1.0 + 2.0**-52))))))
    counts["gravitySign flipped"] = G.n_differing(G.compare_fields(*G.residual_case(
        inp, 0, grid_it=1, grid_override=lambda gr: gr.replace(gravitySign=-gr.gravitySign))))
    print("residual-flow negative controls:", counts)
    for name, n in counts.items():
        assert n > 0, (name, counts)


# ----------------------------------------------------------------------------------------------- 3. gradients

def _tensor_fn(inp, it):
    """x -> {tensor name: [tile,k,j,i]}: GMREDI_CALC_TENSOR (bolus path included) on the dumped inputs of `it`, x =
    the sigma fields and the REAL GMREDI.h parameters the bolus path reads."""
    ds = G.oracle(inp)
    e = G.grid_gate.experiment(G.EXP, inp)
    cfg, sz = e.cfg, e.cfg.size
    grid, params = G.dump_grid(inp, it, ds), G.dump_params(inp, it, ds)
    gm = G.setup_gm(inp)
    gm = gm.replace(**{n: G.wrap(n, jnp.asarray(ds.field(it, "S00_begin", n)), sz) for n in G.TENSOR})
    state = G.StateH(hMixLayer=G.xy("hMixLayer", ds.field(it, "P03_mxlayer", "hMixLayer")[:, 0], sz))
    floats = ("GM_Scrit", "GM_Sd", "GM_Small_Number", "GM_slopeSqCutoff", "GM_Kmin_horiz")
    x0 = {n: jnp.asarray(ds.field(it, "P02_rho_sigma_ivdc", n)) for n in ("sigmaX", "sigmaY", "sigmaR")}
    x0.update({n: jnp.float64(getattr(gm, n)) for n in floats})
    x0["GM_inpK3dGM"] = gm.GM_inpK3dGM.data
    x0["GM_inpK3dRedi"] = gm.GM_inpK3dRedi.data

    def f(x):
        g = gm.replace(**{n: x[n] for n in floats},
                       GM_inpK3dGM=G.with_data(gm.GM_inpK3dGM, x["GM_inpK3dGM"]),
                       GM_inpK3dRedi=G.with_data(gm.GM_inpK3dRedi, x["GM_inpK3dRedi"]))
        out = G.gmredi_calc_tensor(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy,
                                   *(G.xyz(n, x[n], sz) for n in ("sigmaX", "sigmaY", "sigmaR")), 0.0, 0,
                                   cfg=cfg, grid=grid, params=params, gm=g, state=state)
        return {n: getattr(out, n).data for n in G.TENSOR}
    return f, x0, grid


def test_gradient_finite_and_fd():
    """d(random weights . tensor)/d(sigma, K3D fields, GM parameters) finite on every lane (land, halos, neutral and
    unstable lanes, cut-off lanes); FD h-sweep at smooth wet points through GM_PsiX/Y (local cost)."""
    inp = "input_ad"
    f, x0, grid = _tensor_fn(inp, 0)
    shapes = {n: v.shape for n, v in jax.eval_shape(f, x0).items()}
    rng = np.random.default_rng(11)
    w_all = {n: jnp.asarray(rng.standard_normal(s)) for n, s in shapes.items()}

    def cost(x, w):
        out = f(x)
        return sum(jnp.sum(w[n] * out[n]) for n in sorted(w))
    dJ = jax.jit(jax.grad(cost))
    g = dJ(x0, w_all)
    bad = {n: int(np.count_nonzero(~np.isfinite(np.asarray(v)))) for n, v in g.items()}
    assert not any(bad.values()), bad
    assert all(np.any(np.asarray(g[n]) != 0) for n in ("sigmaX", "sigmaY", "sigmaR", "GM_inpK3dGM", "GM_Scrit"))

    # FD at smooth points: stably stratified wet interior points, cost local to the bolus stream functions
    J = jax.jit(cost)
    sR = np.asarray(x0["sigmaR"])
    mW = np.asarray(grid.maskW.data)
    t, k, j, i = np.nonzero((sR < -1e-4) & (mW == 1))
    sz = G.grid_gate.experiment(G.EXP, inp).cfg.size
    inner = ((j >= sz.OLy + 2) & (j < sz.sNy + sz.OLy - 2) & (i >= sz.OLx + 2) & (i < sz.sNx + sz.OLx - 2)
             & (k >= 2) & (k < sz.Nr - 1))
    picks = rng.choice(np.nonzero(inner)[0], size=12, replace=False)
    checked, worst = 0, 0.0
    for name in ("sigmaX", "sigmaR", "GM_inpK3dGM"):
        for p in picks[:4]:
            idx = (t[p], k[p], j[p], i[p])
            w = {}
            for n, shp in shapes.items():
                a = np.zeros(shp)
                if n in ("GM_PsiX", "GM_PsiY"):
                    win = (idx[0], slice(idx[1] - 1, idx[1] + 2), slice(idx[2] - 2, idx[2] + 3),
                           slice(idx[3] - 2, idx[3] + 3))
                    a[win] = rng.standard_normal(a[win].shape)
                w[n] = jnp.asarray(a)
            gp = float(np.asarray(dJ(x0, w)[name])[idx])
            x = float(np.asarray(x0[name])[idx])
            if gp == 0.0 or x == 0.0:
                continue
            errs = []
            for h in (abs(x) * 10.0 ** -e_ for e_ in range(2, 9)):
                xp = {**x0, name: x0[name].at[idx].add(h)}
                xm = {**x0, name: x0[name].at[idx].add(-h)}
                errs.append(abs((float(J(xp, w)) - float(J(xm, w))) / (2 * h) - gp) / abs(gp))
            print("FD", name, tuple(int(q) for q in idx), gp, min(errs))
            assert min(errs) < 1e-6, (name, idx, gp, errs)
            checked += 1
            worst = max(worst, min(errs))
    print("FD points", checked, "worst best-of-sweep rel. error", worst)
    assert checked >= 6


# -------------------------------------------------------------------------------------------- 4. unported options

def test_unported_options_raise():
    inp = "input_ad"
    f, x0, _ = _tensor_fn(inp, 0)
    gm0 = G.setup_gm(inp)
    # 'gkw91' / 'ac02' are ported (GO lane, cs32x15: gmredi_slope_psi.F:290-312, gated in test_cs32_go)
    for scheme, err in (("ldd97", (NotImplementedError, RuntimeError)),
                        ("fm07", RuntimeError), ("nonsense", RuntimeError)):
        from mitjax.pkg.gmredi.gmredi_slope_psi import gmredi_slope_psi
        e = G.grid_gate.experiment(G.EXP, inp)
        sz = e.cfg.size
        a = G.xy("a", np.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx)), sz)
        with pytest.raises(err):
            gmredi_slope_psi(a, a, a, a, a, a, a, a, 0.0, 2, cfg=e.cfg, params=G.dump_params(inp, 0),
                             gm=gm0.replace(GM_taper_scheme=scheme))


# ------------------------------------------------------------------------------------------- 5. replay harness

# harness run: job 27833466 (commit 5bf771b), registered in reference/replay_goadk/CURRENT; REPLAY OK, every output
# bitwise at the first comparison (dev job 27833576)

def test_replay_bitwise():
    """Every output of the GOADK harness (global_ocean.90x40x15/code_ad, synthetic stratification incl. unstable,
    neutral, steep and cut-off lanes, synthetic K3D fields, real grid with land and partial cells, halos): the GM/Redi
    routines (bolus path, ExtraDiag transports, bolus residual flow), GAD_DST3_ADV_X/Y (calcCFL T/F),
    MOM_U/V_BOTDRAG_COEFF with bottomDragFld (inp_KE T/F) and COST_ATLANTIC_HEAT, bitwise on every point."""
    ours, ref = G.replay_run()
    res = G.compare_fields(ours, ref)
    print({k: v[:2] for k, v in res.items()})
    assert not G.failures(res), G.failures(res)
    assert np.count_nonzero(ref["objf_atl"]) == 1          # one tile holds the section row jsec = 28


def test_replay_negative_controls_bite():
    counts = {}

    def bdrag(objs):
        f = G.CtrlFields(bottomDragFld=G.with_data(objs["grid"].R_low, 0.0 * objs["grid"].R_low.data))   # control 0
        return {**objs, "ctrlf": f}
    counts["bottomDragFld zero"] = G.n_differing(G.compare_fields(*G.replay_run(patch=bdrag)))
    counts["GM_inpK3dGM x(1+2^-52)"] = G.n_differing(G.compare_fields(*G.replay_run(patch=lambda o: {
        **o, "gm": o["gm"].replace(GM_inpK3dGM=G.with_data(o["gm"].GM_inpK3dGM,
                                                            o["gm"].GM_inpK3dGM.data * (1.0 + 2.0**-52)))})))
    print("replay negative controls:", counts)
    for name, n in counts.items():
        assert n > 0, (name, counts)


# ------------------------------------------------------------------------------- 6. GRDCHK_GET_POSITION (nbeg = 0)

@pytest.mark.parametrize("inp", VARIANTS)
def test_grdchk_positions_match_oracle(inp):
    """data.grdchk without nbeg (nbeg = 0, grdchk_readparms.F:82): GRDCHK_GET_POSITION sets nbeg to the packed index
    of (iGloPos, jGloPos, kGloPos) and nend = nbeg + nend (grdchk_get_position.F:188-199); then GRDCHK_LOC finds the
    4 points. Gate: the (i, j, k, bi, bj, iobc, rec) of every point == the oracle's "grdchk pos:" lines, icomp and
    ncvarcomp == its "ph-test icomp, ncvarcomp, ichknum" lines, our two "grad-res" lines == its lines (zero-adxx FD
    run of the variant, job 27832347)."""
    from mitjax.io import stdout as so
    from mitjax.pkg.grdchk.grdchk_get_position import grdchk_get_position
    pts, gm, s = G.grdchk_case(inp)
    lines = G.fd_stdout(inp)
    orc = so.grdchk(lines).points
    ours = [(r.itilepos, r.jtilepos, r.layer, r.itile, r.jtile, r.obcspos, r.icvrec) for _, r in pts]
    assert len(ours) == 4
    assert ours == [(p.i, p.j, p.k, p.bi, p.bj, p.iobc, p.rec) for p in orc]
    ph = [tuple(int(x) for x in ln.text.split()[-3:]) for ln in lines if "ph-test icomp, ncvarcomp, ichknum" in ln.text]
    assert ph == [(c, gm.ncvarcomp, n) for n, (c, _) in enumerate(pts, 1)], (ph, gm.ncvarcomp)
    e = G.grid_gate.experiment(G.EXP, inp)
    sz = e.cfg.size
    kw = dict(maskC=np.asarray(G.oracle(inp).field(0, "G00_geometry", "maskC")), sz=sz, iLocTile=s["iGloTile"],
              jLocTile=s["jGloTile"], ncvargrd="c", ncvarrecs=1, ncvarnrmax=sz.Nr if G.CONTROL[inp][0] == 3 else 1,
              ncvarxmax=sz.sNx, ncvarymax=sz.sNy, nwettile=gm.nwettile)
    gp = grdchk_get_position(nbeg=s["nbeg"], nend=s["nend"], iGloPos=s["iGloPos"], jGloPos=s["jGloPos"],
                             kGloPos=s["kGloPos"], obcsglo=s["obcsglo"], recglo=s["recglo"], **kw)
    i0 = next(n for n, ln in enumerate(lines) if "grad-res exact position met" in ln.text)
    assert [ln.text for ln in lines[i0:i0 + 2]] == gp.lines


def test_grdchk_get_position_negative_controls_bite():
    """Planted errors give other positions: kGloPos + 1 (another layer), nend not shifted by nbeg (fewer points), the
    mask shifted by one point."""
    inp = "input_ad.kapredi"
    good = [(r.itilepos, r.jtilepos, r.layer) for _, r in G.grdchk_case(inp)[0]]
    from mitjax.pkg.grdchk import grdchk_get_position as gpm
    orig = gpm.grdchk_get_position

    def no_shift(**kw):
        r = orig(**kw)
        r.nend = kw["nend"]
        return r
    gpm.grdchk_get_position = no_shift
    try:
        unshifted = G.grdchk_case(inp)[0]
    finally:
        gpm.grdchk_get_position = orig
    assert len(unshifted) != len(good)
    mC = np.roll(np.asarray(G.oracle(inp).field(0, "G00_geometry", "maskC")), 1, axis=-1)
    assert [(r.itilepos, r.jtilepos, r.layer) for _, r in G.grdchk_case(inp, maskC=mC)[0]] != good


# --------------------------------------------------------------------------------------- 7. controls (genarr)

@pytest.mark.parametrize("inp", VARIANTS)
def test_control_zero_effective_and_identity(inp):
    """CTRL_MAP_INI_GENARR with the run's own (zero) control file and weight: the effective record == the oracle's
    `xx_<name>.effective.0000000000` (interior, bitwise; for kapgm/kapredi this is the K3D background of our
    GMREDI_INIT_VARIA after the map); theta after the map == S00_begin theta of iteration 0 at every point (halos
    included: the map's EXCH_XYZ_RL)."""
    sz = G.grid_gate.experiment(G.EXP, inp).cfg.size
    out, eff, ref, _ = G.control_case(inp)
    dim, name, target = G.CONTROL[inp]
    o, r = G.interior(eff[(dim, 1)].data, sz), G.interior(ref, sz)
    assert o.shape == r.shape and np.all(np.isfinite(o)) and np.all(np.isfinite(r))
    assert np.array_equal(o, r) and np.array_equal(o.view(np.int64), r.view(np.int64))
    th, s00 = np.asarray(out["theta"].data), G.oracle(inp).field(0, "S00_begin", "theta")
    assert np.array_equal(th, s00) and np.array_equal(th.view(np.int64), s00.view(np.int64))


def test_control_negative_controls_bite():
    """A planted control (1e-3 at the wet points of tile 2) changes the effective record against the zero-control
    oracle; S00 theta of iteration 1 instead of 0 differs from our mapped theta."""
    inp = "input_ad"
    sz = G.grid_gate.experiment(G.EXP, inp).cfg.size
    _, eff, ref, _ = G.control_case(inp, xx_override=lambda a: a + np.where(np.arange(a.shape[0])[:, None, None, None]
                                                                             == 2, 1e-3, 0.0))
    n = int(np.count_nonzero(G.interior(eff[(3, 1)].data, sz) != G.interior(ref, sz)))
    out, _, _, _ = G.control_case(inp)
    m = int(np.count_nonzero(np.asarray(out["theta"].data) != G.oracle(inp).field(1, "S00_begin", "theta")))
    print("control negative controls:", n, m)
    assert n > 0 and m > 0


# planted-control oracle: lane A's dumps-on runs job27833840-ctrlxx (doInitXX = .FALSE., planted control files
# $MJX_REFERENCE/ctrl_planted/0542f7a-*, seed 20261001), dumps of iterations 0-3

@pytest.mark.parametrize("inp", VARIANTS)
def test_control_planted_vs_ctrlxx_oracle(inp):
    """Planted control (the ctrlxx run's own xx file, doInitXX = .FALSE.): the effective record bitwise; input_ad:
    theta after the map == S00_begin theta (iteration 0) at every point; kapgm / kapredi: GMREDI_CALC_TENSOR with the
    mapped K3D field == P05 at every dumped iteration (0-3) of the ctrlxx dumps (where the control enters the model)."""
    from mitjax.io.dump import DumpSet
    top = G.run_top(inp, "ctrlxx")
    ds = DumpSet(top / "dumps")
    sz = G.grid_gate.experiment(G.EXP, inp).cfg.size
    out, eff, ref, _ = G.control_case(inp, kind="ctrlxx", ds=ds)
    dim, name, target = G.CONTROL[inp]
    o, r = G.interior(eff[(dim, 1)].data, sz), G.interior(ref, sz)
    assert np.array_equal(o.view(np.int64), r.view(np.int64))
    if target == "theta":
        th, s00 = np.asarray(out["theta"].data), ds.field(0, "S00_begin", "theta")
        assert np.array_equal(th.view(np.int64), s00.view(np.int64))
    if target in ("GM_inpK3dGM", "GM_inpK3dRedi"):
        for it in ds.iterations():
            ours, refT = G.tensor_case(inp, it, ds=ds, gm_override=lambda g: g.replace(**{target: out[target]}))
            assert not G.failures(G.compare_fields(ours, refT)), (it, target)


# ---------------------------------------------------------------------------------- 8. GAD_ADVECTION dispatch

def test_gad_advection_dispatch_dst3_vs_replay():
    """GAD_ADVECTION's X/Y dispatch of scheme 30 (gad_advection.F:419-422, :640-643: calcCFL .TRUE., deltaTLev(k),
    the transport, the level-k velocity, maskLocW/S) on the GOADK harness inputs == the harness's GAD_DST3_ADV_X/Y
    outputs (calcCFL .TRUE.), every level, bitwise. (No in-model gate: GAD_ADVECTION raises for the code_ad build's
    ALLOW_AUTODIFF arms.)"""
    import jax.numpy as jnp
    from mitjax.pkg.generic_advdiff import gad_advection as GA
    from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_kernel_cfg
    from mitjax.pkg.generic_advdiff.gad_h import ENUM_DST3
    from mitjax.tests import gmredi_gate as GG
    io = G._io()
    rd = G.replay_rundir()
    size, f3, f2, par, one_d = io.read_inputs(rd)
    out2, g = io.read_outputs2(rd, size), io.read_grid(rd, size)
    e = G.grid_gate.experiment(G.EXP, "input_ad")
    sz = e.cfg.size
    grid = GG.replay_grid(g, sz)
    kc = gad_kernel_cfg(e.cfg)
    dT = jnp.float64(par["deltaTloc"])
    for k in range(1, sz.Nr + 1):
        def two(name, a):
            return G.xy(name, a[:, k-1], sz)
        mW, mS = two("maskLocW", np.asarray(grid.maskW.data)), two("maskLocS", np.asarray(grid.maskS.data))
        T, prior = two("tracer", f3["Tracer"]), two("af", f3["dfP"])
        ax = GA._scheme_x(ENUM_DST3, k, dT, two("uTrans", f3["uTr"]), two("uFld", f3["uFld"]), mW, T, prior, kc, grid)
        ay = GA._scheme_y(ENUM_DST3, k, dT, two("vTrans", f3["vTr"]), two("vFld", f3["vFld"]), mS, T, prior, kc, grid)
        for ours, ref in ((ax, out2["dst3x_cflT"][:, k-1]), (ay, out2["dst3y_cflT"][:, k-1])):
            o = np.asarray(ours.data)
            assert np.array_equal(o.view(np.int64), ref.view(np.int64)), k


@pytest.mark.parametrize("inp", VARIANTS)
def test_ctrl_init_ctrlvar_files(inp):
    """CTRL_INIT_CTRLVAR as CTRL_INIT_FIXED registers the genarr control (ctrl_init_fixed.F:231-261: varRecs 1,
    sNx, sNy, Nr or 1, 'c', 'Arr3D' / 'Arr2D'): the file names of CTRL_SET_FNAME are the files of the runs
    (`xx_<name>.0000000000` in the yardstick run, `adxx_<name>.0000000000` in the FD run), and the first-guess record
    of CTRL_SET_GLOBFLD_XYZ / _XY (doInitXX, optimcycle 0) == the yardstick run's file, bitwise (zeros)."""
    from mitjax.pkg.ctrl.ctrl_init_ctrlvar import ctrl_init_ctrlvar
    sz = G.grid_gate.experiment(G.EXP, inp).cfg.size
    dim, name, _ = G.CONTROL[inp]
    nz = sz.Nr if dim == 3 else 1
    var, fname, recs = ctrl_init_ctrlvar(name, 1, 1, 1, 1, 1, sz.sNx, sz.sNy, nz, "c", f"Arr{dim}D", False, sz=sz,
                                         doInitXX=True, optimcycle=0)
    assert fname[0] == f"{name}.0000000000" and fname[1] == f"ad{name}.0000000000"
    top, fd = G.run_top(inp, "yardstick"), G.run_top(inp, "fdzero")
    assert (top / "rundir" / f"{fname[0]}.data").exists() and (fd / "rundir" / f"{fname[1]}.data").exists()
    ref = G.read_global(top / "rundir" / f"{fname[0]}.data", nz, sz)
    assert len(recs) == 1
    ours = G.interior(recs[0][:, :nz], sz)
    assert np.array_equal(ours.view(np.int64), G.interior(ref, sz).view(np.int64))
    assert var.ncvarnrmax == nz and var.ncvargrd == "c"


def test_control_planted_negative_controls_bite():
    """On the planted-control oracle: the map without its EXCH_XYZ_RL (theta halos), the unmapped K3D background in
    GMREDI_CALC_TENSOR (kapgm), and the zero control (effective record) each differ from the oracle."""
    from mitjax.io.dump import DumpSet
    from mitjax.pkg.ctrl import ctrl_map_genarr as cmg
    sz = G.grid_gate.experiment(G.EXP, "input_ad").cfg.size
    counts = {}
    ds = DumpSet(G.run_top("input_ad", "ctrlxx") / "dumps")
    orig = cmg.EXCH_XYZ_RL
    cmg.EXCH_XYZ_RL = lambda fld, ex: fld
    try:
        out, _, _, _ = G.control_case("input_ad", kind="ctrlxx", ds=ds)
    finally:
        cmg.EXCH_XYZ_RL = orig
    counts["no EXCH_XYZ_RL (theta)"] = int(np.count_nonzero(np.asarray(out["theta"].data)
                                                            != ds.field(0, "S00_begin", "theta")))
    _, eff0, ref, _ = G.control_case("input_ad", kind="ctrlxx", ds=ds, xx_override=lambda a: 0.0 * a)
    counts["zero control (effective)"] = int(np.count_nonzero(G.interior(eff0[(3, 1)].data, sz)
                                                              != G.interior(ref, sz)))
    dsg = DumpSet(G.run_top("input_ad.kapgm", "ctrlxx") / "dumps")
    counts["unmapped K3D (kapgm P05)"] = G.n_differing(G.compare_fields(*G.tensor_case("input_ad.kapgm", 0, ds=dsg)))
    print("planted-control negative controls:", counts)
    for name, n in counts.items():
        assert n > 0, (name, counts)


# ------------------------------------------------------------------------- 9. exchange maps of the code_ad layout

def test_code_ad_exchange_map_registered_and_reproduces_probe():
    """The registered map file of global_ocean.90x40x15/code_ad (exch_maps.VARIANT_PROBES / VARIANT_MAP_SHA256, made by
    scripts/make_exch_maps.py --code code_ad) loads with its sha256 and reproduces every exchange probe output of the
    run it was made from bitwise; the code/ layout of the experiment still loads its own map."""
    import importlib.util
    from mitjax import paths
    from mitjax.eesupp import exch_maps as EM
    m = EM.load_maps(G.EXP, code="code_ad")
    assert m.layout.tag() == "t4_45x20_ol3x3"
    spec = importlib.util.spec_from_file_location("_mjx_make_exch_maps", paths.REPO / "scripts" / "make_exch_maps.py")
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    bad = mm.check_against_probe(m, G.oracle("input_ad"), 0)
    assert sum(bad.values()) == 0, {k: v for k, v in bad.items() if v}
    assert EM.load_maps(G.EXP).layout.tag() == "t36_10x10_ol3x3"
    assert EM.load_maps(G.EXP, code="code").layout.tag() == "t36_10x10_ol3x3"
