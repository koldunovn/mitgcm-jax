"""The cube dynamics experiments end to end (plan Task 25, lane B), tier 1x: adjustment.cs-32x32x1/input with the
production set-up (mitjax/drivers/model.Model: cube exchanger from the W2 topology, curvilinear grid, the exch2
global IO map of MDS_READ_FIELD, USE_OLD_EXTERNAL_FORCING) on the run directory of lane A's dumps-on run.

Gates: the initial State (S00_begin of iteration 0: 33 fields) and every stage FORWARD_STEP probes in steps 1-3
(S00 ... S16, 10 stages per step) bitwise on every point of every tile, halos included, vs lane A's jdon dumps; the
whole run through the run driver (%MON records identical to the oracle's STDOUT, testreport digits recorded); P=4 and
P=6 (subprocess, 6 fake devices) == P=1 bitwise after 3 steps.
Negative controls (must bite): the global files read without the exch2 IO map (bathymetry and initial eta on the
wrong tiles: the initial State differs); the old external forcing arm without its difference (g_arr += gU instead
of g_arr += gU_new - gU: step 2 differs from S06 on).
Also (session 10): the whole run checks every output file both runs wrote byte for byte (pickup.ckptA);
MDS_WRITE_FIELD's useSingleCpuIO branch round-trips the oracle's snapshots byte for byte; solid-body's step 1 up to
S05 plus DYNAMICS' CALC_PHI_HYD level loop (D00a, S06 totPhiHyd / phiHydLow) bitwise, and the Exner factors vs the
oracle's PHrefC / PHrefF.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from mitjax import paths
from mitjax.tests import cube_run_gate as C

EXP = ("adjustment.cs-32x32x1", "input")


def _initial_bad(r):
    st, ff, phi0 = r.m.initial_carry()[:3]                    # (+ flow, + package state with GM/Redi)
    vals = {**C.rg.state_fields(st), **{n: getattr(ff, n) for n in ff.names()}, "phi0surf": phi0}
    res = C.compare_stage(r, r.its[0], "S00_begin", vals)
    return C.bad(res), len(res)


def test_adjustment_cs_steps_bitwise():
    r = C.cube_run(*EXP)
    bad, n = _initial_bad(r)
    assert n >= 33 and bad == {}, (n, bad)
    out, _ = C.run_steps(r, 3)
    assert {it for it, _ in out} == {0, 1, 2}
    assert all(len([1 for (i, _) in out if i == it]) >= 10 for it in (0, 1, 2)), sorted(out)
    assert {k: v for k, v in out.items() if v} == {}


def test_adjustment_cs_negative_controls(monkeypatch):
    from mitjax.drivers import model as DM
    from mitjax.model.src import apply_forcing as AF
    from mitjax.io import mds
    # the global files read with the default tile placement instead of the exch2 IO map (mds.E2ioLayout, ADVECT
    # lane's port of mdsio_read_field.F:120-129, :396-441)
    monkeypatch.setattr(mds.E2ioLayout, "from_w2", classmethod(lambda cls, w2, size: None))
    r = C.CubeRun(*EXP)
    bad, _ = _initial_bad(r)
    assert {"etaN", "hFacC"} <= set(bad), sorted(bad)
    monkeypatch.undo()
    # the USE_OLD_EXTERNAL_FORCING difference dropped
    orig = AF._old_external_forcing

    def planted(g_arr, g_state, k, ext, *a, **kw):
        from mitjax.farray import loop_i, loop_j
        sz = kw["cfg"].size
        j, i = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        g_new = ext(g_state, *a[:4], k, a[4], **kw)
        return g_arr.at[i, j].set(g_arr[i, j] + g_new[i, j, k])
    monkeypatch.setattr(AF, "_old_external_forcing", planted)
    r = C.CubeRun(*EXP)
    out, _ = C.run_steps(r, 2)
    # step 1 starts from rest (gU = 0 after MOM_FLUXFORM: no advection, no viscosity, Coriolis of u = v = 0), so
    # the planted term is 0 there; from step 2 on gU /= 0
    assert out[(1, "S06_dynamics")] and not out[(1, "S04_oceanic_phys")], sorted(k for k, v in out.items() if v)
    monkeypatch.undo()
    assert orig is AF._old_external_forcing


def test_adjustment_cs_whole_run():
    """The whole run through the run driver: %MON records identical; testreport digits of every check-list variable
    vs results/ and vs the oracle, next to the yardstick (round-off-level means: no threshold here, Nikolay
    decides; the digits are printed); every MDSIO file the oracle wrote with useSingleCpuIO = .TRUE. (the pickup,
    the state snapshots, their .meta) byte-identical (MDS_WRITE_FIELD's single-CPU branch, lane B)."""
    from mitjax.tests import advect_gate as ag
    m, res, o = C.whole_run(*EXP)
    diffs, rows, nblocks = ag.run_verdict(*EXP, res, o)
    print("digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    files, missing = C.output_file_diffs(m, o)
    print("output files compared:", len(files), "differing:", sorted(f for f, ok in files.items() if not ok),
          "oracle-only:", missing)
    assert nblocks == 25 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    assert all(d4 >= 13 or d4 == 16 for _, _, _, d4 in rows if d4 is not None), rows
    # round-off means (plan decisions 2026-10-02): Vav has 4 digits vs results/ (yardstick 4); gated relative to the
    # field size instead (|ours - results/| / max(|Vmn|, |Vmx|) <= 1e-13 at every monitor time)
    rel = C.roundoff_rel(res.records, *EXP, ["Vav"])
    print("adjustment.cs round-off means, |ours - results/| / field size:", rel)
    assert all(x <= 1e-13 for x in rel.values()), rel
    assert files and all(files.values()), sorted(f for f, ok in files.items() if not ok)
    assert any(f.startswith("pickup") for f in files), sorted(files)


def test_adjustment_cs_sharded_p4():
    r = C.cube_run(*EXP)
    assert C.sharded_vs_single(r, 4) == {}


P6 = r"""
import os
from mitjax.xla_flags import gate_xla_flags
os.environ["XLA_FLAGS"] = gate_xla_flags("").replace("device_count=4", "device_count=6")
import jax
from mitjax.tests import cube_run_gate as C
assert len(jax.devices()) == 6
print("P6", C.sharded_vs_single(C.cube_run("adjustment.cs-32x32x1", "input"), 6))
"""


def test_adjustment_cs_sharded_p6():
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env["PYTHONPATH"] = str(paths.REPO)
    r = subprocess.run([sys.executable, "-c", P6], env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "P6 {}" in r.stdout, r.stdout


# ---------------------------------------------------------------------------------------------------------------------
# p coordinates (solid-body.cs-32x32x1, adjustment.cs-32x32x1/input.nlfs: ATMOSPHERIC, IDEALG)

SB = ("solid-body.cs-32x32x1", "input")


def test_pcoords_fixed_grid_equals_g00():
    """INITIALISE_FIXED's grid in p coordinates (INI_VERTICAL_GRID's rF(Nr+1) = top_Pres branch, INI_DEPTHS'
    p-coordinate arms, the curvilinear grid, masks, INI_CORI): every G00 field the Grid holds, bitwise. input.nlfs
    differs only in recip_hFacW/S, which NONLIN_FRSURF rescales in INITIALISE_VARIA (State fields there)."""
    import dataclasses
    bad, n = C.fixed_grid_g00_bad(*SB)
    assert n >= 79 and bad == {}, (n, bad)
    bad, n = C.fixed_grid_g00_bad("adjustment.cs-32x32x1", "input.nlfs")
    assert n >= 82 and set(bad) <= {"recip_hFacW", "recip_hFacS"}, (n, bad)
    # negative control: the axis anchored at a planted top_Pres
    from mitjax.config.params import load
    from mitjax.model.src.ini_parms import ini_parms_grid
    from mitjax.pkg.exch2.w2_eeboot import exch2_topology, w2_eeboot
    e = load(*SB)
    gp = ini_parms_grid(e, exch2_topology(w2_eeboot(e)[0]))
    bad, _ = C.fixed_grid_g00_bad(*SB, gp=dataclasses.replace(gp, top_Pres=1.0))
    assert {"rF", "rC"} <= set(bad), sorted(bad)


def test_solid_body_initial_state(monkeypatch):
    """The solid-body rotation of its own INI_VEL / INI_PSURF (verification/solid-body.cs-32x32x1/code) and the rest of
    INITIALISE_VARIA in p coordinates: S00_begin bitwise (32 fields). Control: the unconditional
    EXCH_UV_XYZ_RL( uVel, vVel, .TRUE. ) of its INI_VEL with signs off -> uVel/vVel differ on the rotated edges."""
    r = C.CubeRun(*SB)
    bad, n = _initial_bad(r)
    assert n >= 32 and bad == {}, (n, bad)
    from mitjax.verification.solid_body_cs_32x32x1.code import ini_vel as IV
    orig = IV.EXCH_UV_XYZ_RL
    monkeypatch.setattr(IV, "EXCH_UV_XYZ_RL", lambda u, v, ws, *, ex: orig(u, v, False, ex=ex))
    bad, _ = _initial_bad(C.CubeRun(*SB))
    assert {"uVel", "vVel"} & set(bad), sorted(bad)


def test_solid_body_steps_bitwise(monkeypatch):
    """solid-body.cs-32x32x1 (ATMOSPHERIC, IDEALG, Nr = 1, salt stepping only, VECINV, integr_GeoPot = 2 by default):
    every stage FORWARD_STEP probes in steps 1-3 bitwise on every point of every tile vs lane A's jdon dumps
    (DO_ATMOSPHERIC_PHYS's fluidIsAir rhoInSitu at S04, CALC_PHI_HYD's ATMOSPHERIC arm at D00a, MOM_VECINV with the
    cube passes at D00c, ...). theta = tRef and atm_Rq = 0 make rhoInSitu = alphaRho = 0 here, so the ddPIm / ddPIp
    products are 0: the Exner factors are checked against the oracle's PHrefC / PHrefF (cube_run_gate.phiref_bad:
    SET_REF_STATE's phiRef from the same factors), bitwise, for solid-body and adjustment.cs/input.nlfs. Controls
    (one step each): atm_Rq planted (1e-3; the run's is 0.) -> rhoInSitu differs at S04 where salt /= 0; MOM_VECINV
    without the W2 tile view's corner flags (every exch2_is*edge 0) -> gU / gV differ at D00c; the factor
    (rC(1)/atm_Po)**atm_kappa one ulp up -> PHrefC differs."""
    import types
    import numpy as np
    from mitjax.model.src import dynamics as DY
    from mitjax.model.src.ini_parms import _rk
    r = C.CubeRun(*SB)
    out, _ = C.run_steps(r, 3)
    n = C.run_steps.ncompared
    print("solid-body stages compared:", dict(sorted(n.items())))
    assert {it for it, _ in out} == {0, 1, 2}
    assert all(len([1 for (i, _) in out if i == it]) >= 15 for it in (0, 1, 2)), sorted(out)
    assert all((it, "D00c_mom_vecinv") in out for it in (0, 1, 2)), sorted(out)
    assert {k: v for k, v in out.items() if v} == {}
    for e in (SB, ("adjustment.cs-32x32x1", "input.nlfs")):
        assert C.phiref_bad(C.CubeRun(*e) if e != SB else r) == {"PHrefF": [], "PHrefC": []}, e
    # controls
    rc = C.CubeRun(*SB)
    rc.m.params = rc.m.params.replace(traced={"atm_Rq": np.float64(1e-3)})
    out, _ = C.run_steps(rc, 1)
    assert "rhoInSitu" in out[(0, "S04_oceanic_phys")], sorted(out[(0, "S04_oceanic_phys")])
    orig = DY._w2_tile_view

    def no_edges(grid, params):
        v = orig(grid, params)
        return types.SimpleNamespace(**{n: (a if n == "exch2_myFace" else 0*a) for n, a in vars(v).items()})
    monkeypatch.setattr(DY, "_w2_tile_view", no_edges)
    out, _ = C.run_steps(C.CubeRun(*SB), 1)
    assert {"gU_k001", "gV_k001"} & set(out[(0, "D00c_mom_vecinv")]), sorted(out[(0, "D00c_mom_vecinv")])
    monkeypatch.undo()
    pk = np.asarray(r.m.params.rC_Po_kappa.data).copy()
    pk[0] = np.nextafter(pk[0], np.inf)
    r.m.params = r.m.params.replace(traced={"rC_Po_kappa": _rk("rC_Po_kappa", pk.size, pk)})
    assert C.phiref_bad(r)["PHrefC"] == [0]


def test_solid_body_whole_run():
    """solid-body through the run driver: %MON records identical to the oracle STDOUT; testreport digits of every
    check-list variable vs results/ and vs the oracle printed next to the yardstick (no threshold); every output file
    both runs wrote byte-identical (the pickup among them)."""
    from mitjax.tests import advect_gate as ag
    m, res, o = C.whole_run(*SB)
    diffs, rows, nblocks = ag.run_verdict(*SB, res, o)
    print("solid-body digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    files, missing = C.output_file_diffs(m, o)
    print("solid-body output files compared:", sorted(files), "differing:",
          sorted(f for f, ok in files.items() if not ok), "oracle-only:", len(missing))
    assert nblocks == 26 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    assert files and all(files.values()), sorted(f for f, ok in files.items() if not ok)
    assert any(f.startswith("pickup") for f in files), sorted(files)


P6_SB = P6.replace('C.cube_run("adjustment.cs-32x32x1", "input")', 'C.cube_run("solid-body.cs-32x32x1", "input")')


def test_solid_body_sharded_p6():
    """solid-body at P=6 (subprocess, 6 fake devices: one face per device) == P=1 bitwise after 3 steps."""
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env["PYTHONPATH"] = str(paths.REPO)
    r = subprocess.run([sys.executable, "-c", P6_SB], env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "P6 {}" in r.stdout, r.stdout


def test_nlfs_initial_state(monkeypatch):
    """adjustment.cs-32x32x1/input.nlfs (nonlinFreeSurf = 3 without r*, p coordinates): INITIALISE_VARIA's
    CALC_SURF_DR(etaH, startTime, -1) + UPDATE_SURF_DR(.TRUE.) + UPDATE_CG2D + INTEGR_CONTINUITY + CALC_SURF_DR(etaH,
    startTime, nIter0): S00_begin bitwise (46 fields: hFacC/W/S, recip_hFac*, the CG2D operator, ...). Control:
    CALC_SURF_DR without its _EXCH_XY_RS of hFac_surfC (calc_surf_dr.F:214) -> hFacC differs on the halo ring that
    the C loop (0..sNy+1) does not reach."""
    r = C.CubeRun("adjustment.cs-32x32x1", "input.nlfs")
    bad, n = _initial_bad(r)
    assert n >= 46 and bad == {}, (n, bad)
    from mitjax.model.src import calc_surf_dr as CS
    monkeypatch.setattr(CS, "EXCH_XY_RS", lambda phi, *, ex: phi)
    bad, _ = _initial_bad(C.CubeRun("adjustment.cs-32x32x1", "input.nlfs"))
    assert "hFacC" in bad, sorted(bad)


NLFS = ("adjustment.cs-32x32x1", "input.nlfs")


def test_nlfs_steps_bitwise(monkeypatch):
    """adjustment.cs-32x32x1/input.nlfs (ATMOSPHERIC, nonlinFreeSurf = 3, select_rStar = 0): every stage FORWARD_STEP
    probes in steps 1-3 bitwise on every point of every tile vs lane A's jdon dumps, through the per-step
    UPDATE_SURF_DR(.TRUE.) (forward_step.F:852, the State's hFac_surf*), UPDATE_CG2D (:869) and CALC_SURF_DR(etaH)
    (:957). Control: CALC_SURF_DR dropped from FORWARD_STEP -> step 2's UPDATE_SURF_DR / UPDATE_CG2D read stale
    hFac_surf: its S08 operator (and hFac of S00 at iteration 2) differ (hFac_surf* themselves are not dumped)."""
    from mitjax.model.src import forward_step as FS
    r = C.CubeRun(*NLFS)
    out, _ = C.run_steps(r, 3)
    n = C.run_steps.ncompared
    print("input.nlfs stages compared:", dict(sorted(n.items())))
    assert {it for it, _ in out} == {0, 1, 2}
    assert all(len([1 for (i, _) in out if i == it]) >= 10 for it in (0, 1, 2)), sorted(out)
    assert {k: v for k, v in out.items() if v} == {}
    monkeypatch.setattr(FS, "nlfs_calc_r_star", lambda myTime, myIter, *, state, **kw: (state, None))
    out, _ = C.run_steps(C.CubeRun(*NLFS), 2)
    assert not any(out[k] for k in out if k[0] == 0), {k: v for k, v in out.items() if v}
    bad = set().union(*(v for k, v in out.items() if k[0] == 1))
    assert {"aW2d", "aS2d", "aC2d"} & bad, sorted(bad)


def test_nlfs_whole_run():
    """input.nlfs through the run driver: %MON records identical to the oracle STDOUT; testreport digits of every
    check-list variable vs results/ and vs the oracle printed next to the yardstick (Tsd/Uav/Vav are round-off
    with 0 digits there: recorded, no threshold); every output file both runs wrote byte-identical."""
    from mitjax.tests import advect_gate as ag
    m, res, o = C.whole_run(*NLFS)
    diffs, rows, nblocks = ag.run_verdict(*NLFS, res, o)
    print("input.nlfs digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    files, missing = C.output_file_diffs(m, o)
    print("input.nlfs output files compared:", sorted(files), "differing:",
          sorted(f for f, ok in files.items() if not ok), "oracle-only:", len(missing))
    assert nblocks == 21 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    assert all(files.values()), sorted(f for f, ok in files.items() if not ok)
    # round-off means (plan decisions 2026-10-02): Tsd, Uav, Vav have 0 digits vs results/ (yardstick 0); gated
    # relative to the field size (|ours - results/| / max(|min|, |max|) of the field <= 1e-13 at every monitor time)
    rel = C.roundoff_rel(res.records, *NLFS, ["Tsd", "Uav", "Vav"])
    print("input.nlfs round-off means, |ours - results/| / field size:", rel)
    assert all(x <= 1e-13 for x in rel.values()), rel
    # control: the last Vav record planted at 1e-12 of the field size off -> the gate fails
    recs = list(res.records)
    last = max(n for n, ln in enumerate(recs) if "%MON dynstat_vvel_mean" in ln)
    vmax = float(next(ln for ln in reversed(recs) if "%MON dynstat_vvel_max" in ln).split("=")[1])
    val = float(recs[last].split("=")[1])
    recs[last] = recs[last].split("=")[0] + "= " + f"{val + 1e-12*abs(vmax):.13E}"
    assert C.roundoff_rel(recs, *NLFS, ["Vav"])["Vav"] > 1e-13


CS32 = ("global_ocean.cs32x15", "input")


def test_cs32x15_initial_state(monkeypatch):
    """global_ocean.cs32x15/input (nIter0 = 72000 from its pickup; r*, ALLOW_NONHYDROSTATIC / ALLOW_SOLVE4_PS_AND_DRAG /
    ALLOW_ADDFLUID / SHORTWAVE_HEATING / ALLOW_BALANCE_FLUXES compiled with their switches off, cg2dTargetResWunit):
    the Model's initial carry vs S00_begin of lane A's jdon run, every field bitwise on every point of every tile.
    Control: the global files read without the exch2 IO map -> the initial State differs."""
    from mitjax.io import mds
    r = C.CubeRun(*CS32)
    bad, n = _initial_bad(r)
    print("cs32x15 S00_begin fields compared:", n, "bad:", sorted(bad))
    assert n >= 30 and bad == {}, (n, bad)
    tol = float(np.sqrt(r.m.cg2dh.cg2dTolerance_sq))
    assert f"{tol:.15E}" == "5.810494711292602E-07", tol      # STDOUT 'INI_CG2D: cg2dTolerance =' (1PE22.15)
    monkeypatch.setattr(mds.E2ioLayout, "from_w2", classmethod(lambda cls, w2, size: None))
    bad, _ = _initial_bad(C.CubeRun(*CS32))
    assert bad, "control did not bite"


def _cs32_p02_bad(monkeypatch, r):
    """DO_OCEANIC_PHYS of cs32x15's step 1 on the initial carry up to its P02_rho_sigma_ivdc probe (FREEZE_SURFACE,
    FIND_RHO_2D of every level, the upward k loop with GRAD_SIGMA's cube fills and CALC_IVDC) vs the oracle's P02
    dump, every field bitwise. EXTERNAL_FORCING_SURF and the GMREDI tensor / exchange (not ported for this run: SHORTWAVE
    / BALANCE forcing, GM_ExtraDiag, GM_AdvForm) are replaced by pass-throughs: P02 does not read what they write."""
    from mitjax.model.src import do_oceanic_phys as DOP
    import mitjax.pkg.gmredi.gmredi_calc_tensor as GCT
    import mitjax.pkg.gmredi.gmredi_do_exch as GDE
    monkeypatch.setattr(DOP, "external_forcing_surf",
                        lambda *a, state, phi0surf, **k: (a[6], state, phi0surf))     # a[6]: ff
    monkeypatch.setattr(GCT, "gmredi_calc_tensor", lambda *a, gm, **k: gm)
    monkeypatch.setattr(GDE, "gmredi_do_exch", lambda *a, gm, **k: gm)
    m = r.m
    c = m.initial_carry()
    t, it = m.start_counters()
    got = {}
    DOP.do_oceanic_phys(t, it, cfg=m.cfg, grid=m.grid, params=m.params, fp=m.fp, eos=m.eos, state=c[0], ff=c[1],
                        phi0surf=c[2], probe=lambda st, v: got.update({st: v}), gm=c[4]["gm"],
                        gm_params=m.params)
    res = C.compare_stage(r, r.its[0], "P02_rho_sigma_ivdc", got["P02_rho_sigma_ivdc"])
    return C.bad(res), len(res)


def test_cs32x15_grad_sigma_cube(monkeypatch):
    """GRAD_SIGMA's cubed-sphere fills (FILL_CS_CORNER_TR_RL 1 and 2 on rhoLoc, grad_sigma.F:59-62, :72-75) in
    cs32x15's first DO_OCEANIC_PHYS: P02_rho_sigma_ivdc (sigmaX, sigmaY, sigmaR, rhoInSitu, IVDConvCount, ...)
    bitwise (`_cs32_p02_bad`). Control: the fills dropped -> sigmaX / sigmaY differ in the tile corners."""
    from mitjax.model.src import grad_sigma as GS
    r = C.CubeRun(*CS32)
    bad, n = _cs32_p02_bad(monkeypatch, r)
    print("cs32x15 P02 fields compared:", n, "bad:", sorted(bad))
    assert n >= 6 and bad == {}, (n, bad)
    monkeypatch.setattr(GS, "_fill_cs", lambda d, fld, sz, g: fld)
    bad, _ = _cs32_p02_bad(monkeypatch, r)
    assert {"sigmaX", "sigmaY"} & set(bad), sorted(bad)


MIN = ("adjustment.cs-32x32x1", "input_min")


def _min_printout_bad(w2, io):
    """{'log': differing records of w2_tile_topology.0000.log (or length mismatch), 'stdout': differing W2 records
    of STDOUT} vs lane A's plain run of the minimal case (code_min: EEBOOT and the W2 set-up only)."""
    from mitjax.eesupp.print import STANDARD_MESSAGE_UNIT
    rundir = C.run_top(*MIN, kind="minimal") / "rundir"
    log = (rundir / "w2_tile_topology.0000.log").read_text().split("\n")
    assert log[-1] == ""
    log = log[:-1]
    out = (rundir / "output.txt").read_text().split("\n")
    ours, so = io.records(w2.W2_oUnit), io.records(STANDARD_MESSAGE_UNIT)
    i0 = out.index(so[0])
    return {"log": abs(len(log) - len(ours)) + sum(a != b for a, b in zip(log, ours)),
            "stdout": sum(a != b for a, b in zip(out[i0:i0 + len(so)], so)) + max(0, i0 + len(so) - len(out)),
            "log_records": len(ours), "stdout_records": len(so)}


def test_adjustment_cs_min_w2_printout(monkeypatch):
    """adjustment.cs-32x32x1/input_min (code_min: genmake2 -standarddirs eesupp + exch2 + debug; main.F runs EEBOOT
    only, so no physics): our W2_EEBOOT print-out equals the oracle's character for character -- the topology report
    w2_tile_topology.0000.log (W2_printMsg = -1; 1228 records, without W2_CUMSUM_USE_MATRIX, code_min's default
    W2_OPTIONS.h) and the W2 lines of STDOUT (9 records). Control: W2_myTileList(1,1)/(2,1) swapped after
    W2_MAP_PROCS -> the log differs."""
    from mitjax.config.params import load
    from mitjax.eesupp.print import MessageUnits
    from mitjax.pkg.exch2 import w2_eeboot as EB
    e = load(*MIN)
    assert e.cfg.code_dir == "code_min" and not e.cfg.cpp.W2_CUMSUM_USE_MATRIX
    w2, io = EB.w2_eeboot(e)
    bad = _min_printout_bad(w2, io)
    assert bad == {"log": 0, "stdout": 0, "log_records": 1228, "stdout_records": 9}, bad
    orig = EB.w2_map_procs

    def swapped(w2, **kw):
        out = orig(w2, **kw)
        a, b = w2.W2_myTileList[1, 1], w2.W2_myTileList[2, 1]
        w2.W2_myTileList[1, 1], w2.W2_myTileList[2, 1] = b, a
        return out
    monkeypatch.setattr(EB, "w2_map_procs", swapped)
    w2, io = EB.w2_eeboot(e, MessageUnits())
    assert _min_printout_bad(w2, io)["log"] > 0


def test_single_cpu_writer_roundtrip(tmp_path):
    """MDS_WRITE_FIELD's useSingleCpuIO branch (mdsio_write_field.F:269-364, GATHER_2D_R4 with the exch2 global layout)
    without a model run: the oracle's Eta / U snapshots (float32, 192 x 32 global map) read back into tiles
    (MDS_READ_FIELD's exch2 layout, mds.E2ioLayout) and written again give the oracle's .data and .meta byte for
    byte. Control: the same write without the exch2 layout (Nx x Ny of SIZE.h, tiles side by side) differs."""
    import filecmp
    import numpy as np
    from mitjax.farray import FArray
    from mitjax.io import mds as M
    from mitjax.pkg.mdsio.mdsio_write_field import MdsContext, mds_write_field
    from mitjax.tests import monitor_gate as mg
    r = C.cube_run(*EXP)
    od = mg.oracle(*EXP).stdout_path.parent
    w2 = r.m.w2 if getattr(r.m, "w2", None) is not None else None
    if w2 is None:
        from mitjax.pkg.exch2.w2_eeboot import w2_eeboot
        w2 = w2_eeboot(r.e)[0]
    sz = r.e.cfg.size
    e2io = M.E2ioLayout.from_w2(w2, sz)
    L = M.MdsLayout(sz.sNx, sz.sNy, sz.nSx, sz.nSy)
    for name, it in (("Eta", 12), ("U", 24)):
        base = f"{name}.{it:010d}"
        vals = M.read_field(od / f"{base}.data", 32, L, nNz=1, irecord=1, e2io=e2io)
        arr = M.into_tiles(vals, sz.OLx, sz.OLy)[:, 0]
        fld = FArray(arr, name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))
        ctx = MdsContext(tmp_path, sz, exch2=True, w2=w2, useSingleCpuIO=True)
        mds_write_field(base, 32, False, False, "RL", 1, 1, 1, fld, 1, it, mds=ctx)
        assert filecmp.cmp(tmp_path / f"{base}.data", od / f"{base}.data", shallow=False), base
        assert filecmp.cmp(tmp_path / f"{base}.meta", od / f"{base}.meta", shallow=False), base
    d2 = tmp_path / "control"
    d2.mkdir()
    ctx = MdsContext(d2, sz, exch2=False, useSingleCpuIO=True)
    mds_write_field(base, 32, False, False, "RL", 1, 1, 1, fld, 1, it, mds=ctx)
    assert not filecmp.cmp(d2 / f"{base}.data", od / f"{base}.data", shallow=False)
    assert np.fromfile(d2 / f"{base}.data", ">f4").size == np.fromfile(od / f"{base}.data", ">f4").size
