"""Gates of the M3 sub-lane COLMIX: pkg/pp81, pkg/my82, pkg/opps against gfortran (plan Task 29).

Oracle: the COLMIX replay harness (reference/replay_colmix; vermix/code build, THE_MAIN_LOOP replaced), one run per
variant (input.pp81, input.my82, input.opps; OPPS_CHECK refuses OPPS together with PP81 or MY82), listed in
reference/replay_colmix/CURRENT (run directories relative to $MJX_RUNS). Each run holds 48 synthetic samples of a
5 x 5 (1 x 1 + halos 2) x 26-level tile with land columns, variable column depth, sheared and statically unstable
profiles (reference/replay_colmix/replay_io.py), replayed per pass: 1 = the namelist eosType (MDJWF), 2 = JMD95Z
(INI_EOS re-run by the harness), 3 = JMD95Z with useGCMwVel = .TRUE. (OPPS). The 48 samples are fed to the ported
routines as 48 tiles (the routines are tile-local). Second oracle: vermix's substep dumps (P08_pp81, P09_my82,
T05_opps), teacher-forced, of lane A's registered jdon runs (steps 0-2) and of the lane's all-steps runs of the same
binary (reference/replay_colmix/JDALL, steps 0-19).

Gates: bitwise (bit patterns, all points incl. halos and land, both finite) for the INIT_VARIA fields, CALC outputs,
CALC_DIFF (kArg = 0 and kArg = k), CALC_VISC (k = 1..Nr), OPPS_INTERFACE's theta/salt and OPPS_CALC's convection
count; READPARMS-derived values (PP81 RiLimit, MY82 alpha/beta, OPPS e2) exact. Negative controls: a planted
rounding-level change in each package, measured to bite. Gradients: finite on every lane, tangent vs adjoint dot
test, central FD at smooth points. All tier1x (no tier-1 test). JAX transforms are used here, never in mitjax/pkg.
"""

import functools
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax import paths
from mitjax.farray import FArray
from mitjax.tests.col_replay import Common, bit_equal

REPO = Path(__file__).resolve().parents[2]


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rio = _load_by_path("_mjx_replay_colmix_io", REPO / "reference" / "replay_colmix" / "replay_io.py")


def current_runs():
    out = {}
    for ln in (REPO / "reference" / "replay_colmix" / "CURRENT").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            exp, inp, rel = ln.split()
            out[inp] = Path(paths.RUNS) / rel
    return out


class _Size:
    """vermix's SIZE.h with the 48 samples as tiles (nSx = NSAMP)."""

    def __init__(self, sz, ntiles):
        for k in ("sNx", "sNy", "OLx", "OLy", "Nr", "nPx", "nPy"):
            setattr(self, k, getattr(sz, k))
        self.nSx, self.nSy = ntiles, 1


class _Cfg:
    def __init__(self, c, ntiles):
        self.cpp, self._c = c.cpp, c
        self.size = _Size(c.size, ntiles)

    def use_flag(self, name):
        return self._c.use_flag(name)


@functools.lru_cache(maxsize=None)
def experiment(inp):
    from mitjax.config import params as cp
    return cp.load("vermix", inp)


class Replay:
    """One harness run: inputs, Fortran outputs, dumped parameters; the commons for samples `sel` as tiles."""

    def __init__(self, inp):
        runs = current_runs()
        if inp not in runs:
            raise FileNotFoundError(f"no COLMIX replay run for vermix/{inp} in reference/replay_colmix/CURRENT")
        self.inp, self.rundir = inp, runs[inp]
        self.exp = experiment(inp)
        sz = self.exp.cfg.size
        self.sz = {k: getattr(sz, k) for k in rio.SIZE_KEYS}
        self.inputs = rio.read_inputs(self.rundir, self.sz)
        self.out = rio.read_outputs(self.rundir, self.sz)
        self.g = rio.read_grid(self.rundir)
        self.orig = rio.read_orig(self.rundir, self.sz)

    def cfg(self, n):
        return _Cfg(self.exp.cfg, n)

    def samples(self, name, sel):
        """[n, tile=1, ...] -> [n (as tiles), ...] for the selected samples."""
        a = self.inputs[name] if name in self.inputs else self.out[name]
        return a[sel][:, 0]

    def commons(self, sel, eos_type="JMD95Z"):
        """(cfg, grid, params, eos, state) for the samples `sel` (a slice or index array) as tiles."""
        n = len(np.arange(rio.NSAMP)[sel])
        cfg = self.cfg(n)
        s = cfg.size
        Nr = s.Nr
        ij = dict(i=(1-s.OLx, s.sNx+s.OLx), j=(1-s.OLy, s.sNy+s.OLy))
        f3 = lambda name: FArray(jnp.asarray(self.samples(name, sel)), name, k=(1, Nr), **ij)   # noqa: E731
        g = self.g
        grid = Common(maskC=f3("maskC"), maskW=f3("maskW"), maskS=f3("maskS"),
                      kLowC=FArray(jnp.asarray(self.samples("kLowC", sel).astype(np.int32)), "kLowC", **ij),
                      kSurfC=FArray(jnp.asarray(self.samples("kSurfC", sel).astype(np.int32)), "kSurfC", **ij),
                      rC=FArray(jnp.asarray(g["rC"]), "rC", k=(1, Nr), tiled=False),
                      rF=FArray(jnp.asarray(g["rF"]), "rF", k=(1, Nr+1), tiled=False),
                      drF=FArray(jnp.asarray(g["drF"]), "drF", k=(1, Nr), tiled=False),
                      recip_drF=FArray(jnp.asarray(g["recip_drF"]), "recip_drF", k=(1, Nr), tiled=False),
                      recip_drC=FArray(jnp.asarray(g["recip_drC"]), "recip_drC", k=(1, Nr+1), tiled=False))
        vec = lambda name, n_: FArray(jnp.asarray(g[name]), name, k=(1, n_), tiled=False)   # noqa: E731
        par = dict(viscArNr=vec("viscArNr", Nr), diffKrNrS=vec("diffKrNrS", Nr), dTtracerLev=vec("dTtracerLev", Nr),
                   tRef=vec("tRef", Nr), sRef=vec("sRef", Nr), pRef4EOS=vec("pRef4EOS", Nr),
                   phiRef=vec("phiRef", 2*Nr+1))
        par.update({k: jnp.float64(g[k]) for k in ("rhoConst", "rhoNil", "gravity", "mass2rUnit", "surf_pRef")})
        static = {"selectP_inEOS_Zc": int(g["selectP_inEOS_Zc"]), "usingPCoords": bool(g["usingPCoords"]),
                  "usingZCoords": not bool(g["usingPCoords"])}
        params = Common(static, {}, **par)
        eos = our_eos(eos_type, bool(g["usingPCoords"]))
        state = Common(theta=f3("theta"), salt=f3("salt"), uVel=f3("uVel"), vVel=f3("vVel"), wVel=f3("wVel"),
                       totPhiHyd=f3("totPhiHyd"))
        return cfg, grid, params, eos, state


def our_eos(eos_type, usingPCoords=False):
    """EOS.h from our INI_EOS port for eosType (gated against the harness's own EOS.h in test_ini_eos_coefficients)."""
    from types import SimpleNamespace
    from mitjax.model.src.ini_eos import UNSET_RL, ini_eos
    p = SimpleNamespace(fluidIsWater=True, eosType=eos_type.ljust(6)[:6], usingPCoords=usingPCoords,
                        tAlpha=UNSET_RL, sBeta=UNSET_RL)
    cfg = SimpleNamespace(cpp=SimpleNamespace(TARGET_NEC_SX=False))
    return ini_eos(cfg=cfg, params=p)


@functools.lru_cache(maxsize=None)
def replay(inp):
    return Replay(inp)


def check(label, got, want):
    nd, nt, fin = bit_equal(got, want)
    print(f"{label}: {nt - nd}/{nt} points bitwise equal, finite={fin}")
    assert fin, f"{label}: non-finite values"
    assert nd == 0, f"{label}: {nd} of {nt} points differ (max |diff| " \
                    f"{np.max(np.abs(np.asarray(got) - np.asarray(want))):.3e})"


ALL = slice(None)


# ---------------------------------------------------------------------------------------------------------------
# PP81

def pp81_common(R, sel, mod=None):
    from mitjax.pkg.pp81.pp81_h import PP81, declare
    g = R.g
    pp = PP81({"PP81isOn": True, "PPnRi": int(g["PPnRi"]),
               **{k: np.float64(g[k]) for k in ("PPviscMin", "PPdiffMin", "PPviscMax", "PPnu0", "PPalpha",
                                                "RiLimit")}})
    cfg = R.cfg(len(np.arange(rio.NSAMP)[sel]))
    return pp.replace(PPviscAr=declare("PPviscAr", cfg.size, jnp.asarray(R.samples("viscAr0", sel))),
                      PPdiffKr=declare("PPdiffKr", cfg.size, jnp.asarray(R.samples("diffKr0", sel))))


def run_pp81(R, sel, eos_type="JMD95Z", calc=None):
    """PP81_CALC, PP81_CALC_DIFF (kArg 0 and k), PP81_CALC_VISC (k = 1..Nr) as the harness calls them."""
    from mitjax.pkg.pp81 import pp81_calc, pp81_calc_diff, pp81_calc_visc
    calc = calc or pp81_calc.pp81_calc
    cfg, grid, params, eos, state = R.commons(sel, eos_type)
    pp = pp81_common(R, sel)
    s = cfg.size
    f3 = lambda name: FArray(jnp.asarray(R.samples(name, sel)), name, k=(1, s.Nr),   # noqa: E731
                             i=(1-s.OLx, s.sNx+s.OLx), j=(1-s.OLy, s.sNy+s.OLy))

    def f(grid, params, eos, state, pp, kX0, kU, kV):
        pp = calc(None, 0.0, 0, cfg=cfg, grid=grid, params=params, eos=eos, state=state, pp=pp)
        full = (1-s.OLx, s.sNx+s.OLx, 1-s.OLy, s.sNy+s.OLy)
        kXa = pp81_calc_diff.pp81_calc_diff(*full, 0, s.Nr, kX0, cfg=cfg, params=params, pp=pp)
        kXk = kX0
        for k in range(1, s.Nr+1):
            kXk = pp81_calc_diff.pp81_calc_diff(*full, k, s.Nr, kXk, cfg=cfg, params=params, pp=pp)
            kU, kV = pp81_calc_visc.pp81_calc_visc(2-s.OLx, s.sNx+s.OLx, 2-s.OLy, s.sNy+s.OLy, k, kU, kV,
                                                   grid=grid, params=params, pp=pp)
        return pp.PPviscAr.data, pp.PPdiffKr.data, kXa.data, kXk.data, kU.data, kV.data

    return jax.jit(f)(grid, params, eos, state, pp, f3("kappaRx0"), f3("kappaRU0"), f3("kappaRV0"))


OUT_NAMES = ("vis", "dif", "kapX0", "kapXk", "kapU", "kapV")


def test_pp81_readparms_and_init_varia():
    """PP81_READPARMS (RiLimit from viscArNr(1) as READPARMS saw it, the other parameters) and PP81_INIT_VARIA."""
    from mitjax.pkg.pp81.pp81_init_varia import pp81_init_varia
    from mitjax.pkg.pp81.pp81_readparms import pp81_readparms
    R = replay("input.pp81")
    g = R.g
    Nr = R.sz["Nr"]
    pp = pp81_readparms(R.exp, params=Common(viscArNr=FArray(jnp.asarray(g["viscArNr_orig"]), "viscArNr",
                                                             k=(1, Nr), tiled=False)))
    for k in ("PPviscMin", "PPdiffMin", "PPviscMax", "PPnu0", "PPalpha", "RiLimit"):
        assert np.float64(pp.__getattr__(k)).tobytes() == np.float64(g[k]).tobytes(), (k, pp.__getattr__(k), g[k])
    assert pp.PPnRi == int(g["PPnRi"])
    cfg, grid, params, eos, state = R.commons(slice(0, 1))
    cfg1 = R.cfg(1)
    pp = pp81_init_varia(cfg=cfg1, params=params, pp=pp)
    check("PP81_INIT_VARIA PPviscAr", pp.PPviscAr.data, R.out["ini_vis"])
    check("PP81_INIT_VARIA PPdiffKr", pp.PPdiffKr.data, R.out["ini_dif"])


def test_pp81_replay_jmd95z():
    """PP81_CALC / CALC_DIFF / CALC_VISC bitwise on all 48 samples (pass 2: JMD95Z)."""
    R = replay("input.pp81")
    got = run_pp81(R, ALL)
    for name, a in zip(OUT_NAMES, got):
        check(f"pp81 {name}", a, R.out[f"{name}_p2"][:, 0])


def test_pp81_replay_mdjwf():
    """PP81_CALC / CALC_DIFF / CALC_VISC bitwise on all 48 samples, pass 1: vermix's own eosType MDJWF."""
    R = replay("input.pp81")
    got = run_pp81(R, ALL, eos_type="MDJWF")
    for name, a in zip(OUT_NAMES, got):
        check(f"pp81 {name} (MDJWF)", a, R.out[f"{name}_p1"][:, 0])


def _planted(modname, old, new):
    """A copy of mitjax.pkg.<modname> with `old` replaced by `new` (exactly one occurrence)."""
    path = REPO / "mitjax" / "pkg" / (modname.replace(".", "/") + ".py")
    src = path.read_text()
    assert src.count(old) == 1, f"planted: {old!r} occurs {src.count(old)} times in {path.name}"
    name = f"_mjx_planted_{modname.replace('.', '_')}_{abs(hash((old, new)))}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src.replace(old, new), str(path), "exec"), mod.__dict__)
    return mod


def test_pp81_negative_control():
    """A rounding-level re-association in PP81_CALC (PPviscAr/denom -> PPviscAr*(1/denom)) must fail the gate."""
    R = replay("input.pp81")
    bad = _planted("pp81.pp81_calc", "MAX(PPviscAr[i, j, K]/denom,", "MAX(PPviscAr[i, j, K]*(1.0/denom),")
    got = run_pp81(R, ALL, calc=bad.pp81_calc)
    nd, nt, _ = bit_equal(got[1], R.out["dif_p2"][:, 0])
    print(f"planted PPdiffKr: {nd}/{nt} points differ")
    assert nd > 0


# ---------------------------------------------------------------------------------------------------------------
# MY82

def my82_common(R, sel):
    from mitjax.pkg.my82.my82_h import MY82, declare
    g = R.g
    my = MY82({"MYisOn": True, **{k: np.float64(g[k]) for k in ("alpha1", "alpha2", "beta1", "beta2", "beta3",
                                                                 "beta4", "RiMax", "MYhblScale", "MYviscMax",
                                                                 "MYdiffMax")}})
    cfg = R.cfg(len(np.arange(rio.NSAMP)[sel]))
    return my.replace(MYviscAr=declare("MYviscAr", cfg.size, jnp.asarray(R.samples("viscAr1", sel))),
                      MYdiffKr=declare("MYdiffKr", cfg.size, jnp.asarray(R.samples("diffKr1", sel))),
                      MYhbl=declare("MYhbl", cfg.size, jnp.asarray(R.samples("MYhbl0", sel))))


def run_my82(R, sel, eos_type="JMD95Z", calc=None):
    from mitjax.pkg.my82 import my82_calc, my82_calc_diff, my82_calc_visc
    calc = calc or my82_calc.my82_calc
    cfg, grid, params, eos, state = R.commons(sel, eos_type)
    my = my82_common(R, sel)
    s = cfg.size
    f3 = lambda name: FArray(jnp.asarray(R.samples(name, sel)), name, k=(1, s.Nr),   # noqa: E731
                             i=(1-s.OLx, s.sNx+s.OLx), j=(1-s.OLy, s.sNy+s.OLy))

    def f(grid, params, eos, state, my, kX0, kU, kV):
        my = calc(None, 0.0, 0, cfg=cfg, grid=grid, params=params, eos=eos, state=state, my=my)
        full = (1-s.OLx, s.sNx+s.OLx, 1-s.OLy, s.sNy+s.OLy)
        kXa = my82_calc_diff.my82_calc_diff(*full, 0, s.Nr, kX0, cfg=cfg, params=params, my=my)
        kXk = kX0
        for k in range(1, s.Nr+1):
            kXk = my82_calc_diff.my82_calc_diff(*full, k, s.Nr, kXk, cfg=cfg, params=params, my=my)
            kU, kV = my82_calc_visc.my82_calc_visc(2-s.OLx, s.sNx+s.OLx, 2-s.OLy, s.sNy+s.OLy, k, kU, kV,
                                                   grid=grid, params=params, my=my)
        return my.MYviscAr.data, my.MYdiffKr.data, kXa.data, kXk.data, kU.data, kV.data, my.MYhbl.data

    return jax.jit(f)(grid, params, eos, state, my, f3("kappaRx0"), f3("kappaRU0"), f3("kappaRV0"))


def test_my82_readparms_and_init_varia():
    """MY82_READPARMS (MYviscMax, MYdiffMax, MYhblScale, RiMax) and MY82_INIT_VARIA (alpha1..beta4, fields)."""
    from mitjax.pkg.my82.my82_init_varia import my82_init_varia
    from mitjax.pkg.my82.my82_readparms import my82_readparms
    R = replay("input.my82")
    g = R.g
    my = my82_readparms(R.exp)
    cfg, grid, params, eos, state = R.commons(slice(0, 1))
    my = my82_init_varia(cfg=R.cfg(1), params=params, my=my)
    for k in ("MYviscMax", "MYdiffMax", "MYhblScale", "RiMax", "alpha1", "alpha2", "beta1", "beta2", "beta3",
              "beta4"):
        assert np.float64(my.__getattr__(k)).tobytes() == np.float64(g[k]).tobytes(), (k, my.__getattr__(k), g[k])
    check("MY82_INIT_VARIA MYviscAr", my.MYviscAr.data, R.out["ini_vis"])
    check("MY82_INIT_VARIA MYdiffKr", my.MYdiffKr.data, R.out["ini_dif"])
    check("MY82_INIT_VARIA MYhbl", my.MYhbl.data, R.out["ini_hbl"])


def test_my82_replay_jmd95z():
    """MY82_CALC / CALC_DIFF / CALC_VISC bitwise on all 48 samples (pass 2: JMD95Z), MYhbl included."""
    R = replay("input.my82")
    got = run_my82(R, ALL)
    for name, a in zip(OUT_NAMES, got[:6]):
        check(f"my82 {name}", a, R.out[f"{name}_p2"][:, 0])
    check("my82 MYhbl", got[6], R.out["hbl_p2"][:, 0])


def test_my82_replay_mdjwf():
    """MY82 routines bitwise on all 48 samples, pass 1: vermix's own eosType MDJWF."""
    R = replay("input.my82")
    got = run_my82(R, ALL, eos_type="MDJWF")
    for name, a in zip(OUT_NAMES, got[:6]):
        check(f"my82 {name} (MDJWF)", a, R.out[f"{name}_p1"][:, 0])
    check("my82 MYhbl (MDJWF)", got[6], R.out["hbl_p1"][:, 0])


def test_my82_negative_control():
    """A rounding-level re-association in MY82_CALC (SM's quotient) must fail the gate."""
    R = replay("input.my82")
    bad = _planted("my82.my82_calc", "SHtmp*(beta1-beta2*RiFlux)/(beta3-beta4*RiFlux))",
                   "SHtmp*((beta1-beta2*RiFlux)/(beta3-beta4*RiFlux)))")
    got = run_my82(R, ALL, calc=bad.my82_calc)
    nd, nt, _ = bit_equal(got[0], R.out["vis_p2"][:, 0])
    print(f"planted MYviscAr: {nd}/{nt} points differ")
    assert nd > 0


# ---------------------------------------------------------------------------------------------------------------
# OPPS

def opps_common(R, gcm):
    from mitjax.pkg.opps.opps_h import OPPS
    g = R.g
    return OPPS({"OPPSisOn": True, "useGCMwVel": bool(gcm), "MAX_ABE_ITERATIONS": int(g["MAX_ABE_ITERATIO"]),
                 "OPPSdebugLevel": int(g["OPPSdebugLevel"]),
                 "PlumeRadius": np.float64(g["PlumeRadius"]), "STABILITY_THRESHOLD": np.float64(g["STABILITY_THRESH"]),
                 "FRACTIONAL_AREA": np.float64(g["FRACTIONAL_AREA"]),
                 "MAX_FRACTIONAL_AREA": np.float64(g["MAX_FRACTIONAL_A"]),
                 "VERTICAL_VELOCITY": np.float64(g["VERTICAL_VELOCIT"]),
                 "ENTRAINMENT_RATE": np.float64(g["ENTRAINMENT_RATE"]), "e2": np.float64(g["e2"])})


def run_opps(R, sel, gcm, eos_type="JMD95Z", nTimeMax=None, interface=None, counters=False):
    """OPPS_INTERFACE on the full tile as the harness calls it; nTimeMax None = the model's bound opps_h.NTIME_MAX."""
    from mitjax.pkg.opps import opps_h, opps_interface
    interface = interface or opps_interface.opps_interface
    nTimeMax = opps_h.NTIME_MAX if nTimeMax is None else nTimeMax
    cfg, grid, params, eos, state = R.commons(sel, eos_type)
    op = opps_common(R, gcm)
    s = cfg.size

    def f(grid, params, eos, state, op):
        th, sa, cnt, ctr = interface(1-s.OLx, s.sNx+s.OLx, 1-s.OLy, s.sNy+s.OLy, 0.0, 0, cfg=cfg, grid=grid,
                                     params=params, eos=eos, state=state, op=op, nTimeMax=nTimeMax)
        return th.data, sa.data, cnt, ctr

    th, sa, cnt, ctr = jax.jit(f)(grid, params, eos, state, op)
    return (th, sa, cnt, ctr) if counters else (th, sa, cnt)


def test_opps_readparms():
    from mitjax.pkg.opps.opps_readparms import opps_readparms
    R = replay("input.opps")
    op = opps_readparms(R.exp)
    ref = opps_common(R, False)
    for k in ("PlumeRadius", "STABILITY_THRESHOLD", "FRACTIONAL_AREA", "MAX_FRACTIONAL_AREA", "VERTICAL_VELOCITY",
              "ENTRAINMENT_RATE", "e2"):
        assert np.float64(op.__getattr__(k)).tobytes() == np.float64(ref.__getattr__(k)).tobytes(), k
    assert op.MAX_ABE_ITERATIONS == ref.MAX_ABE_ITERATIONS and op.useGCMwVel is False


@pytest.mark.parametrize("p,gcm,eos_type", [(1, False, "MDJWF"), (2, False, "JMD95Z"), (3, True, "JMD95Z")],
                         ids=["mdjwf", "jmd95z", "jmd95z-gcmw"])
def test_opps_replay(p, gcm, eos_type):
    """OPPS_INTERFACE theta/salt and OPPS_CALC's convection count bitwise on all 48 samples (the model's static
    time-loop bound opps_h.NTIME_MAX; no overflow)."""
    from mitjax.pkg.opps import opps_calc, opps_h
    R = replay("input.opps")
    th, sa, cnt, ctr = run_opps(R, ALL, gcm, eos_type=eos_type, counters=True)
    opps_calc.opps_calc_host(ctr, 0, opps_h.NTIME_MAX)                      # raises on an overflow
    check(f"opps theta p{p}", th, R.out[f"theta_p{p}"][:, 0])
    check(f"opps salt p{p}", sa, R.out[f"salt_p{p}"][:, 0])
    check(f"opps count p{p}", cnt, R.out[f"cnt_p{p}"][:, 0])
    c = R.out[f"cnt_p{p}"][:, 0]
    print(f"plumes that mixed: {int(c.sum())}; columns with >= 2 plumes: {int(np.sum(c.sum(axis=1) >= 2))}")
    assert c.sum() > 0


def test_opps_time_loop_bound():
    """A too small bound is never silent: nTimeMax = 1 gives NaN on the overflowing columns only (every finite point
    is the oracle's), a nonzero per-tile count, and opps_calc_host STOPs."""
    from mitjax.pkg.opps import opps_calc
    R = replay("input.opps")
    th1, _, _, ctr = run_opps(R, ALL, False, nTimeMax=1, counters=True)
    want = R.out["theta_p2"][:, 0]
    bad = ~np.isfinite(np.asarray(th1))
    same = np.asarray(th1)[~bad] == want[~bad]
    n = int(np.sum(np.asarray(ctr["ntimeOver"])))
    print(f"nTimeMax=1: {n} overflowing plumes, {int(bad.sum())} NaN points, {int((~same).sum())} finite points "
          "differing")
    assert n > 0 and bad.sum() > 0 and same.all()
    with pytest.raises(RuntimeError, match="ntime > nTimeMax"):
        opps_calc.opps_calc_host(ctr, 1, 1)


def test_opps_negative_control():
    """Two planted errors must fail the gate: the time-step association (rounding level) and STATE1's kRef."""
    from mitjax.pkg.opps import opps_interface
    R = replay("input.opps")
    want = R.out["theta_p2"][:, 0]
    for old, new in (("t = jnp.where(on3 & inner, t + (Fm1 - F)*dt3*rdr, t)",
                      "t = jnp.where(on3 & inner, t + (Fm1 - F)*(dt3*rdr), t)"),
                     ("t = jnp.where(on3 & top, t - F*dt3*rdr, t)", "t = jnp.where(on3 & top, t - F*(dt3*rdr), t)"),
                     ("Pk2 = _lev(P, k2+1, 1)", "Pk2 = _lev(P, k2, 1)")):
        bad = _planted("opps.opps_calc", old, new)
        src = (REPO / "mitjax/pkg/opps/opps_interface.py").read_text()
        name = f"_mjx_planted_opps_interface_{abs(hash((old, new)))}"
        spec = importlib.util.spec_from_loader(name, loader=None)
        mod = importlib.util.module_from_spec(spec)
        exec(compile(src, "opps_interface.py", "exec"), mod.__dict__)
        mod.opps_calc = bad.opps_calc
        th, _, _ = run_opps(R, ALL, False, interface=mod.opps_interface)
        nd, nt, _ = bit_equal(th, want)
        print(f"planted {new!r}: {nd}/{nt} theta points differ")
        assert nd > 0, new
    del opps_interface


# ---------------------------------------------------------------------------------------------------------------
# INI_EOS (MDJWF, JMD95Z) and the vermix PARAMS.h values of our INI_PARMS vs what the oracle's INITIALISE_FIXED left

def test_ini_eos_coefficients():
    """INI_EOS: our MDJWF coefficients == the harness's EOS.h before any override (vermix's namelist eosType), and
    our JMD95Z coefficients == the harness's EOS.h after its INI_EOS re-run (bit patterns)."""
    R = replay("input.opps")
    mine = our_eos("MDJWF")
    for name in ("eosMDJWFnum", "eosMDJWFden"):
        check(f"INI_EOS MDJWF {name}", np.asarray(getattr(mine, name).data), R.orig[name])
    assert float(mine.eosRefP0) == R.orig["eosRefP0"]
    mine = our_eos("JMD95Z")
    for name in ("eosJMDCFw", "eosJMDCSw", "eosJMDCKFw", "eosJMDCKSw", "eosJMDCKP"):
        check(f"INI_EOS JMD95Z {name}", np.asarray(getattr(mine, name).data), R.g[name])


def vermix_params(inp):
    """{viscArNr, diffKrNrS} (Nr) of our INI_PARMS for vermix/inp: ini_parms_dyn, ini_parms_tracer."""
    from mitjax.model.src import ini_parms as ip
    from mitjax.model.src import ini_parms_tracer as ipt
    exp = experiment(inp)
    mp = ip.ini_parms(exp)
    P = ip.ini_parms_dyn(exp, mp.grid, mp.time, mp.init)
    out = {"viscArNr": np.asarray(P.viscArNr.data)}
    T = ipt.ini_parms_tracer(exp, P, mp.time, mp.init)                     # vermix sets diffKzS (old name)
    out["diffKrNrS"] = np.asarray(T.diffKrNrS.data)
    return out


def test_vermix_params_vs_oracle():
    """viscArNr, diffKrNrS of our INI_PARMS == what vermix's INITIALISE_FIXED left (cm_orig.bin)."""
    R = replay("input.pp81")
    mine = vermix_params("input.pp81")
    for name in ("viscArNr", "diffKrNrS"):
        check(f"INI_PARMS {name}", mine[name], R.orig[name])


# ---------------------------------------------------------------------------------------------------------------
# in-model gates against the oracle's substep dumps of vermix (reference/reference_runs.py, kind jdon), teacher-forced:
# PP81_CALC == P08_pp81, MY82_CALC == P09_my82, OPPS_INTERFACE == T05_opps at every dumped iteration, all points.
# Grid from the dump's G00_geometry, EOS from our INI_EOS (MDJWF), the EOS / time parameters INI_PARMS does not
# provide yet (phiRef, pRef4EOS, surf_pRef, selectP_inEOS_Zc, dTtracerLev, rhoConst, gravity, mass2rUnit) from the
# harness's dump of the same build (INITIALISE_FIXED values; the harness overrides none of them); viscArNr from our
# ini_parms_dyn, diffKrNrS from the oracle (`params_src="oracle"`) or from our ini_parms_tracer
# (`params_src="ini_parms"`; the diffKzS alias came with GGL90 session 2).

def _jdall_runs():
    """{input_dir: run top} of the lane's all-steps dump runs (reference/replay_colmix/JDALL, relative to $MJX_RUNS):
    lane A's frozen jaxdump binary of vermix with JAXDUMP_STEPS = 0..19 (the registered jdon runs dump 0..2, where
    OPPS mixes nothing)."""
    out = {}
    for ln in (REPO / "reference" / "replay_colmix" / "JDALL").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            exp, inp, rel = ln.split()
            out[inp] = Path(paths.RUNS) / rel
    return out


@functools.lru_cache(maxsize=None)
def dumpset(inp, kind="jdon"):
    """The substep dumps of vermix/inp: kind "jdon" = lane A's registered run (steps 0..2), "jdall" = the lane's
    all-steps run of the same binary."""
    sys.path.insert(0, str(REPO / "reference"))
    import reference_runs as rr
    from mitjax.io.dump import DumpSet
    if kind == "jdall":
        return DumpSet(_jdall_runs()[inp] / "dumps")
    return DumpSet(rr.run_top(rr.find_run("vermix", inp, "jdon")) / "dumps")


def test_jdall_runs_match_registered():
    """The all-steps runs are the registered runs' model: same binary (sha256), %MON lines identical to lane A's jdon
    run of the same variant (dumping more steps changes no value), and the dumps of steps 0..2 bit-identical."""
    import json
    sys.path.insert(0, str(REPO / "reference"))
    import reference_runs as rr
    for inp, top in _jdall_runs().items():
        reg = rr.run_top(rr.find_run("vermix", inp, "jdon"))
        ma, mb = (json.loads((t / "MANIFEST.json").read_text())["binary"]["sha256"] for t in (top, reg))
        assert ma == mb, inp
        mon = lambda t: [ln for ln in (t / "rundir" / "output.txt").read_text().splitlines() if "%MON" in ln]  # noqa
        assert mon(top) == mon(reg) and len(mon(top)) > 0, f"{inp}: %MON differs"
        for it in (0, 1, 2):
            for f in sorted((reg / "dumps").glob(f"jd_{it:010d}_t*.bin")):
                assert (top / "dumps" / f.name).read_bytes() == f.read_bytes(), f"{inp} {f.name}"
        print(f"{inp}: binary, {len(mon(top))} %MON lines and the steps 0-2 dumps identical; "
              f"{len(dumpset(inp, 'jdall').iterations())} dumped steps")


def dump_commons(inp, params_src):
    """(cfg, grid, params, eos) of the vermix model run (one tile)."""
    from mitjax.model.src import ini_parms as ip
    R, d = replay(inp), dumpset(inp)
    cfg = R.cfg(1)
    s = cfg.size
    Nr = s.Nr
    ij = dict(i=(1-s.OLx, s.sNx+s.OLx), j=(1-s.OLy, s.sNy+s.OLy))
    G = lambda n: d.field(0, "G00_geometry", n)                             # noqa: E731
    v1 = lambda n, m: FArray(jnp.asarray(G(n)[0, :, 0, 0]), n, k=(1, m), tiled=False)   # noqa: E731
    grid = Common(maskC=FArray(jnp.asarray(G("maskC")), "maskC", k=(1, Nr), **ij),
                  maskW=FArray(jnp.asarray(G("maskW")), "maskW", k=(1, Nr), **ij),
                  maskS=FArray(jnp.asarray(G("maskS")), "maskS", k=(1, Nr), **ij),
                  kLowC=FArray(jnp.asarray(G("kLowC")[:, 0].astype(np.int32)), "kLowC", **ij),
                  kSurfC=FArray(jnp.asarray(G("kSurfC")[:, 0].astype(np.int32)), "kSurfC", **ij),
                  rC=v1("rC", Nr), rF=v1("rF", Nr+1), drF=v1("drF", Nr), recip_drF=v1("recip_drF", Nr),
                  recip_drC=v1("recip_drC", Nr+1))
    exp = experiment(inp)
    mp = ip.ini_parms(exp)
    P = ip.ini_parms_dyn(exp, mp.grid, mp.time, mp.init)
    if params_src == "oracle":
        diffKrNrS = R.orig["diffKrNrS"]
    else:
        diffKrNrS = vermix_params(inp)["diffKrNrS"]
    g = R.g
    vec = lambda name, a, n_: FArray(jnp.asarray(a), name, k=(1, n_), tiled=False)    # noqa: E731
    par = dict(viscArNr=P.viscArNr, diffKrNrS=vec("diffKrNrS", diffKrNrS, Nr), tRef=P.tRef, sRef=P.sRef,
               dTtracerLev=vec("dTtracerLev", g["dTtracerLev"], Nr), pRef4EOS=vec("pRef4EOS", g["pRef4EOS"], Nr),
               phiRef=vec("phiRef", g["phiRef"], 2*Nr+1), surf_pRef=jnp.float64(g["surf_pRef"]),
               rhoConst=P.rhoConst, gravity=P.gravity, mass2rUnit=P.mass2rUnit, rhoNil=P.rhoNil)
    for k in ("rhoConst", "gravity", "mass2rUnit"):                          # ours == the harness's dump
        assert np.float64(par[k]).tobytes() == np.float64(g[k]).tobytes(), k
    static = {"selectP_inEOS_Zc": int(g["selectP_inEOS_Zc"]), "usingPCoords": bool(g["usingPCoords"]),
              "usingZCoords": not bool(g["usingPCoords"])}
    return cfg, grid, Common(static, {}, **par), our_eos(P.eosType)


def _f3(a, name, s):
    return FArray(jnp.asarray(a), name, k=(1, s.Nr), i=(1-s.OLx, s.sNx+s.OLx), j=(1-s.OLy, s.sNy+s.OLy))


PARAMS_SRC = [pytest.param("oracle", id="params-oracle"),
              pytest.param("ini_parms", id="params-ini_parms")]   # diffKzS alias merged (GGL90 s2)


@pytest.mark.parametrize("kind", ["jdon", "jdall"])
@pytest.mark.parametrize("params_src", PARAMS_SRC)
def test_pp81_vs_dumps(params_src, kind, calc=None):
    """PP81_CALC on the model's own state == P08_pp81 (PPviscAr, PPdiffKr), every dumped iteration, all points;
    the prior (points PP81_CALC does not write) is PP81_INIT_VARIA's at the first and the previous dump after."""
    from mitjax.pkg.pp81 import pp81_calc
    from mitjax.pkg.pp81.pp81_h import declare
    from mitjax.pkg.pp81.pp81_init_varia import pp81_init_varia
    from mitjax.pkg.pp81.pp81_readparms import pp81_readparms
    inp = "input.pp81"
    d = dumpset(inp, kind)
    cfg, grid, params, eos = dump_commons(inp, params_src)
    s = cfg.size
    pp = pp81_init_varia(cfg=cfg, params=params, pp=pp81_readparms(experiment(inp), params=params))
    calc = calc or pp81_calc.pp81_calc
    f = jax.jit(lambda state, pp, grid, params, eos: calc(
        None, 0.0, 0, cfg=cfg, grid=grid, params=params, eos=eos, state=state, pp=pp))
    for it in d.iterations():
        S04 = lambda n: d.field(it, "S04_oceanic_phys", n)                   # noqa: E731
        S00 = lambda n: d.field(it, "S00_begin", n)                          # noqa: E731
        state = Common(theta=_f3(S04("theta"), "theta", s), salt=_f3(S04("salt"), "salt", s),
                       uVel=_f3(S00("uVel"), "uVel", s), vVel=_f3(S00("vVel"), "vVel", s),
                       totPhiHyd=_f3(d.field(it, "P02_rho_sigma_ivdc", "totPhiHyd"), "totPhiHyd", s))
        out = f(state, pp, grid, params, eos)
        for n in ("PPviscAr", "PPdiffKr"):
            check(f"P08 {n} it {it}", getattr(out, n).data, d.field(it, "P08_pp81", n))
        pp = pp.replace(PPviscAr=declare("PPviscAr", s, jnp.asarray(d.field(it, "P08_pp81", "PPviscAr"))),
                        PPdiffKr=declare("PPdiffKr", s, jnp.asarray(d.field(it, "P08_pp81", "PPdiffKr"))))


@pytest.mark.parametrize("kind", ["jdon", "jdall"])
@pytest.mark.parametrize("params_src", PARAMS_SRC)
def test_my82_vs_dumps(params_src, kind, calc=None):
    """MY82_CALC on the model's own state == P09_my82 (MYviscAr, MYdiffKr, MYhbl), every dumped iteration."""
    from mitjax.pkg.my82 import my82_calc
    from mitjax.pkg.my82.my82_h import declare
    from mitjax.pkg.my82.my82_init_varia import my82_init_varia
    from mitjax.pkg.my82.my82_readparms import my82_readparms
    inp = "input.my82"
    d = dumpset(inp, kind)
    cfg, grid, params, eos = dump_commons(inp, params_src)
    s = cfg.size
    my = my82_init_varia(cfg=cfg, params=params, my=my82_readparms(experiment(inp)))
    calc = calc or my82_calc.my82_calc
    f = jax.jit(lambda state, my, grid, params, eos: calc(
        None, 0.0, 0, cfg=cfg, grid=grid, params=params, eos=eos, state=state, my=my))
    for it in d.iterations():
        S04 = lambda n: d.field(it, "S04_oceanic_phys", n)                   # noqa: E731
        S00 = lambda n: d.field(it, "S00_begin", n)                          # noqa: E731
        state = Common(theta=_f3(S04("theta"), "theta", s), salt=_f3(S04("salt"), "salt", s),
                       uVel=_f3(S00("uVel"), "uVel", s), vVel=_f3(S00("vVel"), "vVel", s),
                       totPhiHyd=_f3(d.field(it, "P02_rho_sigma_ivdc", "totPhiHyd"), "totPhiHyd", s))
        out = f(state, my, grid, params, eos)
        P09 = lambda n: d.field(it, "P09_my82", n)                           # noqa: E731
        check(f"P09 MYviscAr it {it}", out.MYviscAr.data, P09("MYviscAr"))
        check(f"P09 MYdiffKr it {it}", out.MYdiffKr.data, P09("MYdiffKr"))
        check(f"P09 MYhbl it {it}", out.MYhbl.data, P09("MYhbl")[:, 0])
        my = my.replace(MYviscAr=declare("MYviscAr", s, jnp.asarray(P09("MYviscAr"))),
                        MYdiffKr=declare("MYdiffKr", s, jnp.asarray(P09("MYdiffKr"))),
                        MYhbl=declare("MYhbl", s, jnp.asarray(P09("MYhbl")[:, 0])))


@pytest.mark.parametrize("kind", ["jdon", "jdall"])
@pytest.mark.parametrize("params_src", PARAMS_SRC)
def test_opps_vs_dumps(params_src, kind, interface=None):
    """OPPS_INTERFACE (iMin:iMax = 1:sNx, jMin:jMax = 1:sNy, as TRACERS_CORRECTION_STEP calls it) on the state after
    THERMODYNAMICS (S05; DYNAMICS' totPhiHyd from S06) == T05_opps (theta, salt), every dumped iteration."""
    from mitjax.pkg.opps import opps_calc, opps_h, opps_interface
    from mitjax.pkg.opps.opps_readparms import opps_readparms
    inp = "input.opps"
    d = dumpset(inp, kind)
    cfg, grid, params, eos = dump_commons(inp, params_src)
    s = cfg.size
    op = opps_readparms(experiment(inp))
    interface = interface or opps_interface.opps_interface
    f = jax.jit(lambda state, grid, params, eos, op: interface(
        1, s.sNx, 1, s.sNy, 0.0, 0, cfg=cfg, grid=grid, params=params, eos=eos, state=state, op=op))
    changed = 0
    for it in d.iterations():
        S05 = lambda n: d.field(it, "S05_thermodynamics_sync", n)            # noqa: E731
        state = Common(theta=_f3(S05("theta"), "theta", s), salt=_f3(S05("salt"), "salt", s),
                       wVel=_f3(d.field(it, "S06_dynamics", "wVel"), "wVel", s),
                       totPhiHyd=_f3(d.field(it, "S06_dynamics", "totPhiHyd"), "totPhiHyd", s))
        th, sa, cnt, ctr = f(state, grid, params, eos, op)
        opps_calc.opps_calc_host(ctr, it, opps_h.NTIME_MAX)
        check(f"T05 theta it {it}", th.data, d.field(it, "T05_opps", "theta"))
        check(f"T05 salt it {it}", sa.data, d.field(it, "T05_opps", "salt"))
        changed += int(np.sum(d.field(it, "T05_opps", "theta") != S05("theta")))
        print(f"it {it}: plumes {int(np.asarray(cnt).sum())}")
    print(f"{kind}: {changed} theta points changed by OPPS over {len(d.iterations())} dumped steps")
    if kind == "jdall":
        assert changed > 0, "OPPS changed nothing in the dumped steps: the gate would not see an error"


def _module_with(modname, **attrs):
    """A copy of mitjax.pkg.<modname> with module attributes replaced (e.g. a planted callee)."""
    path = REPO / "mitjax" / "pkg" / (modname.replace(".", "/") + ".py")
    name = f"_mjx_copy_{modname.replace('.', '_')}_{abs(hash(tuple(sorted(attrs))))}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    mod.__dict__.update(attrs)
    return mod


def test_dump_gates_negative_controls():
    """The in-model gates bite on vermix's own columns: the pressure level of the first FIND_RHO_2D of PP81 / MY82
    RI_NUMBER (kRef = K -> Km1) and NINT -> INT in OPPS's ntime (:334; vermix's plumes of steps 18-19 are one level
    deep, where STATE1's kRef plant of the replay does not change the outcome) must each fail their dump gate."""
    ri = _planted("pp81.pp81_ri_number", "rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, K,",
                  "rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, Km1,")
    bad = _module_with("pp81.pp81_calc", pp81_ri_number=ri.pp81_ri_number)
    with pytest.raises(AssertionError, match="points differ"):
        test_pp81_vs_dumps("oracle", "jdall", calc=bad.pp81_calc)
    ri = _planted("my82.my82_ri_number", "rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, K,",
                  "rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, Km1,")
    bad = _module_with("my82.my82_calc", my82_ri_number=ri.my82_ri_number)
    with pytest.raises(AssertionError, match="points differ"):
        test_my82_vs_dumps("oracle", "jdall", calc=bad.my82_calc)
    oc = _planted("opps.opps_calc", "r = jnp.where(x >= 0.0, jnp.floor(x + 0.5), -jnp.floor(-x + 0.5))",
                  "r = jnp.where(x >= 0.0, jnp.floor(x), -jnp.floor(-x))")
    bad = _module_with("opps.opps_interface", opps_calc=oc.opps_calc)
    with pytest.raises(AssertionError, match="points differ"):
        test_opps_vs_dumps("oracle", "jdall", interface=bad.opps_interface)


# ---------------------------------------------------------------------------------------------------------------
# gradients: finite on every lane (land, halos, inactive levels), tangent vs adjoint dot test, central FD

GRAD_SEL = slice(0, 4)


def _grad_checks(label, fwd, x0, seed=7, fd_points=4, fd_rel=1e-5):
    """fwd: x [theta, salt, (u, v)] stacked -> scalar-valued pytree output (flattened). Finite grad, dot test, FD."""
    rng = np.random.default_rng(seed)
    y0 = fwd(x0)
    w = jnp.asarray(rng.standard_normal(y0.shape))
    J = lambda x: jnp.sum(w*fwd(x))                                         # noqa: E731
    g = jax.jit(jax.grad(J))(x0)
    g = np.asarray(g)
    assert np.all(np.isfinite(g)), f"{label}: {np.sum(~np.isfinite(g))} non-finite gradient lanes"
    v = jnp.asarray(rng.standard_normal(x0.shape))
    _, tl = jax.jit(lambda x, v: jax.jvp(fwd, (x,), (v,)))(x0, v)
    _, vjp = jax.vjp(fwd, x0)
    (ad,) = jax.jit(vjp)(w)
    lhs, rhs = float(jnp.sum(w*tl)), float(jnp.sum(ad*v))
    rel = abs(lhs - rhs)/max(abs(lhs), 1e-300)
    print(f"{label}: grad finite on {g.size} lanes, |g|max {np.abs(g).max():.3e}; dot test {lhs:.15e} vs "
          f"{rhs:.15e} (rel {rel:.1e})")
    assert rel < 1e-12
    # central FD along the largest-gradient lanes (smooth points: the switches are far, checked by an h-sweep)
    Jj = jax.jit(J)
    idx = np.argsort(-np.abs(g).ravel())[:fd_points]
    for c in idx:
        e = np.zeros(g.size)
        e[c] = 1.0
        e = jnp.asarray(e.reshape(g.shape))
        fds = []
        for h in (1e-4, 1e-5, 1e-6):
            fds.append((float(Jj(x0 + h*e)) - float(Jj(x0 - h*e)))/(2*h))
        best = min(abs(f - g.ravel()[c])/abs(g.ravel()[c]) for f in fds)
        print(f"  lane {c}: AD {g.ravel()[c]:.10e}, FD {[f'{f:.10e}' for f in fds]} (best rel {best:.1e})")
        assert best < fd_rel, f"{label}: FD mismatch at lane {c}"   # measured (job 27840898): <= 9.9e-7


def test_pp81_gradients():
    R = replay("input.pp81")
    from mitjax.pkg.pp81 import pp81_calc
    cfg, grid, params, eos, state = R.commons(GRAD_SEL)
    pp = pp81_common(R, GRAD_SEL)
    x0 = jnp.stack([state.theta.data, state.salt.data, state.uVel.data, state.vVel.data])

    def fwd(x):
        st = state.replace(theta=FArray(x[0], "theta", tiled=True, _dims=state.theta.dims),
                           salt=FArray(x[1], "salt", tiled=True, _dims=state.salt.dims),
                           uVel=FArray(x[2], "uVel", tiled=True, _dims=state.uVel.dims),
                           vVel=FArray(x[3], "vVel", tiled=True, _dims=state.vVel.dims))
        p = pp81_calc.pp81_calc(None, 0.0, 0, cfg=cfg, grid=grid, params=params, eos=eos, state=st, pp=pp)
        return jnp.stack([p.PPviscAr.data, p.PPdiffKr.data])

    _grad_checks("pp81", fwd, x0)


def test_my82_gradients():
    R = replay("input.my82")
    from mitjax.pkg.my82 import my82_calc
    cfg, grid, params, eos, state = R.commons(GRAD_SEL)
    my = my82_common(R, GRAD_SEL)
    x0 = jnp.stack([state.theta.data, state.salt.data, state.uVel.data, state.vVel.data])

    def fwd(x):
        st = state.replace(theta=FArray(x[0], "theta", tiled=True, _dims=state.theta.dims),
                           salt=FArray(x[1], "salt", tiled=True, _dims=state.salt.dims),
                           uVel=FArray(x[2], "uVel", tiled=True, _dims=state.uVel.dims),
                           vVel=FArray(x[3], "vVel", tiled=True, _dims=state.vVel.dims))
        m = my82_calc.my82_calc(None, 0.0, 0, cfg=cfg, grid=grid, params=params, eos=eos, state=st, my=my)
        return jnp.concatenate([m.MYviscAr.data.ravel(), m.MYdiffKr.data.ravel(), m.MYhbl.data.ravel()])

    _grad_checks("my82", fwd, x0)


def test_opps_gradients():
    R = replay("input.opps")
    from mitjax.pkg.opps import opps_interface
    cfg, grid, params, eos, state = R.commons(GRAD_SEL)
    op = opps_common(R, False)
    s = cfg.size
    x0 = jnp.stack([state.theta.data, state.salt.data])

    def fwd(x):
        st = state.replace(theta=FArray(x[0], "theta", tiled=True, _dims=state.theta.dims),
                           salt=FArray(x[1], "salt", tiled=True, _dims=state.salt.dims))
        th, sa, _, _ = opps_interface.opps_interface(1-s.OLx, s.sNx+s.OLx, 1-s.OLy, s.sNy+s.OLy, 0.0, 0,
                                                     cfg=cfg, grid=grid, params=params, eos=eos, state=st, op=op)
        return jnp.stack([th.data, sa.data])

    _grad_checks("opps", fwd, x0)
