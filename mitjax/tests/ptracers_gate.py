"""Helpers of the PTRACERS lane gates (mitjax/tests/test_ptracers.py, plan Task 26): the replay harness runs named by
reference/replay_ptracers/CURRENT (relative to $MJX_REFERENCE), the common blocks the replayed routines read built
from the harness's own dump (pt_grid.bin), one jitted driver per replayed routine (every float a traced argument,
called for every tile as the harness calls it), planted-error copies of a module, and the substep-dump helpers.
JAX transforms are used here, never in mitjax/model or mitjax/pkg.
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.farray import FArray
from mitjax.model.src.ini_eos import EOS, _VECTORS
from mitjax.pkg.ptracers.ptracers_fields_h import PtracersFields
from mitjax.pkg.ptracers.ptracers_params_h import PtracersParams
from mitjax.tests.col_replay import Cfg, Common, bit_equal, experiment

REPO = Path(__file__).resolve().parents[2]


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


replay_io = _load_by_path("_mjx_replay_ptracers_io", REPO / "reference" / "replay_ptracers" / "replay_io.py")


def current_runs():
    """[(experiment, input_dir, rundir)] from reference/replay_ptracers/CURRENT."""
    out = []
    for ln in (REPO / "reference" / "replay_ptracers" / "CURRENT").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            exp, inp, rel = ln.split()
            out.append((exp, inp, Path(paths.REFERENCE) / rel))
    return out


def planted(relpath, old, new):
    """A copy of the module at `relpath` (repository-relative) with `old` replaced by `new` (exactly once)."""
    path = REPO / relpath
    src = path.read_text()
    if src.count(old) != 1:
        raise ValueError(f"planted: {old!r} occurs {src.count(old)} times in {path.name}")
    name = f"_mjx_planted_{path.stem}_{abs(hash((old, new)))}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src.replace(old, new), str(path), "exec"), mod.__dict__)
    return mod


def _ij(size):
    return {"i": (1 - size.OLx, size.sNx + size.OLx), "j": (1 - size.OLy, size.sNy + size.OLy)}


class _Trunc(dict):
    """Record names are CHARACTER*16 (the_main_loop.F COL_WR): a longer Fortran name is read by its first 16
    characters (PTRACERS_numInUse -> PTRACERS_numInUs)."""

    def __getitem__(self, k):
        return dict.__getitem__(self, k[:16])


class Replay:
    """One PTRACERS harness run: inputs, Fortran outputs, the dumped grid/parameters, the experiment config."""

    def __init__(self, exp, inp, rundir):
        self.exp, self.inp, self.rundir = exp, inp, Path(rundir)
        self.experiment = experiment(exp, inp)
        self.cfg = Cfg(self.experiment.cfg)
        self.cfg.experiment, self.cfg.code_dir = self.experiment.cfg.experiment, self.experiment.cfg.code_dir
        self.cfg.exp_dir = self.experiment.cfg.exp_dir                  # config.params.code_path (docs plan Task 2)
        self.size = self.experiment.cfg.size
        sz = {k: getattr(self.size, k) for k in replay_io.SIZE_KEYS}
        self.inputs = replay_io.read_inputs(self.rundir, sz)
        self.out = replay_io.read_outputs(self.rundir, sz)
        self.g = replay_io.read_grid(self.rundir, sz)

    def f3(self, a, name):
        return FArray(jnp.asarray(a), name, k=(1, self.size.Nr), **_ij(self.size))

    def f2(self, a, name):
        return FArray(jnp.asarray(a), name, **_ij(self.size))

    def commons(self):
        """(grid, params, eos, ptr, cg2dh, cg2d_params) of the dumped values (floats traced when passed to jit)."""
        sz = self.size
        g = {k[:16]: v for k, v in self.g.items()}
        g = _Trunc(g)
        Nr = sz.Nr
        r = lambda n, a: FArray(jnp.asarray(np.atleast_1d(a)), n, k=(1, Nr), tiled=False)
        grid = Common(hFacC=self.f3(g["hFacC"], "hFacC"), recip_hFacC=self.f3(g["recip_hFacC"], "recip_hFacC"),
                      maskC=self.f3(g["maskC"], "maskC"), rA=self.f2(g["rA"], "rA"),
                      drF=r("drF", g["drF"]), recip_drF=r("recip_drF", g["recip_drF"]),
                      rF=FArray(jnp.asarray(g["rF"]), "rF", k=(1, Nr + 1), tiled=False),
                      **({"recip_drC": FArray(jnp.asarray(g["recip_drC"]), "recip_drC", k=(1, Nr + 1), tiled=False)}
                         if "recip_drC" in g else {}),
                      rkSign=jnp.float64(g["rkSign"]), gravitySign=jnp.float64(g["gravitySign"]))
        static = {"usingZCoords": True, "usingPCoords": False, "fluidIsAir": False, "useShelfIce": False,
                  "useDiagnostics": False, "usePTRACERS": bool(g["usePTRACERS"]), "nonlinFreeSurf":
                  int(g["nonlinFreeSurf"]), "interDiffKr_pCell": False, "pCellMix_select": 0,
                  "cg2dMinItersNSA": int(g["cg2dMinItersNSA"]),
                  "numItersMax": 200,                    # tutorial_tracer_adjsens/code_ad/tamc.h:94
                  "printResidualFreq": 0, "debugLevel": 1}
        par = {n: r(n, g[n]) for n in ("tRef", "sRef", "viscArNr", "dTtracerLev")}
        par.update({n: jnp.float64(g[n]) for n in ("rhoNil", "rhoConst", "cAdjFreq", "deltaTClock", "ivdc_kappa",
                                                   "diffKrBL79surf", "diffKrBL79deep", "diffKrBL79Ho",
                                                   "diffKrBL79scl")})
        params = Common(static, {}, **par)
        vec = {n: FArray(jnp.zeros(hi - lo + 1), n, k=(lo, hi), tiled=False) for n, (lo, hi) in _VECTORS.items()}
        eos = EOS(equationOfState="LINEAR", eosRefP0=jnp.float64(0.), tAlpha=jnp.float64(g["tAlpha"]),
                  sBeta=jnp.float64(g["sBeta"]), **vec)
        ptr = PtracersParams(
            dict(PTRACERS_numInUse=int(g["PTRACERS_numInUse"]), PTRACERS_StepFwd=(bool(g["PTRACERS_StepFwd"]),),
                 PTRACERS_linFSConserve=(bool(g["PTRACERS_linFSConserve"]),),
                 PTRACERS_addSrelax2EmP=bool(g["PTRACERS_addSrelax2EmP"]),
                 PTRACERS_EvPrRn_ne_UNSET=(bool(g["PTRACERS_EvPrRn"] != 1.234567e5),)),
            dict(PTRACERS_dTLev=r("PTRACERS_dTLev", g["PTRACERS_dTLev"]),
                 PTRACERS_diffKrNr=(r("PTRACERS_diffKrNr", g["PTRACERS_diffKrNr"]),),
                 lambdaTr1ClimRelax=jnp.float64(g["lambdaTr1ClimRelax"])))
        from mitjax.model.src.cg2d_h import CG2DH
        cg2dh = CG2DH(aW2d=self.f2(g["aW2d"], "aW2d"), aS2d=self.f2(g["aS2d"], "aS2d"),
                      aC2d=self.f2(g["aC2d"], "aC2d"), pW=self.f2(g["pW"], "pW"), pS=self.f2(g["pS"], "pS"),
                      pC=self.f2(g["pC"], "pC"), cg2dNorm=jnp.float64(g["cg2dNorm"]),
                      cg2dTolerance_sq=jnp.float64(g["cg2dTolerance_sq"]), cg2dNormaliseRHS=False)
        return grid, params, eos, ptr, cg2dh

    def ptf(self, p3=None, sf=None):
        z3 = jnp.zeros((self.size.nSx*self.size.nSy, self.size.Nr) + self.inputs["cgB"].shape[1:])
        p = self.f3(p3 if p3 is not None else z3, "pTracer")
        s = self.f2(sf if sf is not None else jnp.zeros(self.inputs["cgB"].shape), "surfaceForcingPTr")
        return PtracersFields(pTracer=(p,), gpTrNm1=(self.f3(z3, "gpTrNm1"),), surfaceForcingPTr=(s,))


def state_of(cfg, **fields):
    """A State of the build, every field NaN, with `fields` (FArrays) set."""
    from mitjax.model.state import empty_state
    st = empty_state(cfg)
    return st.replace(**fields)


def compare(got, want):
    """bit_equal of an FArray / array against a Fortran record."""
    return bit_equal(getattr(got, "data", got), want)
