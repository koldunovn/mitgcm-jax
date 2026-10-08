"""CG2D.h: the two-dimensional solver's operator, preconditioner and constants (model/inc/CG2D.h @63cdc0b), and the
PARAMS.h / AUTODIFF_PARAMS.h values that INI_CG2D, UPDATE_CG2D and CG2D read (`Cg2dParams`, set by `ini_parms_cg2d`).

`CG2DH` holds the common blocks of CG2D.h:
    COMMON /CG2D_I_L/  cg2dNormaliseRHS                                  (CG2D.h:19-20)
    COMMON /CG2D_I_RS/ aW2d, aS2d, aC2d, pW, pS, pC                      (CG2D.h:32-42), _RS = Real*8 in every M1 build
    COMMON /CG2D_I_RL/ cg2dNorm, cg2dTolerance_sq                         (CG2D.h:35-36, 43)
The six arrays are FArrays declared (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy); cg2dNorm and cg2dTolerance_sq are float64
scalars and pytree leaves (traced when passed to jit: setup constants are never closed over, [E§11], [L-XLA-4]);
cg2dNormaliseRHS is static (it selects code: CG2D's normalisation; INI_CG2D sets it from the namelist value
cg2dTargetResWunit, decided on the host by `ini_parms_cg2d`).

`ini_parms_cg2d(exp)` sets the run-time parameters these routines read as SET_DEFAULTS / INI_PARMS @63cdc0b set them
(cited per value, defaults through params_io.fortran_default). They belong to `ini_parms.py` once INI_PARMS is ported
as a whole (M1 core lane, Task 11); they live here until then, as GridParams lives in mitjax/model/grid.py.
"""

from dataclasses import dataclass

import jax
import numpy as np

from mitjax.farray import FArray
from mitjax.params_io import RunParams, fortran_default, params_pytree

# eesupp/inc/EEPARAMS.h:88-95 @63cdc0b
debLevZero = 0      # EEPARAMS.h:90  PARAMETER ( debLevZero=0 )
debLevA = 1         # EEPARAMS.h:91  PARAMETER ( debLevA=1 )
debLevB = 2         # EEPARAMS.h:92  PARAMETER ( debLevB=2 )
debLevD = 4         # EEPARAMS.h:94  PARAMETER ( debLevD=4 )
debLevE = 5         # EEPARAMS.h:95  PARAMETER ( debLevE=5 )
UNSET_I = 123456789  # EEPARAMS.h:85  PARAMETER ( UNSET_I = 123456789 )

ARRAYS = ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC")   # CG2D.h:37-42


def declare_xy(name, size, fill=np.nan, ntiles=None):
    """A CG2D.h array (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy) (CG2D.h:37-42), every point `fill`."""
    sNx, sNy, OLx, OLy = size.sNx, size.sNy, size.OLx, size.OLy
    nt = size.nSx * size.nSy if ntiles is None else ntiles
    data = jax.numpy.full((nt, sNy + 2 * OLy, sNx + 2 * OLx), fill, dtype=jax.numpy.float64)
    return FArray(data, name, i=(1 - OLx, sNx + OLx), j=(1 - OLy, sNy + OLy))


@jax.tree_util.register_pytree_node_class
class CG2DH:
    """The CG2D.h common blocks (see the module docstring). Immutable: `replace(**fields)` returns a new one, as a
    routine writes the common block."""

    __slots__ = ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC", "cg2dNorm", "cg2dTolerance_sq", "cg2dNormaliseRHS")

    def __init__(self, aW2d, aS2d, aC2d, pW, pS, pC, cg2dNorm, cg2dTolerance_sq, cg2dNormaliseRHS):
        for k, v in zip(self.__slots__, (aW2d, aS2d, aC2d, pW, pS, pC, cg2dNorm, cg2dTolerance_sq,
                                         cg2dNormaliseRHS)):
            object.__setattr__(self, k, v)

    def __setattr__(self, k, v):
        raise AttributeError("CG2DH is immutable: use replace()")

    def replace(self, **kw):
        vals = {k: getattr(self, k) for k in self.__slots__}
        for k in kw:
            if k not in vals:
                raise AttributeError(f"CG2D.h has no variable {k}")
        vals.update(kw)
        return CG2DH(**vals)

    def tree_flatten(self):
        leaves = tuple(getattr(self, k) for k in self.__slots__[:-1])
        return leaves, (self.cg2dNormaliseRHS,)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(*leaves, *aux)


@params_pytree
@dataclass(frozen=True)
class Cg2dParams:
    """PARAMS.h (and AUTODIFF_PARAMS.h) values read by INI_CG2D, UPDATE_CG2D and CG2D, as INI_PARMS leaves them.
    `float` fields are pytree leaves (traced jit arguments, never decide a branch); the others are static."""
    implicSurfPress: float
    implicDiv2DFlow: float
    freeSurfFac: float
    deltaTMom: float
    deltaTFreeSurf: float
    cg2dTargetResidual: float
    cg2dpcOffDFac: float
    cg2dTargetResWunit_le_0: bool       # cg2dTargetResWunit .LE. zeroRL (ini_cg2d.F:148), decided on the host
    cg2dMaxIters: int
    cg2dUseMinResSol: int
    cg2dPreCondFreq: int
    printResidualFreq: int
    debugLevel: int
    nIter0: int
    deepAtmosphere: bool
    nonlinFreeSurf: int
    selectImplicitDrag: int
    momStepping: bool
    useSRCGSolver: bool
    useNSACGSolver: bool
    cg2dFullAdjoint: bool
    # PTRACERS lane (CG2D_NSA, tutorial_tracer_adjsens): PARAMS.h cg2dMinItersNSA and the build's tamc.h numItersMax
    # (None without ALLOW_AUTODIFF_TAMC, where tamc.h is not included)
    cg2dMinItersNSA: int = 0
    numItersMax: object = None
    # PTRACERS lane (INI_CG2D with cg2dTargetResWunit > 0, ini_cg2d.F:152-162): PARAMS.h cg2dTargetResWunit
    cg2dTargetResWunit: float = np.float64(-1.0)     # set_defaults.F:289  cg2dTargetResWunit = -1.
    # not a Fortran parameter: the CG2D derivative option of mitjax/ad/modes.py ("run": as the run's cg2dFullAdjoint
    # in TAF's CG2D_MAD; "exact": the full implicit derivative); backward-only, read by cg2d.cg2d_solve
    mjx_cg2d_derivative: str = "run"


def _tamc_numItersMax(exp):
    """PTRACERS lane: `PARAMETER ( numItersMax = N )` of the tamc.h the build compiles (genmake2 takes the experiment's
    code directory's copy first, else pkg/autodiff/tamc.h:99)."""
    import re
    from mitjax import paths
    from mitjax.config.params import code_path
    cfg = exp.cfg
    for d in (code_path(cfg), paths.UPSTREAM / "pkg" / "autodiff"):
        f = d / "tamc.h"
        if f.exists():
            m = re.search(r"^\s+PARAMETER\s*\(\s*numItersMax\s*=\s*(\d+)\s*\)", f.read_text(), re.M | re.I)
            if m is None:
                raise ValueError(f"{f}: no PARAMETER numItersMax")
            return int(m.group(1))
    raise FileNotFoundError("tamc.h")


def _get(rp, exp, group, key, citation, fname="data", condition=None):
    """A scalar from the run's parameter file, else the cited Fortran default (params_io.fortran_default)."""
    return rp.get(fname, group, key, default=fortran_default(citation, key, exp, condition=condition))


def ini_parms_cg2d(exp):
    """The values of Cg2dParams for the experiment `exp` (mitjax.config.params.Experiment), set as SET_DEFAULTS and
    INI_PARMS @63cdc0b set them (model/src/set_defaults.F, model/src/ini_parms.F; AUTODIFF_READPARMS for
    cg2dFullAdjoint), each statement cited. Raises for parameter values the ported code does not cover."""
    cfg = exp.cfg
    rp = RunParams(exp.run)
    SD = "model/src/set_defaults.F"
    IP = "model/src/ini_parms.F"
    # PARM01
    implicSurfPress = _get(rp, exp, "PARM01", "implicSurfPress", f"{SD}:252")
    implicDiv2DFlow = _get(rp, exp, "PARM01", "implicDiv2DFlow", f"{SD}:253")
    implicitFreeSurface = _get(rp, exp, "PARM01", "implicitFreeSurface", f"{SD}:250")
    rigidLid = _get(rp, exp, "PARM01", "rigidLid", f"{SD}:251")
    nonlinFreeSurf = _get(rp, exp, "PARM01", "nonlinFreeSurf", f"{SD}:257")
    selectImplicitDrag = _get(rp, exp, "PARM01", "selectImplicitDrag", f"{SD}:211")
    momStepping = _get(rp, exp, "PARM01", "momStepping", f"{SD}:192")
    deepAtmosphere = _get(rp, exp, "PARM04", "deepAtmosphere", f"{SD}:71")
    # debugLevel: set_defaults.F:240-246 (IF ( debugMode ) debLevD ELSE debLevB, debLevA #ifdef ALLOW_AUTODIFF), then
    # the PARM01 READ. debugMode: eeset_parms.F:125 (eedata), then PARM01.
    if rp.has("data", "PARM01", "debugLevel"):
        debugLevel = rp.get("data", "PARM01", "debugLevel")
    else:
        debugMode = rp.get("data", "PARM01", "debugMode") if rp.has("data", "PARM01", "debugMode") else \
            rp.get("eedata", "EEPARMS", "debugMode", default=fortran_default("eesupp/src/eeset_parms.F:125",
                                                                           "debugMode", exp))
        if debugMode:
            debugLevel = debLevD                                                # set_defaults.F:241
        elif cfg.cpp.ALLOW_AUTODIFF:
            debugLevel = debLevA                                                # set_defaults.F:243-245
        else:
            debugLevel = debLevB                                                # set_defaults.F:243
    # ini_parms.F:467-481: no barotropic solver selected => implicitFreeSurface; freeSurfFac (no SET_DEFAULTS line)
    if not rigidLid and not implicitFreeSurface:
        implicitFreeSurface = True                                              # ini_parms.F:478
    if rp.has("data", "PARM01", "freeSurfFac") and not (implicitFreeSurface or rigidLid):
        raise NotImplementedError("INI_PARMS: freeSurfFac from data without implicitFreeSurface/rigidLid")
    freeSurfFac = None
    if implicitFreeSurface:
        freeSurfFac = 1.0                                                       # ini_parms.F:480  1. _d 0
    if rigidLid:
        freeSurfFac = 0.0                                                       # ini_parms.F:481  0. _d 0
    # ini_parms.F:787-789: printResidualFreq = -1; IF ( debugLevel.GE.debLevE ) printResidualFreq = 1  (before the
    # PARM02 READ at :937, which may override it)
    printResidualFreq = -1
    if debugLevel >= debLevE:
        printResidualFreq = 1
    if rp.has("data", "PARM02", "printResidualFreq"):
        printResidualFreq = rp.get("data", "PARM02", "printResidualFreq")
    # PARM02
    if rp.has("data", "PARM02", "cg2dChkResFreq"):                              # ini_parms.F:951-956 (retired)
        raise ValueError('S/R INI_PARMS: unused "cg2dChkResFreq" is no longer allowed in file "data"')
    cg2dMaxIters = _get(rp, exp, "PARM02", "cg2dMaxIters", f"{SD}:286")
    cg2dTargetResidual = _get(rp, exp, "PARM02", "cg2dTargetResidual", f"{SD}:288")
    cg2dTargetResWunit = _get(rp, exp, "PARM02", "cg2dTargetResWunit", f"{SD}:289")
    cg2dpcOffDFac = _get(rp, exp, "PARM02", "cg2dpcOffDFac", f"{SD}:291")
    cg2dPreCondFreq = _get(rp, exp, "PARM02", "cg2dPreCondFreq", f"{SD}:292")
    useNSACGSolver = _get(rp, exp, "PARM02", "useNSACGSolver", f"{SD}:296")
    useSRCGSolver = _get(rp, exp, "PARM02", "useSRCGSolver", f"{SD}:297")
    # cg2dUseMinResSol: set_defaults.F:290 UNSET_I, then ini_parms.F:1585-1589
    if rp.has("data", "PARM02", "cg2dUseMinResSol"):
        cg2dUseMinResSol = rp.get("data", "PARM02", "cg2dUseMinResSol")
    else:
        cg2dUseMinResSol = UNSET_I                                              # set_defaults.F:290
    if cg2dUseMinResSol == UNSET_I:                                             # ini_parms.F:1585
        cg2dUseMinResSol = 0                                                    # ini_parms.F:1586
        topoFile = _get(rp, exp, "PARM05", "topoFile", f"{SD}:361")
        bathyFile = _get(rp, exp, "PARM05", "bathyFile", f"{SD}:360")
        # usingCartesianGrid as INI_PARMS leaves it: set_defaults.F:81-90, then ini_parms.F:1325-1348 (no grid
        # requested => Cartesian)
        grids = [_get(rp, exp, "PARM04", name, f"{SD}:{n}")
                 for name, n in (("usingCartesianGrid", 81), ("usingSphericalPolarGrid", 83),
                                 ("usingCurvilinearGrid", 86), ("usingCylindricalGrid", 90))]
        if sum(int(g) for g in grids) > 1:                                      # ini_parms.F:1325-1336
            raise ValueError("S/R INI_PARMS: More than one coordinate system requested")
        usingCartesianGrid = grids[0] or sum(int(g) for g in grids) < 1         # ini_parms.F:1337-1348
        if topoFile.strip() == "" and bathyFile.strip() == "" and usingCartesianGrid:   # ini_parms.F:1587-1588
            cg2dUseMinResSol = 1
    # PARM03: time steps (set_defaults.F:300-302, ini_parms.F:968, :1043-1068)
    deltaT = _get(rp, exp, "PARM03", "deltaT", f"{SD}:300")
    deltaTMom = _get(rp, exp, "PARM03", "deltaTMom", f"{SD}:301")
    deltaTFreeSurf = _get(rp, exp, "PARM03", "deltaTFreeSurf", f"{SD}:302")
    deltaTtracer = _get(rp, exp, "PARM03", "deltaTtracer", f"{IP}:968")
    if rp.nml.var("data", "PARM03", "dTtracerLev") is not None and rp.nml.var("data", "PARM03", "dTtracerLev").value:
        raise NotImplementedError("INI_PARMS: dTtracerLev in data (ini_parms.F:1034-1042) is not ported here")
    nIter0 = _get(rp, exp, "PARM03", "nIter0", f"{SD}:307")
    if deltaT == 0.:                                                            # ini_parms.F:1043
        # PTRACERS lane (tutorial_advection_in_gyre sets neither deltaT nor deltaTClock): deltaTClock has no
        # default statement; COMMON /PARM_R/ (PARAMS.h) is zero-initialised static storage of the gfortran oracle,
        # so a run that does not set it reads 0. here, as ini_parms.ini_parms_time takes it
        deltaT = rp.get("data", "PARM03", "deltaTClock") if rp.has("data", "PARM03", "deltaTClock") else 0.0
    if deltaT == 0.:
        deltaT = deltaTtracer                                                   # ini_parms.F:1044
    if deltaT == 0.:
        deltaT = deltaTMom                                                      # ini_parms.F:1045
    if deltaT == 0.:
        deltaT = deltaTFreeSurf                                                 # ini_parms.F:1046
    if deltaT == 0.:
        raise ValueError("S/R INI_PARMS: need to specify in file \"data\", namelist \"PARM03\" a model timestep")
    if deltaTMom == 0.:
        deltaTMom = deltaT                                                      # ini_parms.F:1057
    if deltaTFreeSurf == 0.:
        deltaTFreeSurf = deltaTMom                                              # ini_parms.F:1068
    # AUTODIFF_PARAMS.h: cg2dFullAdjoint (pkg/autodiff/autodiff_readparms.F:87, data.autodiff AUTODIFF_PARM01)
    cg2dFullAdjoint = False
    if cfg.cpp.ALLOW_AUTODIFF:
        cg2dFullAdjoint = rp.get("data.autodiff", "AUTODIFF_PARM01", "cg2dFullAdjoint",
                                 default=fortran_default("pkg/autodiff/autodiff_readparms.F:87", "cg2dFullAdjoint",
                                                         exp))
    # PTRACERS lane: CG2D_NSA's values (cg2d_nsa.F:118-123, :250): cg2dMinItersNSA (PARM02, set_defaults.F:287) and
    # numItersMax, the PARAMETER of the build's tamc.h (the experiment's code directory's, else pkg/autodiff/tamc.h;
    # read only under ALLOW_AUTODIFF_TAMC)
    cg2dMinItersNSA = _get(rp, exp, "PARM02", "cg2dMinItersNSA", f"{SD}:287")
    numItersMax = _tamc_numItersMax(exp) if cfg.cpp.ALLOW_AUTODIFF_TAMC else None
    return Cg2dParams(
        cg2dMinItersNSA=int(cg2dMinItersNSA), numItersMax=numItersMax,
        cg2dTargetResWunit=np.float64(cg2dTargetResWunit),
        implicSurfPress=np.float64(implicSurfPress), implicDiv2DFlow=np.float64(implicDiv2DFlow),
        freeSurfFac=np.float64(freeSurfFac), deltaTMom=np.float64(deltaTMom),
        deltaTFreeSurf=np.float64(deltaTFreeSurf), cg2dTargetResidual=np.float64(cg2dTargetResidual),
        cg2dpcOffDFac=np.float64(cg2dpcOffDFac), cg2dTargetResWunit_le_0=bool(cg2dTargetResWunit <= 0.0),
        cg2dMaxIters=int(cg2dMaxIters), cg2dUseMinResSol=int(cg2dUseMinResSol), cg2dPreCondFreq=int(cg2dPreCondFreq),
        printResidualFreq=int(printResidualFreq), debugLevel=int(debugLevel), nIter0=int(nIter0),
        deepAtmosphere=bool(deepAtmosphere), nonlinFreeSurf=int(nonlinFreeSurf),
        selectImplicitDrag=int(selectImplicitDrag), momStepping=bool(momStepping), useSRCGSolver=bool(useSRCGSolver),
        useNSACGSolver=bool(useNSACGSolver), cg2dFullAdjoint=bool(cg2dFullAdjoint))
