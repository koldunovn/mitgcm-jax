"""INI_PARMS (model/src/ini_parms.F @63cdc0b), with the SET_DEFAULTS defaults it starts from and the SET_PARMS / LOAD_REF_FILES
statements that complete the parameters: the PARAMS.h / SET_GRID.h values the ported routines read.

Not a port of INI_PARMS as a whole (a namelist reader of ~900 lines, most of it checks and printouts): the namelist
values come from mitjax/config (the run's parameter files, read with the Fortran's own namelist declarations), every
value the files do not set from the cited SET_DEFAULTS line (`fortran_default`: the line must be live in this
build, REVIEW_M0 #3), and the derivation statements of ini_parms.F / set_parms.F that the model needs are ported here
as code, each with its line. Gated by the oracle's STDOUT parameter printout (CONFIG_SUMMARY, mitjax/io/stdout.py),
mitjax/tests/test_init.py.

Three groups, one dataclass each (frozen; host-side values, never traced):
  * `GridParams` (`ini_parms_grid`): what the grid routines read (moved here from mitjax/model/grid.py, Task 10);
  * `TimeParams` (`ini_parms_time`): the time-stepping parameters of PARM03 and their derivation
    (ini_parms.F:1031-1182, set_parms.F:310-328);
  * `InitParams` (`ini_parms_init`): what INITIALISE_VARIA's field initialisation reads (PARM01 tRef/sRef via
    LOAD_REF_FILES, PARM03 pickup controls, PARM05 initial-condition files, the PARM01 switches that select branches).
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

import numpy as np

from mitjax.farray import FArray
from mitjax.model.grid import PI, UNSET_RL, UNSET_RS
from mitjax.params_io import RunParams, fortran_default

PRECFLOAT32 = 32            # EEPARAMS.h:63  PARAMETER ( precFloat32 = 32 )
PRECFLOAT64 = 64            # EEPARAMS.h:64  PARAMETER ( precFloat64 = 64 )
UNSET_I = 123456789         # EEPARAMS.h:84  PARAMETER ( UNSET_I    = 123456789  )

SD = "model/src/set_defaults.F"
IP = "model/src/ini_parms.F"


# ---------------------------------------------------------------------------------------------------------------
# helpers

def _blank(s):
    return s is None or str(s).strip() == ""


def _get(exp, rp, group, key, citation=None, fname="data"):
    """A scalar from the run's file, else the cited Fortran default (params_io.fortran_default, checked against this
    build)."""
    return rp.get(fname, group, key, default=fortran_default(citation, key, exp) if citation else None)


def _get_or_unset(rp, group, key, unset, fname="data"):
    """A scalar from the run's file, else the UNSET value the cited default assigns (the line reads `key = UNSET_*`)."""
    return rp.get(fname, group, key) if rp.has(fname, group, key) else unset


def _vector(rp, group, key, n, unset):
    """A 1-D REAL array of the namelist (elements the file sets), `unset` elsewhere (Fortran index 1..n)."""
    v = rp.nml.var("data", group, key)
    out = np.full(n, unset, dtype=np.float64)
    if v is not None and v.value is not None and not isinstance(v.value, str):
        try:
            items = dict(v.value).items()
        except (TypeError, ValueError):
            items = ()
        for (i,), x in items:
            out[i - 1] = x
    return out


# ---------------------------------------------------------------------------------------------------------------
# grid

@dataclass(frozen=True)
class Exch2Topology:
    """W2_EXCH2_TOPOLOGY.h values INI_LOCAL_GRID and LOAD_GRID_SPACING read under ALLOW_EXCH2 (single process):
    exch2_tBasex/y per W2 tile (0-based list index = tile number - 1), exch2_mydNx/y per tile (W2_EXCH2_TOPOLOGY.h:71-72
    declares them per tile; the readers take exch2_mydNx(1)), W2_myTileList in the order bi + (bj-1)*nSx;
    exch2_myFace per tile (INI_CURVILINEAR_GRID's file names; empty when the source does not know it)."""
    exch2_tBasex: tuple
    exch2_tBasey: tuple
    exch2_mydNx: tuple
    exch2_mydNy: tuple
    W2_myTileList: tuple
    source: str
    exch2_myFace: tuple = ()


@dataclass(frozen=True)
class GridParams:
    """The PARAMS.h / SET_GRID.h values read by the grid routines, as INI_PARMS leaves them (see ini_parms_grid)."""
    usingCartesianGrid: bool
    usingSphericalPolarGrid: bool
    usingCurvilinearGrid: bool
    usingCylindricalGrid: bool
    usingZCoords: bool
    usingPCoords: bool
    fluidIsWater: bool
    rotateGrid: bool
    deepAtmosphere: bool
    setInterFDr: bool
    setCenterDr: bool
    selectSigmaCoord: int
    selectCoriMap: int
    useMin4hFacEdges: bool
    rSphere: float
    recip_rSphere: float
    xgOrigin: float
    ygOrigin: float
    rF1: float                      # rF(1) as INI_PARMS leaves it: Ro_SeaLevel or UNSET_RS (ini_parms.F:1371-1387)
    seaLev_Z: float
    top_Pres: float
    rSigmaBnd: float
    cosPower: float
    hFacMin: float
    hFacMinDr: float
    f0: float
    beta: float
    fPrime: float
    omega: float
    latBandClimRelax: float
    delX: np.ndarray = field(repr=False)     # SET_GRID.h:34  delX(grid_maxNx)
    delY: np.ndarray = field(repr=False)     # SET_GRID.h:35  delY(grid_maxNy)
    delR: np.ndarray = field(repr=False)     # PARAMS.h delR(Nr)
    delRc: np.ndarray = field(repr=False)    # PARAMS.h delRc(Nr+1)
    delXFile: str = " "
    delYFile: str = " "
    delRFile: str = " "
    delRcFile: str = " "
    hybSigmFile: str = " "
    bathyFile: str = " "
    topoFile: str = " "
    addWwallFile: str = " "
    addSwallFile: str = " "
    buoyancyRelation: str = "OCEANIC"
    readBinaryPrec: int = 32
    useMNC: bool = False
    mnc_read_bathy: bool = False
    exch2: Exch2Topology | None = None
    # curvilinear grid input (lane B, Task 22): PARAMS.h horizGridFile (set_defaults.F:70 ' '),
    # radius_fromHorizGrid as ini_parms.F:1358-1360 leaves it, MNC_PARAMS.h readgrid_mnc (mnc_readparms.F:108)
    horizGridFile: str = " "
    radius_fromHorizGrid: float = 0.0
    readgrid_mnc: bool = False


def grid_max(cfg):
    """(grid_maxNx, grid_maxNy): the declared lengths of delX, delY (SET_GRID.h:19-26):
    ALLOW_EXCH2: W2_maxXStackNx, W2_maxYStackNy (pkg/exch2/W2_EXCH2_SIZE.h:37-42: W2_maxNbTiles = nSx*nSy*nPx*nPy*2,
    W2_maxXStackNx = W2_maxNbTiles*sNx, W2_maxYStackNy = W2_maxNbTiles*sNy); else Nx, Ny."""
    sz = cfg.size
    if cfg.cpp.ALLOW_EXCH2:
        if (cfg.experiment and _code_override(cfg, "W2_EXCH2_SIZE.h")):
            raise NotImplementedError("SET_GRID.h: an experiment's own W2_EXCH2_SIZE.h is not ported")
        W2_maxNbTiles = sz.nSx * sz.nSy * sz.nPx * sz.nPy * 2                  # W2_EXCH2_SIZE.h:37
        return W2_maxNbTiles * sz.sNx, W2_maxNbTiles * sz.sNy                  # W2_EXCH2_SIZE.h:39, :42
    return sz.Nx, sz.Ny                                                        # SET_GRID.h:24-25


def _code_override(cfg, name):
    from mitjax.config.params import code_path
    return (code_path(cfg) / name).exists()


def ini_parms_grid(exp, exch2=None):
    """The PARAMS.h / SET_GRID.h values the grid routines read, set as INI_PARMS @63cdc0b sets them:
    defaults from SET_DEFAULTS (model/src/set_defaults.F, cited per value), the namelist values of `data`
    (PARM01, PARM03, PARM04, PARM05) and `data.mnc`, then the derivations of ini_parms.F and set_parms.F (cited).

    `exp`: mitjax.config.params.Experiment. `exch2`: an Exch2Topology (required under ALLOW_EXCH2)."""
    cfg = exp.cfg
    rp = RunParams(exp.run)
    sz = cfg.size
    # buoyancyRelation -> vertical coordinate and fluid (set_defaults.F:176; ini_parms.F:447-465)
    buoyancyRelation = _get(exp, rp, "PARM01", "buoyancyRelation", f"{SD}:176")
    usingPCoords = usingZCoords = fluidIsWater = False
    if buoyancyRelation == "ATMOSPHERIC":                                       # ini_parms.F:451-453
        usingPCoords = True                                                     # (fluidIsAir = .TRUE.: not a
        # GridParams field; ini_parms_dyn raises on usingPCoords, ini_parms.F:1569-1575)
    elif buoyancyRelation == "OCEANICP":
        raise NotImplementedError("INI_PARMS: buoyancyRelation='OCEANICP' is not ported")
    elif buoyancyRelation == "OCEANIC":                                         # ini_parms.F:457-459
        usingZCoords = True
        fluidIsWater = True
    else:
        raise ValueError(f"S/R INI_PARMS: Bad value of buoyancyRelation {buoyancyRelation!r}")
    # rotation (set_defaults.F:111-116; ini_parms.F:487-495)
    f0 = _get(exp, rp, "PARM01", "f0", f"{SD}:111")
    beta = _get(exp, rp, "PARM01", "beta", f"{SD}:112")
    fPrime = _get(exp, rp, "PARM01", "fPrime", f"{SD}:113")
    rotationPeriod = _get(exp, rp, "PARM01", "rotationPeriod", f"{SD}:115")
    omega = _get_or_unset(rp, "PARM01", "omega", UNSET_RL)                       # set_defaults.F:116
    if omega == UNSET_RL:                                                       # ini_parms.F:487-490
        omega = 0.0
        if rotationPeriod != 0.0:
            omega = 2.0 * PI / rotationPeriod
    elif omega == 0.0:
        rotationPeriod = 0.0
    hFacMin = _get(exp, rp, "PARM01", "hFacMin", f"{SD}:180")
    # hFacMinDr: ini_parms.F:418-420 (UNSET), 643-645 (Dz, Dp, then hFacMinDrDefault, set_defaults.F:181)
    hFacMinDr = _get_or_unset(rp, "PARM01", "hFacMinDr", UNSET_RL)
    hFacMinDz = _get_or_unset(rp, "PARM01", "hFacMinDz", UNSET_RL)
    hFacMinDp = _get_or_unset(rp, "PARM01", "hFacMinDp", UNSET_RL)
    if hFacMinDr == UNSET_RL:
        hFacMinDr = hFacMinDz
    if hFacMinDr == UNSET_RL:
        hFacMinDr = hFacMinDp
    if hFacMinDr == UNSET_RL:
        hFacMinDr = fortran_default(f"{SD}:181", "hFacMinDrDefault", exp).value
    cosPower = _get(exp, rp, "PARM01", "cosPower", f"{SD}:152")
    selectCoriMap = _get(exp, rp, "PARM01", "selectCoriMap", f"{SD}:92")
    # set_defaults.F:355 readBinaryPrec = precFloat32; EEPARAMS.h:63 PARAMETER ( precFloat32 = 32 )
    readBinaryPrec = rp.get("data", "PARM01", "readBinaryPrec") if rp.has("data", "PARM01", "readBinaryPrec") \
        else PRECFLOAT32
    # PARM03: latBandClimRelax (ini_parms.F:967 UNSET_RL)
    latBandClimRelax = _get_or_unset(rp, "PARM03", "latBandClimRelax", UNSET_RL)
    # PARM04 (set_defaults.F:42-97; ini_parms.F:1204-1212 delZ/delP/delR/dxSpacing/dySpacing = UNSET_RL)
    usingCartesianGrid = _get(exp, rp, "PARM04", "usingCartesianGrid", f"{SD}:81")
    usingSphericalPolarGrid = _get(exp, rp, "PARM04", "usingSphericalPolarGrid", f"{SD}:83")
    usingCurvilinearGrid = _get(exp, rp, "PARM04", "usingCurvilinearGrid", f"{SD}:86")
    usingCylindricalGrid = _get(exp, rp, "PARM04", "usingCylindricalGrid", f"{SD}:90")
    deepAtmosphere = _get(exp, rp, "PARM04", "deepAtmosphere", f"{SD}:71")
    rotateGrid = _get(exp, rp, "PARM04", "rotateGrid", f"{SD}:94")
    phiEuler = _get(exp, rp, "PARM04", "phiEuler", f"{SD}:95")
    thetaEuler = _get(exp, rp, "PARM04", "thetaEuler", f"{SD}:96")
    psiEuler = _get(exp, rp, "PARM04", "psiEuler", f"{SD}:97")
    useMin4hFacEdges = _get(exp, rp, "PARM04", "useMin4hFacEdges", f"{SD}:55")
    selectSigmaCoord = _get(exp, rp, "PARM04", "selectSigmaCoord", f"{SD}:48")
    rSigmaBnd = _get_or_unset(rp, "PARM04", "rSigmaBnd", UNSET_RL)               # set_defaults.F:47
    seaLev_Z = _get_or_unset(rp, "PARM04", "seaLev_Z", UNSET_RL)                 # set_defaults.F:45
    top_Pres = _get_or_unset(rp, "PARM04", "top_Pres", UNSET_RL)                 # set_defaults.F:46
    rSphere = _get_or_unset(rp, "PARM04", "rSphere", UNSET_RL)                   # set_defaults.F:84
    xgOrigin = _get_or_unset(rp, "PARM04", "xgOrigin", UNSET_RL)                 # set_defaults.F:72
    ygOrigin = _get_or_unset(rp, "PARM04", "ygOrigin", UNSET_RL)                 # set_defaults.F:73
    Ro_SeaLevel = _get_or_unset(rp, "PARM04", "Ro_SeaLevel", UNSET_RL)           # ini_parms.F:381
    dxSpacing = _get_or_unset(rp, "PARM04", "dxSpacing", UNSET_RL)               # ini_parms.F:1211
    dySpacing = _get_or_unset(rp, "PARM04", "dySpacing", UNSET_RL)               # ini_parms.F:1212
    files = {}
    for key, line in (("delXFile", 68), ("delYFile", 69), ("delRFile", 42), ("delRcFile", 43),
                      ("hybSigmFile", 44)):
        files[key] = _get(exp, rp, "PARM04", key, f"{SD}:{line}")
    for key, line in (("bathyFile", 360), ("topoFile", 361), ("addWwallFile", 362), ("addSwallFile", 363)):
        files[key] = _get(exp, rp, "PARM05", key, f"{SD}:{line}")
    horizGridFile = _get(exp, rp, "PARM04", "horizGridFile", f"{SD}:70")                            # ' '
    radius_fromHorizGrid = _get_or_unset(rp, "PARM04", "radius_fromHorizGrid", UNSET_RL)  # set_defaults.F:87
    for key in ("thetaMin", "phiMin", "rkFac", "groundAtK1"):
        if rp.has("data", "PARM04", key):
            raise NotImplementedError(f"INI_PARMS: retired parameter {key} is not ported")

    # delX, delY: set_defaults.F:74-79 (UNSET_RL over the declared length), namelist, then ini_parms.F:1292-1323.
    # Declared delX(grid_maxNx), delY(grid_maxNy) (SET_GRID.h:19-26; W2_maxXStackNx under ALLOW_EXCH2); only
    # elements 1..gridNx are set by dxSpacing and read (ini_parms.F:1304, load_grid_spacing.F:116,
    # ini_local_grid.F:128-149), the rest stays UNSET_RL as SET_DEFAULTS leaves it.
    grid_maxNx, grid_maxNy = grid_max(cfg)
    if cfg.cpp.ALLOW_EXCH2:
        if exch2 is None:
            raise ValueError("ini_parms_grid: ALLOW_EXCH2 needs the W2 topology (Exch2Topology)")
        gridNx, gridNy = exch2.exch2_mydNx[0], exch2.exch2_mydNy[0]           # ini_parms.F:326-328
    else:
        gridNx, gridNy = sz.Nx, sz.Ny                                           # ini_parms.F:330-331
    delX = _vector(rp, "PARM04", "delX", grid_maxNx, UNSET_RL)
    delY = _vector(rp, "PARM04", "delY", grid_maxNy, UNSET_RL)
    goptCount = int(delX[0] != UNSET_RL) + int(dxSpacing != UNSET_RL) + int(not _blank(files["delXFile"]))
    if goptCount > 1:                                                           # ini_parms.F:1293-1302
        raise ValueError("Too many specifications for delX:Specify only one of delX, dxSpacing or delXfile")
    if dxSpacing != UNSET_RL:                                                   # ini_parms.F:1303-1307
        delX[:gridNx] = dxSpacing
    goptCount = int(delY[0] != UNSET_RL) + int(dySpacing != UNSET_RL) + int(not _blank(files["delYFile"]))
    if goptCount > 1:                                                           # ini_parms.F:1309-1318
        raise ValueError("Too many specifications for delY:Specify only one of delY, dySpacing or delYfile")
    if dySpacing != UNSET_RL:                                                   # ini_parms.F:1319-1323
        delY[:gridNy] = dySpacing
    goptCount = sum(int(x) for x in (usingCartesianGrid, usingSphericalPolarGrid, usingCurvilinearGrid,
                                     usingCylindricalGrid))
    if goptCount > 1:                                                           # ini_parms.F:1325-1336
        raise ValueError("S/R INI_PARMS: More than one coordinate system requested")
    if goptCount < 1:                                                           # ini_parms.F:1337-1348
        usingCartesianGrid = True
    if rSphere == UNSET_RL:                                                     # ini_parms.F:1350-1357
        if usingCurvilinearGrid and radius_fromHorizGrid != UNSET_RL:           # :1351-1352
            rSphere = radius_fromHorizGrid                                      # :1353
        else:
            rSphere = 6370.e3                                                   # :1355  rSphere = 6370. _d 3
    if radius_fromHorizGrid == UNSET_RL:                                        # :1358-1360
        radius_fromHorizGrid = rSphere
    if rSphere != 0.0:                                                          # ini_parms.F:1361-1365
        recip_rSphere = 1.0 / rSphere
    else:
        recip_rSphere = 0.0
    if phiEuler != 0.0 or thetaEuler != 0.0 or psiEuler != 0.0:                 # ini_parms.F:1367-1368
        rotateGrid = True
    if Ro_SeaLevel != UNSET_RL:                                                 # ini_parms.F:1371-1387
        if usingZCoords and seaLev_Z != UNSET_RL:
            raise ValueError('S/R INI_PARMS: Cannot set both "Ro_SeaLevel" and "seaLev_Z"')
        rF1 = Ro_SeaLevel
    else:
        rF1 = UNSET_RS
    if top_Pres == UNSET_RL:                                                    # ini_parms.F:1388
        top_Pres = 0.0
    if seaLev_Z == UNSET_RL:                                                    # ini_parms.F:1389
        seaLev_Z = 0.0
    if xgOrigin == UNSET_RL:                                                    # ini_parms.F:1391
        xgOrigin = 0.0
    if ygOrigin == UNSET_RL:                                                    # ini_parms.F:1392-1406 (0. for all)
        ygOrigin = 0.0
    # delRc / setCenterDr: ini_parms.F:1409-1428 (delRc = UNSET_RL, set_defaults.F:52-54)
    delRc = _vector(rp, "PARM04", "delRc", sz.Nr + 1, UNSET_RL)
    setCenterDr = False
    for k in range(1, sz.Nr + 2):
        if delRc[k - 1] == UNSET_RL:
            if setCenterDr:
                raise ValueError(f"S/R INI_PARMS: No value for delRc at k = {k}")
        else:
            if k == 1:
                setCenterDr = True
            if not setCenterDr:
                raise ValueError(f"S/R INI_PARMS: No value for delRc at k < {k}")
    # delR / setInterFDr: ini_parms.F:1430-1454 (delZ, delP, delR = UNSET_RL, ini_parms.F:1205-1209)
    delZ = _vector(rp, "PARM04", "delZ", sz.Nr, UNSET_RL)
    delP = _vector(rp, "PARM04", "delP", sz.Nr, UNSET_RL)
    delR = _vector(rp, "PARM04", "delR", sz.Nr, UNSET_RL)
    zIn, pIn, rIn = False, False, setCenterDr                                   # ini_parms.F:350-352, 1429
    setInterFDr = False
    for k in range(1, sz.Nr + 1):
        if delZ[k - 1] != UNSET_RL:
            zIn = True
        if delP[k - 1] != UNSET_RL:
            pIn = True
        if delR[k - 1] != UNSET_RL:
            rIn = True
        if delR[k - 1] == UNSET_RL:
            delR[k - 1] = delZ[k - 1]
        if delR[k - 1] == UNSET_RL:
            delR[k - 1] = delP[k - 1]
        if delR[k - 1] == UNSET_RL:
            if setInterFDr:
                raise ValueError(f"S/R INI_PARMS: No value for delZ/delP/delR at k = {k}")
        else:
            if k == 1:
                setInterFDr = True
            if not setInterFDr:
                raise ValueError(f"S/R INI_PARMS: No value for delZ/delP/delR at k < {k}")
    if rp.has("data", "PARM01", "hFacMinDz"):
        zIn = True                                                              # ini_parms.F:640
    if rp.has("data", "PARM01", "hFacMinDp"):
        pIn = True                                                              # ini_parms.F:641
    if rp.has("data", "PARM01", "hFacMinDr"):
        rIn = True                                                              # ini_parms.F:642
    if int(zIn) + int(pIn) + int(rIn) > 1:                                      # ini_parms.F:1456-1465
        raise ValueError("S/R INI_PARMS: Cannot mix z, p and r in the input data.")
    if not _blank(files["delRcFile"]) or not _blank(files["delRFile"]):         # ini_parms.F:1467-1484
        raise NotImplementedError("INI_PARMS: delRFile / delRcFile is not ported")
    # set_parms.F:73-81: selectCoriMap default
    if selectCoriMap == -1:
        if usingCartesianGrid or usingCylindricalGrid:
            selectCoriMap = 1
        else:
            selectCoriMap = 2
    # MNC bathymetry read switch (ini_depths.F:106): useMNC (packages_boot.F), mnc_read_bathy
    # (pkg/mnc/mnc_readparms.F:111)
    useMNC = cfg.use_flag("useMNC") if any(k.lower() == "usemnc" for k, _ in cfg.use) else False
    mnc_read_bathy = False
    if useMNC:
        mnc_read_bathy = rp.get("data.mnc", "MNC_01", "mnc_read_bathy",
                                default=fortran_default("pkg/mnc/mnc_readparms.F:111", "mnc_read_bathy", exp))
    readgrid_mnc = False
    if useMNC:                                                                  # INI_CURVILINEAR_GRID's MNC branch
        readgrid_mnc = rp.get("data.mnc", "MNC_01", "readgrid_mnc",
                              default=fortran_default("pkg/mnc/mnc_readparms.F:108", "readgrid_mnc", exp))
    return GridParams(
        usingCartesianGrid=usingCartesianGrid, usingSphericalPolarGrid=usingSphericalPolarGrid,
        usingCurvilinearGrid=usingCurvilinearGrid, usingCylindricalGrid=usingCylindricalGrid,
        usingZCoords=usingZCoords, usingPCoords=usingPCoords, fluidIsWater=fluidIsWater, rotateGrid=rotateGrid,
        deepAtmosphere=deepAtmosphere, setInterFDr=setInterFDr, setCenterDr=setCenterDr,
        selectSigmaCoord=selectSigmaCoord, selectCoriMap=selectCoriMap, useMin4hFacEdges=useMin4hFacEdges,
        rSphere=float(rSphere), recip_rSphere=float(recip_rSphere), xgOrigin=float(xgOrigin),
        ygOrigin=float(ygOrigin), rF1=float(rF1), seaLev_Z=float(seaLev_Z), top_Pres=float(top_Pres),
        rSigmaBnd=float(rSigmaBnd), cosPower=float(cosPower), hFacMin=float(hFacMin), hFacMinDr=float(hFacMinDr),
        f0=float(f0), beta=float(beta), fPrime=float(fPrime), omega=float(omega),
        latBandClimRelax=float(latBandClimRelax), delX=delX, delY=delY, delR=delR, delRc=delRc,
        buoyancyRelation=buoyancyRelation, readBinaryPrec=int(readBinaryPrec), useMNC=bool(useMNC),
        mnc_read_bathy=bool(mnc_read_bathy), exch2=exch2, horizGridFile=horizGridFile,
        radius_fromHorizGrid=float(radius_fromHorizGrid), readgrid_mnc=bool(readgrid_mnc), **files)


# ---------------------------------------------------------------------------------------------------------------
# time stepping

def _nint(x):
    """Fortran NINT: nearest integer, halves away from zero (Python round() rounds halves to even)."""
    return int(np.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


@dataclass(frozen=True)
class TimeParams:
    """PARAMS.h time-stepping values as INI_PARMS and SET_PARMS leave them (ini_parms_time)."""
    deltaT: float
    deltaTClock: float
    deltaTMom: float
    deltaTFreeSurf: float
    deltaTtracer: float          # ini_parms.F local, read in PARM03 (kept: dTtracerLev(1) and checks use it)
    dTtracerLev: np.ndarray = field(repr=False)     # PARAMS.h dTtracerLev(Nr)
    baseTime: float = 0.0
    startTime: float = 0.0
    endTime: float = 0.0
    nIter0: int = 0
    nTimeSteps: int = 0
    nEndIter: int = 0


def ini_parms_time(exp):
    """The time-stepping parameters (namelist PARM03) as INI_PARMS @63cdc0b and SET_PARMS derive them.

    Defaults: set_defaults.F:300-312 (deltaT, deltaTMom, deltaTFreeSurf, dTtracerLev(k) = 0. _d 0 in the DO loop
    :303-305, baseTime, nIter0 = -1, startTime = UNSET_RL, nTimeSteps, nEndIter, endTime); deltaTtracer = 0. _d 0
    (ini_parms.F:968, before the PARM03 READ). deltaTClock has no default statement: it lives in COMMON /PARM_R/
    (PARAMS.h) with no DATA statement, i.e. in zero-initialised static storage (.bss) of the gfortran oracle, and
    ini_parms.F:1059 tests it against 0.; a run that does not set it starts from 0. Derivations: ini_parms.F:1031-1068
    (time-step sizes), :1121-1182 (start/end time and iteration counts), set_parms.F:310-328 (endTime adjusted to a
    whole number of steps). Error branches raise with the Fortran message."""
    cfg = exp.cfg
    rp = RunParams(exp.run)
    Nr = cfg.size.Nr
    deltaT = _get(exp, rp, "PARM03", "deltaT", f"{SD}:300")
    deltaTMom = _get(exp, rp, "PARM03", "deltaTMom", f"{SD}:301")
    deltaTFreeSurf = _get(exp, rp, "PARM03", "deltaTFreeSurf", f"{SD}:302")
    deltaTClock = rp.get("data", "PARM03", "deltaTClock") if rp.has("data", "PARM03", "deltaTClock") else 0.0
    deltaTtracer = _get(exp, rp, "PARM03", "deltaTtracer", f"{IP}:968")
    dTtracerLev = _vector(rp, "PARM03", "dTtracerLev", Nr, 0.0)                 # set_defaults.F:303-305
    baseTime = _get(exp, rp, "PARM03", "baseTime", f"{SD}:306")
    nIter0 = _get(exp, rp, "PARM03", "nIter0", f"{SD}:307")
    startTime = _get_or_unset(rp, "PARM03", "startTime", UNSET_RL)               # set_defaults.F:308
    nTimeSteps = _get(exp, rp, "PARM03", "nTimeSteps", f"{SD}:309")
    nEndIter = _get(exp, rp, "PARM03", "nEndIter", f"{SD}:311")
    endTime = _get(exp, rp, "PARM03", "endTime", f"{SD}:312")
    # :1033-1042  o Time step size
    if deltaTtracer != dTtracerLev[0] and deltaTtracer != 0. and dTtracerLev[0] != 0.:
        raise ValueError("S/R INI_PARMS: deltaTtracer & dTtracerLev(1) not equal")
    elif dTtracerLev[0] != 0.:
        deltaTtracer = dTtracerLev[0]
    if deltaT == 0.:                                                            # :1043-1046
        deltaT = deltaTClock
    if deltaT == 0.:
        deltaT = deltaTtracer
    if deltaT == 0.:
        deltaT = deltaTMom
    if deltaT == 0.:
        deltaT = deltaTFreeSurf
    if deltaT == 0.:                                                            # :1047-1056
        raise ValueError('S/R INI_PARMS: need to specify in file "data", namelist "PARM03"\n'
                         "S/R INI_PARMS:  a model timestep (in s) deltaT or deltaTClock= ?")
    if deltaTMom == 0.:                                                         # :1057
        deltaTMom = deltaT
    if deltaTtracer == 0.:                                                      # :1058
        deltaTtracer = deltaT
    if deltaTClock == 0.:                                                       # :1059
        deltaTClock = deltaT
    for k in range(1, Nr + 1):                                                  # :1060-1062
        if dTtracerLev[k - 1] == 0.:
            dTtracerLev[k - 1] = deltaTtracer
    if deltaTFreeSurf == 0.:                                                    # :1068
        deltaTFreeSurf = deltaTMom
    # :1121-1134  o start time & nIter0
    if startTime == UNSET_RL and nIter0 == -1:
        startTime = baseTime
        nIter0 = 0
    elif startTime == UNSET_RL:
        startTime = baseTime + deltaTClock * float(nIter0)                      # DFLOAT(nIter0)
    elif nIter0 == -1:
        nIter0 = _nint((startTime - baseTime) / deltaTClock)
    elif baseTime == 0.:
        baseTime = startTime - deltaTClock * float(nIter0)
    # :1136-1154
    if nTimeSteps == 0 and nEndIter != 0:
        nTimeSteps = nEndIter - nIter0
    if nTimeSteps == 0 and endTime != 0.:
        nTimeSteps = _nint((endTime - startTime) / deltaTClock)
    if nEndIter == 0 and nTimeSteps != 0:
        nEndIter = nIter0 + nTimeSteps
    if nEndIter == 0 and endTime != 0.:
        nEndIter = _nint((endTime - baseTime) / deltaTClock)
    if endTime == 0. and nTimeSteps != 0:
        endTime = startTime + deltaTClock * float(nTimeSteps)
    if endTime == 0. and nEndIter != 0:
        endTime = baseTime + deltaTClock * float(nEndIter)
    # :1157-1182  o Consistent?
    if startTime != baseTime + deltaTClock * float(nIter0):
        raise ValueError("S/R INI_PARMS: startTime, baseTime and nIter0 are inconsistent")
    if nEndIter != nIter0 + nTimeSteps:
        raise ValueError("S/R INI_PARMS: nIter0, nTimeSteps and nEndIter are inconsistent")
    if nTimeSteps != _nint((endTime - startTime) / deltaTClock):
        raise ValueError("S/R INI_PARMS: both endTime and nTimeSteps have been set")
    # set_parms.F:310-328: adjust endTime for a sub-timestep mismatch
    tmpVar = startTime + deltaTClock * float(nTimeSteps)
    if endTime != tmpVar:
        if abs(endTime - tmpVar) > deltaTClock * 1.e-6:                         # 1. _d -6
            raise NotImplementedError("SET_PARMS: (endTime-baseTime) not multiple of time-step (warning branch)")
        endTime = tmpVar
    return TimeParams(deltaT=float(deltaT), deltaTClock=float(deltaTClock), deltaTMom=float(deltaTMom),
                      deltaTFreeSurf=float(deltaTFreeSurf), deltaTtracer=float(deltaTtracer),
                      dTtracerLev=dTtracerLev, baseTime=float(baseTime), startTime=float(startTime),
                      endTime=float(endTime), nIter0=int(nIter0), nTimeSteps=int(nTimeSteps),
                      nEndIter=int(nEndIter))


# ---------------------------------------------------------------------------------------------------------------
# initial conditions

@dataclass(frozen=True)
class InitParams:
    """What INITIALISE_VARIA's field initialisation reads (ini_parms_init)."""
    tRef: np.ndarray = field(repr=False)     # PARAMS.h tRef(Nr), as LOAD_REF_FILES leaves it
    sRef: np.ndarray = field(repr=False)     # PARAMS.h sRef(Nr)
    hydrogThetaFile: str = " "
    hydrogSaltFile: str = " "
    uVelInitFile: str = " "
    vVelInitFile: str = " "
    pSurfInitFile: str = " "
    maskIniTemp: bool = True
    maskIniSalt: bool = True
    checkIniTemp: bool = True
    checkIniSalt: bool = True
    allowFreezing: bool = False
    pickupSuff: str = " "
    pickupStrictlyMatch: bool = True
    pickup_read_mdsio: bool = True
    usePickupBeforeC54: bool = False
    rwSuffixType: int = 0
    readBinaryPrec: int = 32
    exactConserv: bool = False
    nonlinFreeSurf: int = 0
    select_rStar: int = 0
    momStepping: bool = True
    nonHydrostatic: bool = False
    storePhiHyd4Phys: bool = False
    AdamsBashforthGt: bool = False
    AdamsBashforthGs: bool = False
    AdamsBashforth_T: bool = False
    AdamsBashforth_S: bool = False
    hFacInf: float = 0.2
    useOffLine: bool = False
    useMNC: bool = False
    mnc_read_theta: bool = False
    mnc_read_salt: bool = False
    selectSigmaCoord: int = 0
    quasiHydrostatic: bool = False
    staggerTimeStep: bool = False


def load_ref_files_tref_sref(exp):
    """tRef, sRef as LOAD_REF_FILES leaves them (model/src/load_ref_files.F:45-104 @63cdc0b): unset levels take the
    level above, level 1 the tracerDefault `20.` / `30.`, under fluidIsAir `300.` / `0.` (REAL*4 literals, exact;
    lane B) or thetaConst. tRefFile / sRefFile raise (no M1 variant sets them)."""
    rp = RunParams(exp.run)
    Nr = exp.cfg.size.Nr
    tRefFile = _get(exp, rp, "PARM01", "tRefFile", f"{SD}:57")
    sRefFile = _get(exp, rp, "PARM01", "sRefFile", f"{SD}:58")
    if not _blank(tRefFile) or not _blank(sRefFile):
        raise NotImplementedError("LOAD_REF_FILES: tRefFile / sRefFile is not ported")
    thetaConst = _get_or_unset(rp, "PARM01", "thetaConst", UNSET_RL)            # set_defaults.F:61
    tRef = _vector(rp, "PARM01", "tRef", Nr, UNSET_RL)                          # set_defaults.F:63
    sRef = _vector(rp, "PARM01", "sRef", Nr, UNSET_RL)                          # set_defaults.F:64
    fluidIsAir = _get(exp, rp, "PARM01", "buoyancyRelation", f"{SD}:176") == "ATMOSPHERIC"   # ini_parms.F:451-453
    tracerDefault = 20.                                                         # load_ref_files.F:47
    if fluidIsAir:                                                              # :48 (lane B)
        tracerDefault = 300.
    if thetaConst != UNSET_RL:                                                  # :49
        tracerDefault = thetaConst
    for k in range(1, Nr + 1):                                                  # :50-53
        if tRef[k - 1] == UNSET_RL:
            tRef[k - 1] = tracerDefault
        tracerDefault = tRef[k - 1]
    tracerDefault = 30.                                                         # :79
    if fluidIsAir:                                                              # :80 (lane B)
        tracerDefault = 0.
    for k in range(1, Nr + 1):                                                  # :81-84
        if sRef[k - 1] == UNSET_RL:
            sRef[k - 1] = tracerDefault
        tracerDefault = sRef[k - 1]
    return tRef, sRef


def ini_parms_init(exp):
    """The parameters of the field initialisation (INI_FIELDS and callees, READ_PICKUP), cited per value."""
    rp = RunParams(exp.run)
    cfg = exp.cfg
    tRef, sRef = load_ref_files_tref_sref(exp)
    g = lambda grp, key, line: _get(exp, rp, grp, key, f"{SD}:{line}")         # noqa: E731
    readBinaryPrec = rp.get("data", "PARM01", "readBinaryPrec") if rp.has("data", "PARM01", "readBinaryPrec") \
        else PRECFLOAT32                                                        # set_defaults.F:355 = precFloat32
    storePhiHyd4Phys = _store_phi_hyd4phys(exp, rp)
    ab = gad_init_fixed_ab_flags(exp, rp)
    useMNC = _use(cfg, "useMNC")
    mnc = {"mnc_read_theta": False, "mnc_read_salt": False}
    if useMNC:                                                                  # pkg/mnc/mnc_readparms.F:112-113
        for key, line in (("mnc_read_salt", 112), ("mnc_read_theta", 113)):
            mnc[key] = rp.get("data.mnc", "MNC_01", key,
                              default=fortran_default(f"pkg/mnc/mnc_readparms.F:{line}", key, exp))
    return InitParams(
        tRef=tRef, sRef=sRef,
        hydrogThetaFile=g("PARM05", "hydrogThetaFile", 365), hydrogSaltFile=g("PARM05", "hydrogSaltFile", 364),
        uVelInitFile=g("PARM05", "uVelInitFile", 384), vVelInitFile=g("PARM05", "vVelInitFile", 385),
        pSurfInitFile=g("PARM05", "pSurfInitFile", 386),
        maskIniTemp=g("PARM05", "maskIniTemp", 366), maskIniSalt=g("PARM05", "maskIniSalt", 367),
        checkIniTemp=g("PARM05", "checkIniTemp", 368), checkIniSalt=g("PARM05", "checkIniSalt", 369),
        allowFreezing=g("PARM01", "allowFreezing", 220),
        pickupSuff=g("PARM03", "pickupSuff", 337), pickupStrictlyMatch=g("PARM03", "pickupStrictlyMatch", 338),
        pickup_read_mdsio=g("PARM03", "pickup_read_mdsio", 342),
        usePickupBeforeC54=g("PARM01", "usePickupBeforeC54", 225), rwSuffixType=g("PARM03", "rwSuffixType", 357),
        readBinaryPrec=int(readBinaryPrec),
        exactConserv=g("PARM01", "exactConserv", 254), nonlinFreeSurf=g("PARM01", "nonlinFreeSurf", 257),
        select_rStar=g("PARM01", "select_rStar", 260), momStepping=g("PARM01", "momStepping", 192),
        nonHydrostatic=g("PARM01", "nonHydrostatic", 215), storePhiHyd4Phys=storePhiHyd4Phys,
        hFacInf=float(g("PARM01", "hFacInf", 258)), useOffLine=_use(cfg, "useOffLine"), useMNC=useMNC,
        mnc_read_theta=mnc["mnc_read_theta"], mnc_read_salt=mnc["mnc_read_salt"],
        selectSigmaCoord=g("PARM04", "selectSigmaCoord", 48), quasiHydrostatic=g("PARM01", "quasiHydrostatic", 216),
        staggerTimeStep=g("PARM01", "staggerTimeStep", 183), **ab)


def _use(cfg, name):
    """A package switch as packages_boot.F leaves it (mitjax/config use flags); .FALSE. if the build has none."""
    return cfg.use_flag(name) if any(k.lower() == name.lower() for k, _ in cfg.use) else False


# GAD.h:23-33 @63cdc0b
ENUM_CENTERED_2ND = 2       # GAD.h:25  PARAMETER(ENUM_CENTERED_2ND=2)
ENUM_UPWIND_3RD = 3         # GAD.h:29  PARAMETER(ENUM_UPWIND_3RD=3)
ENUM_CENTERED_4TH = 4       # GAD.h:33  PARAMETER(ENUM_CENTERED_4TH=4)


def gad_init_fixed_ab_flags(exp, rp=None):
    """AdamsBashforthGt/Gs, AdamsBashforth_T/S as GAD_INIT_FIXED sets them (pkg/generic_advdiff/gad_init_fixed.F:142-162
    @63cdc0b; READ_PICKUP and WRITE_PICKUP read them). Under ALLOW_GENERIC_ADVDIFF only; without it READ_PICKUP sees
    the .FALSE. PARAMETERs of read_pickup.F:70-79. To move into pkg/generic_advdiff/gad_init_fixed.py when GAD_INIT_FIXED
    is ported as a whole (GAD lanes / Task 13)."""
    rp = RunParams(exp.run) if rp is None else rp
    out = dict(AdamsBashforthGt=False, AdamsBashforthGs=False, AdamsBashforth_T=False, AdamsBashforth_S=False)
    if not exp.cfg.cpp.ALLOW_GENERIC_ADVDIFF:
        return out                                                              # read_pickup.F:70-79
    tempAdvScheme = _get(exp, rp, "PARM01", "tempAdvScheme", f"{SD}:226")
    saltAdvScheme = _get(exp, rp, "PARM01", "saltAdvScheme", f"{SD}:227")
    tempStepping = _get(exp, rp, "PARM01", "tempStepping", f"{SD}:194")
    saltStepping = _get(exp, rp, "PARM01", "saltStepping", f"{SD}:198")
    doAB_onGtGs = _get(exp, rp, "PARM03", "doAB_onGtGs", f"{SD}:316")
    if tempAdvScheme in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH):    # :147-151
        out["AdamsBashforthGt"] = tempStepping
    if saltAdvScheme in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH):    # :152-156
        out["AdamsBashforthGs"] = saltStepping
    if not doAB_onGtGs:                                                         # :157-162
        out["AdamsBashforth_T"] = out["AdamsBashforthGt"]
        out["AdamsBashforth_S"] = out["AdamsBashforthGs"]
        out["AdamsBashforthGt"] = False
        out["AdamsBashforthGs"] = False
    return out


def _store_phi_hyd4phys(exp, rp):
    """storePhiHyd4Phys (set_parms.F:272-306 @63cdc0b): selectP_inEOS_Zc defaulted from eosType (:275-282), range
    checks (:283-299), then storePhiHyd4Phys = selectP_inEOS_Zc.GE.2 .OR. useAtm_Phys (:304, :306)."""
    sel = select_p_in_eos_zc(exp, rp)
    useAtm_Phys = exp.cfg.use_flag("useAtm_Phys") if any(k.lower() == "useatm_phys" for k, _ in exp.cfg.use) \
        else False
    return bool(sel >= 2 or useAtm_Phys)                                         # :304, :306


def select_p_in_eos_zc(exp, rp):
    """selectP_inEOS_Zc as SET_PARMS leaves it (set_parms.F:272-299 @63cdc0b, z coordinates)."""
    eosType = _get(exp, rp, "PARM01", "eosType", f"{SD}:175")
    nonHydrostatic = _get(exp, rp, "PARM01", "nonHydrostatic", f"{SD}:215")
    sel = _get_or_unset(rp, "PARM01", "selectP_inEOS_Zc", UNSET_I)               # set_defaults.F:177
    # usingZCoords: every ported buoyancyRelation is OCEANIC (ini_parms_grid)
    if sel == UNSET_I:                                                          # :275-282
        sel = 2 if eosType in ("JMD95P", "UNESCO", "MDJWF", "TEOS10") else 0
    elif sel < 0 or sel > 3:                                                    # :283-288
        raise ValueError(f"SET_PARMS: selectP_inEOS_Zc={sel:9d} : invalid selection")
    elif not nonHydrostatic:                                                    # :289-290
        sel = min(sel, 2)  # MINMAX-INT: integer (no tie or NaN case)
    if eosType in ("LINEAR", "POLY3 ", "POLY3") and sel != 0:                   # :292-299
        raise ValueError(f"SET_PARMS: selectP_inEOS_Zc={sel:9d} : invalid with eosType={eosType}")
    return int(sel)


def set_ref_state_hyd_press_1d(selectP_inEOS_Zc):
    """set_ref_state.F:178-188 @63cdc0b: IF ( selectP_inEOS_Zc.GE.1 ) CALL FIND_HYD_PRESS_1D( pRef4EOS, pRefIntF,
    rhoRef, ... ), the reference pressure in hydrostatic balance with the reference density. FIND_HYD_PRESS_1D is not
    ported. With selectP_inEOS_Zc = 1 ("use pRef = integral{-g*rho(Tref,Sref,pRef)*dz}", PARAMS.h:178-182) the EOS
    reads that pRef4EOS (PRESSURE_FOR_EOS, pressure_for_eos.F:90-95; OPPS STATE1, opps_calc.F:500-501:
    selectP_inEOS_Zc.LE.1), so the value 1 is refused. With 2 or 3 pRef4EOS is not read, and the updated rhoRef /
    dBdrRef feed only code that is not ported (cg3d, implicit internal gravity waves); 0 does not call it. Lane API
    session 2 (Nikolay 2026-10-07)."""
    if selectP_inEOS_Zc == 1:
        raise NotImplementedError("SET_REF_STATE: selectP_inEOS_Zc = 1 needs FIND_HYD_PRESS_1D "
                                  "(set_ref_state.F:178-188), which is not ported")


def set_ref_state_eos(params, grid, exp):
    """The EOS reference pressure of SET_REF_STATE and the surface reference pressure, added to `params` (R5 arm,
    plan Task 16: FIND_RHO_2D / PRESSURE_FOR_EOS of the JMD95Z / JMD95P / UNESCO equations of state read them):
      * surf_pRef (PARAMS.h; data PARM01 or set_defaults.F:103 `101325. _d 0`; INI_EOS's EOS_CHECK changes it only
        between its own save and restore, ini_eos.F:549/:607-..);
      * selectP_inEOS_Zc (static, set_parms.F:272-299);
      * phiRef(2*Nr+1), pRef4EOS(Nr) of set_ref_state.F:51-100 @63cdc0b for buoyancyRelation = 'OCEANIC' with
        gravityFile = ' ' (:85-100; the other branches raise): host float64, one IEEE operation per Fortran operation
        in the Fortran's order, from the grid's rC, rF, gravitySign (INI_VERTICAL_GRID).
    Returns the new Params."""
    rp = RunParams(exp.run)
    Nr = exp.cfg.size.Nr
    buoyancyRelation = _get(exp, rp, "PARM01", "buoyancyRelation", f"{SD}:176")
    if buoyancyRelation.strip() != "OCEANIC":
        raise NotImplementedError("SET_REF_STATE: phiRef / pRef4EOS only for buoyancyRelation='OCEANIC'")
    if rp.has("data", "PARM05", "gravityFile"):
        raise NotImplementedError("SET_REF_STATE: gravityFile is not ported")
    surf_pRef = _get(exp, rp, "PARM01", "surf_pRef", f"{SD}:103")
    top_Pres = _get_or_unset(rp, "PARM04", "top_Pres", UNSET_RL)                 # set_defaults.F:46
    if top_Pres == UNSET_RL:                                                    # ini_parms.F:1388
        top_Pres = 0.0
    rhoConst = np.float64(params.rhoConst)
    recip_rhoConst = np.float64(params.recip_rhoConst)
    gravity = np.float64(params.gravity)
    gravitySign = np.float64(grid.gravitySign)
    rC = np.asarray(grid.rC.data, np.float64)
    rF = np.asarray(grid.rF.data, np.float64)
    phiRef = np.zeros(2 * Nr + 1)                                               # :51-53  phiRef(k) = 0.
    pRef4EOS = np.zeros(Nr)                                                     # :61  0. _d 0
    pRefIntF = np.zeros(Nr + 1)                                                 # :65  0. _d 0
    phiRef[0] = np.float64(top_Pres) * recip_rhoConst                           # :85  phiRef(1)
    pRefIntF[0] = np.float64(top_Pres)                                          # :86
    for k in range(1, Nr + 1):                                                  # :88-99
        phiRef[2*k - 1] = phiRef[0] + (rC[k-1] - rF[0])*gravity*gravitySign      # phiRef(2*k)
        phiRef[2*k] = phiRef[0] + (rF[k] - rF[0])*gravity*gravitySign             # phiRef(2*k+1)
        pRef4EOS[k-1] = pRefIntF[0] + rhoConst*(rC[k-1] - rF[0])*gravity*gravitySign
        pRefIntF[k] = pRefIntF[0] + rhoConst*(rF[k] - rF[0])*gravity*gravitySign
    sel = select_p_in_eos_zc(exp, rp)
    set_ref_state_hyd_press_1d(sel)                                             # :178-188
    return params.replace(static={"selectP_inEOS_Zc": sel},
                          traced={"surf_pRef": jnp.float64(surf_pRef), "phiRef": _rk("phiRef", 2 * Nr + 1, phiRef),
                                  "pRef4EOS": _rk("pRef4EOS", Nr, pRef4EOS)})


# ---------------------------------------------------------------------------------------------------------------
# model I/O: pickups and the output schedule (host side: DO_WRITE_PICKUP, WRITE_PICKUP, MDS_WRITE_FIELD)

@dataclass(frozen=True)
class IOParams:
    """PARAMS.h values of the run's output that the driver decides on the host (ini_parms_io)."""
    pChkPtFreq: float = 0.0
    chkPtFreq: float = 0.0
    writePickupAtEnd: bool = True
    pickup_write_mdsio: bool = True
    globalFiles: bool = False
    useSingleCpuIO: bool = False
    dumpFreq: float = 0.0
    mdsioLocalDir: str = " "
    the_run_name: str = " "
    # GO lane (M1 acceptance): MNC_PARAMS.h pickup_write_mnc, set by the driver from MNC_READPARMS (data.mnc, else
    # mnc_readparms.F:97 .FALSE.) when useMNC; WRITE_PICKUP's MNC branch (write_pickup.F:389) tests it
    pickup_write_mnc: bool = False


def ini_parms_io(exp):
    """pChkPtFreq, chkPtFreq (set_defaults.F:339-340 `deltaT*0`: an expression, ported as code), writePickupAtEnd,
    pickup_write_mdsio, globalFiles, useSingleCpuIO, dumpFreq (:346 `deltaT*0`), mdsioLocalDir, the_run_name, each
    from the run's `data` or its cited SET_DEFAULTS line."""
    rp = RunParams(exp.run)
    tp = ini_parms_time(exp)
    g = lambda grp, key, line: _get(exp, rp, grp, key, f"{SD}:{line}")         # noqa: E731
    expr0 = lambda grp, key: (rp.get("data", grp, key) if rp.has("data", grp, key)   # noqa: E731
                              else tp.deltaT * 0)
    return IOParams(
        pChkPtFreq=float(expr0("PARM03", "pChkPtFreq")),                       # set_defaults.F:339
        chkPtFreq=float(expr0("PARM03", "chkPtFreq")),                         # set_defaults.F:340
        writePickupAtEnd=g("PARM03", "writePickupAtEnd", 345),
        pickup_write_mdsio=g("PARM03", "pickup_write_mdsio", 343),
        globalFiles=g("PARM01", "globalFiles", 217),
        useSingleCpuIO=g("PARM01", "useSingleCpuIO", 218),
        dumpFreq=float(expr0("PARM03", "dumpFreq")),                           # set_defaults.F:346
        mdsioLocalDir=g("PARM05", "mdsioLocalDir", 396),
        the_run_name=g("PARM05", "the_run_name", 398))


# ---------------------------------------------------------------------------------------------------------------
# all of them

@dataclass(frozen=True)
class ModelParams:
    """The three parameter groups of a run (host-side, static): `grid` (GridParams), `time` (TimeParams), `init`
    (InitParams)."""
    grid: GridParams
    time: TimeParams
    init: InitParams


def ini_parms(exp, exch2=None):
    """GridParams, TimeParams and InitParams of an experiment (see the module docstring)."""
    return ModelParams(grid=ini_parms_grid(exp, exch2), time=ini_parms_time(exp), init=ini_parms_init(exp))


# ---------------------------------------------------------------------------------------------------------------
# dynamics: the PARAMS.h values FORWARD_STEP and its callees read (plan Task 12)

@jax.tree_util.register_pytree_node_class
class Params:
    """PARAMS.h by Fortran name for the time-stepping routines: `params.viscAh`, `params.no_slip_sides`.

    `static` values (LOGICAL, INTEGER, CHARACTER: they select branches at trace time, IF on a namelist value ->
    Python `if`) are pytree aux data; `traced` values (REAL scalars as float64, REAL arrays as FArrays, e.g.
    rhoFacC(Nr)) are the leaves, traced when the pytree is a jit argument, never closed over [L-XLA-4]. The same
    interface as the MOM lane's test stand-in (mitjax/tests/mom_replay.Params), which it replaces."""

    def __init__(self, static, traced):
        object.__setattr__(self, "_s", dict(static))
        object.__setattr__(self, "_t", dict(traced))

    def __getattr__(self, name):
        s, t = object.__getattribute__(self, "_s"), object.__getattribute__(self, "_t")
        if name in t:
            return t[name]
        if name in s:
            v = s[name]
            if v is None:
                raise AttributeError(f"PARAMS.h {name} is not set for this case")
            return v
        raise AttributeError(f"PARAMS.h {name} is not provided (mitjax/model/src/ini_parms.py ini_parms_dyn)")

    def __setattr__(self, name, value):
        raise AttributeError("Params is immutable; use params.replace(static=..., traced=...)")

    def replace(self, static=None, traced=None):
        return Params({**self._s, **(static or {})}, {**self._t, **(traced or {})})

    def static_float(self, name):
        """The concrete host value of a REAL that selects a branch (the COL lane's interface, e.g. hMixCriteria);
        the traced copy of the same value is `params.NAME`."""
        f = object.__getattribute__(self, "_s").get("_floats", ())
        for k, v in f:
            if k == name:
                return v
        raise AttributeError(f"PARAMS.h {name}: no host value (Params static_float)")

    def static_items(self):
        return dict(self._s)

    def traced_items(self):
        return dict(self._t)

    def tree_flatten(self):
        keys = tuple(sorted(self._t))
        return tuple(self._t[k] for k in keys), (keys, tuple(sorted(self._s.items())))

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        keys, static = aux
        return cls(dict(static), dict(zip(keys, leaves)))


def _rk(name, n, values):
    """A not-tiled PARAMS.h REAL array (Nr) or (Nr+1) / (2*Nr) as an FArray, values float64."""
    return FArray(jnp.asarray(np.asarray(values, dtype=np.float64)), name, k=(1, n), tiled=False)


def ini_parms_dyn(exp, gp, tp, ip):
    """The PARAMS.h values of the time step (FORWARD_STEP and callees; the MOM kernels' `params.NAME`), set as
    SET_DEFAULTS / INI_PARMS / SET_PARMS / LOAD_REF_FILES / SET_REF_STATE @63cdc0b set them, each value cited.
    `gp`, `tp`, `ip`: the GridParams, TimeParams, InitParams of the same experiment. Branches of the derivations
    that the M1 variants do not take raise. Returns a `Params`.

    Not here: the cg2d values (cg2d_h.Cg2dParams: implicSurfPress, implicDiv2DFlow, freeSurfFac, ... are copied in
    from `ini_parms_cg2d` so that every routine reads them from one object)."""
    from mitjax.model.src.cg2d_h import ini_parms_cg2d
    cfg = exp.cfg
    rp = RunParams(exp.run)
    Nr = cfg.size.Nr
    g = lambda grp, key, line, src=SD: _get(exp, rp, grp, key, f"{src}:{line}")    # noqa: E731
    cg = ini_parms_cg2d(exp)

    # ---- PARM01 values read as they are (namelist or SET_DEFAULTS line)
    momStepping = g("PARM01", "momStepping", 192)
    momAdvection = g("PARM01", "momAdvection", 187)
    momViscosity = g("PARM01", "momViscosity", 186)
    momForcing = g("PARM01", "momForcing", 188)
    momTidalForcing = g("PARM01", "momTidalForcing", 189)
    useCoriolis = g("PARM01", "useCoriolis", 190)
    momPressureForcing = g("PARM01", "momPressureForcing", 191)
    vectorInvariantMomentum = g("PARM01", "vectorInvariantMomentum", 193)
    tempStepping = g("PARM01", "tempStepping", 194)
    saltStepping = g("PARM01", "saltStepping", 198)
    useNHMTerms = g("PARM01", "useNHMTerms", 203)
    useSmag3D = g("PARM01", "useSmag3D", 205)
    implicitViscosity = g("PARM01", "implicitViscosity", 210)
    selectImplicitDrag = g("PARM01", "selectImplicitDrag", 211)
    momImplVertAdv = g("PARM01", "momImplVertAdv", 212)
    nonHydrostatic = g("PARM01", "nonHydrostatic", 215)
    quasiHydrostatic = g("PARM01", "quasiHydrostatic", 216)
    useCDscheme = g("PARM01", "useCDscheme", 230)
    implicitIntGravWave = g("PARM01", "implicitIntGravWave", 182)
    staggerTimeStep = g("PARM01", "staggerTimeStep", 183)
    applyExchUV_early = g("PARM01", "applyExchUV_early", 184)
    exactConserv = g("PARM01", "exactConserv", 254)
    linFSConserveTr = g("PARM01", "linFSConserveTr", 255)
    selectNHfreeSurf = g("PARM01", "selectNHfreeSurf", 261)
    selectAddFluid = g("PARM01", "selectAddFluid", 262)
    useRealFreshWaterFlux = g("PARM01", "useRealFreshWaterFlux", 263)
    no_slip_sides = g("PARM01", "no_slip_sides", 132)
    no_slip_bottom = g("PARM01", "no_slip_bottom", 133)
    bottomVisc_pCell = g("PARM01", "bottomVisc_pCell", 134)
    sideDragFactor = g("PARM01", "sideDragFactor", 135)
    bottomDragLinear = g("PARM01", "bottomDragLinear", 136)
    bottomDragQuadratic = g("PARM01", "bottomDragQuadratic", 137)
    zRoughBot = g("PARM01", "zRoughBot", 138)
    selectBotDragQuadr = g("PARM01", "selectBotDragQuadr", 139)
    viscAh = g("PARM01", "viscAh", 118)
    viscAhGrid = g("PARM01", "viscAhGrid", 121)
    viscAhMax = g("PARM01", "viscAhMax", 124)
    viscA4 = g("PARM01", "viscA4", 140)
    viscA4Grid = g("PARM01", "viscA4Grid", 141)
    viscA4GridMax = g("PARM01", "viscA4GridMax", 142)
    viscA4GridMin = g("PARM01", "viscA4GridMin", 143)
    viscA4Max = g("PARM01", "viscA4Max", 144)
    visc_smag = {n: g("PARM01", n, line) for n, line in (
        ("viscC2leith", 126), ("viscC2leithD", 127), ("viscC2LeithQG", 128), ("viscC2smag", 129),
        ("viscC4leith", 146), ("viscC4leithD", 147), ("viscC4smag", 148))}
    visc_files = {n: g("PARM05", n, line) for n, line in (
        ("viscAhDfile", 371), ("viscAhZfile", 372), ("viscA4Dfile", 373), ("viscA4Zfile", 374))}
    gravity = g("PARM01", "gravity", 101)
    rhoNil = g("PARM01", "rhoNil", 104)
    HeatCapacity_Cp = g("PARM01", "HeatCapacity_Cp", 174)
    eosType = g("PARM01", "eosType", 175)
    sIceLoadFac = g("PARM01", "sIceLoadFac", 179)
    ivdc_kappa = g("PARM01", "ivdc_kappa", 221)
    integr_GeoPot = g("PARM01", "integr_GeoPot", 282)
    # ini_parms.F locals with their pre-READ defaults (ini_parms.F:342-346), then the namelist
    useJamartWetPoints = g("PARM01", "useJamartWetPoints", 342, IP)
    useEnergyConservingCoriolis = g("PARM01", "useEnergyConservingCoriolis", 343, IP)
    use3dCoriolis = g("PARM01", "use3dCoriolis", 345, IP)
    metricTerms = g("PARM01", "metricTerms", 346, IP)
    # PARM03
    abEps = g("PARM03", "abEps", 317)
    momDissip_In_AB = g("PARM03", "momDissip_In_AB", 315)
    forcing_In_AB = g("PARM03", "forcing_In_AB", 969, IP)
    cAdjFreq = g("PARM03", "cAdjFreq", 327)
    monitorFreq = g("PARM03", "monitorFreq", 351)
    # alph_AB, beta_AB (ALLOW_ADAMSBASHFORTH_3): read by ini_parms_tracer (ADVECT lane, plan Task 14) for the tracer
    # AB3 of ADAMS_BASHFORTH3; the momentum AB3 (TIMESTEP with ALLOW_ADAMSBASHFORTH_3) is refused below

    # ---- unported values a variant could set: refuse
    # rhoConstFresh (PARM01) is read only by the forcing routines, through ini_parms_forcing (ini_parms.F:484 there)
    for grp, key in (("PARM01", "selectSigmaCoord"), ("PARM05", "gravityFile"),
                     ("PARM01", "viscAp"), ("PARM01", "viscArNr"), ("PARM01", "viscAhD"),
                     ("PARM01", "viscAhZ"), ("PARM01", "viscA4D"), ("PARM01", "viscA4Z")):
        if rp.has("data", grp, key):
            raise NotImplementedError(f"INI_PARMS: {key} in data is not ported (ini_parms_dyn)")
    # ALLOW_ADAMSBASHFORTH_3 for the momentum: TIMESTEP's ADAMS_BASHFORTH3 of gU, gV (GO lane, cs32x15/input_ad);
    # alph_AB, beta_AB from ini_parms_tracer, mom_StartAB below and CHECK_PICKUP (read_pickup.check_pickup)

    # ---- INI_PARMS derivations
    gBaro = _get_or_unset(rp, "PARM01", "gBaro", UNSET_RL)                       # set_defaults.F:102
    if gBaro == UNSET_RL:
        gBaro = gravity                                                         # ini_parms.F:482
    rhoConst = _get_or_unset(rp, "PARM01", "rhoConst", UNSET_RL)                 # set_defaults.F:105
    if rhoConst == UNSET_RL:
        rhoConst = rhoNil                                                       # ini_parms.F:483
    rhoConstFresh = _get_or_unset(rp, "PARM01", "rhoConstFresh", UNSET_RL)       # set_defaults.F:109  UNSET_RL
    if rhoConstFresh == UNSET_RL:
        rhoConstFresh = rhoConst                                                # ini_parms.F:484
    viscAhD = viscAhZ = viscA4D = viscA4Z = UNSET_RL                            # ini_parms.F:405-408
    if viscAhD == UNSET_RL:
        viscAhD = viscAh                                                        # ini_parms.F:512
    if viscAhZ == UNSET_RL:
        viscAhZ = viscAh                                                        # ini_parms.F:513
    if viscA4D == UNSET_RL:
        viscA4D = viscA4                                                        # ini_parms.F:514
    if viscA4Z == UNSET_RL:
        viscA4Z = viscA4                                                        # ini_parms.F:515
    viscAr = _get_or_unset(rp, "PARM01", "viscAr", UNSET_RL)                     # ini_parms.F:411
    viscArNr = np.full(Nr, UNSET_RL)                                            # set_defaults.F:150
    # PTRACERS lane (tutorial_advection_in_gyre sets viscAz): ini_parms.F:409 viscAz = UNSET_RL before the READ,
    # :523 IF ( viscAr .EQ. UNSET_RL ) viscAr = viscAz (viscAp, :524, stays refused above)
    viscAz = _get_or_unset(rp, "PARM01", "viscAz", UNSET_RL)
    if viscAr == UNSET_RL:
        viscAr = viscAz
    vertSetCount = 0                                                            # ini_parms.F:525-528
    if viscAr == UNSET_RL:                                                      # ini_parms.F:535-536
        viscAr = fortran_default(f"{SD}:130", "viscArDefault", exp).value
    if vertSetCount == 0:                                                       # ini_parms.F:543-547
        viscArNr[:] = viscAr
    if cfg.cpp.ALLOW_MOM_COMMON:                                                # ini_parms.F:549-554
        if selectBotDragQuadr == -1 and bottomDragQuadratic != 0.:
            selectBotDragQuadr = 0
        if selectBotDragQuadr == -1 and zRoughBot != 0.:
            selectBotDragQuadr = 0
    selectCoriScheme = _get_or_unset(rp, "PARM01", "selectCoriScheme", UNSET_I)  # set_defaults.F:232
    if selectCoriScheme == UNSET_I:                                             # ini_parms.F:661-667
        selectCoriScheme = 0
        if useJamartWetPoints:
            selectCoriScheme = 1
        if useEnergyConservingCoriolis and not vectorInvariantMomentum:
            selectCoriScheme = selectCoriScheme + 2
    select3dCoriScheme = _get_or_unset(rp, "PARM01", "select3dCoriScheme", UNSET_I)  # set_defaults.F:231
    if select3dCoriScheme == UNSET_I:                                           # ini_parms.F:705-708
        select3dCoriScheme = 0
        if use3dCoriolis:
            select3dCoriScheme = 1
    elif select3dCoriScheme != 0 and not use3dCoriolis:
        raise ValueError("S/R INI_PARMS: select3dCoriScheme /= 0 with use3dCoriolis=F")
    selectMetricTerms = _get_or_unset(rp, "PARM01", "selectMetricTerms", UNSET_I)    # set_defaults.F:204
    if selectMetricTerms == UNSET_I:                                            # ini_parms.F:716-719
        selectMetricTerms = 0
        if metricTerms:
            selectMetricTerms = 1
    elif selectMetricTerms != 0 and not metricTerms:
        raise ValueError("S/R INI_PARMS: selectMetricTerms /= 0 with metricTerms=F")
    recip_rhoConst = 1.0 / rhoConst                                             # ini_parms.F:763  1. _d 0 / rhoConst
    recip_gravity = 1.0 / gravity                                               # ini_parms.F:784
    momForcingOutAB = _get_or_unset(rp, "PARM03", "momForcingOutAB", UNSET_I)    # set_defaults.F:313
    if momForcingOutAB == UNSET_I:                                              # ini_parms.F:1095-1098
        momForcingOutAB = 1
        if forcing_In_AB:
            momForcingOutAB = 0
    monitorSelect = _get_or_unset(rp, "PARM03", "monitorSelect", UNSET_I)        # set_defaults.F:353
    if monitorSelect == UNSET_I:                                                # ini_parms.F:1198-1201
        monitorSelect = 2
        if gp.fluidIsWater:
            monitorSelect = 3
    if gp.usingPCoords:                                                         # ini_parms.F:1569-1575
        mass2rUnit = gravity                                                    # :1570 (lane B, Task 25)
        rUnit2mass = recip_gravity                                              # :1571
    else:
        mass2rUnit = recip_rhoConst                                             # :1573
        rUnit2mass = rhoConst                                                   # :1574
    # diagFreq = deltaT*0 (set_defaults.F:348: an expression, ported as code): 0. unless set
    diagFreq = rp.get("data", "PARM03", "diagFreq") if rp.has("data", "PARM03", "diagFreq") else tp.deltaT * 0
    # ADVECT lane (plan Task 14): ini_parms.F:1187-1197, monitorFreq unset (set_defaults.F:351 `-1.`) -> from the
    # output frequencies (advect_xy: dumpFreq; advect_xz sets it); `0.` REAL*4 exact
    if monitorFreq < 0.:                                                        # :1187
        io = ini_parms_io(exp)
        monitorFreq = 0.                                                        # :1188
        if io.dumpFreq != 0.:                                                   # :1189
            monitorFreq = io.dumpFreq
        if diagFreq != 0. and diagFreq < monitorFreq:                           # :1190-1191
            monitorFreq = diagFreq
        if io.chkPtFreq != 0. and io.chkPtFreq < monitorFreq:                   # :1192-1193
            monitorFreq = io.chkPtFreq
        if io.pChkPtFreq != 0. and io.pChkPtFreq < monitorFreq:                 # :1194-1195
            monitorFreq = io.pChkPtFreq
        if monitorFreq == 0.:                                                   # :1196
            monitorFreq = tp.deltaTClock
    # PTRACERS lane (tutorial_tracer_adjsens: cAdjFreq = -1.): ini_parms.F:1104-1106 IF ( cAdjFreq .LT. 0. ) cAdjFreq =
    # deltaTClock (the static host value TRACERS_CORRECTION_STEP tests against 0.; CONVECTIVE_ADJUSTMENT reads it)
    if cAdjFreq < 0.:
        cAdjFreq = tp.deltaTClock

    # ---- SET_PARMS derivations (set_parms.F @63cdc0b)
    if gp.usingCartesianGrid:                                                   # :55-58
        selectMetricTerms = 0
        useNHMTerms = False
    if gp.usingCylindricalGrid:                                                 # :59-63
        raise NotImplementedError("SET_PARMS: usingCylindricalGrid is not ported")
    if gp.usingCurvilinearGrid:                                                 # :65-67
        selectMetricTerms = 0
    if not (nonHydrostatic or quasiHydrostatic):                                # :82-83
        select3dCoriScheme = 0
    if gp.selectCoriMap in (0, 1) and gp.fPrime == 0.:                          # :84-85
        select3dCoriScheme = 0
    nonHydrostatic = momStepping and nonHydrostatic                             # :88
    quasiHydrostatic = momStepping and quasiHydrostatic                         # :89
    momAdvection = momStepping and momAdvection                                 # :90
    momViscosity = momStepping and momViscosity                                 # :91
    momForcing = momStepping and momForcing                                     # :92
    momTidalForcing = momForcing and momTidalForcing                            # :93
    useCoriolis = momStepping and useCoriolis                                   # :94
    if not useCoriolis:                                                         # :95
        select3dCoriScheme = 0
    useCDscheme = momStepping and useCDscheme                                   # :96
    momPressureForcing = momStepping and momPressureForcing                     # :97
    implicitIntGravWave = momPressureForcing and implicitIntGravWave            # :98
    momImplVertAdv = momAdvection and momImplVertAdv                            # :99
    useNHMTerms = momAdvection and useNHMTerms                                  # :100
    if not momAdvection:                                                        # :101
        selectMetricTerms = 0
    implicitViscosity = momViscosity and implicitViscosity                      # :102
    useSmag3D = momViscosity and useSmag3D                                      # :103
    use3Dsolver = nonHydrostatic or implicitIntGravWave                         # :104
    calc_wVelocity = momStepping or exactConserv                                # :105
    useVariableVisc = useHarmonicVisc = useBiharmonicVisc = False
    if cfg.cpp.ALLOW_MOM_COMMON:                                                # :131-157
        useVariableVisc = (viscAhGrid != 0. or viscA4Grid != 0.
                           or visc_smag["viscC2smag"] != 0. or visc_smag["viscC4smag"] != 0.
                           or visc_smag["viscC2leith"] != 0. or visc_smag["viscC2leithD"] != 0.
                           or visc_smag["viscC2LeithQG"] != 0.
                           or visc_smag["viscC4leith"] != 0. or visc_smag["viscC4leithD"] != 0.
                           or not _blank(visc_files["viscAhDfile"]) or not _blank(visc_files["viscAhZfile"])
                           or not _blank(visc_files["viscA4Dfile"]) or not _blank(visc_files["viscA4Zfile"]))
        useHarmonicVisc = (viscAh != 0. or viscAhD != 0. or viscAhZ != 0. or viscAhGrid != 0.
                           or visc_smag["viscC2smag"] != 0. or visc_smag["viscC2leith"] != 0.
                           or visc_smag["viscC2leithD"] != 0. or visc_smag["viscC2LeithQG"] != 0.
                           or not _blank(visc_files["viscAhDfile"]) or not _blank(visc_files["viscAhZfile"]))
        useBiharmonicVisc = (viscA4 != 0. or viscA4D != 0. or viscA4Z != 0. or viscA4Grid != 0.
                             or visc_smag["viscC4smag"] != 0. or visc_smag["viscC4leith"] != 0.
                             or visc_smag["viscC4leithD"] != 0.
                             or not _blank(visc_files["viscA4Dfile"]) or not _blank(visc_files["viscA4Zfile"]))
        useVariableVisc = momViscosity and useVariableVisc
        useHarmonicVisc = momViscosity and useHarmonicVisc
        useBiharmonicVisc = momViscosity and useBiharmonicVisc
    if (bottomDragQuadratic == 0. and zRoughBot == 0.) or not momViscosity:    # :158-159
        selectBotDragQuadr = -1
    useShelfIce = _use(cfg, "useShelfIce")
    topoFile = g("PARM05", "topoFile", 361)
    # GO lane: CALC_GRAD_PHI_HYD's generalForm (calc_grad_phi_hyd.F:160-162), static: useShelfIce .OR. (usingPCoords
    # .AND. rF(Nr+1).NE.0) .OR. (usingZCoords .AND. (topoFile.NE.' ' .OR. rF(1).NE.0)); rF(1) as INI_VERTICAL_GRID
    # sets it (ini_vertical_grid.F:136-139: seaLev_Z when INI_PARMS left it UNSET_RS, z coordinates, water)
    rF1 = gp.rF1
    if rF1 == UNSET_RS and gp.usingZCoords and gp.fluidIsWater:
        rF1 = gp.seaLev_Z
    gradPhiHyd_generalForm = bool(useShelfIce or (gp.usingZCoords and (not _blank(topoFile) or rF1 != 0.)))
    uniformFreeSurfLev = gp.usingZCoords                                        # :162
    uniformFreeSurfLev = gp.usingZCoords and not useShelfIce and _blank(topoFile)   # :165-166
    # momentum on/off factors (set_parms.F:198-237): 1. _d 0 / 0. _d 0
    vfFacMom = 1.0 if momViscosity else 0.0
    afFacMom = 1.0 if momAdvection else 0.0
    foFacMom = 1.0 if momForcing else 0.0
    cfFacMom = 1.0 if useCoriolis else 0.0
    pfFacMom = 1.0 if momPressureForcing else 0.0
    mtFacMom = 1.0 if selectMetricTerms >= 1 else 0.0

    # ---- CD scheme coefficients (GO lane): PARM03 tauCD (set_defaults.F:328 `0. _d 0`), rCD, epsAB_CD (ini_parms.F
    # locals preset at :965-966: rCD = -1. _d 0, epsAB_CD = UNSET_RL), then ini_parms.F:1114-1119 when the namelist
    # useCDscheme is set (INI_PARMS tests the value as read, before SET_PARMS' momStepping .AND. at set_parms.F:96)
    tauCD = g("PARM03", "tauCD", 328)
    rCD = _get_or_unset(rp, "PARM03", "rCD", -1.0)                               # ini_parms.F:965
    epsAB_CD = _get_or_unset(rp, "PARM03", "epsAB_CD", UNSET_RL)                 # ini_parms.F:966
    if g("PARM01", "useCDscheme", 230):                                          # :1114
        if tauCD == 0.:
            tauCD = tp.deltaTMom                                                # :1116
        if rCD < 0.:
            rCD = 1.0 - tp.deltaTMom/tauCD                                      # :1117  1. _d 0 - deltaTMom/tauCD
        if epsAB_CD == UNSET_RL:
            epsAB_CD = abEps                                                    # :1118

    # ---- INI_MODEL_IO (ini_model_io.F:125-133): AB starting level; startFromPickupAB2 (set_defaults.F) raises
    if rp.has("data", "PARM03", "startFromPickupAB2"):
        raise NotImplementedError("INI_MODEL_IO: startFromPickupAB2 is not ported")
    tempStartAB = tp.nIter0                                                     # :125
    mom_StartAB = tempStartAB                                                   # :127, :132

    # ---- LOAD_REF_FILES (gravityFile blank, load_ref_files.F:156-164) and SET_REF_STATE (set_ref_state.F:51-80)
    gravFacC = np.ones(Nr)                                                      # load_ref_files.F:158  1. _d 0
    recip_gravFacC = np.ones(Nr)                                                # :159
    gravFacF = np.ones(Nr + 1)                                                  # :162
    recip_gravFacF = np.ones(Nr + 1)                                            # :163
    if gp.buoyancyRelation not in ("OCEANIC", "ATMOSPHERIC"):
        raise NotImplementedError("SET_REF_STATE: only buoyancyRelation='OCEANIC' / 'ATMOSPHERIC' is ported")
    # lane B: the anelastic factors (set_ref_state.F:358-386) depend on `usingZCoords .AND. rhoRefFile .NE. ' '`
    # (LOAD_REF_FILES' rho1Ref, load_ref_files.F:107-115), not on a CPP option (an ALLOW_NONHYDROSTATIC build with
    # rhoRefFile unset, global_ocean.cs32x15, keeps rhoFac = 1)
    if gp.usingZCoords and not _blank(_get(exp, rp, "PARM01", "rhoRefFile", f"{SD}:59")):
        raise NotImplementedError("SET_REF_STATE: the anelastic rhoFac from rhoRefFile (:358-386) is not ported")
    rUnit2z = np.ones(Nr)                                                       # set_ref_state.F:57  1. _d 0
    z2rUnit = np.ones(Nr)                                                       # set_ref_state.F:58  1. _d 0
    rVel2wUnit = np.ones(Nr + 1)                                                # set_ref_state.F:68  1. _d 0
    wUnit2rVel = np.ones(Nr + 1)                                                # :69
    atm_traced = {}
    if gp.buoyancyRelation == "ATMOSPHERIC":                                    # lane B (Task 25)
        rVel2wUnit, wUnit2rVel = _set_ref_state_atm_wunit(exp, cfg, gp, ip, gravity, rVel2wUnit, wUnit2rVel)
        atm_traced = _atm_traced(exp, cfg, gp, ip)
    rhoFacC = np.ones(Nr)                                                       # :74
    recip_rhoFacC = np.ones(Nr)                                                 # :75
    rhoFacF = np.ones(Nr + 1)                                                   # :78
    recip_rhoFacF = np.ones(Nr + 1)                                             # :79

    # ---- MOM_INIT_FIXED (pkg/mom_common/mom_init_fixed.F:45-53): COMMON /MOM_GRID_COPY/ deepFacAdv(Nr)
    # ADVECT lane: a build without pkg/mom_common (advect_xz) has no MOM_INIT_FIXED and no deepFacAdv (no Params
    # entry: a read fails); only the momentum code reads it, so such a build is refused only with momStepping
    mom_traced = {}
    if not cfg.cpp.ALLOW_MOM_COMMON:
        if momStepping:
            raise NotImplementedError("MOM_INIT_FIXED: a momentum-stepping build without ALLOW_MOM_COMMON is not "
                                      "ported")
    else:
        deepFacAdv = np.ones(Nr)                                                # :46  1. _d 0
        if not cfg.cpp.flag("MOM_USE_OLD_DEEP_VERT_ADV", "MOM_COMMON_OPTIONS.h") and useNHMTerms:   # :48-53
            # GO lane: deepFacAdv(k) = deepFacC(k) (:50-52); deepFacC is SET_GRID_FACTORS' (set_grid_factors.F:48-53:
            # 1. _d 0 without deepAtmosphere, mitjax/model/src/set_grid_factors.py); the deep-atmosphere values
            # (:60-86) are not built here
            if gp.deepAtmosphere:
                raise NotImplementedError("MOM_INIT_FIXED: deepFacAdv = deepFacC under deepAtmosphere is not ported")
            deepFacC = np.ones(Nr)                                              # set_grid_factors.F:49  1. _d 0
            deepFacAdv = deepFacC.copy()                                        # mom_init_fixed.F:50-52
        mom_traced["deepFacAdv"] = _rk("deepFacAdv", Nr, deepFacAdv)

    # ---- GAD_INIT_FIXED (pkg/generic_advdiff/gad_init_fixed.F:117-123): the SOM advection flags
    tempAdvection = tempStepping and g("PARM01", "tempAdvection", 195)            # set_parms.F:242
    saltAdvection = saltStepping and g("PARM01", "saltAdvection", 199)            # set_parms.F:245
    tempSOM_Advection = saltSOM_Advection = False
    if cfg.cpp.ALLOW_GENERIC_ADVDIFF:
        ENUM_SOM_PRATHER, ENUM_SOM_LIMITER = 80, 81                             # GAD.h:57, :61
        tAdv = g("PARM01", "tempAdvScheme", 226)
        sAdv = g("PARM01", "saltAdvScheme", 227)
        tempSOM_Advection = ENUM_SOM_PRATHER <= tAdv <= ENUM_SOM_LIMITER and tempAdvection   # :117-120
        saltSOM_Advection = ENUM_SOM_PRATHER <= sAdv <= ENUM_SOM_LIMITER and saltAdvection   # :121-123

    useDiagnostics = _use(cfg, "useDiagnostics")
    static = dict(
        tempAdvection=tempAdvection, saltAdvection=saltAdvection, tempSOM_Advection=tempSOM_Advection,
        saltSOM_Advection=saltSOM_Advection, usePTRACERS=_use(cfg, "usePTRACERS"),
        momStepping=momStepping, momAdvection=momAdvection, momViscosity=momViscosity, momForcing=momForcing,
        momTidalForcing=momTidalForcing, useCoriolis=useCoriolis, momPressureForcing=momPressureForcing,
        vectorInvariantMomentum=vectorInvariantMomentum, tempStepping=tempStepping, saltStepping=saltStepping,
        useNHMTerms=useNHMTerms, useSmag3D=useSmag3D, implicitViscosity=implicitViscosity,
        selectImplicitDrag=selectImplicitDrag, momImplVertAdv=momImplVertAdv, nonHydrostatic=nonHydrostatic,
        quasiHydrostatic=quasiHydrostatic, useCDscheme=useCDscheme, implicitIntGravWave=implicitIntGravWave,
        staggerTimeStep=staggerTimeStep, applyExchUV_early=applyExchUV_early, exactConserv=exactConserv,
        linFSConserveTr=linFSConserveTr, selectNHfreeSurf=selectNHfreeSurf, selectAddFluid=selectAddFluid,
        useRealFreshWaterFlux=useRealFreshWaterFlux, no_slip_sides=no_slip_sides, no_slip_bottom=no_slip_bottom,
        bottomVisc_pCell=bottomVisc_pCell, selectBotDragQuadr=selectBotDragQuadr, selectCoriScheme=selectCoriScheme,
        select3dCoriScheme=select3dCoriScheme, selectMetricTerms=selectMetricTerms, momDissip_In_AB=momDissip_In_AB,
        momForcingOutAB=momForcingOutAB, monitorSelect=monitorSelect, use3Dsolver=use3Dsolver,
        calc_wVelocity=calc_wVelocity, useVariableVisc=useVariableVisc, useHarmonicVisc=useHarmonicVisc,
        useBiharmonicVisc=useBiharmonicVisc, uniformFreeSurfLev=uniformFreeSurfLev, eosType=eosType,
        integr_GeoPot=integr_GeoPot, mom_StartAB=mom_StartAB, nIter0=tp.nIter0,
        uniformLin_PhiSurf=_get(exp, rp, "PARM01", "uniformLin_PhiSurf", f"{SD}:256"),   # lane B (Task 25)
        rigidLid=_get(exp, rp, "PARM01", "rigidLid", f"{SD}:251"), select_rStar=ip.select_rStar,
        nonlinFreeSurf=ip.nonlinFreeSurf, usePickupBeforeC54=ip.usePickupBeforeC54, useOffLine=ip.useOffLine,
        storePhiHyd4Phys=ip.storePhiHyd4Phys, allowFreezing=ip.allowFreezing,
        usingCartesianGrid=gp.usingCartesianGrid, usingSphericalPolarGrid=gp.usingSphericalPolarGrid,
        usingCylindricalGrid=gp.usingCylindricalGrid, usingCurvilinearGrid=gp.usingCurvilinearGrid,
        rotateGrid=gp.rotateGrid, deepAtmosphere=gp.deepAtmosphere, usingPCoords=gp.usingPCoords,
        usingZCoords=gp.usingZCoords, fluidIsWater=gp.fluidIsWater,
        fluidIsAir=gp.buoyancyRelation == "ATMOSPHERIC",                       # ini_parms.F:451-453 (lane B)
        buoyancyRelation=gp.buoyancyRelation, useDiagnostics=useDiagnostics, useOBCS=_use(cfg, "useOBCS"),
        useShelfIce=useShelfIce, useGMRedi=_use(cfg, "useGMRedi"), useKPP=_use(cfg, "useKPP"),
        # GO lane: the run-time package switches DO_OCEANIC_PHYS / TEMP_INTEGRATE test (`IF (useX)`), as PACKAGES_BOOT
        # leaves them (cfg.use; .FALSE. for a package the build does not compile: PACKAGES_CHECK stops a run that
        # switches on an uncompiled package, and the `#ifdef ALLOW_X` around each site removes the test)
        **{n: _use(cfg, n) for n in ("usePP81", "useKL10", "useMY82", "useGGL90", "useSALT_PLUME",
                                     "useDOWN_SLOPE", "useBBL")},
        debugLevel=cg.debugLevel, cg2dMaxIters=cg.cg2dMaxIters, cg2dUseMinResSol=cg.cg2dUseMinResSol,
        printResidualFreq=cg.printResidualFreq, useSRCGSolver=cg.useSRCGSolver, useNSACGSolver=cg.useNSACGSolver,
        cg2dFullAdjoint=cg.cg2dFullAdjoint,
        # REAL values that only decide host-side output (monitor / print frequencies), never traced
        monitorFreq=float(monitorFreq), diagFreq=float(diagFreq), cAdjFreq=float(cAdjFreq),
        ivdc_kappa_ne_0=bool(ivdc_kappa != 0.),
        # REAL comparisons that select code, decided on the host from the value above (cf. cg2dTargetResWunit_le_0)
        bottomDragLinear_ne_0=bool(bottomDragLinear != 0.),                    # mom_fluxform.F:273
        selectSigmaCoord_ne_0=bool(gp.selectSigmaCoord != 0),                   # calc_phi_hyd.F:97 useFVgradPhi
        selectSigmaCoord=gp.selectSigmaCoord, gradPhiHyd_generalForm=gradPhiHyd_generalForm,
        implicSurfPress_ne_1=bool(cg.implicSurfPress != 1.),                    # dynamics.F:353 (1. REAL*4, exact)
        implicDiv2DFlow_eq_1=bool(cg.implicDiv2DFlow == 1.),                    # update_etah.F:54, :84 (1. _d 0)
        interViscAr_pCell=g("PARM04", "interViscAr_pCell", 1214, IP),
        pCellMix_select=g("PARM04", "pCellMix_select", 1216, IP),
        _floats=(),
    )
    f64 = np.float64
    traced = dict(
        viscAh=f64(viscAh), viscAhD=f64(viscAhD), viscAhZ=f64(viscAhZ), viscAhGrid=f64(viscAhGrid),
        viscAhMax=f64(viscAhMax), viscA4=f64(viscA4), viscA4D=f64(viscA4D), viscA4Z=f64(viscA4Z),
        viscA4Grid=f64(viscA4Grid), viscA4Max=f64(viscA4Max), viscA4GridMax=f64(viscA4GridMax),
        viscA4GridMin=f64(viscA4GridMin), sideDragFactor=f64(sideDragFactor), bottomDragLinear=f64(bottomDragLinear),
        bottomDragQuadratic=f64(bottomDragQuadratic), afFacMom=f64(afFacMom), vfFacMom=f64(vfFacMom),
        cfFacMom=f64(cfFacMom), mtFacMom=f64(mtFacMom), pfFacMom=f64(pfFacMom), foFacMom=f64(foFacMom),
        abEps=f64(abEps), implicSurfPress=f64(cg.implicSurfPress), implicDiv2DFlow=f64(cg.implicDiv2DFlow),
        freeSurfFac=f64(cg.freeSurfFac), deltaTMom=f64(tp.deltaTMom), deltaTFreeSurf=f64(tp.deltaTFreeSurf),
        deltaTClock=f64(tp.deltaTClock), startTime=f64(tp.startTime), rhoConst=f64(rhoConst), recip_rhoConst=f64(recip_rhoConst),
        rhoNil=f64(rhoNil), gravity=f64(gravity), recip_gravity=f64(recip_gravity), gBaro=f64(gBaro),
        mass2rUnit=f64(mass2rUnit), rUnit2mass=f64(rUnit2mass), recip_rSphere=f64(gp.recip_rSphere),
        HeatCapacity_Cp=f64(HeatCapacity_Cp), sIceLoadFac=f64(sIceLoadFac), rhoConstFresh=f64(rhoConstFresh),
        tauCD=f64(tauCD), rCD=f64(rCD), epsAB_CD=f64(epsAB_CD), deltaTtracer=f64(tp.deltaTtracer),
        viscArNr=_rk("viscArNr", Nr, viscArNr), gravFacC=_rk("gravFacC", Nr, gravFacC),
        recip_gravFacC=_rk("recip_gravFacC", Nr, recip_gravFacC), gravFacF=_rk("gravFacF", Nr + 1, gravFacF),
        recip_gravFacF=_rk("recip_gravFacF", Nr + 1, recip_gravFacF), rhoFacC=_rk("rhoFacC", Nr, rhoFacC),
        recip_rhoFacC=_rk("recip_rhoFacC", Nr, recip_rhoFacC), rhoFacF=_rk("rhoFacF", Nr + 1, rhoFacF),
        recip_rhoFacF=_rk("recip_rhoFacF", Nr + 1, recip_rhoFacF),
        rVel2wUnit=_rk("rVel2wUnit", Nr + 1, rVel2wUnit), wUnit2rVel=_rk("wUnit2rVel", Nr + 1, wUnit2rVel),
        rUnit2z=_rk("rUnit2z", Nr, rUnit2z), z2rUnit=_rk("z2rUnit", Nr, z2rUnit),
        tRef=_rk("tRef", Nr, ip.tRef), sRef=_rk("sRef", Nr, ip.sRef), **mom_traced, **atm_traced,
    )
    prm = Params(static, traced)
    if cfg.cpp.NONLIN_FRSURF:
        # GO lane: the r* values of the RSTAR lane's ini_parms_rstar (hFacInf, hFacSup, selectKEscheme,
        # doResetHFactors), folded in here when the lanes merged
        from mitjax.model.src.ini_parms_rstar import ini_parms_rstar
        prm = ini_parms_rstar(exp, prm)
    if cfg.cpp.ALLOW_MOM_COMMON:                                                # VECINV lane: ini_parms_vecinv
        vs, vt = ini_parms_vecinv(exp)
        prm = prm.replace(static=vs, traced=vt)
    return prm


def _atm_consts(exp, rp):
    """atm_Po, atm_Cp, atm_Rd, atm_kappa as SET_DEFAULTS / INI_PARMS leave them (set_defaults.F:277-280,
    ini_parms.F:496-500; lane B)."""
    g = lambda key, line: _get(exp, rp, "PARM01", key, f"{SD}:{line}")         # noqa: E731
    atm_Po = g("atm_Po", 277)
    atm_Cp = g("atm_Cp", 278)
    atm_Rd = _get_or_unset(rp, "PARM01", "atm_Rd", UNSET_RL)                    # set_defaults.F:279
    atm_kappa = rp.get("data", "PARM01", "atm_kappa") if rp.has("data", "PARM01", "atm_kappa") \
        else 2.0 / 7.0                                                          # set_defaults.F:280  2. _d 0 / 7. _d 0
    if atm_Rd == UNSET_RL:                                                      # ini_parms.F:496-500
        atm_Rd = atm_Cp * atm_kappa
    else:
        atm_kappa = atm_Rd / atm_Cp
    return atm_Po, atm_Cp, atm_Rd, atm_kappa


def _atm_traced(exp, cfg, gp, ip):
    """The PARAMS.h values the fluidIsAir arms of DO_ATMOSPHERIC_PHYS and CALC_PHI_HYD read (lane B, Task 25):
    atm_Rq (set_defaults.F:281 `0. _d 0`), thetaConst (set_defaults.F:61 UNSET_RL, then load_ref_files.F:74
    tRef(1)), atm_Cp; and the Exner factors of CALC_PHI_HYD's ddPIm / ddPIp (calc_phi_hyd.F:503-516, :537-540,
    :569-582), evaluated on the host as the libm calls they are: rF_Po_kappa(k) = (rF(k)/atm_Po)**atm_kappa
    (k = 1..Nr+1), rC_Po_kappa(k) = (rC(k)/atm_Po)**atm_kappa (k = 1..Nr), `**` through math.pow (glibc 2.28's
    pow, the one the oracle binary imports; XLA's pow is not it), rF, rC from INI_VERTICAL_GRID as in
    _set_ref_state_atm_wunit. CALC_PHI_HYD keeps the arithmetic around them (atm_Cp*(a - b)*halfRL) in the
    Fortran order."""
    import math
    from mitjax.model.grid import Grid
    from mitjax.model.src.ini_vertical_grid import ini_vertical_grid
    rp = RunParams(exp.run)
    atm_Po, atm_Cp, atm_Rd, atm_kappa = _atm_consts(exp, rp)
    atm_Rq = _get(exp, rp, "PARM01", "atm_Rq", f"{SD}:281")
    thetaConst = _get_or_unset(rp, "PARM01", "thetaConst", UNSET_RL)            # set_defaults.F:61
    if thetaConst == UNSET_RL:                                                  # load_ref_files.F:74
        thetaConst = float(np.asarray(ip.tRef).ravel()[0])
    Nr = cfg.size.Nr
    vg = ini_vertical_grid(Grid(), cfg=cfg, params=gp)
    rF = [float(x) for x in np.asarray(vg.rF.data).ravel()]                     # rF(1..Nr+1)
    rC = [float(x) for x in np.asarray(vg.rC.data).ravel()]                     # rC(1..Nr)
    f64 = np.float64
    return dict(atm_Rq=f64(atm_Rq), thetaConst=f64(thetaConst), atm_Cp=f64(atm_Cp),
                rF_Po_kappa=_rk("rF_Po_kappa", Nr + 1, [math.pow(x / atm_Po, atm_kappa) for x in rF]),
                rC_Po_kappa=_rk("rC_Po_kappa", Nr, [math.pow(x / atm_Po, atm_kappa) for x in rC]))


def _set_ref_state_atm_wunit(exp, cfg, gp, ip, gravity, rVel2wUnit, wUnit2rVel):
    """The rVel2wUnit / wUnit2rVel part of SET_REF_STATE's ATMOSPHERIC arm (set_ref_state.F:288-303; lane B, Task 25)
    on the host: atm_Po, atm_Cp, atm_kappa, atm_Rd as SET_DEFAULTS / INI_PARMS leave them (set_defaults.F:277-280,
    ini_parms.F:496-500), rF from INI_VERTICAL_GRID (mitjax/model/src/ini_vertical_grid.py, the same statements),
    `**` through math.pow (the C library's pow: glibc 2.28 libm, the library the oracle binary links). The other
    reference profiles of the arm (dBdrRef, z2rUnit, phiRef: :256-287, :304-337) are not carried by Params."""
    import math
    from mitjax.model.grid import Grid
    from mitjax.model.src.ini_vertical_grid import ini_vertical_grid
    rp = RunParams(exp.run)
    atm_Po, atm_Cp, atm_Rd, atm_kappa = _atm_consts(exp, rp)
    Nr = cfg.size.Nr
    vg = ini_vertical_grid(Grid(), cfg=cfg, params=gp)
    rF = [None] + [float(x) for x in np.asarray(vg.rF.data).ravel()]            # rF(1..Nr+1)
    tRef = [None] + [float(x) for x in np.asarray(ip.tRef).ravel()]             # tRef(1..Nr)
    rVel2wUnit, wUnit2rVel = np.array(rVel2wUnit), np.array(wUnit2rVel)
    for k in range(1, Nr + 2):                                                  # :288-303
        if k == 1:
            thetaLoc = tRef[k]
        elif k > Nr:
            thetaLoc = tRef[k - 1]
        else:
            thetaLoc = (tRef[k] + tRef[k - 1]) * 0.5                            # 0.5 _d 0
        if thetaLoc > 0.0 and rF[k] > 0.0:
            conv_theta2T = math.pow(rF[k] / atm_Po, atm_kappa)                  # (rF(k)/atm_Po)**atm_kappa
            wUnit2rVel[k - 1] = gravity * rF[k] / (atm_Rd * conv_theta2T * thetaLoc)
            rVel2wUnit[k - 1] = 1.0 / wUnit2rVel[k - 1]                         # 1. _d 0 / wUnit2rVel(k)
    return rVel2wUnit, wUnit2rVel


# ---------------------------------------------------------------------------------------------------------------
# VECINV lane (plan Task 23): the PARAMS.h / EEPARAMS.h values MOM_VECINV, MOM_CALC_VISC and MOM_INIT_FIXED read
# beyond ini_parms_dyn; gated against the replay harness's record of the run's PARAMS.h (mitjax/tests/test_vecinv.py)

def ini_parms_vecinv(exp):
    """(static, traced) for pkg/mom_vecinv and pkg/mom_common: the vorticity/KE/shear selectors, the Leith and
    Smagorinsky switches, the static host flags `<coefficient>_ne_0` of the REAL tests that select code in
    MOM_CALC_VISC (:183-193; a gradient or a perturbation of those coefficients must not cross 0), the traced bounds
    and coefficients, viscAhW/viscA4W (MOM_INIT_FIXED, ALLOW_NONHYDROSTATIC) and useCubedSphereExchange (eedata).
    Reads only the run's files (standalone: the cube experiments' ini_parms_dyn still raises in SET_REF_STATE)."""
    rp = RunParams(exp.run)
    g = lambda grp, key, line, src=SD: _get(exp, rp, grp, key, f"{src}:{line}")    # noqa: E731
    f64 = np.float64
    upwindVorticity = g("PARM01", "upwindVorticity", 235)
    highOrderVorticity = g("PARM01", "highOrderVorticity", 236)
    selectVortScheme = _get_or_unset(rp, "PARM01", "selectVortScheme", UNSET_I)  # set_defaults.F:233
    if g("PARM01", "SadournyCoriolis", 344, IP):                                # ini_parms.F:694-703
        if selectVortScheme == UNSET_I:
            selectVortScheme = 2
        if selectVortScheme != 2:
            raise ValueError(f"S/R INI_PARMS: selectVortScheme={selectVortScheme:5d} conflicts with \"SadournyCoriolis\"")
    if g("PARM01", "vectorInvariantMomentum", 193) and selectVortScheme == UNSET_I:   # set_parms.F:184-189
        selectVortScheme = 1
        if upwindVorticity:
            selectVortScheme = 0
        if highOrderVorticity:
            selectVortScheme = 0
    coef = {n: g("PARM01", n, line) for n, line in (
        ("viscC2leith", 126), ("viscC2leithD", 127), ("viscC2LeithQG", 128), ("viscC2smag", 129),
        ("viscC4leith", 146), ("viscC4leithD", 147), ("viscC4smag", 148))}
    viscAhD = _get_or_unset(rp, "PARM01", "viscAhD", UNSET_RL)                   # ini_parms.F:405
    viscA4D = _get_or_unset(rp, "PARM01", "viscA4D", UNSET_RL)                   # ini_parms.F:407
    if viscAhD == UNSET_RL:
        viscAhD = g("PARM01", "viscAh", 118)                                     # ini_parms.F:512
    if viscA4D == UNSET_RL:
        viscA4D = g("PARM01", "viscA4", 140)                                     # ini_parms.F:514
    viscAhW = _get_or_unset(rp, "PARM01", "viscAhW", UNSET_RL)                   # ini_parms.F:403
    viscA4W = _get_or_unset(rp, "PARM01", "viscA4W", UNSET_RL)                   # ini_parms.F:404
    if viscAhW == UNSET_RL:
        viscAhW = viscAhD                                                       # ini_parms.F:517
    if viscA4W == UNSET_RL:
        viscA4W = viscA4D                                                       # ini_parms.F:518
    static = dict(
        selectVortScheme=selectVortScheme, upwindVorticity=upwindVorticity, highOrderVorticity=highOrderVorticity,
        useJamartMomAdv=g("PARM01", "useJamartMomAdv", 234), useAbsVorticity=g("PARM01", "useAbsVorticity", 237),
        upwindShear=g("PARM01", "upwindShear", 238), selectKEscheme=int(g("PARM01", "selectKEscheme", 239)),
        useFullLeith=g("PARM01", "useFullLeith", 206), useAreaViscLength=g("PARM01", "useAreaViscLength", 207),
        useStrainTensionVisc=g("PARM01", "useStrainTensionVisc", 208),
        useCubedSphereExchange=rp.get("eedata", "EEPARMS", "useCubedSphereExchange", default=fortran_default(
            "eesupp/src/eeset_parms.F:106", "useCubedSphereExchange", exp)),
        **{f"{n}_ne_0": bool(v != 0.) for n, v in coef.items()})
    traced = dict(viscAhGridMax=f64(g("PARM01", "viscAhGridMax", 123)),
                  viscAhGridMin=f64(g("PARM01", "viscAhGridMin", 122)),
                  viscAhReMax=f64(g("PARM01", "viscAhReMax", 125)), viscA4ReMax=f64(g("PARM01", "viscA4ReMax", 145)),
                  viscAhW=f64(viscAhW), viscA4W=f64(viscA4W), **{n: f64(v) for n, v in coef.items()})
    return static, traced
