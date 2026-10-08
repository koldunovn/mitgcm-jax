"""The model state: the time-varying common-block variables of DYNVARS.h and SURFACE.h, as Fortran-index arrays
(plan Task 11).

`State` holds every DYNVARS.h / SURFACE.h variable that THIS build declares (CPP options of `cfg`) and that the ported
model reads or writes, each an `FArray` with its Fortran name and declared bounds (`DECLARATIONS`: header, line,
shape kind, type, CPP condition). Storage `[tile, k, j, i]` as GRID.h (mitjax/model/grid.py). A pytree (the arrays
are the leaves, the field names static); immutable: a routine returns `state.replace(uVel=..., ...)`.

The Adams-Bashforth-3 histories `guNm(..., 2)` (DYNVARS.h:57-60, ALLOW_ADAMSBASHFORTH_3) carry a sixth dimension
(the two previous time levels); a field of kind "xyz2" is a tuple of two FArrays (Fortran index m = 1, 2 at Python
positions 0, 1), named `guNm_1`, `guNm_2` as the jaxdump records name them.

Not carried (no dead fields [F§9]; each would be added with the routine that reads it):
  * NH_VARS.h QHydGwNm (lane B carries phi_nh, dPhiNH, gW, gwNm1 of ALLOW_NONHYDROSTATIC builds, cs32x15; and
    DYNVARS.h dU_psFacX/dV_psFacY of ALLOW_SOLVE4_PS_AND_DRAG): QHydGwNm (ALLOW_QHYD_STAGGER_TS,
    global_ocean.90x40x15) is read only with quasiHydrostatic .AND. staggerTimeStep, which CONFIG_CHECK forbids
    (config_check.F:178-182);
  * SURFACE.h /SIGMA_CHANGE/ (etaHw, etaHs, dEtaWdt, dEtaSdt): read only under selectSigmaCoord != 0 (no M1 variant;
    INI_PARMS raises for it); /SURF_FIXED/ Bo_surf, recip_Bo, phi0surf and /SURF_CORREC/: fixed fields and
    diagnostics of INI_LINEAR_PHISURF / the tracer correction (not state; with their routines);
  * DYNVARS.h fields under options no ported build sets (ALLOW_SMAG_3D_DIFFUSIVITY,
    ALLOW_BL79_LAT_VARY, ALLOW_LEITH_QG, INCLUDE_SOUNDSPEED_CALC_CODE): `fields_of(cfg)`
    raises if a build sets one (unported option);
  * FFIELDS.h (forcing) and the other package states (GM/Redi, ...): with their routines (Task 12 INI_FFIELDS /
    INI_FORCING and the forcing driver). CD_CODE_VARS.h (ALLOW_CD_CODE) is carried: INI_PSURF writes etaNm1; the
    other four are written by CD_CODE_INI_VARS (Task 15a) and stay NaN until it is ported.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.model.grid import bounds, n_tiles

# name -> (header, line, shape kind, type, CPP condition or None). Shape kinds as GRID.h (grid.bounds): xy, xyz;
# "xyz2": (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,2). Conditions: "AB3" = ALLOW_ADAMSBASHFORTH_3, "AB2" = its
# absence, other names = the CPP macro.
DECLARATIONS = {
    # DYNVARS.h COMMON /DYNVARS_R/ (DYNVARS.h:35-66)
    "etaN": ("DYNVARS.h", 48, "xy", "RL", None),
    "uVel": ("DYNVARS.h", 49, "xyz", "RL", None), "vVel": ("DYNVARS.h", 50, "xyz", "RL", None),
    "wVel": ("DYNVARS.h", 51, "xyz", "RL", None), "theta": ("DYNVARS.h", 52, "xyz", "RL", None),
    "salt": ("DYNVARS.h", 53, "xyz", "RL", None),
    "gU": ("DYNVARS.h", 54, "xyz", "RL", None), "gV": ("DYNVARS.h", 55, "xyz", "RL", None),
    "guNm": ("DYNVARS.h", 57, "xyz2", "RL", "AB3"), "gvNm": ("DYNVARS.h", 58, "xyz2", "RL", "AB3"),
    "gtNm": ("DYNVARS.h", 59, "xyz2", "RL", "AB3"), "gsNm": ("DYNVARS.h", 60, "xyz2", "RL", "AB3"),
    "guNm1": ("DYNVARS.h", 62, "xyz", "RL", "AB2"), "gvNm1": ("DYNVARS.h", 63, "xyz", "RL", "AB2"),
    "gtNm1": ("DYNVARS.h", 64, "xyz", "RL", "AB2"), "gsNm1": ("DYNVARS.h", 65, "xyz", "RL", "AB2"),
    # lane B (Task 25, adjustment.cs): DYNVARS.h COMMON /DYNVARS_OLD/ (:68-73) under USE_OLD_EXTERNAL_FORCING
    "gT": ("DYNVARS.h", 71, "xyz", "RL", "USE_OLD_EXTERNAL_FORCING"),
    "gS": ("DYNVARS.h", 72, "xyz", "RL", "USE_OLD_EXTERNAL_FORCING"),
    # DYNVARS.h COMMON /DYNVARS_R_2/ (:75-77)
    "etaH": ("DYNVARS.h", 77, "xy", "RL", None),
    # DYNVARS.h COMMON /DYNVARS_DIFFKR/ (:79-85)
    "diffKr": ("DYNVARS.h", 84, "xyz", "RL", "ALLOW_3D_DIFFKR|ALLOW_DIFFKR_CONTROL"),
    # DYNVARS.h COMMON /DYNVARS_DIAG/ (:117-125)
    "rhoInSitu": ("DYNVARS.h", 121, "xyz", "RL", None), "totPhiHyd": ("DYNVARS.h", 122, "xyz", "RL", None),
    "phiHydLow": ("DYNVARS.h", 123, "xy", "RL", None), "hMixLayer": ("DYNVARS.h", 124, "xy", "RL", None),
    "IVDConvCount": ("DYNVARS.h", 125, "xyz", "RL", None),
    # M3 lane MLAdjust: DYNVARS.h /DYNVARS_sigmaR/ sigmaRfield (:127-134, #ifdef ALLOW_LEITH_QG; DO_OCEANIC_PHYS writes
    # it, MOM_VISC_QGL_STRETCH reads it)
    "sigmaRfield": ("DYNVARS.h", 133, "xyz", "RL", "ALLOW_LEITH_QG"),
    # lane B (Task 25, global_ocean.cs32x15): DYNVARS.h COMMON /DYNVARS_DRAG_IN_PS/ (:136-147) under
    # ALLOW_SOLVE4_PS_AND_DRAG; NH_VARS.h COMMON /NH_VARS_R/ (:18-43) under ALLOW_NONHYDROSTATIC (AB2: gwNm1)
    "dU_psFacX": ("DYNVARS.h", 145, "xyz", "RL", "ALLOW_SOLVE4_PS_AND_DRAG"),
    "dV_psFacY": ("DYNVARS.h", 146, "xyz", "RL", "ALLOW_SOLVE4_PS_AND_DRAG"),
    "phi_nh": ("NH_VARS.h", 34, "xyz", "RL", "ALLOW_NONHYDROSTATIC"),
    "dPhiNH": ("NH_VARS.h", 35, "xy", "RL", "ALLOW_NONHYDROSTATIC"),
    "gW": ("NH_VARS.h", 36, "xyz", "RL", "ALLOW_NONHYDROSTATIC"),
    "gwNm1": ("NH_VARS.h", 40, "xyz", "RL", "ALLOW_NONHYDROSTATIC"),
    # SURFACE.h COMMON /ETA_UPDATES/ (SURFACE.h:42-45)
    "etaHnm1": ("SURFACE.h", 43, "xy", "RL", None), "dEtaHdt": ("SURFACE.h", 44, "xy", "RL", None),
    "PmEpR": ("SURFACE.h", 45, "xy", "RL", None),
    # SURFACE.h COMMON /SURF_CHANGE/ (:54-62), /LOCAL_CALC_SURF_DR/ (:67-68), /RSTAR_CHANGE/ (:82-99), NONLIN_FRSURF
    "hFac_surfC": ("SURFACE.h", 57, "xy", "RS", "NONLIN_FRSURF"),
    "hFac_surfW": ("SURFACE.h", 58, "xy", "RS", "NONLIN_FRSURF"),
    "hFac_surfS": ("SURFACE.h", 59, "xy", "RS", "NONLIN_FRSURF"),
    "hFac_surfNm1C": ("SURFACE.h", 60, "xy", "RS", "NONLIN_FRSURF"),
    "hFac_surfNm1W": ("SURFACE.h", 61, "xy", "RS", "NONLIN_FRSURF"),
    "hFac_surfNm1S": ("SURFACE.h", 62, "xy", "RS", "NONLIN_FRSURF"),
    "Rmin_surf": ("SURFACE.h", 68, "xy", "RL", "NONLIN_FRSURF"),
    "rStarFacC": ("SURFACE.h", 87, "xy", "RL", "NONLIN_FRSURF"),
    "rStarFacW": ("SURFACE.h", 88, "xy", "RL", "NONLIN_FRSURF"),
    "rStarFacS": ("SURFACE.h", 89, "xy", "RL", "NONLIN_FRSURF"),
    "pStarFacK": ("SURFACE.h", 90, "xy", "RL", "NONLIN_FRSURF"),
    "rStarFacNm1C": ("SURFACE.h", 91, "xy", "RL", "NONLIN_FRSURF"),
    "rStarFacNm1W": ("SURFACE.h", 92, "xy", "RL", "NONLIN_FRSURF"),
    "rStarFacNm1S": ("SURFACE.h", 93, "xy", "RL", "NONLIN_FRSURF"),
    "rStarExpC": ("SURFACE.h", 94, "xy", "RL", "NONLIN_FRSURF"),
    "rStarExpW": ("SURFACE.h", 95, "xy", "RL", "NONLIN_FRSURF"),
    "rStarExpS": ("SURFACE.h", 96, "xy", "RL", "NONLIN_FRSURF"),
    "rStarDhCDt": ("SURFACE.h", 97, "xy", "RL", "NONLIN_FRSURF"),
    "rStarDhWDt": ("SURFACE.h", 98, "xy", "RL", "NONLIN_FRSURF"),
    "rStarDhSDt": ("SURFACE.h", 99, "xy", "RL", "NONLIN_FRSURF"),
    # pkg/cd_code/CD_CODE_VARS.h COMMON /DYNVARS_CD/ (:3-13), ALLOW_CD_CODE: etaNm1 is written by INI_PSURF
    # (ini_psurf.F:65-76) and READ_PICKUP's CD part; uVelD, vVelD, uNM1, vNM1 by CD_CODE_INI_VARS (Task 15a)
    "uVelD": ("CD_CODE_VARS.h", 9, "xyz", "RL", "ALLOW_CD_CODE"),
    "vVelD": ("CD_CODE_VARS.h", 10, "xyz", "RL", "ALLOW_CD_CODE"),
    "etaNm1": ("CD_CODE_VARS.h", 11, "xy", "RL", "ALLOW_CD_CODE"),
    "uNM1": ("CD_CODE_VARS.h", 12, "xyz", "RL", "ALLOW_CD_CODE"),
    "vNM1": ("CD_CODE_VARS.h", 13, "xyz", "RL", "ALLOW_CD_CODE"),
    # ---- M1 lane GO (plan Task 15a): under NONLIN_FRSURF the GRID.h thickness factors and the CG2D.h operator are
    # time-varying common-block variables (UPDATE_R_STAR / UPDATE_SURF_DR write the first six every step,
    # forward_step.F:475, :839; UPDATE_CG2D the CG2D.h arrays, :866-871): carried with the State and copied into the
    # step's Grid / CG2DH where FORWARD_STEP starts (mitjax/model/src/forward_step.py, `nlfs_*`)
    "hFacC": ("GRID.h", 504, "xyz", "RS", "NONLIN_FRSURF"),
    "hFacW": ("GRID.h", 505, "xyz", "RS", "NONLIN_FRSURF"),
    "hFacS": ("GRID.h", 506, "xyz", "RS", "NONLIN_FRSURF"),
    "recip_hFacC": ("GRID.h", 507, "xyz", "RS", "NONLIN_FRSURF"),
    "recip_hFacW": ("GRID.h", 508, "xyz", "RS", "NONLIN_FRSURF"),
    "recip_hFacS": ("GRID.h", 509, "xyz", "RS", "NONLIN_FRSURF"),
    "aW2d": ("CG2D.h", 37, "xy", "RS", "NONLIN_FRSURF"), "aS2d": ("CG2D.h", 38, "xy", "RS", "NONLIN_FRSURF"),
    "aC2d": ("CG2D.h", 39, "xy", "RS", "NONLIN_FRSURF"), "pW": ("CG2D.h", 40, "xy", "RS", "NONLIN_FRSURF"),
    "pS": ("CG2D.h", 41, "xy", "RS", "NONLIN_FRSURF"), "pC": ("CG2D.h", 42, "xy", "RS", "NONLIN_FRSURF"),
    # pkg/mom_fluxform/MOM_FLUXFORM.h COMMON /LOCAL_MOM_CALC_RTRANS/ (:20-25): MOM_CALC_RTRANS' r* transports, written at
    # k = 1 and carried from level to level of DYNAMICS (NONLIN_FRSURF with select_rStar /= 0)
    "dWtransC": ("MOM_FLUXFORM.h", 23, "xy", "RL", "NONLIN_FRSURF"),
    "dWtransU": ("MOM_FLUXFORM.h", 24, "xy", "RL", "NONLIN_FRSURF"),
    "dWtransV": ("MOM_FLUXFORM.h", 25, "xy", "RL", "NONLIN_FRSURF"),
    # ---- ADVECT lane (plan Task 14): pkg/generic_advdiff/GAD_SOM_VARS.h COMMON /GAD_SOM_VARS_R/ (:24-28), under
    # GAD_ALLOW_TS_SOM_ADV (:16) as GAD_OPTIONS.h sets it: the 1rst & 2nd order moments of T and S, 6th dimension
    # nSOM = 3+6 (GAD.h:93), a tuple of nSOM FArrays named som_T_1 .. som_T_9 (n = x, y, z, xx, yy, zz, xy, xz, yz)
    "som_T": ("GAD_SOM_VARS.h", 27, "xyzSOM", "RL", "SOM"),
    "som_S": ("GAD_SOM_VARS.h", 28, "xyzSOM", "RL", "SOM"),
}
# GO lane: the NONLIN_FRSURF variable parts of GRID.h and CG2D.h (above), by destination
NLFS_GRID = ("hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS")
NLFS_CG2D = ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC")

# DYNVARS.h options whose fields are not carried (see the module docstring): a build that sets one is refused
UNPORTED_OPTIONS = ("ALLOW_SMAG_3D_DIFFUSIVITY", "ALLOW_BL79_LAT_VARY",
                    "INCLUDE_SOUNDSPEED_CALC_CODE")


def _declared(cond, cfg):
    if cond is None:
        return True
    if cond == "AB3":
        return bool(cfg.cpp.ALLOW_ADAMSBASHFORTH_3)
    if cond == "AB2":
        return not cfg.cpp.ALLOW_ADAMSBASHFORTH_3
    if cond == "SOM":       # ADVECT lane: GAD_SOM_VARS.h:16 as the pkg sees it (its own GAD_OPTIONS.h include)
        return bool(cfg.cpp.ALLOW_GENERIC_ADVDIFF) and bool(cfg.cpp.flag("GAD_ALLOW_TS_SOM_ADV", "GAD_OPTIONS.h"))
    return any(bool(getattr(cfg.cpp, c)) for c in cond.split("|"))


def fields_of(cfg):
    """The State fields this build declares, in DECLARATIONS order (then the PTRACERS lane's per-tracer fields)."""
    for opt in UNPORTED_OPTIONS:
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"State: {opt} (its DYNVARS.h / NH_VARS.h fields) is not ported")
    if cfg.cpp.ALLOW_NONHYDROSTATIC and cfg.cpp.ALLOW_ADAMSBASHFORTH_3:        # lane B: NH_VARS.h:37-38 gwNm
        raise NotImplementedError("State: NH_VARS.h gwNm (ALLOW_NONHYDROSTATIC with ALLOW_ADAMSBASHFORTH_3) is not "
                                  "ported")
    return tuple(n for n, (_, _, _, _, c) in DECLARATIONS.items() if _declared(c, cfg)) + ptracers_fields_of(cfg)


# ---- PTRACERS lane (plan Task 26): pkg/ptracers/PTRACERS_FIELDS.h COMMON /PTRACERS_FIELDS/ (:23-29) under
# ALLOW_PTRACERS, the last dimension (PTRACERS_num, PTRACERS_SIZE.h of the build) as one field per tracer named as
# the jaxdump records name them (pTracer_01, gpTrNm1_01, surfaceForcingPTr_01); under PTRACERS_ALLOW_DYN_STATE the
# 2nd-order moments of PTRACERS_MOD.h (_Ptracers_som(:,:,:,:,:,n,iTracer), n = 1..nSOM, GAD.h:93) as ptracers_som_01,
# a tuple of nSOM FArrays (PTRACERS_INIT_VARIA writes them for the SOM tracers only; the others stay NaN, never read).
# The conversion to and from the routines' PtracersFields: mitjax/pkg/ptracers/ptracers_fields_h.py.
PTRACERS_DECLARATIONS = {"pTracer": ("PTRACERS_FIELDS.h", 23, "xyz"), "gpTrNm1": ("PTRACERS_FIELDS.h", 25, "xyz"),
                         "surfaceForcingPTr": ("PTRACERS_FIELDS.h", 27, "xy"),
                         "ptracers_som": ("PTRACERS_MOD.h", 0, "xyzSOM")}


def ptracers_fields_of(cfg):
    """The PTRACERS_FIELDS.h (and PTRACERS_MOD.h SOM) fields of the build: () without ALLOW_PTRACERS."""
    if not cfg.cpp.ALLOW_PTRACERS:
        return ()
    from mitjax.pkg.ptracers.ptracers_readparms import _num
    som = bool(cfg.cpp.flag("PTRACERS_ALLOW_DYN_STATE", "PTRACERS_OPTIONS.h"))
    return tuple(f"{base}_{n:02d}" for n in range(1, _num(cfg) + 1)
                 for base in ("pTracer", "gpTrNm1", "surfaceForcingPTr") + (("ptracers_som",) if som else ()))


def _ptracers_decl(name):
    base, _, num = name.rpartition("_")
    if base in PTRACERS_DECLARATIONS and num.isdigit() and len(num) == 2:
        return PTRACERS_DECLARATIONS[base]
    return None


def declare(name, size, fill=np.nan):
    """A State array with its declared bounds, every point `fill` (NaN marks points no routine has written yet)."""
    pd = _ptracers_decl(name)                                                   # PTRACERS lane
    if pd is not None:
        if pd[2] == "xyzSOM":
            return tuple(_one(f"{name}_{m}", "xyz", size, fill) for m in range(1, 3+6+1))
        return _one(name, pd[2], size, fill)
    hdr, line, kind, typ, _ = DECLARATIONS[name]
    if kind == "xyz2":
        return tuple(_one(f"{name}_{m}", "xyz", size, fill) for m in (1, 2))
    if kind == "xyzSOM":    # ADVECT lane: GAD_SOM_VARS.h:27-28, sixth dimension nSOM = 3+6 (GAD.h:93)
        return tuple(_one(f"{name}_{m}", "xyz", size, fill) for m in range(1, 3+6+1))
    return _one(name, kind, size, fill)


def _one(name, kind, size, fill):
    b = bounds(kind, size)
    dims = [b[a][1] - b[a][0] + 1 for a in ("k", "j", "i") if a in b]
    return FArray(jnp.full((n_tiles(size),) + tuple(dims), fill, dtype=jnp.float64), name, **b)


@jax.tree_util.register_pytree_node_class
class State:
    """The DYNVARS.h / SURFACE.h state by Fortran name: `state.uVel`. Immutable; `state.replace(name=value)`."""

    __slots__ = ("_f",)

    def __init__(self, fields=None):
        object.__setattr__(self, "_f", dict(fields or {}))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in DECLARATIONS:
            hdr, line, _, _, cond = DECLARATIONS[name]
            raise AttributeError(f"{name} ({hdr}:{line}{', ' + cond if cond else ''}) is not in this State")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("State is immutable; use state.replace(...)")

    def __contains__(self, name):
        return name in self._f

    def names(self):
        return tuple(n for n in DECLARATIONS if n in self._f) + tuple(          # PTRACERS lane: per-tracer fields
            n for n in self._f if n not in DECLARATIONS)

    def replace(self, **fields):
        for k in fields:
            if k not in self._f:
                raise KeyError(f"{k} is not a field of this State (mitjax/model/state.py DECLARATIONS, fields_of)")
        return State({**self._f, **fields})

    def tree_flatten(self):
        keys = tuple(sorted(self._f))
        return tuple(self._f[k] for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, leaves):
        return cls(dict(zip(keys, leaves)))

    def __repr__(self):
        return f"State({', '.join(self.names())})"


def empty_state(cfg):
    """Every declared field of the build, NaN everywhere: the storage before INITIALISE_VARIA writes it (the
    Fortran's COMMON blocks hold zeros at load; every M1 field is written by INI_DYNVARS / INI_NLFS_VARS before it is
    read, and a NaN left anywhere shows a point the port never wrote)."""
    return State({n: declare(n, cfg.size) for n in fields_of(cfg)})
