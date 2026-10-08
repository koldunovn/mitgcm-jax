"""INI_FFIELDS (model/src/ini_ffields.F @63cdc0b) and the FFIELDS.h common blocks it initialises, as Fortran-index
arrays by Fortran name.

`FFields` holds the FFIELDS.h variables (model/inc/FFIELDS.h @63cdc0b) that THIS build declares and the ported
forcing routines read or write, each an `FArray` with its declared bounds (`DECLARATIONS`: line, shape kind, CPP
condition). A pytree (the arrays are the leaves; field names and `loadedRec` are static), immutable:
`ff.replace(fu=..., ...)`, as a Fortran routine writes the common block. Kept beside the core lane's State
(mitjax/model/state.py: DYNVARS.h, SURFACE.h), which leaves FFIELDS.h to the forcing routines (its docstring).

`loadedRec(nSx,nSy)` (FFIELDS.h:186-187, INTEGER) is the time record EXTERNAL_FIELDS_LOAD last read; every tile holds
the same value (external_fields_load.F:237-241 sets all tiles together), so it is one static Python int here: the
record sequencing runs on the host (see external_fields_load.py).

Not carried (no M1 build defines the option; `fields_of` raises if a build does): ALLOW_GEOTHERMAL_FLUX,
ALLOW_FRICTION_HEATING, ALLOW_EDDYPSI, EXCLUDE_FFIELDS_LOAD. PTRACERS lane: SHORTWAVE_HEATING's Qsw0/1 and SWFrac3D
are carried (tutorial_tracer_adjsens); lane B (Task 25, global_ocean.cs32x15): ALLOW_ADDFLUID (addMass) and
ALLOW_BALANCE_FLUXES (weight2BalanceFlx). botDragU/V (FFIELDS.h, written by the bottom-drag routines) and
adjustColdSST_diag (a diagnostic of FREEZE_SURFACE) are carried as INI_FFIELDS sets them.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.grid import bounds, n_tiles

# name -> (FFIELDS.h line of the declaration, shape kind, type, CPP condition or None)
DECLARATIONS = {
    "fu": (100, "xy", "RS", None), "fv": (101, "xy", "RS", None),
    "Qnet": (102, "xy", "RS", None), "Qsw": (103, "xy", "RS", None),
    "EmPmR": (104, "xy", "RS", None), "saltFlux": (105, "xy", "RS", None),
    "SST": (106, "xy", "RS", None), "SSS": (107, "xy", "RS", None),
    "lambdaThetaClimRelax": (108, "xy", "RS", None), "lambdaSaltClimRelax": (109, "xy", "RS", None),
    "phiTide2d": (110, "xy", "RS", None), "pLoad": (111, "xy", "RS", None), "sIceLoad": (112, "xy", "RS", None),
    "gcmSST": (119, "xy", "RL", None),
    # /TDFIELDS/ (FFIELDS.h:189-228, #ifndef EXCLUDE_FFIELDS_LOAD)
    "taux0": (203, "xy", "RS", None), "tauy0": (204, "xy", "RS", None), "Qnet0": (205, "xy", "RS", None),
    "EmPmR0": (206, "xy", "RS", None), "saltFlux0": (207, "xy", "RS", None), "SST0": (208, "xy", "RS", None),
    "SSS0": (209, "xy", "RS", None),
    "taux1": (210, "xy", "RS", None), "tauy1": (211, "xy", "RS", None), "Qnet1": (212, "xy", "RS", None),
    "EmPmR1": (213, "xy", "RS", None), "saltFlux1": (214, "xy", "RS", None), "SST1": (215, "xy", "RS", None),
    "SSS1": (216, "xy", "RS", None),
    "pLoad0": (226, "xy", "RS", "ATMOSPHERIC_LOADING"), "pLoad1": (227, "xy", "RS", "ATMOSPHERIC_LOADING"),
    # PTRACERS lane (tutorial_tracer_adjsens/code_ad): SHORTWAVE_HEATING's /TDFIELDS/ Qsw0, Qsw1 (FFIELDS.h:217-220)
    # and /FFIELDS_SWFRAC/ SWFrac3D(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1,nSx,nSy) (FFIELDS.h:144-153, set by INI_FORCING)
    "Qsw0": (218, "xy", "RS", "SHORTWAVE_HEATING"), "Qsw1": (219, "xy", "RS", "SHORTWAVE_HEATING"),
    "SWFrac3D": (152, "xyzp1", "RS", "SHORTWAVE_HEATING"),
    # lane B (Task 25, global_ocean.cs32x15): /FFIELDS_ADD_FLUID/ (:121-124), /FFIELDS_W2BALANCE/ (:137-142)
    "addMass": (123, "xyz", "RL", "ALLOW_ADDFLUID"),
    "weight2BalanceFlx": (141, "xy", "RS", "ALLOW_BALANCE_FLUXES"),
    # /SURFACE_FORCING/ (FFIELDS.h:263-267)
    "surfaceForcingU": (263, "xy", "RL", None), "surfaceForcingV": (264, "xy", "RL", None),
    "surfaceForcingT": (265, "xy", "RL", None), "surfaceForcingS": (266, "xy", "RL", None),
    "adjustColdSST_diag": (267, "xy", "RL", None),
    # /FFIELDS_botDrag/
    "botDragU": (274, "xy", "RS", None), "botDragV": (275, "xy", "RS", None),
}

UNPORTED_OPTIONS = ("ALLOW_GEOTHERMAL_FLUX", "ALLOW_FRICTION_HEATING", "ALLOW_EDDYPSI", "EXCLUDE_FFIELDS_LOAD")


def fields_of(cfg):
    """The FFIELDS.h fields this build declares, in DECLARATIONS order."""
    for opt in UNPORTED_OPTIONS:
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"FFIELDS.h: {opt} (its fields and code paths) is not ported")
    return tuple(n for n, (_, _, _, c) in DECLARATIONS.items() if c is None or getattr(cfg.cpp, c))


def declare(name, size, fill=np.nan):
    """An FFIELDS.h array with its declared bounds, every point `fill` (NaN: not yet written)."""
    _, kind, _, _ = DECLARATIONS[name]
    if kind == "xyzp1":                     # PTRACERS lane: SWFrac3D's vertical dimension Nr+1 (FFIELDS.h:152)
        b = dict(bounds("xy", size), k=(1, size.Nr + 1))
    else:
        b = bounds(kind, size)
    dims = [b[a][1] - b[a][0] + 1 for a in ("k", "j", "i") if a in b]
    return FArray(jnp.full((n_tiles(size),) + tuple(dims), fill, dtype=jnp.float64), name, **b)


@jax.tree_util.register_pytree_node_class
class FFields:
    """The FFIELDS.h common blocks by Fortran name (`ff.fu`) plus `loadedRec` (static int). Immutable."""

    __slots__ = ("_f", "loadedRec")

    def __init__(self, fields=None, loadedRec=0):
        object.__setattr__(self, "_f", dict(fields or {}))
        object.__setattr__(self, "loadedRec", int(loadedRec))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in DECLARATIONS:
            line, _, _, cond = DECLARATIONS[name]
            raise AttributeError(f"{name} (FFIELDS.h:{line}{', ' + cond if cond else ''}) is not in this build")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("FFields is immutable; use ff.replace(...)")

    def __contains__(self, name):
        return name in self._f

    def names(self):
        return tuple(self._f)

    def replace(self, loadedRec=None, **fields):
        unknown = set(fields) - set(self._f)
        if unknown:
            raise KeyError(f"not FFIELDS.h fields of this build: {sorted(unknown)}")
        new = dict(self._f)
        new.update(fields)
        return FFields(new, self.loadedRec if loadedRec is None else loadedRec)

    def tree_flatten(self):
        keys = tuple(self._f)
        return tuple(self._f[k] for k in keys), (keys, self.loadedRec)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        keys, loadedRec = aux
        return cls(dict(zip(keys, leaves)), loadedRec)

    def __repr__(self):
        return f"FFields({', '.join(self._f)}; loadedRec={self.loadedRec})"


def ini_ffields(*, cfg):
    """INI_FFIELDS( myThid )   @63cdc0b model/src/ini_ffields.F:6-148

    C     | SUBROUTINE INI_FFIELDS
    C     | o Initialise to zero all FFIELDS.h arrays

    :43-90 every FFIELDS.h 2-D array of this build = 0. _d 0 on every point (halos included), loadedRec = 0
    (:91-93); :126-141 surfaceForcingU/V/T/S, adjustColdSST_diag, lambdaThetaClimRelax, lambdaSaltClimRelax,
    botDragU/V = 0. _d 0. SHORTWAVE_HEATING (PTRACERS lane): Qsw0, Qsw1 = 0. _d 0 (:76-79); SWFrac3D is declared
    unset (NaN) for INI_FORCING. Lane B (Task 25): weight2BalanceFlx (ALLOW_BALANCE_FLUXES, :135-137) with the 2-D
    fields; addMass = 0. _d 0 on every level (ALLOW_ADDFLUID, the k loop :96-103)."""
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    out = {}
    for name in fields_of(cfg):
        kind = DECLARATIONS[name][1]
        if kind == "xy":
            out[name] = declare(name, sz).at[i, j].set(0.)                       # :43-90, :126-141
        elif name == "addMass":                                                 # :96-103 (lane B)
            from mitjax.farray import loops_kji
            k3, j3, i3 = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
            out[name] = declare(name, sz).at[i3, j3, k3].set(0.)                # :100
        else:                                                                   # SWFrac3D: INI_FORCING (PTRACERS lane, :130-187)
            out[name] = declare(name, sz)
    return FFields(out, loadedRec=0)                                             # :91-93
