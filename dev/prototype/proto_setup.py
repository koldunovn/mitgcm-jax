"""Shared setup of the Task 8 style prototype: the static `cfg`, a common-block pytree, and the replay data turned into
the arguments of the two styles (`style_farray.py`: FArrays; `style_slices.py`: plain [tile, (k), j, i] arrays).

Which experiment's options and why: `global_ocean.90x40x15/code` (plan Task 8: an M1/M2 experiment that compiles all
three routines). Its build has 36 tiles of 10x10 with OLx = OLy = 3 and Nr = 15 (code/SIZE.h), so the tile axis
and multi-cell halos are exercised; ALLOW_AUTODIFF is undefined (no pkg/autodiff), ALLOW_DIAGNOSTICS is defined
(pkg/diagnostics), MOM_VI_ORIGINAL_VISCA4 and OLD_DST3_FORMULATION are undefined (effective macros of the build:
`options/{MOM_COMMON,MOM_VECINV,GAD}_OPTIONS.h.dM`, written by reference/check_build.py options). Its `input_ad`
variants (M2) run exactly these routines (vector-invariant momentum, DST3 advection).
"""

import re
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Cfg(NamedTuple):
    """Static configuration (hashable; never a differentiated argument)."""
    sNx: int
    sNy: int
    OLx: int
    OLy: int
    Nr: int
    ALLOW_AUTODIFF: bool            # MOM_COMMON_OPTIONS.h (via CPP_OPTIONS.h, PACKAGES_CONFIG.h)
    ALLOW_DIAGNOSTICS: bool         # MOM_VECINV_OPTIONS.h (PACKAGES_CONFIG.h)
    MOM_VI_ORIGINAL_VISCA4: bool    # MOM_VECINV_OPTIONS.h
    OLD_DST3_FORMULATION: bool      # GAD_OPTIONS.h
    useDiagnostics: bool            # PARAMS.h (data.pkg); the replay harness sets it .FALSE.


# routine -> the options header its .F includes first (pkg/mom_common/mom_calc_ke.F:1, ...)
OPTIONS_HEADER = {"ALLOW_AUTODIFF": "MOM_COMMON_OPTIONS.h", "ALLOW_DIAGNOSTICS": "MOM_VECINV_OPTIONS.h",
                  "MOM_VI_ORIGINAL_VISCA4": "MOM_VECINV_OPTIONS.h", "OLD_DST3_FORMULATION": "GAD_OPTIONS.h"}


def defined_macros(dm_file):
    """Macro names #define'd in a `cpp -dM` listing."""
    return {m.group(1) for m in re.finditer(r"^#define\s+(\w+)", Path(dm_file).read_text(), re.M)}


def cfg_from_build(options_dir, size, useDiagnostics=False):
    """Cfg from a build's effective macros (`<build>/options/*.h.dM`) and its SIZE.h values."""
    flags = {name: name in defined_macros(Path(options_dir) / f"{hdr}.dM") for name, hdr in OPTIONS_HEADER.items()}
    return Cfg(sNx=size["sNx"], sNy=size["sNy"], OLx=size["OLx"], OLy=size["OLy"], Nr=size["Nr"],
               useDiagnostics=useDiagnostics, **flags)


@jax.tree_util.register_pytree_node_class
class Common:
    """A Fortran common block (GRID.h, PARAMS.h): fields by their Fortran names; a pytree (names static)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def tree_flatten(self):
        keys = tuple(sorted(self.__dict__))
        return tuple(self.__dict__[k] for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, children):
        return cls(**dict(zip(keys, children)))


# GRID.h fields the three routines read, with their Fortran declarations (dims in Fortran order; model/inc/GRID.h)
GRID_DECL = {
    "hFacW": "ijk", "hFacS": "ijk", "recip_hFacC": "ijk", "recip_hFacW": "ijk", "recip_hFacS": "ijk",
    "maskW": "ijk", "maskS": "ijk", "rAw": "ij", "rAs": "ij", "recip_rA": "ij", "recip_dxC": "ij",
    "recip_dyC": "ij", "recip_dxG": "ij", "recip_dyG": "ij", "cosFacU": "j", "cosFacV": "j",
    "recip_deepFacC": "k"}
PARAM_NAMES = ("viscAhD", "viscAhZ", "viscA4D", "viscA4Z")


def bounds(cfg):
    """Declared bounds of the i, j, k dimensions (SIZE.h): (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, 1:Nr)."""
    return {"i": (1 - cfg.OLx, cfg.sNx + cfg.OLx), "j": (1 - cfg.OLy, cfg.sNy + cfg.OLy), "k": (1, cfg.Nr)}


def make_grid(cfg, g, style):
    """GRID.h common block for a style: FArrays (style "farray") or plain arrays (style "slices")."""
    from mitjax.farray import FArray
    b = bounds(cfg)
    out = {}
    for name, dims in GRID_DECL.items():
        a = jnp.asarray(g[name], jnp.float64)
        if style == "farray":
            out[name] = FArray(a, name, tiled=(name != "recip_deepFacC"), **{d: b[d] for d in dims})
        else:
            out[name] = a
    return Common(**out)


def make_params(g):
    """PARAMS.h floats as traced leaves (0-d float64 arrays), never closed over [E§5]."""
    return Common(**{n: jnp.asarray(g[n], jnp.float64) for n in PARAM_NAMES})


def level_array(cfg, a, name, style):
    """One level [tile, j, i] of an argument array as the style wants it (a 2-D tiled FArray, or as is)."""
    if style == "slices":
        return a
    from mitjax.farray import FArray
    b = bounds(cfg)
    return FArray(a, name, i=b["i"], j=b["j"])


def unwrap(x):
    return x.data if hasattr(x, "data") and hasattr(x, "dims") else x


def synthetic_grid(cfg, ntiles, seed=7):
    """A grid with land and partial cells for tests that run without the replay output (numpy, [tile, k, j, i])."""
    rng = np.random.default_rng(seed)
    Ny, Nx = cfg.sNy + 2 * cfg.OLy, cfg.sNx + 2 * cfg.OLx
    s3, s2 = (ntiles, cfg.Nr, Ny, Nx), (ntiles, Ny, Nx)
    g = {}
    for c in ("W", "S"):
        h = rng.uniform(0.05, 1.0, s3)
        h[rng.uniform(size=s3) < 0.3] = 0.0
        g[f"hFac{c}"] = h
        g[f"mask{c}"] = (h > 0).astype(float)
        g[f"recip_hFac{c}"] = np.where(h > 0, 1.0 / np.where(h > 0, h, 1.0), 0.0)
    hc = rng.uniform(0.05, 1.0, s3)
    hc[rng.uniform(size=s3) < 0.3] = 0.0
    g["recip_hFacC"] = np.where(hc > 0, 1.0 / np.where(hc > 0, hc, 1.0), 0.0)
    for n in ("rAw", "rAs"):
        g[n] = rng.uniform(1e10, 2e11, s2)
    g["recip_rA"] = 1.0 / rng.uniform(1e10, 2e11, s2)
    for n in ("recip_dxC", "recip_dyC", "recip_dxG", "recip_dyG"):
        g[n] = 1.0 / rng.uniform(1e5, 5e5, s2)
    g["cosFacU"] = rng.uniform(0.2, 1.0, (ntiles, Ny))
    g["cosFacV"] = rng.uniform(0.2, 1.0, (ntiles, Ny))
    g["recip_deepFacC"] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, cfg.Nr)
    g.update(viscAhD=5.0e5, viscAhZ=2.5e5, viscA4D=1.0e14, viscA4Z=0.6e14)
    return g
