"""INI_GRID: model/src/ini_grid.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import Grid, declare
from mitjax.model.src.ini_cartesian_grid import ini_cartesian_grid
from mitjax.model.src.ini_curvilinear_grid import ini_curvilinear_grid
from mitjax.model.src.ini_spherical_polar_grid import ini_spherical_polar_grid
from mitjax.model.src.ini_vertical_grid import ini_vertical_grid
from mitjax.model.src.load_grid_spacing import load_grid_spacing
from mitjax.ops.safe import safe_div

# the horizontal GRID.h arrays INI_GRID initialises (:66-111), in the Fortran order
_ZEROED = ("xC", "yC", "xG", "yG", "dxC", "dyC", "dxG", "dyG", "dxF", "dyF", "dxV", "dyU", "rA", "rAz", "rAw", "rAs",
           "recip_dxG", "recip_dyG", "recip_dxC", "recip_dyC", "recip_dxF", "recip_dyF", "recip_dxV", "recip_dyU",
           "recip_rA", "recip_rAs", "recip_rAw", "recip_rAz", "tanPhiAtU", "tanPhiAtV")
# (reciprocal, length) pairs of :149-180, in the Fortran order
_RECIPROCALS = (("recip_dxG", "dxG"), ("recip_dyG", "dyG"), ("recip_dxC", "dxC"), ("recip_dyC", "dyC"),
                ("recip_dxF", "dxF"), ("recip_dyF", "dyF"), ("recip_dxV", "dxV"), ("recip_dyU", "dyU"),
                ("recip_rA", "rA"), ("recip_rAs", "rAs"), ("recip_rAw", "rAw"), ("recip_rAz", "rAz"))


def ini_grid(grid=None, *, cfg, params, ex=None, rw=None):
    """INI_GRID( myThid )   @63cdc0b model/src/ini_grid.F:8-238

    C     !DESCRIPTION:
    C     These arrays are used throughout the code in evaluating gradients,
    C     integrals and spatial avarages. This routine is called separately
    C     by each thread and initializes only the region of the domain it is
    C     "responsible" for.
    C     !CALLING SEQUENCE:
    C     INI_GRID
    C      |   -- LOAD_GRID_SPACING
    C      |   -- INI_VERTICAL_GRID
    C      |    / INI_CARTESIAN_GRID
    C      |   /  INI_SPHERICAL_POLAR_GRID
    C      |   \\  INI_CURVILINEAR_GRID
    C      |    \\ INI_CYLINDER_GRID

    Returns (grid, delX, delY, latBandClimRelax): the GRID.h fields of the horizontal and vertical grid, and the
    SET_GRID.h / PARAMS.h values LOAD_GRID_SPACING set. `grid`: the Grid so far (a new one if None). `ex`, `rw`: the
    exchanger and run directory, needed by the curvilinear grid only (its file reads and exchanges).

    The initialisation (:66-111) and reciprocal (:149-180) loops are pointwise over the full tile and run vectorised;
    `IF (dxG .NE. 0.) recip_dxG = 1. _d 0/dxG` keeps the prior zero elsewhere, with the division guarded before it
    (mitjax/ops/safe.py: the guarded lane divides by 1, so every lane is finite). Ported: Cartesian and spherical-polar
    grids (:130-133), curvilinear (:134-135, lane B Task 22: mitjax/model/src/ini_curvilinear_grid.py);
    cylindrical (:136-137) raises. The ALLOW_MONITOR statistics printout (:184-232,
    MON_PRINTSTATS_RS of the grid arrays, STDOUT only) is not ported: it writes no model variable.
    """
    sz = cfg.size
    grid = Grid() if grid is None else grid

    delX, delY, latBandClimRelax = load_grid_spacing(cfg=cfg, params=params)       # :57

    grid = ini_vertical_grid(grid, cfg=cfg, params=params)                          # :60

    grid = init_horizontal(grid, sz)                                                # :66-111

    # :129-146  horizontal grid and coordinate system
    if params.usingCartesianGrid:
        grid = ini_cartesian_grid(grid, delX, delY, cfg=cfg, params=params)
    elif params.usingSphericalPolarGrid:
        grid = ini_spherical_polar_grid(grid, delX, delY, cfg=cfg, params=params)
    elif params.usingCurvilinearGrid:
        if ex is None or rw is None:
            raise ValueError("INI_GRID: INI_CURVILINEAR_GRID needs the exchanger `ex` and the run directory `rw`")
        grid = ini_curvilinear_grid(grid, cfg=cfg, params=params, ex=ex, rw=rw)
    elif params.usingCylindricalGrid:
        raise NotImplementedError("INI_GRID: INI_CYLINDER_GRID is not ported")
    else:
        raise ValueError("S/R INI_GRID: No grid coordinate system has been selected\n"
                         "ABNORMAL END: S/R INI_GRID")

    grid = reciprocals(grid, sz)                                                    # :149-180

    return grid, delX, delY, latBandClimRelax


def init_horizontal(grid, sz):
    """ini_grid.F:66-111: initialise (everywhere) all horizontal grid arrays (a part of INI_GRID, separated so that
    the curvilinear grid gate can build the horizontal grid of a p-coordinate experiment, lane B Task 22)."""
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    fields = {}
    for name in _ZEROED:
        fields[name] = declare(name, sz).at[i, j].set(0.)
    fields["angleCosC"] = declare("angleCosC", sz).at[i, j].set(1.)
    fields["angleSinC"] = declare("angleSinC", sz).at[i, j].set(0.)
    fields["u2zonDir"] = declare("u2zonDir", sz).at[i, j].set(1.)
    fields["v2zonDir"] = declare("v2zonDir", sz).at[i, j].set(0.)
    for name in ("cosFacU", "cosFacV", "sqCosFacU", "sqCosFacV"):
        fields[name] = declare(name, sz).at[j].set(1.)
    return grid.replace(**fields)


def reciprocals(grid, sz):
    """ini_grid.F:149-180: reciprocals of the grid lengths (a part of INI_GRID, see init_horizontal)."""
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    fields = {}
    for rname, name in _RECIPROCALS:
        d = getattr(grid, name)[i, j]
        r = getattr(grid, rname)
        fields[rname] = r.at[i, j].set(safe_div(1.0, d, d != 0., fill=r[i, j]))    # 1. _d 0/dxG(i,j,bi,bj)
    return grid.replace(**fields)
