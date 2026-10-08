"""INI_DEPTHS: model/src/ini_depths.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.eesupp.exch_rs import EXCH_XY_RS
from mitjax.model.grid import declare


def _blank(s):
    return str(s).strip() == ""


def ini_depths(grid, *, cfg, params, ex, rw):
    """INI_DEPTHS( myThid )   @63cdc0b model/src/ini_depths.F:7-298

    C     | SUBROUTINE INI_DEPTHS
    C     | o define R_position of Lower and Surface Boundaries
    C     |atmosphere orography:
    C     | define either in term of P_topo or converted from Z_topo
    C     |ocean bathymetry:
    C     | The depths of the bottom of the model is specified in
    C     | terms of an XY map with one depth for each column of
    C     | grid cells. Depths do not have to coincide with the
    C     | model levels. The model lopping algorithm makes it
    C     | possible to represent arbitrary depths.
    C     | The mode depths map also influences the models topology
    C     | By default the model domain wraps around in X and Y.
    C     | This default doughnut topology can be changed by
    C     | setting elements in the depths map to zero.
    C     | Domain boundaries can be on cell faces.

    Returns `grid` with R_low, Ro_surf (GRID.h) and topoZ (SURFACE.h).

    `ex`: the experiment's exchanger (mitjax/eesupp.exchange.Exchanger, Task 7b); EXCH_XY_RS (mitjax/eesupp/exch_rs.py, the
    RL exchange for _RS = Real*8) is called where the Fortran calls `_EXCH_XY_RS` (:135, :217). `rw`: the model-input
    reader providing `READ_REC_XY_RS(fName, field, iRec, myIter)` (mitjax/pkg/rw/read_rec.py RW: pkg/rw/read_rec.F:22
    over MDS_READ_FIELD, mitjax/io/mds.py), called at :119; it returns `field` with the tile interiors read from the
    file and the halos untouched.

    Ported for the M1 variants (docs/coverage/*.md): z coordinates; R_low from `bathyFile` (:119) or, with no
    bathyFile (advect_xy), rF(Nr+1) on the interior (:94-102); Ro_surf = rF(1) on the interior (no topoFile,
    :155-166); the domain closing at |yC| >= 90 for the spherical grid (:246-268, all points incl. halos).
    Not ported (raise): pressure coordinates (:60-67, :150-153, :224-245), topoFile (:168-214), the MNC bathymetry
    read (`useMNC .AND. mnc_read_bathy`, :105-116), OBCS_CHECK_DEPTHS (:283-290, pkg/obcs not compiled in M1).
    Not ported because they write no model variable: PLOT_FIELD_XYRS (:137-141, :270-275, plotLevel >= debLevC:
    STDOUT only) and, under ALLOW_EXCH2, EXCH2_CHECK_DEPTHS (:292-295, pkg/exch2/exch2_check_depths.F: counts
    unconnected wet edge points of blank tiles and prints warnings only; global_ocean.90x40x15 has no blank tile).
    The point loops are pointwise and run vectorised.
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    rF, yC = grid.rF, grid.yC

    if params.usingPCoords and not _blank(params.bathyFile) and not _blank(params.topoFile):    # :60-67
        raise ValueError("S/R INI_DEPTHS: both bathyFile & topoFile are specified: select the right one !\n"
                         "ABNORMAL END: S/R INI_DEPTHS")

    # :69-82  0) Initialize R_low and Ro_surf (define an empty domain)
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    R_low = declare("R_low", sz).at[i, j].set(0.)                   # 0. _d 0
    Ro_surf = declare("Ro_surf", sz).at[i, j].set(0.)
    topoZ = declare("topoZ", sz).at[i, j].set(0.)

    # :88-131  1) R_low = the lower (in r sense) boundary of the fluid column
    if params.usingPCoords or _blank(params.bathyFile):            # :91-102
        j = loop_j(1, sNy)
        i = loop_i(1, sNx)
        R_low = R_low.at[i, j].set(rF[Nr+1])
    else:
        if cfg.cpp.ALLOW_MNC and params.useMNC and params.mnc_read_bathy:                    # :105-116
            raise NotImplementedError("INI_DEPTHS: bathymetry read through MNC (mnc_read_bathy) is not ported")
        R_low = rw.READ_REC_XY_RS(params.bathyFile, R_low, 1, 0)   # :119
    R_low = EXCH_XY_RS(R_low, ex=ex)                                # :135  _EXCH_XY_RS(R_low, myThid)

    # :146-214  2) Ro_surf = surface boundary
    if params.usingPCoords and not _blank(params.bathyFile):       # :150-153
        raise NotImplementedError("INI_DEPTHS: Po_surf from bathyFile (pressure coordinates) is not ported")
    elif _blank(params.topoFile):                                   # :155-166
        j = loop_j(1, sNy)
        i = loop_i(1, sNx)
        Ro_surf = Ro_surf.at[i, j].set(rF[1])
    else:                                                           # :168-214
        raise NotImplementedError("INI_DEPTHS: topoFile is not ported")
    Ro_surf = EXCH_XY_RS(Ro_surf, ex=ex)                            # :217  _EXCH_XY_RS(Ro_surf, myThid)

    # :221-268  3) close the domain
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    if params.usingPCoords:                                         # :224-245 (lane B, Task 25)
        if params.usingSphericalPolarGrid:                          # :238-240
            Ro_surf = Ro_surf.at[i, j].set(jnp.where(jnp.abs(yC[i, j]) >= 90., rF[Nr+1], Ro_surf[i, j]))
    else:                                                           # :246-268
        if params.usingSphericalPolarGrid:
            R_low = R_low.at[i, j].set(jnp.where(jnp.abs(yC[i, j]) >= 90., rF[1], R_low[i, j]))

    if cfg.cpp.ALLOW_OBCS and cfg.use_flag("useOBCS"):              # :283-290
        raise NotImplementedError("INI_DEPTHS: OBCS_CHECK_DEPTHS is not ported")

    return grid.replace(R_low=R_low, Ro_surf=Ro_surf, topoZ=topoZ)
