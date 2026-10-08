"""GAD_SOM_FILL_CS_CORNER: pkg/generic_advdiff/gad_som_fill_cs_corner.F @63cdc0b (fill the corner-halo regions of
the volume and of every SOM moment of one level on the cubed sphere)."""

from mitjax.eesupp.fill_cs_corner_ag_rl import fill_cs_corner_ag_rl
from mitjax.eesupp.fill_cs_corner_tr_rl import fill_cs_corner_tr_rl


def gad_som_fill_cs_corner(fill4dirX, sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, *,
                           corners, useCubedSphereExchange, sNx, sNy, OLx, OLy):
    """GAD_SOM_FILL_CS_CORNER( fill4dirX, sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz,
    bi, bj, myThid )   @63cdc0b pkg/generic_advdiff/gad_som_fill_cs_corner.F:6-87

    C     | o Fill corner-halo regions of all tracer moments with proper values

    Returns the 11 arrays. Arrays: raw [T, ny, nx] storage of one level (every tile at once); `corners` [T, 4]
    (SW, SE, NW, NE) of FILL_CS_CORNER_*, cleared on the tiles that do not make this call (the one addition to the
    Fortran signature, as for the eesupp fills). :65-69 selectDir = 1 (fill4dirX) or 2; :70-85 the TR fills of
    sm_v, sm_o, sm_z, sm_zz (withSigns .FALSE.) and sm_xy (.TRUE.), the AG fills of (sm_x, sm_y) (.TRUE.),
    (sm_xx, sm_yy) (.FALSE.), (sm_xz, sm_yz) (.TRUE.), in the Fortran order (each fill reads and writes its own
    arrays only)."""
    kw = dict(sNx=sNx, sNy=sNy, OLx=OLx, OLy=OLy)
    if useCubedSphereExchange:                                                  # :63
        selectDir = 1 if fill4dirX else 2                                       # :64-68
        sm_v = fill_cs_corner_tr_rl(selectDir, False, sm_v, corners, True, **kw)          # :69-70
        sm_o = fill_cs_corner_tr_rl(selectDir, False, sm_o, corners, True, **kw)          # :71-72
        sm_x, sm_y = fill_cs_corner_ag_rl(fill4dirX, True, sm_x, sm_y, corners, True, **kw)   # :73-74
        sm_z = fill_cs_corner_tr_rl(selectDir, False, sm_z, corners, True, **kw)          # :75-76
        sm_xx, sm_yy = fill_cs_corner_ag_rl(fill4dirX, False, sm_xx, sm_yy, corners, True, **kw)   # :77-78
        sm_zz = fill_cs_corner_tr_rl(selectDir, False, sm_zz, corners, True, **kw)        # :79-80
        sm_xy = fill_cs_corner_tr_rl(selectDir, True, sm_xy, corners, True, **kw)         # :81-82
        sm_xz, sm_yz = fill_cs_corner_ag_rl(fill4dirX, True, sm_xz, sm_yz, corners, True, **kw)   # :83-84
    return sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz
