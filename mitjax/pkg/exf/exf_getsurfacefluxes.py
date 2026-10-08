"""EXF_GETSURFACEFLUXES: pkg/exf/exf_getsurfacefluxes.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def exf_getsurfacefluxes(myTime, myIter, f, *, cfg, exf=None, ctrl=None):
    """EXF_GETSURFACEFLUXES( myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_getsurfacefluxes.F:3-171

    C     | o Mid-level routine for enabling the use of flux fields as control variables.

    Every statement sits under ALLOW_CTRL (:69-169): without it the routine does nothing. Lane M4ADCOL
    (1D_ocean_ice_column/code_ad): ALLOW_CTRL with ALLOW_GENTIM2D_CONTROL and useCTRL: `ctrl` = dict(xx_gentim2d=
    {iarr: FArray}, files={iarr: xx_gentim2d_file}) of this step; the loop :105-137 (no ALLOW_ROTATE_UV_CONTROLS)
    adds xx_gentim2d to ustress / vstress / hflux / sflux where the file name starts with xx_tauu / xx_tauv /
    xx_hflux / xx_sflux (interior). The block :140-167 (useCTRL .AND. .NOT.useAtmWind) holds only
    ALLOW_ROTATE_UV_CONTROLS code (raises when compiled). Returns `f`."""
    if not (cfg.cpp.flag("ALLOW_CTRL") and cfg.use_flag("useCTRL")):
        return f
    if not cfg.cpp.flag("ALLOW_GENTIM2D_CONTROL", "CTRL_OPTIONS.h"):
        raise NotImplementedError("EXF_GETSURFACEFLUXES: ALLOW_CTRL without ALLOW_GENTIM2D_CONTROL is not ported")
    if cfg.cpp.flag("ALLOW_ROTATE_UV_CONTROLS", "CTRL_OPTIONS.h"):
        raise NotImplementedError("EXF_GETSURFACEFLUXES: ALLOW_ROTATE_UV_CONTROLS (:118-125, :140-165) is not ported")
    if ctrl is None:
        raise ValueError("EXF_GETSURFACEFLUXES: useCTRL needs the gentim2d controls `ctrl`")
    sz = cfg.size
    j = loop_j(1, sz.sNy)                                                      # :108
    i = loop_i(1, sz.sNx)                                                      # :109
    f = dict(f)
    for iarr in sorted(ctrl["xx_gentim2d"]):                                   # :110 DO iarr = 1, maxCtrlTim2D
        fname = ctrl["files"][iarr]
        for cname, fld in (("xx_tauu", "ustress"), ("xx_tauv", "vstress"),     # :112-117
                           ("xx_hflux", "hflux"), ("xx_sflux", "sflux")):      # :126-131
            if fname[:len(cname)] == cname:                                    # (1:n) .EQ. 'xx_...'
                f[fld] = f[fld].at[i, j].set(f[fld][i, j] + ctrl["xx_gentim2d"][iarr][i, j])
    return f
