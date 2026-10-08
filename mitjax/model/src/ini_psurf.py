"""INI_PSURF: model/src/ini_psurf.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_XY_RL
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.rw.read_rec import READ_FLD_XY_RL


def ini_psurf(state, *, cfg, grid, params, ex, rw):
    """INI_PSURF( myThid )   @63cdc0b model/src/ini_psurf.F:7-107

    C     | SUBROUTINE INI_PSURF
    C     | o Set model initial free-surface height/pressure.
    C     | There are several options for setting the initial
    C     | surface displacement (r unit) field.
    C     |  1. Inline code
    C     |  2. Two-dimensional data from a file.

    :49-57 etaN = 0. _d 0 on every point; :59-63 pSurfInitFile read (READ_FLD_XY_RL) and _EXCH_XY_RL; :65-76
    (ALLOW_CD_CODE) etaNm1 = etaN; :79-89 etaH = etaN, etaHnm1 = etaN, dEtaHdt = 0. _d 0 on every point. The
    ALLOW_SHELFICE branch (:91-104) raises (no M1 build compiles pkg/shelfice)."""
    sz = cfg.size
    ip = params.init
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    etaN = state.etaN.at[i, j].set(0.)                                         # :53
    if str(ip.pSurfInitFile).strip() != "":                                    # :59-63
        etaN = READ_FLD_XY_RL(ip.pSurfInitFile, " ", etaN, 0, rw=rw)
        etaN = EXCH_XY_RL(etaN, ex=ex)
    out = {"etaN": etaN}
    if cfg.cpp.ALLOW_CD_CODE:                                                  # :65-76
        out["etaNm1"] = state.etaNm1.at[i, j].set(etaN[i, j])
    out["etaH"] = state.etaH.at[i, j].set(etaN[i, j])                          # :83
    out["etaHnm1"] = state.etaHnm1.at[i, j].set(etaN[i, j])                    # :84
    out["dEtaHdt"] = state.dEtaHdt.at[i, j].set(0.)                            # :85
    if cfg.cpp.ALLOW_SHELFICE:                                                 # :91-104
        raise NotImplementedError("INI_PSURF: the ALLOW_SHELFICE load anomaly is not ported")
    return state.replace(**out)
