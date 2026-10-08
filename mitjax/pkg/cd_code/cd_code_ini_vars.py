"""CD_CODE_INI_VARS: pkg/cd_code/cd_code_ini_vars.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.cd_code.cd_code_read_pickup import cd_code_read_pickup


def cd_code_ini_vars(state, *, cfg, params, ex, rw):
    """CD_CODE_INI_VARS( myThid )   @63cdc0b pkg/cd_code/cd_code_ini_vars.F:3-67

    Called by PACKAGES_INIT_VARIABLES when useCDscheme (packages_init_variables.F:200-210; the caller decides).
    Returns the State with CD_CODE_VARS.h uNM1, vNM1, uVelD, vVelD (every point, every level) and etaNm1 (every
    point) set to 0. _d 0 (:37-58), then, when `nIter0.NE.0 .OR. pickupSuff.NE.' '` (:60), CD_CODE_READ_PICKUP(
    nIter0 ) (:61). `params`: the ModelParams of ini_parms (time.nIter0, init.pickupSuff, ...)."""
    if not cfg.cpp.ALLOW_CD_CODE:
        return state
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :41, :52
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :42, :53
    st = {}
    for n in ("uNM1", "vNM1", "uVelD", "vVelD"):                                # :40-48  (DO K=1,Nr)
        a = getattr(state, n)
        for k in range(1, sz.Nr+1):
            a = a.at[i, j, k].set(0.)                                           # :43-46  0. _d 0
        st[n] = a
    st["etaNm1"] = state.etaNm1.at[i, j].set(0.)                                # :54  0. _d 0
    state = state.replace(**st)
    nIter0 = params.time.nIter0
    if nIter0 != 0 or str(params.init.pickupSuff).strip() != "":                # :60
        state = cd_code_read_pickup(state, nIter0, cfg=cfg, params=params, ex=ex, rw=rw)    # :61
    return state
