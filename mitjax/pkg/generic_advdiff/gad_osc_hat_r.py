"""pkg/generic_advdiff/gad_osc_hat_r.F: WENO oscillation derivatives in R.

    o GAD_OSC_LOC_R   ported
    o GAD_OSC_HAT_R   ported

Column routines: the Fortran works on one column (ix,iy) of the caller's `do iy / do ix` loops (column arrays
`(1-3:Nr+3)`); here all columns at once (each column reads and writes only itself), so the column arrays carry the
column indices (declared `(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, 1-3:Nr+3)`), `ix, iy` are the caller's column loops and the
`ir` loops are vectorised together with them (`loops_kji`). `ohat(1:2, :)` is {1: ohat(1,:), 2: ohat(2,:)}.
"""

from mitjax.farray import Loop, loops_kji


def _range(ir):
    return (ir.first, ir.last) if isinstance(ir, Loop) else (ir, ir)


def gad_osc_loc_r(ir, mask, fbar, ohat, *, ix, iy, cfg):
    """GAD_OSC_LOC_R(ir, mask, fbar, ohat)   @63cdc0b pkg/generic_advdiff/gad_osc_hat_r.F:10-91

    `ir` is one level (an int) or a k loop over levels that all take the same branch of the IF on ir (:27-28, :50,
    :68); `ix, iy` (keywords; not Fortran arguments) are the caller's column loops. Returns ohat. `0.25`, `0.50`,
    `0.` are D literals.
    """
    Nr = cfg.Nr
    ohat = dict(ohat)
    first, last = _range(ir)
    floc = {}                                                           # :25  _RL floc(-1:+1)

    if first > +1-3 and last < Nr+3:                                    # :27-28
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[ix, iy, +0+ir]                                  # :32

        floc[-1] = (floc[+0] +                                          # :34-37
                    mask[ix, iy, ir-1]*(fbar[ix, iy, ir-1]-floc[+0]))
        floc[+1] = (floc[+0] +
                    mask[ix, iy, ir+1]*(fbar[ix, iy, ir+1]-floc[+0]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy, ir].set(floc[+1]*0.25            # :41-42
                                               - floc[-1]*0.25)

        ohat[+2] = ohat[+2].at[ix, iy, ir].set(floc[+1]*0.25            # :44-46
                                               - floc[+0]*0.50
                                               + floc[-1]*0.25)
        return ohat
    if first != last:
        raise ValueError(f"GAD_OSC_LOC_R: the levels {first}..{last} do not all take one branch of the IF on ir")

    if ir == +1-3:                                                      # :50
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[ix, iy, +0+ir]                                  # :54

        floc[+1] = (floc[+0] +                                          # :56-57
                    mask[ix, iy, ir+1]*(fbar[ix, iy, ir+1]-floc[+0]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy, ir].set(floc[+1]*0.50            # :61-62
                                               - floc[+0]*0.50)

        ohat[+2] = ohat[+2].at[ix, iy, ir].set(0.)                      # :64

    if ir == Nr+3:                                                      # :68
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[ix, iy, +0+ir]                                  # :72

        floc[-1] = (floc[+0] +                                          # :74-75
                    mask[ix, iy, ir-1]*(fbar[ix, iy, ir-1]-floc[+0]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy, ir].set(floc[+0]*0.50            # :79-80
                                               - floc[-1]*0.50)

        ohat[+2] = ohat[+2].at[ix, iy, ir].set(0.)                      # :82
    return ohat


def gad_osc_hat_r(ix, iy, mask, fbar, ohat, *, cfg):
    """GAD_OSC_HAT_R(bi,bj,ix,iy, mask, fbar, ohat, myThid)   @63cdc0b pkg/generic_advdiff/gad_osc_hat_r.F:95-127

    C     | OSC_HAT_R: compute WENO oscillation derivatives in R.          |

    The level loop `do ir = +1-3, Nr+3` (:118-122) is vectorised: GAD_OSC_LOC_R is called for its first level, its
    interior and its last level (the three branches of its IF on ir); each level writes only ohat(:,ir).
    Returns ohat.
    """
    Nr = cfg.Nr
    ir, iy, ix = loops_kji((+1-3+1, Nr+3-1), (iy.first, iy.last), (ix.first, ix.last))   # :118 (with the columns)
    ohat = gad_osc_loc_r(+1-3, mask, fbar, ohat, ix=ix, iy=iy, cfg=cfg)                  # :118-122
    ohat = gad_osc_loc_r(ir, mask, fbar, ohat, ix=ix, iy=iy, cfg=cfg)
    ohat = gad_osc_loc_r(Nr+3, mask, fbar, ohat, ix=ix, iy=iy, cfg=cfg)
    return ohat
