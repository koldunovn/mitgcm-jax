"""pkg/generic_advdiff/gad_osc_hat_x.F: WENO oscillation derivatives in X.

    o GAD_OSC_LOC_X   ported
    o GAD_OSC_HAT_X   ported

Row routines: the Fortran works on one row iy of the caller's `do iy` loop (row arrays `(1-OLx:sNx+OLx)`); here all
rows at once (each row reads and writes only itself), so the row arrays carry the row index j (they are declared
like the caller's 2-D fields, `(1-OLx:sNx+OLx, 1-OLy:sNy+OLy)`) and `iy` is the caller's row loop. `ohat(1:2, :)` is
{1: ohat(1,:), 2: ohat(2,:)}.
"""

from mitjax.farray import Loop, loop_i


def _range(ix):
    return (ix.first, ix.last) if isinstance(ix, Loop) else (ix, ix)


def gad_osc_loc_x(ix, mask, fbar, ohat, *, iy, cfg):
    """GAD_OSC_LOC_X(ix, mask, fbar, ohat)   @63cdc0b pkg/generic_advdiff/gad_osc_hat_x.F:10-99

    `ix` is one point (an int) or a loop over points that all take the same branch of the IF on ix (:27-28, :50,
    :72); `iy` (keyword; not a Fortran argument) is the caller's row loop. Returns ohat. `0.25`, `0.50` are D
    literals.
    """
    sNx, OLx = cfg.sNx, cfg.OLx
    ohat = dict(ohat)
    first, last = _range(ix)
    floc = {}                                                           # :25  _RL floc(-2:+2)

    if first > +1-OLx and last < sNx+OLx:                              # :27-28
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[+0+ix, iy]                                      # :32

        floc[-1] = (floc[+0] +                                          # :34-37
                    mask[ix-1, iy]*(fbar[ix-1, iy]-floc[+0]))
        floc[+1] = (floc[+0] +
                    mask[ix+1, iy]*(fbar[ix+1, iy]-floc[+0]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy].set(floc[+1]*0.25                # :41-42
                                           - floc[-1]*0.25)

        ohat[+2] = ohat[+2].at[ix, iy].set(floc[+1]*0.25                # :44-46
                                           - floc[+0]*0.50
                                           + floc[-1]*0.25)
        return ohat
    if first != last:
        raise ValueError(f"GAD_OSC_LOC_X: the points {first}..{last} do not all take one branch of the IF on ix")

    if ix == +1-OLx:                                                    # :50
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[+0+ix, iy]                                      # :54

        floc[+1] = (floc[+0] +                                          # :56-59
                    mask[ix+1, iy]*(fbar[ix+1, iy]-floc[+0]))
        floc[+2] = (floc[+1] +
                    mask[ix+2, iy]*(fbar[ix+2, iy]-floc[+1]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy].set(floc[+1]*0.50                # :63-64
                                           - floc[+0]*0.50)

        ohat[+2] = ohat[+2].at[ix, iy].set(floc[+2]*0.25                # :66-68
                                           - floc[+1]*0.50
                                           + floc[+0]*0.25)

    if ix == sNx+OLx:                                                   # :72
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[+0+ix, iy]                                      # :76

        floc[-1] = (floc[+0] +                                          # :78-81
                    mask[ix-1, iy]*(fbar[ix-1, iy]-floc[+0]))
        floc[-2] = (floc[-1] +
                    mask[ix-2, iy]*(fbar[ix-2, iy]-floc[-1]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy].set(floc[+0]*0.50                # :85-86
                                           - floc[-1]*0.50)

        ohat[+2] = ohat[+2].at[ix, iy].set(floc[+0]*0.25                # :88-90
                                           - floc[-1]*0.50
                                           + floc[-2]*0.25)
    return ohat


def gad_osc_hat_x(kk, iy, mask, fbar, ohat, *, cfg):
    """GAD_OSC_HAT_X(bi,bj,kk,iy, mask, fbar, ohat, myThid)   @63cdc0b pkg/generic_advdiff/gad_osc_hat_x.F:103-135

    C     | OSC_HAT_X: compute WENO oscillation derivatives in X.          |

    The point loop `do ix = 1-OLx+0, sNx+OLx-0` (:126-130) is vectorised: GAD_OSC_LOC_X is called for its first
    point, its interior and its last point (the three branches of its IF on ix); each point writes only ohat(:,ix).
    Returns ohat.
    """
    sNx, OLx = cfg.sNx, cfg.OLx
    ohat = gad_osc_loc_x(1-OLx+0, mask, fbar, ohat, iy=iy, cfg=cfg)                        # :126-130
    ohat = gad_osc_loc_x(loop_i(1-OLx+1, sNx+OLx-1), mask, fbar, ohat, iy=iy, cfg=cfg)
    ohat = gad_osc_loc_x(sNx+OLx-0, mask, fbar, ohat, iy=iy, cfg=cfg)
    return ohat
