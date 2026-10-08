"""pkg/generic_advdiff/gad_osc_hat_y.F: WENO oscillation derivatives in Y.

    o GAD_OSC_LOC_Y   ported
    o GAD_OSC_HAT_Y   ported

Row routines: the Fortran works on one row ix of the caller's `do ix` loop (row arrays `(1-OLy:sNy+OLy)`); here all
rows at once (each row reads and writes only itself), so the row arrays carry the index i of that loop (they are declared
like the caller's 2-D fields, `(1-OLx:sNx+OLx, 1-OLy:sNy+OLy)`) and `ix` is the caller's row loop. `ohat(1:2, :)` is
{1: ohat(1,:), 2: ohat(2,:)}.
"""

from mitjax.farray import Loop, loop_j


def _range(iy):
    return (iy.first, iy.last) if isinstance(iy, Loop) else (iy, iy)


def gad_osc_loc_y(iy, mask, fbar, ohat, *, ix, cfg):
    """GAD_OSC_LOC_Y(iy, mask, fbar, ohat)   @63cdc0b pkg/generic_advdiff/gad_osc_hat_y.F:10-99

    `iy` is one point (an int) or a loop over points that all take the same branch of the IF on iy (:27-28, :50,
    :72); `ix` (keyword; not a Fortran argument) is the caller's row loop. Returns ohat. `0.25`, `0.50` are D
    literals.
    """
    sNy, OLy = cfg.sNy, cfg.OLy
    ohat = dict(ohat)
    first, last = _range(iy)
    floc = {}                                                           # :25  _RL floc(-2:+2)

    if first > +1-OLy and last < sNy+OLy:                              # :27-28
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[ix, +0+iy]                                      # :32

        floc[-1] = (floc[+0] +                                          # :34-37
                    mask[ix, iy-1]*(fbar[ix, iy-1]-floc[+0]))
        floc[+1] = (floc[+0] +
                    mask[ix, iy+1]*(fbar[ix, iy+1]-floc[+0]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy].set(floc[+1]*0.25                # :41-42
                                           - floc[-1]*0.25)

        ohat[+2] = ohat[+2].at[ix, iy].set(floc[+1]*0.25                # :44-46
                                           - floc[+0]*0.50
                                           + floc[-1]*0.25)
        return ohat
    if first != last:
        raise ValueError(f"GAD_OSC_LOC_Y: the points {first}..{last} do not all take one branch of the IF on iy")

    if iy == +1-OLy:                                                    # :50
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[ix, +0+iy]                                      # :54

        floc[+1] = (floc[+0] +                                          # :56-59
                    mask[ix, iy+1]*(fbar[ix, iy+1]-floc[+0]))
        floc[+2] = (floc[+1] +
                    mask[ix, iy+2]*(fbar[ix, iy+2]-floc[+1]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy].set(floc[+1]*0.50                # :63-64
                                           - floc[+0]*0.50)

        ohat[+2] = ohat[+2].at[ix, iy].set(floc[+2]*0.25                # :66-68
                                           - floc[+1]*0.50
                                           + floc[+0]*0.25)

    if iy == sNy+OLy:                                                   # :72
#     ================ mask local stencil: expand from centre outwards
        floc[+0] = fbar[ix, +0+iy]                                      # :76

        floc[-1] = (floc[+0] +                                          # :78-81
                    mask[ix, iy-1]*(fbar[ix, iy-1]-floc[+0]))
        floc[-2] = (floc[-1] +
                    mask[ix, iy-2]*(fbar[ix, iy-2]-floc[-1]))

#     ================ calc. 1st & 2nd derivatives over masked stencil
        ohat[+1] = ohat[+1].at[ix, iy].set(floc[+0]*0.50                # :85-86
                                           - floc[-1]*0.50)

        ohat[+2] = ohat[+2].at[ix, iy].set(floc[+0]*0.25                # :88-90
                                           - floc[-1]*0.50
                                           + floc[-2]*0.25)
    return ohat


def gad_osc_hat_y(kk, ix, mask, fbar, ohat, *, cfg):
    """GAD_OSC_HAT_Y(bi,bj,kk,ix, mask, fbar, ohat, myThid)   @63cdc0b pkg/generic_advdiff/gad_osc_hat_y.F:103-135

    C     | OSC_HAT_Y: compute WENO oscillation derivatives in Y.          |

    The point loop `do iy = 1-OLy+0, sNy+OLy-0` (:126-130) is vectorised: GAD_OSC_LOC_Y is called for its first
    point, its interior and its last point (the three branches of its IF on iy); each point writes only ohat(:,iy).
    Returns ohat.
    """
    sNy, OLy = cfg.sNy, cfg.OLy
    ohat = gad_osc_loc_y(1-OLy+0, mask, fbar, ohat, ix=ix, cfg=cfg)                        # :126-130
    ohat = gad_osc_loc_y(loop_j(1-OLy+1, sNy+OLy-1), mask, fbar, ohat, ix=ix, cfg=cfg)
    ohat = gad_osc_loc_y(sNy+OLy-0, mask, fbar, ohat, ix=ix, cfg=cfg)
    return ohat
