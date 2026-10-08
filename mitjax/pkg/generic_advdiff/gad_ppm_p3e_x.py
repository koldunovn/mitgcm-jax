"""pkg/generic_advdiff/gad_ppm_p3e_x.F: PPM edge values in X (GAD_PPM_P3E_X).

Row routine (see gad_osc_hat_x.py): the row arrays carry the row index j and `iy` is the caller's row loop.
"""

from mitjax.farray import loop_i


def gad_ppm_p3e_x(kk, iy, mask, fbar, edge, *, cfg):
    """GAD_PPM_P3E_X(bi,bj,kk,iy, mask, fbar, edge, myThid)   @63cdc0b pkg/generic_advdiff/gad_ppm_p3e_x.F:3-63

    C     | PPM_P3E_X: approximate edge values with degree-3 polynomials.  |
    C     | Fixed grid-spacing variant in X.                               |

    Returns edge. The point loop (:30-58) runs on all its points at once (each point reads only inputs). `2.`,
    `1/12`, `7/12` are D literals (the quotients folded by gfortran are the correctly rounded doubles, as here).
    """
    sNx, OLx = cfg.sNx, cfg.OLx
    mloc, floc = {}, {}                                                 # :26-27  mloc(-2:+1), floc(-2:+1)

    ix = loop_i(1-OLx+2, sNx+OLx-1)                                     # :30

#     ================ mask local stencil: expand from centre outwards
    mloc[-1] = mask[ix-1, iy]                                           # :33-34
    mloc[+0] = mask[ix+0, iy]

    floc[-1] = (fbar[ix+0, iy]                                          # :36-39
                + mloc[-1]*(fbar[ix-1, iy]-fbar[ix+0, iy]))
    floc[+0] = (fbar[ix-1, iy]
                + mloc[+0]*(fbar[ix+0, iy]-fbar[ix-1, iy]))

    mloc[-2] = mask[ix-2, iy] * mloc[-1]                                # :41

    ftmp = 2. * floc[-1] - floc[+0]                                     # :43-45
    floc[-2] = (ftmp
                + mloc[-2]*(fbar[ix-2, iy]-ftmp))

    mloc[+1] = mask[ix+1, iy] * mloc[+0]                                # :47

    ftmp = 2. * floc[+0] - floc[-1]                                     # :49-51
    floc[+1] = (ftmp
                + mloc[+1]*(fbar[ix+1, iy]-ftmp))

#     ================ centred, 3rd-order interpolation for edge value
    edge = edge.at[ix, iy].set(                                         # :54-56
           -(1. / 12.)*(floc[-2]+floc[+1])
           + (7. / 12.)*(floc[-1]+floc[+0]))
    return edge
