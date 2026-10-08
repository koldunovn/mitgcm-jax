"""pkg/generic_advdiff/gad_ppm_p3e_y.F: PPM edge values in Y (GAD_PPM_P3E_Y).

Row routine (see gad_osc_hat_y.py): the row arrays (along j) carry the index i of the caller's `do ix` loop, and `ix` is that loop.
"""

from mitjax.farray import loop_j


def gad_ppm_p3e_y(kk, ix, mask, fbar, edge, *, cfg):
    """GAD_PPM_P3E_Y(bi,bj,kk,ix, mask, fbar, edge, myThid)   @63cdc0b pkg/generic_advdiff/gad_ppm_p3e_y.F:3-64

    C     | PPM_P3E_Y: approximate edge values with degree-3 polynomials.  |
    C     | Fiyed grid-spacing variant in Y.                               |

    Returns edge. The point loop (:31-59) runs on all its points at once (each point reads only inputs). `2.`,
    `1/12`, `7/12` are D literals (the quotients folded by gfortran are the correctly rounded doubles, as here).
    """
    sNy, OLy = cfg.sNy, cfg.OLy
    mloc, floc = {}, {}                                                 # :26-27  mloc(-2:+1), floc(-2:+1)

    iy = loop_j(1-OLy+2, sNy+OLy-1)                                     # :31

#     ================ mask local stencil: expand from centre outwards
    mloc[-1] = mask[ix, iy-1]                                           # :34-35
    mloc[+0] = mask[ix, iy+0]

    floc[-1] = (fbar[ix, iy+0]                                          # :37-40
                + mloc[-1]*(fbar[ix, iy-1]-fbar[ix, iy+0]))
    floc[+0] = (fbar[ix, iy-1]
                + mloc[+0]*(fbar[ix, iy+0]-fbar[ix, iy-1]))

    mloc[-2] = mask[ix, iy-2] * mloc[-1]                                # :42

    ftmp = 2. * floc[-1] - floc[+0]                                     # :44-46
    floc[-2] = (ftmp
                + mloc[-2]*(fbar[ix, iy-2]-ftmp))

    mloc[+1] = mask[ix, iy+1] * mloc[+0]                                # :48

    ftmp = 2. * floc[+0] - floc[-1]                                     # :50-52
    floc[+1] = (ftmp
                + mloc[+1]*(fbar[ix, iy+1]-ftmp))

#     ================ centred, 3rd-order interpolation for edge value
    edge = edge.at[ix, iy].set(                                         # :55-57
           -(1. / 12.)*(floc[-2]+floc[+1])
           + (7. / 12.)*(floc[-1]+floc[+0]))
    return edge
