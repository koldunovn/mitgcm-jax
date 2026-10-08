"""pkg/generic_advdiff/gad_ppm_p3e_r.F: PPM edge values in R (GAD_PPM_P3E_R).

Column routine (see gad_osc_hat_r.py): the column arrays carry the column indices and `ix, iy` are the caller's
column loops.
"""

from mitjax.farray import loops_kji


def gad_ppm_p3e_r(ix, iy, mask, fbar, edge, *, cfg):
    """GAD_PPM_P3E_R(bi,bj,ix,iy, mask, fbar, edge, myThid)   @63cdc0b pkg/generic_advdiff/gad_ppm_p3e_r.F:3-63

    C     | PPM_P3E_R: approximate edge values with degree-3 polynomials.  |
    C     | Fixed grid-spacing variant in R.                               |

    Returns edge (declared (1-0:Nr+1) per column). The level loop (:30-58) is vectorised with the column loops
    (each level reads only inputs). Literals as GAD_PPM_P3E_X.
    """
    Nr = cfg.Nr
    mloc, floc = {}, {}                                                 # :26-27  mloc(-2:+1), floc(-2:+1)

    ir, iy, ix = loops_kji((+1, Nr+1), (iy.first, iy.last), (ix.first, ix.last))   # :30

#     ================ mask local stencil: expand from centre outwards
    mloc[-1] = mask[ix, iy, ir-1]                                       # :33-34
    mloc[+0] = mask[ix, iy, ir+0]

    floc[-1] = (fbar[ix, iy, ir+0]                                      # :36-39
                + mloc[-1]*(fbar[ix, iy, ir-1]-fbar[ix, iy, ir+0]))
    floc[+0] = (fbar[ix, iy, ir-1]
                + mloc[+0]*(fbar[ix, iy, ir+0]-fbar[ix, iy, ir-1]))

    mloc[-2] = mask[ix, iy, ir-2] * mloc[-1]                            # :41

    ftmp = 2. * floc[-1] - floc[+0]                                     # :43-45
    floc[-2] = (ftmp
                + mloc[-2]*(fbar[ix, iy, ir-2]-ftmp))

    mloc[+1] = mask[ix, iy, ir+1] * mloc[+0]                            # :47

    ftmp = 2. * floc[+0] - floc[-1]                                     # :49-51
    floc[+1] = (ftmp
                + mloc[+1]*(fbar[ix, iy, ir+1]-ftmp))

#     ================ centred, 3rd-order interpolation for edge value
    edge = edge.at[ix, iy, ir].set(                                     # :54-56
           -(1. / 12.)*(floc[-2]+floc[+1])
           + (7. / 12.)*(floc[-1]+floc[+0]))
    return edge
