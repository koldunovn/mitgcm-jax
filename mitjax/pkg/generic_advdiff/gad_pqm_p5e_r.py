"""pkg/generic_advdiff/gad_pqm_p5e_r.F: PQM edge values and slopes in R (GAD_PQM_P5E_R).

Column routine (see gad_osc_hat_r.py): the column arrays carry the column indices and `ix, iy` are the caller's
column loops.
"""

from mitjax.farray import loops_kji


def gad_pqm_p5e_r(ix, iy, mask, fbar, edge, *, cfg, grid):
    """GAD_PQM_P5E_R(bi,bj,ix,iy, mask, fbar, edge, myThid)   @63cdc0b pkg/generic_advdiff/gad_pqm_p5e_r.F:3-79

    C     | PQM_P5E_R: approximate edge values with degree-5 polynomials.  |
    C     | Fixed grid-spacing variant in R.                               |

    Returns edge = {1: edge(1,:), 2: edge(2,:)} (declared (1-0:Nr+1) per column). The level loop (:31-74) is
    vectorised with the column loops (each level reads only inputs). Literals as GAD_PQM_P5E_X.
    """
    Nr = cfg.Nr
    recip_drC = grid.recip_drC
    edge = dict(edge)
    mloc, floc = {}, {}                                                 # :27-28  mloc(-3:+2), floc(-3:+2)

    ir, iy, ix = loops_kji((+1, Nr+1), (iy.first, iy.last), (ix.first, ix.last))   # :31

#     ================ mask local stencil: expand from centre outwards
    mloc[-1] = mask[ix, iy, ir-1]                                       # :34-35
    mloc[+0] = mask[ix, iy, ir+0]

    floc[-1] = (fbar[ix, iy, ir+0]                                      # :37-40
                + mloc[-1]*(fbar[ix, iy, ir-1]-fbar[ix, iy, ir+0]))
    floc[+0] = (fbar[ix, iy, ir-1]
                + mloc[+0]*(fbar[ix, iy, ir+0]-fbar[ix, iy, ir-1]))

    mloc[-2] = mask[ix, iy, ir-2] * mloc[-1]                            # :42-43
    mloc[-3] = mask[ix, iy, ir-3] * mloc[-2]

    ftmp = 2. * floc[-1] - floc[+0]                                     # :45-50
    floc[-2] = (ftmp
                + mloc[-2]*(fbar[ix, iy, ir-2]-ftmp))
    ftmp = 2. * floc[-2] - floc[-1]
    floc[-3] = (ftmp
                + mloc[-3]*(fbar[ix, iy, ir-3]-ftmp))

    mloc[+1] = mask[ix, iy, ir+1] * mloc[+0]                            # :52-53
    mloc[+2] = mask[ix, iy, ir+2] * mloc[+1]

    ftmp = 2. * floc[+0] - floc[-1]                                     # :55-60
    floc[+1] = (ftmp
                + mloc[+1]*(fbar[ix, iy, ir+1]-ftmp))
    ftmp = 2. * floc[+1] - floc[+0]
    floc[+2] = (ftmp
                + mloc[+2]*(fbar[ix, iy, ir+2]-ftmp))

#     ================ centred, 5th-order interpolation for edge value
    edge[1] = edge[1].at[ix, iy, ir].set(                               # :63-66
        + (1. / 60.)*(floc[-3]+floc[+2])
        - (8. / 60.)*(floc[-2]+floc[+1])
        + (37. / 60.)*(floc[-1]+floc[+0]))

    edge[2] = edge[2].at[ix, iy, ir].set((                              # :68-72
        - (1. / 90.)*(floc[-3]-floc[+2])
        + (5. / 36.)*(floc[-2]-floc[+1])
        - (49. / 36.)*(floc[-1]-floc[+0])
                                         ) * recip_drC[ir])
    return edge
