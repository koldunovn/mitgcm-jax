"""pkg/generic_advdiff/gad_pqm_p5e_y.F: PQM edge values and slopes in Y (GAD_PQM_P5E_Y).

Row routine (see gad_osc_hat_y.py): the row arrays (along j) carry the index i of the caller's `do ix` loop, and `ix` is that loop.
"""

from mitjax.farray import loop_j


def gad_pqm_p5e_y(kk, ix, mask, fbar, edge, *, cfg, grid):
    """GAD_PQM_P5E_Y(bi,bj,kk,ix, mask, fbar, edge, myThid)   @63cdc0b pkg/generic_advdiff/gad_pqm_p5e_y.F:3-80

    C     | PQM_P5E_Y: approximate edge values with degree-5 polynomials.  |
    C     | Fiyed grid-spacing variant in Y.                               |
    C       edge      :: EDGE(1,:) = VALUE, EDGE(2,:) = DF/DY

    Returns edge = {1: edge(1,:), 2: edge(2,:)}. The point loop (:32-75) runs on all its points at once (each point
    reads only inputs). All fractions are D literals (the quotients folded by gfortran are the correctly rounded
    doubles, as here).
    """
    sNy, OLy = cfg.sNy, cfg.OLy
    recip_dyC = grid.recip_dyC
    edge = dict(edge)
    mloc, floc = {}, {}                                                 # :27-28  mloc(-3:+2), floc(-3:+2)

    iy = loop_j(1-OLy+3, sNy+OLy-2)                                     # :32

#     ================ mask local stencil: expand from centre outwards
    mloc[-1] = mask[ix, iy-1]                                           # :35-36
    mloc[+0] = mask[ix, iy+0]

    floc[-1] = (fbar[ix, iy+0]                                          # :38-41
                + mloc[-1]*(fbar[ix, iy-1]-fbar[ix, iy+0]))
    floc[+0] = (fbar[ix, iy-1]
                + mloc[+0]*(fbar[ix, iy+0]-fbar[ix, iy-1]))

    mloc[-2] = mask[ix, iy-2] * mloc[-1]                                # :43-44
    mloc[-3] = mask[ix, iy-3] * mloc[-2]

    ftmp = 2. * floc[-1] - floc[+0]                                     # :46-51
    floc[-2] = (ftmp
                + mloc[-2]*(fbar[ix, iy-2]-ftmp))
    ftmp = 2. * floc[-2] - floc[-1]
    floc[-3] = (ftmp
                + mloc[-3]*(fbar[ix, iy-3]-ftmp))

    mloc[+1] = mask[ix, iy+1] * mloc[+0]                                # :53-54
    mloc[+2] = mask[ix, iy+2] * mloc[+1]

    ftmp = 2. * floc[+0] - floc[-1]                                     # :56-61
    floc[+1] = (ftmp
                + mloc[+1]*(fbar[ix, iy+1]-ftmp))
    ftmp = 2. * floc[+1] - floc[+0]
    floc[+2] = (ftmp
                + mloc[+2]*(fbar[ix, iy+2]-ftmp))

#     ================ centred, 5th-order interpolation for edge value
    edge[1] = edge[1].at[ix, iy].set(                                   # :64-67
        + (1. / 60.)*(floc[-3]+floc[+2])
        - (8. / 60.)*(floc[-2]+floc[+1])
        + (37. / 60.)*(floc[-1]+floc[+0]))

    edge[2] = edge[2].at[ix, iy].set((                                  # :69-73
        - (1. / 90.)*(floc[-3]-floc[+2])
        + (5. / 36.)*(floc[-2]-floc[+1])
        - (49. / 36.)*(floc[-1]-floc[+0])
                                     ) * recip_dyC[ix, iy])
    return edge
