"""pkg/generic_advdiff/gad_pqm_p5e_x.F: PQM edge values and slopes in X (GAD_PQM_P5E_X).

Row routine (see gad_osc_hat_x.py): the row arrays carry the row index j and `iy` is the caller's row loop.
"""

from mitjax.farray import loop_i


def gad_pqm_p5e_x(kk, iy, mask, fbar, edge, *, cfg, grid):
    """GAD_PQM_P5E_X(bi,bj,kk,iy, mask, fbar, edge, myThid)   @63cdc0b pkg/generic_advdiff/gad_pqm_p5e_x.F:3-79

    C     | PQM_P5E_X: approximate edge values with degree-5 polynomials.  |
    C     | Fixed grid-spacing variant in X.                               |
    C       edge      :: EDGE(1,:) = VALUE, EDGE(2,:) = DF/DX

    Returns edge = {1: edge(1,:), 2: edge(2,:)}. The point loop (:31-74) runs on all its points at once (each point
    reads only inputs). All fractions are D literals (the quotients folded by gfortran are the correctly rounded
    doubles, as here).
    """
    sNx, OLx = cfg.sNx, cfg.OLx
    recip_dxC = grid.recip_dxC
    edge = dict(edge)
    mloc, floc = {}, {}                                                 # :27-28  mloc(-3:+2), floc(-3:+2)

    ix = loop_i(1-OLx+3, sNx+OLx-2)                                     # :31

#     ================ mask local stencil: expand from centre outwards
    mloc[-1] = mask[ix-1, iy]                                           # :34-35
    mloc[+0] = mask[ix+0, iy]

    floc[-1] = (fbar[ix+0, iy]                                          # :37-40
                + mloc[-1]*(fbar[ix-1, iy]-fbar[ix+0, iy]))
    floc[+0] = (fbar[ix-1, iy]
                + mloc[+0]*(fbar[ix+0, iy]-fbar[ix-1, iy]))

    mloc[-2] = mask[ix-2, iy] * mloc[-1]                                # :42-43
    mloc[-3] = mask[ix-3, iy] * mloc[-2]

    ftmp = 2. * floc[-1] - floc[+0]                                     # :45-50
    floc[-2] = (ftmp
                + mloc[-2]*(fbar[ix-2, iy]-ftmp))
    ftmp = 2. * floc[-2] - floc[-1]
    floc[-3] = (ftmp
                + mloc[-3]*(fbar[ix-3, iy]-ftmp))

    mloc[+1] = mask[ix+1, iy] * mloc[+0]                                # :52-53
    mloc[+2] = mask[ix+2, iy] * mloc[+1]

    ftmp = 2. * floc[+0] - floc[-1]                                     # :55-60
    floc[+1] = (ftmp
                + mloc[+1]*(fbar[ix+1, iy]-ftmp))
    ftmp = 2. * floc[+1] - floc[+0]
    floc[+2] = (ftmp
                + mloc[+2]*(fbar[ix+2, iy]-ftmp))

#     ================ centred, 5th-order interpolation for edge value
    edge[1] = edge[1].at[ix, iy].set(                                   # :63-66
        + (1. / 60.)*(floc[-3]+floc[+2])
        - (8. / 60.)*(floc[-2]+floc[+1])
        + (37. / 60.)*(floc[-1]+floc[+0]))

    edge[2] = edge[2].at[ix, iy].set((                                  # :68-72
        - (1. / 90.)*(floc[-3]-floc[+2])
        + (5. / 36.)*(floc[-2]-floc[+1])
        - (49. / 36.)*(floc[-1]-floc[+0])
                                     ) * recip_dxC[ix, iy])
    return edge
