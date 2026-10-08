"""GAD_SOM_PREP_CS_CORNER: pkg/generic_advdiff/gad_som_prep_cs_corner.F @63cdc0b (on the cubed sphere, fill the
corner-halo regions of the volume and the SOM moments before an X or Y sweep, storing / restoring them across
passes)."""

import jax.numpy as jnp

from mitjax.pkg.generic_advdiff.gad_som_fill_cs_corner import gad_som_fill_cs_corner

CORNER_NAMES = ("SW", "SE", "NE", "NW")       # smCorners(:,:,c,:), c = 1..4 (:35: SW, SE, NE, NW)


def corner_slices(sNx, sNy, OLx, OLy):
    """The storage (j0, i0) slices of the four corner-halo blocks as :125-168 address them: smCorners(i,j,c,n) holds
    the point (i-OLx, j-OLy) (SW), (sNx+i, j-OLy) (SE), (sNx+i, sNy+j) (NE), (i-OLx, sNy+j) (NW), i = 1..OLx,
    j = 1..OLy (Fortran -> storage: index - (1-OL))."""
    lo_s, lo_n = slice(0, OLy), slice(sNy + OLy, sNy + 2*OLy)
    lo_w, lo_e = slice(0, OLx), slice(sNx + OLx, sNx + 2*OLx)
    return {"SW": (lo_s, lo_w), "SE": (lo_s, lo_e), "NE": (lo_n, lo_e), "NW": (lo_n, lo_w)}


def gad_som_prep_cs_corner(smVol, smTr0, smTr, smCorners, prep4dirX, overlapOnly, interiorOnly,
                           N_edge, S_edge, E_edge, W_edge, iPass, *, active, sNx, sNy, OLx, OLy):
    """GAD_SOM_PREP_CS_CORNER( smVol, smTr0, smTr, smCorners, prep4dirX, overlapOnly, interiorOnly,
    N_edge, S_edge, E_edge, W_edge, iPass, k, myNz, bi, bj, myThid )
    @63cdc0b pkg/generic_advdiff/gad_som_prep_cs_corner.F:6-228

    C     | o when using Cubed-Sphere Grid, fill corner-halo regions
    C     |   of all Tracer-moments with proper values
    C     smCorners  :: Temporary storage of Corner-halo-regions values
    C                   ( 3rd dim = Number of corners = 4 : SW, SE, NE, NW )
    C     prep4dirX  :: True = prepare for X direction advection
    C                   otherwise, prepare for Y direction advection.

    Every tile at once: smVol, smTr0 and the nSOM moments smTr are the raw [T, ny, nx] storage of level k (the
    caller's level sections); smCorners a dict {corner: [T, 11, OLy, OLx]} (index 0 = smVol, 1 = smTr0, 1+n = moment
    n: the Fortran's -1:nSOM); overlapOnly, interiorOnly and the edges are [T,1,1] per-tile flags; prep4dirX and
    iPass are static. `active` [T,1,1]: the tiles that make this call (gad_som_advect.F:351-362, the one addition to
    the Fortran signature); on the others nothing changes. Returns (smVol, smTr0, smTr, smCorners).

    :65-68 the corner flags; :70-173 overlapOnly: DO jPass = iPass,2 the fill for X when (jPass.EQ.2 .AND.
    prep4dirX) .OR. (jPass.EQ.1 .AND. .NOT.prep4dirX), else for Y (:81-121), then with jPass.EQ.1 the store of the
    corner blocks (:123-170); :175-222 .NOT.interiorOnly: the corner blocks written back from smCorners."""
    kw = dict(sNx=sNx, sNy=sNy, OLx=OLx, OLy=OLy)
    southWestCorner = S_edge & W_edge                                           # :65
    southEastCorner = S_edge & E_edge                                           # :66
    northEastCorner = N_edge & E_edge                                           # :67
    northWestCorner = N_edge & W_edge                                           # :68
    flags = {"SW": southWestCorner, "SE": southEastCorner, "NE": northEastCorner, "NW": northWestCorner}
    # FILL_CS_CORNER_*'s own corner flags (SW, SE, NW, NE; fill_cs_corner_tr_rl.F:75-82), on the tiles that fill
    fill_corners = jnp.stack([southWestCorner[:, 0, 0], southEastCorner[:, 0, 0], northWestCorner[:, 0, 0],
                              northEastCorner[:, 0, 0]], axis=-1)
    arrs = [smVol, smTr0] + list(smTr)
    ovl = active & overlapOnly
    for jPass in range(iPass, 3):                                               # :79  DO jPass = iPass,2
        fillX = (jPass == 2 and prep4dirX) or (jPass == 1 and not prep4dirX)    # :81-82
        filled = gad_som_fill_cs_corner(fillX, *arrs, corners=fill_corners & ovl[:, 0, 0, None],   # :84-96 / :107-119
                                        useCubedSphereExchange=True, **kw)
        arrs = list(filled)
        if jPass == 1:                                                          # :123-170
            sl = corner_slices(sNx, sNy, OLx, OLy)
            for c in CORNER_NAMES:
                js, is_ = sl[c]
                block = jnp.stack([a[:, js, is_] for a in arrs], axis=1)        # [T, 11, OLy, OLx]
                st = (ovl & flags[c])[:, :, :, None]
                smCorners = {**smCorners, c: jnp.where(st, block, smCorners[c])}
    back = active & ~overlapOnly & ~interiorOnly                                # :175
    sl = corner_slices(sNx, sNy, OLx, OLy)
    for c in CORNER_NAMES:                                                      # :178-221
        js, is_ = sl[c]
        put = (back & flags[c])[:, :, :]
        arrs = [a.at[:, js, is_].set(jnp.where(put, smCorners[c][:, m], a[:, js, is_]))
                for m, a in enumerate(arrs)]
    return arrs[0], arrs[1], arrs[2:], smCorners
