"""The index and coefficient tables of the exch2 two-component (C-grid vector) exchange: EXCH2_RX2_CUBE with
EXCH2_PUT_RX2 / EXCH2_GET_RX2, and EXCH2_UV_3D_RX / EXCH2_UV_DGRID_3D_RX which call it (pkg/exch2 @63cdc0b).

    @63cdc0b pkg/exch2/exch2_rx2_cube.template:8-320, exch2_put_rx2.template:134-330, exch2_get_rx2.template:116-165,
             exch2_uv_3d_rx.template:70-77, exch2_uv_dgrid_3d_rx.template:86-88

What the Fortran computes (one EXCH2_RX2_CUBE call = one pass): first every tile PUTs, for each neighbour entry N,
two send buffers built from its arrays as they are before the pass (exch2_rx2_cube.template:111-164, all PUTs before
any GET):
    val1 = sa1*array1(isl,jsl,k) + sa2*array2(isl,jsl,k),  sa1 = pi(1), sa2 = pj(1)     (exch2_put_rx2.template:149-153,
                                                                                        :229-230)
    val2 = sa1*array1(isl,jsl,k) + sa2*array2(isl,jsl,k),  sa1 = pi(2), sa2 = pj(2)     (:248-253, :317-318)
with pi = (pij(1), pij(2)), pj = (pij(3), pij(4)) of the sender's entry (:139-142), ABS of them when signOption is
.FALSE. (:151-154, :250-253), and the source index (isl, jsl) of the target point (itl, jtl) from the index relation
and the offsets oIs1/oJs1 (buffer 1), oIs2/oJs2 (buffer 2) of EXCH2_GET_UV_BOUNDS (:171-176, :259-264): BOTH arrays
are read at the same (isl, jsl). Then every tile GETs its neighbours' buffers in loop order, array1 over the ranges
1, array2 over the ranges 2 (exch2_rx2_cube.template:266-317, exch2_get_rx2.template:145-165): a plain copy of the
buffer. So the exchanged value is a sum of two products, not a copy: without rotation (pij = identity) the u halo is
`1*u + 0*v` and the v halo `0*u + 1*v` at the same source index, and a -0 becomes +0 where the other component there
is >= +0 (NaN where it is Inf or NaN).

This module replays those loops on the W2 topology (host side, integers only) and returns, per pass and per array,
for every point of one level of the flattened [tile, j, i] storage: `src` (flat source index, the point's own index
where the pass does not write it), `written` (bool), `sa1`, `sa2` (int8: -1, 0, 1). The model applies them as
`where(written, sa1*A1[src] + sa2*A2[src], A)` on the arrays before the pass (mitjax/eesupp/exchange.py,
sharded_exchange.py); a level-independent table serves every level (the k loop of the buffers, :167/:255, only
repeats the (i, j) pattern).

EXCH2_UV_3D_RX (exch2_uv_3d_rx.template:70-77): two passes, EXCH_IGNORE_CORNERS then EXCH_UPDATE_CORNERS, fieldCode
'Cg', the caller's withSigns; its useCubedSphereExchange corner fixes (:79-...) are not ported (raise).
EXCH2_UV_DGRID_3D_RX (exch2_uv_dgrid_3d_rx.template:86-88): EXCH2_UV_3D_RX( vPhi, uPhi, .FALSE., ... ) -- array1 is v,
array2 is u, signs off whatever the caller passes; its useCubedSphereExchange part (:91-...) is not ported.
Only the single-process path is replayed (W2_myCommFlag 'P'; 'M' raises in W2_MAP_PROCS already).
"""

import numpy as np

from mitjax.pkg.exch2.exch2_get_uv_bounds import exch2_get_uv_bounds


def _flat(layout, tile, i, j):
    L = layout
    return ((tile - 1) * L.ny + (j - 1 + L.OLy)) * L.nx + (i - 1 + L.OLx)


def _range(lo, hi, stride):
    return range(lo, hi + stride, stride)


def exch2_rx2_cube_tables(w2, layout, signOption, fieldCode, updateCorners):
    """One EXCH2_RX2_CUBE( array1, array2, signOption, fieldCode, OLx, OLx, OLy, OLy, myNz, OLx, OLy, cornerMode )
    pass as tables: {1: (src, written, sa1, sa2), 2: (...)} for array1 / array2 (numpy, [nTiles*ny*nx])."""
    L = layout
    sz = w2.size
    if (L.sNx, L.sNy, L.nTiles) != (sz.sNx, sz.sNy, w2.exch2_nTiles):
        raise ValueError(f"layout {L} does not match the W2 topology")
    myOLw = myOLe = L.OLx                                                    # exch2_uv_3d_rx.template: OLw = OLx ...
    myOLs = myOLn = L.OLy
    exchWidthX = L.OLx
    i1Lo, i1Hi, j1Lo, j1Hi = 1 - myOLw, sz.sNx + myOLe, 1 - myOLs, sz.sNy + myOLn   # exch2_rx2_cube.template:92-95
    i2Lo, i2Hi, j2Lo, j2Hi = i1Lo, i1Hi, j1Lo, j1Hi                                  # :98-101

    # ---- PUT: every tile's send buffers, from the arrays before the pass (:111-164; exch2_put_rx2.template)
    bufs = {}
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            thisTile = w2.W2_myTileList[bi, bj]
            for N in range(1, w2.exch2_nNeighbours[thisTile] + 1):
                farTile = w2.exch2_neighbourId[N, thisTile]
                oN = w2.exch2_opposingSend[N, thisTile]
                (tIlo1, tIhi1, tJlo1, tJhi1, tIlo2, tIhi2, tJlo2, tJhi2, tiStride, tjStride,
                 oIs1, oJs1, oIs2, oJs2) = exch2_get_uv_bounds(fieldCode, exchWidthX, updateCorners, farTile, oN,
                                                               w2=w2)
                tgT = w2.exch2_neighbourId[N, thisTile]                        # put_rx2:134
                itb, jtb = w2.exch2_tBasex[tgT], w2.exch2_tBasey[tgT]          # :135-136
                isb, jsb = w2.exch2_tBasex[thisTile], w2.exch2_tBasey[thisTile]   # :137-138
                pi = (w2.exch2_pij[1, N, thisTile], w2.exch2_pij[2, N, thisTile])   # :139-140
                pj = (w2.exch2_pij[3, N, thisTile], w2.exch2_pij[4, N, thisTile])   # :141-142
                out = []
                for comp, (tIlo, tIhi, tJlo, tJhi, oIs, oJs, lo_hi) in (
                        (1, (tIlo1, tIhi1, tJlo1, tJhi1, oIs1, oJs1, (i1Lo, i1Hi, j1Lo, j1Hi))),
                        (2, (tIlo2, tIhi2, tJlo2, tJhi2, oIs2, oJs2, (i2Lo, i2Hi, j2Lo, j2Hi)))):
                    sa1, sa2 = (pi[0], pj[0]) if comp == 1 else (pi[1], pj[1])    # :149-150 / :248-249
                    if not signOption:                                            # :151-154 / :250-253
                        sa1, sa2 = abs(sa1), abs(sa2)
                    entries = []
                    for jtl in _range(tJlo, tJhi, tjStride):                      # :168 / :256
                        for itl in _range(tIlo, tIhi, tiStride):                  # :169 / :257
                            itc = itl + itb
                            jtc = jtl + jtb
                            isc = pi[0] * itc + pi[1] * jtc + oIs
                            jsc = pj[0] * itc + pj[1] * jtc + oJs
                            isl = isc - isb
                            jsl = jsc - jsb
                            iLo, iHi, jLo, jHi = lo_hi
                            if not (iLo <= isl <= iHi and jLo <= jsl <= jHi):     # :201-222 (W2_E2_DEBUG_ON STOPs)
                                raise RuntimeError(f"EXCH2_PUT_RX2: source ({isl},{jsl}) out of bounds (tile "
                                                   f"{thisTile}, neighbour {N}, buffer {comp})")
                            entries.append((_flat(L, thisTile, isl, jsl), sa1, sa2))   # :229-231 / :317-319
                    out.append(entries)
                bufs[(thisTile, N)] = out

    # ---- GET: copy the buffers into the halos, tile and neighbour order (:266-317; exch2_get_rx2.template)
    n = L.npoints
    tabs = {c: [np.arange(n, dtype=np.int64), np.zeros(n, bool), np.zeros(n, np.int8), np.zeros(n, np.int8)]
            for c in (1, 2)}
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            thisTile = w2.W2_myTileList[bi, bj]
            for N in range(1, w2.exch2_nNeighbours[thisTile] + 1):
                (tIlo1, tIhi1, tJlo1, tJhi1, tIlo2, tIhi2, tJlo2, tJhi2, tiStride, tjStride,
                 _, _, _, _) = exch2_get_uv_bounds(fieldCode, exchWidthX, updateCorners, thisTile, N, w2=w2)
                soT = w2.exch2_neighbourId[N, thisTile]                         # get_rx2:116
                oNb = w2.exch2_opposingSend[N, thisTile]                        # :117
                if w2.W2_myCommFlag[N, bi, bj] != "P":                           # :121-143
                    raise NotImplementedError("EXCH2_GET_RX2: commSetting other than 'P' is not ported")
                buf = bufs[(soT, oNb)]
                for comp, (tIlo, tIhi, tJlo, tJhi) in ((1, (tIlo1, tIhi1, tJlo1, tJhi1)),
                                                       (2, (tIlo2, tIhi2, tJlo2, tJhi2))):
                    entries = buf[comp - 1]
                    k = 0
                    src, wr, s1, s2 = tabs[comp]
                    for jtl in _range(tJlo, tJhi, tjStride):                     # :147 / :158
                        for itl in _range(tIlo, tIhi, tiStride):                 # :148 / :159
                            p = _flat(L, thisTile, itl, jtl)
                            src[p], s1[p], s2[p] = entries[k]                    # :150-151 / :161-162
                            wr[p] = True
                            k += 1
                    if k != len(entries):
                        raise RuntimeError("EXCH2_GET_RX2: buffer length differs from the PUT's")
    return {c: tuple(tabs[c]) for c in (1, 2)}


def exch2_uv_3d_rx_tables(w2, layout, withSigns, useCubedSphereExchange=False):
    """EXCH2_UV_3D_RX( Uphi, Vphi, withSigns, myNz ) (exch2_uv_3d_rx.template:70-77): the two passes, corners
    ignored then updated, fieldCode 'Cg'. Returns [pass1, pass2], each {1: tables of Uphi, 2: tables of Vphi}."""
    if useCubedSphereExchange:                                               # :79-...
        raise NotImplementedError("EXCH2_UV_3D_RX: the useCubedSphereExchange corner fixes are not ported")
    return [exch2_rx2_cube_tables(w2, layout, withSigns, "Cg", False),      # :70-73 EXCH_IGNORE_CORNERS
            exch2_rx2_cube_tables(w2, layout, withSigns, "Cg", True)]       # :74-77 EXCH_UPDATE_CORNERS


def exch2_uv_dgrid_3d_rx_tables(w2, layout, useCubedSphereExchange=False):
    """EXCH2_UV_DGRID_3D_RX( uPhi, vPhi, withSigns, myNz ): EXCH2_UV_3D_RX( vPhi, uPhi, .FALSE., myNz )
    (exch2_uv_dgrid_3d_rx.template:86-88), so array1 is vPhi and array2 is uPhi, signs off whatever withSigns is."""
    if useCubedSphereExchange:                                               # :91-...
        raise NotImplementedError("EXCH2_UV_DGRID_3D_RX: the useCubedSphereExchange part is not ported")
    return exch2_uv_3d_rx_tables(w2, layout, False)
