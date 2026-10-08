"""The cubed-sphere pass structure of GAD_ADVECTION (pkg/generic_advdiff/gad_advection.F @63cdc0b): per tile and
advection pass, which direction is advected, which part of the local tracer is updated, and the FILL_CS_CORNER_*
calls in their order. Host-side, static (a function of the facet number and the tile's facet-edge flags); written
for the ADVECT lane's GAD_ADVECTION on the cube (lane B, plan Task 22 item (c)), which today raises on
useCubedSphereExchange (gad_advection.py).

    @63cdc0b pkg/generic_advdiff/gad_advection.F:252-275 (npass, nCFace, edges), :334-338 (FILL_CS_CORNER_UV_RS of
             maskLocW, maskLocS once per level), :349-370 (pass flags), :384-395 / :459-462 (X fills),
             :605-616 / :680-683 (Y fills)

    sched = gad_cs_pass_schedule(nCFace, N_edge, S_edge, E_edge, W_edge)      # npass = 3 passes
    sched[ipass-1] = {"overlapOnly", "interiorOnly", "calc_fluxes_X", "calc_fluxes_Y",
                      "events": (("fill", 1), ("flux", "X"), ("fill", 2), ("update", "X"), ...)}

Events, per pass, in the Fortran order (the k loop around the passes is the caller's; the FILL_CS_CORNER_UV_RS of the
masks, :334-338, precedes the passes of every level and is not part of a pass):
  X block (IF (calc_fluxes_X), :384): when `.NOT.overlapOnly .OR. N_edge .OR. S_edge` (:389): ("fill", 1) if
  overlapOnly (:392-395, FILL_CS_CORNER_TR_RL( 1, .FALSE., localTij )), ("flux", "X"), ("fill", 2) if
  overlapOnly .AND. ipass.EQ.1 (:459-462); then always ("update", "X") (the localTij update of :474-591, its ranges
  from overlapOnly / interiorOnly and the edges).
  Y block (IF (calc_fluxes_Y), :605) likewise with `.NOT.overlapOnly .OR. E_edge .OR. W_edge` (:610),
  ("fill", 2) before (:613-616) and ("fill", 1) after for ipass 1 (:680-683), then ("update", "Y").
withSigns is .FALSE. in every fill of GAD_ADVECTION. Gate: test_cube.py::test_gad_cs_pass_schedule (a gfortran
build of the Fortran statements of :349-370 and the fill conditions, extracted verbatim by line number from the
pinned file, run for every facet, edge combination and pass).
"""

from mitjax.pkg.exch2.w2_exch2_h import imod

NPASS_CS = 3                       # gad_advection.F:253  npass = 3 (useCubedSphereExchange)


def gad_cs_pass_flags(nCFace, ipass):
    """(overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y) of pass `ipass` on facet nCFace
    (gad_advection.F:347-365, the useCubedSphereExchange arm)."""
    interiorOnly = False                                                      # :347
    overlapOnly = False                                                       # :348
    if ipass == 1:                                                            # :351
        overlapOnly = imod(nCFace, 3) == 0                                    # :352
        interiorOnly = imod(nCFace, 3) != 0                                   # :353
        calc_fluxes_X = nCFace == 6 or nCFace == 1 or nCFace == 2             # :354
        calc_fluxes_Y = nCFace == 3 or nCFace == 4 or nCFace == 5             # :355
    elif ipass == 2:                                                          # :356
        overlapOnly = imod(nCFace, 3) == 2                                    # :357
        interiorOnly = imod(nCFace, 3) == 1                                   # :358
        calc_fluxes_X = nCFace == 2 or nCFace == 3 or nCFace == 4             # :359
        calc_fluxes_Y = nCFace == 5 or nCFace == 6 or nCFace == 1             # :360
    else:
        interiorOnly = True                                                   # :362
        calc_fluxes_X = nCFace == 5 or nCFace == 6                            # :363
        calc_fluxes_Y = nCFace == 2 or nCFace == 3                            # :364
    return overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y


def gad_cs_pass_schedule(nCFace, N_edge, S_edge, E_edge, W_edge):
    """The three passes of one tile (module docstring)."""
    out = []
    for ipass in range(1, NPASS_CS + 1):
        overlapOnly, interiorOnly, cX, cY = gad_cs_pass_flags(nCFace, ipass)
        ev = []
        if cX:                                                                # :384
            if (not overlapOnly) or N_edge or S_edge:                         # :389
                if overlapOnly:                                               # :392
                    ev.append(("fill", 1))                                    # :393
                ev.append(("flux", "X"))
                if overlapOnly and ipass == 1:                                # :459
                    ev.append(("fill", 2))                                    # :460
            ev.append(("update", "X"))
        if cY:                                                                # :605
            if (not overlapOnly) or E_edge or W_edge:                         # :610
                if overlapOnly:                                               # :613
                    ev.append(("fill", 2))                                    # :614
                ev.append(("flux", "Y"))
                if overlapOnly and ipass == 1:                                # :680
                    ev.append(("fill", 1))                                    # :681
            ev.append(("update", "Y"))
        out.append({"overlapOnly": overlapOnly, "interiorOnly": interiorOnly, "calc_fluxes_X": cX,
                    "calc_fluxes_Y": cY, "events": tuple(ev)})
    return out


def tile_schedules(w2):
    """[nTiles] schedules of the W2 topology's tiles, tile order W2_myTileList(bi,bj) (gad_advection.F:254-259:
    nCFace = exch2_myFace, the edges from exch2_is*edge)."""
    sz = w2.size
    out = []
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            t = w2.W2_myTileList[bi, bj]
            out.append(gad_cs_pass_schedule(w2.exch2_myFace[t], w2.exch2_isNedge[t] == 1,
                                            w2.exch2_isSedge[t] == 1, w2.exch2_isEedge[t] == 1,
                                            w2.exch2_isWedge[t] == 1))
    return out
