"""GRDCHK_GET_POSITION: pkg/grdchk/grdchk_get_position.F @63cdc0b (GOADK lane, M2: global_ocean.90x40x15/input_ad*,
data.grdchk without nbeg, i.e. nbeg = 0 from grdchk_readparms.F:82)."""

from dataclasses import dataclass


@dataclass
class GetPosition:
    nbeg: int
    nend: int
    ierr: int
    lines: list            # the STDOUT lines the routine writes (standardMessageUnit), in order


def grdchk_get_position(*, nbeg, nend, maskC, sz, iLocTile, jLocTile, iGloPos, jGloPos, kGloPos, obcsglo, recglo,
                        ncvargrd, ncvarrecs, ncvarnrmax, ncvarxmax, ncvarymax, nwettile, myProcId=0,
                        grdchkwhichproc=0):
    """GRDCHK_GET_POSITION( myThid )   @63cdc0b pkg/grdchk/grdchk_get_position.F:9-243

    Host-side integer bookkeeping (no traced value). Finds the packed control index of the grid point
    (iGloPos, jGloPos, kGloPos) of tile (iLocTile, jLocTile) by counting the wet points of the control's mask in
    irec, k, j, i order, and sets nbeg to it and `nend = nbeg + nend` (:188-211); when the point itself is dry the
    next wet point is taken ('closest next position'). Returns the new nbeg, nend, ierr and the two STDOUT lines.

    maskC: numpy [tile, k, j, i] with halos (Fortran i at i-1+OLx); tile (bi, bj) is storage tile bi-1 + (bj-1)*nSx.
    nwettile: [tile, k] (GRDCHK_GET_MASK; read only by the ELSEIF of :219-223, which cannot run: it is the ELSE of
    `IF ( ierr .ne. 0 )` guarded by `ierr .NE. 0` again; ported as written).

    Ported for this build (ALLOW_OBCS_CONTROL, ALLOW_SHELFICE undefined): ncvargrd 'c' (maskC), 's' and 'w' raise
    (not executed: the M2 controls are all 'c'); ncvargrd 'm' prints 'Ooops' in the Fortran (:92-97) and raises here.
    The loop `GOTO 1234` is a return. myProcId / grdchkwhichproc: single process (0; grdchk_readparms.F:140-145).
    """
    if ncvargrd != "c":
        raise NotImplementedError(f"GRDCHK_GET_POSITION: ncvargrd {ncvargrd!r} is not ported")
    t_of = lambda bi, bj: bi - 1 + (bj - 1) * sz.nSx               # noqa: E731
    itile = iLocTile                                                # :64-70
    jtile = jLocTile
    itilepos = iGloPos
    jtilepos = jGloPos
    layer = kGloPos
    obcspos = obcsglo
    icvrec = recglo
    lines = []
    ierr = None
    if myProcId == grdchkwhichproc:                                 # :77
        ierr = -5                                                   # :80-89
        pastit = -1
        wetlocal = 0.
        itest = 0
        icomptest = 0
        irecwrk = 1
        kwrk = 1
        jwrk = 1
        iwrk = 1
        nobcsmax = 1                                                # :92-100 (ncvargrd .NE. 'm')
        for irec in range(irecwrk, ncvarrecs + 1):                  # :103
            iobcs = (irec - 1) % nobcsmax + 1                       # :104
            bi = itile                                              # :105-106
            bj = jtile
            for k in range(kwrk, ncvarnrmax + 1):                   # :107
                if ierr != 0:                                       # :112
                    for j in range(jwrk, ncvarymax + 1):            # :113
                        for i in range(iwrk, ncvarxmax + 1):        # :114
                            if ierr != 0:                           # :115
                                m = float(maskC[t_of(bi, bj), k - 1, j - 1 + sz.OLy, i - 1 + sz.OLx])
                                if m > 0.:                          # :117-119
                                    icomptest = icomptest + 1
                                wetlocal = m                        # :120
                                if (i == itilepos and j == jtilepos and k == layer and bi == itile
                                        and bj == jtile and iobcs == obcspos and irec == icvrec):   # :181-187
                                    pastit = 0                      # :188
                                    if wetlocal != 0:               # :189-200
                                        nbeg = icomptest
                                        nend = nbeg + nend
                                        ierr = 0
                                        lines.append(" grad-res exact position met: ")
                                        lines.append(" grad-res " + "".join(f"{v:5d}" for v in (
                                            grdchkwhichproc, nbeg, itilepos, jtilepos, layer, itile, jtile)))
                                        return GetPosition(nbeg, nend, ierr, lines)
                                elif pastit == 0 and wetlocal != 0:   # :201-212
                                    nbeg = icomptest
                                    nend = nbeg + nend
                                    ierr = 0
                                    lines.append(" grad-res closest next position: ")
                                    lines.append(" grad-res " + "".join(f"{v:5d}" for v in (
                                        grdchkwhichproc, nbeg, itilepos, jtilepos, layer, itile, jtile)))
                                    return GetPosition(nbeg, nend, ierr, lines)
                        iwrk = 1                                    # :216
                    jwrk = 1                                        # :218
                elif ierr != 0:                                     # :219-223 (unreachable, ported as written)
                    itest = itest + int(nwettile[t_of(bi, bj), k - 1])
                    iwrk = 1
                    jwrk = 1
    return GetPosition(nbeg, nend, ierr, lines)                     # :234  1234 CONTINUE
