"""pkg/grdchk for the JAX-FD gradient check: which control components the Fortran checks, where they sit in our
arrays, and the finite-difference formula (host-side integer bookkeeping; no traced values except the control).

    @63cdc0b pkg/ctrl/ctrl_init_wet.F:105-126 (wet counts), pkg/grdchk/grdchk_get_mask.F:40-170,
             pkg/grdchk/grdchk_loc.F:11-365, pkg/grdchk/grdchk_main.F:202-205, 224-475 (loop, memory, gfd),
             pkg/grdchk/grdchk_getxx.F (the perturbation xx + localEps)

GRDCHK_GET_POSITION (nbeg = 0) is mitjax/pkg/grdchk/grdchk_get_position.py (GOADK lane). Only ncvargrd 'c' and
one record are ported (ALLOW_OBCS_CONTROL, ALLOW_SHELFICE undefined). The FD driver (Task 16) perturbs the control array at each
point with +-|grdchk_eps|, runs the forward model and the cost, and forms `fd_gradient`; the forward runs need the
time step, which is not part of this lane.
"""

from dataclasses import dataclass

import numpy as np


def ctrl_init_wet_nwetctile(maskC, sz):
    """nwetctile(bi,bj,k) (ctrl_init_wet.F:105-126 with jmin..jmax = 1..sNy, imin..imax = 1..sNx, :61-64):
    `maskC .ne. 0.` counted per tile and level. maskC: numpy [tile, k, j, i] with halos. -> int [tile, k]."""
    inner = maskC[:, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx]
    return (inner != 0.0).sum(axis=(2, 3)).astype(np.int64)


@dataclass
class GrdchkMask:
    nwettile: np.ndarray        # [tile, k] (iobcs = 1)
    ncvarcomp: int
    maxncvarcomps: int


def grdchk_get_mask(nwetctile, *, ncvargrd, ncvarnrmax, ncvarxmax, ncvarymax, ncvarrecs):
    """GRDCHK_GET_MASK (grdchk_get_mask.F:63-170) for ncvargrd 'c' and nobcsmax = 1: nwettile = nwetctile (:73-79);
    ncvarcomp = sum over tiles (bj, bi) and k = 1..ncvarnrmax of nwettile, times ncvarrecs (:156-170)."""
    if ncvargrd != "c":
        raise NotImplementedError(f"GRDCHK_GET_MASK: ncvargrd {ncvargrd!r} not ported")
    nt = nwetctile.shape[0]
    nwettile = np.zeros((nt, ncvarnrmax), np.int64)                     # :66-72
    nwettile[:, :] = nwetctile[:, :ncvarnrmax]                          # :73-79
    ncvarcomp = 0
    maxncvarcomps = 0
    for t in range(nt):                                                 # :158-168 (bj, bi: storage order)
        for k in range(ncvarnrmax):
            ncvarcomp = ncvarcomp + int(nwettile[t, k])
            maxncvarcomps = maxncvarcomps + ncvarxmax * ncvarymax
    return GrdchkMask(nwettile, ncvarcomp * ncvarrecs, maxncvarcomps * ncvarrecs)


@dataclass
class LocResult:
    icvrec: int
    itile: int
    jtile: int
    layer: int
    obcspos: int
    itilepos: int
    jtilepos: int
    icglom1: int
    itest: int
    ierr: int


def grdchk_loc(icomp, ichknum, mem, *, gm, maskC, sz, iLocTile, jLocTile, ncvargrd, ncvarrecs, ncvarnrmax,
               ncvarxmax, ncvarymax):
    """GRDCHK_LOC( icomp, ichknum, icvrec, itile, jtile, layer, obcspos, itilepos, jtilepos, icglom1, itest, ierr,
                   myThid )

    Literal port of the search (grdchk_loc.F:91-307): it resumes from the previous check's position (`mem`, the
    *mem arrays of GRDCHK_MAIN, :464-475) and walks the wet points of tile (iLocTile, jLocTile) in k, j, i order,
    counting with nwettile, until the icomp-th component; `goto 1234` is the return. maskC: numpy [tile, k, j, i]
    with halos (Fortran index i at i-1+OLx). Tile (bi, bj) is storage tile bi-1 + (bj-1)*nSx."""
    if ncvargrd != "c":
        raise NotImplementedError(f"GRDCHK_LOC: ncvargrd {ncvargrd!r} not ported")
    ierr = -5                                                           # :91
    icglom1 = 0                                                         # :92
    icomploc = 0                                                        # :93
    out = dict(icvrec=None, itile=None, jtile=None, layer=None, obcspos=None, itilepos=None, jtilepos=None)
    itest = None
    t_of = lambda bi, bj: bi - 1 + (bj - 1) * sz.nSx                    # noqa: E731
    mask = lambda i, j, k, bi, bj: maskC[t_of(bi, bj), k - 1, j - 1 + sz.OLy, i - 1 + sz.OLx]   # noqa: E731
    if icomp > 0:                                                       # :100
        if icomp <= gm.ncvarcomp:                                       # :101
            if ichknum == 1:                                            # :103-113
                itest = 0
                icomptest = 0
                irecwrk = 1
                kwrk = 1
                jwrk = 1
                iwrk = 1
                icglo = 0
            else:                                                       # :114-126
                p = mem[ichknum - 1]
                itest = p["itest"]
                icomptest = p["icomp"]
                irecwrk = p["icvrec"]
                kwrk = p["layer"]
                icglo = p["icglo"]
                jwrk = p["jtilepos"]
                iwrk = p["itilepos"]
                iwrk = iwrk + 1
            nobcsmax = 1                                                # :129-137 (ncvargrd .NE. 'm')
            for irec in range(irecwrk, ncvarrecs + 1):                  # :144
                iobcs = (irec - 1) % nobcsmax + 1                       # :146
                bj = jLocTile                                           # :149
                bi = iLocTile                                           # :150
                for k in range(kwrk, ncvarnrmax + 1):                   # :152
                    icglo = icglo + int(gm.nwettile[t_of(bi, bj), k - 1])        # :153
                    icglom1 = icglo - int(gm.nwettile[t_of(bi, bj), k - 1])      # :154
                    if ierr != 0 and (icglom1 < icomp <= icglo):        # :160-161
                        if icomptest == 0:                              # :170-172
                            icomptest = icglom1
                        icomploc = icomp                                # :174
                        out.update(icvrec=irec, itile=bi, jtile=bj)     # :175-177
                        for j in range(jwrk, ncvarymax + 1):            # :181
                            for i in range(iwrk, ncvarxmax + 1):        # :182
                                if ierr != 0:                           # :183
                                    if mask(i, j, k, bi, bj) > 0.0:     # :184-189
                                        icomptest = icomptest + 1
                                        itmp = i
                                        jtmp = j
                                    if icomploc == icomptest:           # :250-262
                                        out.update(itilepos=itmp, jtilepos=jtmp, layer=k, obcspos=iobcs)
                                        ierr = 0
                                        return LocResult(icglom1=icglom1, itest=itest, ierr=ierr, **out)
                            iwrk = 1                                    # :266
                        jwrk = 1                                        # :268
                    elif ierr != 0:                                     # :270
                        if icomptest == icomp - 1:                      # :271-272
                            icomptest = icomptest
                        else:                                           # :273-279
                            icomptest = 0
                            for kk in range(1, k + 1):
                                icomptest = icomptest + int(gm.nwettile[t_of(bi, bj), kk - 1])
                        iwrk = 1                                        # :281
                        jwrk = 1                                        # :282
                kwrk = 1                                                # :287
        else:                                                           # :302-326
            ierr = -4 if icomp > gm.maxncvarcomps else -3
            out = dict.fromkeys(out, -1)
            out["icvrec"] = -1
    else:                                                               # :328-352
        ierr = -2 if icomp < 0 else -1
        out = dict.fromkeys(out, -1)
    return LocResult(icglom1=icglom1, itest=itest, ierr=ierr, **out)


def grdchk_points(*, nbeg, nstep, nend, position=None, **kw):
    """The positions GRDCHK_MAIN checks (grdchk_main.F:227-242, memory :464-475): [(icomp, LocResult)].

    GOADK lane: with nbeg = 0, GRDCHK_GET_POSITION first sets nbeg and nend = nbeg + nend (grdchk_main.F:227,
    grdchk_get_position.F:188-211); `position` then holds its data.grdchk inputs (iGloPos, jGloPos, kGloPos, obcsglo,
    recglo)."""
    if nbeg == 0:                                                       # :227 (GOADK lane)
        from mitjax.pkg.grdchk.grdchk_get_position import grdchk_get_position
        if position is None:
            raise ValueError("GRDCHK_MAIN: nbeg = 0 needs `position` (iGloPos, jGloPos, kGloPos, obcsglo, recglo)")
        gp = grdchk_get_position(nbeg=nbeg, nend=nend, maskC=kw["maskC"], sz=kw["sz"], iLocTile=kw["iLocTile"],
                                 jLocTile=kw["jLocTile"], ncvargrd=kw["ncvargrd"], ncvarrecs=kw["ncvarrecs"],
                                 ncvarnrmax=kw["ncvarnrmax"], ncvarxmax=kw["ncvarxmax"], ncvarymax=kw["ncvarymax"],
                                 nwettile=kw["gm"].nwettile, **position)
        nbeg, nend = gp.nbeg, gp.nend
    mem, out = {}, []
    for icomp in range(nbeg, nend + 1, nstep):                          # :229
        ichknum = (icomp - nbeg) // nstep + 1                           # :232
        r = grdchk_loc(icomp, ichknum, mem, **kw)                       # :242
        mem[ichknum] = dict(icvrec=r.icvrec, itilepos=r.itilepos, jtilepos=r.jtilepos, layer=r.layer,
                            icomp=icomp, itest=r.itest, icglo=r.icglom1)
        out.append((icomp, r))
    return out


def storage_index(r, sz):
    """(tile, j, i) storage indices of a check point in our [tile, j, i] 2-D arrays (Fortran i at i-1+OLx)."""
    return (r.itile - 1 + (r.jtile - 1) * sz.nSx, r.jtilepos - 1 + sz.OLy, r.itilepos - 1 + sz.OLx)


def grdchk_epsfac(useCentralDiff):
    """grdchk_main.F:202-206: 2. _d 0 with central differences, else 1. _d 0."""
    return 2.0 if useCentralDiff else 1.0


def fd_gradient(fcpertplus, fcpertminus, grdchk_eps, epsfac):
    """gfd as grdchk_main.F:428-432 computes it."""
    if grdchk_eps == 0.0:
        return fcpertplus - fcpertminus
    return (fcpertplus - fcpertminus) / (epsfac * grdchk_eps)
