"""COST_GENCOST_ALL with COST_GENERIC, COST_GENLOOP, COST_GENCAL, COST_GENREAD   @63cdc0b
pkg/ecco/cost_gencost_all.F:3-101, cost_generic.F:12-141, :147-518, cost_gencal.F:7-121, cost_genread.F:7-108
(lane M4ADCOL session 3)

COST_DRIVER calls COST_GENCOST_ALL under useECCO (cost_driver.F:44-53) after the loop. Split as the rest of the
port: `gencost_setup` is the host part (file names, records, the reads of the data and uncertainty files, which do
not depend on the model state: COST_GENCAL, READ_REC_LEV_RL of the data record, ECCO_READWEI); `cost_gencost_all`
is the traced part on the bar records the run carried (ECCO_READBAR, ECCO_MULT, ECCO_DIFFMSK, ECCO_ADDCOST), and
returns the misfit fields the host writes (`write_misfits`, WRITE_REC_XYZ_RL :448-455).

Ported for the gencost of the builds: flag 1, 3-D (nnzbar = nnzobs = Nr) or 2-D (nnzbar = nnzobs = 1, :60-66; lane
M4ADLAB session 4), ylocmask 'c' (:75; cost_generic.F:312-318: localmask = maskC with overlaps), and of the
pre-processing (cost_generic.F:271-304) 'mean' (domean), 'offset' (dooffset) and 'mindepth' (domaskmindepth with
topomin = gencost_preproc_r) -- anything else raises (doanom, dovarwei, nosumsq, smooth, clim, factor). Without
'mean' COST_GENERIC makes the one COST_GENLOOP call of :123-139 with addVariaCost = .TRUE. (the record misfits); with
it, the one call of :107-121 with addVariaCost = .FALSE. (the misfit of the record mean, :472-503). After the record
loop COST_GENLOOP fills localdifmeanOut and localdifmsk (:472-477, ECCO_CP / ECCO_DIVFIELD of localdifsum and
localdifnum, zero without domean): without domean they land in COST_GENERIC's local localdifmean1, which nothing reads
before its return -- computed only with domean. COST_GENCOST_GLBMEAN (empty), _BOXMEAN (flag -3), _BPV4 ('bpv4-grace'),
_MOC (flag -5) do nothing for these gencost (the flags they need raise in ecco_readparms.py; bpv4-grace raises here);
using_cost_transp / altim / sst / seaice are .FALSE. (:93-96).
"""

import os

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray, loops_kji
from mitjax.pkg.cal.cal_h import idiv, imod
from mitjax.pkg.ecco.ecco_readparms import _eq
from mitjax.pkg.ecco.ecco_toolbox import (ecco_addcost, ecco_addmask, ecco_cp, ecco_diffmsk, ecco_divfield,
                                          ecco_maskmindepth, ecco_mult, ecco_offset, ecco_readwei)


def cost_gencal(localbarfile, localobsfile, irec, localstartdate, localperiod, *, eccoiter, cal, dTtracerLev1,
                rundir):
    """cost_gencal( localbarfile, localobsfile, irec, localstartdate, localperiod, fname1, fname2, localrec, obsrec,
    exst, mythid )   @63cdc0b pkg/ecco/cost_gencal.F:7-121 -> (fname1, fname2, localrec, obsrec, exst)."""
    fname1 = f"{localbarfile.rstrip(' ')}.{eccoiter:010d}"                  # :70-72 '(2a,i10.10)'
    if localperiod == dTtracerLev1:                                         # :74-77
        localrec, obsrec, yday = irec, irec, 0
    elif localperiod == 86400.:                                             # :78-99
        raise NotImplementedError("COST_GENCAL: daily fields (localperiod = 86400) are not ported")
    else:                                                                   # :100-106 monthly fields
        obsrec = irec
        mody = idiv(cal.modelStartDate[0], 10000)
        modm = idiv(cal.modelStartDate[0], 100) - mody*100
        yday = mody + idiv(modm - 1 + irec - 1, 12)
        localrec = 1 + imod(modm - 1 + irec - 1, 12)
    il = localobsfile.rstrip(" ")
    fname2 = f"{il}_{yday:4d}"                                              # :108-110 '(2a,i4)'
    exst = os.path.exists(os.path.join(str(rundir), fname2))                # :111
    if not exst:                                                            # :112-116 cyclic data set
        fname2 = il
        exst = os.path.exists(os.path.join(str(rundir), fname2))
    return fname1, fname2, localrec, obsrec, exst


def _preproc_flags(g):
    """COST_GENLOOP's pre/post-processing switches (cost_generic.F:271-304) for gencost `g` -> dict(domean, dooffset,
    domaskmindepth, topomin, dosumsq, fac). The switches of other runs raise."""
    f = dict(dosumsq=True, domean=False, doanom=False, dovarwei=False, dosmooth=False, dooffset=False,
             domaskmindepth=False, doclim=False, fac=1.0, topomin=None)                 # :271-281
    for k2 in range(len(g.gencost_preproc)):                                            # :283-304
        pre, pos = g.gencost_preproc[k2].rstrip(" "), g.gencost_posproc[k2].rstrip(" ")
        if pre == "mean":
            f["domean"] = True
        if pre == "anom":
            f["doanom"] = True
        if pre == "variaweight":
            f["dovarwei"] = True
        if pre == "nosumsq":
            f["dosumsq"] = False
        if pre == "offset":
            f["dooffset"] = True
        if pre == "mindepth":
            f["domaskmindepth"] = True
            f["topomin"] = float(g.gencost_preproc_r[k2])
        if pos == "smooth":
            f["dosmooth"] = True
        if pre == "clim":
            f["doclim"] = True
        if pre == "factor":
            f["fac"] = float(g.gencost_preproc_r[k2])
        if pre not in ("", "mean", "offset", "mindepth") or pos != "":
            raise NotImplementedError(f"COST_GENLOOP: gencost {g.k} preproc '{pre}' / posproc '{pos}' not ported")
    return f


def gencost_setup(ep, *, rw, cal, eccoiter, dTtracerLev1, sz):
    """The host part of COST_GENCOST_ALL for every gencost it evaluates (:47-87) -> [dict(g, nnz, pp, recs=[dict(irec,
    obs, weight)])]: nnz (:60-66), the switches of `_preproc_flags`, and per obs record irec = 1..nrecloop (:359-365)
    the weight (:368-378: COST_GENCAL on the errfile with jrec = 1, ECCO_ZERO, ECCO_READWEI when localrec, obsrec > 0
    and the file exists: READ_REC_LEV_RL( fname3, cost_iprec, Nr, 1, nnzobs, localWeight, localrec, 1 ),
    ecco_toolbox.F:950-951) and the data record (:380-397: COST_GENCAL, ECCO_ZERO with spzero, READ_REC_LEV_RL(
    fname2, cost_iprec, Nr, 1, nnzobs, localobs, localrec, 1 )) as interior values [tile, nnz, sNy, sNx] (REAL*8
    numpy; a record of nnzobs levels, MDS_READ_FIELD with kLo = 1, kHi = nnzobs)."""
    out = []
    ntiles = sz.nSx*sz.nSy
    for g in ep.gencost:
        if g.gencost_flag == -1 and g.using_gencost and g.gencost_name.rstrip() == "bpv4-grace":
            raise NotImplementedError("COST_GENCOST_BPV4 is not ported")
        if not (g.using_gencost and g.gencost_flag == 1 and not g.gencost_is1d):   # :49-50
            continue
        nnz = sz.Nr if g.gencost_is3d else 1                                     # :60-66
        b = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy), k=(1, nnz))
        pp = _preproc_flags(g)
        if _eq(g.gencost_datafile, " "):                                         # COST_GENERIC :107, :123
            raise NotImplementedError("COST_GENERIC: a gencost without datafile is not ported")
        recs = []
        nrecloop = g.gencost_nrec                                                # :359 (no clim: :362)
        for irec in range(1, nrecloop + 1):                                      # :365
            jrec = 1                                                             # :369 (dovarwei off)
            _, fname3, localrec, obsrec, exst = cost_gencal(
                g.gencost_barfile, g.gencost_errfile, jrec, g.gencost_startdate, g.gencost_period, eccoiter=eccoiter,
                cal=cal, dTtracerLev1=dTtracerLev1, rundir=rw.rundir)            # :371-373
            weight = np.zeros((ntiles, nnz, sz.sNy, sz.sNx))                     # :374
            if localrec > 0 and obsrec > 0 and exst:                             # :375-377
                raw = rw.mds_read_field(fname3, ep.cost_iprec, nnz, FArray(jnp.zeros((ntiles, nnz) + (
                    sz.sNy + 2*sz.OLy, sz.sNx + 2*sz.OLx)), "localWeight", **b), localrec)
                weight = ecco_readwei(np.asarray(raw.data)[:, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx],
                                      pp["dosumsq"])
            fname1, fname2, localrec, obsrec, exst = cost_gencal(
                g.gencost_barfile, g.gencost_datafile, irec, g.gencost_startdate, g.gencost_period,
                eccoiter=eccoiter, cal=cal, dTtracerLev1=dTtracerLev1, rundir=rw.rundir)   # :381-383
            obs = np.full((ntiles, nnz, sz.sNy, sz.sNx), np.float64(g.gencost_spzero))   # :392
            if localrec > 0 and obsrec > 0 and exst:                             # :393-397
                raw = rw.mds_read_field(fname2, ep.cost_iprec, nnz, FArray(jnp.zeros((ntiles, nnz) + (
                    sz.sNy + 2*sz.OLy, sz.sNx + 2*sz.OLx)), "localobs", **b), localrec)
                obs = np.asarray(raw.data)[:, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx].copy()
            recs.append(dict(irec=irec, fname1=fname1, weight=weight, obs=obs))
        out.append(dict(g=g, nnz=nnz, pp=pp, recs=recs))
    return out


def cost_gencost_all(ecco_rec, setup, *, maskC, sz, R_low=None, ex=None):
    """COST_GENCOST_ALL( myiter, mytime, mythid ) traced: for each evaluated gencost k, COST_GENERIC (objf_local =
    num_local = 0. _d 0 :86-91) -> COST_GENLOOP (:107-139). The record loop (:365-469): localbar = record irec of the
    bar file (COST_GENREAD -> ECCO_READBAR, :387-389), ECCO_MULT by fac (:390), ECCO_DIFFMSK (:399-401),
    ECCO_MASKMINDEPTH (:403-404), ECCO_ADDMASK with domean (:409-411); with addVariaCost (no domean) ECCO_ADDCOST
    (:442-446) and the record misfit (:448-455). After the loop (:472-503): localdifmeanOut = ECCO_CP / ECCO_DIVFIELD
    of localdifsum by localdifnum, localdifmsk of localdifnum by itself, ECCO_OFFSET (dooffset), and with domean
    ECCO_ADDCOST of the mean misfit with the last record's weight and its misfit at record 1.
    `ecco_rec`: {k: (FArray [tile, nnz, sNy, sNx] per record)} (the carried records); `setup`: gencost_setup's (traced arrays
    allowed for obs / weight); `R_low` (GRID.h, ECCO_MASKMINDEPTH); `ex` the exchange context of a sharded run
    (GLOBAL_SUM_TILE_RL). Returns ({k: (objf_gencost, num_gencost)} [tile], {k: [(irec, localdif)]} the misfit records,
    {k: (volGlob, theMean)} of ECCO_OFFSET)."""
    res, mis, offs = {}, {}, {}
    for s in setup:
        g, nnz, pp = s["g"], s["nnz"], s["pp"]
        k, j, i = loops_kji((1, nnz), (1, sz.sNy), (1, sz.sNx))
        localmask = maskC[i, j, k]                                                 # :312-318 (ylocmask 'c')
        nT = localmask.shape[0]
        objf = jnp.zeros((nT,), jnp.float64)                                       # cost_generic.F:89
        num = jnp.zeros((nT,), jnp.float64)                                        # :90
        zero = jnp.zeros_like(localmask)
        localdifsum, localdifnum = zero, zero                                      # :268-269 ECCO_ZERO
        addVariaCost = not pp["domean"]                                            # COST_GENERIC :107-139
        mis[g.k] = []
        localweight = zero
        for r in s["recs"]:
            localweight = jnp.asarray(r["weight"])                                 # :374-377
            localbar = ecco_rec[g.k][r["irec"] - 1].data                           # :387-389
            localbar = ecco_mult(localbar, pp["fac"])                              # :390
            localdif, difmask = ecco_diffmsk(localbar, r["obs"], localmask, g.gencost_spmin, g.gencost_spmax,
                                             g.gencost_spzero)                     # :399-401
            if pp["domaskmindepth"]:                                               # :403-404
                difmask = ecco_maskmindepth(difmask, R_low[i, j], pp["topomin"])
            if pp["domean"]:                                                       # :409-411 (doanom raises)
                localdifsum, localdifnum = ecco_addmask(localdif, difmask, localdifsum, localdifnum)
            if addVariaCost:                                                       # :413
                objf, num = ecco_addcost(localdif, localweight, difmask, pp["dosumsq"], objf, num)   # :442-446
                if g.gencost_outputlevel > 0:                                      # :448
                    mis[g.k].append((r["irec"], localdif))
        if pp["domean"]:
            localdifmeanOut = ecco_divfield(ecco_cp(localdifsum, zero), localdifnum)   # :472-474
            localdifmsk = ecco_divfield(ecco_cp(localdifnum, zero), localdifnum)       # :475-476
            if pp["dooffset"]:                                                     # :478-484
                localdifmeanOut, volGlob, theMean = ecco_offset(localdifmeanOut, localdifmsk, ex=ex)
                offs[g.k] = (volGlob, theMean)
            objf, num = ecco_addcost(localdifmeanOut, localweight, localdifmsk, pp["dosumsq"], objf, num)  # :486-492
            if g.gencost_outputlevel > 0:                                          # :495-502
                mis[g.k].append((1, localdifmeanOut))
        elif pp["dooffset"]:
            raise NotImplementedError("COST_GENLOOP: 'offset' without 'mean' is not ported")
        res[g.k] = (objf, num)
    return res, mis, offs


def ecco_offset_lines(offs, ep):
    """ECCO_OFFSET's STDOUT records (ecco_toolbox.F:795-806, master thread; host), per gencost with dooffset in
    COST_GENCOST_ALL's order: '(3A,1PE21.14)' 'ecco_offset: # of nonzero constributions to mean of ', fname, ' = ',
    volGlob and 'ecco_offset:                         Global mean of ', fname, ' = ', theMean; fname =
    localbarfile(1:il) (cost_generic.F:479-481)."""
    from mitjax.io.fortran_format import fortran_write
    out = []
    for g in ep.gencost:
        if g.k in offs:
            vol, mean = offs[g.k]
            fn = g.gencost_barfile.rstrip(" ")
            out.append(fortran_write("(3A,1PE21.14)", "ecco_offset: # of nonzero constributions to mean of ", fn,
                                     " = ", float(vol)))
            out.append(fortran_write("(3A,1PE21.14)", "ecco_offset:                         Global mean of ", fn,
                                     " = ", float(mean)))
    return out


def _write_xy_or_xyz(fname, vals, irec, *, writeBinaryPrec, globalFile, eccoiter, mds):
    """WRITE_REC_XY_RL (nNz = 1, pkg/rw/write_rec.F:190-199) or WRITE_REC_XYZ_RL (nNz = Nr, :316-325):
    MDS_WRITE_FIELD( fName, writeBinaryPrec, globalFile, useCurrentDir = .FALSE., 'RL', nNz, 1, nNz, field, iRec,
    myIter ) of COST_GENLOOP's local `vals` (interior values [tile, nnz, sNy, sNx]; the Fortran local is declared
    (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy), cost_generic.F:232-243, ECCO_ZERO'd on every point, :258-269, and only
    its levels 1..nnz interior are ever set).

    WRITE_REC_XY_RL declares its dummy `field(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)` (write_rec.F:162): the Nr-level
    actual argument is passed by sequence association, so the 2-D tile (bi,bj) the routine writes is the
    ((bi-1)+(bj-1)*nSx+1)-th xy slab of the local's storage -- with Nr >= nSx*nSy that is level
    (bi-1)+(bj-1)*nSx+1 of tile (1,1): level 1 (the misfit) for tile (1,1), the zero levels 2.. for every other tile.
    Literal here (the storage flattened in Fortran order, tile = (bj-1)*nSx + bi - 1): the oracle's misfit_sst /
    misfit_mdt / weight_mdt files of lab_sea/input_ad (4 tiles of one process) hold zeros in tiles 2-4
    (docs/ISSUES_UPSTREAM.md)."""
    from mitjax.pkg.mdsio.mdsio_write_field import mds_write_field
    sz = mds.size
    vals = np.asarray(vals)
    nnz = vals.shape[1]
    J, I = sz.sNy + 2*sz.OLy, sz.sNx + 2*sz.OLx
    b = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
    if nnz == 1 and sz.Nr != 1:                                           # WRITE_REC_XY_RL of an Nr-level local
        T = vals.shape[0]
        store = np.zeros((T, sz.Nr, J, I))                               # ECCO_ZERO (:258-269)
        store[:, :1, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = vals
        fld = FArray(jnp.asarray(store.reshape(T*sz.Nr, J, I)[:T]), "localdif", **b)   # sequence association
    else:
        full = np.zeros((vals.shape[0], nnz, J, I))                      # :258-269 ECCO_ZERO
        full[:, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = vals
        fld = (FArray(jnp.asarray(full[:, 0]), "localdif", **b) if nnz == 1
               else FArray(jnp.asarray(full), "localdif", k=(1, nnz), **b))
    mds_write_field(fname, writeBinaryPrec, globalFile, False, "RL", nnz, 1, nnz, fld, irec, eccoiter, mds=mds)


def write_misfits(mis, ep, *, eccoiter, writeBinaryPrec, globalFile, mds, setup=None):
    """COST_GENLOOP's file output on the host, in its order per gencost: the misfits WRITE_REC_XY(Z)_RL(
    'misfit_'//outname, localdif, irec, eccoiter ) (:448-455, or the mean misfit at record 1, :495-502) and with
    outlev > 1 and no dovarwei the weight 'weight_'//outname at record 1 (:504-513; the last record's localweight)."""
    for g in ep.gencost:
        for irec, localdif in mis.get(g.k, []):
            fname3 = "misfit_" + g.gencost_name.rstrip(" ")                       # :449-450, :496-497
            _write_xy_or_xyz(fname3, localdif, irec, writeBinaryPrec=writeBinaryPrec, globalFile=globalFile,
                             eccoiter=eccoiter, mds=mds)
        if g.gencost_outputlevel > 1:                                             # :504-513 (:457-466 dovarwei raises)
            s = [x for x in (setup or []) if x["g"].k == g.k]
            if len(s) != 1:
                raise ValueError(f"write_misfits: gencost {g.k} with outputlevel > 1 needs its setup (weight)")
            fname3 = "weight_" + g.gencost_name.rstrip(" ")                       # :506-507
            _write_xy_or_xyz(fname3, s[0]["recs"][-1]["weight"], 1, writeBinaryPrec=writeBinaryPrec,
                             globalFile=globalFile, eccoiter=eccoiter, mds=mds)
