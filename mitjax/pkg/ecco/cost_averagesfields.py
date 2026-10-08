"""COST_AVERAGESFIELDS with COST_AVERAGESFLAGS, COST_GENCOST_ASSIGNPERIOD, COST_GENCOST_CUSTOMIZE,
COST_AVERAGESGENERIC; COST_AVERAGESINIT   @63cdc0b pkg/ecco/cost_averagesfields.F:3-131, cost_averagesflags.F:4-300,
cost_gencost_assignperiod.F:3-110, cost_gencost_customize.F:20-300, cost_averagesgeneric.F:3-213,
cost_averagesinit.F:3-80 (lane M4ADCOL session 3)

COST_AVERAGESFIELDS runs at the start of every step (the_main_loop.F:663-673, with the step-start clock of the
ALLOW_AUTODIFF arm :650-653: myTime = startTime + deltaTClock*(iloop-1)) and once after the loop with endTime
(:737-742). Its calendar part (COST_AVERAGESFLAGS, the record bookkeeping sum1*/ *rec of ECCO.h, which persists
between calls, and COST_GENCOST_ASSIGNPERIOD) is host integer work on concrete times: `averages_table` evaluates it
for the n+1 calls of a run and reduces it, per gencost k, to the branch COST_AVERAGESGENERIC takes, sum1gen(k) and
genrec(k) -- the per-call table a traced step reads with its loop counter (call c = iloop; c = nTimeSteps+1 is the
call after the loop). COST_GENCOST_CUSTOMIZE and COST_AVERAGESGENERIC are traced (`cost_averagesfields`).

The bar file (ACTIVE_WRITE_XYZ of gencost_barfile.<eccoiter>, record genrec, cost_averagesgeneric.F:130-139,
:173-182) is kept as its records in the carry (`rec`, one interior FArray [tile, nnz, j, i] per record; session 4 of
lane M4ADLAB: tiled FArrays, so that P = N shards them); COST_GENLOOP's ECCO_READBAR reads
them back from there, and the host writes the file after the loop (`write_barfiles`): a forward ACTIVE_WRITE_3D_RL is
MDS_WRITE_FIELD with ctrlprec = 64 (active_file_control.F:771-785, ctrl_readparms.F:229), so the file holds the
REAL*8 values and the read-back is exact.
"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray, loops_kji
from mitjax.pkg.cal.cal_addtime import cal_AddTime
from mitjax.pkg.cal.cal_compdates import cal_CompDates
from mitjax.pkg.cal.cal_copydate import cal_CopyDate
from mitjax.pkg.cal.cal_fulldate import cal_FullDate
from mitjax.pkg.cal.cal_getdate import cal_GetDate
from mitjax.pkg.cal.cal_h import idiv, imod, nMonthYear
from mitjax.pkg.cal.cal_numints import cal_NumInts
from mitjax.pkg.cal.cal_timeinterval import cal_TimeInterval
from mitjax.pkg.cal.cal_timepassed import cal_TimePassed
from mitjax.pkg.ecco.ecco_readparms import _eq, _sub

# COST_AVERAGESGENERIC's branches (cost_averagesgeneric.F:107, :142, :156, :184, :202)
SNAPSHOT, ASSIGN, DIVIDE, ACCUMULATE = 0, 1, 2, 3


class EccoRecs:
    """ECCO.h's record bookkeeping that COST_AVERAGESFLAGS writes through its O arguments (sum1day, dayrec, sum1mon,
    monrec, sum1year, yearrec: cost_averagesfields.F:82-84); the common block's static zero at the start."""

    def __init__(self):
        self.sum1day = self.dayrec = self.sum1mon = self.monrec = self.sum1year = self.yearrec = 0


def cost_averagesflags(myiter, mytime, r, *, cal):
    """cost_AveragesFlags( myiter, mytime, mythid, first, last, startofday, ..., sum1year, yearrec )
    @63cdc0b pkg/ecco/cost_averagesflags.F:4-300 on concrete (myiter, mytime); `r` (EccoRecs) updated in place.
    Returns dict(first, last, startofday, startofmonth, startofyear, inday, inmonth, inyear, endofday, endofmonth,
    endofyear)."""
    mydate = cal_GetDate(myiter, mytime, cal=cal)                                         # :109
    nextdate = cal_GetDate(myiter + 1, np.float64(mytime) + cal.modelStep, cal=cal)       # :110
    timediff = cal_TimeInterval(-cal.modelStep, "secs", cal=cal)                          # :112
    prevdate = cal_AddTime(mydate, timediff, cal=cal)                                     # :113
    msd = cal.modelStartDate
    if cal_CompDates(msd, mydate):                                                        # :122-130
        first = True
        r.dayrec = 0
        r.monrec = 0
        r.yearrec = 0
    else:
        first = False
    last = cal_CompDates(cal.modelEndDate, mydate)                                        # :133-138
    mydateday, prevdateday = imod(mydate[0], 100), imod(prevdate[0], 100)                 # :141-142
    startofday = mydateday != prevdateday                                                 # :143-147
    nextdateday = imod(nextdate[0], 100)                                                  # :150-151
    endofday = mydateday != nextdateday                                                   # :152-156
    inday = mydateday == prevdateday and mydateday == nextdateday                         # :161-166
    if last or endofday:                                                                  # :169-192
        if mydate[0] == msd[0]:
            targetdate = cal_CopyDate(msd)
            r.dayrec = 1
        else:
            targetdate = [mydate[0], 0, mydate[2], mydate[3]]                             # :174-177
            datediff = cal_TimePassed(msd, targetdate, cal=cal)                           # :178-179
            r.dayrec = datediff[0] + 1 if datediff[1] == 0 else datediff[0] + 2           # :180-184
        timediff = cal_TimeInterval(cal.modelStep, "secs", cal=cal)                       # :186
        r.sum1day = cal_NumInts(targetdate, mydate, timediff, cal=cal) + 1                # :188-189
    else:
        r.sum1day = 0                                                                     # :191
    mydatemonth = imod(idiv(mydate[0], 100), 100)                                         # :195
    prevdatemonth = imod(idiv(prevdate[0], 100), 100)                                     # :196
    startofmonth = mydatemonth != prevdatemonth                                           # :197-201
    nextdatemonth = imod(idiv(nextdate[0], 100), 100)                                     # :205
    endofmonth = mydatemonth != nextdatemonth                                             # :206-210
    inmonth = mydatemonth == prevdatemonth and mydatemonth == nextdatemonth               # :215-220
    if last or endofmonth:                                                                # :223-248
        if idiv(mydate[0], 100)*100 == idiv(msd[0], 100)*100:                             # :224-226
            targetdate = cal_CopyDate(msd)
            r.monrec = 1
        else:
            targetdate1 = idiv(mydate[0], 100)*100 + 1                                    # :228
            targetdate2 = 0
            targetdate = list(cal_FullDate(targetdate1, targetdate2, cal=cal))            # :230-231
            if idiv(mydate[0], 10000) == idiv(msd[0], 10000):                             # :232-240
                r.monrec = imod(idiv(mydate[0], 100), 100) - imod(idiv(msd[0], 100), 100) + 1
            else:
                r.monrec = (imod(idiv(mydate[0], 100), 100) + nMonthYear - imod(idiv(msd[0], 100), 100) + 1
                            + (idiv(mydate[0], 10000) - idiv(msd[0], 10000) - 1)*nMonthYear)
        timediff = cal_TimeInterval(cal.modelStep, "secs", cal=cal)                       # :242
        r.sum1mon = cal_NumInts(targetdate, mydate, timediff, cal=cal) + 1                # :244-245
    else:
        r.sum1mon = 0                                                                     # :247
    mydateyear, prevdateyear = idiv(mydate[0], 10000), idiv(prevdate[0], 10000)           # :251-252
    startofyear = mydateyear != prevdateyear                                              # :253-257
    nextdateyear = idiv(nextdate[0], 10000)                                               # :261
    endofyear = mydateyear != nextdateyear                                                # :262-266
    inyear = mydateyear == prevdateyear and mydateyear == nextdateyear                    # :271-276
    if last or endofyear:                                                                 # :279-298
        if idiv(mydate[0], 10000) == idiv(msd[0], 10000):
            targetdate = cal_CopyDate(msd)
            r.yearrec = 1
        else:
            targetdate = list(cal_FullDate(idiv(mydate[0], 10000)*10000 + 101, 0, cal=cal))   # :285-288
            r.yearrec = idiv(mydate[0], 10000) - idiv(msd[0], 10000) + 1                  # :289
        timediff = cal_TimeInterval(cal.modelStep, "secs", cal=cal)                       # :292
        r.sum1year = cal_NumInts(targetdate, mydate, timediff, cal=cal) + 1               # :294-295
    else:
        r.sum1year = 0                                                                    # :297
    return dict(first=first, last=last, startofday=startofday, startofmonth=startofmonth, startofyear=startofyear,
                inday=inday, inmonth=inmonth, inyear=inyear, endofday=endofday, endofmonth=endofmonth,
                endofyear=endofyear)


def cost_gencost_assignperiod(f, r, g, myiter, niter0):
    """cost_gencost_assignperiod( ... )   @63cdc0b pkg/ecco/cost_gencost_assignperiod.F:3-110, for gencost `g`
    (:55-105): -> (startofgen, endofgen, ingen, sum1gen, genrec), or None when the k is not assigned (:56)."""
    if not (g.using_gencost and (g.gencost_flag >= 1 or not _eq(g.gencost_avgperiod, "     "))):
        return None
    ap = g.gencost_avgperiod.rstrip(" ")
    if ap in ("day", "DAY"):                                                              # :58-63
        return f["startofday"], f["endofday"], f["inday"], r.sum1day, r.dayrec
    if ap in ("month", "MONTH"):                                                          # :64-69
        return f["startofmonth"], f["endofmonth"], f["inmonth"], r.sum1mon, r.monrec
    if ap in ("year", "YEAR"):                                                            # :70-75
        return f["startofyear"], f["endofyear"], f["inyear"], r.sum1year, r.yearrec
    if ap in ("step", "STEP"):                                                            # :76-81
        return True, True, True, 1, 1 + myiter - niter0
    if ap in ("const", "CONST"):                                                          # :82-88 (with a print*)
        raise NotImplementedError("COST_GENCOST_ASSIGNPERIOD: 'const' (print* 'gf-const') is not ported")
    raise RuntimeError("gencost_avgperiod wrongly specified")                             # :89-91


def averages_table(ep, *, cal, nIter0, startTime, deltaTClock, nTimeSteps, endTime):
    """The host part of the n+1 COST_AVERAGESFIELDS calls of a run (module docstring) -> {k: dict(branch, sum1,
    rec: int32 arrays [nTimeSteps+1], indexed by call - 1; nrec, writes: [(call, rec)])} for the gencost k that
    COST_AVERAGESFIELDS averages (using_gencost and not gencost_barskip, :97-98)."""
    r = EccoRecs()
    calls = [np.float64(startTime) + np.float64(deltaTClock)*np.float64(c - 1) for c in range(1, nTimeSteps + 1)]
    calls.append(np.float64(endTime))                                                     # the_main_loop.F:740
    out = {g.k: dict(branch=[], sum1=[], rec=[], nrec=g.gencost_nrec, writes=[]) for g in ep.gencost
           if g.using_gencost and not g.gencost_barskip}
    for c, mytime in enumerate(calls, start=1):
        myiter = nIter0 + int((mytime - np.float64(startTime))/np.float64(deltaTClock) + 0.5)   # :69
        f = cost_averagesflags(myiter, mytime, r, cal=cal)                                # :76-85
        for g in ep.gencost:
            if g.k not in out:
                continue
            a = cost_gencost_assignperiod(f, r, g, myiter, nIter0)                        # :88-94
            if a is None:
                raise RuntimeError(f"COST_AVERAGESFIELDS: gencost {g.k} averaged with no period assigned")
            startofgen, endofgen, ingen, sum1gen, genrec = a
            first, last = f["first"], f["last"]
            if startofgen and endofgen:                                                   # :107
                b = SNAPSHOT
            elif first or startofgen:                                                     # :142
                b = ASSIGN
            elif last or endofgen:                                                        # :156
                b = DIVIDE
            elif ingen and not (first or startofgen) and not (last or endofgen):          # :184-186
                b = ACCUMULATE
            else:
                raise RuntimeError("in cost_averagesgeneric")                             # :201-203 stop
            t = out[g.k]
            t["branch"].append(b)
            t["sum1"].append(sum1gen)
            t["rec"].append(genrec)
            if b in (SNAPSHOT, DIVIDE):
                if not 1 <= genrec <= g.gencost_nrec:
                    raise NotImplementedError(f"COST_AVERAGESGENERIC: gencost {g.k} writes record {genrec} of "
                                              f"{g.gencost_nrec}: not ported")
                t["writes"].append((c, genrec))
    for t in out.values():
        for n in ("branch", "sum1", "rec"):
            t[n] = np.asarray(t[n], np.int32)
        if sorted({rr for _, rr in t["writes"]}) != list(range(1, t["nrec"] + 1)):
            raise NotImplementedError(f"COST_AVERAGESFIELDS: bar records written {t['writes']} do not cover the "
                                      f"{t['nrec']} records COST_GENLOOP reads: not ported")
    return out


def customize_arm(g):
    """The arm of COST_GENCOST_CUSTOMIZE's IF chain (cost_gencost_customize.F:122-262) that gencost `g` takes, for the
    arms ported: 'm_eta' (:122-125: barfile(1:5) .EQ. 'm_eta' .and. barfile(1:9) .NE. 'm_eta_dyn'), 'm_sst'
    (:132-134), 'm_theta' (:226-231), 'm_salt' (:232-237; lane M4ADCOL). The arms between them in the chain
    (m_boxmean, m_horflux :126-131; m_sss, m_drifterUE/VN, m_bp :135-147; the EXF, CTRL and SEAICE arms :164-223)
    test prefixes and names that these barfiles do not match; any other barfile raises (set-up)."""
    bf = g.gencost_barfile
    if g.gencost_name.rstrip() in ("siv4-conc", "siv4-deconc", "siv4-exconc"):         # :207-217 (name tests)
        raise NotImplementedError(f"COST_GENCOST_CUSTOMIZE: {g.gencost_name.rstrip()} not ported")
    if _eq(_sub(bf, 5), "m_eta") and not _eq(_sub(bf, 9), "m_eta_dyn"):                # :122-123
        return "m_eta"
    if _eq(_sub(bf, 9), "m_boxmean") or _eq(_sub(bf, 9), "m_horflux"):                # :126-131
        raise NotImplementedError(f"COST_GENCOST_CUSTOMIZE: barfile {bf.rstrip()} not ported")
    if _eq(_sub(bf, 5), "m_sst"):                                                      # :132
        return "m_sst"
    if _eq(_sub(bf, 7), "m_theta"):                                                    # :226
        return "m_theta"
    if _eq(_sub(bf, 6), "m_salt"):                                                     # :232
        return "m_salt"
    raise NotImplementedError(f"COST_GENCOST_CUSTOMIZE: barfile {bf.rstrip()} not ported")


def check_customize(ep):
    """COST_GENCOST_CUSTOMIZE's arms ported (`customize_arm`): every averaged gencost must take one of them, and a
    2-D arm (m_eta, m_sst) must be a 2-D gencost, a 3-D arm (m_theta, m_salt) a 3-D one (ECCO_READPARMS' is3d,
    ecco_readparms.F:747-756). The EXF locals zontau, mertau (:89-104) and zonwind, merwind (ROTATE_UV2EN_RL,
    :110-111) are read only by the m_ustress / m_vstress / m_uwind / m_vwind arms."""
    for g in ep.gencost:
        if g.using_gencost and not g.gencost_barskip:
            arm = customize_arm(g)
            if g.gencost_is3d != (arm in ("m_theta", "m_salt")):
                raise NotImplementedError(f"COST_GENCOST_CUSTOMIZE: {arm} with gencost_is3d = {g.gencost_is3d}")


def cost_gencost_customize(g, *, theta, salt, maskC, sz, m_eta=None):
    """The ported arms of COST_GENCOST_CUSTOMIZE for gencost `g` (`customize_arm`), over the interior (iMin..iMax =
    1..sNx, jMin..jMax = 1..sNy, :82-85):
    m_eta (:124-125): gencost_modfld(i,j,bi,bj,k) = m_eta(i,j,bi,bj)*maskC(i,j,1,bi,bj) (ECCO.h m_eta, written by
    ECCO_PHYS at the end of the previous FORWARD_STEP or by ECCO_INIT_VARIA's call);
    m_sst (:133-134): gencost_modfld(i,j,bi,bj,k) = THETA(i,j,1,bi,bj)*maskC(i,j,1,bi,bj);
    m_theta / m_salt (:229-230, :235-236): gencost_mod3d(i,j,k2,bi,bj,kk) = theta(i,j,k2,bi,bj)*maskC(i,j,k2,bi,bj)
    (salt), k2 = 1..Nr.
    Returns the interior values [tile, nnz, sNy, sNx] (nnz = 1 for gencost_modfld; the halo of the field is never
    read: COST_AVERAGESGENERIC loops 1..sNx/sNy)."""
    arm = customize_arm(g)
    if arm in ("m_eta", "m_sst"):
        k, j, i = loops_kji((1, 1), (1, sz.sNy), (1, sz.sNx))       # level 1 only (a k axis of length 1)
        if arm == "m_eta":
            return m_eta[i, j] * maskC[i, j, k]                                           # :124-125
        return theta[i, j, k] * maskC[i, j, k]                                            # :133-134
    k, j, i = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))
    if arm == "m_theta":
        return theta[i, j, k] * maskC[i, j, k]                                            # :229-230
    return salt[i, j, k] * maskC[i, j, k]                                                 # :235-236


def cost_averagesgeneric(localbar, localfld, branch, sum1loc):
    """cost_averagesgeneric( localbarfile, localbar, localfld, xx_localbar_mean_dummy, first, last, startofloc,
    endofloc, inloc, sum1loc, locrec, nnz, myThid )   @63cdc0b pkg/ecco/cost_averagesgeneric.F:3-213

    On the interior values [tile, nnz, sNy, sNx] with the traced branch of `averages_table`:
    SNAPSHOT (:107-118) and ASSIGN (:142-154): localbar = localfld; DIVIDE (:156-170): localbar = (localbar +
    localfld)/float(sum1loc) -- FLOAT of the INTEGER is REAL*4 (exact for these counts), promoted to REAL*8 in the
    REAL*8 division; ACCUMULATE (:184-199): localbar = localbar + localfld. Guards before the operation: the divisor
    is 1 on the lanes that do not divide (sum1 = 0 there)."""
    div = branch == DIVIDE
    s1 = jnp.where(div, sum1loc, 1).astype(jnp.float32).astype(jnp.float64)
    return jnp.where((branch == SNAPSHOT) | (branch == ASSIGN), localfld,
                     jnp.where(div, (localbar + localfld) / s1,
                               jnp.where(branch == ACCUMULATE, localbar + localfld, localbar)))


def cost_averagesinit(ep, *, sz, ntiles):
    """COST_AVERAGESINIT   @63cdc0b pkg/ecco/cost_averagesinit.F:3-80 (from ECCO_COST_INIT_VARIA :60): the
    interiors of gencost_barfld / modfld / bar3d / mod3d and gencost_dummy set to 0. _d 0 (:51-75). Returns the
    carried part of ECCO.h: {"bar": {k: FArray}, "rec": {k: (FArray, ...)}} for the averaged gencost k:
    gencost_bar3d(..., gencost_pointer3d(k)) with nnz = Nr levels for a 3-D gencost, gencost_barfld(..., k) with
    nnz = 1 for a 2-D one (lane M4ADLAB session 4), and the bar file's records (one FArray per record). Only the
    interiors are ever written or read (COST_AVERAGESGENERIC loops 1..sNx/sNy): the FArrays are declared on the
    interior, i = 1..sNx, j = 1..sNy, k = 1..nnz (data [tile, nnz, sNy, sNx]), tiled -- so that a sharded run
    (P = N) carries each device's own tiles, as every other tiled field of the carry."""
    bar, rec = {}, {}
    for g in ep.gencost:
        if g.using_gencost and not g.gencost_barskip:
            nnz = sz.Nr if g.gencost_is3d else 1
            z = jnp.zeros((ntiles, nnz, sz.sNy, sz.sNx), jnp.float64)
            bar[g.k] = _interior(z, "gencost_bar", nnz, sz)
            rec[g.k] = tuple(_interior(z, "gencost_barrec", nnz, sz) for _ in range(g.gencost_nrec))
    return {"bar": bar, "rec": rec}


def _interior(data, name, nnz, sz):
    """An FArray of interior values [tile, nnz, sNy, sNx] (i = 1..sNx, j = 1..sNy, k = 1..nnz), tiled."""
    return FArray(data, name, i=(1, sz.sNx), j=(1, sz.sNy), k=(1, nnz))


def cost_averagesfields(ecco, call, *, ep, tab, theta, salt, maskC, sz):
    """cost_averagesfields( mytime, mythid )   @63cdc0b pkg/ecco/cost_averagesfields.F:3-131: COST_GENCOST_CUSTOMIZE
    (:95), then COST_AVERAGESGENERIC for each averaged gencost (:97-123; nnz = 1 for a 2-D gencost :99-107, Nr for a
    3-D one :109-118) with its ACTIVE_WRITE into the carried records (`rec`) on the SNAPSHOT / DIVIDE branches at
    record genrec. `call` (traced int32): the call number (iloop; nTimeSteps+1 after the loop); `tab`: {k: (branch,
    sum1, rec)} int32 arrays (averages_table). The m_eta arm reads ECCO.h's m_eta from ecco["phys"]."""
    bar, recs = dict(ecco["bar"]), dict(ecco["rec"])
    for g in ep.gencost:
        if g.k not in bar:
            continue
        b, s1, r = (jnp.asarray(x)[call - 1] for x in tab[g.k])
        m_eta = ecco["phys"]["m_eta"] if "phys" in ecco else None
        fld = cost_gencost_customize(g, theta=theta, salt=salt, maskC=maskC, sz=sz, m_eta=m_eta)
        new = cost_averagesgeneric(bar[g.k].data, fld, b, s1)
        write = (b == SNAPSHOT) | (b == DIVIDE)
        recs[g.k] = tuple(FArray(jnp.where(write & (r == irec), new, x.data), x.name, tiled=x.tiled, _dims=x.dims)
                          for irec, x in enumerate(recs[g.k], start=1))      # ACTIVE_WRITE of record genrec
        bar[g.k] = FArray(new, bar[g.k].name, tiled=bar[g.k].tiled, _dims=bar[g.k].dims)
    return dict(ecco, bar=bar, rec=recs)


def write_barfiles(ecco, ep, table, *, eccoiter, mds):
    """The ACTIVE_WRITE_XY / ACTIVE_WRITE_XYZ calls of COST_AVERAGESGENERIC (cost_averagesgeneric.F:129-135,
    :178-184), on the host after the loop, in call order: ACTIVE_WRITE_3D_RL with myNr = 1 (XY, active_file.F:391-397)
    or Nr (XYZ), globalFile = .FALSE., useCurrentDir = .FALSE. -> MDS_WRITE_FIELD( '<barfile>.<eccoiter>'
    ('(2a,i10.10)'), ctrlprec = 64, .FALSE., .FALSE., 'RL', myNr, 1, myNr, localbar, iRec, myOptimIter = eccoiter )
    (active_file_control.F:780-785) with the record's values."""
    from mitjax.pkg.mdsio.mdsio_write_field import mds_write_field
    sz = mds.size
    for g in ep.gencost:
        if g.k not in table:
            continue
        fname = f"{g.gencost_barfile.rstrip(' ')}.{eccoiter:010d}"
        for _, irec in table[g.k]["writes"]:
            vals = np.asarray(ecco["rec"][g.k][irec - 1].data)
            nnz = vals.shape[1]
            full = np.zeros((vals.shape[0], nnz, sz.sNy + 2*sz.OLy, sz.sNx + 2*sz.OLx))
            full[:, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = vals
            b = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
            fld = (FArray(jnp.asarray(full[:, 0]), "localbar", **b) if not g.gencost_is3d
                   else FArray(jnp.asarray(full), "localbar", k=(1, sz.Nr), **b))
            mds_write_field(fname, 64, False, False, "RL", nnz, 1, nnz, fld, irec, eccoiter, mds=mds)
