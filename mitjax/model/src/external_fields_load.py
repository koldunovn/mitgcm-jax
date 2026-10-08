"""EXTERNAL_FIELDS_LOAD: model/src/external_fields_load.F @63cdc0b, with GET_PERIODIC_INTERVAL
(eesupp/src/get_periodic_interval.F @63cdc0b) that it calls.

Time-record sequencing [P§3]: which record is read at which step and the interpolation weights are computed on the
host, in float64 with the Fortran's own operations (`get_periodic_interval`), from the concrete model time of the
step; the reads (READ_REC_XY_RS) are host file reads as in the Fortran. The routine therefore takes concrete
`myTime`, `myIter` and runs eagerly, step by step. A traced time loop hoists this part: the per-step record indices
and weights are a function of (nIter0, deltaTClock, step) only, so a driver can precompute them on the host for every
step (the R5 section at the end of this file).

GET_PERIODIC_INTERVAL lives in eesupp/src; it is ported here because mitjax/eesupp belongs to another lane (to move
to mitjax/eesupp/get_periodic_interval.py when the lanes merge).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.eesupp.exch_rs import EXCH_UV_XY_RS, EXCH_XY_RS
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.io.fortran_format import fortran_write
from mitjax.model.src.ini_parms_forcing import debLevB, debLevC, debLevZero


def _nint(x):
    """Fortran NINT of a REAL*8 (round half away from zero; gfortran lround). x - trunc(x) is exact."""
    x = float(x)
    n = math.trunc(x)
    f = x - n
    if f >= 0.5:
        n += 1
    elif f <= -0.5:
        n -= 1
    return int(n)


def _int(x):
    """Fortran INT of a REAL*8 (truncation toward zero)."""
    return int(math.trunc(float(x)))


def _mod_r8(a, p):
    """Fortran MOD(a, p) of REAL*8 arguments: gfortran evaluates it with fmod (exact remainder, sign of a)."""
    return float(np.fmod(np.float64(a), np.float64(p)))


def get_periodic_interval(cycleLength, recSpacing, deltaT, currentTime):
    """GET_PERIODIC_INTERVAL( tRec0, tRec1, tRec2, wght1, wght2, cycleLength, recSpacing, deltaT, currentTime,
    myThid )   @63cdc0b eesupp/src/get_periodic_interval.F:7-137

    C     | o Provide time-record indices arround current time
    C     |   from a periodic, regularly spaced, time sequence
    C     | From a regularly-spaced sequence of time records
    C     | this routine returns the index of the two records
    C     | surrounding the current time and the record index of
    C     | corresponding to the previous time-step ; also provides
    C     | the weighting factor for a linear interpolation

    Returns (tRec0, tRec1, tRec2, wght1, wght2) as Python ints and float64 values (host). Every operation is one
    IEEE double operation in the Fortran's order; REAL*4 literals `0.5` and `0.` are exact. The non-periodic branch
    (cycleLength = 0, :96-114) raises: no M1 variant takes it. The STOPs of :71-94 raise with the Fortran message."""
    cycleLength, recSpacing, deltaT, currentTime = (np.float64(cycleLength), np.float64(recSpacing),
                                                    np.float64(deltaT), np.float64(currentTime))
    tRec0 = 0                                                                   # :65
    tRec1 = 0                                                                   # :66
    tRec2 = 0                                                                   # :67
    wght1 = np.float64(0.)                                                      # :68
    wght2 = np.float64(0.)                                                      # :69
    if cycleLength < 0. or recSpacing <= 0.:                                    # :71-81
        raise ValueError("ABNORMAL END: S/R GET_PERIODIC_INTERVAL (cycleLength >= 0 and recSpacing > 0 required; "
                         f"cycleLength={cycleLength}, recSpacing={recSpacing})")
    nbRec = _nint(cycleLength / recSpacing)                                     # :83
    tmpTime = np.float64(nbRec) * recSpacing                                    # :85
    if cycleLength != tmpTime:                                                  # :86-94
        raise ValueError("ABNORMAL END: S/R GET_PERIODIC_INTERVAL (cycleLength not multiple of recSpacing)")
    if cycleLength == 0.:                                                       # :96-114
        raise NotImplementedError("GET_PERIODIC_INTERVAL: the non-periodic branch (cycleLength = 0) is not ported")
    # :118-119  locTime = currentTime - recSpacing*0.5 + cycleLength*( 2 - NINT(currentTime/cycleLength) )
    locTime = (currentTime - recSpacing * np.float64(0.5)) \
        + cycleLength * np.float64(2 - _nint(currentTime / cycleLength))
    tmpTime = np.float64(_mod_r8(locTime, cycleLength))                         # :122
    tRec1 = 1 + _int(tmpTime / recSpacing)                                      # :123
    tRec2 = 1 + (tRec1 % nbRec)                                                 # :124 MOD of positive integers
    wght2 = (tmpTime - recSpacing * np.float64(tRec1 - 1)) / recSpacing         # :127
    wght1 = np.float64(1.) - wght2                                              # :128
    tmpTime = np.float64(_mod_r8(locTime - deltaT, cycleLength))                # :131
    tRec0 = 1 + _int(tmpTime / recSpacing)                                      # :132
    return tRec0, tRec1, tRec2, wght1, wght2


def external_fields_load(myTime, myIter, ff, *, cfg, fp, rw, ex, stdout=None):
    """EXTERNAL_FIELDS_LOAD( myTime, myIter, myThid )   @63cdc0b model/src/external_fields_load.F:7-361

    C     | o Control reading of fields from external source.
    C     | External source field loading routine.
    C     | This routine is called every time we want to
    C     | load a a set of external fields. The routine decides
    C     | which fields to load and then reads them in.

    `ff` FFIELDS.h (ini_ffields.FFields), `fp` the forcing parameters (ini_parms_forcing), `rw` the run directory
    context of pkg/rw, `ex` the exchanger; `stdout` (a list, or None to discard) receives the records the routine
    WRITEs to standardMessageUnit (:77-84 with ALLOW_DEBUG and debugLevel >= debLevB, :105-112 debugLevel >=
    debLevZero, :340-352 debugLevel >= debLevC). Returns the new FFields.

    Branches: periodicExternalForcing (:63); the reload test is the ALLOW_AUTODIFF one (:92), its
    STORE_LOADEDREC_TEST form (:94, GOADK lane: global_ocean.90x40x15/code_ad) or the plain one (:100); every file of
    PARM05 that a build reads (:114-173, ATMOSPHERIC_LOADING pLoadFile :205-212). SHORTWAVE_HEATING with surfQswFile
    raises (PTRACERS lane: without it the routine touches no SW field); ALLOW_GEOTHERMAL_FLUX is not ported
    (FFIELDS.h fields_of raises). Concrete myTime/myIter (host; see the module docstring)."""
    if cfg.cpp.EXCLUDE_FFIELDS_LOAD:                                            # :53
        raise NotImplementedError("EXTERNAL_FIELDS_LOAD: EXCLUDE_FFIELDS_LOAD is not ported")
    if not fp.periodicExternalForcing:                                          # :63
        return ff
    if cfg.cpp.SHORTWAVE_HEATING and fp.surfQswFile.strip():                    # :174-187, :227-230, :306-313
        raise NotImplementedError("EXTERNAL_FIELDS_LOAD: SHORTWAVE_HEATING surfQswFile (Qsw0/Qsw1) is not ported")
    sz = cfg.size
    out = []

    def write(fmt, *vals):
        if stdout is not None:
            out.append(fortran_write(fmt, *vals))

    # :69-72
    intimeP, intime0, intime1, bWght, aWght = get_periodic_interval(
        fp.externForcingCycle, fp.externForcingPeriod, fp.deltaTClock, myTime)
    loadedRec = ff.loadedRec                                                    # loadedRec(bi,bj), :74-75
    if cfg.cpp.ALLOW_DEBUG and fp.debugLevel >= debLevB:                        # :76-85
        write("(A,I10,A,4I5,A,2F14.10)", " EXTERNAL_FIELDS_LOAD,", myIter, " : iP,iLd,i0,i1=", intimeP, loadedRec,
              intime0, intime1, " ; Wght=", float(bWght), float(aWght))
    if cfg.cpp.ALLOW_AUTODIFF:                                                  # :86-95
        if cfg.cpp.STORE_LOADEDREC_TEST:                                        # GOADK lane (global_ocean code_ad)
            reload = intime1 != loadedRec                                       # :94
        else:
            reload = (intime0 != intimeP) or (myIter == fp.nIter0)              # :92
    else:
        reload = intime1 != loadedRec                                           # :100

    new = {}
    if reload:
        if fp.debugLevel >= debLevZero:                                         # :105-112
            write("(A,I10,A,2(2I5,A))", " EXTERNAL_FIELDS_LOAD, it=", myIter, " : Reading new data, i0,i1=",
                  intime0, intime1, " (prev=", intimeP, loadedRec, " )")
        g = dict((n, getattr(ff, n)) for n in ff.names())
        if fp.zonalWindFile.strip():                                           # :114-119
            g["taux0"] = rw.READ_REC_XY_RS(fp.zonalWindFile, g["taux0"], intime0, myIter)
            g["taux1"] = rw.READ_REC_XY_RS(fp.zonalWindFile, g["taux1"], intime1, myIter)
        if fp.meridWindFile.strip():                                           # :120-125
            g["tauy0"] = rw.READ_REC_XY_RS(fp.meridWindFile, g["tauy0"], intime0, myIter)
            g["tauy1"] = rw.READ_REC_XY_RS(fp.meridWindFile, g["tauy1"], intime1, myIter)
        if fp.surfQFile.strip():                                               # :126-130
            g["Qnet0"] = rw.READ_REC_XY_RS(fp.surfQFile, g["Qnet0"], intime0, myIter)
            g["Qnet1"] = rw.READ_REC_XY_RS(fp.surfQFile, g["Qnet1"], intime1, myIter)
        elif fp.surfQnetFile.strip():                                          # :131-136
            g["Qnet0"] = rw.READ_REC_XY_RS(fp.surfQnetFile, g["Qnet0"], intime0, myIter)
            g["Qnet1"] = rw.READ_REC_XY_RS(fp.surfQnetFile, g["Qnet1"], intime1, myIter)
        if fp.EmPmRfile.strip():                                               # :137-155
            g["EmPmR0"] = rw.READ_REC_XY_RS(fp.EmPmRfile, g["EmPmR0"], intime0, myIter)
            g["EmPmR1"] = rw.READ_REC_XY_RS(fp.EmPmRfile, g["EmPmR1"], intime1, myIter)
            j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
            i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
            g["EmPmR0"] = g["EmPmR0"].at[i, j].set(g["EmPmR0"][i, j]*fp.rhoConstFresh)    # :148
            g["EmPmR1"] = g["EmPmR1"].at[i, j].set(g["EmPmR1"][i, j]*fp.rhoConstFresh)    # :149
        if fp.saltFluxFile.strip():                                            # :156-161
            g["saltFlux0"] = rw.READ_REC_XY_RS(fp.saltFluxFile, g["saltFlux0"], intime0, myIter)
            g["saltFlux1"] = rw.READ_REC_XY_RS(fp.saltFluxFile, g["saltFlux1"], intime1, myIter)
        if fp.thetaClimFile.strip():                                           # :162-167
            g["SST0"] = rw.READ_REC_XY_RS(fp.thetaClimFile, g["SST0"], intime0, myIter)
            g["SST1"] = rw.READ_REC_XY_RS(fp.thetaClimFile, g["SST1"], intime1, myIter)
        if fp.saltClimFile.strip():                                            # :168-173
            g["SSS0"] = rw.READ_REC_XY_RS(fp.saltClimFile, g["SSS0"], intime0, myIter)
            g["SSS1"] = rw.READ_REC_XY_RS(fp.saltClimFile, g["SSS1"], intime1, myIter)
        if cfg.cpp.ATMOSPHERIC_LOADING and fp.pLoadFile.strip():               # :205-212
            g["pLoad0"] = rw.READ_REC_XY_RS(fp.pLoadFile, g["pLoad0"], intime0, myIter)
            g["pLoad1"] = rw.READ_REC_XY_RS(fp.pLoadFile, g["pLoad1"], intime1, myIter)
        # :214-234 exchanges (every call, in this order)
        g["SST0"] = EXCH_XY_RS(g["SST0"], ex=ex)                                # :215
        g["SST1"] = EXCH_XY_RS(g["SST1"], ex=ex)                                # :216
        g["SSS0"] = EXCH_XY_RS(g["SSS0"], ex=ex)                                # :217
        g["SSS1"] = EXCH_XY_RS(g["SSS1"], ex=ex)                                # :218
        g["taux0"], g["tauy0"] = EXCH_UV_XY_RS(g["taux0"], g["tauy0"], True, ex=ex)   # :219
        g["taux1"], g["tauy1"] = EXCH_UV_XY_RS(g["taux1"], g["tauy1"], True, ex=ex)   # :220
        g["Qnet0"] = EXCH_XY_RS(g["Qnet0"], ex=ex)                              # :221
        g["Qnet1"] = EXCH_XY_RS(g["Qnet1"], ex=ex)                              # :222
        g["EmPmR0"] = EXCH_XY_RS(g["EmPmR0"], ex=ex)                            # :223
        g["EmPmR1"] = EXCH_XY_RS(g["EmPmR1"], ex=ex)                            # :224
        g["saltFlux0"] = EXCH_XY_RS(g["saltFlux0"], ex=ex)                      # :225
        g["saltFlux1"] = EXCH_XY_RS(g["saltFlux1"], ex=ex)                      # :226
        if cfg.cpp.ATMOSPHERIC_LOADING:                                         # :231-234
            g["pLoad0"] = EXCH_XY_RS(g["pLoad0"], ex=ex)
            g["pLoad1"] = EXCH_XY_RS(g["pLoad1"], ex=ex)
        new = {k: g[k] for k in g if g[k] is not getattr(ff, k)}
        ff = ff.replace(loadedRec=intime1, **new)                               # :236-241 loadedRec = intime1

    ff = _interpolate(ff, float(bWght), float(aWght), cfg=cfg, fp=fp)              # :246-337

    if cfg.cpp.ALLOW_DEBUG and fp.debugLevel >= debLevC:                        # :340-352 (point (1,sNy) of tile 1)
        p = lambda f: float(np.asarray(f.data)[0, sz.sNy - 1 + sz.OLy, sz.OLx])   # noqa: E731
        write("(A,1P4E12.4)", " EXTERNAL_FIELDS_LOAD: (fu0,1),fu,fv=", p(ff.taux0), p(ff.taux1), p(ff.fu), p(ff.fv))
        write("(A,1P4E12.4)", " EXTERNAL_FIELDS_LOAD: SST,SSS,Q,E-P=", p(ff.SST), p(ff.SSS), p(ff.Qnet),
              p(ff.EmPmR))
    if stdout is not None:
        stdout.extend(out)
    return ff


def _interpolate(ff, b, a, *, cfg, fp):
    """external_fields_load.F:246-337: the time interpolation of the two loaded records with the weights bWght (`b`)
    and aWght (`a`): host floats (eager calls) or traced float64 (the preloaded scan path, same operations)."""
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    upd = {}
    if fp.thetaClimFile.strip():                                               # :249-256
        upd["SST"] = ff.SST.at[i, j].set(b*ff.SST0[i, j] + a*ff.SST1[i, j])
    if fp.saltClimFile.strip():                                                # :257-264
        upd["SSS"] = ff.SSS.at[i, j].set(b*ff.SSS0[i, j] + a*ff.SSS1[i, j])
    if fp.zonalWindFile.strip():                                               # :265-272
        upd["fu"] = ff.fu.at[i, j].set(b*ff.taux0[i, j] + a*ff.taux1[i, j])
    if fp.meridWindFile.strip():                                               # :273-280
        upd["fv"] = ff.fv.at[i, j].set(b*ff.tauy0[i, j] + a*ff.tauy1[i, j])
    if fp.surfQnetFile.strip() or fp.surfQFile.strip():                       # :281-289
        upd["Qnet"] = ff.Qnet.at[i, j].set(b*ff.Qnet0[i, j] + a*ff.Qnet1[i, j])
    if fp.EmPmRfile.strip():                                                   # :290-297
        upd["EmPmR"] = ff.EmPmR.at[i, j].set(b*ff.EmPmR0[i, j] + a*ff.EmPmR1[i, j])
    if fp.saltFluxFile.strip():                                                # :298-305
        upd["saltFlux"] = ff.saltFlux.at[i, j].set(b*ff.saltFlux0[i, j] + a*ff.saltFlux1[i, j])
    if cfg.cpp.ATMOSPHERIC_LOADING and fp.pLoadFile.strip():                  # :326-334
        upd["pLoad"] = ff.pLoad.at[i, j].set(b*ff.pLoad0[i, j] + a*ff.pLoad1[i, j])
    return ff.replace(**upd)


# ---- R5: the periodic forcing of a traced time loop (plan Task 16) ---------------------------------------------------

# the record arrays EXTERNAL_FIELDS_LOAD reads into (and exchanges) on a reload, :113-234
SLOTS = ("taux0", "taux1", "tauy0", "tauy1", "Qnet0", "Qnet1", "EmPmR0", "EmPmR1", "saltFlux0", "saltFlux1",
         "SST0", "SST1", "SSS0", "SSS1", "pLoad0", "pLoad1")


@jax.tree_util.register_pytree_node_class
class ForcingPreload:
    """EXTERNAL_FIELDS_LOAD's host part for a whole run, done before the time loop: `slots[name]` the record arrays
    after each reload (a tuple with one tiled FArray per reload, each exactly what the Fortran's reads and exchanges
    leave in the FFIELDS.h record array; FArrays so that a TileSharding places them like every tiled field), and per
    step iloop (index 0 unused): `irec[iloop]` the set the step sees, `bWght`, `aWght` the interpolation weights. A
    pytree: jit argument of the scan (never closed over)."""

    def __init__(self, slots, irec, bWght, aWght):
        self.slots, self.irec, self.bWght, self.aWght = slots, irec, bWght, aWght

    def tree_flatten(self):
        keys = tuple(sorted(self.slots))
        return (tuple(self.slots[k] for k in keys), self.irec, self.bWght, self.aWght), keys

    @classmethod
    def tree_unflatten(cls, keys, leaves):
        sl, irec, b, a = leaves
        return cls(dict(zip(keys, sl)), irec, b, a)


def preload_periodic_forcing(ff, nTimeSteps, *, cfg, fp, rw, ex, nIter0, startTime, deltaTClock, stdout=None):
    """Run EXTERNAL_FIELDS_LOAD's record logic on the host for steps iloop = 1..nTimeSteps of a run that starts
    from `ff` (FFIELDS.h after INI_FORCING): at each step the start-of-step clock (forward_step.F:429-430 in an
    ALLOW_AUTODIFF build, myIter = nIter0 + iloop-1, myTime = startTime + deltaTClock*(iloop-1); the plain build's
    carried counters hold the same values, the_main_loop.step_start_counters), the eager `external_fields_load`
    (the reload decision, the reads and the exchanges of :63-241, its STDOUT records into `stdout`), keeping the
    record arrays after every reload as one set. Returns (ForcingPreload, ff after the last step's loads).
    The weights are the step's GET_PERIODIC_INTERVAL values (:69-72); a step without a reload sees the set of the
    last reload, as the Fortran's record arrays keep their values between reloads."""
    if not fp.periodicExternalForcing:
        raise ValueError("preload_periodic_forcing: periodicExternalForcing is .FALSE.")
    sets, irec, bW, aW = [], [0], [0.], [0.]
    last = None
    for iloop in range(1, nTimeSteps + 1):
        myIter = nIter0 + (iloop - 1)                                            # forward_step.F:429
        myTime = float(np.float64(startTime) + np.float64(deltaTClock) * np.float64(iloop - 1))   # :430
        _, _, _, b, a = get_periodic_interval(fp.externForcingCycle, fp.externForcingPeriod, fp.deltaTClock,
                                              myTime)
        ff = external_fields_load(myTime, myIter, ff, cfg=cfg, fp=fp, rw=rw, ex=ex, stdout=stdout)
        cur = tuple(getattr(ff, n) for n in SLOTS if n in ff)
        if last is None or any(x is not y for x, y in zip(cur, last)):
            sets.append({n: getattr(ff, n) for n in SLOTS if n in ff})
            last = cur
        irec.append(len(sets) - 1)
        bW.append(float(b))
        aW.append(float(a))
    slots = {n: tuple(s[n] for s in sets) for n in sets[0]}
    return ForcingPreload(slots, jnp.asarray(irec, jnp.int32), jnp.asarray(bW, jnp.float64),
                          jnp.asarray(aW, jnp.float64)), ff


def external_fields_load_preloaded(iloop, ff, *, cfg, fp, pre):
    """EXTERNAL_FIELDS_LOAD in a traced time loop: the record arrays of step `iloop` (traced int) from the host
    preload (`preload_periodic_forcing`: the reads, exchanges and reload decisions of :63-241 already done), then
    the interpolation :246-337 with the step's weights. `loadedRec` (static) is left as it is: under preloading the
    reload bookkeeping is the host's."""
    if not fp.periodicExternalForcing:                                          # :63
        return ff
    k = pre.irec[iloop]
    new = {n: FArray(jnp.stack([f.data for f in pre.slots[n]])[k], n, tiled=getattr(ff, n).tiled,
                     _dims=getattr(ff, n).dims) for n in pre.slots}
    ff = ff.replace(**new)
    return _interpolate(ff, pre.bWght[iloop], pre.aWght[iloop], cfg=cfg, fp=fp)  # :246-337
