"""MONITOR   @63cdc0b pkg/monitor/monitor.F:8-204

The host-side boundary of the monitor: MONITOR is called by the driver on the concrete state (after initialisation,
model/src/initialise_varia.F:383, and at the end of every step, forward_step.F:1154), outside any differentiated or
jitted program. Its per-point expressions run as eager jax.numpy on the state arrays, the ordered sums through
mitjax/eesupp, extrema and the final scalar arithmetic on the host (numpy float64), and the text through
mitjax/io/fortran_format.py. Nothing it computes flows back into the model state; mon_trAdvCFL (set in
THERMODYNAMICS by mon_calc_advcfl.py) is only printed.

Input interface (Fortran names; to be reconciled with mitjax/model/state.py of the core lane):
* cfg (static): sNx, sNy, OLx, OLy, Nr, nSx, nSy (SIZE.h); monitorSelect, monitor_stdio, usingPCoords,
  usingSphericalPolarGrid, useCubedSphereExchange, fluidIsAir, fluidIsWater, useCoriolis, selectCoriMap,
  nonHydrostatic, select_rStar, useMNC, monitor_mnc, useAIM (PARAMS.h / EEPARAMS.h / MNC_PARAMS.h); CPP flags
  ALLOW_MNC, ALLOW_NONHYDROSTATIC, ALLOW_AIM, NONLIN_FRSURF, MONITOR_TEST_HFACZ.
* params (floats): monitorFreq, deltaTClock, deltaTMom, dTtracerLev (list of Nr), rUnit2mass.
* grid (FArrays with their GRID.h / SURFACE.h declarations): rA, rAw, rAs, rAz, recip_rA, recip_rAz, dxC, dyC, dxG,
  dyG, recip_dxC, recip_dyC, hFacC, hFacW, hFacS, recip_hFacC, maskC, maskInC, maskInW, maskInS, fCoriG, yG,
  kSurfC (integer), Bo_surf, h0FacC (r* only); 1-D: drF, drC, recip_drF, recip_drC, deepFacC, deepFac2C, deepFac2F,
  recip_deepFac2C, rhoFacC, rhoFacF, recip_rhoFacC.
* state (FArrays): uVel, vVel, wVel, theta, salt, etaN (DYNVARS.h); Qnet, Qsw, EmPmR, fu, fv, phi0surf (FFIELDS.h);
  rStarDhCDt (SURFACE.h, r* only).
* mon: MonitorCommon (MONITOR.h), set up by mon_init.
"""

from mitjax.farray import FArray
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.monitor.mon_advcfl import mon_advcfl
from mitjax.pkg.monitor.mon_advcflw import mon_advcflw
from mitjax.pkg.monitor.mon_advcflw2 import mon_advcflw2
from mitjax.ops.fortran_minmax_host import MAX
from mitjax.pkg.monitor.mon_ke import mon_ke
from mitjax.pkg.monitor.mon_out import mon_out_i, mon_out_rl
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.mon_solution import mon_solution
from mitjax.pkg.monitor.mon_surfcor import mon_surfcor
from mitjax.pkg.monitor.mon_vort3 import mon_vort3
from mitjax.pkg.monitor.mon_writestats_rl import mon_writestats_rl
from mitjax.pkg.monitor.mon_writestats_rs import mon_writestats_rs
from mitjax.eesupp.different_multiple import different_multiple
from mitjax.eesupp.print import SQUEEZE_RIGHT, print_message
from mitjax.pkg.monitor.monitor_h import mon_foot_max, mon_string_none


def level_view(fld, name=None):
    """A 2-D field (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy) passed to a dummy declared (...,myNr,nSx,nSy) with myNr=1:
    the same storage with a k axis of length 1 (Fortran passes the address; nothing is copied in the Fortran)."""
    (_, ilo, ihi), (_, jlo, jhi) = [d for d in fld.dims if d[0] == "i"][0], [d for d in fld.dims if d[0] == "j"][0]
    return FArray(fld.data[:, None], name or fld.name, i=(ilo, ihi), j=(jlo, jhi), k=(1, 1))


def level_slice(fld, k, name=None):
    """`fld(1-OLx,1-OLy,k,1,1)` passed to a dummy with myNr=1 (monitor.F:127, 129): level k of a 3-D field."""
    (_, ilo, ihi), (_, jlo, jhi) = [d for d in fld.dims if d[0] == "i"][0], [d for d in fld.dims if d[0] == "j"][0]
    (_, klo, _) = [d for d in fld.dims if d[0] == "k"][0]
    return FArray(fld.data[:, k - klo:k - klo + 1], name or fld.name, i=(ilo, ihi), j=(jlo, jhi), k=(1, 1))


def vec_head(vec, n, name=None):
    """The first n entries of a 1-D array passed to a dummy declared (n) (e.g. drF to arrDr(myNr) with myNr=1)."""
    (_, klo, _) = vec.dims[0]
    return FArray(vec.data[:n], name or vec.name, k=(1, n), tiled=False)


def monitor(myTime, myIter, *, cfg, params, grid, state, mon, ex=None):
    """MONITOR( myTime, myIter, myThid )

    C     Monitor key dynamical variables: calculate over the full domain
    C      some simple statistics (e.g., min,max,average) and write them.

    MASTER_CPU_IO is true (one process, one thread). #ifdef ALLOW_MNC (:60-74): monitor output to MNC (useMNC and
    monitor_mnc) is not ported and raises. Records go to mon.units[mon.mon_ioUnit]."""
    g, s = grid, state
    Nr = cfg.Nr
    if not different_multiple(params.monitorFreq, myTime, params.deltaTClock):    # :48
        return

    mon.mon_write_stdout = bool(cfg.monitor_stdio)                  # :54-58
    mon.mon_write_mnc = False                                       # :59
    if cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc:            # :60-74
        raise NotImplementedError("MONITOR: monitor output to MNC (useMNC with monitor_mnc) is not ported")
    if mon.mon_write_stdout:                                        # :77-87
        _banner("// Begin MONITOR dynamic field statistics", mon)

    thickFacC = FArray(g.drF.data*g.deepFac2C.data*g.rhoFacC.data,           # :93-96
                       "thickFacC", k=(1, Nr), tiled=False)
    thickFacF = FArray(g.drC.data[:Nr]*g.deepFac2F.data[:Nr]*g.rhoFacF.data[:Nr],
                       "thickFacF", k=(1, Nr), tiled=False)
    dummyRL = [0.0] * 6

    mon_set_pref("time", mon=mon)                                   # :99-101
    mon_out_i("_tsnumber", myIter, mon_string_none, cfg=cfg, mon=mon)
    mon_out_rl("_secondsf", myTime, mon_string_none, cfg=cfg, mon=mon)

    statsTemp = [0.0] * 6
    if cfg.monitorSelect >= 1:                                      # :103-121
        mon_set_pref("dynstat", mon=mon)
        dummyRL = mon_writestats_rl(1, level_view(s.etaN), "_eta", level_view(g.maskInC), g.maskInC, g.rA,
                                    vec_head(g.drF, 1), dummyRL, cfg=cfg, mon=mon, ex=ex)
        dummyRL = mon_writestats_rl(Nr, s.uVel, "_uvel", g.hFacW, g.maskInW, g.rAw, thickFacC, dummyRL,
                                    cfg=cfg, mon=mon, ex=ex)
        dummyRL = mon_writestats_rl(Nr, s.vVel, "_vvel", g.hFacS, g.maskInS, g.rAs, thickFacC, dummyRL,
                                    cfg=cfg, mon=mon, ex=ex)
        dummyRL = mon_writestats_rl(Nr, s.wVel, "_wvel", g.maskC, g.maskInC, g.rA, thickFacF, dummyRL,
                                    cfg=cfg, mon=mon, ex=ex)
        statsTemp = mon_writestats_rl(Nr, s.theta, "_theta", g.hFacC, g.maskInC, g.rA, thickFacC, statsTemp,
                                      cfg=cfg, mon=mon, ex=ex)
        dummyRL = mon_writestats_rl(Nr, s.salt, "_salt", g.hFacC, g.maskInC, g.rA, thickFacC, dummyRL,
                                    cfg=cfg, mon=mon, ex=ex)
    else:
        statsTemp[0] = 1.0
        statsTemp[1] = 0.0
    if cfg.monitorSelect >= 3 and cfg.nSx == 1 and cfg.nSy == 1:     # :122-131
        k = 1
        if cfg.usingPCoords:
            k = Nr
        dummyRL = mon_writestats_rl(1, level_slice(s.theta, k), "_sst", level_view(g.maskInC), g.maskInC, g.rA,
                                    vec_head(g.drF, 1), dummyRL, cfg=cfg, mon=mon, ex=ex)
        dummyRL = mon_writestats_rl(1, level_slice(s.salt, k), "_sss", level_view(g.maskInC), g.maskInC, g.rA,
                                    vec_head(g.drF, 1), dummyRL, cfg=cfg, mon=mon, ex=ex)

    if cfg.monitorSelect >= 3:                                      # :134-146
        mon_set_pref("forcing", mon=mon)
        for fld, name, msk, area in ((s.Qnet, "_qnet", g.maskInC, g.rA), (s.Qsw, "_qsw", g.maskInC, g.rA),
                                     (s.EmPmR, "_empmr", g.maskInC, g.rA), (s.fu, "_fu", g.maskInW, g.rAw),
                                     (s.fv, "_fv", g.maskInS, g.rAs)):
            dummyRL = mon_writestats_rs(1, level_view(fld), name, level_view(msk), msk, area,
                                        vec_head(g.drF, 1), dummyRL, cfg=cfg, mon=mon, ex=ex)

    if cfg.monitorSelect >= 2:                                      # :149-154
        mon_set_pref("trAdv_CFL", mon=mon)
        mon_out_rl("_u", mon.mon_trAdvCFL[0], mon_foot_max, cfg=cfg, mon=mon)
        mon_out_rl("_v", mon.mon_trAdvCFL[1], mon_foot_max, cfg=cfg, mon=mon)
        mon_out_rl("_w", mon.mon_trAdvCFL[2], mon_foot_max, cfg=cfg, mon=mon)

    mon_set_pref("advcfl", mon=mon)                                 # :157-163
    dT = MAX(params.dTtracerLev[0], params.deltaTMom, p="a")        # :158
    mon_advcfl("_uvel", s.uVel, g.recip_dxC, dT, cfg=cfg, mon=mon, ex=ex)
    mon_advcfl("_vvel", s.vVel, g.recip_dyC, dT, cfg=cfg, mon=mon, ex=ex)
    mon_advcflw("_wvel", s.wVel, vec_head(g.recip_drC, Nr), dT, cfg=cfg, mon=mon, ex=ex)
    mon_advcflw2("_W_hf", s.wVel, g.recip_hFacC, g.recip_drF, dT, cfg=cfg, mon=mon, ex=ex)

    mon_ke(myIter, cfg=cfg, grid=g, state=s, mon=mon, ex=ex, params=params)   # :166 (params: lane B, AM)

    if cfg.monitorSelect >= 2:                                      # :169
        mon_vort3(myIter, cfg=cfg, grid=g, state=s, mon=mon, ex=ex)
    if cfg.monitorSelect >= 2:                                      # :172
        mon_surfcor(cfg=cfg, params=params, grid=g, state=s, mon=mon, ex=ex)

    mon_solution(statsTemp, myTime, myIter, cfg=cfg, grid=g, state=s, mon=mon, ex=ex)   # :175

    if mon.mon_write_stdout:                                        # :182-192
        _banner("// End MONITOR dynamic field statistics", mon)
    mon.mon_write_stdout = False                                    # :194-195
    mon.mon_write_mnc = False


def _banner(text, mon):
    """The three PRINT_MESSAGE records around the block (monitor.F:78-86, 183-191)."""
    line = fortran_write("(2A)", "// ==========================", "=============================")
    print_message(line, mon.mon_ioUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)
    print_message(fortran_write("(A)", text), mon.mon_ioUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)
    print_message(line, mon.mon_ioUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)
