"""ADMONITOR: pkg/monitor/monitor_ad.F @63cdc0b (the adjoint monitor of an adjoint build: `%MON ad_*` blocks).

In MITgcm the adjoint model calls ADMONITOR where the forward model called MONITOR, in the reverse sweep; it writes
statistics of the adjoint variables adEtaN, aduVel, advVel, adwVel, adTheta, adSalt at that point. Here the
adjoint variables are the cotangents of the State that the gradient driver hands over
(mitjax/drivers/adjoint_run.py: the step-boundary cotangents through grad.value_and_grad's stats hook, with
COST_TILE's contribution added, because COST_TILE follows MONITOR in FORWARD_STEP); with mon_AdVarExch = 2 (the
autodiff_readparms.F:76 default) the caller passes the copies COPY_ADVAR_OUTP / COPY_AD_UV_OUTP made (their
ADEXCH: mitjax/ad/adexch.py). This module writes the records only (the MONITOR.h state `mon`, as pkg/monitor).
"""

from mitjax.pkg.monitor.mon_out import mon_out_i, mon_out_rl
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.mon_writestats_rl import mon_writestats_rl
from mitjax.pkg.monitor.monitor import _banner, level_slice, level_view, vec_head
from mitjax.pkg.monitor.monitor_h import different_multiple, mon_string_none


def admonitor(myTime, myIter, *, cfg, params, grid, ad, mon, adjMonitorFreq, mon_AdVarExch=2, ex=None):
    """ADMONITOR( myTime, myIter, myThid )   @63cdc0b pkg/monitor/monitor_ad.F:9-305

    C     Monitor key dynamical variables: calculate over the full domain
    C      some simple statistics (e.g., min,max,average) and write them.
    (adjoint variables)

    `ad`: the adjoint fields after COPY_ADVAR_OUTP / COPY_AD_UV_OUTP (:162-180): adEtaN (var2Du, 2-D), aduVel,
    advVel, adwVel, adTheta, adSalt (3-D). `adjMonitorFreq` (PARAMS.h, data PARM03; set_defaults.F:352 `0.`).
    Ported: the mon_AdVarExch = 2 arm (:160-193, the default); PTRACERS lane: monitorSelect >= 4 (ad_forcing,
    :181-216, its mon_AdVarExch = 2 arm: `ad` then holds adQnet, adQsw, adEmPmR, adfu, adfv). Raise: mon_AdVarExch
    /= 2 (:131-159), ALLOW_MNC output (:79-93). The package calls after the block (ADSEAICE_MONITOR, :257-259: not
    compiled in M2 builds; ADPTRACERS_MONITOR :262-264: mitjax/pkg/ptracers/ptracers_monitor_ad.py, called by the
    driver after this routine)."""
    if mon_AdVarExch != 2:
        raise NotImplementedError("ADMONITOR: mon_AdVarExch /= 2 (:131-159) is not ported")
    if not different_multiple(adjMonitorFreq, myTime, params.deltaTClock):     # :76
        return
    mon.mon_write_stdout = bool(cfg.monitor_stdio)                            # :78-83
    mon.mon_write_mnc = False                                                 # :84
    if cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc:                      # :85-99
        raise NotImplementedError("ADMONITOR: monitor output to MNC is not ported")
    if mon.mon_write_stdout:                                                  # :101-111
        _banner("// Begin AD_MONITOR dynamic field statistics", mon)
    g = grid
    Nr = cfg.Nr
    dummyRL = [0.0] * 6
    mon_set_pref("ad_time", mon=mon)                                          # :115
    mon_out_i("_tsnumber", myIter, mon_string_none, cfg=cfg, mon=mon)         # :116
    mon_out_rl("_secondsf", myTime, mon_string_none, cfg=cfg, mon=mon)        # :117
    mon_set_pref("ad_dynstat", mon=mon)                                       # :119
    drF1 = vec_head(g.drF, 1)
    drF = vec_head(g.drF, Nr)
    drC = vec_head(g.drC, Nr)
    dummyRL = mon_writestats_rl(1, level_view(ad.adEtaN), "_adeta", level_view(g.maskInC), g.maskInC, g.rA,
                                drF1, dummyRL, cfg=cfg, mon=mon, ex=ex)       # :163-164
    dummyRL = mon_writestats_rl(Nr, ad.aduVel, "_aduvel", g.hFacW, g.maskInW, g.rAw, drF, dummyRL,
                                cfg=cfg, mon=mon, ex=ex)                      # :167-168
    dummyRL = mon_writestats_rl(Nr, ad.advVel, "_advvel", g.hFacS, g.maskInS, g.rAs, drF, dummyRL,
                                cfg=cfg, mon=mon, ex=ex)                      # :169-170
    dummyRL = mon_writestats_rl(Nr, ad.adwVel, "_adwvel", g.maskC, g.maskInC, g.rA, drC, dummyRL,
                                cfg=cfg, mon=mon, ex=ex)                      # :172-173
    dummyRL = mon_writestats_rl(Nr, ad.adTheta, "_adtheta", g.hFacC, g.maskInC, g.rA, drF, dummyRL,
                                cfg=cfg, mon=mon, ex=ex)                      # :176-177
    dummyRL = mon_writestats_rl(Nr, ad.adSalt, "_adsalt", g.hFacC, g.maskInC, g.rA, drF, dummyRL,
                                cfg=cfg, mon=mon, ex=ex)                      # :178-179
    if cfg.monitorSelect >= 3 and cfg.nSx == 1 and cfg.nSy == 1:              # :180-187
        k = 1
        if cfg.usingPCoords:
            k = Nr
        dummyRL = mon_writestats_rl(1, level_slice(ad.adTheta, k), "_adsst", level_view(g.maskInC), g.maskInC,
                                    g.rA, drF1, dummyRL, cfg=cfg, mon=mon, ex=ex)
        dummyRL = mon_writestats_rl(1, level_slice(ad.adSalt, k), "_adsss", level_view(g.maskInC), g.maskInC,
                                    g.rA, drF1, dummyRL, cfg=cfg, mon=mon, ex=ex)
    if cfg.monitorSelect >= 4:                                                # :181-216 (PTRACERS lane)
        # the mon_AdVarExch = 2 arm (:196-215): the caller passes the COPY_ADVAR_OUTP (vType 11) / COPY_AD_UV_OUTP
        # (vType 33) copies adQnet, adQsw (SHORTWAVE_HEATING; None otherwise), adEmPmR, adfu, adfv
        mon_set_pref("ad_forcing", mon=mon)                                   # :182
        mInC, mInW, mInS = level_view(g.maskInC), level_view(g.maskInW), level_view(g.maskInS)
        dummyRL = mon_writestats_rl(1, level_view(ad.adQnet), "_adqnet", mInC, g.maskInC, g.rA, drF1, dummyRL,
                                    cfg=cfg, mon=mon, ex=ex)                  # :198-200
        if ad.adQsw is not None:                                              # :201-205 #ifdef SHORTWAVE_HEATING
            dummyRL = mon_writestats_rl(1, level_view(ad.adQsw), "_adqsw", mInC, g.maskInC, g.rA, drF1, dummyRL,
                                        cfg=cfg, mon=mon, ex=ex)
        dummyRL = mon_writestats_rl(1, level_view(ad.adEmPmR), "_adempmr", mInC, g.maskInC, g.rA, drF1,
                                    dummyRL, cfg=cfg, mon=mon, ex=ex)         # :206-208
        dummyRL = mon_writestats_rl(1, level_view(ad.adfu), "_adfu", mInW, g.maskInW, g.rAw, drF1, dummyRL,
                                    cfg=cfg, mon=mon, ex=ex)                  # :209-212
        dummyRL = mon_writestats_rl(1, level_view(ad.adfv), "_adfv", mInS, g.maskInS, g.rAs, drF1, dummyRL,
                                    cfg=cfg, mon=mon, ex=ex)                  # :213-214
    if mon.mon_write_stdout:                                                  # :242-252
        _banner("// End AD_MONITOR dynamic field statistics", mon)
    mon.mon_write_stdout = False                                              # :253
    mon.mon_write_mnc = False                                                 # :254
    del dummyRL
