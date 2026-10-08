"""SEAICE_TRACER_PHYS: pkg/seaice/seaice_tracer_phys.F @63cdc0b (lane M4LAB session 4: ALLOW_SITRACER of
lab_sea/code)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.safe import safe_div


def seaice_tracer_phys(myTime, myIter, sf, *, cfg, sp):
    """SEAICE_TRACER_PHYS( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_tracer_phys.F:7-274

    C     | o Time step SItr/SItrEFF as a result of
    C     |   seaice thermodynamics and specific tracer physics

    `sf` SEAICE.h with SEAICE_TRACER.h (SItracer, SItrBucket, and the SItrHEFF / SItrAREA snapshots SEAICE_GROWTH
    took). Returns sf with SItracer and SItrBucket of the tracers in use (DO iTr = 1, SItrNumInUse, :63), interior.
    Per tracer: the exchange values SItrFromOcean0 / SItrFromFlood0 / SItrExpand0 (:67-73); with SItrMate 'HEFF' the
    three thermodynamic increments and the flooding (:86-132: growth dilutes by HEFFprev/HEFFpost and brings
    SItrFromOcean, melt moves meltPart*SItracer into SItrBucket), else the cover expansion (:142-154, expandFact =
    AREAprev/AREApost where the cover grows); the 'age' tracer (:158-171) grows by SEAICE_deltaTtherm where its mate
    (SItrHEFF(5) or SItrAREA(3)) is > 0, else 0; 'one' (:174-175) has no process of its own; then SItrBucket is
    emptied (:220-231; 'grease' keeps it). The point loops are independent: vectorised; the IF tests are wheres with
    the divisions guarded on their arm (safe_div). Raises: SItrName 'salinity' (:75-83, :203-218: it changes
    saltFlux / saltPlumeFlux) and 'ridge' (:176-190) (SEAICE_INIT_FIXED refuses them first),
    ALLOW_SITRACER_DEBUG_DIAG (:51-53, :90-93, ...: diagnostics only). The ALLOW_DIAGNOSTICS fill (:194-201) is
    output only. REAL literals `1. _d 0` / `0. _d 0` double."""
    if cfg.cpp.flag("ALLOW_SITRACER_DEBUG_DIAG", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_TRACER_PHYS: ALLOW_SITRACER_DEBUG_DIAG is not ported")
    sz = cfg.size
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    sf = dict(sf)
    SItrHEFF, SItrAREA = sf["SItrHEFF"], sf["SItrAREA"]
    for iTr in range(1, sp.SItrNumInUse + 1):                                  # :63
        name, mate = sp.SItrName[iTr-1], sp.SItrMate[iTr-1]
        if name in ("salinity", "ridge"):
            raise NotImplementedError(f"SEAICE_TRACER_PHYS: SItrName = '{name}' is not ported")
        SItracer = sf["SItracer"][i, j, iTr]
        SItrBucket = sf["SItrBucket"][i, j, iTr]
        # 0) set ice-ocean and ice-snow exchange values (:67-73; the 'salinity' arm :75-83 raised above)
        SItrFromOcean = sp.SItrFromOcean0[iTr-1]                               # :69
        SItrFromFlood = sp.SItrFromFlood0[iTr-1]                               # :70
        SItrExpand = sp.SItrExpand0[iTr-1]                                     # :71
        # 1) seaice thermodynamics processes
        if mate == "HEFF":                                                     # :86
            for jTh in range(1, 3 + 1):                                        # :100
                HEFFprev = SItrHEFF[i, j, jTh]                                 # :101
                HEFFpost = SItrHEFF[i, j, jTh+1]                               # :102
                grow = HEFFpost > HEFFprev                                     # :106
                growFact = jnp.where(grow, safe_div(HEFFprev, HEFFpost, grow), 1.)   # :104, :106
                meltPart = jnp.where(HEFFpost < HEFFprev, HEFFprev-HEFFpost, 0.)   # :105, :107
                SItracer = (SItracer*growFact                                  # :109-110
                            + SItrFromOcean*(1. - growFact))
                SItrBucket = (SItrBucket                                       # :111-112
                              - HEFFpost*SItrFromOcean*(1. - growFact))
                SItrBucket = (SItrBucket                                       # :113-114
                              + meltPart*SItracer)
            HEFFprev = SItrHEFF[i, j, 4]                                       # :118
            HEFFpost = SItrHEFF[i, j, 5]                                       # :119
            grow = HEFFpost > HEFFprev                                         # :120
            growFact = jnp.where(grow, safe_div(HEFFprev, HEFFpost, grow), 1.)   # :117, :120
            SItracer = (SItracer*growFact                                      # :121-122
                        + SItrFromFlood*(1. - growFact))
            SItrBucket = (SItrBucket                                           # :125-126
                          - HEFFpost*SItrFromFlood*(1. - growFact))
        else:                                                                  # :134-155
            AREAprev = SItrAREA[i, j, 2]                                       # :145
            AREApost = SItrAREA[i, j, 3]                                       # :146
            grow = AREApost > AREAprev                                         # :149
            expandFact = jnp.where(grow, safe_div(AREAprev, AREApost, grow), 1.)   # :148-149
            SItracer = (SItracer*expandFact                                    # :151-152
                        + SItrExpand*(1. - expandFact))
        # 2) very ice tracer processes
        if name == "age":                                                      # :158-171
            alive = (((SItrHEFF[i, j, 5] > 0.) & (mate == "HEFF"))             # :162-164
                     | ((SItrAREA[i, j, 3] > 0.) & (mate == "AREA")))
            SItracer = jnp.where(alive, SItracer + sp.SEAICE_deltaTtherm, 0.)   # :165-168
        # 'one' (:174-175): no specific process
        # 3) ice-ocean exchange (:194-218: the diagnostics fill is output only; 'salinity' raised above)
        if name != "grease":                                                   # :228-229 empty bucket
            SItrBucket = jnp.zeros_like(SItrBucket)
        sf["SItracer"] = sf["SItracer"].at[i, j, iTr].set(SItracer)
        sf["SItrBucket"] = sf["SItrBucket"].at[i, j, iTr].set(SItrBucket)
    return sf
