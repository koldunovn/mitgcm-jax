"""SEAICE_REG_RIDGE: pkg/seaice/seaice_reg_ridge.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.pkg.seaice.seaice_params_h import ONE, siEps


def seaice_reg_ridge(myTime, myIter, sf, *, cfg, sp, op):
    """SEAICE_REG_RIDGE( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_reg_ridge.F:12-404

    C     | o this routine has two purposes:
    C     |   (1) clean up after advection (undershoots etc.);
    C     |   (2) driver for ice ridging;

    `sf` SEAICE.h (dict). Returns sf with d_HEFFbyNEG, d_HSNWbyNEG, saltFluxAdjust (reset on all points :97-109),
    HEFF, HSNOW, AREA, TICES (IT = 1..SEAICE_multDim), HSALT, SItrAREA.
    Ported: the no-ITD build (:173-187, :248-258, :280-290, :365-383) with or (lane M4OFF) without
    SEAICE_VARIABLE_SALINITY (:105, :295-309),
    DISABLE_AREA_FLOOR undefined (:263), ALLOW_SITRACER's SItrAREA(:,:,1) (:376-378, lane M4LAB session 4);
    SEAICE_MODIFY_GROWTH_ADJ raises when compiled (ALLOW_AUTODIFF alone: no value, lane M4ADCOL).
    EXF_SEAICE_FRACTION (lane M4ADLAB session 3): d_AREAbyRLX / d_HEFFbyRLX reset on all points (:101-104); the
    relaxation towards exf_iceFraction (:120-148) runs only with SEAICE_tauAreaObsRelax > 0, which
    SEAICE_READPARMS refuses (not ported), so it is not taken. The point loops are independent: vectorised. The pointwise IFs (:222, :255,
    :286, :302) are wheres; every arm is finite (no division). REAL literals `0. _d 0` double, `0` / `0.0` exact."""
    for o in ("SEAICE_ITD", "SEAICE_MODIFY_GROWTH_ADJ", "DISABLE_AREA_FLOOR"):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_REG_RIDGE: {o} is not ported")
    # lane M4ADCOL (1D_ocean_ice_column/code_ad): ALLOW_AUTODIFF alone changes no value: its arms are the includes
    # (:5-6, :45-47), the tape key tkey (:79-82, :93-95) and CADJ directives; the two value arms (:115, :387) also
    # need SEAICE_MODIFY_GROWTH_ADJ, which raises above
    seaice_fraction = cfg.cpp.flag("EXF_SEAICE_FRACTION", "EXF_OPTIONS.h")   # lane M4ADLAB session 3
    salinity = cfg.cpp.flag("SEAICE_VARIABLE_SALINITY", "SEAICE_OPTIONS.h")    # lane M4OFF: both builds
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    sf = dict(sf)
    recip_deltaTtherm = ONE / sp.SEAICE_deltaTtherm                            # :88
    j = loop_j(1-OLy, sNy+OLy)                                                 # :97-109
    i = loop_i(1-OLx, sNx+OLx)
    sf["d_HEFFbyNEG"] = sf["d_HEFFbyNEG"].at[i, j].set(0.0)                    # :99
    sf["d_HSNWbyNEG"] = sf["d_HSNWbyNEG"].at[i, j].set(0.0)                    # :100
    if seaice_fraction:
        sf["d_AREAbyRLX"] = sf["d_AREAbyRLX"].at[i, j].set(0.0)                # :102
        sf["d_HEFFbyRLX"] = sf["d_HEFFbyRLX"].at[i, j].set(0.0)                # :103
    if salinity:
        sf["saltFluxAdjust"] = sf["saltFluxAdjust"].at[i, j].set(0.0)          # :106
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    # :179-187 remove negative values
    HEFF, HSNOW, AREA = sf["HEFF"], sf["HSNOW"], sf["AREA"]
    d_HEFFbyNEG = MAX(-HEFF[i, j], 0.0, p="a")                                 # :181
    HEFF = HEFF.at[i, j].set(HEFF[i, j]+d_HEFFbyNEG)                           # :182
    d_HSNWbyNEG = MAX(-HSNOW[i, j], 0.0, p="a")                                # :183
    HSNOW = HSNOW.at[i, j].set(HSNOW[i, j]+d_HSNWbyNEG)                        # :184
    AREA = AREA.at[i, j].set(MAX(AREA[i, j], 0.0, p="a"))                      # :185
    # :218-234 remove very thin ice
    thin = HEFF[i, j] <= siEps                                                 # :222
    tmpscal1 = jnp.where(thin, -HEFF[i, j], 0.0)                               # :220, :223
    tmpscal2 = jnp.where(thin, -HSNOW[i, j], 0.0)                              # :221, :224
    TICES = sf["TICES"]
    for IT in range(1, sp.SEAICE_multDim + 1):                                 # :225-227
        TICES = TICES.at[i, j, IT].set(jnp.where(thin, op.celsius2K, TICES[i, j, IT]))
    sf["TICES"] = TICES
    HEFF = HEFF.at[i, j].set(HEFF[i, j]+tmpscal1)                              # :229
    HSNOW = HSNOW.at[i, j].set(HSNOW[i, j]+tmpscal2)                           # :230
    sf["d_HEFFbyNEG"] = sf["d_HEFFbyNEG"].at[i, j].set(d_HEFFbyNEG+tmpscal1)   # :231
    sf["d_HSNWbyNEG"] = sf["d_HSNWbyNEG"].at[i, j].set(d_HSNWbyNEG+tmpscal2)   # :232
    # :253-258 no ice: no area
    AREA = AREA.at[i, j].set(jnp.where((HEFF[i, j] == 0.0) & (HSNOW[i, j] == 0.0), 0.0, AREA[i, j]))   # :255-256
    # :284-290 area floor
    AREA = AREA.at[i, j].set(jnp.where((HEFF[i, j] > 0) | (HSNOW[i, j] > 0),   # :286-288
                                       MAX(AREA[i, j], sp.SEAICE_area_floor, p="a"), AREA[i, j]))
    if salinity:                                                               # :300-309 SEAICE_VARIABLE_SALINITY
        HSALT = sf["HSALT"]
        neg = (HSALT[i, j] < 0.0) | (HEFF[i, j] == 0.0)                        # :302-303
        sf["saltFluxAdjust"] = sf["saltFluxAdjust"].at[i, j].set(              # :304-305
            jnp.where(neg, - sf["HEFFM"][i, j] * HSALT[i, j] * recip_deltaTtherm, sf["saltFluxAdjust"][i, j]))
        sf["HSALT"] = HSALT.at[i, j].set(jnp.where(neg, 0.0, HSALT[i, j]))     # :306
    # :370-383 area_max
    if cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h"):                    # :376-378 (lane M4LAB session 4)
        sf["SItrAREA"] = sf["SItrAREA"].at[i, j, 1].set(AREA[i, j])            # :377 pre-ridging AREA
    AREA = AREA.at[i, j].set(MIN(AREA[i, j], sp.SEAICE_area_max, p="a"))       # :381
    sf["HEFF"], sf["HSNOW"], sf["AREA"] = HEFF, HSNOW, AREA
    return sf
