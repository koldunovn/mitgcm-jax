"""RBCS_FIELDS_LOAD: pkg/rbcs/rbcs_fields_load.F @63cdc0b (M3 Task 30)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XYZ_RS
from mitjax.farray import FArray, loops_kji
from mitjax.pkg.rbcs.rbcs_fields_h import rbcs_array


def _intervals(params):
    """(intimeP, intime0, intime1, bWght, aWght) of :72-94 for rbcsForcingPeriod = 0 (:88-94); the periodic branch
    (GET_PERIODIC_INTERVAL, :72-87) raises."""
    if params.rbcsForcingPeriod > 0.0:                                  # :72-87
        raise NotImplementedError("RBCS_FIELDS_LOAD: periodic relaxation records (rbcsForcingPeriod > 0) are not "
                                  "ported")
    return 1, 1, 1, 0.5, 0.5                                            # :89-93  aWght = bWght = .5 _d 0


def rbcs_preload(*, cfg, params, rw):
    """The host part of RBCS_FIELDS_LOAD: the records its reads return (READ_REC_XYZ_RS of relaxTFile, records
    intime0 and intime1, :164-165; MDS_READ_FIELD( relaxTFile, readBinaryPrec, .FALSE., 'RS', Nr, 1, Nr, dummyRL,
    field, iRec, myThid ), read_rec.F:137-188), as the interior of NaN-halo arrays; the step's reload copies the
    interior into rbct0 / rbct1 as MDS_READ_FIELD does. Returns {"rbct0": rec, "rbct1": rec} (empty without
    useRBCtemp or relaxTFile). Not ported (raise): useRBCuVel/vVel files (:125-155), useRBCsalt files (:170-183),
    rbcsSingleTimeFiles (:157-162)."""
    sz = cfg.size
    _, intime0, intime1, _, _ = _intervals(params)
    if (params.useRBCuVel and str(params.relaxUFile).strip()) or (params.useRBCvVel and str(params.relaxVFile).strip()):
        raise NotImplementedError("RBCS_FIELDS_LOAD: the U/V relaxation records are not ported")   # :125-155
    if params.useRBCsalt and str(params.relaxSFile).strip():           # :170-183
        raise NotImplementedError("RBCS_FIELDS_LOAD: the salinity relaxation records are not ported")
    out = {}
    if params.useRBCtemp and str(params.relaxTFile).strip():           # :156-169
        if params.rbcsSingleTimeFiles:                                  # :157-162
            raise NotImplementedError("RBCS_FIELDS_LOAD: rbcsSingleTimeFiles is not ported")
        name = str(params.relaxTFile).strip()
        out["rbct0"] = rw.mds_read_field(name, rw.readBinaryPrec, sz.Nr, rbcs_array("rbct0", sz), intime0)  # :164
        out["rbct1"] = rw.mds_read_field(name, rw.readBinaryPrec, sz.Nr, rbcs_array("rbct1", sz), intime1)  # :165
    return out


def rbcs_fields_load(myTime, myIter, *, cfg, params, rbcs, pre, ex):
    """RBCS_FIELDS_LOAD( myTime, myIter, myThid )   @63cdc0b pkg/rbcs/rbcs_fields_load.F:5-284

    C     | SUBROUTINE RBCS_FIELDS_LOAD
    C     | o Control reading of fields from external source.

    `rbcs`: the RBCS_FIELDS.h state (rbcs_init_varia's dict, a carry); `pre`: rbcs_preload's records. The load test
    of the non-ALLOW_AUTODIFF_TAMC form (:106: intime1 .NE. rbcsLdRec(bi,bj) of the first tile, :70-71) is traced:
    when it holds, rbct0 / rbct1 take the records' interior (READ_REC_XYZ_RS, :164-165), EXCH_XYZ_RS (:167-168), and
    rbcsLdRec = intime1 on every tile (:218-222); then RBCtemp = bWght*rbct0 + aWght*rbct1 and RBCsalt likewise on
    every point (:244-253). ALLOW_AUTODIFF_TAMC (:101 the other load test) raises; the "Reading new data" message
    (:111-118, debugLevel >= debLevZero) is the host's (not printed here). Returns the new state."""
    if cfg.cpp.ALLOW_AUTODIFF_TAMC:                                     # :96-101
        raise NotImplementedError("RBCS_FIELDS_LOAD: the ALLOW_AUTODIFF_TAMC load test is not ported")
    sz = cfg.size
    _, _, intime1, bWght, aWght = _intervals(params)                    # :72-94
    k3, j3, i3 = loops_kji((1, sz.Nr), (1 - sz.OLy, sz.sNy + sz.OLy), (1 - sz.OLx, sz.sNx + sz.OLx))
    out = dict(rbcs)
    load = rbcs["rbcsLdRec"][0] != intime1                              # :70-71, :106  bi = myBxLo, bj = myByLo
    if "rbct0" in pre:                                                  # :156-169
        ki, ji, ii = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))
        for n in ("rbct0", "rbct1"):
            rd = rbcs[n].at[ii, ji, ki].set(pre[n][ii, ji, ki])         # :164-165 READ_REC_XYZ_RS (the interior)
            rd = EXCH_XYZ_RS(rd, ex=ex)                                 # :167-168
            out[n] = FArray(jnp.where(load, rd.data, rbcs[n].data), rd.name, tiled=rd.tiled, _dims=rd.dims)
    out["rbcsLdRec"] = jnp.where(load, jnp.full_like(rbcs["rbcsLdRec"], intime1), rbcs["rbcsLdRec"])   # :218-222
    if not cfg.cpp.DISABLE_RBCS_MOM and (params.useRBCuVel or params.useRBCvVel):  # :230-242
        raise NotImplementedError("RBCS_FIELDS_LOAD: RBCuVel / RBCvVel are not ported")
    out["RBCtemp"] = out["RBCtemp"].at[i3, j3, k3].set(bWght * out["rbct0"][i3, j3, k3]
                                                       + aWght * out["rbct1"][i3, j3, k3])          # :247-248
    out["RBCsalt"] = out["RBCsalt"].at[i3, j3, k3].set(bWght * out["rbcs0"][i3, j3, k3]
                                                       + aWght * out["rbcs1"][i3, j3, k3])          # :249-250
    return out
