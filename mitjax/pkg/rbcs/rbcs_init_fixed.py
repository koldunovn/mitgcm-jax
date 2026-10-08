"""RBCS_INIT_FIXED: pkg/rbcs/rbcs_init_fixed.F @63cdc0b (M3 Task 30)."""

from mitjax.eesupp.exch_rs import EXCH_XYZ_RS
from mitjax.farray import loops_kji
from mitjax.pkg.rbcs.rbcs_fields_h import rbcs_array
from mitjax.pkg.rbcs.rbcs_readparms import maskLEN


def rbcs_init_fixed(*, cfg, params, grid, rw, ex):
    """RBCS_INIT_FIXED( myThid )   @63cdc0b pkg/rbcs/rbcs_init_fixed.F:4-324

    C     | S/R RBCS_INIT_FIXED
    C     | Initialize RBCS fixed variables (masks)

    `params`: Params with the RBCS_PARAMS.h values (rbcs_readparms). Returns {"RBC_mask": [maskLEN FArrays],
    "RBC_maskU", "RBC_maskV"} (RBCS_FIELDS.h): RBC_maskU/V = 0 (:189-204, DISABLE_RBCS_MOM undefined);
    RBC_mask(irbc) = 0 (:207-226), then for each relaxMaskFile set: READ_FLD_XYZ_RS, EXCH_XYZ_RS, times maskC on
    every point (:229-253). The messages :41-186 only print. Not ported (raise): the U/V masks of useRBCuVel /
    useRBCvVel (:255-311), the debugLevel >= debLevC output (:247-251, WRITE_FLD_XYZ_RS)."""
    sz = cfg.size
    if params.useRBCuVel or params.useRBCvVel:                         # :255-311
        raise NotImplementedError("RBCS_INIT_FIXED: the U/V relaxation masks (useRBCuVel/useRBCvVel) are not ported")
    k3, j3, i3 = loops_kji((1, sz.Nr), (1 - sz.OLy, sz.sNy + sz.OLy), (1 - sz.OLx, sz.sNx + sz.OLx))
    out = {}
    if not cfg.cpp.DISABLE_RBCS_MOM:                                    # :189-204
        out["RBC_maskU"] = rbcs_array("RBC_maskU", sz).at[i3, j3, k3].set(0.)       # :197  0. _d 0
        out["RBC_maskV"] = rbcs_array("RBC_maskV", sz).at[i3, j3, k3].set(0.)       # :198
    masks = []
    for irbc in range(1, maskLEN + 1):                                  # :207-226
        masks.append(rbcs_array("RBC_mask", sz).at[i3, j3, k3].set(0.))            # :217  0. _d 0
    for irbc in range(1, maskLEN + 1):                                  # :229-253
        name = params.relaxMaskTrFile[irbc - 1]
        if str(name).strip() != "":
            # :231-232  READ_FLD_XYZ_RS( relaxMaskTrFile(irbc), ' ', RBC_mask(..,irbc), 0, myThid ): MDS_READ_FIELD
            # ( fullName, readBinaryPrec, .FALSE., 'RS', Nr, 1, Nr, dummyRL, field, 1, myThid )
            m = rw.mds_read_field(str(name).strip(), rw.readBinaryPrec, sz.Nr, masks[irbc - 1], 1)
            m = EXCH_XYZ_RS(m, ex=ex)                                   # :233
            m = m.at[i3, j3, k3].set(m[i3, j3, k3] * grid.maskC[i3, j3, k3])          # :234-245
            if params.debugLevel >= 3:                                  # :247-251 (debLevC = 3, EEPARAMS.h)
                raise NotImplementedError("RBCS_INIT_FIXED: the debugLevel >= debLevC RBC_mask output is not ported")
            masks[irbc - 1] = m
    out["RBC_mask"] = masks
    return out
