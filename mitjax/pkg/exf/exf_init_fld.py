"""EXF_INIT_FLD: pkg/exf/exf_init_fld.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.exf.exf_filter_rl import exf_filter_rl


def exf_init_fld(fldName, fldFile, fldMask, fldPeriod, fld_inScale, fldConst, fldArr, fld0, fld1, *, cfg, exf, grid,
                 params, rw):
    """EXF_INIT_FLD( fldName, fldFile, fldMask, fldPeriod, fld_inScale, fldConst, fldArr, fld0, fld1, myThid )
    @63cdc0b pkg/exf/exf_init_fld.F:3-148

    C     | o Initialise one generic external forcing field: constant value, or the time-constant field read
    C     |   once (fldPeriod = 0)

    Returns (fldArr, fld0, fld1). Host routine (file read)."""
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    fldArr = fldArr.at[i, j].set(fldConst)                                     # :90
    fld0 = fld0.at[i, j].set(fldConst)                                         # :91
    fld1 = fld1.at[i, j].set(fldConst)                                         # :92
    if fldFile.strip() and fldPeriod == 0.:                                    # :98
        count = 1                                                              # :99
        fldArr = rw.READ_REC_3D_RL(fldFile, exf.exf_iprec, 1, fldArr, count, 0)   # :125-126
        fldArr = exf_filter_rl(fldArr, fldMask, cfg=cfg, grid=grid, params=params)   # :132
        j = loop_j(1, sz.sNy)
        i = loop_i(1, sz.sNx)
        fldArr = fldArr.at[i, j].set(fld_inScale*fldArr[i, j])                 # :139
    return fldArr, fld0, fld1
