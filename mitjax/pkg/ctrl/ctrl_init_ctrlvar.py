"""CTRL_INIT_CTRLVAR: pkg/ctrl/ctrl_init_ctrlvar.F @63cdc0b (GOADK lane, M2: global_ocean.90x40x15/code_ad)."""

from dataclasses import dataclass

from mitjax.pkg.ctrl.ctrl_set_fname import ctrl_set_fname
from mitjax.pkg.ctrl.ctrl_set_globfld_xyz import ctrl_set_globfld_xy, ctrl_set_globfld_xyz

# CTRL.h:17-24 PARAMETER( maxcvars = 0 + maxCtrlArr2D + maxCtrlArr3D + maxCtrlTim2D ) (ALLOW_OBCS_CONTROL undefined;
# CTRL_SIZE.h:21-28 all three = 1 in pkg/ctrl, not overridden in global_ocean.90x40x15/code_ad)
maxcvars = 0 + 1 + 1 + 1


@dataclass(frozen=True)
class CtrlVar:
    """CTRL.h's per-variable settings of control ivar (Fortran names)."""
    ivar: int
    ncvarindex: int
    ncvarrecs: int
    ncvarrecstart: int
    ncvarrecsend: int
    ncvarxmax: int
    ncvarymax: int
    ncvarnrmax: int
    ncvargrd: str
    ncvartype: str
    ncvarfname: str


def ctrl_init_ctrlvar(xx_fname, ivar, varIndex, varRecs, varStartRec, varEndRec, varNxMax, varNyMax, varNrMax,
                      varGrid, varType, costfinal_exists, *, sz, doInitXX, optimcycle, doAdmTlm=False,
                      yadprefix="ad"):
    """CTRL_INIT_CTRLVAR( xx_fname, ivar, varIndex, varRecs, varStartRec, varEndRec, varNxMax, varNyMax, varNrMax,
                          varGrid, varType, costfinal_exists, myThid )   @63cdc0b pkg/ctrl/ctrl_init_ctrlvar.F:9-179

    Host-side. Returns (CtrlVar (the CTRL.h entries :90-99), fname(1..3) of CTRL_SET_FNAME, the first-guess control
    records CTRL_SET_GLOBFLD_XYZ / _XY write to fname(1) with ( doInitXX .AND. optimcycle.EQ.0 ) .OR. doAdmTlm
    (:140-151; None when not written)).
    Branches (forward-only code_ad build): CTRL_DO_PACK_UNPACK_ONLY undefined (:104); ALLOW_ADJOINT_RUN /
    ALLOW_TANGENTLINEAR_RUN undefined (:135-139, :144-148 not compiled); yadprefix 'ad' (the g_ check :113-127 is
    the tangent-linear prefix only). varType 'Arr3D' and '..2D'; 'SecXZ' / 'SecYZ' raise (not ported); any other
    type is the Fortran STOP (:170-171). An ivar outside 0..maxcvars is the Fortran STOP (:79-87)."""
    if ivar < 0 or ivar > maxcvars:                                 # :79-87
        raise RuntimeError("ABNORMAL END: S/R CTRL_INIT_CTRLVAR")
    var = None
    if ivar != 0:                                                   # :88-100
        var = CtrlVar(ivar, varIndex, varRecs, varStartRec, varEndRec, varNxMax, varNyMax, varNrMax, varGrid,
                      varType, xx_fname)
    fname = ctrl_set_fname(xx_fname, optimcycle=optimcycle, yadprefix=yadprefix)    # :106-107 (ctrlDir blank)
    records = None
    if not costfinal_exists:                                        # :132
        write = (doInitXX and optimcycle == 0) or doAdmTlm
        if varType == "Arr3D":                                      # :134-142
            records = ctrl_set_globfld_xyz(varRecs, sz=sz) if write else None
        elif varType[3:5] == "2D":                                  # :143-151
            records = ctrl_set_globfld_xy(varRecs, sz=sz) if write else None
        elif varType in ("SecXZ", "SecYZ"):                         # :152-169
            raise NotImplementedError(f"CTRL_INIT_CTRLVAR: varType {varType!r} is not ported")
        else:
            raise RuntimeError("CTRL_INIT_CTRLVAR: varType option not implemented")      # :171
    return var, fname, records
