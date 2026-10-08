"""SEAICE_COST_FINAL   @63cdc0b pkg/seaice/seaice_cost_final.F:12-100 (lane M4ADCOL)"""

from mitjax.eesupp.global_sum import global_sum_tile
from mitjax.io.fortran_format import fortran_write


def seaice_cost_final(cost, *, cfg, sp, ex=None):
    """SEAICE_COST_FINAL( ifc, optimcycle, myThid )

    Under ALLOW_COST and ALLOW_COST_ICE (:45): tile_fc(bi,bj) = tile_fc + mult_ice_export*objf_ice_export
    + mult_ice*objf_ice (:65-67; ALLOW_SEAICE_COST_EXPORT undefined: no :57-60 sums), f_ice and no_ice by
    GLOBAL_SUM_TILE_RL (:76-77). Returns (cost, f_ice, no_ice) -- (cost, None, None) without ALLOW_COST_ICE; `seaice_cost_final_lines` prints the STDOUT record
    (:79-80, a WRITE to standardMessageUnit without the PID.TID prefix) and the costfunction_seaice file (:83-95)."""
    if not cfg.cpp.flag("ALLOW_COST_ICE", "SEAICE_OPTIONS.h"):         # :45-97 (lane M4ADLAB: lab_sea/code_ad):
        return cost, None, None                                        # no term, no line, no file
    if cfg.cpp.flag("ALLOW_SEAICE_COST_EXPORT", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_COST_FINAL: ALLOW_SEAICE_COST_EXPORT (:57-60) is not ported")
    cost.tile_fc = (cost.tile_fc                                       # :65-67
                    + sp.mult_ice_export * cost.objf_ice_export
                    + sp.mult_ice * cost.objf_ice)
    gs = global_sum_tile if ex is None else ex.global_sum_tile
    f_ice = gs(cost.objf_ice)                                          # :76
    no_ice = gs(cost.num_ice)                                          # :77
    return cost, f_ice, no_ice


def seaice_cost_final_lines(f_ice, no_ice, mult_ice):
    """(the STDOUT record of :79-80, the costfunction_seaice.<optimcycle> line of :90-91), host floats."""
    out = fortran_write("(A,1PE22.14,1PE22.14,1X,1PE9.2)", " --> f_ice     =", float(f_ice), float(no_ice),
                        float(mult_ice))
    cf = fortran_write("(A,1PE22.14,1PE22.14,1X,1PE9.2)", "f_ice   =", float(f_ice), float(no_ice), float(mult_ice))
    return out, cf
