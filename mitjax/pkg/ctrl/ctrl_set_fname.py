"""CTRL_SET_FNAME: pkg/ctrl/ctrl_set_fname.F @63cdc0b (GOADK lane, M2: global_ocean.90x40x15/code_ad)."""

from mitjax.pkg.ctrl.ctrl_readparms import MAX_LEN_FNAM


def ctrl_set_fname(xx_fname, *, optimcycle, yadprefix):
    """CTRL_SET_FNAME( xx_fname, fname, myThid )   @63cdc0b pkg/ctrl/ctrl_set_fname.F:6-64

    C     o get filename for control variable and adjoint thereof

    Host-side: returns fname(1..3) (blank-stripped): '<xx_fname>.<optimcycle %10.10d>', '<dir>/<yadprefix><name>.<..>',
    '<dir>/hn<name>.<..>' (:44-57; the directory part up to the last '/' of xx_fname, :43-48). yadprefix: 'ad' (CTRL.h,
    ctrl_readparms.F:442; 'g_' only #ifdef ALLOW_TANGENTLINEAR_RUN, :439). A name longer than MAX_LEN_FNAM-13 is the
    Fortran STOP (:59-60)."""
    il = len(xx_fname.rstrip(" "))                                  # :41  ILNBLNK( xx_fname )
    if not (il > 0 and il + 13 <= MAX_LEN_FNAM):                    # :43
        raise RuntimeError("ABNORMAL END: S/R CTRL_SET_FNAME")      # :60
    ic = 0                                                          # :44-48
    for pos in range(il, 0, -1):
        if xx_fname[pos - 1] == "/":
            ic = pos
            break
    head, tail = xx_fname[:ic], xx_fname[ic:il]
    return (f"{xx_fname[:il]}.{optimcycle:010d}",                   # :50-51
            f"{head}{yadprefix}{tail}.{optimcycle:010d}",          # :52-54
            f"{head}hn{tail}.{optimcycle:010d}")                   # :55-57
