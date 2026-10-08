"""GRDCHK_PRINT and the per-check lines of GRDCHK_MAIN: pkg/grdchk/grdchk_print.F, grdchk_main.F @63cdc0b (lane API,
docs plan 20261006 Task 6). Host-side text only: the numbers come from the driver (mitjax/drivers/grdchk.py).

Every message goes through PRINT_MESSAGE( msgBuf, standardMessageUnit, SQUEEZE_RIGHT, myThid ) of one process and
one thread: the prefix "(PID.TID 0000.0001) " and the message without trailing blanks; a blank message is the prefix
alone (eesupp/src/print.F:122-128). The `ADM` lines are mitjax/drivers/adjoint_run.adm_lines. Not ported: the
ALLOW_TANGENTLINEAR_RUN arms (TLM header :106-109, :160-164) -- this is the adjoint's grdchk -- and the
`grad-res` / `ph-` debug WRITEs to standardMessageUnit without the prefix (grdchk_main.F:208-221, :466-487).
"""

import math

from mitjax.io.fortran_format import fortran_write

PREFIX = "(PID.TID 0000.0001) "
MAXGRDCHECKS = 4000                 # GRDCHK.h:20  parameter ( maxgrdchecks = 4000 )


def _pm(msg):
    """PRINT_MESSAGE with SQUEEZE_RIGHT (print.F:86-91, :122-128)."""
    return PREFIX + msg.rstrip(" ") if msg.strip(" ") else PREFIX


def check_start_lines(ichknum):
    """grdchk_main.F:235-238: WRITE(msgBuf,'(A,I4,A)') '====== Starts gradient-check number', ichknum, ..."""
    return [_pm(fortran_write("(A,I4,A)", "====== Starts gradient-check number", ichknum, " (=ichknum) ======="))]


def check_position_lines(itilepos, jtilepos, layer, itile, jtile, obcspos, icvrec):
    """grdchk_main.F:263-266: WRITE(msgBuf,'(A,3I5,A,2I4,A,I3,A,I4)') 'grdchk pos: i,j,k=', ..."""
    return [_pm(fortran_write("(A,3I5,A,2I4,A,I3,A,I4)", "grdchk pos: i,j,k=", itilepos, jtilepos, layer,
                              " ; bi,bj=", itile, jtile, " ; iobc=", obcspos, " ; rec=", icvrec))]


def check_end_lines(ichknum, ierr):
    """grdchk_main.F:521-524: WRITE(msgBuf,'(A,I4,A,I3,A)') '====== End of gradient-check number', ichknum, ..."""
    return [_pm(fortran_write("(A,I4,A,I3,A)", "====== End of gradient-check number", ichknum, " (ierr=", ierr,
                              ") ======="))]


def ratio_ad(adxxmemo, gfd):
    """grdchk_main.F:434-438: ratio_ad = ABS( adxxmemo - gfd ) when adxxmemo = 0, else 1. - gfd/adxxmemo."""
    if adxxmemo == 0.0:
        return abs(adxxmemo - gfd)
    return 1.0 - gfd / adxxmemo


def grdchk_print(ichknum, ierr_grdchk, mem, *, grdchk_eps, grdchkvarname):
    """GRDCHK_PRINT( ichknum, ierr_grdchk, myThid )   @63cdc0b pkg/grdchk/grdchk_print.F:4-215 (adjoint arm).

    mem: the GRDCHK.h *mem arrays as a list of dicts, entry i-1 for check i, with the keys xxmemref, xxmempert,
    adxxmem, fcrmem, fcppmem, fcpmmem, gfdmem, ratioadmem, bimem, bjmem, ilocmem, jlocmem, klocmem, icompmem,
    ierrmem (grdchk_main.F:448-472). Returns the printed lines."""
    out = []
    iL = len(grdchkvarname.rstrip(" "))                                           # :63 ILNBLNK
    out.append(_pm(" "))                                                          # :66-68
    out.append(_pm("// ======================================================="))  # :69-72
    out.append(_pm("// Gradient check results  >>> START <<<"))                   # :73-76
    out.append(_pm("// ======================================================="))  # :77-80
    out.append(_pm(" "))                                                          # :81-83
    out.append(_pm(fortran_write("(A,1PE13.6,3A)", " EPS =", grdchk_eps,         # :87-91
                                 ' ; grdchk CTRL var/file name: "', grdchkvarname[:iL], '"')))
    out.append(_pm(" "))                                                          # :92-94
    out.append(_pm(fortran_write("(A,2X,4A,3(3X,A),11X,A)", "grdchk output h.p:", "Id", " Itile", " Jtile",
                                 " LAYER", "bi", "bj", "X(Id)", "X(Id)+/-EPS")))   # :96-100
    out.append(_pm(fortran_write("(A,2X,A,A4,1X,2A21)", "grdchk output h.c:", "Id", "FC", "FC1", "FC2")))  # :101-105
    out.append(_pm(fortran_write("(A,2X,A,2X,2A18,4X,A18)", "grdchk output h.g:", "Id",            # :110-114
                                 "FC1-FC2/(2*EPS)", "ADJ GRAD(FC)", "1-FDGRD/ADGRD")))
    if ierr_grdchk == 0:                                                          # :118-122
        numchecks = ichknum
    else:
        numchecks = MAXGRDCHECKS
    ratio_RMS = 0.0                                                               # :124
    for i in range(1, numchecks + 1):                                             # :125
        r = mem[i - 1]
        out.append(_pm(" "))                                                      # :144-146
        out.append(_pm(fortran_write("(A,I4,3I6,2I5,1x,1P2E17.9)", "grdchk output (p):", i, r["ilocmem"],
                                     r["jlocmem"], r["klocmem"], r["bimem"], r["bjmem"], r["xxmemref"],
                                     r["xxmempert"])))                            # :147-152
        if r["ierrmem"] == 0:                                                     # :153
            out.append(_pm(fortran_write("(A,I4,1P3E21.13)", "grdchk output (c):", i, r["fcrmem"], r["fcppmem"],
                                         r["fcpmmem"])))                          # :154-159
            ratio_RMS = ratio_RMS + r["ratioadmem"] * r["ratioadmem"]             # :165
            out.append(_pm(fortran_write("(A,I4,3x,1P3E21.13)", "grdchk output (g):", i, r["gfdmem"],
                                         r["adxxmem"], r["ratioadmem"])))         # :166-168
        else:                                                                     # :172-186
            msg = {-1: " Component does not exist (zero)", -2: " Component does not exist (negative)",
                   -3: " Component does not exist (too large)", -4: " Component does not exist (land point)"}
            out.append(_pm(msg.get(r["ierrmem"]) or fortran_write("(A,I6,A)", " Unknown error (ierr=",
                                                                  r["ierrmem"], " )")))
    if ichknum > 1:                                                               # :190
        ratio_RMS = ratio_RMS / ichknum
    if ratio_RMS > 0.0:                                                           # :191
        ratio_RMS = math.sqrt(ratio_RMS)
    out.append(_pm(" "))                                                          # :192-194
    out.append(_pm(fortran_write("(A,I4,A,1P1E21.13)", "grdchk  summary  :  RMS of ", ichknum, " ratios =",
                                 ratio_RMS)))                                     # :195-198
    out.append(_pm(" "))                                                          # :199-201
    out.append(_pm("// ======================================================="))  # :202-205
    out.append(_pm("// Gradient check results  >>> END <<<"))                     # :206-209
    out.append(_pm("// ======================================================="))  # :210-213
    out.append(_pm(" "))                                                          # :214-216
    return out


def start_lines(fcref):
    """grdchk_main.F:140-147 and :184-186: the banner '// Gradient-check starts (grdchk_main)' and
    WRITE(msgBuf,'(A,1PE22.14)') 'grdchk reference fc: fcref       =', fcref."""
    bar = "// ======================================================="
    return [_pm(bar), _pm("// Gradient-check starts (grdchk_main)"), _pm(bar),
            _pm(fortran_write("(A,1PE22.14)", "grdchk reference fc: fcref       =", fcref))]


def perturb_lines(fcpertplus, fcpertminus):
    """grdchk_main.F:369-371 and :410-412: WRITE(msgBuf,'(A,1PE22.14)') 'grdchk perturb(+)fc: fcpertplus  =' ... and
    'grdchk perturb(-)fc: fcpertminus ='."""
    return [_pm(fortran_write("(A,1PE22.14)", "grdchk perturb(+)fc: fcpertplus  =", fcpertplus)),
            _pm(fortran_write("(A,1PE22.14)", "grdchk perturb(-)fc: fcpertminus =", fcpertminus))]
