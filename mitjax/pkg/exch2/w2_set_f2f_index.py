"""W2_SET_F2F_INDEX (pkg/exch2/w2_set_f2f_index.F @63cdc0b): check the facet topology and set up the index
correspondence matrix of connected facet edges (facet_pij, facet_oi, facet_oj).

    @63cdc0b pkg/exch2/w2_set_f2f_index.F:9-266

Ported as executed by the single facet (global_ocean.90x40x15) and the 6-facet cube (cubed-sphere experiments): every
edge connected (no disconnected-edge warning, :54-59), connections N<-S, S<-N, E<-W, W<-E (:132-147) and the ones that
change orientation, N<-W, S<-E, E<-S, W<-N (:149-184). The error branches raise with the Fortran message. `jj = INT(facet_link)`, `ii = MOD(NINT(facet_link*10.), 10)`: the product is REAL*4 (facet_link is Real*4,
`10.` a REAL*4 literal).
"""

from mitjax.eesupp.print import SQUEEZE_RIGHT, internal_write, print_message
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.exch2.w2_exch2_h import idiv, imod, int_, nint, r4

edge = ("N", "S", "E", "W")         # :39  DATA edge / 'N' , 'S' , 'E' , 'W' /


def _edge(i):
    return edge[i - 1]


def _link(w2, i, j):
    """(jj, ii) = (INT(facet_link(i,j)), MOD( NINT(facet_link(i,j)*10.), 10 ))."""
    x = w2.facet_link[i, j]
    return int_(x), imod(nint(x * r4(10.0)), 10)


def w2_set_f2f_index(w2, *, io, myThid=1):
    """W2_SET_F2F_INDEX( myThid ) on the W2Common `w2`."""
    msgBuf = internal_write("(2A)", "W2_SET_F2F_INDEX:", " index matrix for connected Facet-Edges:")   # :41-42
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                               # :43
    prtFlag = abs(w2.W2_printMsg) >= 2 or (w2.W2_printMsg != 0 and io.myProcId == 0)              # :44-45

    # --   Check Topology (:48-95)
    errCnt = 0
    for j in range(1, w2.nFacets + 1):
        for i in range(1, 5):
            jj, ii = _link(w2, i, j)                                          # :52-53
            if w2.facet_link[i, j] == r4(0.0):                                # :54
                raise NotImplementedError("W2_SET_F2F_INDEX: disconnected facet edge (w2_set_f2f_index.F:55-59) "
                                          "is not ported")
            elif jj < 1 or jj > w2.nFacets or ii < 1 or ii > 4:               # :60-65
                errCnt = errCnt + 1
            else:
                j1, i1 = _link(w2, ii, jj)                                    # :68-69
                if j1 != j or i1 != i:                                        # :70-86
                    errCnt = errCnt + 1
    if errCnt > 0:                                                            # :90-95
        raise RuntimeError(f" W2_SET_F2F_INDEX: found{errCnt:3d} Topology errors; ABNORMAL END: S/R W2_SET_F2F_INDEX")

    # --   Check length of connection (facet size) between connected facet-edges (:98-209)
    errCnt = 0
    for j in range(1, w2.nFacets + 1):
        for i in range(1, 5):
            # -    Length of N or S Edge = x-size, E or W Edge = y-size
            lo = 2 * (j - 1) + idiv(i + 1, 2)                                # :102
            lo = w2.facet_dims[lo]                                            # :103
            jj, ii = _link(w2, i, j)                                          # :105-106
            if jj >= 1:                                                       # :107
                ll = 2 * (jj - 1) + idiv(ii + 1, 2)                          # :108
                ll = w2.facet_dims[ll]                                        # :109
                if lo != ll:                                                  # :110-118
                    errCnt = errCnt + 1
                # -    Default: Same orientation (works for N <-> S or E <-> W) (:127-130)
                w2.facet_pij[1, i, j] = 1
                w2.facet_pij[2, i, j] = 0
                w2.facet_pij[3, i, j] = 0
                w2.facet_pij[4, i, j] = 1
                # --   1rst: cases with same orientation
                if i == 1 and ii == 2:                                        # :132  N <-- S
                    w2.facet_oi[i, j] = 0                                     # :134
                    w2.facet_oj[i, j] = +w2.facet_dims[2 * j]                 # :135
                elif i == 2 and ii == 1:                                      # :136  S <-- N
                    w2.facet_oi[i, j] = 0                                     # :138
                    w2.facet_oj[i, j] = -w2.facet_dims[2 * jj]                # :139
                elif i == 3 and ii == 4:                                      # :140  E <-- W
                    w2.facet_oi[i, j] = +w2.facet_dims[2 * j - 1]             # :142
                    w2.facet_oj[i, j] = 0                                     # :143
                elif i == 4 and ii == 3:                                      # :144  W <-- E
                    w2.facet_oi[i, j] = -w2.facet_dims[2 * jj - 1]            # :146
                    w2.facet_oj[i, j] = 0                                     # :147
                # --   2nd : cases where orientation changes
                elif i == 1 and ii == 4:                                      # :149  N <-- W
                    w2.facet_pij[1, i, j] = 0                                 # :152
                    w2.facet_pij[2, i, j] = -1                                # :153
                    w2.facet_pij[3, i, j] = 1                                 # :154
                    w2.facet_pij[4, i, j] = 0                                 # :155
                    w2.facet_oi[i, j] = lo + 1                                # :156
                    w2.facet_oj[i, j] = +w2.facet_dims[2 * j]                 # :157
                elif i == 2 and ii == 3:                                      # :158  S <-- E
                    w2.facet_pij[1, i, j] = 0                                 # :161
                    w2.facet_pij[2, i, j] = -1                                # :162
                    w2.facet_pij[3, i, j] = 1                                 # :163
                    w2.facet_pij[4, i, j] = 0                                 # :164
                    w2.facet_oi[i, j] = lo + 1                                # :165
                    w2.facet_oj[i, j] = -w2.facet_dims[2 * jj - 1]            # :166
                elif i == 3 and ii == 2:                                      # :167  E <-- S
                    w2.facet_pij[1, i, j] = 0                                 # :170
                    w2.facet_pij[2, i, j] = 1                                 # :171
                    w2.facet_pij[3, i, j] = -1                                # :172
                    w2.facet_pij[4, i, j] = 0                                 # :173
                    w2.facet_oi[i, j] = +w2.facet_dims[2 * j - 1]             # :174
                    w2.facet_oj[i, j] = lo + 1                                # :175
                elif i == 4 and ii == 1:                                      # :176  W <-- N
                    w2.facet_pij[1, i, j] = 0                                 # :179
                    w2.facet_pij[2, i, j] = 1                                 # :180
                    w2.facet_pij[3, i, j] = -1                                # :181
                    w2.facet_pij[4, i, j] = 0                                 # :182
                    w2.facet_oi[i, j] = -w2.facet_dims[2 * jj]                # :183
                    w2.facet_oj[i, j] = lo + 1                                # :184
                else:                                                         # :185-193
                    errCnt = errCnt + 1
                # --   Print resulting index matrix (:196-200)
                if prtFlag:
                    io.write(w2.W2_oUnit, fortran_write(
                        "(2(3A,I3),A,4I3,A,2I6)", "  ", _edge(i), ".Edge Facet", j, " <-- ",
                        _edge(ii), ".Edge Facet", jj,
                        " : pij=", *(w2.facet_pij[k, i, j] for k in range(1, 5)),
                        " ; oi,oj=", w2.facet_oi[i, j], w2.facet_oj[i, j]))
    if errCnt > 0:                                                            # :204-209
        raise RuntimeError(f" W2_SET_F2F_INDEX: found{errCnt:3d} Connection errors; "
                           "ABNORMAL END: S/R W2_SET_F2F_INDEX")

    # --   Check indices correspondence matrix reciprocity (:215-263)
    errCnt = 0
    for j in range(1, w2.nFacets + 1):
        for i in range(1, 5):
            jj, ii = _link(w2, i, j)                                          # :219-220
            if jj >= 1:                                                       # :221
                p = w2.facet_pij
                # -      Matrix product (:229-236)
                chk1 = p[1, i, j] * p[1, ii, jj] + p[2, i, j] * p[3, ii, jj]
                chk2 = p[1, i, j] * p[2, ii, jj] + p[2, i, j] * p[4, ii, jj]
                chk3 = p[3, i, j] * p[1, ii, jj] + p[4, i, j] * p[3, ii, jj]
                chk4 = p[3, i, j] * p[2, ii, jj] + p[4, i, j] * p[4, ii, jj]
                # -      Offsets (:238-243)
                chk5 = p[1, i, j] * w2.facet_oi[ii, jj] + p[2, i, j] * w2.facet_oj[ii, jj] + w2.facet_oi[i, j]
                chk6 = p[3, i, j] * w2.facet_oi[ii, jj] + p[4, i, j] * w2.facet_oj[ii, jj] + w2.facet_oj[i, j]
                if chk1 != 1 or chk2 != 0 or chk5 != 0 or chk3 != 0 or chk4 != 1 or chk6 != 0:   # :244-254
                    errCnt = errCnt + 1
    if errCnt > 0:                                                            # :258-263
        raise RuntimeError(f" W2_SET_F2F_INDEX: found{errCnt:3d} bugs in Matrix product; "
                           "ABNORMAL END: S/R W2_SET_F2F_INDEX")
    return w2
