"""W2 set-up gate helpers (test_w2_setup.py; not a test file). The oracle for global_ocean.90x40x15/input:

* the W2 print-out of the registered dumps-on run: `w2_tile_topology.0000.log` (W2_printMsg = -1 sends the
  topology report there, pkg/exch2/w2_eeboot.F:85-95) and the W2 lines of its STDOUT (W2_READPARMS, W2_EEBOOT);
* the topology the oracle's dump headers carry (jaxdump.F writes exch2_tBasex/y of each W2 tile;
  mitjax/tests/grid_gate.exch2_topology), which the grid gate has used so far;
* the measured exchange maps ($MJX_REFERENCE/exch_maps, from lane A's exchange probe): the topology implies them
  through EXCH2_3D_RL (exch2_3d_rx.template: EXCH2_RX1_CUBE with EXCH_IGNORE_CORNERS, then with
  EXCH_UPDATE_CORNERS), which `simulate_exch2_3d` replays on the topology arrays.
"""

import copy
import dataclasses

import numpy as np

from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.print import STANDARD_MESSAGE_UNIT
from mitjax.model.src.ini_parms import ini_parms_grid
from mitjax.pkg.exch2.exch2_get_scal_bounds import exch2_get_scal_bounds
from mitjax.pkg.exch2.w2_eeboot import exch2_topology, w2_eeboot
from mitjax.tests import grid_gate as gg

EXP, INP = "global_ocean.90x40x15", "input"
LOG = "w2_tile_topology.0000.log"
_CACHE = {}


def setup():
    """(experiment, W2Common, MessageUnits) of the W2 set-up of global_ocean (cached; callers that mutate copy)."""
    if "w2" not in _CACHE:
        e = gg.experiment(EXP, INP)
        w2, io = w2_eeboot(e)
        _CACHE["w2"] = (e, w2, io)
    return _CACHE["w2"]


def oracle_printout():
    """(log-file records, STDOUT records) of the oracle run."""
    _, _, rundir = gg.oracle(EXP, INP)
    log = (rundir / LOG).read_text().split("\n")
    out = (rundir / "output.txt").read_text().split("\n")
    assert log[-1] == "", "log file does not end with a newline"
    return log[:-1], out


def printout_mismatches(io, w2):
    """{'log': n differing records (or length mismatch), 'stdout': n differing} vs the oracle, character for
    character. The STDOUT block is located at its first record (W2_READPARMS)."""
    log, out = oracle_printout()
    ours = io.records(w2.W2_oUnit)
    nlog = abs(len(log) - len(ours)) + sum(a != b for a, b in zip(log, ours))
    so = io.records(STANDARD_MESSAGE_UNIT)
    try:
        i0 = out.index(so[0])
    except ValueError:
        return {"log": nlog, "stdout": len(so)}
    nout = sum(a != b for a, b in zip(out[i0:i0 + len(so)], so)) + max(0, i0 + len(so) - len(out))
    return {"log": nlog, "stdout": nout, "log_records": len(ours), "stdout_records": len(so)}


def header_mismatches(ex2):
    """Fields of an Exch2Topology that differ from the dump-header topology: tBasex/y and W2_myTileList on every
    tile, exch2_mydNx/y on every tile equal to the facet size (single facet)."""
    ds, _, _ = gg.oracle(EXP, INP)
    ref = gg.exch2_topology(ds)
    bad = []
    for name in ("exch2_tBasex", "exch2_tBasey", "W2_myTileList"):
        if tuple(getattr(ex2, name)) != tuple(getattr(ref, name)):
            bad.append(name)
    for name in ("exch2_mydNx", "exch2_mydNy"):
        if len(getattr(ex2, name)) != len(ref.exch2_tBasex) or set(getattr(ex2, name)) != {getattr(ref, name)[0]}:
            bad.append(name)
    return bad


def gridparams_mismatches(ex2):
    """GridParams fields (all but `exch2` itself) that differ between ini_parms_grid with `ex2` and with the
    dump-header topology."""
    e = gg.experiment(EXP, INP)
    ds, _, _ = gg.oracle(EXP, INP)
    a, b = ini_parms_grid(e, ex2), ini_parms_grid(e, gg.exch2_topology(ds))
    bad = []
    for f in dataclasses.fields(a):
        if f.name == "exch2":
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        same = np.array_equal(x, y) if isinstance(x, np.ndarray) else (x == y)
        if not same:
            bad.append(f.name)
    return bad


def simulate_exch2_3d(w2, layout):
    """Source (flat index into the pre-exchange [tile, j, i] array) of every point after EXCH2_3D_RL, replayed on the
    topology: two EXCH2_RX1_CUBE passes (corners ignored, then updated; exch2_3d_rx.template). In a pass every PUT
    reads the array as it was before the pass (exch2_rx1_cube.template: all EXCH2_PUT_RX1 of all tiles, then all
    EXCH2_GET_RX1), with the source index of exch2_put_rx1.template (pij, oi, oj of the sender's entry), and the GETs
    write in tile order (W2_myTileList) and neighbour order. A source index outside the source tile's array (what
    W2_E2_DEBUG_ON stops on) gives -1."""
    L = layout
    sz = w2.size
    arr = np.arange(L.npoints).reshape(L.shape2d)
    for corners in (False, True):
        old = arr.copy()
        for bj in range(1, sz.nSy + 1):
            for bi in range(1, sz.nSx + 1):
                t = w2.W2_myTileList[bi, bj]
                for N in range(1, w2.exch2_nNeighbours[t] + 1):
                    s = w2.exch2_neighbourId[N, t]
                    oN = w2.exch2_opposingSend[N, t]
                    tIlo, tIhi, tJlo, tJhi, tiS, tjS = exch2_get_scal_bounds("T ", L.OLx, corners, t, N, w2=w2)
                    p1, p2, p3, p4 = (w2.exch2_pij[k, oN, s] for k in range(1, 5))
                    oi, oj = w2.exch2_oi[oN, s], w2.exch2_oj[oN, s]
                    for jtl in range(tJlo, tJhi + tjS, tjS):
                        for itl in range(tIlo, tIhi + tiS, tiS):
                            itc, jtc = itl + w2.exch2_tBasex[t], jtl + w2.exch2_tBasey[t]
                            isl = p1 * itc + p2 * jtc + oi - w2.exch2_tBasex[s]
                            jsl = p3 * itc + p4 * jtc + oj - w2.exch2_tBasey[s]
                            if not (1 - L.OLx <= isl <= L.sNx + L.OLx and 1 - L.OLy <= jsl <= L.sNy + L.OLy):
                                # exch2_put_rx1.template (W2_E2_DEBUG_ON): "isl/jsl out of bounds" STOP
                                arr[t - 1, jtl - 1 + L.OLy, itl - 1 + L.OLx] = -1
                                continue
                            arr[t - 1, jtl - 1 + L.OLy, itl - 1 + L.OLx] = old[s - 1, jsl - 1 + L.OLy,
                                                                                isl - 1 + L.OLx]
    return arr.reshape(-1)


def map_mismatches(w2):
    """{map key: n points where the replayed exchange differs from the measured map} for the scalar maps of
    EXCH_XY_RL and EXCH_3D_RL (src where the map writes the point; the point's own index where it keeps it)."""
    maps = EM.load_maps(EXP)
    sim = simulate_exch2_3d(w2, maps.layout)
    own = np.arange(maps.layout.npoints)
    out = {}
    for key in ("XY", "3D"):
        src, comp, sign = maps.maps[key]
        want = np.where(comp == 0, own, src)
        out[key] = int(np.count_nonzero(sim != want)) + int(np.count_nonzero((comp != 0) & (comp != 1)))
        out[key] += int(np.count_nonzero(sign != 1))
    return out


def planted(mutate):
    """A deep copy of the set-up's W2Common with `mutate(w2)` applied (the planted errors of the controls)."""
    _, w2, _ = setup()
    w2 = copy.deepcopy(w2)
    mutate(w2)
    return w2


__all__ = ["setup", "printout_mismatches", "header_mismatches", "gridparams_mismatches", "simulate_exch2_3d",
           "map_mismatches", "planted", "exch2_topology", "w2_eeboot"]
