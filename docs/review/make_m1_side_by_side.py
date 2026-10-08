#!/usr/bin/env python3
"""Writes docs/review/M1_SIDE_BY_SIDE.html: ten routines ported since M0, the Fortran at `pinned` (63cdc0b) on the
left and the Python on the right, rows aligned by the `# :NN` citations of the Python (plan Task 18, readability
review). A local, self-contained file: no external resources; it is not published anywhere.

    python docs/review/make_m1_side_by_side.py            (from the repository root; needs $MJX_UPSTREAM)

Each excerpt is given as segments: a Python range located by two unique text fragments (so that the page can be
regenerated when the code moves) and a Fortran line range at `pinned` (fixed). Inside a segment, a Python line whose
comment cites `:NN` with NN in the Fortran range is an anchor; the longest increasing chain of anchors is kept, so a
stray citation cannot scramble the alignment, and each anchor shares a row with its Fortran line. Every code cell
carries `data-src` (`up:<path>:<line>` or `repo:<path>:<line>`); mitjax/tests/test_m1_side_by_side.py checks every
cell against the files.
"""

import html
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mitjax import paths  # noqa: E402

OUT = REPO / "docs" / "review" / "M1_SIDE_BY_SIDE.html"
PIN = "pinned"

# (title, lane, [(python file, start text, end text, fortran file | None, f_lo, f_hi)], what to look at, gate)
ROUTINES = [
    ("CALC_R_STAR: new column thickness and the r* counters", "RSTAR (Task 15a)",
     [("mitjax/model/src/calc_r_star.py", "# :99-110  Compute the new column thikness",
       "icntc2 = jnp.sum(big, axis=(1, 2), dtype=jnp.int32)", "model/src/calc_r_star.F", 99, 199)],
     ["Deviations: none in this excerpt. The one accepted guard of the routine (Nikolay 2026-10-01) is at "
      "calc_r_star.F:306-311 (a zero rStarExp denominator becomes 1, with a host-side count that is reported).",
      "The divisions by tmpfldW/tmpfldS are <code>safe_div</code> on the Fortran's own <code>kSurfW.LE.Nr</code> "
      "test, so dry points give finite derivatives; the forward value equals the Fortran's where it divides.",
      "The STOP of :201-243 cannot run inside a compiled step: the counters of :183-199 are computed per tile and "
      "returned; the host decides the STOP and the warnings (<code>calc_r_star_host</code>).",
      "<code>rStarAreaWeight</code> (:64-68) is a static switch; the simple-average arm is gated in the replay "
      "harness only (no M1 variant runs it)."],
     "test_rstar.py (tier 1x): test_calc_r_star_bitwise (global_ocean.90x40x15 and advect_xz/input.nlfs dumps, "
     "every point incl. halos), test_replay_harness_bitwise_and_stderr (76 warning lines equal to STDERR), "
     "test_rstar_P4_equals_P1_global_ocean. Tier 1x job 27837912 (master c01069d): 16 of 17 passed; the failure is "
     "test_unported_options_raise, a test that now calls UPDATE_SURF_DR without <code>state=</code>, not this "
     "routine. Whole run: global_ocean.90x40x15/input through the run CLI, job 27838020 (16 digits)."),
    ("MOM_VECINV: vorticity, divergence and the dissipation terms of one level", "VECINV (Task 23)",
     [("mitjax/pkg/mom_vecinv/mom_vecinv.py", "#     Make local copies of horizontal flow field",
       "guDiss, gvDiss, cfg=cfg, grid=grid, params=p)", "pkg/mom_vecinv/mom_vecinv.F", 264, 436)],
     ["Deviations: none.",
      "Per-site MAX/MIN: the three MINs of h0FacZ (:318-320) carry <code>p=\"b\"</code> from the oracle's site "
      "table (which argument gfortran returns on a tie or NaN).",
      "The land masks of vort3/vort3BC and strain/strainBC (:289-297, :335-343) are <code>jnp.where</code> on "
      "<code>hFacZ == 0</code>, both branches finite.",
      "momViscosity, useVariableVisc, useBiharmonicVisc, no_slip_sides are static; unported options "
      "(ALLOW_LEITH_QG, useStrainTensionVisc) raise. The routine takes the level k as a Python integer; DYNAMICS "
      "calls it in an unrolled loop."],
     "test_vecinv.py (tier 1x): gfortran replay of every routine on the 90x40x15 code_ad, cs32x15 and solid-body "
     "grids, dump stage D00c, negative controls, finite gradients, FD and dot test, P=4. Tier 1x job 27837912 "
     "(master c01069d): 17 passed. In the model: test_goadk_model.py, free-running steps 0-2 of "
     "global_ocean.90x40x15/input_ad bitwise at every dumped stage (merged eea66a1)."),
    ("GAD_ADVECTION: the three cube passes (one level)", "ADVECT_CS (Task 25), schedule by lane B (Task 22)",
     [("mitjax/pkg/generic_advdiff/gad_advection.py", "def _cs_pass_flags(nCFace, ipass):",
       "return overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y", "pkg/generic_advdiff/gad_advection.F",
       347, 370),
      ("mitjax/pkg/generic_advdiff/gad_advection.py", "for ipass in range(1, NPASS_CS+1):",
       "localTij = localTij.at[i, j].set(jnp.where(sel, newTij, localTij[i, j]))",
       "pkg/generic_advdiff/gad_advection.F", 342, 591)],
     ["Deviations: none.",
      "Where-masks over the tile axis: the Fortran decides per tile (by its face number) whether a pass computes X "
      "or Y fluxes and which points it updates. Here every tile runs every statement; the per-tile flags are "
      "<code>[tile,1,1]</code> boolean arrays, and <code>sel</code> writes only the points the Fortran's loops "
      "would write (overlapOnly / interiorOnly ranges with the facet edges).",
      "FILL_CS_CORNER_TR_RL is applied only on the tiles whose flag is set (the others get their corner flags "
      "cleared): check the fill directions against :392-395 and :459-462.",
      "The compressible branch (GAD_MULTIDIM_COMPRESSIBLE, not defined in the M2 cube builds) divides "
      "<code>tmpTrac/newVol</code> on every point before the select, without a guard.",
      "afx (:580-585) is kept for diagnostics only and is not in this function."],
     "test_advect_cs.py (tier 1x): advect_cs initial state and steps 1-3 bitwise incl. corner halos, three "
     "negative controls (corner fill skipped, passes swapped, SOM corner store skipped), P=6 == P=1, the pass "
     "schedule equal to lane B's gfortran replay (288 cases), the whole run through the CLI. Tier 1x job 27837912 "
     "(master c01069d): 9 passed. Dev jobs 27834077, 27834254 (P=6), 27834348 (CLI run)."),
    ("SOLVE_PENTADIAGONAL: forward elimination and back substitution", "GO (Task 15a, implicit vertical advection)",
     [("mitjax/model/src/solve_pentadiagonal.py", "# :341-350  k = 1: just copy terms",
       "errCode = jnp.where(jnp.any(fw[\"err\"], axis=0), 1, errCode)", "model/src/solve_pentadiagonal.F", 334, 376),
      ("mitjax/model/src/solve_pentadiagonal.py", "def normalise(cp, dp, ep, yp):",
       "return dp, ep, yp, jnp.any(~nz, axis=(1, 2))", "model/src/solve_pentadiagonal.F", 377, 397),
      ("mitjax/model/src/solve_pentadiagonal.py", "# :398-423  Backward sweep (starting from bottom)",
       "y5d = set_levels(y5d, bw, range(1, Nr-1))", "model/src/solve_pentadiagonal.F", 398, 424)],
     ["Deviations: none. Nr &lt; 3 raises (the routine assumes the k &ge; 3 sweep).",
      "<code>scan_k</code>: the forward sweep reads levels k-1 and k-2, so the carry holds two levels; the first "
      "two levels are single statements because the compiled loop body cannot branch on k.",
      "The pivot test <code>tmpVar.NE.0.</code> (:381) guards the division (<code>safe_div</code>), so a zero pivot "
      "gives the Fortran's 0 and a finite derivative. errCode is one value per tile.",
      "The normalisation (:377-392) is written once as a local function and called at k = 1, 2 and inside the scan."],
     "test_replay_go.py::test_solve_pentadiagonal_bitwise (tier 1x): gfortran replay on the global_ocean grid "
     "(the gate found the missing <code>errCode = 0</code> of :208); in the model through advect_xz/input.nlfs "
     "(implicit vertical advection, stage T23, test_r3_nlfs.py). Tier 1x job 27837912 (master c01069d): "
     "test_replay_go 4 passed, test_r3_nlfs 4 passed."),
    ("GMREDI_CALC_PSI_BOLUS: GM bolus stream functions", "GOADK (Task 24)",
     [("mitjax/pkg/gmredi/gmredi_calc_psi_bolus.py", "halfSign = halfRL*gravitySign",
       "return gm.replace(GM_PsiX=GM_PsiX, GM_PsiY=GM_PsiY)", "pkg/gmredi/gmredi_calc_psi_bolus.F", 87, 197)],
     ["Deviations: none. <code>half_K</code> (:147-148) is not computed: it is read only in the branch without "
      "GM_USE_K3D_GM, which raises in this port.",
      "The k loop calls GMREDI_SLOPE_PSI once per level and carries nothing between levels except the outputs: "
      "an unrolled Python loop in the Fortran order, each level vectorised over (i, j).",
      "The ALLOW_AUTODIFF zeroing of the locals (:94-103) is ported (the code_ad build defines it)."],
     "test_goadk.py (tier 1x): test_calc_tensor_bolus_bitwise_vs_dumps and "
     "test_residual_flow_bolus_bitwise_vs_dumps (the bolus stream function and transport vs the oracle dumps of "
     "global_ocean.90x40x15/input_ad and variants), test_replay_bitwise (53 outputs), negative controls. "
     "Tier 1x job 27837912 (master c01069d): test_goadk 34 passed. GOADK dev jobs 27833371 (44 passed incl. test_gmredi) and 27833696."),
    ("CONVECTIVE_ADJUSTMENT: the level loop and its clock test", "PTRACERS (Task 26)",
     [("mitjax/model/src/convective_adjustment.py",
       "doIt = different_multiple(params.cAdjFreq, myTime, params.deltaTClock)",
       "return state.replace(theta=theta, salt=salt), ptf", "model/src/convective_adjustment.F", 60, 184)],
     ["Deviations: none.",
      "REAL IF decided on the host: <code>IF ( rkSign*gravitySign .GT. 0. )</code> (:95-107) is taken from "
      "usingZCoords, as the code's own comment states (:96); the pressure-coordinate branch raises.",
      "<code>IF ( DIFFERENT_MULTIPLE(cAdjFreq,myTime,deltaTClock) )</code> reads the traced model clock: every level "
      "is computed and the adjusted fields are selected by <code>jnp.where</code> (a traced DIFFERENT_MULTIPLE, "
      "same operations as the host one).",
      "The level loop is a recursion (level k-1 mixed at iteration k is read again at k+1): an unrolled Python loop "
      "in the Fortran order. ConvectCount feeds diagnostics only; it is computed and dropped."],
     "test_ptracers.py (tier 1x): test_replay_convective_adjustment (gfortran replay incl. PTRACERS_CONVECT), "
     "test_dump_convective_adjustment_tracer_adjsens, test_negative_control_clock, "
     "test_gradient_convective_adjustment. Tier 1x job 27837912 (master c01069d): test_ptracers 39 passed. PTRACERS dev job 27833654, "
     "regression job 27834372."),
    ("EXCH2_Z_3D_RX: the cube corner statements, replayed into gather tables", "lane B (Task 22)",
     [("mitjax/pkg/exch2/exch2_cube_tables.py", "def exch2_z_3d(sym, c, w2, useCubedSphereExchange):",
       "sym.copy(c, t, i, sNy + 1, c, 1, j)", "pkg/exch2/exch2_z_3d_rx.template",
       56, 178)],
     ["Deviations: none. This is host code: the Fortran statements are executed once on symbolic arrays (each "
      "point records its source point, component and sign), giving one gather table per exchange; the model's "
      "exchange is that gather.",
      "The k loop is absent: the table is the same for every level.",
      "W2_FILL_NULL_REGIONS must be undefined (the <code>#ifdef</code> blocks are not replayed); "
      "cube_exchange_programs checks it."],
     "test_cube.py (tier 1x): test_cube_exchanges_equal_replay (every exchange routine vs its gfortran replay on "
     "the four cube experiments), test_cube_exchange_negative_controls, test_cube_sharded_p4/p6_equals_single, "
     "test_cube_transpose. Tier 1x job 27837912 (master c01069d): test_cube 34 passed. Lane B jobs 27833717, 27834541, 27837362."),
    ("INI_CURVILINEAR_GRID: reading the facet files, exchanges, radius rescaling", "lane B (Task 22)",
     [("mitjax/model/src/ini_curvilinear_grid.py", "for bj in range(1, sz.nSy + 1):",
       "anglesAreSet = False\n", "model/src/ini_curvilinear_grid.F", 261, 366),
      ("mitjax/model/src/ini_curvilinear_grid.py", "# exchanges (:374-384), in the Fortran order",
       "return grid.replace(angleSinC=sinC, angleCosC=cosC)", "model/src/ini_curvilinear_grid.F", 374, 420)],
     ["Deviations: none. Host code: the grid is built once before the model is compiled.",
      "Citation slip: the line <code>tmpFac = params.rSphere / params.radius_fromHorizGrid</code> cites "
      "<code>:420</code>; the Fortran statement is at :391 (the alignment below ignores that anchor).",
      "<code>IF ( rSphere.NE.radius_fromHorizGrid )</code> (:390) is a Python <code>if</code> on host values "
      "(the grid is fixed; no derivative with respect to rSphere).",
      "The eleven exchanges run in the Fortran order, through the cube exchanger."],
     "test_cube.py (tier 1x): test_cube_grid_equals_replay_and_oracle (35 GRID.h fields bitwise on every point "
     "incl. halos and corners, vs the gfortran replay and the G00_geometry dumps, four cube experiments), "
     "test_cube_grid_negative_controls (records swapped, rescaling skipped, withSigns flipped). "
     "Tier 1x job 27837912 (master c01069d): test_cube 34 passed. Lane B job 27837362."),
    ("CTRL_MAP_GENARR3D: adding a 3-D control to the model field", "GOADK (Task 24)",
     [("mitjax/pkg/ctrl/ctrl_map_genarr.py", "doscaling = _flags(genarr3d, \"CTRL_MAP_GENARR3D\")",
       "return fld, fld, wgen", "pkg/ctrl/ctrl_map_genarr.F", 276, 417)],
     ["Deviations: none. The control and weight records are arrays (the JAX control vector), not files; the "
      "file-name bookkeeping (:312-319) has no counterpart.",
      "<code>IF ( wgenarr3d .GT. 0. )</code> chooses between two values: it stays a <code>jnp.where</code>, with "
      "the SQRT guarded (a lane with weight &le; 0 takes SQRT(1.) and is then replaced by 0).",
      "The exchange is skipped for xx_uvel/xx_vvel exactly as :397-399; log10ctrl raises."],
     "test_goadk.py (tier 1x): test_control_zero_effective_and_identity, test_control_negative_controls_bite "
     "(effective records of xx_theta, xx_kapgm, xx_kapredi bitwise vs lane A's planted-control runs); "
     "test_goadk_model.py::test_negative_control_planted_control_vs_zero_oracle. Tier 1x job 27837912 (master c01069d): test_goadk 34 "
     "passed. GOADK dev job 27834324."),
    ("Adjoint (drivers/adjoint_run.py): gradient, GRDCHK finite differences, ADM lines", "R5 (Task 16)",
     [("mitjax/drivers/adjoint_run.py", "class Adjoint:",
       ("return self._jvp(self.theta0 if theta is None else theta, v, self.model if model is None else model,", 1),
       None,
       0, 0),
      ("mitjax/drivers/adjoint_run.py", "def grdchk_fd(self, points, *, eps=None, theta=None, model=None):",
       ("out.append((p, fp, fm, fd_gradient(fp, fm, eps, grdchk_epsfac(True))))", 1), "pkg/grdchk/grdchk_main.F",
       353, 432),
      ("mitjax/drivers/adjoint_run.py", "def adm_lines(fc, adjoint_gradient, finite_diff_grad):",
       ("out.append((PREFIX + fortran_write(\"(A30,1PE22.14)\", lit, float(v))).rstrip())", 1),
       "pkg/grdchk/grdchk_main.F", 509, 519)],
     ["This is a driver, not a port: the adjoint is <code>jax.value_and_grad</code> of the checkpointed time loop "
      "(one checkpoint per step) instead of TAF's generated code; JAX transforms appear only here and in the other "
      "drivers, never in model/ or pkg/.",
      "The exact JAX gradient is the default; no data.autodiff approximation is used by this experiment.",
      "The finite differences follow GRDCHK_MAIN: forward runs with the control at the point +eps and -eps, "
      "gfd = (fcpertplus - fcpertminus)/(grdchk_epsfac*grdchk_eps); the ADM lines use the Fortran format "
      "(A30,1PE22.14)."],
     "test_r5_adjoint.py (tier 1x): admGrd equal to TAF's results/output_adm.txt in every printed digit at the 3 "
     "points (rel. 1.25e-15, 1.22e-15, 1.73e-15), admCst and admFwd likewise, trust protocol (FD h-sweep, "
     "tangent/adjoint dot test, repeat bitwise), negative control (a cost weight times 1+1e-7). Jobs 27834137 "
     "(values), 27834391 (FD test failed on an unrounded comparison, rewritten), 27834769 (passed); "
     "docs/M1_ACCEPTANCE.md section 2."),
]

ANCHOR = re.compile(r"(?<![\w.]):(\d+)")


def upstream_lines(path):
    r = subprocess.run(["git", "-C", str(paths.UPSTREAM), "show", f"{PIN}:{path}"], capture_output=True, check=True)
    return r.stdout.decode("latin-1").split("\n")


def locate(lines, start, end, path):
    """1-based (first, last) of the unique line containing `start` and the first line at or after it containing
    `end` (a trailing newline in `end` means the line must end with it)."""
    hits = [n for n, ln in enumerate(lines, 1) if start in ln]
    if len(hits) != 1:
        raise ValueError(f"{path}: start text {start!r} found {len(hits)} times")
    extra = 0
    if isinstance(end, tuple):
        end, extra = end
    whole = end.endswith("\n")
    e = end.rstrip("\n")
    for n in range(hits[0], len(lines) + 1):
        ln = lines[n - 1]
        if (ln.rstrip().endswith(e) if whole else e in ln):
            return hits[0], n + extra
    raise ValueError(f"{path}: end text {end!r} not found after line {hits[0]}")


def anchor_of(line, lo, hi):
    if "#" not in line:
        return None
    m = ANCHOR.search(line[line.index("#"):])
    if m and lo <= int(m.group(1)) <= hi:
        return int(m.group(1))
    return None


def lis(pairs):
    """Longest strictly increasing (in the Fortran line) subsequence of [(py, f)]."""
    best, prev = [], {}
    tails = []
    for idx, (_, f) in enumerate(pairs):
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if pairs[tails[mid]][1] < f:
                lo = mid + 1
            else:
                hi = mid
        prev[idx] = tails[lo - 1] if lo > 0 else None
        if lo == len(tails):
            tails.append(idx)
        else:
            tails[lo] = idx
    k = tails[-1] if tails else None
    while k is not None:
        best.append(pairs[k])
        k = prev[k]
    return best[::-1]


def rows(pfile, plines, p0, p1, ffile, flines, f0, f1):
    """[(f_line | None, p_line | None, anchored)] aligning Python p0..p1 with Fortran f0..f1."""
    if ffile is None:
        return [(None, n, False) for n in range(p0, p1 + 1)]
    anchors = lis([(n, a) for n in range(p0, p1 + 1) if (a := anchor_of(plines[n - 1], f0, f1)) is not None])
    out, fp, pp = [], f0, p0
    for pn, fn in anchors + [(p1 + 1, f1 + 1)]:
        left, right = list(range(fp, fn)), list(range(pp, pn))
        for k in range(max(len(left), len(right))):
            out.append((left[k] if k < len(left) else None, right[k] if k < len(right) else None, False))
        if pn <= p1:
            out.append((fn, pn, True))
        fp, pp = fn + 1, pn + 1
    return out


def cell(kind, path, n, text):
    if n is None:
        return '<td class="n"></td><td class="c"></td>'
    return (f'<td class="n">{n}</td><td class="c" data-src="{kind}:{path}:{n}">{html.escape(text, quote=False)}'
            f'</td>')


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--code:#f3f1ea;--anchor:#fff3c4;--accent:#3b5b8c}
@media (prefers-color-scheme: dark){:root{--bg:#1b1c1e;--fg:#e8e6e1;--muted:#a09d96;--line:#34363a;--code:#232528;
--anchor:#3d3720;--accent:#8fb0e0}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:1500px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6em;margin:0 0 .3em}h2{font-size:1.2em;margin:2.2em 0 .3em;color:var(--accent)}
p,li{max-width:110ch}.meta{color:var(--muted);font-size:.92em}
nav ol{columns:2;max-width:110ch}
.gate{border-left:3px solid var(--accent);padding:.3em .8em;background:var(--code);max-width:110ch}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;margin-top:.8em}
table{border-collapse:collapse;width:100%;font:12.5px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
th{position:sticky;top:0;background:var(--code);text-align:left;padding:4px 6px;border-bottom:1px solid var(--line)}
td{vertical-align:top;padding:0 6px}td.c{white-space:pre;width:50%}
td.n{color:var(--muted);text-align:right;user-select:none;width:3.5em;border-left:1px solid var(--line)}
tr.a td{background:var(--anchor)}tr.seg td{border-top:2px solid var(--line);padding:2px 6px;color:var(--muted)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}
"""


def build():
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    parts = [f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>M1 side by side</title><style>{CSS}</style></head><body><main>
<h1>M1 readability review: ten routines side by side</h1>
<p class="meta" id="provenance" data-mitjax="{commit}" data-upstream="63cdc0b">Fortran: MITgcm master at 63cdc0b
(branch <code>pinned</code> of <code>$MJX_UPSTREAM</code>). Python: mitjax at {commit}. Written by
<code>docs/review/make_m1_side_by_side.py</code>; every code line below is checked against those files by
<code>mitjax/tests/test_m1_side_by_side.py</code>. Local file for the plan's Task 18 review; not published.</p>
<p>Each section shows a part of one routine. Rows are aligned by the <code># :NN</code> comments of the Python, which
cite the Fortran line; an aligned pair is highlighted. A Fortran statement that spans several lines is aligned on
its first line. Where the Python is vectorised, one statement stands for a whole DO nest, so the Fortran column is
the longer one. The notes say what deserves a careful look; the gate is the test that compares the routine with
gfortran (bitwise unless stated) and the job that ran it. Tier 1x job 27837912 tested master c01069d; every
excerpt below is identical in that commit (checked when the page was written). Deviations from the Fortran: none
in these ten.</p>
<nav><ol>"""]
    for k, r in enumerate(ROUTINES, 1):
        parts.append(f'<li><a href="#r{k}">{html.escape(r[0])}</a></li>')
    parts.append("</ol></nav>")
    for k, (title, lane, segs, notes, gate) in enumerate(ROUTINES, 1):
        parts.append(f'<section class="routine" id="r{k}"><h2>{k}. {html.escape(title)}</h2>')
        parts.append(f'<p class="meta">Lane: {html.escape(lane)}.</p><ul>')
        parts += [f"<li>{n}</li>" for n in notes]
        parts.append(f'</ul><p class="gate"><b>Gate.</b> {gate}</p>')
        parts.append('<div class="wrap"><table><thead><tr><th></th><th>Fortran (63cdc0b)</th><th></th>'
                     '<th>Python (mitjax)</th></tr></thead><tbody>')
        for pfile, start, end, ffile, f0, f1 in segs:
            plines = (REPO / pfile).read_text().split("\n")
            p0, p1 = locate(plines, start, end, pfile)
            flines = upstream_lines(ffile) if ffile else None
            head = (f"{ffile}:{f0}-{f1}" if ffile else "no Fortran counterpart") + f"  |  {pfile}:{p0}-{p1}"
            parts.append(f'<tr class="seg"><td colspan="4">{html.escape(head)}</td></tr>')
            for fn, pn, anchored in rows(pfile, plines, p0, p1, ffile, flines, f0, f1):
                left = cell("up", ffile, fn, flines[fn - 1] if fn else "")
                right = cell("repo", pfile, pn, plines[pn - 1] if pn else "")
                tr = '<tr class="a">' if anchored else "<tr>"
                parts.append(f"{tr}{left}{right}</tr>")
        parts.append("</tbody></table></div></section>")
    parts.append("</main></body></html>\n")
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)
