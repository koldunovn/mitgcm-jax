#!/usr/bin/env python3
"""Build dev/prototype/side_by_side.html (plan Task 8): for each routine three columns, Fortran `.F` at 63cdc0b |
Fortran-index style | plain-slices style, aligned on the Fortran line anchors (`# :NN`) of the Python code, plus the
measurements (dev/prototype/measurements.json). Self-contained HTML (no external resources).

    make_side_by_side.py [--out dev/prototype/side_by_side.html]
"""

import argparse
import ast
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from mitjax import paths  # noqa: E402

ROUTINES = [("mom_calc_ke", "pkg/mom_common/mom_calc_ke.F"),
            ("gad_dst3_adv_x", "pkg/generic_advdiff/gad_dst3_adv_x.F"),
            ("mom_vi_hdissip", "pkg/mom_vecinv/mom_vi_hdissip.F")]
ANCHOR = re.compile(r"#\s*:(\d+)")


def py_function(path, name):
    """(first line number, lines) of a top-level function, docstring included."""
    src = Path(path).read_text().splitlines()
    tree = ast.parse("\n".join(src))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node.lineno, src[node.lineno - 1:node.end_lineno]
    raise KeyError(name)


def fortran_routine(rel):
    """(first line number, lines) from the SUBROUTINE statement (with the preceding CBOP header) to END."""
    lines = (paths.UPSTREAM / rel).read_text().splitlines()
    start = next(n for n, l in enumerate(lines) if re.match(r"\s+SUBROUTINE", l))
    for n in range(start - 1, -1, -1):
        if lines[n].startswith("CBOP"):
            start = n
            break
    return start + 1, lines[start:]


def anchors(first, lines):
    """[(Fortran line, index into lines)] of anchor comments, kept only while increasing."""
    out, last = [], 0
    for n, l in enumerate(lines):
        m = ANCHOR.search(l)
        if m and int(m.group(1)) > last:
            out.append((int(m.group(1)), n))
            last = int(m.group(1))
    return out


def chunks(f_first, f_lines, py_a, py_b):
    """Rows of aligned (fortran, a, b) chunks: lists of (line number, text)."""
    (a_first, a_lines), (b_first, b_lines) = py_a, py_b
    aa = dict(anchors(a_first, a_lines))
    bb = dict(anchors(b_first, b_lines))
    common = sorted(set(aa) & set(bb))
    keep, la, lb = [], -1, -1                       # anchors increasing in all three
    for fl in common:
        if aa[fl] > la and bb[fl] > lb and fl >= f_first:
            keep.append(fl)
            la, lb = aa[fl], bb[fl]
    cuts_f = [0] + [fl - f_first for fl in keep] + [len(f_lines)]
    cuts_a = [0] + [aa[fl] for fl in keep] + [len(a_lines)]
    cuts_b = [0] + [bb[fl] for fl in keep] + [len(b_lines)]
    rows = []
    for m in range(len(cuts_f) - 1):
        rows.append(([(f_first + n, f_lines[n]) for n in range(cuts_f[m], cuts_f[m + 1])],
                     [(a_first + n, a_lines[n]) for n in range(cuts_a[m], cuts_a[m + 1])],
                     [(b_first + n, b_lines[n]) for n in range(cuts_b[m], cuts_b[m + 1])]))
    return rows


def code_cell(items, lang):
    out = []
    for num, text in items:
        t = html.escape(text)
        if lang == "f" and (text[:1] in ("C", "c", "!") or text.lstrip().startswith("!")):
            t = f'<span class="cm">{t}</span>'
        elif lang == "f" and text.startswith("#"):
            t = f'<span class="pp">{t}</span>'
        elif lang == "p":
            m = re.search(r"#.*$", text)
            if m and text.count('"') % 2 == 0:
                t = html.escape(text[:m.start()]) + f'<span class="cm">{html.escape(m.group(0))}</span>'
        out.append(f'<span class="ln">{num:4d}</span>{t}')
    return "\n".join(out)


CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a64; --rule:#dcd8cc; --code:#f3f1ea; --accent:#2f5d8a;
        --cm:#7a776d; --pp:#8a4b2f; --ok:#2e6b3a; --warn:#8a5a00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#171715; --fg:#e9e7e0; --muted:#a19e94; --rule:#3a3933; --code:#22221f; --accent:#8fb6dd;
  --cm:#8d8a80; --pp:#d39a7c; --ok:#8fcf9b; --warn:#e3b65a; } }
:root[data-theme="dark"] { --bg:#171715; --fg:#e9e7e0; --muted:#a19e94; --rule:#3a3933; --code:#22221f;
  --accent:#8fb6dd; --cm:#8d8a80; --pp:#d39a7c; --ok:#8fcf9b; --warn:#e3b65a; }
html, body { background:var(--bg); color:var(--fg); margin:0; }
body { font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; padding: 16px; max-width: 2200px; }
h1 { font-size: 1.5rem; margin: .2rem 0 .4rem; } h2 { font-size: 1.2rem; margin: 2rem 0 .5rem; }
p, li { max-width: 72rem; } .muted { color: var(--muted); }
code { background: var(--code); padding: 0 .25em; border-radius: 3px; font-size: .92em; }
table.m { border-collapse: collapse; margin: .5rem 0 1rem; font-size: .92rem; }
table.m th, table.m td { border-bottom: 1px solid var(--rule); padding: .3rem .6rem; text-align: left; vertical-align: top; }
table.m th { color: var(--muted); font-weight: 600; }
.ok { color: var(--ok); font-weight: 600; } .warn { color: var(--warn); font-weight: 600; }
.sbs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0 10px; overflow-x: auto; }
.sbs .hd { font-weight: 600; padding: .3rem .4rem; border-bottom: 2px solid var(--accent); position: sticky; top: 0;
           background: var(--bg); z-index: 1; }
.sbs pre { margin: 0; padding: .15rem .4rem; background: var(--code); font: 12.5px/1.45 ui-monospace, Menlo,
           Consolas, monospace; white-space: pre; overflow-x: auto; border-top: 1px dashed var(--rule); }
.ln { color: var(--muted); user-select: none; margin-right: .8em; }
.cm { color: var(--cm); } .pp { color: var(--pp); }
@media (max-width: 900px) { .sbs { grid-template-columns: 1fr; } .sbs .hd { position: static; } }
"""


def measurements_html(m):
    if not m:
        return "<p class='muted'>Measurements pending (dev/prototype/measurements.json not written yet).</p>"
    rows = "".join(f"<tr><td>{html.escape(r['gate'])}</td><td class='{r.get('cls', '')}'>{html.escape(r['result'])}"
                   f"</td><td class='muted'>{html.escape(r.get('detail', ''))}</td></tr>" for r in m["gates"])
    t = ("<table class='m'><tr><th>gate</th><th>result</th><th>detail</th></tr>" + rows + "</table>")
    if m.get("timing"):
        trs = "".join(f"<tr><td>{html.escape(r['size'])}</td><td>{html.escape(r['case'])}</td>"
                      f"<td>{r['farray_ms']:.4f}</td><td>{r['slices_ms']:.4f}</td><td>{r['ratio']:.3f}</td>"
                      f"<td>{html.escape(r['spread'])}</td></tr>" for r in m["timing"])
        t += ("<p>A100 (80 GB) per call, median of interleaved repeats, warm runs only: "
              "(t<sub>N</sub> − t<sub>W</sub>)/(N − W), production XLA flags. "
              + html.escape(m.get("timing_note", "")) + "</p><table class='m'><tr><th>size</th><th>routine (case)</th>"
              "<th>Fortran-index ms</th><th>plain slices ms</th><th>ratio</th><th>min..max (ms)</th></tr>"
              + trs + "</table>")
    return t


INTRO = """
<p>Three MITgcm master routines (pinned commit <code>63cdc0b</code>) translated literally into JAX in two array styles. Both
styles produce the same traced program; the choice between them is about reading the code next to the Fortran.
<b>Nothing here is decided yet:</b> Nikolay (and Martin Losch) choose the style for all physics code from this page.</p>
<ul>
<li><b>Fortran-index style</b> (<code>mitjax/farray.py</code>): arrays keep their Fortran declaration
(<code>uFld(1-OLx:sNx+OLx,1-OLy:sNy+OLy)</code>); a <code>DO</code> loop becomes a loop index object
(<code>i = loop_i(1-OLx, sNx+OLx-1)</code>), and <code>uFld[i+1, j]</code> means <code>uFld(i+1,j)</code> for every i, j of the
loops at once. <code>KE = KE.at[i, j].set(expr)</code> is the assignment <code>KE(i,j) = expr</code> inside the loops: the
points the loops do not reach keep their old values, as in Fortran. Indices are checked against the declared
bounds; an error reads e.g. <code>uFld(i+2, j): i+2 = 1..8 is outside the declared -1:7 of dimension 1</code>.</li>
<li><b>Plain-slices style</b>: the same arrays as plain JAX arrays stored <code>[tile, k, j, i]</code> (index order
reversed, tile first); <code>F(lo, hi, OL)</code> turns a Fortran loop range into a Python slice, shifted indices are
separate slices (<code>ip1</code> for i+1), level k is storage index <code>kk = k-1</code>, a 1-D field along j is
<code>cosFacU[:, j, None]</code>.</li>
<li>In both styles: the <code>bi,bj</code> tile loop is implicit (every array holds all tiles); a statement inside the
<code>DO j / DO i</code> loops runs on the whole (i,j) range at once (valid because no point reads a value written
in the same loop); CPP options are static <code>cfg.NAME</code> flags (<code>if cfg.ALLOW_AUTODIFF:</code> mirrors
<code>#ifdef ALLOW_AUTODIFF</code>); GRID.h and PARAMS.h variables keep their names (<code>grid.recip_dxC</code>,
<code>params.viscAhD</code>); <code>x.at[...].set(v)</code> returns a new array (JAX arrays are immutable).
Comments <code># :NN</code> cite the Fortran line.</li>
</ul>
"""


FINDINGS = """
<h2>Findings</h2>
<ul>
<li><b>Harness.</b> A genmake2 build of global_ocean.90x40x15/code with THE_MAIN_LOOP replaced
(<code>reference/replay</code>, after the ECCO port's adx harness): INITIALISE_FIXED gives the experiment's real grid
(65,624 dry and 4,669 partial cells of 138,240 incl. halos); the harness calls the routines on synthetic inputs for
all 36 tiles x 15 levels and writes every output array at every point. Its preprocessed routines are byte-identical to
the oracle build's. cosFacU, cosFacV, recip_deepFacC (all 1 in this experiment) and viscAhZ, viscA4Z (equal to the D
values) are overridden with distinct synthetic values, so a missing factor or a swap shows.</li>
<li><b>XLA:CPU flushes subnormals to zero; gfortran does not.</b> With the gate flags XLA's float64 divide equals
gfortran's for every pair whose operands and quotient are normal numbers, whether the divisor is an array, a traced
scalar or a Python constant. Every difference involves a subnormal, and XLA returns +-0 there. With algsimp on, a
scalar divisor is rewritten to x*(1/d), which changes 13 % (d = 1000) to 36 % (d = 7) of normal values; that rewrite, not the divide
instruction, explains the earlier "13 % of divides off" (fesom_jax).</li>
<li><b>Reverse mode and the upwind formula.</b> In GAD_DST3_ADV_X, uT = 0.5*(u+|u|)*A + 0.5*(u-|u|)*B. Where A is
astronomically large (halo rows beyond the domain's N/S edges, where the grid's recip_dxC is 3.7e10 1/m and the CFL
number 6e14), forward mode gets the exact derivative but reverse mode cancels +-0.5*A*ct and loses the B term: the
adjoint w.r.t. uTrans is 0 at 3 of 9,216 points. Both styles are identical here (bitwise gradients); a TAF adjoint
of the same statement would accumulate the same way. The FD gate is applied at points with CFL <= 1.</li>
<li><b>Timing.</b> Device time is the same (identical programs): with identical call signatures the ratio
Fortran-index / slices is 0.988 to 1.002 at both sizes. With FArrays passed as jit arguments, the small
(global_ocean-size, dispatch-bound, 0.07 to 0.1 ms) calls are 6 to 9 % slower, about 5 to 6 microseconds per call
(reproduced in a second job): JAX flattens each FArray argument (17 grid fields) with a Python callback on every
call. At the LLC90-like size the same comparison gives 0.993 to 0.999. In a model the state crosses the jit boundary
once per call of a whole multi-step program, where this cost does not matter; a driver can also pass plain arrays
and declare the FArrays inside.</li>
</ul>
<h2>Readability trade-offs (for the decision, not a recommendation)</h2>
<table class='m'><tr><th></th><th>Fortran-index style</th><th>plain slices</th></tr>
<tr><td>index order, bounds</td><td>Fortran's: <code>uFld[i+1, j]</code>, <code>hFacW[i, j, k]</code>, loop bounds as in the
DO statement</td><td>reversed storage order <code>[:, j, i]</code>, <code>hFacW[:, kk, j, i]</code>, <code>kk = k-1</code>;
bounds converted by <code>F(lo, hi, OL)</code></td></tr>
<tr><td>shifted indices</td><td><code>i+1</code>, <code>j-1</code> written inline</td><td>one named slice per shift
(<code>im2, im1, ip1, jp1</code>), defined next to the loop</td></tr>
<tr><td>fixed Fortran indices</td><td><code>uT.at[sNx+OLx, j]</code></td><td>storage numbers:
<code>uT.at[:, j, sNx+2*OLx-1]</code></td></tr>
<tr><td>1-D fields</td><td><code>cosFacU[j]</code> broadcasts along i</td><td><code>cosFacU[:, j, None]</code> by hand
(without None it silently aligns j with i when nj == ni)</td></tr>
<tr><td>points a loop does not write</td><td>keep their value (built in)</td><td>keep their value (<code>.at[].set</code> on
the slice)</td></tr>
<tr><td>checks</td><td>bounds, axis (an i loop on a j dimension), declared shape (Nr vs Nr+1), value shape; messages in
Fortran order</td><td>none: an out-of-range slice is clipped silently</td></tr>
<tr><td>what a reader must learn</td><td>FArray, loop_i/loop_j, <code>.at[i, j].set</code>; trust a 310-line module</td>
<td>numpy slicing only</td></tr>
<tr><td>code length</td><td colspan=2>the same (MOM_CALC_KE 52/51, GAD_DST3_ADV_X 25/27, MOM_VI_HDISSIP 98/98 code
lines)</td></tr>
<tr><td>program, gradients, speed</td><td colspan=2>identical programs, bitwise-identical outputs and gradients, equal
device time</td></tr>
</table>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(HERE / "side_by_side.html"))
    a = ap.parse_args(argv)
    mpath = HERE / "measurements.json"
    m = json.loads(mpath.read_text()) if mpath.exists() else None
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width, initial-scale=1'><title>Array Style Comparison</title>"
             f"<style>{CSS}</style></head><body>",
             "<h1>MITgcm in JAX: two array styles, side by side</h1>",
             "<p class='muted'>mitjax plan Task 8 (decision gate). Experiment options: global_ocean.90x40x15/code "
             "(36 tiles of 10x10, OLx=OLy=3, Nr=15; ALLOW_AUTODIFF undefined, ALLOW_DIAGNOSTICS defined, "
             "MOM_VI_ORIGINAL_VISCA4 and OLD_DST3_FORMULATION undefined).</p>",
             INTRO, "<h2>Measurements</h2>", measurements_html(m), FINDINGS]
    for name, rel in ROUTINES:
        f_first, f_lines = fortran_routine(rel)
        pa = py_function(HERE / "style_farray.py", name)
        pb = py_function(HERE / "style_slices.py", name)
        parts.append(f"<h2>{name.upper()} <span class='muted'>({rel} @63cdc0b)</span></h2><div class='sbs'>"
                     f"<div class='hd'>Fortran {rel.split('/')[-1]}</div><div class='hd'>Fortran-index style "
                     f"(style_farray.py)</div><div class='hd'>plain slices (style_slices.py)</div>")
        for fc, ac, bc in chunks(f_first, f_lines, pa, pb):
            h = max(len(fc), len(ac), len(bc))
            cells = []
            for items, lang in ((fc, "f"), (ac, "p"), (bc, "p")):
                body = code_cell(items, lang)
                body += "\n" * (h - len(items))
                cells.append(f"<pre>{body}</pre>")
            parts.append("".join(cells))
        parts.append("</div>")
    parts.append("<p class='muted'>Generated by dev/prototype/make_side_by_side.py from the committed sources.</p>"
                 "</body></html>")
    Path(a.out).write_text("\n".join(parts))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
