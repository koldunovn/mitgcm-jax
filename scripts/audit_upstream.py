#!/usr/bin/env python3
"""c66g -> master audit of the MITgcm Fortran that copied infrastructure encodes (plan Task 7a).

For every file in MODULES (and for any routine or file list a caller passes) it runs
`git -C <upstream> diff checkpoint66g pinned -- <file>` and classifies every hunk as one of

    forward value   changes what a forward run computes (the default for any executable statement)
    I/O             reads or writes files: READ/OPEN/CLOSE/NAMELIST, MDS/READ_FLD/MNC calls, all of pkg/mdsio + pkg/rw
    AD-only         compiled only into adjoint builds: inside #ifdef ALLOW_AUTODIFF / ALLOW_ADJOINT* / ALLOW_TAPENADE /
                    ALLOW_OPENAD ..., CADJ/$TAF directives, *.flow, *_ad.F, *_mad.F, pkg/autodiff. NOTE: ALLOW_AUTODIFF
                    code still runs in the FORWARD sweep of a code_ad build (R5 tutorial_global_oce_optim): "AD-only"
                    means "only in AD builds", not "only in the backward sweep".
    diagnostics     WRITE/PRINT, PRINT_MESSAGE/PRINT_ERROR, WRITE_0D/1D, DEBUG_*, TIMER_*, DIAGNOSTICS_FILL, MON_*,
                    STOP/error counters, IF blocks on debugLevel/debugMode only
    cosmetic        no effect: comments, blank lines, whitespace/case/re-wrapping (token-equal code), declarations,
                    #include of plain headers, CPP comment text, NEC/OpenMP directives, _BEGIN/_END_MASTER

Rules work per changed line, on the statement the line belongs to (continuation lines follow their initial line) and
on the CPP context (#if stack) of that line in its own version of the file. A hunk gets the most severe class of its
lines (forward value > I/O > AD-only > diagnostics > cosmetic). Lines the rules cannot decide (a lone #ifdef/#endif
that wraps existing code, an #include of an *_OPTIONS.h header, a declaration that changes a variable's type, EQUIVALENCE)
make the hunk "undecided" unless another line already makes it forward value. Every undecided hunk needs an entry in
OVERRIDES (class + content sha + one-line reason); an override may also change a rule-decided hunk, but never below
its *floor* (the most severe class among lines decided by a certain rule: comments, CADJ, AD context, diagnostics and
I/O statements; the default "forward value" of an executable statement is not certain). Overrides that name a missing
hunk, carry a stale content sha, fall below the floor, or repeat the rules' own answer are errors.

Usage
    audit_upstream.py                        print the generated tables (markdown)
    audit_upstream.py --update FILE          rewrite the generated section of FILE (docs/AUDIT_C66G_MASTER.md)
    audit_upstream.py --check FILE           exit 1 unless FILE holds exactly the current generated section, no hunk
                                             is undecided, every override is valid, and every forward-value hunk is
                                             referenced (`path#n` or `name.F#n-m`) in FILE's hand-written part
    audit_upstream.py --routines NAME ...    classify the files that define these routines (Task 4: instrumented
                                             routines), print one row per hunk; exit 2 if any hunk is undecided
    audit_upstream.py --files PATH ...       same for repository paths (model/src/forward_step.F, ...)
    audit_upstream.py --show PATH#N ...      print hunks with their class, decision and content sha (for OVERRIDES)
    --upstream DIR                           the MITgcm clone (default $MJX_UPSTREAM, else mitjax/paths.py UPSTREAM)

Library interface (stdlib + git only; load this file by path or import scripts.audit_upstream):
    audit_items(items, upstream=None, overrides=OVERRIDES) -> list[FileAudit]
        items: routine names (FORWARD_STEP, forward_step) and/or repository paths. A routine name is resolved at both
        tags (`git grep` for SUBROUTINE/FUNCTION NAME in *.F, *.template); a routine that lives in different files at
        the two tags is reported as moved and the two files are diffed against each other.
        FileAudit: .old_path, .new_path (None if absent at that tag), .status ("identical", "modified", "added",
        "removed", "moved"), .hunks (list of Hunk: .hid "path#n", .header, .cls, .how ("rule" | "override: <reason>"),
        .floor, .ctx (CPP context of the changed code), .summary), .counts() -> {class: n}.
    classify_texts(old_text, new_text, path="x.F") -> list[Hunk]   rules only, for synthetic tests (git diff --no-index)
"""

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD, NEW = "checkpoint66g", "pinned"
PINNED_SHA = "63cdc0b9602b46bda69df37c5eb9e17396116f67"
C66G_SHA = "4207dc8bfb14bb14d332cc3af0d2621adb17e3d6"

# Hunk shapes depend on git's diff settings: pin every one that a user's config could change.
GIT_DIFF_CONFIG = ("-c", "diff.algorithm=myers", "-c", "diff.indentHeuristic=true", "-c", "diff.interHunkContext=0",
                   "-c", "diff.suppressBlankEmpty=false", "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false")

BEGIN = "<!-- BEGIN AUDIT (written by scripts/audit_upstream.py --update; do not edit) -->"
END = "<!-- END AUDIT -->"

FWD, IO, AD, DIAG, COS = "forward value", "I/O", "AD-only", "diagnostics", "cosmetic"
UNDECIDED = "undecided"
CLASSES = (FWD, IO, AD, DIAG, COS)
SEVERITY = {COS: 0, DIAG: 1, AD: 2, IO: 3, FWD: 4}

# --------------------------------------------------------------------------------------------------------------------
# Modules to copy from the ECCO port (lessons [E§11]) and the Fortran each one encodes. Paths are repository paths of
# MITgcm; a pair (old, new) is a file renamed between the tags. "deferred" modules are audited now and re-audited when
# copied. R module names are relative to the ECCO handoff repository (never a path: the audit does not read it).
# --------------------------------------------------------------------------------------------------------------------
_EE = "eesupp/src/"
_X2 = "pkg/exch2/"
_EXCH_TEMPLATES = [_EE + f"exch_{n}_rx.template" for n in
                   ("xy", "uv_xy", "z_3d", "uv_agrid_3d", "uv_bgrid_3d", "3d", "sm_3d", "uv_dgrid_3d", "uv_3d", "s3d",
                    "xyz", "uv_xyz")]
_EXCH2 = ([_X2 + f"exch2_{n}_rx.template" for n in
           ("3d", "z_3d", "sm_3d", "s3d", "uv_3d", "uv_agrid_3d", "uv_bgrid_3d", "uv_cgrid_3d", "uv_dgrid_3d")]
          + [_X2 + f"exch2_{n}.template" for n in ("rx1_cube", "rx2_cube")]
          + [_X2 + f"exch2_{a}_rx{b}.template" for a in ("get", "put", "send", "recv") for b in (1, 2)]
          + [_X2 + f for f in ("exch2_get_scal_bounds.F", "exch2_get_uv_bounds.F", "W2_EXCH2_SIZE.h",
                               "W2_EXCH2_TOPOLOGY.h", "W2_EXCH2_PARAMS.h", "W2_EXCH2_BUFFER.h", "W2_OPTIONS.h",
                               "w2_readparms.F", "w2_map_procs.F", "w2_e2setup.F", "w2_eeboot.F",
                               "w2_set_single_facet.F", "w2_set_gen_facets.F", "w2_set_cs6_facets.F",
                               "w2_set_myown_facets.F", "w2_set_map_tiles.F", "w2_set_map_cumsum.F",
                               "w2_set_tile2tiles.F", "w2_set_f2f_index.F")]
          + [_EE + f"fill_cs_corner_{n}.F" for n in ("ag_rl", "tr_rl", "uv_rl", "uv_rs")])
_EXCH1 = ([_EE + f for f in ("exch0_rx.template", "exch1_rx.template", "exch1_rx_cube.template",
                             "exch1_uv_rx_cube.template", "exch1_z_rx_cube.template", "exch1_bg_rx_cube.template",
                             "exch_rx_send_put_x.template", "exch_rx_send_put_y.template",
                             "exch_rx_recv_get_x.template", "exch_rx_recv_get_y.template", "exch_init.F",
                             "ini_communication_patterns.F", "exch_cycle_ebl.F")]
          + ["eesupp/inc/" + f for f in ("EXCH.h", "EESUPPORT.h", "EEPARAMS.h", "CPP_EEOPTIONS.h", "CPP_EEMACROS.h")])
_GSUM = [_EE + f for f in ("global_sum_tile.F", "global_sum.F", "global_max.F", "global_sum_singlecpu.F")] + \
        ["eesupp/inc/" + f for f in ("GLOBAL_SUM.h", "GLOBAL_MAX.h", "CPP_EEOPTIONS.h")] + \
        [(_EE + "global_vec_sum.F", _EE + "global_sum_vector.F")]
_MDS = (["pkg/mdsio/" + f for f in ("mdsio_read_field.F", "mdsio_facef_read.F", "mdsio_rd_rec_rl.F",
                                    "mdsio_rd_rec_rs.F", "mdsio_seg4torl.F", "mdsio_seg8torl.F", "mdsio_seg4tors.F",
                                    "mdsio_seg8tors.F", "mdsio_pass_r4torl.F", "mdsio_pass_r8torl.F",
                                    "mdsio_pass_r4tors.F", "mdsio_pass_r8tors.F", "mdsio_buffertorl.F",
                                    "mdsio_buffertors.F", "mdsio_read_meta.F", "mdsio_check4file.F", "MDSIO_OPTIONS.h",
                                    "MDSIO_BUFF_3D.h")]
        + ["pkg/rw/" + f for f in ("read_fld_xy_rl.F", "read_fld_xy_rs.F", "read_fld_xyz_rl.F", "read_fld_xyz_rs.F",
                                   "read_rec.F", "RW_OPTIONS.h")]
        + [_EE + f for f in ("mds_byteswapr4.F", "mds_byteswapr8.F", "mds_reclen.F", "mdsfindunit.F")])
_CG2D = ["model/src/" + f for f in ("cg2d.F", "cg2d_sr.F", "cg2d_nsa.F", "cg2d_ex0.F", "ini_cg2d.F", "update_cg2d.F",
                                    "solve_for_pressure.F")] + \
        ["model/inc/CG2D.h", "model/inc/SOLVE_FOR_PRESSURE.h", "model/inc/CPP_OPTIONS.h", _EE + "exch_s3d_rx.template",
         "pkg/autodiff/cg2d.flow", "pkg/autodiff/cg2d_mad.F", "pkg/autodiff/AUTODIFF_PARAMS.h"]
_DRIVER = ["model/src/" + f for f in ("forward_step.F", "the_main_loop.F", "adams_bashforth2.F", "adams_bashforth3.F")] + \
          ["pkg/autodiff/tamc.h", "pkg/autodiff/tamc_keys.h"]
_JAXDUMP = (["model/src/" + f for f in ("forward_step.F", "do_oceanic_phys.F", "dynamics.F", "solve_for_pressure.F",
                                        "thermodynamics.F", "temp_integrate.F", "salt_integrate.F")]
            + ["pkg/exf/" + f for f in ("exf_getforcing.F", "exf_radiation.F", "exf_bulkformulae.F")]
            + ["pkg/seaice/" + f for f in ("seaice_model.F", "seaice_dynsolver.F", "seaice_lsr.F", "seaice_advdiff.F",
                                           "seaice_growth.F")]
            + ["model/inc/" + f for f in ("SIZE.h", "PARAMS.h", "SURFACE.h", "GRID.h", "DYNVARS.h", "FFIELDS.h",
                                          "CG2D.h")]
            + ["eesupp/inc/EEPARAMS.h", "pkg/exf/EXF_PARAM.h", "pkg/exf/EXF_FIELDS.h", "pkg/seaice/SEAICE_SIZE.h",
               "pkg/seaice/SEAICE_PARAMS.h", "pkg/seaice/SEAICE.h", _EE + "print.F", _EE + "mdsfindunit.F"])
_ADHELP = ["pkg/autodiff/" + f for f in ("autodiff_readparms.F", "autodiff_inadmode_set_ad.F",
                                         "autodiff_inadmode_unset_ad.F", "autodiff_inadmode_set.F",
                                         "autodiff_inadmode_unset.F", "autodiff_inadmode.flow", "AUTODIFF_PARAMS.h",
                                         "AUTODIFF_OPTIONS.h", "cg2d.flow", "zero_adj.F", "g_zero_adj.F")] + \
          ["model/src/temp_integrate.F", "model/src/salt_integrate.F", "model/src/dynamics.F",
           "pkg/mom_common/mom_calc_visc.F", "pkg/gmredi/GMREDI_OPTIONS.h", "pkg/gmredi/gmredi_slope_limit.F",
           "pkg/gmredi/gmredi_slope_psi.F"]
_PARAMS = ["model/src/" + f for f in ("set_defaults.F", "ini_parms.F", "packages_boot.F", "packages_readparms.F")] + \
          ["model/inc/PARAMS.h", "pkg/diagnostics/diagnostics_is_on.F"] + \
          [_EE + f for f in ("nml_set_terminator.F", "nml_change_syntax.F", "eeset_parms.F", "open_copy_data_file.F")] + \
          ["pkg/autodiff/autodiff_readparms.F", _X2 + "w2_readparms.F"]
_LAYOUT = ["model/inc/SIZE.h", _X2 + "w2_e2setup.F", _X2 + "w2_set_single_facet.F", _X2 + "W2_EXCH2_TOPOLOGY.h"]

# (key, title, R modules, mitjax target, files, status)
MODULES = [
    ("exch2", "exch2 maps and the single-device exchanger",
     "parallel/exchange.py, scripts/make_exch_maps.py, jaxdump.F JAXDUMP_EXCH_PROBE",
     "mitjax/eesupp/{exch_maps,exchange}.py, scripts/make_exch_maps.py (Task 7b)", _EXCH_TEMPLATES + _EXCH2, "copy"),
    ("exch1", "eesupp (exch1) exchange: experiments without pkg/exch2",
     "none (the ECCO port probed only exch2; new probe in Task 4)",
     "mitjax/eesupp/exchange.py, same gather-map format (Task 7b)", _EXCH_TEMPLATES + _EXCH1, "new"),
    ("global_sum", "fixed-order global sums",
     "parallel/global_sum.py, Exchanger.global_max", "mitjax/eesupp/global_sum.py (Task 7b)", _GSUM, "copy"),
    ("mds", "MDS readers: readBinaryPrec, global and tiled files, MDS_FACEF_READ",
     "io/mds.py", "mitjax/io/mds.py (Task 7b)", _MDS, "copy"),
    ("cg2d", "cg2d solver, its normalisation and the derivative rule",
     "core/cg2d.py", "mitjax/ad/cg2d_rule.py (Task 7c); forward cg2d in M1 Task 12", _CG2D, "copy"),
    ("driver", "checkpoint stack and gradient drivers",
     "adjoint/checkpoint.py, adjoint/grad.py", "mitjax/drivers/{checkpoint,grad}.py (Task 7c)", _DRIVER, "copy"),
    ("jaxdump", "jaxdump shim: instrumented routines and the headers jaxdump.F reads",
     "reference/jaxdump/{jaxdump.F,JAXDUMP.h,instrument.py}", "reference/jaxdump/ (Task 4)", _JAXDUMP, "copy"),
    ("ad_helpers", "AD helpers: ad_skip, modes (deferred to the first milestone with a data.autodiff approximation)",
     "ops/ad_skip.py, adjoint/modes.py", "mitjax/ad/{ad_skip,modes}.py (deferred, plan Task 7c)", _ADHELP, "deferred"),
    ("params", "params pytree and namelist reader",
     "params_io.py, io/namelist.py", "mitjax/params_io.py, mitjax/io/namelist.py, mitjax/config/ (Task 6)", _PARAMS,
     "copy"),
    ("layout", "layout and Fortran-index helpers", "layout.py, grid/geometry.py",
     "mitjax/farray.py (Task 8), mitjax/eesupp/tiles.py (Task 7b)", _LAYOUT, "copy"),
]
# Modules with no Fortran dependency ([E§11]): sharded exchange (maps + GLOBAL_SUM_TILE only), dump reader and diff
# tools (record format of R's own jaxdump.F), test manifest/runner/verdict, libm (glibc + gfortran, Task 7c).

# --------------------------------------------------------------------------------------------------------------------
# Manual overrides: {(path, hunk number): (class, content sha, reason)}. Only for hunks the rules leave undecided or
# get wrong; see the module docstring for what makes an override invalid.
# --------------------------------------------------------------------------------------------------------------------
OVERRIDES = {
    ("model/src/cg2d.F", 14): (
        DIAG, "ac045dc7c0",
        "new block runs only if debugLevel.GE.debLevE .AND. printResidualFreq.EQ.1 (cg2d.F:392): residual of the "
        "returned x into locals errTile/sumRHS, printed; its _EXCH_XY_RL(cg2d_x) is repeated by "
        "solve_for_pressure.F:315 anyway; the removed lines were commented out"),
    ("pkg/exch2/w2_eeboot.F", 3): (
        DIAG, "48d3e442b5",
        "file name of the w2_tile_topology.<proc>.log listing (digits of the processor number)"),
    ("model/src/ini_parms.F", 17): (
        DIAG, "9edfc57fdb",
        "default printResidualFreq 0 -> -1 (1 if debugLevel>=debLevE, ini_parms.F:787-789): selects cg2d/cg3d "
        "residual printing only (cg2d.F:198,330,392) and its config_summary line"),
    ("eesupp/src/mds_byteswapr4.F", 1): (
        IO, "a961e207ed",
        "the FAST_BYTESWAP variant (arr as integer(kind=4), shifts) is removed; the CHARACTER*(*) byte swap that "
        "remains is the c66g default path (FAST_BYTESWAP was #undef in CPP_EEOPTIONS.h)"),
    ("eesupp/src/mds_byteswapr8.F", 1): (
        IO, "e0e5acc370", "same as mds_byteswapr4.F#1 for 8-byte words"),
    ("eesupp/src/print.F", 2): (
        DIAG, "51a2d04960", "idString CHARACTER*9 -> *13 and a format string: process/thread tag of messages"),
    ("eesupp/src/print.F", 4): (
        DIAG, "d956c01a1e", "same as print.F#2 in the second message routine"),
    ("model/src/temp_integrate.F", 5): (
        AD, "0988666212",
        "#endif of an ALLOW_AUTODIFF block moved above the CADJ STOREs, which get their own ALLOW_AUTODIFF_TAMC "
        "block; only CADJ lines change build membership, no executable statement"),
    ("model/src/salt_integrate.F", 5): (
        AD, "92c00dcf70", "same restructuring as temp_integrate.F#5 for salt"),
    ("model/src/the_main_loop.F", 12): (
        AD, "c868a476de",
        "TAF tape INIT/STORE directives rewritten; ikey_dynamics = 1 moved from an ALLOW_AUTODIFF to an "
        "ALLOW_AUTODIFF_TAMC block; nIter0 assignment unchanged; no plain-build statement changes"),
}

# --------------------------------------------------------------------------------------------------------------------
# Upstream access
# --------------------------------------------------------------------------------------------------------------------


def default_upstream():
    """$MJX_UPSTREAM, else UPSTREAM of mitjax/paths.py (loaded by path: this script runs without the package)."""
    v = os.environ.get("MJX_UPSTREAM", "").strip()
    if v:
        return Path(v).expanduser()
    spec = importlib.util.spec_from_file_location("_mjx_paths", REPO / "mitjax" / "paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Path(mod.UPSTREAM)


class Upstream:
    def __init__(self, path=None):
        self.path = Path(path) if path else default_upstream()
        self._blobs = {}
        # the git ref of each audited commit: its tag where the clone has it, else its sha (a fresh clone of MITgcm
        # has the upstream tag checkpoint66g but not this project's local tag `pinned`: fresh install job 27969756)
        self.ref = {}
        for tag, sha in ((OLD, C66G_SHA), (NEW, PINNED_SHA)):
            got = self.git("rev-parse", "--verify", "-q", f"{tag}^{{commit}}", check=False).strip()
            if got and got != sha:
                raise SystemExit(f"{self.path}: {tag} is {got}, expected {sha}")
            if not got:
                self.git("rev-parse", "--verify", f"{sha}^{{commit}}")       # raises if the clone lacks the commit
            self.ref[tag] = tag if got else sha

    def git(self, *args, check=True):
        r = subprocess.run(["git", "-C", str(self.path), "-c", "core.quotepath=false", *args],
                           capture_output=True, text=True, errors="replace")
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout

    def prefetch(self, paths):
        """Read every (tag, path) blob in one `git cat-file --batch` call (one process instead of two per file)."""
        keys = [(t, p) for p in paths for t in (OLD, NEW) if (t, p) not in self._blobs]
        if not keys:
            return
        r = subprocess.run(["git", "-C", str(self.path), "cat-file", "--batch"], capture_output=True,
                           input="".join(f"{self.ref[t]}:{p}\n" for t, p in keys).encode())
        out, pos = r.stdout, 0
        for key in keys:
            end = out.index(b"\n", pos)
            header = out[pos:end].decode()
            pos = end + 1
            if header.endswith(" missing"):
                self._blobs[key] = None
                continue
            size = int(header.split()[2])
            self._blobs[key] = out[pos:pos + size].decode("latin-1")
            pos += size + 1

    def exists(self, tag, path):
        return self.text(tag, path) is not None

    def text(self, tag, path):
        key = (tag, path)
        if key not in self._blobs:
            r = subprocess.run(["git", "-C", str(self.path), "show", f"{self.ref[tag]}:{path}"], capture_output=True)
            self._blobs[key] = r.stdout.decode("latin-1") if r.returncode == 0 else None
        return self._blobs[key]

    DIFF = [*GIT_DIFF_CONFIG, "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames", "--unified=3"]

    def prefetch_diffs(self, paths):
        """One `git diff OLD NEW -- <paths>` split per file (same output as one call per file)."""
        self._diffs = getattr(self, "_diffs", {})
        todo = [p for p in paths if p not in self._diffs]
        if not todo:
            return
        out = self.git(*self.DIFF, self.ref[OLD], self.ref[NEW], "--", *todo)
        for p in todo:
            self._diffs[p] = ""
        for chunk in re.split(r"(?m)^(?=diff --git )", out):
            m = re.match(r"diff --git a/(\S+) b/", chunk)
            if m and m.group(1) in self._diffs:
                self._diffs[m.group(1)] = chunk

    def diff(self, old_path, new_path):
        """Unified diff (3 lines of context, myers) of old_path at OLD vs new_path at NEW; '' if identical."""
        if old_path == new_path:
            cached = getattr(self, "_diffs", {}).get(old_path)
            return cached if cached is not None else self.git(*self.DIFF, self.ref[OLD], self.ref[NEW], "--", old_path)
        return self.git(*self.DIFF, f"{self.ref[OLD]}:{old_path}", f"{self.ref[NEW]}:{new_path}")

    def define_sites(self, tag, routine):
        """Files at `tag` that define SUBROUTINE/FUNCTION `routine` (case-insensitive) in *.F or *.template."""
        pat = rf"^      *([A-Za-z_*0-9 ]*FUNCTION|SUBROUTINE) +{re.escape(routine)}\b"
        out = self.git("grep", "-l", "-i", "-E", pat, self.ref[tag], "--", "*.F", "*.template", check=False)
        hits = sorted({line.split(":", 1)[1] for line in out.splitlines() if ":" in line})
        # verification/ holds experiment-specific copies, pkg/openad and pkg/tapenade stubs for other AD tools:
        # never the model's routine
        return [h for h in hits if not h.startswith(("verification/", "tools/", "utils/", "doc/", "pkg/openad/",
                                                     "pkg/tapenade/"))]


# --------------------------------------------------------------------------------------------------------------------
# Fortran line analysis
# --------------------------------------------------------------------------------------------------------------------
AD_MACRO = re.compile(r"^(ALLOW_AUTODIFF\w*|ALLOW_ADJOINT\w*|ALLOW_TANGENTLINEAR\w*|ALLOW_ADMTLM|ALLOW_TAPENADE|"
                      r"ALLOW_OPENAD\w*|ALLOW_TAMC\w*|AUTODIFF_\w+|ALLOW_DIVIDED_ADJOINT|ALLOW_GRDCHK|"
                      r"ALLOW_ECCO_OPTIMIZATION|ALLOW_[A-Z0-9_]*_AD)$")
AD_FILE = re.compile(r"(\.flow$|_ad\.F$|_ad\.template$|_tl\.F$|_mad\.F$|_g\.F$|_ad_diff\.list$|"
                     r"^pkg/autodiff/|^pkg/tapenade/|^pkg/openad/|exch_tap_[bd]\.F$|_cube_b\.F$|_12_d\.F$)")
IO_FILE = re.compile(r"(^pkg/mdsio/|^pkg/rw/|^eesupp/src/mds_\w+\.F$|^eesupp/src/mdsfindunit\.F$|"
                     r"^eesupp/src/nml_\w+\.F$|^eesupp/src/open_copy_data_file\.F$)")
DIAG_FILE = re.compile(r"^eesupp/src/print\.F$")
DIAG_CALL = re.compile(r"^CALL(PRINT_MESSAGE|PRINT_ERROR|PRINT_LIST_\w+|WRITE_0D_\w+|WRITE_1D_\w+|WRITE_COPY1D_\w+|"
                       r"DEBUG_\w+|TIMER_\w+|DIAGNOSTICS_\w+|DIAG_\w+|MON_\w+|MONITOR\w*|PLOT_FIELD_\w+|BAR_CHECK|"
                       r"ALL_PROC_DIE|COMM_STATS|PACKAGES_PRINT_MSG|WRITE_FULLARRAY_\w+|PRINT_MAXLOC\w*|PRINT_MINLOC\w*|TIMEAVE_\w+|\w+_TAVE)\b")
IO_CALL = re.compile(r"^CALL(READ_\w+|WRITE_FLD_\w+|WRITE_REC_\w+|WRITE_LOCAL_\w+|WRITE_GLVEC_\w+|MDS_?\w+|MNC_\w+|"
                     r"OPEN_COPY_DATA_FILE|NML_\w+|WRITE_PICKUP\w*|READ_PICKUP\w*|WRITE_STATE\w*|MDSFINDUNIT|"
                     r"ACTIVE_READ\w*|ACTIVE_WRITE\w*|GET_WRITE_GLOBAL_FLD|SET_WRITE_GLOBAL_FLD)\b")
IO_STMT = re.compile(r"^(READ\(|OPEN\(|CLOSE\(|INQUIRE\(|REWIND|BACKSPACE|NAMELIST/|FLUSH)")
DIAG_STMT = re.compile(r"^(WRITE\(|PRINT\*|PRINT'|STOP($|'|\"|\d))")
DECL = re.compile(r"^(_RL|_RS|_R4|_R8|_RX|_RES|REAL\*\d+|REAL|DOUBLEPRECISION|INTEGER\*\d+|INTEGER|LOGICAL\*\d+|LOGICAL"
                  r"|CHARACTER\*\([^)]*\)|CHARACTER\*\d+|CHARACTER|COMPLEX\*\d+|COMPLEX|COMMON/|EXTERNAL|IMPLICITNONE"
                  r"|INTRINSIC)")
DEBUG_COND = re.compile(r"^(DEBUGLEVEL|DEBUGMODE|DEBLEV\w*|\.GE\.|\.GT\.|\.LE\.|\.LT\.|\.EQ\.|\.NE\.|\.AND\.|\.OR\.|"
                        r"\.NOT\.|\(|\)|\d+|DEBUGLEVEL\w*|PRINTRESIDUALFREQ|USEDIAGNOSTICS|DIAGNOSTICS_IS_ON\w*|"
                        r"MOD|MYITER|'[^']*'|,|MYTHID|DEBLEVA|DEBLEVB|DEBLEVC|DEBLEVD|DEBLEVE|DEBLEVZERO)$")
THREAD_MACRO = re.compile(r"^(_BEGIN_MASTER|_END_MASTER|_BARRIER|_BEGIN_CRIT|_END_CRIT)")
CLOSERS = re.compile(r"^(ENDIF|ENDDO|ELSE|CONTINUE|ENDSELECT|\d+CONTINUE)$")


def is_ad_file(path):
    return bool(AD_FILE.search(path))


@dataclass
class Line:
    kind: str           # blank comment ad_dir dir cpp_cond cpp_inc cpp_def code
    head: int = -1      # index of the statement's initial line (code lines)
    ctx: tuple = ()     # CPP conditions enclosing the line ("!X" = negated / #else branch)
    norm: str = ""      # normalised text used for token comparison
    kw: str = ""        # CPP keyword of a CPP line (if, ifdef, ifndef, else, elif, endif, include, define, ...)
    block: tuple = ()   # cpp_cond lines: (first, last) line index of the #if ... #endif group they belong to


def _strip_inline_comment(code):
    out, q = [], None
    for ch in code:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def normalise_code(code):
    """Upper-case, no blanks (outside quotes)."""
    out, q = [], None
    for ch in code:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            out.append(ch)
        elif not ch.isspace():
            out.append(ch.upper())
    return "".join(out)


_CPP = re.compile(r"^\s*#\s*(\w+)(.*)$")


def _cpp_cond(kw, rest):
    rest = re.sub(r"/\*.*?\*/", "", rest).strip()
    rest = re.sub(r"\s+", " ", rest)
    if kw == "ifdef":
        return rest.split()[0] if rest else ""
    if kw == "ifndef":
        return "!" + (rest.split()[0] if rest else "")
    return rest


def _norm_cond(rest):
    """CPP line text as tokens joined by single blanks (CPP blanks separate tokens, so `#define A B` != `#define AB`);
    comments dropped; `defined (X)` == `defined X`; one pair of parentheses around the whole condition dropped."""
    t = re.sub(r"/\*.*?\*/", "", rest)
    t = re.sub(r"defined\s*\(\s*(\w+)\s*\)", r"defined \1", t)
    toks = re.findall(r"\"[^\"]*\"|'[^']*'|\w+|&&|\|\||==|!=|<=|>=|\S", t)
    if toks[:1] == ["("] and toks[-1:] == [")"]:
        depth = 0
        for k, tok in enumerate(toks):
            depth += tok == "("
            depth -= tok == ")"
            if depth == 0 and k < len(toks) - 1:
                break
        else:
            toks = toks[1:-1]
    return " ".join(toks)


def analyse(text):
    """Per-line analysis of a fixed-form Fortran/CPP file."""
    lines = text.splitlines()
    info, stack, last_head, open_ = [], [], -1, []
    for idx, raw in enumerate(lines):
        s = raw.rstrip()
        m = _CPP.match(s)
        if not s.strip():
            info.append(Line("blank", ctx=tuple(stack)))
            continue
        if m:
            kw, rest = m.group(1), m.group(2)
            norm = "#" + kw + " " + _norm_cond(rest)
            if kw in ("if", "ifdef", "ifndef"):
                info.append(Line("cpp_cond", ctx=tuple(stack), norm=norm, kw=kw))
                stack.append(_cpp_cond(kw, rest))
                open_.append([idx])
            elif kw in ("else", "elif"):
                outer = tuple(stack[:-1])
                info.append(Line("cpp_cond", ctx=outer, norm=norm, kw=kw))
                if open_:
                    open_[-1].append(idx)
                if stack:
                    prev = stack[-1]
                    neg = prev[1:] if prev.startswith("!") else "!(" + prev + ")"
                    stack[-1] = neg + (" & " + _cpp_cond("if", rest) if kw == "elif" else "")
            elif kw == "endif":
                if stack:
                    stack.pop()
                info.append(Line("cpp_cond", ctx=tuple(stack), norm=norm, kw=kw))
                if open_:
                    grp = open_.pop() + [idx]
                    for k in grp:
                        info[k].block = (grp[0], grp[-1])
            elif kw == "include":
                info.append(Line("cpp_inc", ctx=tuple(stack), norm=norm, kw=kw))
            else:
                info.append(Line("cpp_def", ctx=tuple(stack), norm=norm, kw=kw))
            continue
        c0 = s[0]
        upper = s.lstrip().upper()
        if c0 in "Cc*!" or s.lstrip().startswith("!"):
            if re.match(r"^(CADJ|C\$TAF|!\$TAF|C\$OPENAD|!\$OPENAD|CTAF)", upper):
                info.append(Line("ad_dir", ctx=tuple(stack), norm=normalise_code(s)))
            elif re.match(r"^(C\$OMP|!\$OMP|CDIR|!CDIR|C\$DIR)", upper):
                info.append(Line("dir", ctx=tuple(stack)))
            else:
                info.append(Line("comment", ctx=tuple(stack)))
            continue
        body = s[6:] if len(s) > 6 else ""
        label = s[:5].strip()
        cont = len(s) > 5 and s[5] not in " 0" and not s[:5].strip()
        if "\t" in s[:6]:          # tab-format line: no continuation column
            body, cont, label = s.lstrip(), False, ""
        body = _strip_inline_comment(body)
        norm = normalise_code(label + body)
        if cont and last_head >= 0:
            info.append(Line("code", head=last_head, ctx=info[last_head].ctx, norm=norm))
        else:
            info.append(Line("code", head=idx, ctx=tuple(stack), norm=norm))
            last_head = idx
    return lines, info


def statement(lines, info, head):
    """Normalised text of the statement starting at `head` (initial + continuation lines)."""
    parts = [info[head].norm]
    for j in range(head + 1, len(info)):
        if info[j].kind == "code":
            if info[j].head != head:
                break
            parts.append(info[j].norm)
        elif info[j].kind in ("cpp_cond", "cpp_inc", "cpp_def"):
            continue
    return "".join(parts)


def _defined_names(cond):
    return re.findall(r"[A-Za-z_]\w*", cond.replace("defined", " "))


def ad_positive(cond):
    """True when code under this #if condition is compiled only in AD builds."""
    c = cond.strip()
    if not c or c.startswith("!"):
        return False
    if "||" in c:
        return all(ad_positive(p.strip(" ()")) for p in c.split("||"))
    for part in re.split(r"&&|&", c):
        p = part.strip().strip("()").strip()
        if p.startswith("!"):
            continue
        names = _defined_names(p)
        if len(names) == 1 and AD_MACRO.match(names[0]):
            return True
    return False


def ad_context(ctx):
    return any(ad_positive(c) for c in ctx)


def _ad_normalised(norm):
    return re.sub(r"[A-Z_][A-Z0-9_]*", lambda m: "AD" if AD_MACRO.match(m.group(0)) else m.group(0), norm)


def _logical_if_tail(stmt):
    """For `IF(cond)stmt` return (cond, stmt); for block IF `IF(cond)THEN` return (cond, 'THEN'); else None."""
    if not (stmt.startswith("IF(") or stmt.startswith("ELSEIF(")):
        return None
    start = stmt.index("(")
    depth = 0
    for k in range(start, len(stmt)):
        if stmt[k] == "(":
            depth += 1
        elif stmt[k] == ")":
            depth -= 1
            if depth == 0:
                return stmt[start + 1:k], stmt[k + 1:]
    return None


def _debug_only(cond):
    toks = re.findall(r"\.\w+\.|'[^']*'|[A-Z_][A-Z0-9_]*|\d+|\S", cond)
    return bool(toks) and all(DEBUG_COND.match(t) for t in toks) and \
        any(re.match(r"DEBUG|DEBLEV|PRINTRESIDUALFREQ|USEDIAGNOSTICS|DIAGNOSTICS_IS_ON", t) for t in toks)


def _decl_names(stmt):
    """{name: type} declared by a type statement (normalised text)."""
    m = DECL.match(stmt)
    if not m or stmt.startswith(("COMMON/", "EXTERNAL", "IMPLICITNONE", "INTRINSIC")):
        return {}
    typ = m.group(1)
    rest = stmt[len(typ):]
    base = "REAL8" if typ in ("_RL", "_RS", "_R8", "REAL*8", "DOUBLEPRECISION", "_RX") else typ
    names, depth, cur = {}, 0, ""
    for ch in rest + ",":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            nm = re.match(r"[A-Z_][A-Z0-9_]*", cur)
            if nm:
                names[nm.group(0)] = base
            cur = ""
            continue
        if depth == 0 and ch not in "()":
            cur += ch
    return names


@dataclass
class LineVerdict:
    cls: str
    certain: bool
    rule: str


def assignment_target(stmt):
    """Name assigned by an assignment statement `name[(...)] = ...` (normalised text), else None."""
    m = re.match(r"[A-Z_][A-Z0-9_]*", stmt)
    if not m:
        return None
    k = m.end()
    if k < len(stmt) and stmt[k] == "(":
        depth = 0
        for j in range(k, len(stmt)):
            if stmt[j] == "(":
                depth += 1
            elif stmt[j] == ")":
                depth -= 1
                if depth == 0:
                    k = j + 1
                    break
        else:
            return None
        if k < len(stmt) and stmt[k] == "(":          # substring of a CHARACTER array element: a(i)(1:3) = ...
            return assignment_target(m.group(0) + stmt[k:]) and m.group(0)
    if stmt[k:k + 1] == "=" and stmt[k + 1:k + 2] != "=":
        return m.group(0)
    return None


DIAG_TARGET = {"MSGBUF", "ERRCOUNT", "ERRMSG", "ERRORMESSAGE"}


def classify_statement(stmt, path):
    """(class or None, certain, rule) of one executable or declaration statement (normalised text)."""
    if THREAD_MACRO.match(stmt):
        return COS, True, "thread macro"
    target = assignment_target(stmt)
    if target is not None:
        if target in DIAG_TARGET:
            return DIAG, True, "message/error-count assignment"
        if IO_FILE.search(path):
            return IO, True, "I/O file"
        if DIAG_FILE.search(path):
            return DIAG, True, "message-printing file"
        return FWD, False, "executable statement"
    if DECL.match(stmt):
        return COS, False, "declaration"
    if stmt.startswith(("PARAMETER(", "DATA")):
        return FWD, False, "PARAMETER/DATA"
    if stmt.startswith(("EQUIVALENCE", "SAVE")):
        return None, False, "EQUIVALENCE/SAVE"
    if DIAG_STMT.match(stmt) or DIAG_CALL.match(stmt):
        return DIAG, True, "diagnostics statement"
    if IO_STMT.match(stmt) or IO_CALL.match(stmt):
        return IO, True, "I/O statement"
    li = _logical_if_tail(stmt)
    if li:
        cond, tail = li
        if _debug_only(cond):
            return DIAG, True, "debug-only IF"
        if tail and tail != "THEN":
            c, cert, rule = classify_statement(tail, path)
            if c in (DIAG, IO) and cert:
                return c, True, rule + " (logical IF)"
    if IO_FILE.search(path):
        return IO, True, "I/O file"
    if DIAG_FILE.search(path):
        return DIAG, True, "message-printing file"
    return FWD, False, "executable statement"


@dataclass
class Hunk:
    path: str
    n: int
    header: str
    body: list
    old_start: int
    new_start: int
    cls: str = ""
    how: str = "rule"
    floor: str = COS
    rule_cls: str = ""
    ctx: str = ""
    summary: str = ""
    verdicts: list = field(default_factory=list)

    @property
    def hid(self):
        return f"{self.path}#{self.n}"

    @property
    def sha(self):
        return hashlib.sha1("\n".join(self.body).encode("latin-1", "replace")).hexdigest()[:10]


_HDR = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_hunks(diff_text, path):
    hunks, cur = [], None
    for line in diff_text.splitlines():
        m = _HDR.match(line)
        if m:
            cur = Hunk(path, len(hunks) + 1, line.split(" @@")[0] + " @@", [], int(m.group(1)), int(m.group(3)))
            hunks.append(cur)
        elif cur is not None and line[:1] in (" ", "-", "+"):
            cur.body.append(line)
        elif cur is not None and line.startswith("\\"):
            continue
    return hunks


def _blocks(h):
    """Contiguous runs of changed lines: [(old indices, new indices)] (0-based file line indices)."""
    o, n = h.old_start - 1, h.new_start - 1
    if h.header.split()[1].endswith(",0"):
        o += 1
    if h.header.split()[2].endswith(",0"):
        n += 1
    out, cur = [], None
    for b in h.body:
        t = b[0]
        if t == " ":
            if cur:
                out.append(cur)
                cur = None
            o += 1
            n += 1
        else:
            if cur is None:
                cur = ([], [])
            if t == "-":
                cur[0].append(o)
                o += 1
            else:
                cur[1].append(n)
                n += 1
    if cur:
        out.append(cur)
    return out


def _code_stream(lines, info, idxs):
    return "".join(info[i].norm for i in idxs if info[i].kind in ("code", "cpp_cond", "cpp_inc", "cpp_def", "ad_dir"))


def _cpp_balanced(info, idxs):
    """Indices of cpp_cond lines that open+close a block made only of changed lines (idxs, one side)."""
    idxset, ok, stack = set(idxs), set(), []
    for i in sorted(idxs):
        if info[i].kind != "cpp_cond":
            continue
        kw = info[i].kw
        if kw in ("if", "ifdef", "ifndef"):
            stack.append([i])
        elif kw in ("else", "elif"):
            if stack:
                stack[-1].append(i)
        elif kw == "endif" and stack:
            grp = stack.pop() + [i]
            if all(k in idxset for k in range(grp[0], grp[-1] + 1)):
                ok.update(grp)
    return ok


GUARD = re.compile(r"^_?[A-Z0-9_]+_H_?$")


def _guard_only(L):
    """#ifndef X_H / #define X_H / #endif of an include guard (names ending in _H)."""
    names = re.findall(r"[A-Za-z_]\w*", re.sub(r"^#(ifndef|ifdef|if|define|undef|endif|else|elif) ?", "", L.norm)
                       .replace("defined", " "))
    return bool(names) and all(GUARD.match(n.upper()) for n in names)


def ad_negative(cond):
    """True when code under this #if condition is compiled only in builds WITHOUT AD (#ifndef ALLOW_AUTODIFF ...)."""
    c = cond.strip()
    if c.startswith("!") and not c.startswith("!("):
        return bool(AD_MACRO.match(c[1:].strip()))
    m = re.fullmatch(r"!\s*defined\s*\(?\s*(\w+)\s*\)?", c)
    return bool(m and AD_MACRO.match(m.group(1)))


def _wrapped_verdict(lines, info, i, path, changed=()):
    """Verdict of a CPP conditional line that opens, closes or changes a block around unchanged code. Adding or
    removing `#ifdef X` around a statement adds it to or removes it from the builds without X, so the line takes the
    most severe class of the statements in its #if ... #endif group, judged WITHOUT the group's own condition. A group
    whose condition is `#ifndef <AD macro>` only changes AD builds: AD-only. Changed lines inside the group (`changed`)
    are classified on their own and skipped here."""
    if not info[i].block:
        return None        # unmatched #if/#endif: cannot tell what it wraps
    first, last = info[i].block
    depth = len(info[first].ctx)
    own = _cpp_cond(info[first].kw, info[first].norm.split(" ", 1)[1] if " " in info[first].norm else "")
    if ad_negative(own):
        return LineVerdict(AD, False, "CPP structure change: code dropped/added in AD builds only (#ifndef AD)")
    best = None
    for k in range(first + 1, last):
        L = info[k]
        if k in changed:
            continue
        if L.kind == "ad_dir":
            best = AD if best is None or SEVERITY[AD] > SEVERITY[best] else best
            continue
        if L.kind != "code" or L.head != k:
            continue
        ctx = L.ctx[:depth] + L.ctx[depth + 1:]
        if ad_context(ctx):
            c = AD
        else:
            c, _cert, _rule = classify_statement(statement(lines, info, k), path)
            if c is None:
                return None
        if best is None or SEVERITY[c] > SEVERITY[best]:
            best = c
    return LineVerdict(best or COS, False, "CPP structure change (class of the wrapped statements)")


def _retyped(old, new, olds, news):
    """Names declared with different base types on the two sides of a block (_RL/_RS/_R8/REAL*8 count as one type:
    _RS is real*8 in every oracle build, CPP_EEMACROS.h without -use_real4)."""
    types = []
    for (lines, info), idxs in ((old, olds), (new, news)):
        t = {}
        for i in idxs:
            if info[i].kind == "code" and info[i].head == i:
                t.update(_decl_names(statement(lines, info, i)))
        types.append(t)
    return {n for n in types[0].keys() & types[1].keys() if types[0][n] != types[1][n]}


def classify_hunk(h, old, new):
    """Rule classification of one hunk. old/new = (lines, info) of the two file versions (None if absent)."""
    verdicts = []        # (side, idx, LineVerdict or None)
    ad_by_file = is_ad_file(h.path)
    whole_equal = True
    sides = {"-": old, "+": new}
    # whole-hunk token equality: context + changed lines on each side
    o_idx, n_idx = [], []
    o, n = h.old_start - 1 + (1 if h.header.split()[1].endswith(",0") else 0), \
        h.new_start - 1 + (1 if h.header.split()[2].endswith(",0") else 0)
    for b in h.body:
        if b[0] in " -":
            o_idx.append(o)
            o += 1
        if b[0] in " +":
            n_idx.append(n)
            n += 1
    if old and new:
        whole_equal = _code_stream(*old, o_idx) == _code_stream(*new, n_idx)
    else:
        whole_equal = False
    if whole_equal:
        return COS, COS, "token-equal hunk (comments, blanks, layout, case)", "", []

    for olds, news in _blocks(h):
        block_equal = bool(old and new) and _code_stream(*old, olds) == _code_stream(*new, news)
        # pair identical-modulo-comment CPP lines and conditions that differ only in AD macro names
        pairs_cos, pairs_ad = set(), set()
        if old and new:
            ocpp = [i for i in olds if old[1][i].kind in ("cpp_cond", "cpp_inc", "cpp_def")]
            ncpp = [i for i in news if new[1][i].kind in ("cpp_cond", "cpp_inc", "cpp_def")]
            used = set()
            for i in ocpp:
                for j in ncpp:
                    if j in used:
                        continue
                    if old[1][i].norm == new[1][j].norm:
                        pairs_cos.update({("-", i), ("+", j)})
                        used.add(j)
                        break
                    if _ad_normalised(old[1][i].norm) == _ad_normalised(new[1][j].norm) and \
                            old[1][i].kind == "cpp_cond":
                        pairs_ad.update({("-", i), ("+", j)})
                        used.add(j)
                        break
        retyped = _retyped(old, new, olds, news) if (old and new and not block_equal) else set()
        for side, idxs in (("-", olds), ("+", news)):
            if not idxs:
                continue
            lines, info = sides[side]
            balanced = _cpp_balanced(info, idxs)
            for i in idxs:
                L = info[i]
                if block_equal or L.kind in ("blank", "comment"):
                    v = LineVerdict(COS, True, "comment/blank" if not block_equal else "token-equal block")
                elif L.kind == "dir":
                    v = LineVerdict(COS, True, "NEC/OpenMP directive")
                elif L.kind == "ad_dir":
                    v = LineVerdict(AD, True, "CADJ/TAF directive")
                elif ad_by_file:
                    v = LineVerdict(AD, True, "AD-only file")
                elif L.kind in ("cpp_cond", "cpp_def") and _guard_only(L):
                    v = LineVerdict(COS, True, "include guard")
                elif (side, i) in pairs_cos:
                    v = LineVerdict(COS, True, "CPP comment text")
                elif (side, i) in pairs_ad:
                    v = LineVerdict(AD, True, "CPP condition differs only in AD macro names")
                elif ad_context(L.ctx):
                    v = LineVerdict(AD, True, "inside an AD-only #if")
                elif L.kind == "cpp_cond":
                    if i in balanced:
                        v = LineVerdict(COS, False, "balanced new/removed #if block (contents classified)")
                    else:
                        v = _wrapped_verdict(lines, info, i, h.path, set(idxs))
                elif L.kind == "cpp_inc":
                    if re.search(r"OPTIONS\.H|PACKAGES_CONFIG\.H|CPP_EEMACROS\.H", L.norm.upper()):
                        v = LineVerdict(FWD, False, "#include of an options header (may change the compiled code)")
                    else:
                        v = LineVerdict(COS, False, "#include of a plain header")
                elif L.kind == "cpp_def":
                    name = re.sub(r"^#\w+ ", "", L.norm)
                    nm = re.match(r"[A-Za-z_]\w*", name)
                    if nm and AD_MACRO.match(nm.group(0).upper()):
                        v = LineVerdict(AD, True, "#define/#undef of an AD macro")
                    elif IO_FILE.search(h.path):
                        v = LineVerdict(IO, False, "#define/#undef in an I/O package")
                    else:
                        v = LineVerdict(FWD, False, "#define/#undef")
                else:
                    stmt = statement(lines, info, L.head)
                    c, cert, rule = classify_statement(stmt, h.path)
                    v = LineVerdict(c, cert, rule) if c else None
                    if v is not None and rule == "declaration" and set(_decl_names(stmt)) & retyped:
                        v = None   # a variable changed its type (e.g. INTEGER -> _RL): decide by hand
                verdicts.append((side, i, v))

    # closers (ENDIF/ENDDO/ELSE/CONTINUE) take the class of the other changed code on their side
    for side in "-+":
        lines_info = sides[side]
        if not lines_info:
            continue
        lines, info = lines_info
        mine = [(k, s, i, v) for k, (s, i, v) in enumerate(verdicts) if s == side and v is not None
                and info[i].kind == "code"]
        closers = [(k, s, i, v) for (k, s, i, v) in mine if v.rule == "executable statement"
                   and CLOSERS.match(statement(lines, info, info[i].head))]
        others = [v for (k, s, i, v) in mine if (k, s, i, v) not in closers]
        if closers and others:
            top = max(others, key=lambda v: SEVERITY[v.cls])
            for k, s, i, v in closers:
                verdicts[k] = (s, i, LineVerdict(top.cls, False, "block closer"))

    decided = [v for _, _, v in verdicts if v is not None]
    undecided = [1 for _, _, v in verdicts if v is None]
    floor = max((v.cls for v in decided if v.certain), key=lambda c: SEVERITY[c], default=COS)
    top = max((v.cls for v in decided), key=lambda c: SEVERITY[c], default=COS)
    if undecided and top != FWD:
        cls = UNDECIDED
    else:
        cls = top
    rules = sorted({v.rule for v in decided if v.cls == cls}) if cls != UNDECIDED else ["undecided line(s)"]
    ctxs = sorted({" & ".join(sides[s][1][i].ctx) for s, i, v in verdicts
                   if v is not None and v.cls == cls and sides[s][1][i].ctx and cls != COS})
    return cls, floor, "; ".join(rules), " / ".join(ctxs), verdicts


def _summary(h, old, new, verdicts):
    """First changed line with the hunk's class (code before comments), trimmed."""
    best = None
    for s, i, v in verdicts:
        lines = (old if s == "-" else new)[0]
        txt = lines[i].strip()
        if v is None or v.cls == h.cls:
            best = s + " " + txt
            break
    if best is None:
        for b in h.body:
            if b[0] in "-+" and b[1:].strip():
                best = b[0] + " " + b[1:].strip()
                break
    best = (best or "").replace("`", "'")
    return best if len(best) <= 70 else best[:67] + "..."


# --------------------------------------------------------------------------------------------------------------------
# Files and routines
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class FileAudit:
    old_path: str
    new_path: str
    status: str
    hunks: list
    routines_moved: list = field(default_factory=list)
    modules: list = field(default_factory=list)
    where: str = ""

    @property
    def path(self):
        return self.new_path or self.old_path

    @property
    def label(self):
        if self.old_path and self.new_path and self.old_path != self.new_path:
            return f"{self.old_path} -> {self.new_path}"
        return self.path

    def counts(self):
        out = {c: 0 for c in CLASSES + (UNDECIDED,)}
        for h in self.hunks:
            out[h.cls] += 1
        return out


def _apply_overrides(fa, overrides, problems):
    for h in fa.hunks:
        key = (h.path, h.n)
        if key not in overrides:
            continue
        cls, sha, reason = overrides[key]
        if cls not in CLASSES:
            problems.append(f"override {h.hid}: unknown class {cls!r}")
        elif sha != h.sha:
            problems.append(f"override {h.hid}: stale content sha {sha} (hunk is now {h.sha})")
        elif SEVERITY[cls] < SEVERITY[h.floor]:
            problems.append(f"override {h.hid}: {cls} is below the rules' floor {h.floor}")
        elif cls == h.rule_cls:
            problems.append(f"override {h.hid}: repeats the rules' class {cls}")
        else:
            h.cls, h.how = cls, f"override: {reason}"


def audit_pair(up, old_path, new_path, overrides=None, problems=None):
    """FileAudit of old_path at OLD vs new_path at NEW (either may be None for added/removed files)."""
    overrides = OVERRIDES if overrides is None else overrides
    problems = [] if problems is None else problems
    ot = up.text(OLD, old_path) if old_path else None
    nt = up.text(NEW, new_path) if new_path else None
    if ot is None and nt is None:
        problems.append(f"{old_path or new_path}: absent at both tags")
        return FileAudit(old_path, new_path, "missing", [])
    if ot is None:
        status, old_path = "added", None
    elif nt is None:
        status, new_path = "removed", None
    elif ot == nt:
        status = "identical"
    else:
        status = "modified" if old_path == new_path else "moved"
    path = new_path or old_path
    if status == "identical":
        hunks = []
    elif status in ("added", "removed"):
        text = nt if status == "added" else ot
        hunks = parse_hunks(_diff_texts("" if status == "added" else text, text if status == "added" else ""), path)
    else:
        hunks = parse_hunks(up.diff(old_path, new_path), path)
    oldA = analyse(ot) if ot is not None else None
    newA = analyse(nt) if nt is not None else None
    for h in hunks:
        cls, floor, rule, ctx, verdicts = classify_hunk(h, oldA, newA)
        h.cls, h.rule_cls, h.floor, h.how, h.ctx = cls, cls, floor, "rule: " + rule, ctx
        h.verdicts = verdicts
        h.summary = _summary(h, oldA or newA, newA or oldA, verdicts) if verdicts else _first_change(h)
    fa = FileAudit(old_path, new_path, status, hunks)
    _apply_overrides(fa, overrides, problems)
    return fa


def _first_change(h):
    for b in h.body:
        if b[0] in "-+" and b[1:].strip():
            s = (b[0] + " " + b[1:].strip()).replace("`", "'")
            return s if len(s) <= 70 else s[:67] + "..."
    return ""


def _diff_texts(old_text, new_text):
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a", Path(d) / "b"
        a.write_text(old_text, encoding="latin-1")
        b.write_text(new_text, encoding="latin-1")
        r = subprocess.run(["git", *GIT_DIFF_CONFIG, "diff", "--no-index", "--no-color", "--no-ext-diff",
                            "--no-textconv", "--unified=3", str(a), str(b)], capture_output=True, text=True,
                           errors="replace")
        return r.stdout


def classify_texts(old_text, new_text, path="model/src/x.F"):
    """Rule classification (no overrides) of the hunks between two in-memory versions of a file."""
    hunks = parse_hunks(_diff_texts(old_text, new_text), path)
    oldA, newA = analyse(old_text), analyse(new_text)
    for h in hunks:
        cls, floor, rule, ctx, verdicts = classify_hunk(h, oldA, newA)
        h.cls, h.rule_cls, h.floor, h.how, h.ctx, h.verdicts = cls, cls, floor, "rule: " + rule, ctx, verdicts
        h.summary = _summary(h, oldA, newA, verdicts)
    return hunks


def resolve(up, item):
    """[(old_path, new_path, note)] for a repository path, an (old, new) pair, or a routine name."""
    if isinstance(item, tuple):
        return [(item[0], item[1], "renamed")]
    if "/" in item or re.search(r"\.(F|h|template|flow)$", item):
        return [(item, item, "")]
    o, n = up.define_sites(OLD, item), up.define_sites(NEW, item)
    if not o and not n:
        raise SystemExit(f"routine {item}: no SUBROUTINE/FUNCTION definition at either tag")
    if o == n:
        return [(p, p, "") for p in o]
    if len(o) == 1 and len(n) == 1:
        return [(o[0], n[0], f"{item} moved")]
    out = [(p, p, "") for p in sorted(set(o) & set(n))]
    out += [(p, None, f"{item} not here at {NEW}") for p in sorted(set(o) - set(n))]
    out += [(None, p, f"{item} not here at {OLD}") for p in sorted(set(n) - set(o))]
    return out


def unmatched_overrides(audits, overrides, require_all=False):
    """Problems for override keys that name no hunk of an audited file (or, with require_all, no audited file)."""
    have = {(h.path, h.n) for fa in audits for h in fa.hunks}
    paths = {fa.path for fa in audits}
    out = []
    for key in sorted(overrides):
        if key in have:
            continue
        if key[0] in paths:
            out.append(f"override {key[0]}#{key[1]}: no such hunk")
        elif require_all:
            out.append(f"override {key[0]}#{key[1]}: file is not in MODULES")
    return out


def audit_items(items, upstream=None, overrides=None, problems=None):
    """Audit routine names and/or repository paths; returns [FileAudit] in input order, each file once."""
    up = upstream if isinstance(upstream, Upstream) else Upstream(upstream)
    overrides = OVERRIDES if overrides is None else overrides
    problems = [] if problems is None else problems
    seen, out = set(), []
    for item in items:
        for old_path, new_path, note in resolve(up, item):
            key = (old_path, new_path)
            if key in seen:
                continue
            seen.add(key)
            fa = audit_pair(up, old_path, new_path, overrides, problems)
            if note:
                fa.routines_moved.append(note)
            out.append(fa)
    problems.extend(unmatched_overrides(out, overrides))
    return out


def module_files(modules=None):
    """Unique file entries of MODULES in first-appearance order, and {entry: [module keys]}."""
    order, member = [], {}
    for key, _t, _r, _m, files, _s in (MODULES if modules is None else modules):
        for f in files:
            if f not in member:
                order.append(f)
                member[f] = []
            if key not in member[f]:
                member[f].append(key)
    return order, member


def routines_in(text):
    return sorted({m.group(2).upper() for m in re.finditer(
        r"^ {6,}\s*(SUBROUTINE|[A-Za-z_*0-9 ]*FUNCTION)\s+([A-Za-z_][A-Za-z0-9_]*)", text or "", re.M | re.I)})


def whereabouts(up, fa):
    """For an added/removed file: where its routines are defined at the other tag."""
    if fa.status not in ("added", "removed", "moved"):
        return ""
    here, there = (NEW, OLD) if fa.status == "added" else (OLD, NEW)
    path = fa.old_path if here == OLD else fa.new_path
    names = routines_in(up.text(here, path))
    notes = []
    for nm in names:
        sites = [s for s in up.define_sites(there, nm) if s != path]
        notes.append(f"{nm}: " + (", ".join(sites) if sites else f"absent at {there}"))
    return "; ".join(notes) if notes else "no routines (header or flow file)"


# --------------------------------------------------------------------------------------------------------------------
# Rendering and check
# --------------------------------------------------------------------------------------------------------------------


def _md(s):
    return s.replace("|", "\\|")


def run_audit(upstream=None, overrides=None, modules=None):
    """(Upstream, [FileAudit] of every MODULES file, problems)."""
    up = upstream if isinstance(upstream, Upstream) else Upstream(upstream)
    overrides = OVERRIDES if overrides is None else overrides
    problems = []
    order, member = module_files(modules)
    up.prefetch(sorted({p for e in order for p in (e if isinstance(e, tuple) else (e,))}))
    up.prefetch_diffs(sorted({e for e in order if isinstance(e, str)}))
    audits = []
    for entry in order:
        old_path, new_path = entry if isinstance(entry, tuple) else (entry, entry)
        fa = audit_pair(up, old_path, new_path, overrides, problems)
        fa.modules = member[entry]
        fa.where = whereabouts(up, fa)
        audits.append(fa)
    problems.extend(unmatched_overrides(audits, overrides, require_all=True))
    return up, audits, problems


def render(audits):
    out = [BEGIN, "",
           f"Diff `git diff {OLD} {NEW}` (c66g = `{C66G_SHA[:7]}`, pinned = `{PINNED_SHA[:7]}`), 3 lines of context, "
           "myers. Hunk IDs `path#n` number the hunks of one file in diff order.", "",
           "### Files", "",
           "| file | modules | status | hunks | fwd | I/O | AD | diag | cosm |",
           "|---|---|---|---|---|---|---|---|---|"]
    tot = {c: 0 for c in CLASSES + (UNDECIDED,)}
    for fa in audits:
        c = fa.counts()
        for k in tot:
            tot[k] += c[k]
        und = f" (+{c[UNDECIDED]} undecided)" if c[UNDECIDED] else ""
        out.append(f"| `{fa.label}` | {', '.join(fa.modules)} | {fa.status}{und} | {len(fa.hunks)} | {c[FWD]} | "
                   f"{c[IO]} | {c[AD]} | {c[DIAG]} | {c[COS]} |")
    out.append(f"| **total** | | | {sum(len(f.hunks) for f in audits)} | {tot[FWD]} | {tot[IO]} | {tot[AD]} | "
               f"{tot[DIAG]} | {tot[COS]} |")
    moved = [fa for fa in audits if fa.status in ("added", "removed", "moved")]
    if moved:
        out += ["", "### Files added, removed or renamed between the tags", "",
                "| file | status | its routines at the other tag |", "|---|---|---|"]
        for fa in moved:
            out.append(f"| `{fa.label}` | {fa.status} | {_md(fa.where) if fa.where else '—'} |")
    out += ["", "### Hunks that are not cosmetic", "",
            "| hunk | lines c66g → master | class | decided by | CPP context | first changed line |",
            "|---|---|---|---|---|---|"]
    for fa in audits:
        for h in fa.hunks:
            if h.cls == COS:
                continue
            hdr = h.header[3:-3]
            out.append(f"| `{h.hid}` | {hdr} | {h.cls} | {_md(h.how)} | {_md(h.ctx) if h.ctx else ''} | "
                       f"`{_md(h.summary)}` |")
    out += ["", "### Cosmetic hunks", ""]
    for fa in audits:
        cos = [str(h.n) for h in fa.hunks if h.cls == COS]
        if cos:
            out.append(f"- `{fa.path}`: " + ", ".join(cos))
    out += ["", END]
    return "\n".join(out)


def _section(doc_text):
    if BEGIN not in doc_text or END not in doc_text:
        return None
    return doc_text[doc_text.index(BEGIN):doc_text.index(END) + len(END)]


_REF = re.compile(r"([\w./-]+\.(?:F|h|template|flow))((?:#\d+(?:-\d+)?)(?:,\s*#\d+(?:-\d+)?)*)")
_NUM = re.compile(r"#(\d+)(?:-(\d+))?")


def referenced(hand_text, audits):
    """Hunk IDs referenced in hand-written text: `path#n`, `name.F#n` (basename if unique), ranges `name.F#n-m` and
    lists `name.F#1, #4-6, #9`."""
    by_base = {}
    for fa in audits:
        by_base.setdefault(Path(fa.path).name, set()).add(fa.path)
    refs = set()
    for m in _REF.finditer(hand_text):
        name = m.group(1)
        paths = {name} if "/" in name else by_base.get(name, set())
        if len(paths) != 1:
            continue
        p = next(iter(paths))
        for a, b in _NUM.findall(m.group(2)):
            refs.update((p, k) for k in range(int(a), int(b or a) + 1))
    return refs


def check(doc_text, audits, problems=()):
    """Problems that make --check fail (empty list = the document is complete and current)."""
    out = list(problems)
    cur = _section(doc_text)
    if cur is None:
        out.append("generated-section markers missing")
    elif cur != render(audits):
        out.append("generated table is stale: regenerate with scripts/audit_upstream.py --update")
    for fa in audits:
        for h in fa.hunks:
            if h.cls == UNDECIDED:
                out.append(f"{h.hid}: undecided by the rules and no override")
    hand = doc_text.replace(cur, "") if cur else doc_text
    refs = referenced(hand, audits)
    missing = [h.hid for fa in audits for h in fa.hunks if h.cls == FWD and (h.path, h.n) not in refs]
    if missing:
        out.append(f"{len(missing)} forward-value hunks have no hand-written note: " + ", ".join(missing))
    return out


def update(doc_text, audits):
    cur = _section(doc_text)
    if cur is None:
        raise SystemExit("generated-section markers missing")
    return doc_text.replace(cur, render(audits))


def print_items(audits):
    print("| hunk | lines c66g → master | class | decided by | first changed line |")
    print("|---|---|---|---|---|")
    for fa in audits:
        if not fa.hunks:
            print(f"| `{fa.label}` | — | {fa.status} | | {'; '.join(fa.routines_moved)} |")
        for h in fa.hunks:
            print(f"| `{h.hid}` | {h.header[3:-3]} | {h.cls} | {_md(h.how)} | `{_md(h.summary)}` |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", metavar="FILE")
    g.add_argument("--update", metavar="FILE")
    g.add_argument("--routines", nargs="+", metavar="NAME")
    g.add_argument("--files", nargs="+", metavar="PATH")
    g.add_argument("--show", nargs="+", metavar="PATH#N", help="print hunks (body, class, line verdicts)")
    ap.add_argument("--upstream", metavar="DIR")
    args = ap.parse_args(argv)
    if args.show:
        problems = []
        wanted = [x.rsplit("#", 1) for x in args.show]
        audits = audit_items([w[0] for w in wanted], args.upstream, problems=problems)
        for path, n in wanted:
            for fa in audits:
                for h in fa.hunks:
                    if h.path == path and str(h.n) == n:
                        print(f"== {h.hid} {h.header} {h.cls} [{h.how}] floor={h.floor} sha={h.sha}")
                        print("\n".join(h.body))
        return 1 if problems else 0
    if args.routines or args.files:
        problems = []
        audits = audit_items(args.routines or args.files, args.upstream, problems=problems)
        print_items(audits)
        for p in problems:
            print("FAIL:", p)
        und = sum(fa.counts()[UNDECIDED] for fa in audits)
        if und:
            print(f"{und} undecided hunks: add them to OVERRIDES in scripts/audit_upstream.py")
        return 1 if problems else (2 if und else 0)
    _up, audits, problems = run_audit(args.upstream)
    if args.update:
        path = Path(args.update)
        path.write_text(update(path.read_text(), audits))
        print(f"updated {path}")
        for p in problems:
            print("FAIL:", p)
        return 1 if problems else 0
    if not args.check:
        print(render(audits))
        for p in problems:
            print("FAIL:", p)
        return 1 if problems else 0
    probs = check(Path(args.check).read_text(), audits, problems)
    for p in probs:
        print("FAIL:", p)
    if not probs:
        n = sum(len(fa.hunks) for fa in audits)
        nf = sum(fa.counts()[FWD] for fa in audits)
        print(f"OK: {len(audits)} files, {n} hunks, {nf} forward-value hunks, all classified and noted")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
