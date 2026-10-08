"""Make a run directory for one MITgcm verification variant exactly as testreport does (plan Task 3a).

    python -m mitjax.make_rundir EXPERIMENT INPUT_DIR [--run-id ID] [--binary NAME_OR_DIR]
    python reference/make_rundir.py ...        (the same; a shim kept at the old path)

Part of the package since docs plan 20261006 Task 5 (R4): mitjax/config/namelists.py and mitjax/drivers/run.py use
it at run time. Stdlib only (loadable by path without jax).

INPUT_DIR is the variant's input directory: `input`, `input.<v>` (e.g. input.nlfs), `input_ad` or `input_ad.<v>`, or
`input_min` (adjustment.cs-32x32x1's "minimal" case for code_min, README of that experiment; testreport does not run
it, so it is linked like input_ad: its own directory only, run directory `run`).
The run goes to $MJX_REFERENCE_RUNS/<EXPERIMENT>/<INPUT_DIR>/<ID> (mitjax/paths.py); ID defaults to the UTC time
plus the process id; an existing directory is refused (nothing is overwritten or deleted).

What testreport does (verification/testreport at 63cdc0b) and how it is reproduced here:
  * Run directory: `run` for the main input directory, `tr_run.<v>` for a variant (testreport:1606-1607, 1793),
    inside verification/<experiment>/. prepare_run scripts reach other experiments by relative paths
    (tutorial_global_oce_optim/input_ad/prepare_run links ../../tutorial_global_oce_latlon/input/*.bin and
    ../../isomip/input_ad/ones_64b.bin), so <ID>/verification/ mirrors upstream's verification/: every other
    experiment is a symlink to the read-only clone, verification/<experiment>/ is a real directory holding links
    to the input directories used and the run directory itself. <ID>/rundir links to the run directory.
  * linkdata (testreport:771-834), with testreport's defaults MPI=0 and MULTI_THREAD=f (testreport:1133-1135):
    `linkdata run input` (testreport:1771) or `linkdata tr_run.<v> input.<v> input` (testreport:1800) — the
    same for input_ad (testreport:1323-1332). For each directory in that order (a repeat of the previous one is
    skipped), every entry of `ls -1 | grep -v CVS` (no dot files; any name containing "CVS" skipped) that is not a
    directory and not already readable in the run directory is linked as `ln -sf ../<dir>/<name> <name>` (so the
    variant's files shadow input/'s). With MPI=0 the *.mpi and eedata.mth branches only remove links left by an
    earlier MPI or multi-threaded run, which a new directory cannot have; `*.mpi` files and eedata.mth are linked
    under their own names by the generic loop, as testreport does.
  * then `./prepare_run` if it is executable (testreport:829-831), its output kept in <ID>/prepare_run.log; an
    `Error:` line in its output refuses the directory (the scripts print it and exit 0 when a source is missing).
    After prepare_run every linkdata entry must still be there with the same link target (or still the same regular
    file): `removed_by_prepare_run(before, after)` on the two listings, else the directory is refused.
    The experiment's other directories are linked into the mirror as well (prepare_run may read them).
    Nothing is ever removed (project rule): a prepare_run with an `rm`, `mv` or `unlink` command is refused before
    it runs; one that calls `gunzip` runs with a `gunzip` shim first on PATH (<ID>/prepare_run_shims/gunzip =
    the system gunzip with `--keep`), so the decompressed file is the same while the `.gz` link linkdata made stays
    (global_ocean.cs32x15/input.viscA4/prepare_run: `gunzip -f ${xx}.gz` would unlink that link); recorded in
    MANIFEST.json ("prepare_run": {"shims": [...]}). After prepare_run every entry linkdata made must still
    exist, else the directory is refused.
  * Required inputs (a check testreport does not make; the model would stop or, worse, read zeros): every
    non-blank quoted value of a key ending in `File` in `data`, of a key containing `_weight` in `data.ctrl`, and
    with nIter0 > 0 the pickup `pickup.<nIter0 as I10.10>` must exist in the run directory as `<name>` or
    `<name>.data` (the two global names MDS_READ_FIELD tries, pkg/mdsio/mdsio_read_field.F:229-255) or, for the
    pickup only, as per-tile files `<name>.<iG as I3.3>.<jG as I3.3>.data` (its globalFile=F branch,
    mdsio_read_field.F:446-454, what WRITE_PICKUP writes with globalFiles=F; at least one such file: the tile
    numbers are not derived here). This is an existence check on the namelist text, not a namelist reader (that is
    mitjax/config, plan Task 6).
  * --binary: link `mitgcmuv` to $MJX_REFERENCE/bin/<name>/mitgcmuv after checking it against that directory's
    sha256 file (testreport links ../build/mitgcmuv, testreport:861-864).
On success <ID>/MANIFEST.json lists every entry of the run directory with its link target and the directory it
came from, and <ID>/READY is written last; on a refusal <ID>/REFUSED.txt holds the reason (the directory stays,
nothing is deleted) and the exit code is 1. reference/run.sh runs only a READY directory.

Options that leave testreport's defaults (plan Task 3b; each is recorded in MANIFEST.json):
  * --mpi: linkdata's MPI branch (testreport:780-795 with MPI != 0) for a single-process build that uses the
    tiling of SIZE.h_mpi: every `*.mpi` file of the first input directory is linked under its name without `.mpi`
    (e.g. global_ocean.90x40x15/input/data.exch2.mpi -> data.exch2) before the generic loop, which then links the
    `*.mpi` file under its own name as well. The binary stays serial; only the run directory is testreport -mpi's.
  * --set FILE:GROUP:KEY=VALUE (repeatable): a namelist overlay. FILE, which linkdata would link, is written as a
    regular file instead: the linked file's text with `KEY=VALUE,` replacing the one uncommented line of GROUP that
    assigns KEY (a line assigning anything else as well is refused), or inserted after the group's header line when
    GROUP does not assign KEY. Every change is listed in <ID>/OVERLAY.txt (source, old line, new line). A run with an
    overlay is not testreport's run: its output must never be used as a yardstick.

  * --copy SOURCE:NAME (repeatable; NAME may be <dir>/<file>, the directory made if absent): copy the file SOURCE
    into the run directory as the regular file NAME after
    linkdata and prepare_run (refused when NAME exists there). Used for the zero-valued adjoint control files of the
    forward-only tutorial_global_oce_optim/code_ad gradient check (reference/grdchk_adxx.py); listed with their
    sha256 in MANIFEST.json ("copies").

`linkdata_plan(exp_dir, dirs, mpi)` is the pure part of linkdata (the names, link targets and source directories,
no links made), for callers that need the run directory's file set without making one (mitjax/config).
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ == "mitjax":                  # imported as mitjax.make_rundir (mitjax/config/namelists.py, drivers)
    from mitjax import paths
else:                                       # loaded by path (reference/make_rundir.py, environments without jax):
    _spec = importlib.util.spec_from_file_location("_mjx_paths", Path(__file__).resolve().parent / "paths.py")
    paths = importlib.util.module_from_spec(_spec)          # mitjax/paths.py is stdlib only, while `import mitjax`
    _spec.loader.exec_module(paths)                          # imports jax


def input_dirs(input_dir):
    """testreport's linkdata directory list for a variant (testreport:1771, 1800)."""
    if input_dir == "input_min":     # adjustment.cs-32x32x1/README, "minimal" test case (not run by testreport)
        return ["input_min"], "run"
    m = re.fullmatch(r"(input|input_ad)(?:\.([\w+-]+))?", input_dir)
    if not m:
        raise SystemExit(f"bad input directory {input_dir!r}: expected input, input.<v>, input_ad, input_ad.<v> "
                         "or input_min")
    base, variant = m.groups()
    return ([input_dir, base] if variant else [base]), ("tr_run." + variant if variant else "run")


def ls1(directory):
    """`ls -1 | grep -v CVS` of testreport:816: visible entries (no dot files), names containing CVS dropped.
    Order is irrelevant here: no two entries of one directory share a name."""
    return sorted(n for n in os.listdir(directory) if not n.startswith(".") and "CVS" not in n)


def linkdata_plan(exp_dir, dirs, mpi=False):
    """testreport's linkdata (testreport:771-834) for MULTI_THREAD=f, MPI=0 (default) or MPI != 0 (`mpi`), as a plan
    for a new, empty run directory inside `exp_dir` (the directory holding the input directories): the ordered list
    of (name, link target, source directory) the shell function would create. Nothing is linked.

    MPI branch (testreport:780-795): every `*.mpi` file of the first directory (`find . -name "*.mpi"`) that is
    readable is linked as its name without `.mpi`; the removals of links left by an earlier run cannot apply to a
    new directory. Generic loop (testreport:810-828): for each directory in order, a repeat of the previous one
    skipped, every entry of `ls -1 | grep -v CVS` that is not a directory and not readable yet in the run directory
    (here: not planned yet with a readable target) is linked as ../<dir>/<name>."""
    exp_dir = Path(exp_dir)
    plan, readable = [], {}
    if mpi and dirs and (exp_dir / dirs[0]).is_dir():
        first = exp_dir / dirs[0]
        for root, _, files in sorted(os.walk(first)):
            for f in sorted(files):
                if not f.endswith(".mpi"):
                    continue
                rel = (Path(root) / f).relative_to(first).as_posix()
                if "/" in rel:
                    raise SystemExit(f"{first / rel}: a *.mpi file in a subdirectory is not supported")
                if os.access(first / rel, os.R_OK):
                    yy = rel[:-len(".mpi")]
                    if yy in readable:
                        raise SystemExit(f"two *.mpi files map to {yy}")
                    plan.append((yy, f"../{dirs[0]}/{rel}", dirs[0]))
                    readable[yy] = True
    prev = None
    for ldir in dirs:
        src = exp_dir / ldir
        if src.is_dir() and ldir != prev:
            for name in ls1(src):
                if not (src / name).is_dir() and not readable.get(name, False):
                    if name in readable:   # dangling link: `ln -sf` replaces it (cannot occur here)
                        raise SystemExit(f"unexpected dangling link {name}")
                    plan.append((name, f"../{ldir}/{name}", ldir))
                    readable[name] = os.access(src / name, os.R_OK)
        prev = ldir
    return plan


def linkdata(rundir, dirs, mpi=False, overlays=None):
    """Make the links of linkdata_plan in the new run directory `rundir` (its parent holds the input directories);
    returns {name: source dir}. A name in `overlays` ({name: text}) is written as a regular file with that text
    instead of the link."""
    made = {}
    for name, target, ldir in linkdata_plan(rundir.parent, dirs, mpi):
        if overlays and name in overlays:
            (rundir / name).write_text(overlays[name])
        else:
            (rundir / name).symlink_to(target)
        made[name] = ldir
    return made


# --- namelist overlays (--set) --------------------------------------------------------------------------------

def parse_set(spec):
    """'FILE:GROUP:KEY=VALUE' -> (file, group, key, value)."""
    m = re.fullmatch(r"([^:/]+):(\w+):(\w+)=(.+)", spec)
    if not m:
        raise SystemExit(f"bad --set {spec!r}: expected FILE:GROUP:KEY=VALUE")
    return m.groups()


def _commented(line):
    return line.lstrip().startswith(("#", "!"))


def overlay_namelist(text, group, key, value):
    """(new text, old line or None, new line): `KEY=VALUE,` in namelist GROUP of `text` (see the module docstring)."""
    lines = text.split("\n")
    head = [i for i, ln in enumerate(lines) if not _commented(ln) and re.match(rf"\s*&{group}\s*$", ln, re.I)]
    if len(head) != 1:
        raise SystemExit(f"namelist group &{group}: found {len(head)} times")
    end = next((i for i in range(head[0] + 1, len(lines))
                if not _commented(lines[i]) and re.match(r"\s*(&|/)", lines[i])), None)
    if end is None:
        raise SystemExit(f"namelist group &{group} has no terminator")
    new = f" {key}={value},"
    hits = [i for i in range(head[0] + 1, end)
            if not _commented(lines[i]) and re.search(rf"(^|[\s,]){key}\s*(\(|=)", lines[i], re.I)]
    if len(hits) > 1:
        raise SystemExit(f"&{group} assigns {key} on {len(hits)} lines; overlay refused")
    if hits:
        old = lines[hits[0]]
        if len(re.findall(r"[A-Za-z_]\w*\s*(?:\([^)]*\))?\s*=", old)) != 1 or "(" in old.split("=")[0]:
            raise SystemExit(f"&{group} line {old!r} assigns more than {key} (or an element); overlay refused")
        lines[hits[0]] = new
    else:
        old = None
        lines.insert(head[0] + 1, new)
    return "\n".join(lines), old, new


def build_overlays(exp_dir, dirs, mpi, sets):
    """{file name: overlaid text} and the OVERLAY.txt records for the --set specs, from the files linkdata_plan
    links (several --set on one file apply in order)."""
    plan = {n: (t, d) for n, t, d in linkdata_plan(exp_dir, dirs, mpi)}
    texts, records = {}, []
    for spec in sets:
        fname, group, key, value = parse_set(spec)
        if fname not in plan:
            raise SystemExit(f"--set {spec}: {fname} is not linked by linkdata (no file to overlay)")
        src = Path(exp_dir) / plan[fname][0][len("../"):]   # the link target (a *.mpi file under --mpi)
        text = texts.get(fname, src.read_text())
        texts[fname], old, new = overlay_namelist(text, group, key, value)
        records.append({"file": fname, "source": plan[fname][0], "group": group, "key": key, "value": value,
                        "old_line": old, "new_line": new})
    return texts, records


NAMELIST_STRING = re.compile(r"""^\s*([A-Za-z_]\w*(?:\([^)]*\))?)\s*=\s*(['"])(.*?)\2""", re.M)


def strip_comments(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith(("#", "!")))


def required_inputs(rundir):
    """[(file name, reason)] the run's namelists name as inputs (see the module docstring)."""
    need = []
    data = rundir / "data"
    if data.exists():
        text = strip_comments(data.read_text())
        for key, _, val in NAMELIST_STRING.findall(text):
            if re.fullmatch(r"\w*file", key.split("(")[0], re.I) and val.strip():
                need.append((val.strip(), f"data: {key}"))
        m = re.search(r"^\s*nIter0\s*=\s*(\d+)", text, re.M | re.I)
        suff = re.search(r"""^\s*pickupSuff\s*=\s*['"](.*?)['"]""", text, re.M | re.I)
        if m and int(m.group(1)) > 0 and not (suff and suff.group(1).strip()):
            need.append((f"pickup.{int(m.group(1)):010d}", f"data: nIter0={int(m.group(1))}"))
    ctrl = rundir / "data.ctrl"
    if ctrl.exists():
        for key, _, val in NAMELIST_STRING.findall(strip_comments(ctrl.read_text())):
            if "_weight" in key.lower() and val.strip():
                need.append((val.strip(), f"data.ctrl: {key}"))
    return need


TILE_SUFFIX = re.compile(r"\.\d{3}\.\d{3}\.data")
FACE_SUFFIX = re.compile(r"\.face\d{3}\.bin")


def missing_inputs(rundir):
    def found(n, why):
        if any(os.access(rundir / c, os.R_OK) for c in (n, n + ".data")):
            return True
        # the unit weight the model writes itself: CTRL_INIT_FIXED writes 'wunit' (wunit.data/.meta) unconditionally
        # (pkg/ctrl/ctrl_init_fixed.F:92-110) before any control is mapped (global_ocean.90x40x15/input_ad*)
        if why.startswith("data.ctrl:") and n in ("wunit", "wunit.data"):
            return True
        # per-face grid files of a curvilinear (exch2) grid: horizGridFile//'.face'//I3.3//'.bin'
        # (model/src/ini_curvilinear_grid.F:282-283); at least one, the face count is not derived here
        if why == "data: horizGridFile":
            return any(FACE_SUFFIX.fullmatch(f[len(n):]) and os.access(rundir / f, os.R_OK)
                       for f in os.listdir(rundir) if f.startswith(n + "."))
        # per-tile pickup files (mdsio_read_field.F:446-454): only for the pickup, not for the data: *File inputs
        return why.startswith("data: nIter0=") and any(
            TILE_SUFFIX.fullmatch(f[len(n):]) and os.access(rundir / f, os.R_OK)
            for f in os.listdir(rundir) if f.startswith(n + "."))
    return [(n, why) for n, why in required_inputs(rundir) if not found(n, why)]


def listing(rundir, names=None):
    """{name: link target, or None for anything that is not a symlink} of the run directory's entries (dangling links
    included), restricted to `names` when given (a name not present is left out)."""
    present = set(os.listdir(rundir))
    return {n: (os.readlink(rundir / n) if (rundir / n).is_symlink() else None)
            for n in sorted(present if names is None else present & set(names))}


def removed_by_prepare_run(before, after):
    """Pure: the linkdata entries (`before`: {name: link target or None}, taken before prepare_run ran) that are gone
    or changed in `after` (the same kind of listing taken after it ran): a missing name, a link with another target,
    a link that became a regular file or a regular file that became a link. A replaced link was removed first
    (`ln -sf` unlinks), so a change is a removal too. Entries prepare_run added are not looked at."""
    return sorted(n for n, tgt in before.items() if n not in after or after[n] != tgt)


def removing_commands(script):
    """Commands of a prepare_run script that would remove a file (rm, mv, unlink, rmdir), outside comments."""
    out = []
    for ln in script.split("\n"):
        code = ln.split("#", 1)[0]
        out += re.findall(r"(?:^|[\s;&|(`])(rm|mv|unlink|rmdir)(?=\s)", code)
    return sorted(set(out))


def frozen_binary(spec):
    """Path of a frozen binary: a directory under $MJX_REFERENCE/bin given by name or path, checked against its
    sha256 file."""
    d = Path(spec) if "/" in spec else paths.REFERENCE / "bin" / spec
    exe, sha = d / "mitgcmuv", d / "sha256"
    if not exe.is_file() or not sha.is_file():
        raise SystemExit(f"no frozen binary in {d} (mitgcmuv + sha256 from reference/build.sh)")
    want = sha.read_text().split()[0]
    got = hashlib.sha256(exe.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{exe}: sha256 {got} != {want} recorded in {sha}")
    return exe, want


def entry(path):
    if path.is_symlink():
        tgt = os.readlink(path)
        res = Path(os.path.realpath(path))
        return {"link": tgt, "resolves_to": str(res), "exists": res.exists()}
    return {"file": True, "size": path.stat().st_size}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment")
    ap.add_argument("input_dir")
    ap.add_argument("--run-id")
    ap.add_argument("--binary", help="frozen binary: directory name under $MJX_REFERENCE/bin, or its path")
    ap.add_argument("--upstream", help="MITgcm clone (default $MJX_UPSTREAM)")
    ap.add_argument("--runs", help="run root (default $MJX_REFERENCE_RUNS)")
    ap.add_argument("--mpi", action="store_true", help="linkdata's MPI branch (*.mpi files; serial binary)")
    ap.add_argument("--set", action="append", default=[], metavar="FILE:GROUP:KEY=VALUE",
                    help="namelist overlay (repeatable); the run is then not testreport's run")
    ap.add_argument("--copy", action="append", default=[], metavar="SOURCE:NAME",
                    help="copy SOURCE into the run directory as NAME (repeatable)")
    ap.add_argument("--exp-dir", help="the experiment directory, anywhere on disk (default: <upstream>/verification/"
                    "EXPERIMENT); its parent directory plays the role of verification/ for prepare_run's relative "
                    "paths, as testreport runs inside the experiment directory")
    a = ap.parse_args(argv)
    # the defaults are read only when needed: MJX_REFERENCE_RUNS is unset without the oracle (mitjax/paths.py)
    a.upstream = a.upstream or str(paths.UPSTREAM)
    a.runs = a.runs or str(paths.REFERENCE_RUNS)

    if a.exp_dir:
        exp_src = Path(a.exp_dir).resolve()
        if exp_src.name != a.experiment:
            raise SystemExit(f"--exp-dir {exp_src}: its name is not {a.experiment}")
        verif = exp_src.parent
    else:
        verif = Path(a.upstream) / "verification"
        exp_src = verif / a.experiment
    dirs, run_name = input_dirs(a.input_dir)
    for d in dirs:
        if not (exp_src / d).is_dir():
            raise SystemExit(f"{exp_src / d} is not a directory")
    binary = frozen_binary(a.binary) if a.binary else None
    overlays, overlay_records = build_overlays(exp_src, dirs, a.mpi, a.set) if a.set else ({}, [])
    run_id = a.run_id or (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                          + f"-{os.getpid()}")
    top = Path(a.runs) / a.experiment / a.input_dir / run_id
    if top.exists() or top.is_symlink():
        raise SystemExit(f"{top} exists; choose a new run id (nothing is overwritten)")

    mirror = top / "verification"
    exp_dir = mirror / a.experiment
    rundir = exp_dir / run_name
    exp_dir.mkdir(parents=True)
    for sib in sorted(os.listdir(verif)):
        if sib != a.experiment and (verif / sib).is_dir() and not sib.startswith("."):
            (mirror / sib).symlink_to(verif / sib)
    for d in dirs:
        (exp_dir / d).symlink_to(exp_src / d)
    # the experiment's other entries too (testreport runs inside the experiment directory itself): prepare_run
    # scripts read sibling input directories (global_ocean.cs32x15/input.in_p/prepare_run links from
    # ../input.icedyn and ../input.seaice)
    for sib in sorted(os.listdir(exp_src)):
        if sib not in dirs and sib != run_name and not sib.startswith("."):
            (exp_dir / sib).symlink_to(exp_src / sib)
    rundir.mkdir()
    (top / "rundir").symlink_to(Path("verification") / a.experiment / run_name)

    made = linkdata(rundir, dirs, mpi=a.mpi, overlays=overlays)
    if overlay_records:
        (top / "OVERLAY.txt").write_text("".join(
            f"{r['file']} &{r['group']} {r['key']}={r['value']} (source {r['source']}): "
            f"old {r['old_line']!r} -> new {r['new_line']!r}\n" for r in overlay_records))
    prep = None
    if os.access(rundir / "prepare_run", os.X_OK):
        script = (rundir / "prepare_run").read_text(errors="replace")
        removing = removing_commands(script)
        if removing:
            (top / "REFUSED.txt").write_text(f"prepare_run would remove files: {removing} (nothing is ever removed)\n")
            print(f"REFUSED {top}: prepare_run has removing commands {removing}")
            return 1
        env, shims = dict(os.environ), []
        if re.search(r"(?m)^[^#]*\bgunzip\b", script):
            real = shutil.which("gunzip")
            if real is None:
                raise SystemExit("prepare_run calls gunzip, but no gunzip on PATH")
            sdir = top / "prepare_run_shims"
            sdir.mkdir()
            (sdir / "gunzip").write_text("#!/bin/sh\n# make_rundir.py: prepare_run's gunzip keeps its input "
                                         f"(nothing is removed)\nexec {real} --keep \"$@\"\n")
            (sdir / "gunzip").chmod(0o755)
            env["PATH"] = f"{sdir}:{env.get('PATH', '')}"
            shims.append(f"gunzip -> {real} --keep")
        before = listing(rundir, made)
        res = subprocess.run(["./prepare_run"], cwd=rundir, capture_output=True, text=True, env=env)
        (top / "prepare_run.log").write_text(f"exit {res.returncode}\n--- stdout\n{res.stdout}--- stderr\n{res.stderr}")
        prep = {"exit": res.returncode, "source": made.get("prepare_run")}
        if shims:
            prep["shims"] = shims
        gone = removed_by_prepare_run(before, listing(rundir))
        # a prepare_run whose source directory is missing prints " Error: <dir> not a directory" and exits 0
        errs = [ln.strip() for ln in res.stdout.split("\n") if ln.strip().startswith("Error:")]
        if res.returncode != 0 or gone or errs:
            why = (f"prepare_run exited {res.returncode}" if res.returncode != 0 else
                   f"prepare_run removed or replaced {gone}" if gone else f"prepare_run reported {errs}")
            (top / "REFUSED.txt").write_text(f"{why} (prepare_run.log)\n")
            print(f"REFUSED {top}: {why}")
            return 1

    copies = []
    for spec in a.copy:
        src, _, name = spec.rpartition(":")
        # NAME is a file name, or <dir>/<file> with one plain directory (lane A session 11: lab_sea/input_ad* read
        # their controls from ctrlDir='./ctrl_variables', data.ctrl; the model's own `mkdir -p` of ctrlDir,
        # ctrl_readparms.F:535-554, accepts the existing directory)
        parts = name.split("/")
        if (not src or not name or len(parts) > 2 or any(q in ("", ".", "..") for q in parts)
                or not Path(src).is_file()):
            raise SystemExit(f"bad --copy {spec!r}: expected SOURCE:NAME (NAME a file or <dir>/<file>) with an "
                             "existing SOURCE file")
        if (rundir / name).exists() or (rundir / name).is_symlink():
            (top / "REFUSED.txt").write_text(f"--copy {spec}: {name} exists in the run directory\n")
            print(f"REFUSED {top}: --copy target {name} exists")
            return 1
        data = Path(src).read_bytes()
        if len(parts) == 2:
            (rundir / parts[0]).mkdir(exist_ok=True)
        (rundir / name).write_bytes(data)
        copies.append({"name": name, "source": str(Path(src).resolve()), "sha256": hashlib.sha256(data).hexdigest()})

    if binary:
        (rundir / "mitgcmuv").symlink_to(binary[0])

    manifest = {
        "experiment": a.experiment, "input_dir": a.input_dir, "linkdata_dirs": dirs, "run_dir": str(rundir),
        "upstream": str(a.upstream),
        "upstream_commit": subprocess.run(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip(),
        "binary": {"path": str(binary[0]), "sha256": binary[1]} if binary else None,
        "prepare_run": prep,
        "entries": {n: dict(entry(rundir / n), linked_by=(("overlay " if n in overlays else "linkdata ") + made[n])
                            if n in made else
                            ("make_rundir --binary" if n == "mitgcmuv" and binary else
                             "make_rundir --copy" if n in {c["name"] for c in copies} else "prepare_run"))
                    for n in sorted(os.listdir(rundir))},
        "required_inputs": required_inputs(rundir),
    }
    if a.mpi or overlay_records or copies:   # keys only when used: a default run's MANIFEST.json is unchanged
        manifest["mpi_linkdata"] = a.mpi
        manifest["overlays"] = overlay_records
        manifest["copies"] = copies
    (top / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")

    dangling = [n for n, e in manifest["entries"].items() if "link" in e and not e["exists"]]
    missing = missing_inputs(rundir)
    if dangling or missing:
        why = [f"dangling link {n}" for n in dangling] + [f"missing input {n} ({w})" for n, w in missing]
        (top / "REFUSED.txt").write_text("\n".join(why) + "\n")
        print(f"REFUSED {top}:", *why, sep="\n  ")
        return 1
    (top / "READY").write_text(f"{run_id}\n")
    print(f"RUNDIR {rundir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
