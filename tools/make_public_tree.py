#!/usr/bin/env python3
"""Build the public tree of mitjax (docs plan 20261006 Task 11, decision 13): a NEW directory with a fresh git
repository holding ONE commit of the files CONTENT marks `publish`, and a scan of every file and of the commit message
for private text. It never pushes, adds no remote, uses no network and never deletes anything.

    make_public_tree.py list  [--ref REF] [--page]                  every tracked file: publish / keep, rule, reason
    make_public_tree.py build DEST --message-file F [--ref REF] [--extra-strings FILE] [--test-only]
    make_public_tree.py scan  DIR [--message-file F] [--extra-strings FILE]

`build` refuses if DEST exists. It exports the publish set of REF (default `master`) from this repository with
`git archive`, scans the tree and the message, and only when the scan is clean runs `git init` + one commit in DEST
(message = the file's text; author and committer from the user's git configuration). With a scan hit it leaves the
exported files uncommitted for inspection and exits 1 (`--test-only` commits anyway, for a local install test, with
a first message line that makes every later `scan` of the tree fail). `scan` checks an existing tree (e.g. the
result of `build`, before a push; with its git log). Exit 0 = clean, 1 = scan hit(s) listed as
`file:line: class`, 2 = usage error.

CONTENT is the one machine-readable content list: ordered rules (path, decision, reason, open) where `path` is a file,
a directory (`dir/`) or an fnmatch pattern and the first matching rule decides. A tracked file no rule matches is an
error (`list` and `build` refuse), so a new top-level path cannot slip into the public tree unreviewed. `open=True`
marks a decision still waiting for Nikolay (the content proposal lists them); the proposal applies meanwhile. All
fourteen decisions of 2026-10-08 are taken (Nikolay 2026-10-08), so no rule is open.

SCAN holds the forbidden classes (assistant / session markers, scratch and session paths, UUIDs, this account's
paths, the LICENSE draft marker). ALLOW is the one per-file allowance, Nikolay's decision of 2026-10-08
("levante.env is fine, people can reuse then"): `levante.env` is published as it is, as the DKRZ Levante site file
to copy and adapt, so it may hold the account id and the project work path; any other class in it, and those two
classes in any other file, are hits. The assistant-marker classes follow the "assistant marker" and "session link"
classes of mitjax/tests/test_docs_pages.py (FORBIDDEN), whose stricter set (job IDs, host names) is for the user
pages only: job IDs may stay in the tree (Nikolay 2026-10-08). The patterns are written so that this file does not
match itself (a character class inside each literal). `--extra-strings FILE` adds literal strings, one per line (e.g.
a real session ID), from a file OUTSIDE the tree: hits print the line number of the string in that file, never the
string, and no real ID is ever written into a published file.

`list` also reports dangling references: a published text file naming a kept-out file (its path, or for docs/ and
scripts/ files its file name). They are not scan failures; each is a content decision (publish the target, or
rewrite the reference). Stdlib only.
"""

import argparse
import fnmatch
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PUBLISH, KEEP = "publish", "keep"

# (path or pattern, decision, reason, open). First match wins; "dir/" matches everything below dir.
CONTENT = [
    # ---------------------------------------------------------------- top level
    ("LICENSE", PUBLISH, "MIT licence with MITgcm's notice kept (decision 13; holder line Nikolay 2026-10-08)",
     False),
    ("README.md", PUBLISH, "the entry page", False),
    ("pyproject.toml", PUBLISH, "package metadata; `pip install -e .`", False),
    ("environment.yml", PUBLISH, "the users' conda environment", False),
    ("constraints.txt", PUBLISH, "the exact pinned versions (environment.yml agrees, test_env_pins.py)", False),
    ("conftest.py", PUBLISH, "pytest set-up: gate XLA flags, manifest groups, oracle skips", False),
    (".gitignore", PUBLISH, "ignores data and build files (the agent lines moved to .git/info/exclude, Nikolay "
                            "2026-10-08)", False),
    ("levante.env", PUBLISH, "the DKRZ Levante site file, published as it is for reuse (Nikolay 2026-10-08); its "
                             "paths are the one allowance of the scan (ALLOW)", False),
    ("CL" "AUDE.md", KEEP, "agent instructions (untracked; listed so a tracked copy is never published)", False),
    (".cl" "aude/", KEEP, "agent configuration (untracked; listed for safety)", False),
    ("COMMIT", KEEP, "the tier runners' snapshot stamp (written into a test snapshot, never tracked)", False),
    # ---------------------------------------------------------------- package, notebooks, tools
    ("mitjax/", PUBLISH, "the package: model, packages, drivers, API, tests", False),
    ("notebooks/", PUBLISH, "the seven walk-through notebooks, their sources and README", False),
    ("tools/make_public_tree.py", PUBLISH, "this tool: shows how the public tree is made and scanned "
                                          "(test_public_tree.py tests it)", False),
    ("tools/", PUBLISH, "docs generators (gen_docs, build_notebooks, coverage), oracle tools (cpp_live, "
                        "fortran_minmax_sites, diffdump, step_vs_dump, testreport_upstream.sh)", False),
    ("reference/", PUBLISH, "the oracle-rebuild scripts (optional for users): build, run, dump and replay harnesses, "
                            "registries the tests import (reference_runs.py)", False),
    ("dev/prototype/", PUBLISH, "the Task 8 array-style prototype; read by test_style_prototype.py and cited by "
                                "kernel docstrings", False),
    ("dev/", KEEP, "development experiments not cited by published code", False),
    # ---------------------------------------------------------------- docs: user pages and reference records
    ("docs/install.md", PUBLISH, "user page", False),
    ("docs/running.md", PUBLISH, "user page", False),
    ("docs/gradients.md", PUBLISH, "user page", False),
    ("docs/configurations.md", PUBLISH, "user page", False),
    ("docs/parallel.md", PUBLISH, "user page", False),
    ("docs/checking_a_change.md", PUBLISH, "user page", False),
    ("docs/troubleshooting.md", PUBLISH, "user page", False),
    ("docs/status.md", PUBLISH, "generated status page", False),
    ("docs/fortran_map.md", PUBLISH, "generated Fortran -> Python map", False),
    ("docs/READING_GUIDE.md", PUBLISH, "how to read the code against the Fortran", False),
    ("docs/ISSUES_UPSTREAM.md", PUBLISH, "what the port found in MITgcm (Nikolay 2026-10-08)", False),
    ("docs/KERNEL_GUIDE.md", PUBLISH, "how a kernel is written (Nikolay 2026-10-08)", False),
    ("docs/ENV.md", PUBLISH, "environment and pins (Nikolay 2026-10-08); one account path to rewrite", False),
    ("docs/PORTING_RULES.md", PUBLISH, "the binding rules for a change (sections 1-5); linked from README and user "
                                       "pages", False),
    ("docs/PORTING_RULES_INTERNAL.md", KEEP, "this project's operations (ECCO copy, Levante, git, lanes): the "
                                             "former sections 6-8 of PORTING_RULES.md (Nikolay 2026-10-08)", False),
    ("docs/LESSONS_CARRIED.md", PUBLISH, "lessons IDs cited by PORTING_RULES and code ([E§n]...); its test needs "
                                         "the unpublished digests and skips without them", False),
    ("docs/REFERENCE_RUNS.md", PUBLISH, "generated record of the oracle runs the tests compare with", False),
    ("docs/YARDSTICK.md", PUBLISH, "generated digits of the oracle vs results/ (linked from checking_a_change)",
     False),
    ("docs/PORTABILITY.md", PUBLISH, "measured runtime dependencies (Task 1); cited by code and tests", False),
    ("docs/M1_ACCEPTANCE.md", PUBLISH, "milestone-1 acceptance record; read by test_m1_acceptance_doc.py", False),
    ("docs/R5_CODE_AD_FORWARD.md", PUBLISH, "code_ad forward record; cited by mitjax/tests/code_ad_forward.py",
     False),
    ("docs/AUDIT_C66G_MASTER.md", PUBLISH, "c66g -> master audit table; read by scripts/audit_upstream.py's test",
     False),
    ("docs/coverage/", PUBLISH, "generated gcov coverage per experiment (linked from checking_a_change)", False),
    ("docs/review/", PUBLISH, "M1 side-by-side page (notebook 02 links it) and its generator", False),
    ("docs/PORTING_LESSONS.md", KEEP, "per-session development log (lanes, jobs, process)", False),
    ("docs/REVIEW_M0.md", KEEP, "internal review of milestone 0", False),
    ("docs/DELETION_CANDIDATES.md", KEEP, "project housekeeping", False),
    ("docs/PUBLIC_CONTENT.md", KEEP, "this content proposal, for Nikolay's yes", False),
    ("docs/HANDOFF-*", KEEP, "session handoffs (untracked)", False),
    ("docs/handoff/", KEEP, "lane notes and handoffs", False),
    ("docs/plans/", KEEP, "development plans", False),
    ("docs/brainstorm-*", KEEP, "design records of the development sessions", False),
    # ---------------------------------------------------------------- scripts: shared infrastructure and test inputs
    ("scripts/__init__.py", PUBLISH, "package marker (tests import scripts.*)", False),
    ("scripts/tests/", PUBLISH, "tests (manifest-listed; test_manifest.py needs every listed file)", False),
    ("scripts/run_tier1.sbatch", PUBLISH, "tier-1 runner (checking_a_change.md)", False),
    ("scripts/run_tier1x.sbatch", PUBLISH, "tier-1x runner", False),
    ("scripts/run_tier2.sbatch", PUBLISH, "tier-2 GPU runner", False),
    ("scripts/run_docs_notebooks.sbatch", PUBLISH, "cluster-notebook runner (test_notebooks_cluster.py)", False),
    ("scripts/fresh_install.sbatch", PUBLISH, "the fresh-install check (env from environment.yml + notebook 01)",
     False),
    ("scripts/fresh_install_nb01.sbatch", PUBLISH, "the fresh-install check, step 2 (notebook 01 + user tests)",
     False),
    ("scripts/check_pytest_report.py", PUBLISH, "suite verdicts from the JUnit report", False),
    ("scripts/tier1x_shards.py", PUBLISH, "tier-1x sharding by measured times", False),
    ("scripts/audit_upstream.py", PUBLISH, "upstream-diff audit (smoke test)", False),
    ("scripts/make_env.sbatch", PUBLISH, "builds the environment (ENV.md)", False),
    ("scripts/make_nb_venv.sbatch", PUBLISH, "notebook overlay venv (ENV.md)", False),
    ("scripts/nb_gate.sbatch", PUBLISH, "laptop-notebook gate (ENV.md, manifest_nb.py)", False),
    ("scripts/pip_install_check.sbatch", PUBLISH, "`pip install -e .` check (pyproject, ENV.md)", False),
    ("scripts/noref_probe.py", PUBLISH, "no-reference probe (test_no_reference.py)", False),
    ("scripts/noref_probe.sbatch", PUBLISH, "no-reference probe job (PORTABILITY.md)", False),
    ("scripts/port_gate.sbatch", PUBLISH, "toolchain gate (test_toolchain_system.py)", False),
    ("scripts/make_exch_maps.py", PUBLISH, "oracle exchange maps (exch_maps.py, gate helpers)", False),
    ("scripts/global_sum_ref/", PUBLISH, "global-sum reference program (test_global_sum.py)", False),
    ("scripts/grid_sharding/", PUBLISH, "grid-sharding measurement (cited by eesupp/shard.py)", False),
    ("scripts/r5_shardgrad/", PUBLISH, "sharded-gradient GPU check (tier 2 tests)", False),
    ("scripts/shardgrad/compile_study.py", PUBLISH, "compile study imported by GPU / cs32 tests", False),
    ("scripts/cs32_adjoint.py", PUBLISH, "used by test_cs32_ad_*.py", False),
    ("scripts/cs32_compile_study.py", PUBLISH, "used by test_cs32_ad_adjoint.py", False),
    ("scripts/floor_optim_perturb.py", PUBLISH, "1-ulp floor (test_r5_sharded_grad_cpu.py, decision 18)", False),
    ("scripts/goadk_adjoint.py", PUBLISH, "cited by KERNEL_GUIDE.md and status.md", False),
    ("scripts/m1_acceptance.py", PUBLISH, "used by test_m1_acceptance_doc.py, test_m2accept_pn.py", False),
    ("scripts/m1_acceptance.sbatch", PUBLISH, "M1 acceptance job (M1_ACCEPTANCE.md, tests)", False),
    ("scripts/m4adcol_adjoint.py", PUBLISH, "cited by status.md and test_api.py", False),
    ("scripts/m4adcs32ice_adjoint.py", PUBLISH, "cited by status.md", False),
    ("scripts/m4adlab_adjoint.py", PUBLISH, "cited by status.md", False),
    ("scripts/m4costshard_adexf.py", PUBLISH, "cited by drivers/adjoint_run.py", False),
    ("scripts/m4costshard_audit.py", PUBLISH, "used by test_m4costshard_grad_cpu.py", False),
    ("scripts/m4costshard_cg2dlog.py", PUBLISH, "used by test_m4costshard_grad_cpu.py (decision 18 floor)", False),
    ("scripts/m4costshard_shardgrad.py", PUBLISH, "used by test_m4costshard_grad_cpu.py", False),
    ("scripts/gpu_smoke.py", PUBLISH, "cited by M1_ACCEPTANCE.md", False),
    ("scripts/gpu_smoke.sbatch", PUBLISH, "cited by M1_ACCEPTANCE.md", False),
    ("scripts/step_timing.py", PUBLISH, "cited by M1_ACCEPTANCE.md", False),
    ("scripts/twin_*", PUBLISH, "climate-twin scripts cited by M1_ACCEPTANCE.md", False),
    ("scripts/api_gate.sbatch", KEEP, "per-lane gate job (lane API)", False),
    ("scripts/apif_gate.sbatch", KEEP, "per-lane gate job (lane APIF)", False),
    ("scripts/cli_gate.sbatch", KEEP, "per-lane gate job (lane CLI)", False),
    ("scripts/floor_gate.sbatch", KEEP, "per-lane gate job (lane FLOOR)", False),
    ("scripts/m2_acceptance.sbatch", KEEP, "per-lane acceptance job (M2), cited by nothing published", False),
    ("scripts/cg2d_modes_compare.py", KEEP, "per-lane study script, cited by nothing published", False),
    ("scripts/m4adcs32ice_shardgrad.py", KEEP, "per-lane study script, cited by nothing published", False),
    ("scripts/m4adlab_shardgrad.py", KEEP, "per-lane study script, cited by nothing published", False),
    ("scripts/m4adlab_tieprobe.py", KEEP, "per-lane study script, cited by nothing published", False),
    ("scripts/m4costshard_dynmixbranch.py", KEEP, "per-lane study script, cited by nothing published", False),
    ("scripts/m4costshard_dynmixstep.py", KEEP, "per-lane study script, cited by nothing published", False),
    ("scripts/shardgrad/compile_study.sbatch", KEEP, "per-lane study job, cited by nothing published", False),
]

# Forbidden classes (case-insensitive). Each literal carries a character class so that this file does not match
# itself; the test's planted samples are assembled at run time for the same reason.
SCAN = {
    "assistant name": r"c[l]aude",
    "co-author trailer": r"co-[a]uthored-by",
    "generated-with marker": r"\bgenerated [w]ith\b",
    "session directory": r"c[l]aude-\d+/",
    "scratch path": r"/scr[a]tch/",
    "UUID": r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    "account id": r"a27[0]088",
    "project work path": r"/work/ab09[9]5",
    "home path": r"/home/[a]/",
    "draft marker": r"\[DRAF[T]\b",
    "test-only tree": r"NOT FOR PUBLICATIO[N]",
}
# The one per-file allowance (Nikolay 2026-10-08: "levante.env is fine, people can reuse then"): the published site
# file keeps this account's DKRZ paths. Only these classes, only in this file (repository-relative path); any other hit
# stays a hit.
ALLOW = {"levante.env": frozenset({"account id", "project work path"})}
TEST_ONLY = "TEST ONLY - NOT FOR PUBLICATIO" + "N"
SCAN_RX = {name: re.compile(rx, re.I) for name, rx in SCAN.items()}


# ------------------------------------------------------------------------------------------------- content


def classify(path):
    """(decision, rule, reason, open) of a repository-relative path; None when no rule matches."""
    for pat, decision, reason, is_open in CONTENT:
        if (pat.endswith("/") and path.startswith(pat)) or path == pat or (
                any(c in pat for c in "*?[") and fnmatch.fnmatchcase(path, pat)):
            return decision, pat, reason, is_open
    return None


def tracked(ref, repo=REPO):
    """Files of `ref` (git ls-tree), as repository-relative paths."""
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "-z", ref],
                         check=True, capture_output=True).stdout.decode()
    return sorted(p for p in out.split("\0") if p)


def plan(ref, repo=REPO):
    """{path: (decision, rule, reason, open)} for every tracked file; ValueError naming unclassified files."""
    files = tracked(ref, repo)
    out = {p: classify(p) for p in files}
    missing = sorted(p for p, c in out.items() if c is None)
    if missing:
        raise ValueError("files no CONTENT rule classifies (add a rule in tools/make_public_tree.py): "
                         + ", ".join(missing))
    return out


def is_text(data):
    return b"\0" not in data[:8192]


def dangling(files_text, kept):
    """[(published file, line, kept-out target)] where a published text names a kept-out file (path, or the file
    name for docs/ and scripts/ files whose name is specific enough). A needle counts only where no path character
    precedes it: `P2/PORTING_LESSONS.md`, `notes/HANDOFF-...` and `R/docs/...` are other projects' files (the source
    tags of LESSONS_CARRIED.md), not this repository's."""
    needles = {}
    for k in kept:
        needles[k] = k
        name = k.rsplit("/", 1)[-1]
        if k.startswith(("docs/", "scripts/")) and name not in ("README.md", "__init__.py", "run.sbatch",
                                                                 "CURRENT", "build.sh"):
            needles[name] = k
    dirs = {"docs/handoff/": "docs/handoff/", "docs/plans/": "docs/plans/", "docs/HANDOFF-": "docs/HANDOFF-*",
            "HANDOFF-": "docs/HANDOFF-*"}
    needles.update(dirs)
    rx = re.compile(r"(?<![\w./~-])(?:" + "|".join(re.escape(n) for n in sorted(needles, key=len, reverse=True))
                    + ")")
    out = []
    for path, text in files_text.items():
        for i, line in enumerate(text.splitlines(), 1):
            for target in dict.fromkeys(needles[m.group(0)] for m in rx.finditer(line)):
                out.append((path, i, target))
    return out


def read_ref_files(ref, paths_, repo=REPO):
    """{path: bytes} of the given files at `ref` (one `git archive`)."""
    want = set(paths_)
    data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", ref], check=True,
                          capture_output=True).stdout
    out, modes = {}, {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if m.name in want:
                if m.issym():
                    raise ValueError(f"{m.name}: symbolic link in the publish set (not supported)")
                if m.isfile():
                    out[m.name] = tf.extractfile(m).read()
                    modes[m.name] = m.mode
    lost = want - set(out)
    if lost:
        raise ValueError(f"not in the archive of {ref}: {sorted(lost)[:5]}")
    return out, modes


# ------------------------------------------------------------------------------------------------- scan


def load_extra(path):
    """Literal strings from a file outside the tree (one per line; blank lines and `#` lines ignored)."""
    if path is None:
        return []
    return [(i, s) for i, s in enumerate(Path(path).read_text().splitlines(), 1)
            if s.strip() and not s.lstrip().startswith("#")]


def scan_text(name, text, extra=()):
    """[(name, line, class, excerpt)] of forbidden text; an extra string is reported by its line in the extra file.
    `name` is the file's path relative to the tree root: the classes ALLOW grants that file are not reported."""
    hits = []
    allowed = ALLOW.get(name, frozenset())
    low_extra = [(n, s.lower()) for n, s in extra]
    for i, line in enumerate(text.splitlines(), 1):
        for cls, rx in SCAN_RX.items():
            if cls in allowed:
                continue
            m = rx.search(line)
            if m:
                a = max(0, m.start() - 30)
                hits.append((name, i, cls, line[a:m.end() + 30].strip()))
        low = line.lower()
        for n, s in low_extra:
            if s in low:
                hits.append((name, i, f"extra string #{n}", "(not shown)"))
    return hits


def scan_tree(root, message=None, extra=()):
    """Scan every file under `root` (outside .git) and the commit message; binary files are scanned as latin-1."""
    root = Path(root)
    hits = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if ".git" in rel.parts[:1] or not p.is_file() or p.is_symlink():
            continue
        data = p.read_bytes()
        text = data.decode("utf-8") if is_text(data) and _utf8(data) else data.decode("latin-1")
        hits += scan_text(rel.as_posix(), text, extra)
    if message is not None:
        hits += scan_text("<commit message>", message, extra)
    return hits


def _utf8(data):
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def print_hits(hits, out=sys.stdout):
    by_class = {}
    for name, line, cls, excerpt in hits:
        by_class[cls] = by_class.get(cls, 0) + 1
        print(f"{name}:{line}: {cls}: {excerpt[:140]}", file=out)
    print(f"scan: {len(hits)} hit(s)" + (" -- " + ", ".join(f"{c} {n}" for c, n in sorted(by_class.items()))
                                         if hits else " (clean)"), file=out)


# ------------------------------------------------------------------------------------------------- build


def _git(dest, *args, stdin=None):
    return subprocess.run(["git", "-C", str(dest), "-c", "core.hooksPath=/dev/null", *args], check=True,
                          capture_output=True, text=True, input=stdin)


def build(ref, dest, message, extra=(), repo=REPO, test_only=False):
    """Export, scan, and (clean only) commit. Returns the hits (empty = committed). `test_only` commits despite hits
    for a local install test, with the message's first line TEST_ONLY so that `scan` of that tree always fails."""
    dest = Path(dest)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"{dest} exists: build writes into a NEW directory only (nothing is overwritten)")
    repo_root = Path(repo).resolve()
    if dest.resolve().is_relative_to(repo_root):
        raise ValueError(f"{dest} is inside the repository {repo_root}")
    p = plan(ref, repo)
    pub = sorted(f for f, c in p.items() if c[0] == PUBLISH)
    data, modes = read_ref_files(ref, pub, repo)
    dest.mkdir(parents=True)
    for f in pub:
        t = dest / f
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(data[f])
        t.chmod(0o755 if modes[f] & 0o111 else 0o644)
    hits = scan_tree(dest, message, extra)
    if hits and not test_only:
        return hits
    if test_only:
        message = f"{TEST_ONLY} (scan hits: {len(hits)})\n\n{message}"
    _git(dest, "init", "-q", "-b", "main")
    _git(dest, "add", "-f", "-A")
    staged = sorted(_git(dest, "ls-files", "-z").stdout.split("\0"))
    if [s for s in staged if s] != pub:
        raise RuntimeError("the staged set differs from the publish set")
    _git(dest, "commit", "-q", "-F", "-", stdin=message)
    identity = _git(dest, "log", "-1", "--format=%an <%ae>%n%cn <%ce>%n%B").stdout
    remotes = _git(dest, "remote").stdout.strip()
    if remotes:
        raise RuntimeError(f"unexpected remote(s) in {dest}: {remotes}")
    return (hits if test_only else []) + scan_text("<commit identity + message>", identity, extra)


# ------------------------------------------------------------------------------------------------- page


def page_table(p):
    """Markdown rows: every top-level path, every docs/ and scripts/ file (the content proposal's table)."""
    rows, seen = [], set()
    for f, (decision, rule, reason, is_open) in sorted(p.items()):
        top = f.split("/", 1)[0]
        if top in ("docs", "scripts") and "/" in f:
            key = f
        else:
            key = top + ("/" if "/" in f else "")
        if key in seen and key == top + "/" and top not in ("dev",):
            continue
        if top == "dev" and "/" in f:
            key = "/".join(f.split("/")[:2]) + "/"
            if key in seen:
                continue
        seen.add(key)
        if key.endswith("/") and top not in ("docs", "scripts"):
            n = sum(1 for g in p if g.startswith(key))
            key = f"{key} ({n} files)"
        rows.append(f"| `{key}` | {decision}{' (open)' if is_open else ''} | {reason} |")
    return ["| path | decision | reason |", "|---|---|---|"] + rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list")
    a.add_argument("--ref", default="master")
    a.add_argument("--page", action="store_true", help="print the content table as Markdown")
    b = sub.add_parser("build")
    b.add_argument("dest")
    b.add_argument("--ref", default="master")
    b.add_argument("--message-file", required=True)
    b.add_argument("--extra-strings")
    b.add_argument("--test-only", action="store_true",
                   help="commit despite scan hits, for a local install test only: the message's first line says "
                        "the tree is marked unpublishable and `scan` of it always fails")
    s = sub.add_parser("scan")
    s.add_argument("dir")
    s.add_argument("--message-file")
    s.add_argument("--extra-strings")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        try:
            p = plan(args.ref)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if args.page:
            print("\n".join(page_table(p)))
            return 0
        for f, (decision, rule, reason, is_open) in sorted(p.items()):
            print(f"{decision:7s} {'OPEN ' if is_open else '     '}{f}  [{rule}]")
        pub = [f for f, c in p.items() if c[0] == PUBLISH]
        kept = [f for f, c in p.items() if c[0] == KEEP]
        data, _ = read_ref_files(args.ref, pub)
        texts = {f: d.decode("utf-8", "replace") for f, d in data.items() if is_text(d)}
        dl = dangling(texts, kept)
        for f, line, target in dl:
            print(f"DANGLING {f}:{line} -> {target}")
        print(f"{len(pub)} publish, {len(kept)} keep, {len(dl)} dangling reference(s)")
        return 0

    extra = []
    if args.extra_strings:
        ex = Path(args.extra_strings).resolve()
        for root in (REPO.resolve(), Path(getattr(args, "dest", None) or getattr(args, "dir")).resolve()):
            if ex.is_relative_to(root):
                print(f"ERROR: the extra-strings file must live outside {root}", file=sys.stderr)
                return 2
        extra = load_extra(ex)
    message = Path(args.message_file).read_text() if args.message_file else None

    if args.cmd == "scan":
        if (Path(args.dir) / ".git").exists():        # every commit's identity and message, too
            log = _git(args.dir, "log", "--format=%an <%ae>%n%cn <%ce>%n%B").stdout
            message = (message or "") + "\n" + log
        hits = scan_tree(args.dir, message, extra)
        print_hits(hits)
        return 1 if hits else 0

    try:
        hits = build(args.ref, args.dest, message, extra, test_only=args.test_only)
    except (FileExistsError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print_hits(hits)
    if args.test_only:
        print(f"committed for a local TEST ONLY: {args.dest} (marked unpublishable; nothing was pushed)")
        return 1 if hits else 0
    if hits:
        print(f"NOT committed: {args.dest} holds the exported files for inspection", file=sys.stderr)
        return 1
    print(f"committed: {args.dest} (one commit, no remote; nothing was pushed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
