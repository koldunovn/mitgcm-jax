#!/usr/bin/env python3
"""Check a genmake2 build directory before and after `make`: no requested package silently dropped, strict
floating point, NetCDF present, and (for a forward-only build of a code_ad directory) no adjoint compiled in.

    check_build.py packages BUILD_DIR --rootdir ROOT --mods MODS [--report FILE]
    check_build.py flags    BUILD_DIR [--ieee]
    check_build.py forward  BUILD_DIR
    check_build.py options  BUILD_DIR --out DIR     (effective macros of every *OPTIONS.h; not a check)

Stdlib only (runs in any python). Exit 0 = pass, 1 = a check failed (the reason is printed), 2 = usage error.

packages: the requested package set is derived independently of genmake2's own messages, the way genmake2 defines
it: the first `packages.conf` found in the build directory, then in each `-mods` directory (tools/genmake2:2444-2452),
else the group `default_pkg_list` (tools/genmake2:2453-2459); comments stripped (#...); groups of
`pkg/pkg_groups` (lines `name : members`) expanded recursively (genmake2 expand_pkg_groups, tools/genmake2:221-245;
here by exact name, where genmake2 greps for a line starting with the name, a prefix match); every `-name` entry
disables `name` (tools/genmake2:2483-2500). Each requested package must then appear as `-DALLOW_<NAME>` in the
Makefile's ENABLED_PACKAGES and as `#define ALLOW_<NAME>` in PACKAGES_CONFIG.h (both written by genmake2,
tools/genmake2:2746-2752, 3164, 4059). A missing one is a failure: genmake2 removes mnc, profiles and obsfit with
only a warning when its NetCDF test fails (tools/genmake2:2527-2568). Packages added by the dependency rules
(pkg/pkg_depend) are reported, not failed.

flags: the Makefile's compiler flags (FFLAGS, FOPTIM, F90FLAGS, F90OPTIM, CFLAGS, DEFINES) contain
-ffp-contract=off and -fconvert=big-endian, none of the value-changing options below, -DHAVE_NETCDF in DEFINES,
and with --ieee FOPTIM exactly -O0 (the optfile's IEEE branch).

forward: AD_CONFIG.h (written by `make`, tools/genmake2:3337-3339) undefines ALLOW_ADJOINT_RUN and
ALLOW_TANGENTLINEAR_RUN, so the_model_main.F calls THE_MAIN_LOOP (the_model_main.F:704-714 under ALLOW_AUTODIFF).
"""

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN_FLAGS = ("-ffast-math", "-Ofast", "-funsafe-math-optimizations", "-fassociative-math",
                   "-freciprocal-math", "-ffp-contract=fast", "-ffp-contract=on", "-march=", "-mtune=native",
                   "-mfma", "-mavx", "-fno-signed-zeros", "-ffinite-math-only", "-fno-trapping-math")
REQUIRED_FLAGS = ("-ffp-contract=off", "-fconvert=big-endian")


def strip_comments(text):
    return [ln.split("#", 1)[0].strip() for ln in text.splitlines()]


def read_groups(pkg_groups):
    """{group: [members]} from pkg/pkg_groups lines `name : m1 m2 ...` (comments stripped)."""
    groups = {}
    for ln in strip_comments(Path(pkg_groups).read_text()):
        if ":" not in ln:
            continue
        name, members = ln.split(":", 1)
        name = name.strip()
        if name and members.split():
            groups[name] = members.split()
    return groups


def find_packages_conf(build_dir, mods):
    """The packages.conf genmake2 uses: build dir first, then each -mods directory in order; None = default list."""
    for d in [Path(build_dir)] + [Path(m) for m in mods]:
        if (d / "packages.conf").is_file():
            return d / "packages.conf"
    return None


def requested_packages(packages_conf, groups):
    """Return (requested, disabled, source): the package set genmake2 is asked to compile."""
    if packages_conf is None:
        tokens, source = ["default_pkg_list"], "default_pkg_list (no packages.conf)"
    else:
        tokens = [t for ln in strip_comments(Path(packages_conf).read_text()) for t in ln.split()]
        source = str(packages_conf)

    def expand(tok, seen=()):
        if tok in seen:
            raise ValueError(f"package group cycle: {' -> '.join(seen + (tok,))}")
        if tok in groups:
            return [x for m in groups[tok] for x in expand(m, seen + (tok,))]
        return [tok]

    expanded = [x for t in tokens for x in expand(t)]
    disabled = {t[1:] for t in expanded if t.startswith("-")}
    requested = {t for t in expanded if not t.startswith(("-", "+"))} - disabled
    return requested, disabled, source


def makefile_var(makefile, name):
    """Value of `NAME = ...` in a genmake2 Makefile (with backslash continuations); None if absent."""
    lines = Path(makefile).read_text().splitlines()
    for i, ln in enumerate(lines):
        m = re.match(rf"^{name}\s*=(.*)$", ln)
        if m:
            val = m.group(1)
            while val.endswith("\\") and i + 1 < len(lines):
                i += 1
                val = val[:-1] + " " + lines[i]
            return val.strip()
    return None


def enabled_in_makefile(makefile):
    val = makefile_var(makefile, "ENABLED_PACKAGES")
    if val is None:
        raise ValueError(f"{makefile}: no ENABLED_PACKAGES line")
    return {t[len("-DALLOW_"):] for t in val.split() if t.startswith("-DALLOW_")}


def enabled_in_config_h(config_h):
    return set(re.findall(r"^#define\s+ALLOW_(\w+)", Path(config_h).read_text(), re.M))


def check_packages(build_dir, rootdir, mods, report=None):
    build_dir = Path(build_dir)
    groups = read_groups(Path(rootdir) / "pkg" / "pkg_groups")
    conf = find_packages_conf(build_dir, mods)
    requested, disabled, source = requested_packages(conf, groups)
    want = {p.upper() for p in requested}
    in_mk = enabled_in_makefile(build_dir / "Makefile")
    in_h = enabled_in_config_h(build_dir / "PACKAGES_CONFIG.h")
    missing_mk, missing_h = sorted(want - in_mk), sorted(want - in_h)
    added = sorted((in_mk | in_h) - want)
    lines = [f"package list source: {source}",
             f"requested ({len(want)}): {' '.join(sorted(want))}",
             f"disabled by '-name': {' '.join(sorted(p.upper() for p in disabled)) or '-'}",
             f"enabled in Makefile ENABLED_PACKAGES ({len(in_mk)}): {' '.join(sorted(in_mk))}",
             f"enabled in PACKAGES_CONFIG.h ({len(in_h)}): {' '.join(sorted(in_h))}",
             f"added by dependency rules: {' '.join(added) or '-'}"]
    ok = not missing_mk and not missing_h
    if missing_mk:
        lines.append(f"FAIL: requested but not in ENABLED_PACKAGES: {' '.join(missing_mk)}")
    if missing_h:
        lines.append(f"FAIL: requested but not defined in PACKAGES_CONFIG.h: {' '.join(missing_h)}")
    lines.append("PACKAGES OK" if ok else "PACKAGES DROPPED")
    text = "\n".join(lines) + "\n"
    if report:
        Path(report).write_text(text)
    print(text, end="")
    return ok


def check_flags(build_dir, ieee):
    mk = Path(build_dir) / "Makefile"
    vals = {v: makefile_var(mk, v) or "" for v in ("FFLAGS", "FOPTIM", "F90FLAGS", "F90OPTIM", "CFLAGS", "DEFINES")}
    errors = []
    for v in ("FFLAGS", "F90FLAGS"):
        for f in REQUIRED_FLAGS:
            if f not in vals[v].split():
                errors.append(f"{v} lacks {f}")
    for v, val in vals.items():
        for f in FORBIDDEN_FLAGS:
            if any(tok.startswith(f) for tok in val.split()):
                errors.append(f"{v} has {f}")
    if "-DHAVE_NETCDF" not in vals["DEFINES"].split():
        errors.append("DEFINES lacks -DHAVE_NETCDF (genmake2 found no working NetCDF)")
    if ieee and vals["FOPTIM"].split() != ["-O0"]:
        errors.append(f"FOPTIM is {vals['FOPTIM']!r}, expected '-O0' (-ieee)")
    for v, val in vals.items():
        print(f"{v} = {val}")
    for e in errors:
        print(f"FAIL: {e}")
    print("FLAGS OK" if not errors else "FLAGS BAD")
    return not errors


def check_forward(build_dir):
    text = (Path(build_dir) / "AD_CONFIG.h").read_text()
    errors = []
    for name in ("ALLOW_ADJOINT_RUN", "ALLOW_TANGENTLINEAR_RUN"):
        if re.search(rf"^#define\s+{name}\b", text, re.M):
            errors.append(f"AD_CONFIG.h defines {name}")
        if not re.search(rf"^#undef\s+{name}\b", text, re.M):
            errors.append(f"AD_CONFIG.h does not #undef {name}")
    for e in errors:
        print(f"FAIL: {e}")
    print("FORWARD OK" if not errors else "FORWARD BAD")
    return not errors


def cpp_program(makefile):
    """The preprocessor of `CPPCMD = cat $< | <cpp ...> $(DEFINES) $(INCLUDES) | ...` (tools/genmake2:3380)."""
    val = makefile_var(makefile, "CPPCMD") or ""
    m = re.match(r"cat \$< \| (.*?) \$\(DEFINES\) \$\(INCLUDES\)", val)
    if not m:
        raise ValueError(f"{makefile}: cannot read the preprocessor from CPPCMD = {val!r}")
    return m.group(1).split()


def dump_options(build_dir, out):
    """For every *OPTIONS.h of the build (the links make placed in BUILD_DIR: the -mods copy where there is one,
    else the package's own): a copy, its source path, and the macros in effect after preprocessing it with the
    build's own cpp, DEFINES and INCLUDES (`cpp -dM`, sorted). Fails if a dump is empty."""
    import subprocess

    build_dir, out = Path(build_dir), Path(out)
    out.mkdir()
    mk = build_dir / "Makefile"
    cpp = cpp_program(mk)
    flags = (makefile_var(mk, "DEFINES") or "").split() + (makefile_var(mk, "INCLUDES") or "").split()
    headers = sorted(p for p in build_dir.glob("*OPTIONS.h"))
    if not headers:
        raise ValueError(f"no *OPTIONS.h in {build_dir}")
    with open(out / "SOURCES.txt", "w") as src:
        src.write(f"preprocessor: {' '.join(cpp)} -dM {' '.join(flags)} <header>  (run in {build_dir})\n")
        for h in headers:
            src.write(f"{h.name}  <- {h.resolve()}\n")
            (out / h.name).write_text(h.read_text())
            res = subprocess.run(cpp + ["-dM"] + flags + [h.name], cwd=build_dir, capture_output=True, text=True)
            if res.returncode != 0 or not res.stdout.strip():
                raise ValueError(f"cpp -dM {h.name} failed: {res.stderr.strip()}")
            (out / f"{h.name}.dM").write_text("".join(sorted(res.stdout.splitlines(keepends=True))))
    print(f"OPTIONS {len(headers)} headers -> {out}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("options")
    o.add_argument("build_dir")
    o.add_argument("--out", required=True)
    p = sub.add_parser("packages")
    p.add_argument("build_dir")
    p.add_argument("--rootdir", required=True)
    p.add_argument("--mods", action="append", default=[])
    p.add_argument("--report")
    f = sub.add_parser("flags")
    f.add_argument("build_dir")
    f.add_argument("--ieee", action="store_true")
    w = sub.add_parser("forward")
    w.add_argument("build_dir")
    a = ap.parse_args(argv)
    if a.cmd == "packages":
        ok = check_packages(a.build_dir, a.rootdir, a.mods, a.report)
    elif a.cmd == "flags":
        ok = check_flags(a.build_dir, a.ieee)
    elif a.cmd == "options":
        ok = dump_options(a.build_dir, a.out)
    else:
        ok = check_forward(a.build_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
