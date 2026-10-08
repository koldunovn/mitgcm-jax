"""The package list genmake2 compiles for an experiment's code directory, and its PACKAGES_CONFIG.h (plan Task 6).

A literal port of genmake2 @63cdc0b (MJX_UPSTREAM tools/genmake2), the steps in its order:
  1. package list: the first `packages.conf` in the build directory, then in each -mods directory
     (genmake2:2444-2452), else the group `default_pkg_list` (genmake2:2453-2459); comments (`#...`) stripped,
     words split (genmake2:2466-2474);
  2. group expansion, repeated until nothing matches (genmake2:2478-2483, expand_pkg_groups genmake2:220-246): pkg_groups
     lines `name : members` (after `s/#.*$//` and `s/:/ : /g`, kept when NF>2 and $2==":"); a package word is replaced
     when `grep "^ *word"` matches a line, i.e. a line STARTING with the word (a prefix match, kept literally); the
     replacement is every field but the first two of the matched lines joined by blanks (`echo $line | awk
     '{ $1=""; $2=""; print $0 }'`);
  3. DISABLE: every word containing `-` names a package (its first `-` removed) that is dropped
     (genmake2:2485-2504); ENABLE (`-enable`, not used by testreport) appended; every name must be a directory
     pkg/<name> (genmake2:2505-2513); the list becomes `grep -v "-" | sort | uniq` (genmake2:2514-2518);
  4. NetCDF: without HAVE_NETCDF, mnc/profiles/obsfit are removed (with the literal `sed -e 's/mnc//g'`) and
     disabled (genmake2:2522-2560); with it, the mnc templates are built (genmake2:2561-2575, assumed to succeed:
     the frozen builds' checks show mnc compiled where requested);
  5. dependency rules of pkg/pkg_depend (get_pdepend_list genmake2:822-837; applied genmake2:2614-2700): one rule per
     (first word, other word) of each line; a rule fires when its package is in the list or in STANDARDDIRS
     ("eesupp model", genmake2:2362); `+dep` and plain/`=dep` add dep unless disabled (`+` with dep disabled is an
     error), `-dep` with dep present is an error; passes repeat until two consecutive passes add nothing;
  6. PACKAGES_CONFIG.h: `-UALLOW_<NAME>` for every directory of `ls -1 pkg` (not CVS) not in the list, `-DALLOW_<NAME>`
     for each package in list order (+ `-DALLOW_AIM` after aim_v23), written by the upstream script
     tools/convert_cpp_cmd2defines (genmake2:2722-2758, 4059), which is run here unchanged.
`ls -1` and `sort` follow the locale; for master's package names the C order and the en_US.UTF-8 order are the same
(checked on the clone), and Python's sort is the C order.
Stdlib only.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

STANDARDDIRS = ("eesupp", "model")      # tools/genmake2:2362 (default; -standarddirs is not used by testreport)


@dataclass(frozen=True)
class PackageSet:
    source: str            # the packages.conf used, or "default_pkg_list"
    words: tuple           # the package words before group expansion
    expanded: tuple        # after group expansion
    disable: tuple         # DISABLE (from `-name` words, plus mnc/profiles/obsfit without NetCDF)
    packages: tuple        # final list in genmake2's order (sorted, then dependency additions)
    added_by_rules: tuple  # packages added by pkg_depend rules
    all_dirs: tuple        # `ls -1 pkg` directories (not CVS)

    def __contains__(self, name):
        return name in self.packages

    @property
    def enabled_defines(self):
        out = []
        for p in self.packages:
            out.append(f"-DALLOW_{p.upper()}")
            if p == "aim_v23":                              # genmake2:2751-2756 (the "UGLY HACK")
                out.append("-DALLOW_AIM")
        return tuple(out)

    @property
    def disabled_defines(self):
        return tuple(f"-UALLOW_{n.upper()}" for n in self.all_dirs if n not in self.packages)


def _strip(text):
    return [re.sub(r"#.*$", "", ln) for ln in text.splitlines()]


def read_pkg_groups(path):
    """The lines of pkg_groups that expand_pkg_groups uses, as field lists (genmake2:225-226)."""
    out = []
    for ln in _strip(Path(path).read_text()):
        f = ln.replace(":", " : ").split()
        if len(f) > 2 and f[1] == ":":
            out.append((ln.replace(":", " : "), f))
    return out


def expand_pkg_groups(words, group_lines):
    """One pass of expand_pkg_groups (genmake2:222-246): returns (new words, matched)."""
    new, matched = [], False
    for w in words:
        hits = [f for text, f in group_lines if re.match(r"^ *" + re.escape(w), text)]
        if hits:
            matched = True
            joined = [x for f in hits for x in f]                    # `echo $line`: all matched lines, one line
            new += joined[2:]                                          # `$1=""; $2=""`
        else:
            new.append(w)
    return new, matched


def read_pkg_depend(path):
    """[(pname, dname)] in file order (get_pdepend_list, genmake2:826-827)."""
    rules = []
    for ln in _strip(Path(path).read_text()):
        f = ln.split()
        rules += [(f[0], d) for d in f[1:]]
    return rules


def find_packages_conf(build_dir, mods):
    """genmake2:2446-2452: '.' (the build directory) first, then each -mods directory."""
    for d in [build_dir] + list(mods):
        if d is not None and (Path(d) / "packages.conf").is_file():
            return Path(d) / "packages.conf"
    return None


def package_set(rootdir, mods, build_dir=None, have_netcdf=True):
    """The packages genmake2 compiles for `-rootdir=rootdir -mods=<mods...>` run in `build_dir` (None: a fresh
    build directory, which has no packages.conf)."""
    rootdir = Path(rootdir)
    conf = find_packages_conf(build_dir, mods)
    if conf is None:
        words, source = ["default_pkg_list"], "default_pkg_list"
    else:
        words = [w for ln in _strip(conf.read_text()) for w in ln.split()]
        source = str(conf)
    groups = read_pkg_groups(rootdir / "pkg" / "pkg_groups")
    expanded, matched = list(words), True
    for _ in range(100):
        expanded, matched = expand_pkg_groups(expanded, groups)
        if not matched:
            break
    else:
        raise ValueError(f"package group expansion does not terminate: {expanded}")
    disable = [w.replace("-", "", 1) for w in expanded if "-" in w]
    pack = [p for p in expanded if p not in disable]
    for w in pack:
        if "+" in w:
            raise ValueError(f"package word {w!r}: '+' is not a valid package list entry (genmake2:2709-2716)")
        j = re.sub(r"[-+]", "", w, count=1)
        if not (rootdir / "pkg" / j).is_dir():
            raise ValueError(f"dir '{rootdir}/pkg/{w}' missing for package '{w}' (genmake2:2507-2511)")
    packages = sorted(set(p for p in pack if "-" not in p))
    if not have_netcdf:                                                # genmake2:2522-2560
        joined = " ".join(packages)
        for name in ("mnc", "profiles", "obsfit"):
            if name in packages:
                joined = joined.replace(name, "")                     # literal `sed -e 's/<name>//g'`
        packages = joined.split()
        disable += ["mnc", "profiles", "obsfit"]
    rules = read_pkg_depend(rootdir / "pkg" / "pkg_depend")
    added, ck = [], ""
    while ck != "tt":
        for pname, dname in rules:
            pin = pname in packages or pname in STANDARDDIRS
            plus = "+" if dname.startswith("+") else "-" if dname.startswith("-") else "a"
            dname = re.sub(r"^[=+-]", "", dname)
            din = dname in packages
            if pin and plus != "-" and not din:
                if dname in disable:
                    if plus == "+":
                        raise ValueError(f'"{dname}" is required with pkg "{pname}" (pkg_depend) but is disabled')
                else:
                    packages.append(dname)
                    added.append(dname)
                    ck = ""
            if pin and plus == "-" and din:
                raise ValueError(f'"{dname}" was requested but is disallowed by the dependency rules for "{pname}"')
        ck += "t"
    all_dirs = sorted(n for n in os.listdir(rootdir / "pkg") if (rootdir / "pkg" / n).is_dir() and n != "CVS")
    return PackageSet(source, tuple(words), tuple(expanded), tuple(disable), tuple(packages), tuple(added),
                      tuple(all_dirs))


def packages_config_h(rootdir, pset):
    """PACKAGES_CONFIG.h as genmake2:4059 writes it, by the upstream tools/convert_cpp_cmd2defines itself."""
    args = (["-bPACKAGES_CONFIG_H", "Disabled packages:"] + list(pset.disabled_defines) + [" ", "Enabled packages:"]
            + list(pset.enabled_defines))
    res = subprocess.run(["bash", str(Path(rootdir) / "tools" / "convert_cpp_cmd2defines")] + args,
                         capture_output=True, text=True, check=True)
    return res.stdout


def ad_config_h(rootdir, version="Forward version"):
    """AD_CONFIG.h of a plain `make` (the Makefile target of $(EXECUTABLE), genmake2:3254-3256): the forward version,
    ALLOW_ADJOINT_RUN and ALLOW_TANGENTLINEAR_RUN undefined. The oracle builds are all plain `make` (the code_ad build
    is a forward-only build, reference/README.md); adjoint/tangent targets are not supported here."""
    if version != "Forward version":
        raise NotImplementedError(f"AD_CONFIG.h {version!r}: only the forward build is supported")
    res = subprocess.run(["bash", str(Path(rootdir) / "tools" / "convert_cpp_cmd2defines"), version,
                          "-bAD_CONFIG_H", "-UALLOW_ADJOINT_RUN", "-UALLOW_TANGENTLINEAR_RUN"],
                         capture_output=True, text=True, check=True)
    return res.stdout
