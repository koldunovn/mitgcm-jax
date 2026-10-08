"""FTZ/DAZ Fortran oracle (plan decision 13, Nikolay 2026-10-04; lane A session 12).

XLA:CPU computes with the MXCSR flags FTZ and DAZ set (L-CONF-2), gfortran keeps subnormals; where a gate meets
subnormals (first: global_ocean.cs32x15/input.seaice, SEAICE_LSR's decaying uIce/vIce) the port is gated bitwise
against the FTZ variant of the same builds: the standard build's objects linked with the compiler's crtfastmath.o
(reference/relink_ftz.sh, reference/jobs/ftz_build.sbatch, job 27878830). Checked here:
  * every FTZ build (rr.FTZ_SOURCE): frozen binary + sha256, objects identical to the source build's (sha256 list vs
    the source build directory), the standard relink reproduced the source binary, set_fast_math in the FTZ binary
    only (nm), MXCSR DAZ + FZ at MAIN__ in the FTZ binary only, the FTZ probe flushed (ftz_provenance.txt);
  * the FTZ invisibility triple (rr.FTZ_VARIANTS): registered kinds ftz_plain/ftz_jdoff/ftz_jdon, INVISIBLE dumps off
    and on;
  * the band measurement FTZ vs standard (reference/ftz_band.py, job ftzband): recomputed here from the runs, equal to
    the stored report, BAND CONFINED; the dumped LSR velocities do differ (the band is not empty).
Negative controls (measured): a planted provenance without FZ, a planted VISIBLE verdict, a planted pickup value
outside the band (and one inside it, classified in band), a planted STDOUT token outside the band.
Oracle-dependent: fails (never skips) while a build, run or report is missing.
"""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from mitjax import paths

REPO = Path(__file__).resolve().parents[2]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load(REPO / "reference" / "reference_runs.py", "_mjx_rr_ftz")
band = _load(REPO / "reference" / "ftz_band.py", "_mjx_ftz_band")
EXP = ("global_ocean.cs32x15", "input.seaice")


def _bin(build, exp="global_ocean.cs32x15", code="code"):
    return paths.REFERENCE / "bin" / rr.bin_name(exp, code, build)


def ftz_build_problems(d, src):
    """Problems of an FTZ bin directory d against its standard source bin directory src (empty list = good)."""
    d, src = Path(d), Path(src)
    probs = []
    sha = (d / "sha256").read_text().split()[0]
    ssha = (src / "sha256").read_text().split()[0]
    if rr.sha256(d / "mitgcmuv") != sha:
        probs.append("mitgcmuv != sha256 file")
    if sha == ssha:
        probs.append("FTZ binary equals the standard binary")
    prov = {ln[:12].strip(): ln[12:] for ln in (d / "ftz_provenance.txt").read_text().splitlines() if ln[:1] != " "}
    if f"sha256 {ssha} =" not in prov.get("relink std", ""):
        probs.append("standard relink did not reproduce the source binary")
    if f"sha256 {sha};" not in prov.get("ftz binary", "") or "set_fast_math: FTZ 1, standard 0" not in prov["ftz binary"]:
        probs.append("ftz binary line: sha256 or set_fast_math")
    mx = prov.get("MXCSR", "")
    f_part, _, s_part = mx.partition("; standard binary")
    if not ("DAZ" in f_part.split() and "FZ" in f_part.split()) or "DAZ" in s_part.split() or "FZ" in s_part.split():
        probs.append(f"MXCSR line {mx!r}")
    crt = prov.get("crtfastmath", "")
    if not crt.startswith("/sw/spack-levante/gcc-11.2.0-"):
        probs.append(f"crtfastmath.o not the gfortran 11.2.0 one: {crt!r}")
    probes = [ln for ln in (d / "ftz_provenance.txt").read_text().splitlines() if ln.startswith("probe       1.D")]
    if len(probes) != 2 or not all(ln.rstrip().endswith("0.00000000000000000E+000") for ln in probes):
        probs.append("FTZ probe not flushed")
    if "objects identical" not in prov.get("objects", "") and "sha256 identical" not in prov.get("objects", ""):
        probs.append("objects line")
    return probs


def test_ftz_builds():
    """Each FTZ build: the standard build's objects (sha256 list = the source build directory's, recomputed),
    crtfastmath.o of gfortran 11.2.0 linked, FTZ/DAZ at run time; set_fast_math only in the FTZ binary (nm)."""
    assert set(rr.FTZ_SOURCE) <= set(rr.BUILDS) and set(rr.FTZ_SOURCE.values()) <= set(rr.BUILDS)
    for ftz, std in rr.FTZ_SOURCE.items():
        d, s = _bin(ftz), _bin(std)
        assert d.name == s.name + "-ftz", (d.name, s.name)
        assert ftz_build_problems(d, s) == [], (ftz, ftz_build_problems(d, s))
        listed = {ln.split()[1][2:]: ln.split()[0] for ln in (d / "objects_sha256.txt").read_text().splitlines()}
        bld = paths.REFERENCE / "build" / s.name / "bld"
        assert len(listed) > 900 and listed == {o: rr.sha256(bld / o) for o in listed}, ftz
        assert sorted(p.name for p in bld.glob("*.o")) == sorted(listed), ftz
        nm = {b: subprocess.run(["nm", str(x / "mitgcmuv")], capture_output=True, text=True, check=True).stdout
              for b, x in (("ftz", d), ("std", s))}
        assert " set_fast_math\n" in nm["ftz"] and " set_fast_math\n" not in nm["std"], ftz
        # the build record is the source build's (options, packages, flags checks)
        for f in ("options/CPP_OPTIONS.h.dM", "flags_check.txt", "PACKAGES_CONFIG.h"):
            assert (d / f).read_bytes() == (s / f).read_bytes(), (ftz, f)


def test_ftz_build_control(tmp_path):
    """Negative control: a planted ftz_provenance.txt whose FTZ binary lacks FZ, and one whose standard relink did
    not reproduce the source, are refused."""
    ftz, std = next(iter(rr.FTZ_SOURCE.items()))
    d, s = _bin(ftz), _bin(std)
    for name, edit in (("nofz", lambda t: t.replace("UM PM FZ ]; standard", "UM PM ]; standard")),
                       ("relink", lambda t: t.replace((s / "sha256").read_text().split()[0] + " =", "0" * 64 + " ="))):
        p = tmp_path / name
        p.mkdir()
        for f in ("mitgcmuv", "sha256"):
            (p / f).symlink_to(d / f)
        txt = (d / "ftz_provenance.txt").read_text()
        assert edit(txt) != txt, name
        (p / "ftz_provenance.txt").write_text(edit(txt))
        assert ftz_build_problems(p, s) != [], name


def test_ftz_runs_registered_and_invisible():
    """The FTZ triple of each FTZ variant: registered kinds with the FTZ builds, steps nIter0..+2, never a yardstick;
    INVISIBLE dumps off and on (no new exemption)."""
    for (exp, inp), (job, bp, bj, _, _) in rr.FTZ_VARIANTS.items():
        runs = {r["kind"]: r for r in rr.RUNS if (r["exp"], r["input"]) == (exp, inp) and r["kind"] in rr.FTZ_KINDS}
        assert set(runs) == set(rr.FTZ_KINDS)
        assert runs["ftz_plain"]["build"] == bp and runs["ftz_jdon"]["build"] == runs["ftz_jdoff"]["build"] == bj
        n0 = rr.NITER0.get((exp, inp), 0)
        assert runs["ftz_jdon"]["steps"] == (n0, n0 + 1, n0 + 2) and "NEVER A YARDSTICK" in runs["ftz_plain"]["purpose"]
        assert rr.find_run(exp, inp, "yardstick")["build"] == rr.FTZ_SOURCE[bp]
        lines = rr.invisibility_file(exp, inp, ftz=True).read_text().split("\n")
        assert _invisible(lines, exp, inp), lines[:3]


def _invisible(lines, exp, inp):
    if any(ln.startswith("VISIBLE") for ln in lines):
        return False
    return all(len([ln for ln in lines if ln.startswith(f"INVISIBLE {exp}/{inp} {m}")]) == 1
               for m in ("dumps-off", "dumps-on"))


def test_ftz_invisibility_control():
    """Negative control: the verdict check refuses a planted VISIBLE line."""
    lines = rr.invisibility_file(*EXP, ftz=True).read_text().split("\n")
    assert _invisible(lines, *EXP)
    assert not _invisible(lines + [f"VISIBLE {EXP[0]}/{EXP[1]} dumps-on: planted"], *EXP)


def test_ftz_band_measurement(tmp_path):
    """FTZ vs standard oracle, recomputed from the registered runs: BAND CONFINED for the plain and the dumps-on
    triple members, the report identical to the stored one of the band job; the LSR velocities differ (the band is
    not empty)."""
    for (exp, inp), (job, _, _, bjob, sjob) in rr.FTZ_VARIANTS.items():
        base = paths.REFERENCE_RUNS / exp / inp
        for mode in ("plain", "jdon"):
            stored = rr.ftz_band_file(exp, inp, mode).read_text()
            js = json.loads(rr.ftz_band_file(exp, inp, mode, "json").read_text())
            assert js["problems"] == [] and js["verdict"].startswith("BAND CONFINED"), (mode, js["verdict"])
            out = tmp_path / f"{mode}.txt"
            assert band.main([str(base / f"job{sjob}-{mode}"), str(base / f"job{job}-{mode}"), "--out", str(out)]) == 0
            assert out.read_text() == stored, mode
            if mode == "jdon":
                keys = set(js["dumps"])
                assert any(k.endswith("Y06_lsr/UICE") for k in keys) and any(k.endswith("Y06_lsr/VICE") for k in keys)
                assert all(v["in_band"] for v in js["dumps"].values())
            assert stored.splitlines()[-1 - len(js["problems"])].startswith("BAND CONFINED"), mode


def test_ftz_band_control(tmp_path):
    """Negative controls of the band classifier: a pickup value moved outside the band (one ulp of a normal value) is
    OUTSIDE, a value moved inside it (1e-310 for 0.) is in band; a STDOUT number changed by one digit is outside."""
    run = rr.find_run(*EXP, "ftz_plain")
    src = (rr.run_top(run) / "rundir").resolve() / "pickup_seaice.ckptA.data"
    out = {}
    for name, plant in (("outside", lambda a: np.nextafter(a[np.argmax(np.abs(a))], np.inf)), ("inside", None)):
        a = np.fromfile(src, ">f8").copy()
        if plant is None:
            k = int(np.flatnonzero(a == 0)[0])
            a[k] = 1e-310
        else:
            k = int(np.argmax(np.abs(a)))
            a[k] = plant(a)
        p = tmp_path / name / src.name
        p.parent.mkdir()
        a.astype(">f8").tofile(p)
        shutil.copy(src.with_suffix(".meta"), p.with_suffix(".meta"))
        out[name] = band.compare_binary(src, p, src.name, 15)
    assert len(out["outside"]) == 1 and not next(iter(out["outside"].values()))["in_band"], out["outside"]
    assert len(out["inside"]) == 1 and next(iter(out["inside"].values()))["in_band"], out["inside"]
    line = " %MON seaice_uice_max              =   3.9999999999999997E-01"
    n, probs, _ = band.compare_lines([line], [line.replace("97E-01", "98E-01")], "STDOUT")
    assert n == 1 and len(probs) == 1
    n, probs, _ = band.compare_lines([line], [line.replace("3.9999999999999997E-01", "1.0E-310")], "STDOUT")
    assert n == 1 and len(probs) == 1          # one side normal: outside
    tiny = " %MON seaice_uice_min              =  -1.0000000000000000E-310"
    n, probs, _ = band.compare_lines([tiny], [tiny.replace("-1.0000000000000000E-310", "0.0000000000000000E+00")],
                                     "STDOUT")
    assert n == 1 and probs == []
