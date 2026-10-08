"""Gather maps of the Fortran halo exchanges: replayed from the exchange code at run time (`exchange_maps`, docs plan
20261006 Task 3), measured by the oracle's exchange probe for the gates (plan Task 7b, L-COPY-2).

Run time (`exchange_maps(exp)`, drivers.model.Model's default exchanger): the maps of the experiment's build are
replayed from the Fortran loops -- eesupp exch1 (and EXCH0_RX under DISCONNECTED_TILES) by
mitjax/eesupp/exch1_tables.py, pkg/exch2 (any W2 topology, cubed sphere included) by
mitjax/pkg/exch2/exch2_cube_tables.py -- and cached in $MJX_CACHE/exch_maps under a key that hashes the layout and
the topology (SIZE.h, DISCONNECTED_TILES, the W2 set-up), never the experiment's name; a cache file whose stored key
or layout differs from the computed one is refused. The probe maps below are the reference of the replay's gate
(mitjax/tests/test_exch_replay.py: bitwise equal to every registered probe map), not a run-time input.

The probe (the rest of this docstring, unchanged):

One format for the eesupp (exch1) and the pkg/exch2 exchanges: the maps are not derived from the exchange code, they
are read off the Fortran itself. `JAXDUMP_EXCH_PROBE` (reference/jaxdump/jaxdump.F:893-1031, plan Task 4) fills every
point of two arrays, halos included, with an exact code `comp*cbase + ((tile-1)*(sNy+2OLy) + ja)*(sNx+2OLx) + ia + 1`
(comp 1 in the u-like array, 2 in the v-like one; decoder `mitjax.io.dump.probe_decode`), calls each exchange routine
the model uses and dumps the result. Decoding an exchanged value gives the source point in the PRE-exchange arrays
(it may be a halo point), the source array and the sign; a point holding its own code was not written.

A map, per output array: over the flattened `[tile, j, i]` points of one level (N = nTiles*ny*nx),
    src   int32  flat index of the source point
    comp  int8   0 = the exchange does not write the point (it keeps its value), 1 = copied from the first (u-like or
                 scalar) input, 2 = copied from the second (v-like) input
    sign  int8   +1 or -1 (+1 where kept)
Exchanges are copies with signs: they read no arithmetic, so they preserve -0 and NaN, and their JAX transpose is the
matching scatter-add.

Routines and probe fields (jaxdump.F:913-981; the `name` is the map key):
    name  Fortran call                                  probe outputs
    XY    EXCH_XY_RL( phi )                             xT
    3D    EXCH_3D_RL( phi, 1 )                          x3D
    Z     EXCH_Z_3D_RL( phi, 1 )                        xZ
    SMs   EXCH_SM_3D_RL( phi, .TRUE., 1 )               xSMs
    S3D   EXCH_S3D_RL( phi, 1 ) on (0:sNx+1,0:sNy+1)    xS3D   (copied into and back out of the full array: points
                                                               beyond the width-1 ring keep their value)
    UVs   EXCH_UV_XY_RL( u, v, .TRUE. )                 xUVs_u, xUVs_v
    UVn   EXCH_UV_XY_RL( u, v, .FALSE. )                xUVn_u, xUVn_v
    As/An EXCH_UV_AGRID_3D_RL( u, v, .TRUE./.FALSE., 1) xAs_*, xAn_*
    Bs/Bn EXCH_UV_BGRID_3D_RL( u, v, .TRUE./.FALSE., 1) xBs_*, xBn_*
    Ds    EXCH_UV_DGRID_3D_RL( u, v, .TRUE., 1 )        xDs_u, xDs_v
    UV3s  EXCH_UV_3D_RL( u, v, .TRUE., 1 )              xUV3s_u, xUV3s_v
Signed-zero probes (jaxdump.F:985-1029): +0 and -0 fields through UVs, As, Bs, Ds, UV3s and SMs (zp*/zm*).

pkg/exch2 layouts: the C-grid vector exchanges (UVs, UVn, UV3s, Ds) are not copies in the Fortran (EXCH2_PUT_RX2 sums
sa1*A1 + sa2*A2 at one source index); `load_maps` attaches their per-pass tables from the W2 topology
(`ExchangeMaps.rx2`, mitjax/pkg/exch2/exch2_rx2_cube.py), which the exchangers evaluate instead of the copy. The
measured maps of those kinds remain the gate of the tables (test_exch2_vector.py).

Files: `$MJX_REFERENCE/exch_maps/<exp>-<layout tag>.npz` (scripts/make_exch_maps.py; never in the repository), with
a `.sha256` and a `.json` (provenance) next to it. `MAP_SHA256` below registers each file's sha256; `load_maps`
refuses a file whose sha256 differs. Copied from the ECCO port's `parallel/exchange.py` (`decode`, `build_maps`,
`ExchangeMaps`) with lane A's probe code (R's `1e6*comp + 1e4*tile + 100*(j+OLy) + (i+OLx)` needed sNx+2OLx < 100 and
at most 99 tiles), the exch1 routines, EXCH_S3D_RL and the signed-zero probes added.
"""

import functools
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from mitjax import paths
from mitjax.eesupp.tiles import TileLayout
from mitjax.io.dump import probe_decode

STAGE = "X00_exch_probe"
SCALAR = {"XY": "xT", "3D": "x3D", "Z": "xZ", "SMs": "xSMs", "S3D": "xS3D"}
VECTOR = {"UVs": ("xUVs_u", "xUVs_v"), "UVn": ("xUVn_u", "xUVn_v"), "As": ("xAs_u", "xAs_v"),
          "An": ("xAn_u", "xAn_v"), "Bs": ("xBs_u", "xBs_v"), "Bn": ("xBn_u", "xBn_v"),
          "Ds": ("xDs_u", "xDs_v"), "UV3s": ("xUV3s_u", "xUV3s_v")}
# kinds the probe does not measure (withSigns=.FALSE. variants) that the cube tables replay from the Fortran loops
# (load_cube_maps, mitjax/pkg/exch2/exch2_cube_tables.py)
CUBE_SCALAR = {"SMn": None}
CUBE_VECTOR = {"UV3n": None, "Dn": None}
# signed-zero probes: map name -> (outputs that were dumped); jaxdump.F:1003-1027
ZERO_PROBES = {"UVs": ("_u", "_v"), "As": ("_u", "_v"), "Bs": ("_u", "_v"), "Ds": ("_u", "_v"),
               "UV3s": ("_u", "_v"), "SMs": ("_u",)}
# kinds whose exchange may change signs in some topology (cube-sphere facets); scalar kinds must not
SIGNED_SCALAR = ("SMs",)

# The experiments (code/ layout) of M1 and their probe runs (lane A, plan Task 4: run job 27826873, dumps-on runs).
M1_PROBES = {"tutorial_barotropic_gyre": ("input", "job27826873"), "advect_xy": ("input", "job27826873"),
             "advect_xz": ("input", "job27826873"), "tutorial_baroclinic_gyre": ("input", "job27826873"),
             "global_ocean.90x40x15": ("input", "job27826873"),
             "tutorial_global_oce_optim": ("input_ad", "job27826873")}

# M3 (plan Task 30): experiments of M3 whose code/ build is exch1, with their dumps-on probe runs (lane A session 9,
# reference_runs.M3_TRIPLE); the map files are named and registered as the M1 ones (MAP_SHA256).
M3_PROBES = {"tutorial_reentrant_channel": ("input", "job27840386"),
             "vermix": ("input", "job27840385")}          # vermix lane (lane A's M3 dumps-on run)

# M4 step 3 (lane M4COL): 1D_ocean_ice_column (code/ build, exch1, one 1x1 tile) and its dumps-on probe run (lane A
# session 11, job 27856057); named and registered as the M1 ones (MAP_SHA256).
M4_PROBES = {"1D_ocean_ice_column": ("input", "job27856057"),
             # M4 step 4 (lane M4OFF): offline_exf_seaice code/ build (exch1, 2x2 tiles of 40x21), the dumps-on probe
             # run of input.thermo (lane A session 11, job 27855986)
             "offline_exf_seaice": ("input.thermo", "job27855986"),
             # M4 step 5 (lane M4LAB): lab_sea code/ build (exch1, 2x2 tiles of 10x8, OLx = OLy = 4), the dumps-on
             # probe run of input (lane A session 11, job 27855987)
             "lab_sea": ("input", "job27855987")}

# M2 (plan Task 27, M2 acceptance): the exch1 experiments of M2 whose maps the PTRACERS lane built in memory from the
# probe (test_ptracers_*.py); registered so that `python -m mitjax run` (the driver's Model default exchanger) runs
# them. Probe runs: lane A session 6 (job 27832231: latlon, advection_in_gyre; job 27832229: tracer_adjsens code_ad,
# the experiment's only build). Named and registered as the M1 ones (MAP_SHA256).
M2_PROBES = {"tutorial_global_oce_latlon": ("input", "job27832231"),
             "tutorial_advection_in_gyre": ("input", "job27832231"),
             "tutorial_tracer_adjsens": ("input_ad", "job27832229")}

# sha256 of every generated map file (scripts/make_exch_maps.py prints them); load_maps checks it. Generated
# 2026-10-01 from the probe dumps of run job 27826873 (all six reproduced bitwise by the maps).
MAP_SHA256 = {
    "tutorial_barotropic_gyre-t1_62x62_ol2x2.npz": "a62ab9bfb0b3a7a762cbce79ea6944f7bf4e0979991a78ee39bfe2e292df99d5",
    "advect_xy-t2_20x10_ol3x3.npz": "aaafbdb660bf641e003c944f1fef1cfde2a569d0cfbaf463b0b14c75772186e2",
    "advect_xz-t2_10x1_ol4x4.npz": "413bab0a483d461b2970d3b6a145c8475c4cd743b69f8128576a2f4086a5946d",
    "tutorial_baroclinic_gyre-t4_31x31_ol2x2.npz": "3426cf6cca97d9d7088bb12ad2c32a2ee35a8dbc443086af598cea9f902a2d37",
    "global_ocean.90x40x15-t36_10x10_ol3x3.npz": "1738559f9f506b4ac7b1c948b2e0174592deed3a6db73c751dff6c0f256a9d66",
    "tutorial_global_oce_optim-t4_45x20_ol2x2.npz": "4df7d028380339e31cb160ef576f34a3e60685d109ae1152e518cb8cf2fcc8f1",
    # vermix lane (M3 Task 30): written by scripts/make_exch_maps.py from M3_PROBES["vermix"] (lane A's M3 dumps-on
    # run), reproduced bitwise by the maps; not in M1_PROBES, whose layouts test_exchange.py gates with their counts
    "vermix-t1_1x1_ol2x2.npz": "5bb405394cc9498c15fbdb7f871d0559d3811a364d7ba0791fa599cefbee9d43",
    # M3 (2026-10-02, from job27840386-jdon's probe, reproduced bitwise)
    "tutorial_reentrant_channel-t4_20x10_ol4x4.npz": "bd544c394052457f6adef9821553049acd4ccf0915bce1dcb9e0f93b81336bbd",
    # M2 (2026-10-02, M2_PROBES, each probe reproduced bitwise by scripts/make_exch_maps.py)
    "tutorial_global_oce_latlon-t2_45x40_ol2x2.npz": "f9b32b04f2df2f628a71ecc1071beab2e1d76888345a078f738881fa20f54225",
    "tutorial_advection_in_gyre-t4_30x30_ol4x4.npz": "3ce4db060379db5def43087eddb046a7b126bb601cef4242a323b8a8ab647f24",
    "tutorial_tracer_adjsens-t4_45x20_ol3x3.npz": "c59edc9e61ec5f4206e9366792748670f6ea89f725c273eef5e0ae7b74ac0224",
    # M4 step 3 (2026-10-03, lane M4COL, M4_PROBES, the probe reproduced bitwise by scripts/make_exch_maps.py)
    "1D_ocean_ice_column-t1_1x1_ol2x2.npz": "75c7ce7375cde7ce7e71a28f87810e8196bfcb254dab3fbdc58b015cbdba880d",
    # M4 step 4 (2026-10-03, lane M4OFF, M4_PROBES, the probe reproduced bitwise by scripts/make_exch_maps.py)
    "offline_exf_seaice-t4_40x21_ol3x3.npz": "60681c5eb92655010dbc09bfc8948bad55a9daeae8c895541fcdd6f9a71c7995",
    # M4 step 5 (2026-10-03, lane M4LAB, M4_PROBES, the probe reproduced bitwise by scripts/make_exch_maps.py)
    "lab_sea-t4_10x8_ol4x4.npz": "e90cc8a84aefdb0687b9eff3640d1b463dff48dad54e3babef4e9e2c34be6e4a",
}


# GOADK lane (M2): builds whose tiling differs from the experiment's code/ build get their own map file, named
# <exp>-<code dir>-<layout tag>.npz (so that the M1 glob <exp>-t*.npz does not see it), made by
# scripts/make_exch_maps.py --code from the registered dumps-on probe run of that build:
# {(experiment, code dir): (input, run id)}
VARIANT_PROBES = {("global_ocean.90x40x15", "code_ad"): ("input_ad", "job27832229-jdon")}
VARIANT_MAP_SHA256 = {
    "global_ocean.90x40x15-code_ad-t4_45x20_ol3x3.npz": "c59edc9e61ec5f4206e9366792748670f6ea89f725c273eef5e0ae7b74ac0224",
}


def output_names(name):
    """Map keys of the outputs of one exchange: ("XY",) or ("UVs_u", "UVs_v")."""
    if name in SCALAR or name in CUBE_SCALAR:
        return (name,)
    if name in VECTOR or name in CUBE_VECTOR:
        return (name + "_u", name + "_v")
    raise KeyError(f"no exchange map {name!r}")


@dataclass(frozen=True)
class ExchangeMaps:
    layout: TileLayout
    maps: dict                        # output key -> (src int32 [N], comp int8 [N], sign int8 [N])
    meta: dict = field(default_factory=dict)
    # exch2 C-grid vector exchanges (EXCH2_RX2_CUBE: sums sa1*A1 + sa2*A2, not copies): kind -> (swap, passes),
    # passes = [ {1: (src, written, sa1, sa2), 2: (...)} per EXCH2_RX2_CUBE call ] (pkg/exch2/exch2_rx2_cube.py);
    # empty for exch1 layouts. Not stored in the map file: derived from the W2 topology by load_maps.
    rx2: dict = field(default_factory=dict)

    def save(self, path):
        """Write the .npz, its .sha256 and its .json (provenance); refuses existing files. Returns the sha256."""
        path = Path(path)
        side = [path, path.with_suffix(".npz.sha256"), path.with_suffix(".json")]
        for p in side:
            if p.exists():
                raise FileExistsError(f"{p} exists (map files are never overwritten)")
        arrs = {"layout": self.layout.as_array()}
        for k, (s, c, g) in sorted(self.maps.items()):
            arrs[k + ".src"], arrs[k + ".comp"], arrs[k + ".sign"] = (s.astype(np.int32), c.astype(np.int8),
                                                                      g.astype(np.int8))
        with open(path, "xb") as fh:
            np.savez_compressed(fh, **arrs)
        sha = sha256_file(path)
        side[1].write_text(f"{sha}  {path.name}\n")
        side[2].write_text(json.dumps(dict(self.meta, sha256=sha, layout=self.layout.__dict__), indent=1) + "\n")
        return sha

    @classmethod
    def load(cls, path, sha256=None):
        """Read a map file; sha256: the expected digest (checked when given)."""
        path = Path(path)
        if sha256 is not None:
            got = sha256_file(path)
            if got != sha256:
                raise ValueError(f"{path}: sha256 {got} != registered {sha256}")
        with np.load(path) as z:
            L = TileLayout.from_array(z["layout"])
            names = sorted({k.rsplit(".", 1)[0] for k in z.files if k != "layout"})
            maps = {n: (z[n + ".src"].astype(np.int32), z[n + ".comp"].astype(np.int8),
                        z[n + ".sign"].astype(np.int8)) for n in names}
        meta_p = path.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        return cls(L, maps, meta)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def map_dir():
    return paths.REFERENCE / "exch_maps"


def map_path(exp, layout):
    return map_dir() / f"{exp}-{layout.tag()}.npz"


def load_maps(exp, code=None):
    """The registered map file of an experiment (exactly one; sha256 checked against MAP_SHA256), with the exch2
    C-grid vector tables (`rx2`) when the experiment's build compiles pkg/exch2. GOADK lane: with `code` (the build's
    code directory) a build listed in VARIANT_PROBES takes its own map file (VARIANT_MAP_SHA256); other code
    directories use the experiment's map."""
    if code is not None and (exp, code) in VARIANT_PROBES:
        reg = sorted(n for n in VARIANT_MAP_SHA256 if n.startswith(f"{exp}-{code}-t"))
        if len(reg) != 1:
            raise FileNotFoundError(f"{exp}/{code}: {len(reg)} registered variant map files (VARIANT_MAP_SHA256)")
        return ExchangeMaps.load(map_dir() / reg[0], VARIANT_MAP_SHA256[reg[0]])
    hits = sorted(map_dir().glob(f"{exp}-t*.npz"))
    reg = [p for p in hits if p.name in MAP_SHA256]
    if len(reg) != 1:
        raise FileNotFoundError(f"{exp}: {len(reg)} registered map files in {map_dir()} (files: "
                                f"{[p.name for p in hits]}; registered: {sorted(MAP_SHA256)})")
    m = ExchangeMaps.load(reg[0], MAP_SHA256[reg[0].name])
    if exp in M1_PROBES and uses_exch2(exp, M1_PROBES[exp][0]):
        m = replace(m, rx2=exch2_vector_tables(exp, M1_PROBES[exp][0], m.layout))
    return m


# exch2 vector exchanges that go through EXCH2_RX2_CUBE (W2_USE_R1_ONLY undefined, pkg/exch2/W2_OPTIONS.h:13):
# EXCH_UV_XY_RL and EXCH_UV_XYZ_RL (eesupp/src/exch_uv_xy_rx.template:58, exch_uv_xyz_rx.template:59) and
# EXCH_UV_3D_RL (exch_uv_3d_rx.template:60) call EXCH2_UV_3D_RX with the caller's withSigns; EXCH_UV_DGRID_3D_RL
# calls EXCH2_UV_DGRID_3D_RX (exch_uv_dgrid_3d_rx.template:61) -> EXCH2_UV_3D_RX( vPhi, uPhi, .FALSE. ). The A- and
# B-grid exchanges use EXCH2_RX1_CUBE (copies) and keep their maps.
RX2_KINDS = {"UVs": ("uv3d", True), "UVn": ("uv3d", False), "UV3s": ("uv3d", True), "Ds": ("dgrid", None)}
_RX2_CACHE = {}


@functools.lru_cache(maxsize=None)
def uses_exch2(exp, inp):
    """The experiment's build compiles pkg/exch2 (its package list, mitjax/config/packages.py)."""
    from mitjax.config import cpp_options, namelists, packages
    mods = [paths.UPSTREAM / "verification" / exp / namelists.code_dir_for(inp)]
    tc = cpp_options.default_toolchain()
    return "exch2" in packages.package_set(paths.UPSTREAM, mods, have_netcdf=tc.have_netcdf).packages


def exch2_vector_tables(exp, inp, layout):
    """{kind: (swap, passes)} of the C-grid vector exchanges from the W2 set-up (mitjax/pkg/exch2/w2_eeboot.py) of
    the experiment; swap: array1 is v (EXCH2_UV_DGRID_3D_RX)."""
    key = (exp, inp, layout)
    if key not in _RX2_CACHE:
        from mitjax.config.params import load
        from mitjax.pkg.exch2.exch2_rx2_cube import exch2_uv_3d_rx_tables, exch2_uv_dgrid_3d_rx_tables
        from mitjax.pkg.exch2.w2_eeboot import w2_eeboot
        from mitjax.pkg.exch2.w2_readparms import use_cubed_sphere_exchange
        e = load(exp, inp)
        w2, _ = w2_eeboot(e)
        cube = use_cubed_sphere_exchange(e)
        out = {}
        for kind, (routine, withSigns) in RX2_KINDS.items():
            if routine == "uv3d":
                out[kind] = (False, exch2_uv_3d_rx_tables(w2, layout, withSigns, cube))
            else:
                out[kind] = (True, exch2_uv_dgrid_3d_rx_tables(w2, layout, cube))
        _RX2_CACHE[key] = out
    return _RX2_CACHE[key]


def load_cube_maps(exp, inp):
    """ExchangeMaps of a cubed-sphere (or any pkg/exch2) build from its W2 topology alone: every exchange kind
    replayed from the Fortran loops (mitjax/pkg/exch2/exch2_cube_tables.py), no probe file. Copy kinds go into
    `maps`; the C-grid kinds (UVs, UVn, UV3s, UV3n, Ds, Dn) into `rx2` as (swap, passes, post). Gated against the
    gfortran replay of every exchange (reference/replay_cube, test_cube_exchange.py) and, once lane A's dumps exist,
    against the jaxdump probes."""
    key = ("cube", exp, inp)
    if key not in _RX2_CACHE:
        from mitjax.config.params import load
        from mitjax.pkg.exch2.exch2_cube_tables import cube_exchange_programs
        from mitjax.pkg.exch2.w2_eeboot import w2_eeboot
        from mitjax.pkg.exch2.w2_readparms import use_cubed_sphere_exchange
        e = load(exp, inp)
        if not e.cfg.cpp.ALLOW_EXCH2:
            raise ValueError(f"{exp}: the build does not compile pkg/exch2")
        w2, _ = w2_eeboot(e)
        sz = e.cfg.size
        L = TileLayout(sz.sNx, sz.sNy, sz.OLx, sz.OLy, w2.exch2_nTiles)
        maps, rx2 = cube_exchange_programs(w2, L, use_cubed_sphere_exchange(e), e.cfg.cpp)
        _RX2_CACHE[key] = ExchangeMaps(L, maps, {"source": "W2 topology replay (exch2_cube_tables.py)",
                                                 "experiment": exp, "input": inp}, rx2)
    return _RX2_CACHE[key]


# ---------------------------------------------------------------------------------------------------------------------
# building the maps from a probe dump


def probe_layout(ds, it, stage=STAGE):
    """TileLayout of a probe dump set; tiles must be numbered 1..nTiles and share one shape."""
    ntl = int(ds.scalar(it, stage, "xNtiles"))
    if sorted(ds.tiles_info) != list(range(1, ntl + 1)):
        raise ValueError(f"tiles {sorted(ds.tiles_info)} are not 1..{ntl}")
    shapes = {v[3:] for v in ds.tiles_info.values()}
    if len(shapes) != 1:
        raise ValueError(f"tiles of different shapes {shapes}")
    snx, sny, olx, oly = shapes.pop()
    return TileLayout(snx, sny, olx, oly, ntl)


def decode_field(values, cbase, layout):
    """Exchanged probe values [nTiles, ny, nx] -> (src, comp, sign) int arrays [N] (comp 1/2, sign +-1)."""
    L = layout
    v = np.asarray(values, np.float64)
    if v.shape != L.shape2d:
        raise ValueError(f"probe field shape {v.shape} != {L.shape2d}")
    d = probe_decode(v, cbase, L.nx, L.ny)
    src = ((d["tile"] - 1) * L.ny + d["ja"]) * L.nx + d["ia"]
    if np.any(d["tile"] < 1) or np.any(d["tile"] > L.nTiles):
        raise ValueError("probe source tile outside 1..nTiles")
    return src.reshape(-1), d["comp"].reshape(-1), d["sign"].reshape(-1)


def finish_map(key, src, comp, sign, layout):
    """(src, comp, sign) of the decoded probe -> the map (comp 0 where the point kept its own code)."""
    L = layout
    own = np.arange(L.npoints)
    own_comp = 2 if key.endswith("_v") else 1
    keep = (src == own) & (comp == own_comp) & (sign > 0)
    interior = np.broadcast_to(L.interior(), L.shape2d).reshape(-1)
    if np.any(~keep & interior):
        raise ValueError(f"{key}: the exchange changed an interior point")
    return (src.astype(np.int32), np.where(keep, 0, comp).astype(np.int8), np.where(keep, 1, sign).astype(np.int8))


def build_maps(ds, it=None, stage=STAGE, meta=None):
    """ExchangeMaps from a probe dump set (mitjax.io.dump.DumpSet with stage X00_exch_probe)."""
    it = ds.iterations()[0] if it is None else it
    L = probe_layout(ds, it, stage)
    cbase = ds.scalar(it, stage, "xCbase")

    def fld(name):
        a = ds.field(it, stage, name)
        if a.shape[1] != 1:
            raise ValueError(f"{name}: probe records have {a.shape[1]} levels, expected 1")
        return a[:, 0]

    maps = {}
    for name, f in SCALAR.items():
        s, c, g = decode_field(fld(f), cbase, L)
        if np.any(c != 1) or (name not in SIGNED_SCALAR and np.any(g < 0)):
            raise ValueError(f"scalar exchange {name} took a v-component or a sign")
        maps[name] = finish_map(name, s, c, g, L)
    for name, (fu, fv) in VECTOR.items():
        for key, f in ((name + "_u", fu), (name + "_v", fv)):
            s, c, g = decode_field(fld(f), cbase, L)
            maps[key] = finish_map(key, s, c, g, L)
    return ExchangeMaps(L, maps, dict(meta or {}, iteration=int(it), stage=stage, cbase=float(cbase)))


def probe_inputs(layout, cbase):
    """The coded fields JAXDUMP_PROBE_FILL writes before each exchange: (pu, pv) [nTiles, ny, nx] float64."""
    idx = np.arange(layout.npoints, dtype=np.float64).reshape(layout.shape2d) + 1.0
    return 1.0 * cbase + idx, 2.0 * cbase + idx


def summary(m):
    """{key: (written halo points, from the other component, sign flips, halo points not written)}."""
    L = m.layout
    halo = ~np.broadcast_to(L.interior(), L.shape2d).reshape(-1)
    out = {}
    for k, (s, c, g) in sorted(m.maps.items()):
        other = {"_u": 2, "_v": 1}.get(k[-2:], 0)
        out[k] = (int(np.sum((c > 0) & halo)), int(np.sum(c == other)) if other else 0, int(np.sum(g < 0)),
                  int(np.sum((c == 0) & halo)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# run time: the maps replayed from the exchange code (docs plan 20261006 Task 3)

CACHE_VERSION = 1
_REPLAY_MEMO = {}


def _canonical(v):
    """Bytes of a W2 set-up value that depend on its content only (no addresses, no file paths)."""
    import dataclasses
    if isinstance(v, np.ndarray):
        return f"nd{v.dtype.str}{v.shape}".encode() + np.ascontiguousarray(v).tobytes()
    if hasattr(v, "data") and hasattr(v, "bounds"):                       # w2_exch2_h.FIntArray
        return f"fa{v.bounds}".encode() + _canonical(np.asarray(v.data))
    if dataclasses.is_dataclass(v):                                       # Size: its fields, not SIZE.h's path
        return b"dc" + b"|".join(f.name.encode() + b"=" + _canonical(getattr(v, f.name))
                                 for f in dataclasses.fields(v) if f.name != "source")
    if v is None or isinstance(v, (bool, int, float, str, np.generic)):
        return f"{type(v).__name__}:{v!r}".encode()
    if isinstance(v, (tuple, list)):
        return b"(" + b",".join(_canonical(x) for x in v) + b")"
    raise TypeError(f"no canonical form for {type(v).__name__} in the W2 set-up (the cache key would not be "
                    f"reproducible)")


def _topology_digest(w2):
    """sha256 over every attribute of the W2 set-up (W2_EXCH2_TOPOLOGY.h, W2_EXCH2_PARAMS.h, the process map): what
    the exch2 replay may read, by content."""
    h = hashlib.sha256()
    for name in sorted(vars(w2)):
        h.update(name.encode() + b"\0" + _canonical(getattr(w2, name)) + b"\1")
    return h.hexdigest()


def replay_key(exp):
    """(family, layout, key): the build's exchange family ("exch1", "exch0" = DISCONNECTED_TILES, "exch2",
    "exch2-cube"), its TileLayout and the sha256 of everything the replay reads (layout + topology, no names)."""
    from mitjax.pkg.exch2.w2_readparms import use_cubed_sphere_exchange
    cfg, sz = exp.cfg, exp.cfg.size
    cube = bool(use_cubed_sphere_exchange(exp))
    desc = {"version": CACHE_VERSION, "size": [sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.nSx, sz.nSy, sz.nPx, sz.nPy,
                                               sz.Nx, sz.Ny], "useCubedSphereExchange": cube}
    if cfg.cpp.ALLOW_EXCH2:
        from mitjax.pkg.exch2.w2_eeboot import w2_eeboot
        w2 = w2_eeboot(exp)[0]
        family = "exch2-cube" if cube else "exch2"
        L = TileLayout(sz.sNx, sz.sNy, sz.OLx, sz.OLy, w2.exch2_nTiles)
        desc["w2"] = _topology_digest(w2)
        for opt in ("W2_FILL_NULL_REGIONS", "W2_USE_R1_ONLY"):      # as exch2_cube_tables._check_w2_options
            try:
                desc[opt] = bool(getattr(cfg.cpp, opt))
            except KeyError:
                desc[opt] = False
    else:
        w2 = None
        disc = "DISCONNECTED_TILES" in cfg.cpp.known and bool(cfg.cpp.DISCONNECTED_TILES)
        family = "exch0" if disc else "exch1"
        L = TileLayout(sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.nSx * sz.nSy * sz.nPx * sz.nPy)
    desc["family"] = family
    desc["layout"] = [int(x) for x in L.as_array()]
    key = hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()
    return family, L, key, w2, cube


def cache_path(family, layout, key):
    return paths.CACHE / "exch_maps" / f"{family}-{layout.tag()}-{key[:20]}.npz"


def _read_cache(path, layout, key):
    """The cached copy maps, after checking the stored key and layout (a mismatch is refused, never repaired)."""
    with np.load(path) as z:
        got_key = bytes(z["cache_key"]).decode()
        got_L = TileLayout.from_array(z["layout"])
        if got_key != key or got_L != layout:
            raise ValueError(f"{path}: exchange-map cache refused: stored key {got_key[:20]} / layout {got_L} != "
                             f"computed {key[:20]} / {layout} (a file from another layout or topology)")
        names = sorted({k.rsplit(".", 1)[0] for k in z.files if k not in ("layout", "cache_key")})
        return {n: (z[n + ".src"].astype(np.int32), z[n + ".comp"].astype(np.int8), z[n + ".sign"].astype(np.int8))
                for n in names}


def _write_cache(path, layout, key, maps):
    """Write once, atomically (a new temporary name renamed into place; an existing file is kept, never replaced)."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    arrs = {"layout": layout.as_array(), "cache_key": np.frombuffer(key.encode(), np.uint8)}
    for k, (a, c, g) in sorted(maps.items()):
        arrs[k + ".src"], arrs[k + ".comp"], arrs[k + ".sign"] = a.astype(np.int32), c.astype(np.int8), g.astype(np.int8)
    part = path.with_name(f"{path.name}.part-{os.getpid()}")
    with open(part, "xb") as fh:
        np.savez_compressed(fh, **arrs)
    if not path.exists():
        os.rename(part, path)


def _replay_copy_maps(exp, family, L, w2, cube):
    cfg = exp.cfg
    if family in ("exch1", "exch0"):
        from mitjax.eesupp.exch1_tables import exch1_exchange_maps
        return exch1_exchange_maps(cfg.size, L, cfg.cpp, cube)
    from mitjax.pkg.exch2.exch2_cube_tables import cube_copy_maps
    maps = cube_copy_maps(w2, L, cube, cfg.cpp)
    if family == "exch2":
        # a single-facet (non-cube) exch2 build keeps the probe's kinds: the scalar and A-/B-grid copy kinds (the
        # C-grid kinds go through rx2, as load_maps attaches them)
        keep = set(SCALAR) | {k + s for k in VECTOR if k not in RX2_KINDS for s in ("_u", "_v")}
        maps = {k: v for k, v in maps.items() if k in keep}
    return maps


def exchange_maps(exp, use_cache=True):
    """ExchangeMaps of the build of `exp` (mitjax.config.params.Experiment), replayed from the exchange code; the copy
    maps come from $MJX_CACHE when a file with this layout + topology key exists (checked), else they are replayed
    and written there. The exch2 C-grid tables (`rx2`) are computed from the W2 set-up at every load (as load_maps
    and load_cube_maps did)."""
    family, L, key, w2, cube = replay_key(exp)
    if key in _REPLAY_MEMO:
        return _REPLAY_MEMO[key]
    path = cache_path(family, L, key)
    maps = _read_cache(path, L, key) if (use_cache and path.exists()) else None
    source = "cache" if maps is not None else "replay"
    if maps is None:
        maps = _replay_copy_maps(exp, family, L, w2, cube)
        if use_cache:
            _write_cache(path, L, key, maps)
    rx2 = {}
    if family == "exch2-cube":
        from mitjax.pkg.exch2.exch2_cube_tables import cube_rx2_programs
        rx2 = cube_rx2_programs(w2, L, cube, exp.cfg.cpp)
    elif family == "exch2":
        from mitjax.pkg.exch2.exch2_rx2_cube import exch2_uv_3d_rx_tables, exch2_uv_dgrid_3d_rx_tables
        for kind, (routine, withSigns) in RX2_KINDS.items():       # as exch2_vector_tables
            if routine == "uv3d":
                rx2[kind] = (False, exch2_uv_3d_rx_tables(w2, L, withSigns, cube))
            else:
                rx2[kind] = (True, exch2_uv_dgrid_3d_rx_tables(w2, L, cube))
    m = ExchangeMaps(L, maps, {"source": source, "family": family, "cache_key": key,
                               "cache_file": str(path) if use_cache else None}, rx2)
    if use_cache:
        _REPLAY_MEMO[key] = m
    return m
