"""The thin mitjax API (docs plan 20261006 Task 6): five calls and `exp.results()`, wrappers over the gated drivers.
The API never changes what a kernel computes; every number comes from the driver it names.

    import mitjax
    exp = mitjax.load("<MITgcm>/verification/tutorial_barotropic_gyre", variant="input")
    run = exp.run(out="runs/gyre")                  # drivers/run.py: output.txt (cg2d, %MON lines), pickups
    rep = mitjax.compare(run, exp.results())        # testreport's digits (mitjax/testreport_jax.py)
    ad = mitjax.load(".../1D_ocean_ice_column", variant="input_ad")
    g = ad.gradient(out="runs/col-grad")            # dfc/dxx of the run's own cost and control, adxx_* files
    chk = ad.grdchk(out="runs/col-chk")             # GRDCHK_MAIN's FD vs the adjoint at data.grdchk's points
    mitjax.compare(chk, ad.results())               # testreport's adm digits (admGrd, admCst, admFwd)

What each call wraps:
  load      mitjax.drivers.run.load_experiment (config.params.load: data* namelists, the preprocessed *_OPTIONS.h)
  run       make_rundir + drivers.model.Model + drivers.run.forward, written as drivers/run.run writes it
  results   the experiment's own results/ references (testreport's names, mitjax.testreport_jax.reference_name)
  compare   testreport's comparison (testreport_jax.resolve_checklists / testoutput_run / formatresults)
  gradient  drivers/adjoint_run.GenarrAdjoint (xx_genarr2d/3d) or Adjoint (xx_gentim2d, one program per record) on
            devices = 1; drivers/sharded_grad (jit(shard_map) over a TileSharding of the replayed exchange maps) on
            devices > 1. mode = "run": the run's own data.autodiff switches as backward-only hooks
            (drivers/ad_switches.with_run_switches) and the CG2D derivative as the run's cg2dFullAdjoint says
            (ad/modes.py "run"); mode = "exact": no backward-only switch and the exact CG2D derivative. The SEAICE_LSR
            derivative is A1 where the build defines SEAICE_LSR_ADJOINT_ITER (ad/modes.py "run"), or everywhere with
            lsr_derivative = "sweeps" (plan decision 17); without either a gradient through the LSR raises.
  grdchk    GRDCHK_MAIN: the adjoint gradient and the central (or one-sided) differences of the cost at data.grdchk's
            points (drivers/grdchk.py), printed as output_adm.txt prints them (GRDCHK_PRINT)

Arrays are returned as numpy arrays in mitjax's storage layout [tile, (k,) j, i] with halos (Fortran index i at
storage i - 1 + OLx); `Run.interior(name)` drops the halos, `Run.global_field(name)` assembles the tiles (TileMap).
Unsupported cases stop with the drivers' own errors (a missing file, an unported option or routine, a control the
drivers do not differentiate).

Additions of lane APIF (docs plan 20261006 S5):
  run(out, devices=N)    the forward on N JAX devices: drivers/run.forward(sharding=TileSharding(...)), the step
                         under jit(shard_map) as the P=N whole-run gates run it; the same output.txt, pickups and fields
  Run.fields             the ocean State and, with pkg/seaice, the sea-ice state by its SEAICE.h names (AREA, HEFF,
                         HSNOW, UICE, VICE, ...)
  Run.global_field(name) one global numpy array: exch1 [(k,) Ny, Nx] (tiles bi + (bj-1)*nSx at ((bj-1)*sNy,
                         (bi-1)*sNx)); pkg/exch2 [face, (k,) j, i] (W2 tile tN at exch2_tBasey/x(tN) of face
                         exch2_myFace(tN); points of no tile, e.g. blank tiles, NaN)
  grid()                 the GRID.h arrays of INI_GRID .. INI_CORI by Fortran name (xC, yC, xG, yG, dxC, dyC, dxG, dyG,
                         rA, rAw, rAs, rAz, hFacC/W/S, maskC/W/S, R_low, Ro_surf, rC, rF, drC, drF, ...), without a
                         run directory (GridArrays: a dict with interior() and global_field())
  gradient / grdchk      cg2d_derivative = None (the mode's choice) | "run" | "exact" (mitjax/ad/modes.py); the
                         sea-ice LSR derivative defaults to A1 (lsr_derivative=None: "sweeps" where the build does not
                         tape the sweeps itself, plan decisions 14 and 17); jax.clear_caches() between programs
  devices > the JAX device count: DeviceCountError (a ValueError) before any run directory is made
XLA flags: every call sets them once with mitjax.xla_flags.set_api_xla_flags (the gate flags, --xla_cpu_max_isa=AVX
only on x86-64, the user's own XLA_FLAGS entries win), effective only before JAX's backend starts.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["load", "compare", "Experiment", "Run", "Results", "Gradient", "Grdchk", "GridArrays", "TileMap",
           "DeviceCountError"]

MODES = ("run", "exact")
CG2D_DERIVATIVES = (None, "run", "exact")


def _xla_flags():
    from mitjax.xla_flags import set_api_xla_flags
    set_api_xla_flags()


class DeviceCountError(ValueError, RuntimeError):
    """devices= asks for more JAX devices than there are (a ValueError; also a RuntimeError, which the API raised for
    this case before lane APIF)."""


def _check_devices(devices):
    """int(devices), refused (DeviceCountError) when < 1 or above len(jax.devices()); called before any run
    directory is made."""
    import jax
    n = int(devices)
    have = len(jax.devices())
    if n < 1 or n > have:
        raise DeviceCountError(f"devices={n}: JAX has {have} device(s) ({jax.default_backend()}); on CPU, "
                               "XLA_FLAGS=--xla_force_host_platform_device_count=N set before JAX starts gives N")
    return n


@dataclass(frozen=True)
class TileMap:
    """Where each storage tile's interior goes in a global field (module docstring, Run.global_field).
    kind "exch1": shape (Ny, Nx), origins ((None, y0, x0), ...) per tile in storage order; "exch2": shape
    (nFaces, ny, nx), origins ((face0, y0, x0), ...) with face0 = exch2_myFace(tN) - 1."""
    kind: str
    sNx: int
    sNy: int
    OLx: int
    OLy: int
    shape: tuple
    origins: tuple

    @classmethod
    def of(cls, cfg, w2=None):
        """The map of the build `cfg` (SIZE.h); `w2` the W2 set-up (pkg/exch2 builds, w2_eeboot)."""
        sz = cfg.size
        if sz.nPx * sz.nPy != 1:
            raise NotImplementedError(f"global_field: SIZE.h nPx = {sz.nPx}, nPy = {sz.nPy} (one process only)")
        base = dict(sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx, OLy=sz.OLy)
        if not cfg.cpp.ALLOW_EXCH2:
            origins = tuple((None, (bj - 1) * sz.sNy, (bi - 1) * sz.sNx)
                            for bj in range(1, sz.nSy + 1) for bi in range(1, sz.nSx + 1))
            return cls("exch1", shape=(sz.sNy * sz.nSy, sz.sNx * sz.nSx), origins=origins, **base)
        if w2 is None:
            from mitjax.pkg.exch2.w2_eeboot import w2_eeboot
            raise ValueError(f"TileMap.of: a pkg/exch2 build needs its W2 set-up ({w2_eeboot.__module__})")
        dims = {(int(w2.facet_dims[2 * f - 1]), int(w2.facet_dims[2 * f])) for f in range(1, w2.nFacets + 1)}
        if len(dims) != 1:
            raise NotImplementedError(f"global_field: facets of different sizes {sorted(dims)} (one [face, j, i] "
                                      "array needs equal facets)")
        nx, ny = dims.pop()
        origins = []
        for bj in range(1, sz.nSy + 1):
            for bi in range(1, sz.nSx + 1):
                tN = w2.W2_myTileList[bi, bj]
                origins.append((int(w2.exch2_myFace[tN]) - 1, int(w2.exch2_tBasey[tN]), int(w2.exch2_tBasex[tN])))
        return cls("exch2", shape=(int(w2.nFacets), ny, nx), origins=tuple(origins), **base)

    def interior(self, a):
        return np.asarray(a)[..., self.OLy:self.OLy + self.sNy, self.OLx:self.OLx + self.sNx]

    def global_field(self, a):
        """The global array of a tiled storage array [tile, (k,) sNy+2OLy, sNx+2OLx]; an untiled one (rC, drF, ...)
        unchanged."""
        a = np.asarray(a)
        T, ny, nx = len(self.origins), self.sNy + 2 * self.OLy, self.sNx + 2 * self.OLx
        if a.ndim < 3 or a.shape[0] != T or a.shape[-2:] != (ny, nx):
            if a.ndim >= 1 and a.shape[0] == T and a.ndim >= 3:
                raise ValueError(f"global_field: shape {a.shape} is not [tile, ..., {ny}, {nx}]")
            return a
        v = self.interior(a)
        mid = a.shape[1:-2]
        if self.kind == "exch1":
            out = np.zeros(mid + tuple(self.shape), a.dtype)
            for t, (_, y0, x0) in enumerate(self.origins):
                out[..., y0:y0 + self.sNy, x0:x0 + self.sNx] = v[t]
            return out
        fill = np.nan if np.issubdtype(a.dtype, np.floating) else 0
        out = np.full((self.shape[0],) + mid + tuple(self.shape[1:]), fill, dtype=a.dtype)
        for t, (f, y0, x0) in enumerate(self.origins):
            out[f, ..., y0:y0 + self.sNy, x0:x0 + self.sNx] = v[t]
        return out


class GridArrays(dict):
    """{Fortran name: numpy array} of the GRID.h fields (storage layout; vertical profiles 1-D), with `interior(name)`
    and `global_field(name)` as Run's."""

    def __init__(self, arrays, tilemap):
        super().__init__(arrays)
        self.tilemap = tilemap

    def interior(self, name):
        return self.tilemap.interior(self[name])

    def global_field(self, name):
        return self.tilemap.global_field(self[name])


def load(exp_dir, variant="input"):
    """The experiment in `exp_dir` (anywhere on disk) with input directory `variant` (input, input.X, input_ad, ...)."""
    return Experiment(exp_dir, variant)


@dataclass
class Run:
    """A forward run: `output` is <out>/rundir/output.txt; `fields` the final state's fields (numpy, storage layout):
    the ocean State by its mitjax names and, with pkg/seaice, the sea-ice fields by their SEAICE.h names (AREA, HEFF,
    HSNOW, UICE, VICE, ...); `pickups` the pickup files written; `devices` the JAX devices the run used."""
    out: Path
    rundir: Path
    output: Path
    variant: str
    fields: dict
    pickups: list
    size: object = None
    kind: str = "fwd"
    devices: int = 1
    tilemap: TileMap = field(default=None, repr=False)

    def interior(self, name):
        """fields[name] without halos: [tile, (k,) sNy, sNx]."""
        sz, a = self.size, self.fields[name]
        return a[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx]

    def global_field(self, name):
        """fields[name] as one global array: exch1 [(k,) Ny, Nx] in the Fortran's tile order (bi + (bj-1)*nSx at
        ((bj-1)*sNy, (bi-1)*sNx)); pkg/exch2 [face, (k,) j, i] (TileMap)."""
        return self.tilemap.global_field(self.fields[name])


@dataclass
class Results:
    """The experiment's own testreport references for `variant` (None where results/ has none)."""
    exp_dir: Path
    variant: str
    output: Path = None             # forward: results/output[.X].txt
    output_adm: Path = None         # adjoint: results/output_adm[.X].txt
    output_tlm: Path = None         # tangent linear: results/output_tlm[.X].txt

    def reference(self, kind):
        ref = {"fwd": self.output, "adm": self.output_adm, "tlm": self.output_tlm}[kind]
        if ref is None:
            raise FileNotFoundError(f"{self.exp_dir}/results: no {kind} reference for {self.variant} "
                                    "(testreport's name: mitjax.testreport_jax.reference_name)")
        return ref


@dataclass
class Gradient:
    """dfc/dxx of the run's cost w.r.t. one control: `fc` the cost, `adxx` {record: numpy [tile, (k,) j, i]}, `files`
    the adxx_* files written in <out>/rundir."""
    out: Path
    rundir: Path
    control: str
    mode: str
    devices: int
    fc: float
    adxx: dict
    files: list


@dataclass
class Grdchk:
    """GRDCHK_MAIN's results: `output` (<out>/rundir/output.txt: the grdchk lines), `checks` per point (i, j, k, bi,
    bj, rec, fcref, fcpertplus, fcpertminus, gfd, adjoint_gradient, ierr)."""
    out: Path
    rundir: Path
    output: Path
    control: str
    checks: list
    lines: list = field(repr=False, default_factory=list)
    kind: str = "adm"


class Experiment:
    """An experiment directory with one input variant: `config` is the drivers' configuration
    (mitjax.config.params.Experiment); `model(out)` the drivers' Model in a new run directory."""

    def __init__(self, exp_dir, variant="input"):
        from mitjax.drivers.run import load_experiment
        self.exp_dir = Path(exp_dir).resolve()
        self.variant = variant
        # a missing input directory: FileNotFoundError "no input dir ..." of mitjax.config.params.check_input_dir
        self.config = load_experiment(self.exp_dir, variant)

    def __repr__(self):
        return f"mitjax.Experiment({str(self.exp_dir)!r}, variant={self.variant!r})"

    def model(self, out):
        """drivers.model.Model of this experiment in the new run directory <out>/rundir (make_rundir: testreport's
        linkdata, prepare_run where the variant has one)."""
        from mitjax.drivers.model import Model
        return Model(self.config, self._rundir(out))

    def _rundir(self, out):
        """drivers.run.make_rundir without make_rundir.main's `RUNDIR <path>` line on stdout (prepare_run variants);
        every other line it prints (REFUSED ...) is printed."""
        import contextlib
        import io

        from mitjax.drivers.run import make_rundir
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                return make_rundir(self.exp_dir, self.variant, out)
        finally:
            kept = [ln for ln in buf.getvalue().splitlines() if not ln.startswith("RUNDIR ")]
            if kept:
                print("\n".join(kept))

    # ------------------------------------------------------------------------------------------------ forward run

    def run(self, out, devices=1):
        """The forward run, written as drivers/run.run writes it: <out>/rundir/output.txt (cg2d lines, %MON blocks,
        cost lines) and the pickups the run's schedule asks for. devices > 1: the time loop split over that many JAX
        devices by tiles (drivers/run.forward(sharding=...)); the same output and fields bit for bit (gate:
        mitjax/tests/test_api_followups.py)."""
        from mitjax.drivers.run import forward
        _xla_flags()
        devices = _check_devices(devices)
        m = self.model(out)
        sh = None
        if devices > 1:
            from mitjax.eesupp.exch_maps import exchange_maps
            from mitjax.eesupp.shard import TileSharding
            sh = TileSharding(m.exch_maps if m.exch_maps is not None else exchange_maps(m.exp), devices)
        res = forward(m, sharding=sh)
        path = Path(m.rundir) / "output.txt"
        with open(path, "x") as fh:
            fh.write("\n".join(res.records) + "\n")
        fields = run_fields(res.carry)
        return Run(Path(out), Path(m.rundir), path, self.variant, fields, list(res.pickups), m.cfg.size,
                   devices=devices, tilemap=TileMap.of(m.cfg, getattr(m, "w2", None)))

    def grid(self):
        """The GRID.h arrays of INITIALISE_FIXED's grid part (INI_GRID, SET_GRID_FACTORS, INI_DEPTHS, INI_MASKS_ETC,
        INI_CORI: drivers/model.initialise_fixed_grid, the Model's own builder) by Fortran name, numpy, without a run
        directory: the input files are read where testreport's linkdata finds them (mitjax.config.namelists.
        link_sources); a variant with a prepare_run (files linked from another experiment) reads them from a prepared
        directory under $MJX_CACHE/api_grid/ made once and reused. Returns GridArrays."""
        from mitjax.drivers.model import initialise_fixed_grid
        from mitjax.eesupp.exch_maps import exchange_maps
        from mitjax.eesupp.exchange import Exchanger
        from mitjax.farray import FArray
        from mitjax.model.src.ini_parms import ini_parms
        _xla_flags()
        exp, cfg = self.config, self.config.cfg
        w2 = exch2 = e2io = None
        if cfg.cpp.ALLOW_EXCH2:                                    # as drivers.model.Model.__init__
            from mitjax.pkg.exch2.w2_eeboot import exch2_topology, w2_eeboot
            w2 = w2_eeboot(exp)[0]
            exch2 = exch2_topology(w2)
            if w2.W2_useE2ioLayOut:
                from mitjax.io.mds import E2ioLayout
                e2io = E2ioLayout.from_w2(w2, cfg.size)
        prm = ini_parms(exp, exch2)
        rw = _input_rw(self, prm.init.readBinaryPrec, cfg.size, e2io=e2io)
        g = initialise_fixed_grid(exp, prm.grid, ex=Exchanger(exchange_maps(exp)), rw=rw)
        arrays = {n: np.asarray(getattr(g, n).data) for n in g.names() if isinstance(getattr(g, n), FArray)}
        return GridArrays(arrays, TileMap.of(cfg, w2))

    def results(self):
        """The experiment's results/ references of this variant (testreport's file names; a .gz copy where only that
        exists, as testreport reads it)."""
        from mitjax import testreport_jax as T
        out = {}
        for kind in ("fwd", "adm", "tlm"):
            try:
                name = T.reference_name(self.variant, T.KIND_NAMES[kind])
            except ValueError:
                continue
            p = self.exp_dir / "results" / name
            if not p.is_file() and Path(str(p) + ".gz").is_file():
                p = Path(str(p) + ".gz")
            if p.is_file():
                out[{"fwd": "output", "adm": "output_adm", "tlm": "output_tlm"}[kind]] = p
        return Results(self.exp_dir, self.variant, **out)

    # ------------------------------------------------------------------------------------------------- gradients

    def _mode_arrays(self, m, model, mode, lsr_derivative, cg2d_derivative=None):
        """The Model arrays of `mode` (module docstring); the forward of every result is the forward of `model`.
        lsr_derivative None: "sweeps" (A1) where the build does not tape the LSR sweeps itself (no
        SEAICE_LSR_ADJOINT_ITER), the build's own choice otherwise (plan decisions 14, 17; lane APIF).
        cg2d_derivative None: the mode's (run: the run's cg2dFullAdjoint, exact: "exact"); "run" / "exact" override it
        (mitjax/ad/modes.with_cg2d_derivative)."""
        from mitjax.ad.modes import lsr_derivative as lsr_choice
        from mitjax.ad.modes import with_cg2d_derivative, with_lsr_derivative
        from mitjax.drivers.ad_switches import with_run_switches
        if mode not in MODES:
            raise ValueError(f"gradient mode {mode!r}: one of {MODES}")
        if cg2d_derivative not in CG2D_DERIVATIVES:
            raise ValueError(f"cg2d_derivative {cg2d_derivative!r}: one of {CG2D_DERIVATIVES}")
        if lsr_derivative is not None:
            if "sp" not in model.pkc:
                raise ValueError("lsr_derivative: the run has no pkg/seaice (no SEAICE_LSR)")
            model = model.replace(pkc={**model.pkc, "sp": with_lsr_derivative(model.pkc["sp"], lsr_derivative)})
        elif model.pkc.get("sp") is not None and _lsr_default_applies(m, model.pkc["sp"], lsr_choice):
            model = model.replace(pkc={**model.pkc, "sp": with_lsr_derivative(model.pkc["sp"], "sweeps")})
        if mode == "run":
            model = with_run_switches(m, model)
        else:
            model = model.replace(cg2d_params=with_cg2d_derivative(model.cg2d_params, "exact"))
        if cg2d_derivative is not None:
            model = model.replace(cg2d_params=with_cg2d_derivative(model.cg2d_params, cg2d_derivative))
        return model

    def _control(self, m, control):
        from mitjax.drivers.grdchk import control_by_name, grdchk_settings
        if not (m.cfg.cpp.ALLOW_CTRL and m.cfg.use_flag("useCTRL")):
            raise FileNotFoundError(f"{self.exp_dir}/{self.variant}: no control vector (data.ctrl, read by "
                                    "CTRL_READPARMS, needs pkg/ctrl compiled and useCTRL in data.pkg)")
        if control is None:
            control = grdchk_settings(m.exp)["grdchkvarname"]
        return control_by_name(m, control)

    def _value_and_grad(self, m, ctl, mode, devices, lsr_derivative, records, cg2d_derivative=None):
        """{record: (fc, gradient numpy)} of the control `ctl` (see gradient); jax.clear_caches() after each program
        (as the gentim2d loop always did), so that successive calls do not keep every compiled program."""
        import jax

        from mitjax.drivers import adjoint_run as AR
        if devices > 1:
            from mitjax.drivers import sharded_grad as SG
            from mitjax.eesupp.exch_maps import exchange_maps
            from mitjax.eesupp.shard import TileSharding
            sh = TileSharding(exchange_maps(m.exp), devices)
        out = {}
        if ctl.kind == "genarr":
            a = AR.GenarrAdjoint(m, key=ctl.key)
            model = self._mode_arrays(m, a.model, mode, lsr_derivative, cg2d_derivative)
            if devices == 1:
                fc, g = a.value_and_grad(model=model)
                out[1] = (float(fc), np.asarray(g))
            else:
                f = a.sharded_value_and_grad_fn(sh, model)
                fc, g = f(sh.put_tree(a.theta_farray()), SG.place_model(sh, model), sh.put_tree(a.st0), a.xs,
                          SG.seed(1.0))
                out[1] = (float(fc), np.asarray(g.data)[:sh.layout.nTiles])
            jax.clear_caches()
            return out
        for rec in (records or range(1, ctl.ncvarrecs + 1)):
            if devices == 1:
                a = AR.Adjoint(m, iarr=ctl.iarr, rec=rec)
                model = self._mode_arrays(m, a.model, mode, lsr_derivative, cg2d_derivative)
                fc, g = a.value_and_grad(model=model)
                out[rec] = (float(fc), np.asarray(g))
            else:
                step, model0, st0, xs, theta, params_fn, final_cost = AR.sharded_problem(m, ctl.iarr, rec)
                model = self._mode_arrays(m, model0, mode, lsr_derivative, cg2d_derivative)
                vg = SG.sharded_value_and_grad_fn(sh, step, theta, model, st0, final_cost=final_cost,
                                                  init_fn=lambda t, s: s, params_fn=params_fn, schedule="step")
                fc, g = vg(sh.put_tree(theta), SG.place_model(sh, model), sh.put_tree(st0), xs, SG.seed(1.0))
                out[rec] = (float(fc), np.asarray(g.data)[:sh.layout.nTiles])
            jax.clear_caches()
        return out

    def gradient(self, out, mode="run", devices=1, control=None, lsr_derivative=None, cg2d_derivative=None):
        """The adjoint of the experiment's own cost (COST_FINAL's fc) with respect to its own control `control`
        (default data.grdchk's grdchkvarname; any genarr / gentim2d control of data.ctrl), every record, written as
        adxx_<name>.<optimcycle> in <out>/rundir. mode: "run" or "exact"; devices: tile sharding over that many JAX
        devices (more than JAX has: DeviceCountError before any run directory); lsr_derivative: None (A1, see
        _mode_arrays) or "run" / "sweeps"; cg2d_derivative: None (the mode's), "run" or "exact" (module docstring)."""
        from mitjax.drivers.grdchk import write_adxx
        _xla_flags()
        if mode not in MODES:
            raise ValueError(f"gradient mode {mode!r}: one of {MODES}")
        if cg2d_derivative not in CG2D_DERIVATIVES:
            raise ValueError(f"cg2d_derivative {cg2d_derivative!r}: one of {CG2D_DERIVATIVES}")
        devices = _check_devices(devices)
        m = self.model(out)
        ctl = self._control(m, control)
        res = self._value_and_grad(m, ctl, mode, devices, lsr_derivative, None, cg2d_derivative)
        fcs = {r: fc for r, (fc, _) in res.items()}
        if len(set(fcs.values())) != 1:
            raise AssertionError(f"gradient: the records' programs give different costs {fcs}")
        adxx = {r: g for r, (_, g) in res.items()}
        stem = write_adxx(m, ctl, adxx)
        files = sorted(p for p in Path(m.rundir).iterdir() if p.name.startswith(stem))
        return Gradient(Path(out), Path(m.rundir), ctl.name, mode, int(devices), next(iter(fcs.values())), adxx,
                        files)

    def grdchk(self, out, mode="run", lsr_derivative=None, cg2d_derivative=None):
        """GRDCHK_MAIN (grdchk_main.F:202-524) at data.grdchk's points: the adjoint gradient of the run's cost
        (mode as gradient) and the finite differences (fc at xx +- grdchk_eps, the cost program of the same driver),
        printed into <out>/rundir/output.txt as output_adm.txt prints them (the reference fc; per check: the start
        line, `grdchk pos`, fc+ and fc-, the ADM lines, the end line; then GRDCHK_PRINT)."""
        import jax

        from mitjax.drivers import adjoint_run as AR
        from mitjax.drivers.grdchk import grdchk_positions, grdchk_settings, storage_point
        from mitjax.pkg.grdchk import grdchk_print as gp
        _xla_flags()
        if mode not in MODES:
            raise ValueError(f"grdchk mode {mode!r}: one of {MODES}")
        if cg2d_derivative not in CG2D_DERIVATIVES:
            raise ValueError(f"cg2d_derivative {cg2d_derivative!r}: one of {CG2D_DERIVATIVES}")
        m = self.model(out)
        s = grdchk_settings(m.exp)
        ctl = self._control(m, s["grdchkvarname"])
        pts = grdchk_positions(m, ctl, s)
        if len(pts) > gp.MAXGRDCHECKS:
            raise NotImplementedError("GRDCHK_MAIN: more checks than maxgrdchecks (ierr_grdchk = -1, :525-527)")
        sz = m.cfg.size
        eps = float(s["grdchk_eps"])
        if not s["useCentralDiff"]:
            raise NotImplementedError("GRDCHK_MAIN: useCentralDiff = .FALSE. (one-sided differences) is not wired")
        lines, mem, checks = [], [], []
        by_rec = {}
        for icomp, r in pts:
            if r.ierr == 0:
                by_rec.setdefault(r.icvrec, []).append((icomp, r))
        results = {}
        for rec, rp in by_rec.items():
            if ctl.kind == "genarr":
                a = AR.GenarrAdjoint(m, key=ctl.key)
            else:
                a = AR.Adjoint(m, iarr=ctl.iarr, rec=rec)
            model = self._mode_arrays(m, a.model, mode, lsr_derivative, cg2d_derivative)
            fc, g = a.value_and_grad(model=model)
            g = np.asarray(g)
            jax.clear_caches()                          # lane APIF: between the gradient and the cost programs
            sp = [storage_point(ctl, r, sz) for _, r in rp]
            fd = a.grdchk_fd(sp, eps=eps, model=model)   # grdchk_main.F:359-432 (central: epsfac 2)
            for (icomp, r), p, (_, fp, fm, gfd) in zip(rp, sp, fd):
                results[icomp] = (float(fc), float(g[p]), fp, fm, gfd)
            jax.clear_caches()
        fcs = sorted({v[0] for v in results.values()})
        if len(fcs) > 1:
            raise AssertionError(f"grdchk: the records' programs give different reference costs {fcs}")
        lines += gp.start_lines(fcs[0] if fcs else 0.0)
        for ichknum, (icomp, r) in enumerate(pts, start=1):
            lines += gp.check_start_lines(ichknum)
            lines += gp.check_position_lines(r.itilepos, r.jtilepos, r.layer, r.itile, r.jtile, r.obcspos, r.icvrec)
            fcref, adxxmemo, fp, fm, gfd = results.get(icomp, (0.0, 0.0, 0.0, 0.0, 0.0))
            ratio = gp.ratio_ad(adxxmemo, gfd)
            if r.ierr == 0:
                lines += gp.perturb_lines(fp, fm)
                lines += AR.adm_lines(fcref, adxxmemo, gfd)
            lines += gp.check_end_lines(ichknum, r.ierr)
            mem.append(dict(xxmemref=0.0, xxmempert=-eps, adxxmem=adxxmemo, fcrmem=fcref, fcppmem=fp, fcpmmem=fm,
                            gfdmem=gfd, ratioadmem=ratio, bimem=r.itile, bjmem=r.jtile, ilocmem=r.itilepos,
                            jlocmem=r.jtilepos, klocmem=r.layer, icompmem=icomp, ierrmem=r.ierr))
            checks.append(dict(i=r.itilepos, j=r.jtilepos, k=r.layer, bi=r.itile, bj=r.jtile, rec=r.icvrec,
                               fcref=fcref, fcpertplus=fp, fcpertminus=fm, gfd=gfd, adjoint_gradient=adxxmemo,
                               ratio_ad=ratio, ierr=r.ierr))
        lines += gp.grdchk_print(len(pts), 0, mem, grdchk_eps=eps, grdchkvarname=s["grdchkvarname"])
        path = Path(m.rundir) / "output.txt"
        with open(path, "x") as fh:
            fh.write("\n".join(lines) + "\n")
        return Grdchk(Path(out), Path(m.rundir), path, ctl.name, checks, lines)


def run_fields(carry):
    """{name: numpy} of a driver carry: the ocean State (carry[0]) by its mitjax names, and the sea-ice state
    (carry[4]["seaice"], pkg/seaice) by its SEAICE.h names."""
    state = carry[0]
    fields = {n: np.asarray(getattr(state, n).data) for n in state.names() if hasattr(getattr(state, n), "data")}
    pk = carry[4] if len(carry) > 4 else {}
    for n, v in (pk.get("seaice") or {}).items():
        if not hasattr(v, "data"):
            continue
        if n in fields:
            raise AssertionError(f"Run.fields: the sea-ice field {n} has the name of an ocean field")
        fields[n] = np.asarray(v.data)
    return fields


def _lsr_default_applies(m, sp, lsr_choice):
    """True where the API's default LSR derivative ("sweeps", A1) replaces the build's: the build does not tape the
    LSR sweeps (mitjax.ad.modes.lsr_derivative gives "forward_only", whose derivative raises). Plan decision 14 makes
    A1 the port's LSR derivative where TAF tapes every sweep; decision 17 extends it, by decision, to the build that
    does not (cs32x15 input_ad.seaice: TAF's adjoint of the untaped LSR is wrong by construction, seaice_lsr.F:805-807,
    MITgcm#1041). The forward is unchanged (the scan's forward is the while_loop's bit for bit)."""
    return lsr_choice(m.cfg, sp) == "forward_only"


def _input_rw(exp, readBinaryPrec, size, e2io=None):
    """pkg/rw's RW for Experiment.grid(): MDS_READ_FIELD's names resolved in the experiment's input directories as
    testreport's linkdata would link them (_input_path; no run directory)."""
    from mitjax.pkg.rw.read_rec import RW

    class InputRW(RW):
        def path(self, fName):
            return str(_input_path(exp, str(fName).strip()))
    return InputRW(exp.exp_dir, readBinaryPrec, size, e2io=e2io)


def _input_path(exp, name):
    """The path of input file `name` (or its .data / tile files) of experiment `exp`, without a run directory."""
    import os

    from mitjax import make_rundir as mr
    from mitjax.config.namelists import link_sources
    src = link_sources(exp.exp_dir, exp.variant)
    for key in (name, name + ".data", name + ".001.001.data"):
        if key in src:
            return exp.exp_dir / src[key] / name
    dirs, _ = mr.input_dirs(exp.variant)
    if any(os.access(exp.exp_dir / d / "prepare_run", os.X_OK) for d in dirs):
        return _prepared_dir(exp) / name
    raise FileNotFoundError(f"{exp.exp_dir}/{exp.variant}: input file {name} is not in the input directories "
                            f"({', '.join(dirs)})")


def _prepared_dir(exp):
    """A prepare_run variant's run directory under $MJX_CACHE/api_grid/ (drivers/run.make_rundir, made once per
    experiment path and variant, reused, never removed)."""
    import hashlib

    from mitjax import paths
    key = hashlib.sha256(f"{exp.exp_dir}\0{exp.variant}".encode()).hexdigest()[:16]
    out = Path(paths.CACHE) / "api_grid" / f"{exp.exp_dir.name}-{exp.variant}-{key}"
    if not (out / "rundir").exists():
        exp._rundir(out)
    return out / "rundir"


# ---------------------------------------------------------------------------------------------------- comparison

def compare(run, results, match=None, kind=None):
    """testreport's comparison of `run` (a Run, a Grdchk, or the path of an output file) with the experiment's
    reference `results` (exp.results()): mitjax.testreport_jax's check lists, digits and verdict, for the forward
    (`fwd`) or adjoint (`adm`) kind of the run. Returns testreport_jax.Report (`.summary`, `.verdict`, `.run`)."""
    from mitjax import testreport_jax as T
    match = T.MATCH_CRIT if match is None else match
    if kind is None:
        kind = getattr(run, "kind", None) or {T.KIND_FWD: "fwd", T.KIND_ADM: "adm"}[T.kind_of_variant(
            results.variant)]
    k = T.KIND_NAMES[kind]
    out_path = Path(getattr(run, "output", run))
    ref = results.reference(kind)
    files = T.resolve_checklists(results.exp_dir, results.variant, k)
    checklists = {n: p.read_text() for n, p in files.items()}
    rr = T.testoutput_run(T.read_lines(out_path), T.read_lines(ref), k, checklists)
    base = T.INPUT_DIR[k]
    name = results.exp_dir.name
    expname = name if results.variant == base else f"{name}.{results.variant[len(base) + 1:]}"
    summary, verdict = T.formatresults(expname, rr.results, match)
    return T.Report(expname, k, out_path, ref, files, rr, summary, verdict)
