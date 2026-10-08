"""Helpers of the global_ocean.cs32x15 input_ad.seaice / input_ad.seaice_dynmix gates (M4 step 7, last item; lane
M4ADCS32ICE).

Oracle: lane A's registered dumps-on runs job27855988-jdon of both variants (build
global_ocean.cs32x15-code_ad-63cdc0b-704fd6b-jaxdump, the forward of the code_ad build; registry M4_TRIPLE /
M4_BUILD), dumped iterations 36000-36001 (input_ad.seaice: nTimeSteps 2) and 36000-36002 (input_ad.seaice_dynmix:
nTimeSteps 5). nIter0 = 36000 from pickup.0000036000 + pickup_seaice.0000036000 (linked from input.icedyn by
prepare_run). The standard oracle is the reference: neither variant meets a subnormal (measured, lane M4ADCS32ICE
session 1, dev job 27903286: 0 subnormal values in every dumped record of both jdon runs and in every binary output
of the plain and jdon runs), so no FTZ variant is needed (plan decision 13: measure first).

Teacher forcing and the stage comparison are lane M4CS32ICE's (m4cs32ice_gate.teacher_carry / step_compare: every
dumped stage, every point of every tile incl. halos and cube corners, bit patterns)."""

import functools

EXPS = {"seaice": ("global_ocean.cs32x15", "input_ad.seaice"),
        "dynmix": ("global_ocean.cs32x15", "input_ad.seaice_dynmix")}
JDON = "job27855988-jdon"


def top(v):
    from mitjax import paths
    e, i = EXPS[v]
    return paths.REFERENCE_RUNS / e / i / JDON


FD = "job27856074-fdzero"          # lane A's FD oracle run (registry M4_FD: xx_theta, zero adxx), build plain_m2b


def fd_top(v):
    from mitjax import paths
    e, i = EXPS[v]
    return paths.REFERENCE_RUNS / e / i / FD


def _model(v, namelist=None):
    from mitjax.config.params import load
    from mitjax.drivers.model import Model, with_namelist
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    e = load(*EXPS[v])
    if namelist:
        e = with_namelist(e, namelist)
    return Model(e, top(v) / "rundir")


@functools.lru_cache(maxsize=None)
def run(v):
    """r with .m (the Model on the oracle's dumps-on run directory), .ds, .its, .stages (as cube_run_gate.CubeRun)."""
    from types import SimpleNamespace
    from mitjax.io.dump import DumpSet
    m = _model(v)
    ds = DumpSet(top(v) / "dumps")
    return SimpleNamespace(m=m, ds=ds, its=ds.iterations(), stages=ds.stages)


def planted_model(r, v, namelist):
    from mitjax.tests import m4cs32ice_gate as G
    return G.with_model(r, _model(v, namelist))


def run_compare(r):
    """({(it, stage): bad fields}, {(it, stage): fields compared}): the dumped iterations stepped from the Model's
    own initial carry (no teacher forcing), every dumped stage probed and compared (cs32_gate.run_compare, incl.
    S18_cost_tile's cost.h)."""
    from mitjax.tests import cs32_gate as S
    out, ncmp, _ = S.run_compare(r, len(r.its))
    return {k: b for k, b in out.items() if b}, ncmp


def step(r, it, until=None):
    """({stage: bad fields}, {stage: fields compared}) of FORWARD_STEP of iteration `it` from the oracle's state at
    its start (the Model's own initial carry at nIter0), every dumped stage compared."""
    from mitjax.tests import m4cs32ice_gate as G
    out, ncmp = G.step_compare(r, it, until=until)
    return {s: b for s, b in out.items() if b}, ncmp
