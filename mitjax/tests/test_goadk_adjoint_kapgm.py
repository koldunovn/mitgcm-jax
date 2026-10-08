"""GOADK adjoint gate (M2 Task 24): global_ocean.90x40x15/input_ad.kapgm vs TAF, through the driver Model and
drivers/adjoint_run.GenarrAdjoint. Control: xx_kapgm (genarr3d onto GM_inpK3dGM), wunit.data weights; fc(xx) =
COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENARR(xx))), jax.value_and_grad through the per-step-checkpointed scan.

Gated (helpers: goadk_adjoint_gate.py): `ADM adjoint_gradient` at the 4 grdchk points >= 10 digits vs TAF's
results/output_adm.kapgm.txt (measured 13 13 13 12); the gradient finite, zero off the wet interior, nonzero at 29293
(16 wet points have exactly zero sensitivity); negative control: the control weight x (1 + 1e-7) misses the 10-digit
gate; GRDCHK's finite differences (grdchk_eps = 100): perturbed costs and FD to every printed digit of lane A's FD
oracle, the `ADM ref_cost_function` and `ADM finite-diff_grad` lines identical to TAF's; the tangent-linear vs
adjoint dot test (1e-12 relative; measured 6.5e-14). ~45 min, ~31 GB: the gradient compile ~27 min, the forward ~4
min, the tangent ~11 min (one program at a time).
"""

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

INP = "input_ad.kapgm"


@pytest.fixture(scope="module")
def adj():
    from mitjax.tests import goadk_adjoint_gate as A
    yield from A.adjoint_fixture(INP)


def test_adjoint_gradient_matches_taf(adj):
    from mitjax.tests import goadk_adjoint_gate as A
    A.check_admgrd(adj, nonzero=29293)


def test_grdchk_fd_lines(adj):
    from mitjax.tests import goadk_adjoint_gate as A
    A.check_fd_lines(adj)


def test_dot_test(adj):
    from mitjax.tests import goadk_adjoint_gate as A
    A.check_dot(adj)
