"""Manifest fragment of lane API (docs plan 20261006 S2, Task 6: the thin API; owner: lane API). No tier-1 test (tier 1
stays at 98)."""

MANIFEST = {
    # Task 6: mitjax/api.py against the gated drivers -- run bitwise vs drivers/run.py (barotropic gyre), compare()
    # and its perturbed-output control, gradient + grdchk of 1D_ocean_ice_column/input_ad vs TAF (admGrd >= 10
    # digits, fc / fc+- every printed digit, adxx file), mode run vs exact on global_ocean.cs32x15/input_ad, sharded
    # gradient (tutorial_global_oce_optim, devices=2 fc bitwise), errors, CLI. About 70 min on a CPU node.
    "mitjax/tests/test_api.py": "tier1x",
    # lane API session 2: refusals (a missing variant directory in the config loader; a non-zero first guess,
    # optimcycle /= 0 or doInitXX = .FALSE.; selectP_inEOS_Zc = 1 without FIND_HYD_PRESS_1D), each with its planted
    # guard removal measured. About 10 min on a CPU node (four Model set-ups of 1D_ocean_ice_column).
    "mitjax/tests/test_api_refusals.py": "tier1x",
    # lane APIF (S5): the API follow-ups without a model run -- XLA flags by platform (mocked) and the user's flags
    # winning, devices above the count refused before the run directory, MinMaxDefaultWarning once per (build, site),
    # no RUNDIR line, cg2d_derivative / lsr default options, clear_caches between programs (drivers mocked),
    # build_notebooks names and kept outputs, TileMap tile order; each with its planted control. About 1 min.
    "mitjax/tests/test_api_unit.py": "tier1x",
    # lane APIF (S5): sharded forward exp.run(devices=N) bitwise = devices=1 (global_ocean.90x40x15/input_ad P=3,
    # adjustment.cs-32x32x1/input P=4), Run.global_field and exp.grid() vs the Fortran's own files (T, Eta, XC, ...,
    # hFacC; exch1 and exch2), grid() bitwise vs the Model's grid, the sea-ice fields of Run.fields (1D column vs the
    # driver and the Fortran's AREA / HEFF), the LSR default on cs32x15 input_ad.seaice; planted controls. ~30 min.
    "mitjax/tests/test_api_followups.py": "tier1x",
    # lane CLI (docs plan decision 12 b): `python -m mitjax run` through the API -- API flags = gate flags on x86-64
    # (control: the arm64 set differs), CLI subprocess (no user XLA_FLAGS) = in-process exp.run byte for byte on the
    # barotropic gyre, --devices 2 = 1 on global_ocean.90x40x15/input_ad (control: one ulp planted in the sharded
    # step -> differs), --devices 5 -> exit 2 with DeviceCountError's text and no directory, the RUNDIR line kept
    # (control: the API prints none). About 10 min on a CPU node.
    "mitjax/tests/test_cli_run.py": "tier1x",
}
