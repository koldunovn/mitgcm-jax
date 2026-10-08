"""Manifest fragment of lane M4COSTSHARD (sharded gradients of the cost.h / COST_TEST builds; owner: lane
M4COSTSHARD)."""

MANIFEST = {
    # session 1: global_ocean.cs32x15 code_ad (input_ad, input_ad.seaice, input_ad.seaice_dynmix): the P = 6 trace
    # gate with the gather reverted as its negative control, and the sharded gradients (fc bitwise, finite, padding
    # 0, <= 1e-14 of max|g| vs P = 1; the 5-step input_ad window <= 10 x its P = 1 1-ulp floor, plan decision 18)
    # in subprocesses with 6 fake CPU devices; about 105 min (seven gradient programs of 7-18 min each + the floor)
    "mitjax/tests/test_m4costshard_grad_cpu.py": "tier1x",
    # session 2: Model.cost_final's grid / tables from the `arrays` argument (six cost builds, the set-up poisoned)
    # and SEAICE_COST_TEST's gathered objf_ice chain on planted multi-tile cases at P = 3 (padding) and 4, with their
    # negative controls; a few minutes (six Model set-ups)
    "mitjax/tests/test_m4costshard_cost_paths.py": "tier1x",
}
