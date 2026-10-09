"""Manifest fragment of the GPU race fix (2026-10-09, mitjax/xla_flags.py docstring: `multi_output_fusion`)."""

MANIFEST = {
    # mitjax/tests/hlo_race.py on hand-written HLO modules: the race found, XLA's safe in-place pattern not, two
    # planted variants found (pure Python, seconds; tier 1x: tier 1 holds its budget of 100 tests)
    "mitjax/tests/test_hlo_race.py": "tier1x",
    # the 3-step vjp forward of tutorial_global_oce_optim/input_ad on one GPU: no racing kernel, three calls bitwise
    # equal; control: compiled without multi_output_fusion disabled it holds the racing kernel; skips without a GPU
    "mitjax/tests/test_gpu_race.py": "tier2",
}
