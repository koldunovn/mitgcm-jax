"""Manifest fragment of lane PORT (docs plan 20261006 S1: portability; owner: lane PORT). No tier-1 test (tier 1 stays
at 98)."""

MANIFEST = {
    # Task 2: experiments outside the MITgcm tree run bitwise equal to the in-tree run; decision 11 (an experiment's
    # own unported .F refused by name); negative controls. About 15 min on a CPU node (job 27924823).
    "mitjax/tests/test_experiment_location.py": "tier1x",
    # Task 3: the exchange maps replayed from the exchange code equal every registered probe map bit for bit; planted
    # off-by-one controls (exch1, exch2); the layout + topology cache (hit, name-free key, wrong layout refused).
    # About 3 min (configuration loads with cpp, host-side replays, one Model set-up; 168 s on 8 login cores).
    "mitjax/tests/test_exch_replay.py": "tier1x",
    # Task 4 (R2): the system toolchain (cpp found as genmake2 does, declared DEFINES) equals the oracle's preprocessor
    # on every verification build of the registry (configuration + every preprocessed source); planted DEFINE, no-cpp
    # error and override controls. About 15 min on a CPU node (28 builds, ~30 s each).
    "mitjax/tests/test_toolchain_system.py": "tier1x",
    # Task 4 (R6, decision 7): build identity = hash of the preprocessed options and SIZE.h; every registry build in
    # build_identity.KNOWN; a renamed copy is the known build, an edited one (option, tiling) gets the documented
    # MAX/MIN default with one warning per site. About 2 min (29 configuration loads, no model).
    "mitjax/tests/test_build_identity.py": "tier1x",
    # Task 5: environment.yml (hand-written, for users) agrees with constraints.txt and pyproject.toml; planted
    # mismatches reported. Seconds, but tier1x: tier 1 stays at 98 (the plan's "smoke" would add a tier-1 test).
    "mitjax/tests/test_env_pins.py": "tier1x",
    # Task 5: the no-reference test -- forward (gyre whole run), gradient (1D column input_ad), sharded (P=2 vs P=1)
    # with only MJX_UPSTREAM/MJX_RUNS/MJX_CACHE set and the experiments copied outside the tree; no foreign read
    # but the cpp; planted control: the toolchain read re-pointed at $MJX_REFERENCE is caught. About 6 min wall.
    "mitjax/tests/test_no_reference.py": "tier1x",
}
