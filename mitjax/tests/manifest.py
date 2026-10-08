"""Cost group of every test file — merged from one fragment per area, the one place a test file's tier is decided.

Each area (lane) owns one fragment `mitjax/tests/manifest_<area>.py` holding `MANIFEST = {path: group}`; lanes never
edit this file. All fragments in this directory are merged here; a path listed by two fragments is an error, as is a
fragment without a `MANIFEST` dict. conftest.py applies the group as a pytest marker to every test collected from the
file and refuses to collect a test file that is not listed, so a new file cannot silently land in no tier (or in the
10-minute tier 1 by accident). Paths are relative to the repository root.

Groups:
    smoke   seconds; safe on a login node (`pytest -m smoke`); also part of tier 1
    tier1   fast suite: < 10 min and < 100 tests on one CPU compute node, every commit (scripts/run_tier1.sbatch)
    tier1x  extended CPU suite (nightly): substep dump gates steps 1-3, replay gates, P=4 forward, FD h-sweeps,
            negative controls
    tier2   GPU suite (nightly / milestone): A100 runs, sharded gradients on 4 real GPUs vs 1, repeats
    tier3   milestone climate twins (cost stated and Nikolay's yes before submission)
"""

import importlib.util
from pathlib import Path

GROUPS = ("smoke", "tier1", "tier1x", "tier2", "tier3")

# Tier 1 as run by scripts/run_tier1.sbatch.
TIER1_EXPRESSION = "smoke or tier1"
TIER1_MAX_TESTS = 100

# Directories searched for test files; must equal pyproject's [tool.pytest.ini_options] testpaths, and every test
# file of the repository must sit under one of them (fesom_jax lessons §7: 64 tests in an uncollected directory never
# ran). test_manifest.py checks both.
TEST_DIRS = ("mitjax/tests", "scripts/tests")

FRAGMENT_GLOB = "manifest_*.py"


def merge_fragments(fragments):
    """Merge {fragment name: {path: group}} into one dict; a path listed twice is a ValueError."""
    merged, owner = {}, {}
    for name in sorted(fragments):
        for path, group in fragments[name].items():
            if path in merged:
                raise ValueError(f"{path} is listed by both {owner[path]} and {name}")
            merged[path], owner[path] = group, name
    return merged


def load_fragments(directory):
    """Return {fragment name: MANIFEST dict} for every manifest_<area>.py in `directory` (loaded by path)."""
    out = {}
    for f in sorted(Path(directory).glob(FRAGMENT_GLOB)):
        spec = importlib.util.spec_from_file_location(f"_mjx_manifest_{f.stem}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = getattr(mod, "MANIFEST", None)
        if not isinstance(m, dict):
            raise ValueError(f"{f.name} has no MANIFEST dict")
        out[f.stem] = m
    return out


FRAGMENTS = load_fragments(Path(__file__).resolve().parent)
MANIFEST = merge_fragments(FRAGMENTS)


def audit(root, manifest=None):
    """Return (unlisted, missing, bad_group, outside): test files on disk under TEST_DIRS not in the manifest,
    manifest entries with no file, entries whose group is not one of GROUPS, and test files anywhere in the
    repository outside TEST_DIRS (the runner would never collect them)."""
    manifest = MANIFEST if manifest is None else manifest
    root = Path(root)
    on_disk = {p.relative_to(root).as_posix()
               for d in TEST_DIRS if (root / d).is_dir()
               for p in (root / d).rglob("test_*.py")}
    unlisted = sorted(on_disk - set(manifest))
    missing = sorted(set(manifest) - on_disk)
    bad_group = sorted(f for f, g in manifest.items() if g not in GROUPS)
    skip = {".git", "work", "__pycache__"}
    everywhere = {p.relative_to(root).as_posix() for p in root.rglob("test_*.py")
                  if not skip.intersection(p.relative_to(root).parts)}
    outside = sorted(everywhere - on_disk)
    return unlisted, missing, bad_group, outside
