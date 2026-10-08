"""The one `XLA_FLAGS` string for bitwise CPU gates, composed in one place and assigned once.

    --xla_cpu_max_isa=AVX                     no FMA: XLA:CPU at AVX2 contracts a*b+c into an FMA, while the oracle is
                                              built with -ffp-contract=off (ECCO lessons §5; measured on JMD95Z there:
                                              5896 points differed by up to 4.5e-13 with FMA, 0 without)
    --xla_disable_hlo_passes=algsimp          no algebraic rewrites that change rounding (x/d -> x*(1/d), (x*c1)*c2 ->
                                              x*(c1*c2)); ECCO lessons §5
    --xla_force_host_platform_device_count=4  four fake CPU devices, so sharded forward code runs at P=4 on any CPU

XLA reads `XLA_FLAGS` once, when the first backend initialises (importing jax does not initialise one), so the flags
must be set before any jax computation. A second assignment of the variable replaces the first and silently drops
flags, and a flag given twice keeps only one value, so callers never edit the string themselves: `conftest.py` and
every standalone gate script call `set_gate_xla_flags()` before their first jax computation. Float parameters must
also reach jit as traced arguments, never closed over (ECCO lessons §5); that is the caller's part.

GPU-only test runs (`MJX_XLA_FLAG_SET=gpu`, exported by scripts/run_tier2.sbatch) use the gate set WITHOUT
`--xla_disable_hlo_passes=algsimp` (`flag_set()`): the flag applies to every backend's HLO pipeline and, measured on the
10-step R2 gradient program (lane SHARDGRAD, job 27833512, one A100), makes the XLA GPU compile 2.8 x longer: 701 s
with it, 254 s without (trace 50 s and lower 8 s either way; J bitwise the same); the GPU tests compare against their own measured repeat floors and make no bitwise claim. The CPU
gates (tier 1, tier 1x) keep the full set; any other value of MJX_XLA_FLAG_SET is an error.

The API (mitjax/api.py, docs plan 20261006 S5) sets its own variant, `set_api_xla_flags()`: the same flags, except
  - `--xla_cpu_max_isa=AVX` only on x86-64 (`platform.machine()` x86_64 / AMD64). Evidence (XLA 9b635916, the commit
    jaxlib 0.10.1 = jax 619764c1 was built from, third_party/xla/revision.bzl): xla/backends/cpu/codegen/
    cpu_features.cc CpuFeatureFromString maps "AVX" on every host; on AArch64 ShouldEnableAArch64CpuFeature filters
    only "sve" / "sve2" (for max ISA NEON / SVE) and keeps every feature for an x86 name, so no feature is filtered and
    the target CPU stays the host's (target_machine_options.cc:115-127): the flag is a no-op there, not an error.
    FMA contraction stays on (cpu_aot_loader.cc:52-58 CompilerTargetOptions: AllowFPOpFusion = Fast) and AArch64 has
    FMA in its base ISA, so XLA:CPU on arm64 may contract a*b+c where our x86 gates do not: results can differ from
    Levante in the last bits (plan decision 10: arm64 is best effort). Read from the source, not run (no arm64 here).
  - the user's own XLA_FLAGS entries win: a flag name the user already set is left as the user set it (no error, no
    second copy); the gates (`set_gate_xla_flags`) keep refusing such a conflict.
XLA reads the variable only when the first backend starts, so neither function changes a process whose JAX backend is
already running.

Stdlib only.
"""

import os
import platform

FAKE_DEVICES = 4
GATE_FLAGS = ("--xla_cpu_max_isa=AVX", "--xla_disable_hlo_passes=algsimp",
              f"--xla_force_host_platform_device_count={FAKE_DEVICES}")


def _name(flag):
    return flag.split("=", 1)[0]


# flag names left out under MJX_XLA_FLAG_SET=gpu (module docstring)
GPU_ONLY_DROPPED = ("--xla_disable_hlo_passes",)


def flag_set():
    """The flags `set_gate_xla_flags` sets: GATE_FLAGS, or under MJX_XLA_FLAG_SET=gpu the gate set without the names
    in GPU_ONLY_DROPPED."""
    which = os.environ.get("MJX_XLA_FLAG_SET", "").strip() or "gate"
    if which == "gate":
        return GATE_FLAGS
    if which == "gpu":
        return tuple(f for f in GATE_FLAGS if _name(f) not in GPU_ONLY_DROPPED)
    raise ValueError(f"MJX_XLA_FLAG_SET={which!r}: expected 'gate' (default) or 'gpu'")


def gate_xla_flags(existing="", flags=GATE_FLAGS):
    """Return `existing` with every flag of `flags` (default: the gate set) present exactly once (appended when
    missing).

    Raises ValueError when `existing` already sets one of the gate flags' names to another value, or sets it twice:
    the gate would silently run with the wrong flags.
    """
    have = existing.split()
    out = list(have)
    for flag in flags:
        same_name = [f for f in have if _name(f) == _name(flag)]
        if not same_name:
            out.append(flag)
        elif same_name != [flag]:
            raise ValueError(f"XLA_FLAGS already sets {same_name}; the bitwise gates need exactly {flag!r} "
                             f"(unset XLA_FLAGS or drop those flags)")
    return " ".join(out)


def set_gate_xla_flags():
    """Set `XLA_FLAGS` for the gates in one assignment (the flags of `flag_set()`); returns the string."""
    os.environ["XLA_FLAGS"] = gate_xla_flags(os.environ.get("XLA_FLAGS", ""), flag_set())
    return os.environ["XLA_FLAGS"]


X86_MACHINES = ("x86_64", "amd64")            # platform.machine() of x86-64 Linux / macOS and of Windows
ISA_FLAG = "--xla_cpu_max_isa"


def api_flag_set(machine=None):
    """The flags `set_api_xla_flags` adds: `flag_set()` without --xla_cpu_max_isa when `machine` (default
    platform.machine()) is not x86-64 (module docstring)."""
    m = (platform.machine() if machine is None else machine).lower()
    flags = flag_set()
    if m in X86_MACHINES:
        return flags
    return tuple(f for f in flags if _name(f) != ISA_FLAG)


def merge_user_flags(existing, flags):
    """`existing` (the user's XLA_FLAGS) with every flag of `flags` whose name it does not set appended: the user's
    entries win, nothing is given twice."""
    have = existing.split()
    names = {_name(f) for f in have}
    return " ".join(have + [f for f in flags if _name(f) not in names])


def set_api_xla_flags(machine=None):
    """Set `XLA_FLAGS` for the API in one assignment (merge_user_flags of api_flag_set); returns the string."""
    os.environ["XLA_FLAGS"] = merge_user_flags(os.environ.get("XLA_FLAGS", ""), api_flag_set(machine))
    return os.environ["XLA_FLAGS"]
