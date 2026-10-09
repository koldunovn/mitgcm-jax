"""The one `XLA_FLAGS` string for bitwise CPU gates, composed in one place and assigned once.

    --xla_cpu_max_isa=AVX                     no FMA: XLA:CPU at AVX2 contracts a*b+c into an FMA, while the oracle is
                                              built with -ffp-contract=off (ECCO lessons §5; measured on JMD95Z there:
                                              5896 points differed by up to 4.5e-13 with FMA, 0 without)
    --xla_disable_hlo_passes=algsimp,multi_output_fusion
                                              algsimp: no algebraic rewrites that change rounding (x/d -> x*(1/d),
                                              (x*c1)*c2 -> x*(c1*c2)); ECCO lessons §5.
                                              multi_output_fusion: no multi-output fusions on the GPU (below)
    --xla_force_host_platform_device_count=4  four fake CPU devices, so sharded forward code runs at P=4 on any CPU

XLA reads `XLA_FLAGS` once, when the first backend initialises (importing jax does not initialise one), so the flags
must be set before any jax computation. A second assignment of the variable replaces the first and silently drops
flags, and a flag given twice keeps only one value, so callers never edit the string themselves: `conftest.py` and
every standalone gate script call `set_gate_xla_flags()` before their first jax computation. Float parameters must
also reach jit as traced arguments, never closed over (ECCO lessons §5); that is the caller's part.

GPU runs use the gate set too. The GPU-only set `MJX_XLA_FLAG_SET=gpu` (the gate set WITHOUT
`--xla_disable_hlo_passes=algsimp`, for a GPU compile 2.8 x shorter: 254 s instead of 701 s on the 10-step R2 gradient
program, lane SHARDGRAD, job 27833512) is RETIRED (Nikolay 2026-10-08): with algsimp enabled XLA compiled a wrong
program for tutorial_global_oce_optim/input_ad on 4 A100s. From step 2 the west-east neighbour tiles swapped a
theta-only surface term (gtNm1 at k=1, per-tile mean differences +-2.6e-10 in pairs), and the sharded gradient was
wrong by up to 95 x max|g| (jobs 27980864, 27981958). The same program with algsimp disabled is right to rounding;
2 A100s and 1 A100 are right either way. `flag_set()` now refuses "gpu" with that reason; "gate" (the default) is
the only set.

`multi_output_fusion` is disabled since 2026-10-09: XLA:GPU's multi-output fusion pass built a kernel that races
(tutorial_global_oce_optim/input_ad, one A100, the vjp forward of `python -m mitjax gradient`). One fusion
(`loop_dynamic_update_slice_select_fusion`, 5 outputs) wrote UPDATE_ETAH's etaH := etaN (update_etah.py:22) in place
into etaH's buffer while the same kernel read etaH at other points for etaN (_exact_conserv, integr_continuity.py:167)
and its halo exchange (EXCH_XY_RL, a gather: integr_continuity.py:76). Every call of the same executable gave another
cost (spread 1.5e-5 relative) and gradient, from step 3 on (the first step where etaN differs from etaH at a halo
source); `--xla_gpu_deterministic_ops=true` did not help (jobs 27987414, 27994893). With the pass disabled the 3- and
10-step programs are bitwise reproducible over 10 calls, the fused kernel is gone, the cost and the gradient check
match TAF's output_adm.txt to 15 / 16 digits and the gradient matches the CPU's (job 28005413). XLA:CPU adds its own
multi-output fusion pass only when the backend extra option `xla_cpu_use_multi_output_fusion` is set
(xla/service/cpu/cpu_compiler.cc:1057-1073, cpu_options.cc:156-163 at XLA 9b635916), so the flag changes no CPU
program. mitjax/tests/hlo_race.py finds such kernels in a compiled program (the tier-2 test test_gpu_race.py; its
control compiles without this flag and must find the one above).

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
    second copy); the gates (`set_gate_xla_flags`) keep refusing such a conflict. One exception,
    `--xla_disable_hlo_passes` (UNION_FLAGS): the user's list of passes is kept and the gate set's passes missing from
    it are appended, so a user's own list never re-enables the passes that compile wrong GPU programs.
XLA reads the variable only when the first backend starts, so neither function changes a process whose JAX backend is
already running.

Stdlib only.
"""

import os
import platform

FAKE_DEVICES = 4
GATE_FLAGS = ("--xla_cpu_max_isa=AVX", "--xla_disable_hlo_passes=algsimp,multi_output_fusion",
              f"--xla_force_host_platform_device_count={FAKE_DEVICES}")
# comma-separated lists the API merges with the user's own instead of leaving the user's alone (module docstring)
UNION_FLAGS = ("--xla_disable_hlo_passes",)


def _name(flag):
    return flag.split("=", 1)[0]


RETIRED_GPU = ("MJX_XLA_FLAG_SET=gpu is retired (2026-10-08): with --xla_disable_hlo_passes=algsimp dropped, XLA "
               "compiled a wrong 4-GPU program of tutorial_global_oce_optim (jobs 27980864, 27981958; "
               "mitjax/xla_flags.py docstring); unset MJX_XLA_FLAG_SET, every run uses the gate set")


def flag_set():
    """The flags `set_gate_xla_flags` sets: GATE_FLAGS ("gate", the default and the only set; "gpu" is refused with
    the reason it was retired, module docstring)."""
    which = os.environ.get("MJX_XLA_FLAG_SET", "").strip() or "gate"
    if which == "gate":
        return GATE_FLAGS
    if which == "gpu":
        raise ValueError(RETIRED_GPU)
    raise ValueError(f"MJX_XLA_FLAG_SET={which!r}: expected 'gate' (the default)")


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
    entries win, nothing is given twice. For a flag of UNION_FLAGS the user sets, the items of `flags`' value missing
    from the user's comma list are appended to it."""
    have = existing.split()
    names = {_name(f) for f in have}
    ours = {_name(f): f.split("=", 1)[1].split(",") for f in flags if _name(f) in UNION_FLAGS and "=" in f}
    out = []
    for f in have:
        if _name(f) in ours and "=" in f:
            user = [p for p in f.split("=", 1)[1].split(",") if p]
            f = f"{_name(f)}={','.join(user + [p for p in ours[_name(f)] if p not in user])}"
        out.append(f)
    return " ".join(out + [f for f in flags if _name(f) not in names])


def set_api_xla_flags(machine=None):
    """Set `XLA_FLAGS` for the API in one assignment (merge_user_flags of api_flag_set); returns the string."""
    os.environ["XLA_FLAGS"] = merge_user_flags(os.environ.get("XLA_FLAGS", ""), api_flag_set(machine))
    return os.environ["XLA_FLAGS"]
