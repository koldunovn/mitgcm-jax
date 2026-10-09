# Troubleshooting

## "not ported": reading the errors

mitjax stops instead of running a different model. The error says what is missing and where:

| error | when | what it names |
|---|---|---|
| `UnknownNamelistVariable` | loading | a namelist variable that is not in its group's `NAMELIST` statement in the routines your build compiles (a typo, or a parameter of a package or option that is not compiled) |
| `UnportedRoutine` | loading | an own `.F` in `code/` that is neither the MITgcm file it replaces nor a ported experiment routine ([configurations.md](configurations.md#your-own-f-files)) |
| `UnportedPackage` | setting up the model | a package switched on in `data.pkg` that is not ported |
| `UnsupportedOption` | setting up the model | an option value the port refuses (e.g. reading input through `pkg/mnc`) |
| `NotImplementedError` | setting up the model, or compiling the time step | the Fortran routine, the branch and usually its `.F` lines, e.g. `CALC_VISCOSITY: pCellMix_select > 0 (:170-394) is not ported` |
| `MinMaxDefaultWarning` (a warning) | compiling the time step | a MAX/MIN statement where your build takes the documented default ([configurations.md](configurations.md#maxmin-at-build-dependent-statements)) |

A `NotImplementedError` raised while the time step is compiled comes from deep inside JAX's tracing; the last lines of
the traceback name the routine. Line numbers like `(:170-394)` are lines of the `.F` file of that routine at
`63cdc0b`. Every explicit refusal in the code is listed in [configurations.md](configurations.md#supported-options-generated);
[configurations.md](configurations.md#adding-a-port) says how to port what is missing.

## Compile time and memory

Most of a short run's time is XLA compiling the time step (and, for a gradient, its backward pass); a longer run
reuses the compiled step. Measured on 8 cores of an AMD EPYC 7763 CPU node, including compiling:

| run | wall time | peak memory |
|---|---|---|
| `tutorial_barotropic_gyre` forward with comparison and plots | 109-127 s | 2.0 GiB |
| `1D_ocean_ice_column/input_ad` gradient / gradient check | 322 s / 365 s | 5.3 GiB |
| `tutorial_global_oce_optim/input_ad` gradient, 10 steps, 4 CPU devices | 203-218 s | 4.3 GiB |
| `global_ocean.cs32x15/input_ad.seaice_dynmix` gradient (each mode) | 757-815 s | 18.0 GiB |
| `lab_sea/input_ad` gradient check | 1401 s | 21.5 GiB |

What makes it larger: the number of levels and tiles, the packages (sea ice with LSR is the most expensive), and for
gradients the stored states of the window ([gradients.md](gradients.md#checkpointing)). If memory runs out, run one
program per process, or call `jax.clear_caches()` between big programs in one process.

**`vm.max_map_count`.** Every compiled XLA:CPU kernel holds about 3 memory mappings, and Linux allows a process 65530
by default (`sysctl vm.max_map_count`): about 20 000 live kernels. Several big programs in one process can cross it.
The symptom is an abort inside the XLA compile (`LLVM ERROR: Unable to allocate section memory!`; a bare `Fatal Python
error: Aborted` when the output is captured), on a machine with plenty of free memory. Fixes: `jax.clear_caches()`
and `gc.collect()` between big programs, one big program per process, or a larger `vm.max_map_count` (needs root).

**Thread limits.** XLA sizes its compile thread pools from the number of cores it sees. On a machine with many cores
and a low `ulimit -u` it can abort with `pthread_create ... failed`; restrict the cores (`taskset -c 0-7 python ...`).

## float64

mitjax computes in float64 everywhere: `import mitjax` enables `jax_enable_x64` before any array is created. Import
mitjax before you create JAX arrays of your own: JAX makes float32 arrays until x64 is enabled.

## XLA flags

mitjax sets `XLA_FLAGS` once, before JAX starts its first computation (`mitjax/xla_flags.py`):

- `--xla_cpu_max_isa=AVX`: no fused multiply-add on x86-64 CPUs (MITgcm's reference build has none). The API leaves
  it out on other processors: there it would filter nothing.
- `--xla_disable_hlo_passes=algsimp,multi_output_fusion`: two XLA passes switched off.
  - `algsimp`: no algebraic rewrites that change rounding (`x/d` into `x*(1/d)`). Keep it on GPUs too: with this
    pass enabled, XLA compiled a wrong program for `tutorial_global_oce_optim` on 4 A100 GPUs (neighbouring tiles
    exchanged a surface forcing term from the second step on, and the gradient was wrong), while the same run with
    the flag was right ([parallel.md](parallel.md#what-to-expect)).
  - `multi_output_fusion`: no GPU kernels that compute several outputs at once. With it, XLA built a kernel for the
    gradient of `tutorial_global_oce_optim` that overwrote the free-surface height `etaH` while other threads of the
    same kernel still read it, so every call gave another cost and gradient. Without it the cost is the same on every
    call and the gradient check matches TAF's `output_adm.txt` to 15-16 digits on one A100. XLA:CPU does not run
    this pass by default, so CPU results do not change.
- `--xla_force_host_platform_device_count=4`: four CPU devices for sharded runs ([parallel.md](parallel.md)).

XLA reads `XLA_FLAGS` only when the first backend starts, so set your own `XLA_FLAGS` before your first JAX
computation. For the API and the command line (`python -m mitjax run`, `gradient`, `grdchk`) your own entries win:
a flag you set is left as you set it, except that the passes above are added to your own
`--xla_disable_hlo_passes` list. Without your own `XLA_FLAGS`, on x86-64, both use exactly the flags the
project's tests use. The tests and gate scripts use the strict set: there an `XLA_FLAGS` value that sets one of these
three flags to something else is refused. Without these flags a run still works, but it is no longer bit for bit what
the tests check (and without the two disabled passes a GPU run can be wrong, above); on
x86-64 the fused multiply-adds change the last digits. XLA:CPU allows floating-point contraction, and arm64
processors have fused multiply-add, so on a Mac expect differences in the last bits (read from XLA's source, not
measured).

## Subnormal numbers

XLA on CPU computes with subnormal numbers flushed to zero (FTZ/DAZ; no XLA flag turns this off); gfortran keeps
them. A run only sees the difference where values decay below about 2.2e-308. In the gated runs this happens in one
place: the sea-ice velocities of `global_ocean.cs32x15/input.seaice` decay through the subnormal range at a few
hundred points, so mitjax differs from MITgcm's standard build in bits below 2^-1021 there, while every residual,
iteration count and monitor digit agrees. The project checks that run against a variant of the Fortran build with
flush-to-zero set at program start (plan decision 13). For your own runs: a difference confined to values below
2^-1021 (about 4.5e-308) is this effect, not a bug.

## Other platforms

On macOS arm64 (best effort, not yet run) XLA may round some operations differently from Linux x86_64, so the
testreport digits against `results/` may be lower than the numbers on these pages; the notebooks' asserts allow for
this. The C preprocessor there is clang's. Measured with LLVM 18's `clang-cpp -traditional -P` on Linux
(not Apple's clang): the configuration of all 28 verification builds is the same as with GNU cpp; the preprocessed
text differs only in blank lines and in the exponent letter of `_d` constants (`D0` for `d0`).
