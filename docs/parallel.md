# Parallel runs

mitjax parallelises over MITgcm's tiles: the tiles of `SIZE.h` (`nSx * nSy` of them) are split over JAX devices,
where MITgcm would split them over MPI processes. One code path serves one and many devices.

## How it works

- Every array is stored `[tile, (k,) j, i]` with halos. On P devices each device holds a block of consecutive tiles;
  when P does not divide the number of tiles, the last device gets padding tiles that compute finite values and are
  never read.
- Halos are filled from one exchange map per experiment, the same on one and on P devices: computed from `SIZE.h`
  and the `data.exch2` topology as MITgcm's exchange routines fill them. Across devices the copies are
  `jax.lax.ppermute` calls inside `jax.shard_map` (with `check_vma=True`).
- Global sums keep MITgcm's order: the sum within each tile in the Fortran loop order, then the tiles in tile order,
  on every device count. So a global sum, and with it the CG2D solver's residuals and iteration counts, does not
  depend on P.

## How to use it

```python
run = exp.run(out="runs/optim-p4", devices=4)        # forward on 4 devices
g = exp.gradient(out="runs/optim-grad-p4", devices=4)
```

On the command line, `python -m mitjax gradient ... --devices 4`; the forward run takes the same option,
`python -m mitjax run ... --devices 4`, which goes through `exp.run(out, devices=4)`. `devices` larger than the
number of JAX devices stops with `mitjax.api.DeviceCountError` (a `ValueError`) before anything is written.

- **CPU:** JAX shows one CPU device unless told otherwise. The API adds `--xla_force_host_platform_device_count=4`
  to `XLA_FLAGS` when it is first called, so up to 4 CPU devices are there on any machine (they share the machine's
  cores; this is for checking, not for speed). For another number, set the flag in `XLA_FLAGS` yourself before JAX
  starts: your own entries win ([troubleshooting.md](troubleshooting.md#xla-flags)). If JAX has already started,
  mitjax cannot change its flags.
- **GPU:** one device per GPU. The project's GPU tests run on 1 and 4 NVIDIA A100 80 GB.

## What to expect

**The forward run is bit for bit the same on every device count** on one machine: every state field, every printed
monitor value. Measured on `tutorial_global_oce_optim/input_ad` (4 tiles of 45 x 20, 10 steps) at P = 2, 3 (with
two padding tiles) and 4 against P = 1: no value differs (notebook 07). The tier-1x tests check P = N against
P = 1 for whole runs of most gated experiments ([status.md](status.md), column "sharded").

**The gradient agrees to rounding, not bit for bit.** The backward pass of an exchange adds the contributions of a
halo point to the tile that owns it; with the tiles on other devices it adds them in another order. The bar for a
short window is

    max |g_P - g_1| <= 1e-14 max |g_1|

Measured on `tutorial_global_oce_optim/input_ad` (4 tiles, 10 steps, 4 CPU devices): 7.1e-15 at P = 4, and
1.32e-14 at P = 2 and P = 3, which is above the bar (P = 2 and P = 3 are bit for bit equal to each other: their
device edges are the same). P = 4 is checked against 1e-14. P = 2 and P = 3 are judged against this window's own
rounding floor (see "Longer windows" below). The floor is 2.0e-14, so P = 2 and P = 3 sit at 0.66 x the floor and
P = 4 at 0.35 x. The bar is 10 x floor = 2.0e-13. The tier-1x test checks P = 2, 3 and 4.

**Longer windows** (plan decision 18): a gradient over several steps amplifies a rounding difference as the model
does, independent of sharding. For the 5-step window of `global_ocean.cs32x15/input_ad` the relative difference is
6.0e-14, 1.8e-13 and 2.2e-13 at P = 6, 5 and 2: above 1e-14, with identical CG2D iteration counts at P = 1 and
P = 6, and one ulp injected at P = 1 moves the gradient as much as P = 6 does. Such a window is judged against its
own floor: the largest relative change of the P = 1 gradient when the adjoint state is perturbed by one ulp
(relative, fixed seed) at each step boundary of the window. The bar is

    max |g_P - g_1| <= 10 x floor

(floor 2.27e-13 for that window). Windows that meet 1e-14 keep 1e-14.

**GPU:** P = 4 GPUs are compared with one GPU, not with the CPU. Each program runs twice with the same inputs; the
repeat floor is the largest relative difference between the two runs. |g_P4 - g_P1| must stay within the floors of
P = 1 and P = 4 plus the summation-order difference measured on CPU between the same two programs (for
`tutorial_global_oce_optim`: 7.66e-15, bound 2.06e-14). A gross-bug guard checks linearity on the GPU (a seed of 2
against twice a seed of 1, relative 1e-12) and that a zero seed gives exactly zero. The GPU runs use the CPU flag set
without `--xla_disable_hlo_passes=algsimp`: with it the XLA GPU compile of a 10-step gradient took 701 s, without it
254 s (one A100); the GPU makes no bit-for-bit claim.

## Not supported

- Several MPI processes (`nPx * nPy > 1` in `SIZE.h`): use more tiles and devices instead.
- The gradient check (`grdchk`) on several devices: it has no `devices` argument.
