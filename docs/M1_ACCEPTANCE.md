# M1 acceptance (plan Task 18)

MITgcm master 63cdc0b, M1 variants, measured 2026-10-02 on branch `m1-go` (GO lane, session 6). Every number below
comes from a Slurm job whose ID is given; the job's files are under `$MJX_RUNS`.

## 1. Forward runs through `python -m mitjax run`

Each variant ran on a CPU compute node through `mitjax.drivers.run.run` (the function `python -m mitjax run` calls)
via `scripts/m1_acceptance.sbatch REPO EXP INPUT PN` (a frozen `git archive` of the commit; the files are in
`$MJX_RUNS/m1_acceptance/<job>-<exp>-<input>/`: `COMMIT`, `run/result.json`, `run/cli/rundir/` with `output.txt` and
the pickups). `mitjax/tests/test_m1_acceptance_doc.py` (tier 1x) re-derives every column of the table below except
the last from those files, so the table cannot drift from the runs.

Columns:

- **check-list variables**: the testreport check list of the variant, in its order; the three digits columns list one
  number per variable in the same order (`tools/testreport_jax.py`, testreport's own digit count: 16 = equal, 22 =
  all compared values zero, `--` = no comparison, e.g. `PS` in the advection tests, which print no cg2d lines).
- **ours vs results/**: our `output.txt` vs `verification/<exp>/results/output*.txt`.
- **yardstick**: our gfortran oracle's STDOUT (`$MJX_REFERENCE_RUNS/<exp>/<input>/job*-jdon`) vs results/.
- **ours vs oracle**: our `output.txt` with the oracle STDOUT as the reference.
- **%MON identical**: every `%MON` record and MONITOR banner equal to the oracle's, in order; number of blocks.
- **pickups byte-identical**: every `pickup*` file of the oracle run directory equal byte for byte to ours, and no
  extra pickup file; number of files.
- **P=N == P=1**: the driver's step under `jit(shard_map(check_vma=True))` on N fake CPU devices vs P=1 for the
  whole run (`go_gate.run_steps_p`): every carry leaf and every per-step output bit for bit (`scripts/m1_acceptance.py
  --pn N`; job in brackets).
- **proved by**: the tier-1x test that gates the same P=N equality for the whole run.

<!-- m1-table -->
| variant | job | check-list variables | ours vs results/ | yardstick | ours vs oracle | %MON identical | pickups byte-identical | P=N == P=1 | proved by |
|---|---|---|---|---|---|---|---|---|---|
| tutorial_barotropic_gyre/input | 27838013 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd | 16 16 16 16 22 16 16 16 22 16 16 16 16 16 16 16 16 | 16 16 16 16 22 16 16 16 22 16 16 16 16 16 16 16 16 | 16 16 16 16 22 16 16 16 22 16 16 16 16 16 16 16 16 | yes (11 blocks) | yes (2 files) | n/a (1 tile) | -- |
| tutorial_baroclinic_gyre/input | 27838082 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd | 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 | 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 | 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 | yes (11 blocks) | yes (8 files) | P=4 yes, 10 steps (job 27838014) | test_r2_baroclinic_gyre.py::test_p4_equals_p1_whole_run |
| advect_xy/input | 27838015 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd | -- 14 14 16 16 16 16 16 16 16 16 16 22 16 16 16 22 | -- 14 14 16 16 16 16 16 16 16 16 16 22 16 16 16 22 | -- 16 16 16 16 16 16 16 16 16 16 16 22 16 16 16 22 | yes (6 blocks) | yes (8 files) | P=2 yes, 80 steps (job 27838015) | test_r3_advection.py::test_p2_equals_p1_advect_xy |
| advect_xy/input.ab3_c4 | 27838016 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd | -- 16 16 16 16 16 16 16 16 16 16 16 22 16 16 16 22 | -- 16 16 16 16 16 16 16 16 16 16 16 22 16 16 16 22 | -- 16 16 16 16 16 16 16 16 16 16 16 22 16 16 16 22 | yes (11 blocks) | yes (4 files) | P=2 yes, 100 steps (job 27838016) | test_m1_pn.py::test_pn_equals_p1_whole_run_and_nan_storage |
| advect_xz/input | 27838017 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd | -- 16 16 16 16 16 16 16 16 16 16 16 16 22 22 22 22 | -- 16 16 16 16 16 16 16 16 16 16 16 16 22 22 22 22 | -- 16 16 16 16 16 16 16 16 16 16 16 16 22 22 22 22 | yes (21 blocks) | yes (8 files) | P=2 yes, 200 steps (job 27838017) | test_m1_pn_advect_xz.py::test_p2_equals_p1_whole_run_and_nan_storage |
| advect_xz/input.pqm | 27838018 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd | -- 13 16 16 16 12 16 16 16 16 16 16 16 22 22 22 22 | -- 13 16 16 16 12 16 16 16 16 16 16 16 22 22 22 22 | -- 16 16 16 16 16 16 16 16 16 16 16 16 22 22 22 22 | yes (21 blocks) | yes (4 files) | P=2 yes, 200 steps (job 27838018) | test_m1_pn_advect_xz.py::test_p2_equals_p1_whole_run_and_nan_storage |
| advect_xz/input.nlfs | 27838019 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd | -- 16 16 16 16 14 14 16 14 16 16 16 16 22 22 22 22 | -- 16 16 16 16 14 14 16 14 16 16 16 16 22 22 22 22 | -- 16 16 16 16 16 16 16 16 16 16 16 16 22 22 22 22 | yes (21 blocks) | yes (4 files) | P=2 yes, 200 steps (job 27838019) | test_m1_pn_advect_xz.py::test_p2_equals_p1_whole_run_and_nan_storage |
| global_ocean.90x40x15/input | 27838020 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd sbo_M sboFW sboAc sboAp | 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 | 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 | 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 | yes (11 blocks) | yes (147 files) | P=4 yes, 10 steps (job 27838020) | test_r4c_global_ocean_run.py::test_p4_equals_p1_whole_run |
| tutorial_global_oce_optim/input_ad | 27838021 | PS Tmn Tmx Tav Tsd Smn Smx Sav Ssd Umn Umx Uav Usd Vmn Vmx Vav Vsd | n/a | n/a | 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 | yes (11 blocks) | yes (16 files) | P=4 yes, 10 steps (job 27838021) | test_m1_pn.py::test_pn_equals_p1_whole_run_and_nan_storage |
<!-- /m1-table -->

Verdict against the plan's bar (every check-list variable at or above the yardstick, and at least 13 digits against
our own oracle): met by every variant. Ours vs results/ equals the yardstick in every cell; ours vs the oracle is 16
(or 22 / `--` where the oracle's own comparison is 22 / `--`) everywhere.

Notes:

- `tutorial_global_oce_optim/input_ad` is the forward run of the adjoint build. results/ holds only `output_adm.txt`
  (the TAF run), so there is no forward results/ comparison and no yardstick; the forward check list is compared
  against the oracle (`testreport_jax --kind fwd --reference <oracle STDOUT>`). The adjoint line is section 2.
- `tutorial_baroclinic_gyre/input`: the first run (job 27838014) stopped at WRITE_PICKUP, whose port raised "MNC
  pickups are not ported" for `useMNC` alone; write_pickup.F:389 tests `useMNC .AND. pickup_write_mnc`, and
  `pickup_write_mnc` is .FALSE. here (mnc_readparms.F:97 default, not set in data.mnc). Fixed in commit 970d61a
  (IOParams.pickup_write_mnc from MNC_READPARMS; guard test with negative control in test_r4a_global_ocean.py); the
  rerun 27838082 runs end to end. The P=4 part of 27838014 is independent of the pickup and is the table's P=N job.
- `tutorial_barotropic_gyre/input` has one tile (62x62, nSx = nSy = nPx = nPy = 1), so there is nothing to shard.
- Wall times of the CLI runs (set-up, compile and run, 128-core node): barotropic 90 s, baroclinic 104 s, advect_xy
  41 s / 55 s, advect_xz 175 s / 135 s / 87 s, global_ocean 168 s, optim forward 194 s.

## 2. Adjoint (R5): tutorial_global_oce_optim/input_ad vs TAF

From `mitjax/tests/test_r5_adjoint.py` (lane R5; `drivers/adjoint_run.Adjoint`: `jax.value_and_grad` of
fc(xx) = COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENTIM2D(xx))) through the per-step checkpointed scan, cg2d implicit
rule) and the R5 dev jobs:

| quantity | ours | TAF results/output_adm.txt | digits |
|---|---|---|---|
| admGrd, point 1 (tile 1, j 3, i 44) | -2.7038420344440266e-06 | -2.70384203444403E-06 | every printed digit (rel. 1.25e-15) |
| admGrd, point 2 (i 45) | -2.7739760579595167e-06 | -2.77397605795952E-06 | every printed digit (rel. 1.22e-15) |
| admGrd, point 3 (i 46) | -2.690915009911805e-06 | -2.69091500991181E-06 | every printed digit (rel. 1.73e-15) |
| admCst (ref_cost_function) | 6.200232281823371 | 6.20023228182337E+00 | every printed digit |
| admFwd (GRDCHK FD, grdchk_eps 0.1), points 1-3 | -2.7038416172686652e-06, -2.773975569247966e-06, -2.6909146466636003e-06 | -2.70384161726867E-06, -2.77397556924797E-06, -2.69091464666360E-06 (= our oracle's FD, job27827478-fdzero) | every printed digit |

Jobs: 27834137 (gradient and FD values above; gradient compile 654 s on a CPU node), 27834391 (test_r5_adjoint.py:
gradient vs TAF, trust protocol and negative control passed; the FD test failed on a 15-digit check of the unrounded
FD against the 15-significant-digit printed oracle value, 14 digits by floor), 27834769 (the FD test rewritten to
compare the printed values, `1PE22.14`, passed). Trust protocol (test_gradient_trust_protocol): FD h-sweep h = 1 ...
0.01 at point 1 with the best h within 1e-6 relative of the adjoint, tangent-linear (jvp) vs adjoint dot test in a
random wet-interior direction within 1e-12 relative, a repeat of the gradient bitwise; the gradient finite on every
lane and zero off the wet interior. Negative control: wtheta of level 1 times (1 + 1e-7) changes the printed fc and
fails the 10-digit admGrd gate.

## 3. GPU smoke run: global_ocean.90x40x15/input on one A100

Comparison, defined before the run (`scripts/gpu_smoke.py` docstring, commit 36e1620): pass if the run completes,
every `%MON` value is finite, and testreport gives >= 10 digits GPU vs our CPU run (job 27838020) on every
check-list variable; the measured digits are the CPU-vs-GPU spread.

Job 27838026 (`scripts/gpu_smoke.sbatch`, one NVIDIA A100-SXM4-80GB, `JAX_PLATFORMS=cuda,cpu`,
`MJX_XLA_FLAG_SET=gpu`, 25-minute `timeout`; elapsed 9 min 42 s): the Model built on the CPU backend (18 s), its
Arrays and initial carry placed on the GPU, `run.forward` for the whole run (10 steps, a MONITOR block and the SBO
records every step, no pickups written).

- Completed; 836 `%MON` records, all finite. Pass.
- Digits GPU vs CPU (the same against the oracle and results/, since the CPU run equals the oracle):
  PS 10, Tmn 13, Tmx 16, Tav 16, Tsd 16, Smn 16, Smx 16, Sav 16, Ssd 16, Umn 16, Umx 13, Uav 13, Usd 16, Vmn 16,
  Vmx 13, Vav 12, Vsd 16, sbo_M 12, sboFW 12, sboAc 14, sboAp 16. Minimum 10 (PS, the cg2d initial residual).
- Spread over all `%MON` records: 131 of 836 differ from the CPU run; largest relative difference 2.5e-10
  (`surfExpan_salt_mean`, a mean of cancelling terms of order 1e-8).
- Compile time of the one-step scan program (`_scan.lower(...).compile()`, timed alone): 141 s on the GPU node
  (CPU node: 67-68 s, jobs 27838126 / 27838137).
- The final carry holds NaN in 5 State leaves on the GPU (job 27838145) as on the CPU (job 27838137), the same
  leaves with the same counts (leaves 8, 9, 16, 34, 54 of the State's sorted field names): dWtransU and dWtransV
  1116 points each (31 of the 16x16 points of each of the 36 tiles), pTracer_01 and gpTrNm1_01 (138240 each) and
  surfaceForcingPTr_01 (9216) entirely, as pkg/ptracers is compiled but usePTRACERS is off. These are
  `empty_state`'s NaN for storage no routine writes (mitjax/model/state.py); a read of any of them would reach
  `%MON`, which is finite and equal to the oracle's on the CPU.
- Steady state (job 27838145, `scripts/step_timing.py --gpu`, 10 steps from the initial carry, no host work):
  0.039 s per step on the A100 (first call 0.108 s), against 0.106-0.129 s on a 128-core CPU node; compile 138 s.
- The host side dominates the GPU run: 389 s for `run.forward`'s 10 steps = one compile of the program (its `iloop0`
  argument is a strong int32, the timed lowering's a weak Python int, so the forward compiles again, about 140 s)
  + 10 x 0.04 s + 11 MONITOR events (MONITOR, SBO, CALC_ADVCFL on the host reading GPU arrays) at about 22 s each,
  against 6-7 s each on the CPU node.
- After the change of session 7 (`run.forward` reads ONE host-side copy of the carry per host event, commit
  bd17c80; on a CPU run no copy at all: the 9 CLI runs of section 1 rerun, jobs 27838425-27838433, give
  byte-identical `output.txt` and pickups): rerun job 27838434, `run.forward` 334 s instead of 389 s, about 17.5 s
  per event; the 836 `%MON` records are identical to job 27838026's (digits vs CPU unchanged). What remains is
  MONITOR's own host work: on a CPU node (job 27838517, per event) MONITOR 5.9 s (11.1 s at the first event),
  SBO 6 ms, CALC_ADVCFL 0.17 s, the copy 3 ms (not broken down on the GPU node, where it is about 17 s).

Usage: 0.16 A100-h (smoke, 9 min 42 s) + 0.05 A100-h (step timing, 2 min 52 s) = 0.21 A100-h of the 0.5 allowed;
session 7 rerun 0.14 A100-h (8 min 40 s).

## 4. Tier-3 twin of global_ocean.90x40x15: cost estimate (approved and run 2026-10-02: section 6)

Plan Task 18: multi-year at production dt, JAX vs Fortran, every `%MON` statistic within the spread of two Fortran
runs (L-TOL-16 recommends three for an envelope). Needs Nikolay's yes before any submission.

Set-up: the verification's own dt (deltaTtracer = deltaTClock = deltaTfreesurf = 86400 s, deltaTmom = 1800 s; data
lines 55-59), from its pickup (nIter0 = 36000, the end of the 100-year spin-up, data:50-54), over 10 model years =
3600 steps (360-day year: externForcingCycle = 31104000 s, data:72); monitorFreq = 2592000 s (30 days: 120 MONITOR
blocks; SBO follows it, sbo_readparms.F:68), dumpFreq off, one pickup at the end. Per-year figures scale linearly.

Fortran runs:

- F1: the oracle's plain binary (`reference/bin/global_ocean.90x40x15-code-63cdc0b-f48c062/mitgcmuv`, 36 tiles of
  10x10, one process, one thread) on the twin namelist.
- F2: the same physics differing only at round-off: either the same binary from the pickup with one wet `Theta`
  value changed by 1 ulp (no build), or another tile layout (e.g. 2x2 tiles of 45x20: the global sums and exchanges
  in another order; one genmake2 build, minutes in a compute job). F3 (envelope) costs the same as F2.
- Measured cost: 0.43 s per step (FORWARD_STEP 4.77 s user time for 10 steps minus MONITOR 0.23 s,
  DO_THE_MODEL_IO 0.20 s and DO_WRITE_PICKUP 0.05 s; plain run job27829315, output.txt timers). 3600 steps:
  about 26 min per run, serial. Two or three runs fit side by side on one compute node: one job of about 30 min
  (request 1 h), 0.5 node-h (exclusive node; the `shared` partition would use 2-3 cores only).

JAX run (CPU, one compute node, gate XLA flags as in the acceptance runs):

- Measured: set-up 25 s, compile of a scan program 67-68 s, steady-state step 0.106-0.129 s (median, jobs 27838137
  / 27838126; `scripts/step_timing.py`); host work per MONITOR event (MONITOR, SBO, CALC_ADVCFL on the host) about
  6-7 s (job 27838020: 168 s wall - 25 s set-up - 68 s compile - 10 steps, over 11 events incl. 147 pickup files).
- 3600 steps x 0.13 s = 8 min, + at most 3 compiles (one per distinct chunk length: the 30-day blocks, the last
  chunk) = 3-4 min,
  + 120 MONITOR events x 7 s = 14 min: about 26 min wall, one job of 1 h requested, 0.5 node-h.
- On one A100 instead (jobs 27838145, 27838026): 3600 x 0.039 s = 2.4 min + 3 compiles x 140 s = 7 min + 120
  MONITOR events x 17.5 s (after the host copy) = 35 min: about 45 min, 0.75 A100-h; MONITOR's host statistics
  are the remaining cost.

Total for the twin (10 years): CPU 1-1.5 node-h (Fortran 0.5, JAX CPU 0.5, a rebuilt F2 layout + 0.1), wall time
under 1 h when the jobs run concurrently; the JAX run on one A100 instead: 0.75 A100-h. Not submitted.

## 5. NaN storage in the whole-run carries

The port fills storage no routine has written with NaN (`mitjax/model/state.py` `empty_state`, FArray `local`,
`pkg/cost/cost_init_varia.py`), so that a read of an unwritten point reaches `%MON`. After the whole run of every M1
variant (survey job 27838344, the run driver's step at P = 1) exactly these leaves hold NaN; everything else is
finite. The reasons were read at `pinned`; `mitjax/tests/go_gate.py` `NAN_STORAGE` keeps them with file:line.

| variant | storage (count) | the Fortran | backward pass |
|---|---|---|---|
| advect_xz (all 3) | dWtransC, dWtransU, dWtransV: every point (324 each) | MOM_FLUXFORM.h r* storage; MOM_CALC_RTRANS is a momentum routine and momStepping = .FALSE., the ALLOW_AUTODIFF reset (dynamics.F:326-331) is not compiled: never written, never read | passed through unchanged |
| global_ocean.90x40x15 | dWtransU, dWtransV: row j = 1-OLy and column i = 1-OLx of every tile (1116 each) | mom_calc_rtrans.F:123-129, :140-148 write i >= 2-OLx, j >= 2-OLy; :150-157 read the same range: never written, never read there | MOM_CALC_RTRANS reads only the written range and is linear in dWtrans |
| global_ocean.90x40x15 | pTracer_01, gpTrNm1_01 (138240 each), surfaceForcingPTr_01 (9216) | pkg/ptracers compiled, usePTRACERS = .FALSE.: PTRACERS_INIT_VARIA (packages_init_variables.F:329-334) and every reader skipped | passed through unchanged |
| tutorial_global_oce_optim/input_ad | cost.h objf_temp_tut, objf_hflux_tut (4 each) | not initialised; COST_FINAL writes them (cost_final.F:134-138) before reading them (:180-203), after the time loop | COST_FINAL overwrites them |

In each case the Fortran's COMMON block holds the loader's zero at those points and never uses it during the run,
so the NaN is not observable in the Fortran sense, and no backward pass reads it: the port keeps the NaN.

- `mitjax/tests/test_m1_pn.py`, `test_m1_pn_advect_xz.py`, `test_r4c_global_ocean_run.py::test_p4_equals_p1_whole_run`
  (tier 1x): the non-finite values of each M1 variant's final carry are exactly these (field and count); the
  barotropic gyre, baroclinic gyre, advect_xy (both) have none. Controls (test_negative_controls_bite): theta
  moved by 1 ulp in the P=N carry gives one differing leaf, an extra per-step output entry unequal outputs, one
  planted NaN a different non-finite set.
- `test_m1_pn.py::test_backward_does_not_read_nan_storage`: the VJP of MOM_CALC_RTRANS (the only reader) over the
  k = 1 .. Nr+1 chain from the initial State (dWtrans all NaN), w.r.t. every field it reads, is finite on every lane
  and leaves exactly the rim NaN; control: one NaN in h0FacW (a coefficient) makes the gradient non-finite.
- Whole-step evidence (job 27838416, not a test: compile 6-7 min each): the VJP of one FORWARD_STEP w.r.t. every
  floating carry leaf, for a random cotangent on every floating output, is finite on all 115 leaves of
  advect_xz/input and all 134 of global_ocean.90x40x15/input, with the NaN storage in the input.

## 6. Tier-3 twin of global_ocean.90x40x15: result (lane B session 14, 2026-10-02)

Set-up as in section 4: the verification's own time steps from its pickup (nIter0 = 36000), 3600 steps = 10 model
years, MONITOR every 30 days (121 blocks of 76 `%MON` statistics: 9196 values per run), dumpFreq = 0,
useDiagnostics = .FALSE. (output only), one pickup at the end (pChkPtFreq = 10 years). The variants live in a shadow
MITgcm tree under `$MJX_RUNS/twin/MITgcm` (`scripts/twin_setup.py`: the clone's entries as symlinks, the twin
namelists `input.twin`, and `input.twin3` = the same with the pickup's Theta(46,21,1) one ulp up); jobs
`scripts/twin_fortran.sbatch`, `scripts/twin_jax.sbatch`, `scripts/twin_gpu.sbatch`; the comparison
`scripts/twin_compare.py` (criterion fixed before the runs were looked at: per statistic and monitor time, JAX inside
[min(f1, f2) - |f2 - f1|, max(f1, f2) + |f2 - f1|]).

| run | what | job | wall |
|---|---|---|---|
| F1 | lane A's plain binary (36 tiles 10x10), input.twin | 27840489 | 1482 s |
| F2 | lane A's retile binary (another tile layout; data.exch2 by testreport's MPI linkdata), input.twin | 27840711 | 1420 s |
| F3 | the plain binary, input.twin3 (Theta one ulp up at one point) | 27841031 | 1497 s |
| JAX CPU | `python -m mitjax run` on one CPU node, gate XLA flags | 27840490 | 1480 s |
| JAX GPU | the same on one A100, MJX_XLA_FLAG_SET=gpu | 27840809 | 965 s |

Results (`$MJX_RUNS/twin/compare/*.json`):

- **F2 = F1 bitwise.** The retiled run prints every `%MON` record of F1 for all 10 years: in this model another tile
  layout changes no rounding, so the F1/F2 envelope has zero width. The second option of section 4, the 1-ulp pickup
  (F3), was run as the round-off twin; F3 first differs from F1 at the second monitor time (iteration 36060) and ends
  10 years later at a median relative difference of 5.1e-6 over the 76 statistics.
- **JAX on CPU = F1 bitwise.** All 9196 `%MON` records of the 121 blocks are textually identical to F1's (and F2's),
  and the 144 pickup files the run writes at iteration 39600 are byte-identical to F1's: 100 % inside either envelope
  (the envelope test is trivially passed by equality).
- **JAX on one A100** (no bitwise claim, section 3): inside the F1/F3 envelope for 7195 of 9196 values (78.2 %); per
  model year 62 % (year 1, while the F3 perturbation has not yet reached most statistics: zero spread) and 76-83 %
  in years 2-10. Where the twin spread is nonzero (5608 values), 66.5 % inside; |GPU - F1| / |F3 - F1| has median
  0.95, 90th percentile 7.7. The worst offenders are the first monitor times, where F1 = F3 (zero spread): e.g. the
  initial block's eta_mean differs by 1.8e-8 relative (MONITOR's reductions run on the GPU there), the first month's
  salt_del2 by 2.0e-7. At the end of the 10 years the GPU run's median relative difference to F1 is 8.8e-7, smaller
  than the 1-ulp twin's 5.1e-6. Against the zero-width F1/F2 envelope the GPU run is inside for 37.9 % (equal values
  only). A two-member envelope is narrow (one realisation of the round-off growth; L-TOL-16 recommends three).

Cost: CPU 1.65 node-h (F1 0.41, F2 0.40 plus a 24-s first attempt without data.exch2, F3 0.42, JAX 0.42; exclusive
nodes) of the 2 approved; GPU 0.27 A100-h of the 1 approved.
