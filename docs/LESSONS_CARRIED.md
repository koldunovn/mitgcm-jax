# Lessons carried into mitjax

One deduplicated list of the lessons from three earlier ports, with stable IDs. The sources are the three digests
harvested on 2026-10-01 and kept unchanged in `$MJX_LESSONS` (`mitjax/paths.py` `LESSONS`):

| Tag | Digest | Earlier project |
|---|---|---|
| [E§n] | `ecco_port.md` | ECCO v4r4 port of MITgcm checkpoint66g (c66g) to JAX on the LLC90 grid ("the ECCO port"; its repository is called R) |
| [P§n] | `port2_kokkos.md` | FESOM2 ports: the failed C port, the successful C port, the C++/Kokkos port and the JAMES paper |
| [F§n] | `fesom_jax.md` | FESOM2 → JAX port ("fesom_jax"), incl. its adjoint program |

## How to use

Cite a lesson by its ID in brackets, e.g. `[L-AD-9]`, in commit messages, the per-task lessons log and code
comments. The plan's tags such as [E§6] resolve to IDs through the mapping table at the end of this file. Each entry
states the lesson as a rule or a fact, then *mitjax:* how it applies here where that is not obvious, then *Sources:*
the digest section tag followed by the original citations exactly as the digest gives them. A citation key belongs to
the digest named by the tag in front of it: `PL` after [E§n] is the ECCO port's `R/docs/PORTING_LESSONS.md`, `PL`
after [F§n] is fesom_jax's `~/port_jax/docs/PORTING_LESSONS.md`; the key tables are at the top of each digest.
"[P§0] item n" is item n of that digest's numbered top-ten list. Incidents and numbers come from the earlier projects,
not from mitjax, unless an entry says "mitjax evidence". Fortran `file:line` citations from the ECCO port are at
checkpoint66g; master (63cdc0b) line numbers are re-derived by the Task 7a audit.

IDs are stable: never renumber and never reuse one. To withdraw a lesson, keep its bullet and replace the text by
`- **L-XXX-n Retired.** <date>, <reason>, <successor ID if any>`; a new lesson takes the next free number of its
topic. `mitjax/tests/test_lessons_carried.py` checks the IDs (well-formed, unique, numbered without gaps), that every
entry carries a source tag of an existing digest section, that every ID this file mentions exists, and that the
mapping table covers every digest section and equals the set of entries that cite it. Edit the mapping table together
with the entries.

Topics: L-PROC process and agents · L-ORA oracle, dumps, replay and gates · L-LIT literal translation and
configuration · L-TOL tolerances, comparators, localisation and long runs · L-XLA JAX/XLA numerics · L-ARCH JAX
program structure · L-AD differentiability · L-PAR parallel and sharding · L-PERF performance, GPU and memory ·
L-TEST testing · L-ENV environment and versions · L-OPS operations · L-USER Nikolay's preferences · L-COPY
infrastructure to copy from the ECCO port (the copy audit list) · L-CONF conflicts between the digests.

## The lessons that matter most for M0 and M1

1. [L-ORA-1] Build the oracle and its comparator before any port code; a routine is done only when it matches the
   oracle.
2. [L-ORA-8] Replay every routine on dumped inputs at the rounding floor, then compose with a running-view gate;
   [L-TOL-1] never loosen a tolerance.
3. [L-ORA-9] A gate is trusted only after it fails on a planted error that was measured to bite.
4. [L-LIT-1] Literal translation; a deviation is stopped and reported, never implemented on the way.
5. [L-LIT-8] Values come from the experiment's `data*` files and preprocessed CPP options, never from code defaults;
   [L-LIT-9] from the exact run directory the oracle used.
6. [L-ORA-4] A `code_ad` build computes a different forward model than the `code` build; each needs its own oracle.
7. [L-ORA-12] The per-experiment branch inventory comes from the preprocessor and gcov execution counts, never from
   grep; [L-LIT-10] port everything that inventory uses.
8. [L-XLA-1], [L-XLA-2], [L-XLA-4], [L-XLA-5] Bitwise CPU gates need no FMA, no algsimp, float parameters as traced
   arguments, and one `XLA_FLAGS` string set before the backend starts ([L-CONF-1], [L-CONF-2]).
9. [L-ORA-3], [L-ORA-7] The dump observer is invisible, and every point is gated, halos included.
10. [L-TOL-3], [L-TOL-5] Comparators: `max()` never picks a NaN; refuse to run without masks.
11. [L-AD-1], [L-AD-4] Masked, halo and padding lanes compute finite values; gate the gradient with respect to whole
    initial fields from day one.
12. [L-AD-8], [L-AD-9], [L-AD-10] Never differentiate solver iterations; cg2d gets an implicit-derivative rule with
    tight tangent and transpose solves and zeroed halo lanes.
13. [L-PAR-2], [L-PAR-1] Probe the Fortran exchange (exch2 and exch1) with coded halo points; the forward is bitwise at
    P=1 and P=N.
14. [L-COPY-1] Copy ECCO infrastructure only after the c66g→master diff audit of the Fortran each module encodes
    ([L-COPY-2] … [L-COPY-16]).
15. [L-AD-15] The ECCO port never checked its TAF semantics against TAF output; master's `results/output_adm*.txt` are
    that check.

## L-PROC — Process and agents

- **L-PROC-1 Brainstorm, plan, review the plan, then execute; the review changes the plan.** In the ECCO port the plan
  review (verdict NEEDS REVISION) added an override audit task, the Fortran step order, data staging, the ecco seams,
  early safe-op, full-field gradient and sharding gates, and task splits. Nikolay: "brainstorm" or "plan" means the
  brainstorm skill then the planning skill — reconnaissance, then ask, never deliver and then defend; each agent track
  gets a plan first and runs at high effort. *mitjax:* the step order (packages.conf + data.pkg) differs per
  verification experiment, so a plan states it per experiment. *Sources:* [E§1] BR:41-43; PLAN:12. [F§10]
  mem:pj/feedback-use-brainstorm-plan-skills.md; mem:pj/feedback-subagents-plan-first-high-effort.md.
- **L-PROC-2 Keep agent fleets small and bounded.** 3–5 agents at a time, each on a separate problem with a deliverable,
  a time box and a stated cost; no Workflow tool and no agents that poll Slurm. The ECCO port once ran ten kernel agents
  at once and five wish-list items in parallel. *mitjax:* at most 5 agents, one package or plan task each. *Sources:*
  [E§1] notes/HANDOFF-20260923.md:294; notes/brainstorm-20260924.md:1; GCL:16-29. [F§10]
  mem:pj/feedback-no-dynamic-workflows.md.
- **L-PROC-3 One git worktree per agent, and every commit checked in a clean worktree.** Agents sharing one working tree
  built on each other's uncommitted edits: `init.py` called a method that existed only in another agent's uncommitted
  `exchange.py`, and HEAD failed 3 tier-1 tests. *mitjax:* lanes own disjoint files; shared files such as the test
  manifest are split into per-area fragments. *Sources:* [E§1] PL:231-235.
- **L-PROC-4 Agents die with the session; Slurm jobs survive.** The watchers that were to launch the ECCO stage-1 runs
  died, and the runs had to be launched by hand. Chain follow-up jobs with `--dependency` and list pending job IDs in
  the handoff. *Sources:* [E§1] notes/HANDOFF-20260923.md:318-325.
- **L-PROC-5 Handoff diagnoses and memory notes are hypotheses with a shelf life.** Tag each claim MEASURED (job, file)
  or HYPOTHESIS; the next session re-derives a claim from current code before building on it, and memory is corrected in
  place with date and evidence. In FESOM, "the biharmonic is 20–35 % less stable" was carried for sessions until an
  identical-input probe matched 244659/244659 elements (1.2e-12); a domain expert's "ice velocity is the cause" was
  refuted by `NO_ICE_DYN`; a Kokkos handoff's "GM off by default" and a plan's stale host-stay list were false; a note
  "CORE2 uses PP not KPP" was wrong. *Sources:* [P§0] item 9. [P§1] P2/DT1800_HANDOFF.md:3-26,150; KD/KPL:200,422;
  P2M/feedback_port_used_terms.md:12.
- **L-PROC-6 Reject a hypothesis whose magnitude cannot fit; write down the predicted size of an effect before testing
  it.** "Allreduce order" stayed the lead hypothesis for a 30 % difference (4.62e-5 vs 3.49e-5) that reassociation
  (~1e-15) cannot produce, and nine halo "fixes" each had no observable effect; the cause was a halo write loop. A fix
  that leaves the failing metric unchanged did not fix the failure. *Sources:* [P§1]
  P2/MPI_PORT_REPORT.md:33-41,77-100,168; P2/PORT_EXPERIENCE_REPORT.md:92.
- **L-PROC-7 Verify an agent's finding by hand on one concrete case.** An agent found the `edge_vflux` sign error but
  framed it as a "buffer mismatch"; one hand-computed triangle (+0.289 vs −0.289) proved it. *mitjax:* agents return a
  reproducer (inputs, expected, got, file:line) and own disjoint files. *Sources:* [P§1]
  FPM/project_edge_vflux_sign_fix.md:79-95.
- **L-PROC-8 Commit at every verified state, and log every decision and lesson in the same commit.** A "Phase D PASS"
  with no commit behind it was lost. *mitjax:* commit when a gate passes, with the task's lessons-log
  entry in the same commit. *Sources:* [P§8] P2M/feedback_unique_outdir_per_job.md:14,30. [P§9]
  KM/feedback-document-decisions.md:10; KD/KPL:3-6.
- **L-PROC-9 Record everything that changes a number in each result row.** A configuration field hard-coded as a literal
  mislabelled a headline result. *Sources:* [F§9] PL:6309-6319; PIDX:150-151.
- **L-PROC-10 Record a capability where people land when it bites (a docstring, the project instructions file), not in a
  ticked task write-up, and mark superseded scripts in their docstring the same day.** *Sources:* [F§9] PL:6110-6142.
- **L-PROC-11 Prioritise the question that could kill the approach.** A night of green ticks in fesom_jax avoided it.
  *Sources:* [F§9] PL:5912-5937.

## L-ORA — Oracle, dumps, replay and gates

- **L-ORA-1 Build the oracle and its comparator before any port code; a routine is done only when it matches the
  oracle.** port2's harness paid back its two days at the first divergence; FESOM_port had no harness, reported "All 13
  tests pass" while CORE2 died at step 82 for six days, and its fixes "match Fortran" had no effect on CORE2 (the cause
  was a one-character sign in `edge_vflux`). No tests that encode an agent's own expectation. *mitjax:* one dump oracle
  per experiment family before its physics is ported (Tasks 3a, 4). *Sources:* [E§1] CAT:75-77; BR:6-7. [P§0] item 1.
  [P§1] FP/<agent notes>:52-54; FPM/project_core2_exhaustive.md:26-30; FPM/project_edge_vflux_sign_fix.md:13-20.
- **L-ORA-2 Dump whole fields keyed by global index, with step windows, field lists, and dump points inside internal
  iterations.** The FESOM shim first dumped 5 probe gids hard-coded to values valid on the pi mesh; full-field dumps and
  step windows had to be added later. The shim's two days of cost turned divergence hunts into ~10 min. *mitjax:* whole
  tiles keyed by global (tile,k,j,i), with dump points inside cg2d, the sea-ice solvers and KPP. *Sources:* [P§2]
  ~/port2/fesom2/src/fesom_dump_shim.F90:1-75; ~/port2/inspect_dump.py:1-40; P2/PORT_EXPERIENCE_REPORT.md:436-451.
- **L-ORA-3 Prove the observer is invisible.** With dumps off and with dumps on, the state files and `%MON` are
  byte-identical to the plain build (gfortran -O3, no fast-math, no FMA). *mitjax:* also compare against each
  experiment's `results/output.txt` (Task 4 invisibility test). *Sources:* [E§2] PL:64-65; REF:66-67.
- **L-ORA-4 Every build is its own oracle: `pkg/autodiff` changes the forward model.** It defines ALLOW_AUTODIFF, which
  changes forward branches: the r* reset (`forward_step.F:418-450`), saltPlumeFlux zeroing, IMPLDIFF and static sea-ice
  masks. *mitjax:* an experiment's `code_ad` build computes a different forward than its `code` build, so each gets its
  own oracle (Task 3a builds a forward-only `code_ad` binary). *Sources:* [E§2] PL:30-32, 407-409; EI:37-42.
- **L-ORA-5 Know where `myIter` advances, and make the instrumentation fail loudly on drift.** In c66g
  `forward_step.F:823` advances it right after DYNAMICS, so later stages pass myIter-1, which `instrument.py` asserts;
  missing stages S06-S14 revealed this. Anchors carry expected occurrence counts; mind the 72-column limit, `#ifdef`
  inside a CALL, and pass suffixes for stages in loops, since the index keeps the last record of a repeated key.
  *mitjax:* re-derive the point in master's `forward_step.F` and keep the assert (Task 4). *Sources:* [E§2] PL:66-68;
  reference/jaxdump/instrument.py:9-12; PL:309-311.
- **L-ORA-6 Read dumps lazily and in parallel; a missing stage is information.** One dumped ECCO iteration is 9.5-12.5
  GB; header indexing took 110 s serially and 3 s in parallel. Stage S03 sits inside `IF (useCTRL)`, and the test
  asserts that it is absent. *Sources:* [E§2] PL:89, 307, 312, 393; PL:115.
- **L-ORA-7 Gate every point, halos included.** Points a loop never writes keep their old values; INI_CG2D writes halos
  that UPDATE_CG2D never rewrites; locals are re-initialised only where ALLOW_AUTODIFF_TAMC does it, so seaiceMass
  starts at 1000. *mitjax:* the Fortran-index wrapper (Task 8) defaults to "unwritten points keep the prior array".
  *Sources:* [E§2] KG:29-30; PL:118-119; PL:205; PL:379-380.
- **L-ORA-8 Replay every routine on dumped inputs, at both dumped iterations, then compose the step with a running-view
  gate.** Replay with reference inputs is readable where live-run diffs are not: FESOM's live KPP diffs hit ~52 % of
  nodes, while replay gave 3e-13 (KPP) and 1e-16 (TKE). The running-view gate applies each stage's writes in Fortran
  order, which turns "every stage" into an exact inventory, with the dumped-but-uncomputed values asserted as a set;
  composing the full V4r4 step was then mere wiring (820 stage-field pairs bitwise on the first run). *mitjax:* each
  routine is a pure function — load the oracle's input pytree, call it, diff the output; this is the per-routine
  acceptance test. *Sources:* [E§2] KG:56-60; PL:439-444. [P§2] P2M/feedback_controlled_replay_validation.md:22,31.
- **L-ORA-9 A gate is trusted only after it fails on a planted error that was measured to bite.** The ECCO port's
  planned corner-mask control was a no-op because the corner halos are all dry; for a stage that changes nothing, plant
  the error in the operation itself; jax caches traces per function object, so a control must jit a fresh lambda; a
  probe-only round trip missed a vector-exchange bug that the real-field gate caught. For each failure mode a test
  claims to catch, write down what the broken implementation outputs and check that the assertion fails on it.
  *Sources:* [E§2] PL:126-127; PL:391-392; PL:415-416; PL:86-88. [F§7] PL:6550-6572.
- **L-ORA-10 A gate covers only code that its experiment executes with live data.** The pi mesh had no ice and almost no
  circulation, so tracer advection running backwards passed ("stable 100 steps"); Kokkos device-halo changes validated
  only on pi left the shipped GPU path wrong for ~8 commits (0.4 against a 1e-3 floor on CORE2). *mitjax:* keep a
  coverage table package/branch → experiment and input variant that executes it (Task 5, and the physics-active check
  per rung). *Sources:* [P§0] item 8. [P§1] FPM/project_edge_vflux_sign_fix.md:47; KD/GPU_FIDELITY.md:310;
  KM/feedback-gpu-fidelity-gate.md:20.
- **L-ORA-11 Code the oracle cannot see needs a transcription test or an all-wet synthetic test.** All 8 LLC90 cube
  corners are dry, so the corner fills leave the state unchanged; conservation on an all-wet case shows them (7e-16 with
  the fills, 3e-3 without). *mitjax:* cube-sphere verification experiments with wet corners exercise this code in the
  oracle (M2). *Sources:* [E§2] PL:132-133, 141-142, 177-178, 333-336.
- **L-ORA-12 Take the active branches from the preprocessor and from execution counts, never from grep.** `cpp
  -traditional` with line markers gives `.F` line numbers; a `^#define` grep missed V4r5's `# define
  EXF_LWDOWN_WITH_EMISSIVITY` (an error of 0.7 K in Ts); a gcov line diff, with GCOV_PREFIX per run, found the active
  branches. Check the derived runtime flags in STDOUT too: with DST3 all T/S AB flags are F, so the ECCO plan's AB
  branch was dead. Coverage means execution counts in the experiment, not the existence of a routine: `visc_filt_bcksct`
  was "fully ported, audited, allocated — and DEAD CODE (never called)", and a grep for the Fortran name passes such an
  audit. *mitjax:* Task 5 (`cpp_live.py`, gcov per experiment). *Sources:* [E§2] PL:350; PL:694-696; PL:387-390; PL:70;
  PL:163-165. [P§3] KDI/KPL:1229-1250.
- **L-ORA-13 A branch can be dead while its switch is alive, and gates run at least two steps.** `FESOM_WSPLIT=1`
  announced itself and gave bit-identical output, because the maximum CFLz 0.822 never reached the 1.0 trigger; a
  constant made time-varying is bit-equal at cold start, so step 1 passes and step 2 fails. *mitjax:* tests of threshold
  branches log how many points took the branch; with r* or a nonlinear free surface hFacC equals h0FacC at cold start.
  *Sources:* [P§4] KDI/KPL:1334-1345; KD/KPL:480.
- **L-ORA-14 Use a fast bitwise twin of the oracle and compare the whole STDOUT.** ECCO's mpi13 run (13 ranks × 1 tile)
  is bitwise equal to serial13 and about 5x faster (Nikolay had asked "why serial oracle, super slow"). AD tapes are
  sparse files (165 GB apparent, 0.75 GB data). *Sources:* [E§2] notes/HANDOFF-20260923.md:123; PL:423-433.
- **L-ORA-15 Keep a live twin of every optimised version, and use the working Fortran freely.** `FESOM_KK_VERIFY` runs
  the reference kernel on the live state, diffs and restores, so it can stay on in long runs; per-step trajectories at
  one global node showed the C port erupting at step 4324 while the Fortran stayed smooth through 4400. *mitjax:* keep
  the literal version in the tree as the twin of any optimised version; P=1 is the twin of P=N; add environment-gated,
  bit-neutral dumps to the oracle whenever needed. *Sources:* [P§2] PP/methods.tex:68; KD/KPL:35 (D19);
  P2M/feedback_port_used_terms.md:27-30.
- **L-ORA-16 An identical-input operator probe answers "operator or upstream?".** A deterministic integer hash of the
  global id, fed to both codes for one call, exonerated the FESOM biharmonic (244659/244659 elements, 1.2e-12) after
  sessions of suspicion. *mitjax:* keep an environment-gated hash-field driver in the gfortran oracle for stencil
  routines. *Sources:* [P§2] P2/PORTING_LESSONS.md:116-123; P2/DT1800_HANDOFF.md:5-15.
- **L-ORA-17 A one-routine replay harness gates a second code base cheaply.** It is a genmake2 build whose THE_MAIN_LOOP
  reads every input of the routine from a record file, and it runs in seconds on synthetic or recorded inputs;
  `tools/step_vs_dump.py` reports the first differing stage. *mitjax:* for packages that no convenient verification
  experiment drives; Task 8's replay harness. *Sources:* [E§2] PL:689-693; PL:197.
- **L-ORA-18 Restart-check every new start point.** The published ECCO flux-forced tree cannot read its own pickups (I6
  writer, I5 reader), and only a real restart found it. Before an adjoint from a new date, compare state(t0)+N with an
  independent run field by field and name the fields that differ. *Sources:* [E§2] PL:71-73; EI:10-13; PL:615-617.
- **L-ORA-19 Know what a restart must carry.** An iterative solver's initial guess is state; a "scratch" array can be
  prognostic because of call order; a `save :: lfirst` build from evolving state is a restart bug; judge the restart at
  the restart step, not a day later (round-off grows about 5× per step); "max over the mesh" counts events (7 Baffin
  nodes), so use area-weighted norms. *mitjax:* the pickup holds the cg2d first guess and the AB tendencies; grep for
  SAVEd first-call builds (Task 11 restart gate). *Sources:* [F§9] PL:5669-5735.
- **L-ORA-20 Support a bit-exact start from an oracle pickup from day one: matched restart and injection find a
  systematic drift, more climate runs do not.** FESOM's 63-year deep drift (+1.1e-6 °C/yr, 10× the GPU−C spread) was
  isolated by reading a Fortran restart, comparing every substep and injecting the Fortran ice velocities: a 3e-4
  concentration difference after advection, which came down to one number. *Sources:* [P§7]
  PP/appendix_cport.tex:162-164; P2M/project_longrun_drift.md:18-31.
- **L-ORA-21 The reference must run the same physics: every oracle run carries a manifest.** A comparator defaulting to
  deprecated references with two physics deltas (PP vs KPP, `ice_gamma_fct`) produced a fake −0.236 °C SST bias that
  collapsed to +1e-4 °C once repointed; a comparator Fortran ran scheme 5 against the port's scheme 7 and gave three
  sessions of a false "port less robust" verdict. *mitjax:* each oracle run records the MITgcm sha, optfile, CPP options
  and `data*`/forcing hashes, and a comparator refuses a pair whose manifests differ outside a declared list; diff the
  full physics configuration of both sides before any cross-model verdict. *Sources:* [P§3] KDI/KPL:1229-1250. [P§4]
  KD/GPU_FIDELITY.md:515-522; KD/REFERENCE_RUNS.md:1-12,110-123.
- **L-ORA-22 Oracles and inputs go stale: fingerprint them.** A cached oracle built before a fix "failed" a correct
  binary (T=1.28, 5 cells); the shared mesh's level files changed on /pool (Σnlevels 3832750→3832745) and broke
  months-old bit gates at step 0; a partition regenerated in place would have passed the md5 manifest, because "a
  manifest guards what it hashes"; the /pool CORE2 mesh levels were silently regenerated and the project's own export
  became the only copy. *mitjax:* store each oracle with its source sha; keep MITgcm inputs, grids, pickups and dumps
  under `$MJX_WORK` with sha256 manifests, and protect derived inputs (tile maps, exch2 topology) separately. *Sources:*
  [P§4] KD/KPL:330,455; KDI/KPL:1631-1648. [P§8] KM/feedback-mesh-copies-never-pool.md:11. [F§9] PL:5022-5030.
- **L-ORA-23 "Allocated but never computed" shows up only with the first new consumer.** In FESOM, `sw_alpha`/`sw_beta`
  stayed zero, so a "bit-identical baseline" meant no Redi; `stress_atmice` was ≡0, so the ice felt no wind; readers
  that use only the sign hide value errors (`bvfreq` lacked `N2smth_h` smoothing, −7 % at L1, until KPP `bldepth` read
  its values). The ECCO diff tool's ZERO flag catches "allocated, never computed". *mitjax:* when a package reads an
  existing field in a new way (value instead of sign or mask), replay that field's producer at the floor again.
  *Sources:* [P§5] P2/PORT_EXPERIENCE_REPORT.md:182-200; P2M/feedback_bvfreq_smoothing_gap.md:14-28. [E§11]
  diffdump.py:11.

## L-LIT — Literal translation and configuration

- **L-LIT-1 Literal translation: any difference from the Fortran is a bug, and a deviation needs Nikolay's approval.** A
  deviation carries a `# DEVIATION:` banner, is OFF by default and errors loudly on unknown values; one that seems
  needed is stopped and reported, not implemented. The FESOM C port's shortcuts each cost physics: linearised α/β left
  the OBL stuck at 5 m; a closed-form `ghats` gave wrong magnitudes; `dt/areasvol` imprinted the mesh; an added 500 m
  OBL cap gave wrong physics; an AB_order-3 history shift with AB2 weights gave a Coriolis weight of 2.5. Nikolay:
  literal fidelity is non-negotiable ("VERY frustrated by repeated simplification attempts"); mirror Fortran names and
  structure, with ease of porting as the design criterion. *Sources:* [P§3] ~/port2/FRESH_START.md:6-12,632-646;
  P2/PORT_EXPERIENCE_REPORT.md:51-72,646-674; P2M/feedback_io_system_design.md:9,21; KD/CG_BLOWUPS_M13.md:112-130. [P§9]
  FPM/feedback_no_simplifications.md:7-14; P2M/feedback_port_fidelity.md:7-11. [E§1] KG:9.
- **L-LIT-2 "Algebraically identical" restructuring is not literal: keep the Fortran's tendency arrays, summation order
  and association.** Adding Redi increments onto absolute S instead of into `del_ttf` is invisible at FP64 but loses the
  tendency at FP32 (ulp(35)=2.4e-6; SP salt error ratio 1.95, 1.02 after the fix), and "the byte gates compare the port
  against itself". In time interpolation, `rdate·coef_a + coef_b` cancels two numbers of about 2.4e6; folding `1/denom`
  into the weights moved the field 6e-8 off the reference. Reassociation also changes the adjoint. *mitjax:* gT/gS/gU
  then the update, as MITgcm does; `exf_interp` and every record-time interpolation keep their association order.
  *Sources:* [P§3] KM/project-m16-sp-gap-root-cause.md:8-21. [F§2] PL:788-800.
- **L-LIT-3 Port the live line, not the commented-out alternative.** In FESOM's MFCT the `valuesAB` gradient at
  `oce_tracer_mod.F90:126` is deliberately commented out and `:127` uses `values`; the port first wired the
  symmetric-looking option. *mitjax:* port exactly what the experiment's CPP options compile, not a nearby `C`-commented
  or `#else` variant. *Sources:* [P§3] P2M/feedback_mfct_gradient_from_values.md:13-20.
- **L-LIT-4 Never delete a Fortran factor to suppress a symptom; a symptom-driven deletion is a deviation.** The `0.5*`
  in `ice_strength` was removed as "spurious"; the Arctic pack locked (ice/ocean speed ratio 0.37 vs Fortran 1.11), and
  restoring the factor gave 0.81→1.09. *Sources:* [P§3] P2M/feedback_ice_strength_halffactor.md:10-30.
- **L-LIT-5 Keep the quirks.** c66g examples: `mom_StartAB=nIter0` gives AB2 weights at step 1 after a pickup; SQRTTWO
  is a literal; SOLVE_DIAGONAL_KINNER ignores iMin..iMax; the calendar keeps its REAL*4 literals; GGL90_CALC_VISC masks
  V but not U; FILL_CS_CORNER_TR_RL works in place; stableGmAdjTap hard-codes its slope limits; the fixed Newton count
  IMAX_TICE=10 is visible only bitwise; the smoother needs a real*4 round trip and the model's nIter0. *Sources:* [E§3]
  PL:116-117, 125, 132-136, 153-156, 252-255, 349.
- **L-LIT-6 Port cost and ctrl quirks literally, and flag them.** V4r4: the stress controls are off for the first two
  weekly records; `glosum` is not implemented; COST_AVERAGESFIELDS samples before CTRL_MAP_INI_GENARR; the adjsen box
  test `YC<=151` is always true, and J is divided by the box volume twice. *mitjax:* expect the same class of quirk in
  the verification experiments' cost functions (Task 16). *Sources:* [E§3] PL:633-643; EI:14-20.
- **L-LIT-7 Every constant cites Fortran file:line and its literal value; small scalar defaults are physics.** FESOM's
  AB2 offset was hard-coded 1e-9 against Fortran 0.1, behind a comment "=1e-9 by convention" that blocked discovery for
  sessions; others were `visc_gamma0=0.003` (not 0.03) and `wsplit_maxcfl=1.0`. Use the reference's truncated constants
  (π=3.14159265358979, not `jnp.pi`; full π seeds 1e-13 per rotation). *mitjax:* MITgcm's AB stabilisation (`abEps`) and
  the other `ini_parms.F` defaults are this class; take the value in effect from the oracle's printed parameter summary.
  *Sources:* [P§3] P2/PORTING_LESSONS.md:33-57. [F§2] PL:36-40.
- **L-LIT-8 Values come from the experiment's `data*` files and the build's preprocessed CPP options, never from code
  defaults; there are no Python-side defaults.** The ECCO port's `RunNamelists.get` raises KeyError unless a cited
  Fortran default is passed, and an unported option is a hard error at setup. In FESOM the namelist beat the code
  default where nobody looked: `ref_sss_local` (Arctic SSS rms halved after the fix), `ice_gamma_fct` 0.25 vs 0.5, FCT
  `scale_area` 2e8 vs 5.8e9 (5.4× ice diffusion), an open-water albedo. Configuration desyncs, not kernel bugs, caused
  the largest climate errors in fesom_jax: `ice_dt=500` against `dt=1800` gave an SST RMS 46× too large, and `ice: {}`
  meant EVP, not mEVP. *mitjax:* build the configuration from the oracle's `data*` files, fail on any parameter that is
  read but missing, derive dependent time steps from the master value (Task 6). *Sources:* [E§3]
  mitgcm_jax/params_io.py:30-36; KG:11-14. [P§3] P2M/feedback_namelist_over_codedefault.md:16-38; PP/methods.tex:15;
  PP/appendix_cport.tex:164. [F§9] PL:1998-2012; PL:4557-4568.
- **L-LIT-9 Use the oracle's exact run directory, and never hand-type parameter tables.** FESOM's `opt_visc` and
  `use_wsplit` came from `work_pi` or `config/`, not the target `work_core`; a bootstrap document ("Default parameters
  (CORE2 from namelists)") listed `opt_visc = 5` and `use_wsplit = .true.` where CORE2 uses 7 and `.false.`, and both
  became dt=1800 bugs. *mitjax:* the truth is the exact `input/` or `input.<suffix>/` directory and the `code/` CPP
  headers the oracle used, recorded in its manifest; generate each experiment's parameter manifest from the oracle run
  (`data*` plus the parameter summary in STDOUT) and diff the configuration against it (Task 6 cross-check). *Sources:*
  [P§0] item 2. [P§1] ~/port2/FRESH_START.md:469-471; P2/PORTING_LESSONS.md:20-21. [P§3] P2/PORTING_LESSONS.md:59-66;
  P2M/feedback_port_used_terms.md:18.
- **L-LIT-10 Port everything used in the run one-to-one, not every option; the per-experiment branch inventory is the
  specification.** A grep count of 0 for a used routine is a gap, and comments admitting a deferral ("not in first
  slice") "are confessions, not documentation". The FESOM C stage "settles which of the Fortran's many options and
  defaults are in force", and the assistant "was especially prone to following plausible but inactive branches"; mitjax
  has no such intermediate stage, so the inventory (CPP flags, `data.pkg`, dispatch values) must be explicit. *mitjax:*
  per experiment, diff the oracle's executed call tree against the port's call log. *Sources:* [P§1]
  PP/methods.tex:4,13. [P§3] P2M/feedback_port_used_terms.md:10,20-26. [P§9] P2M/feedback_port_used_terms.md:10.
- **L-LIT-11 The namelist parser handles every terminator.** V4r4 ends groups with `/`, `&` and `&end`, and the first
  parser silently returned empty groups for `&`. Cross-check the parsed key count with the count of `key =` lines.
  *Sources:* [E§3] PL:53-54.
- **L-LIT-12 Watch for hidden configuration couplings; `data.diagnostics` and `data.pkg` are configuration.** `mult_*=0`
  does not switch controls off (theta still moves by up to 8.7 K); MXLDEPTH in `data.diagnostics` turns on
  CALC_OCE_MXLAYER and changes hMixLayer; `packages_boot.F:206-207` sets useCAL when useECCO is on; the T/S atlases are
  read only when nIter0=0. *Sources:* [E§3] PL:201-204; PL:388-389; EI:43-45; PL:653-654; PL:49-52.
- **L-LIT-13 Check the CPP guards before believing a comment.** The "every step" recomputation of seaiceMaskU/V sits
  under `#ifndef ALLOW_AUTODIFF_TAMC`, so the masks are static in V4r4. *Sources:* [E§3] PL:407-409.
- **L-LIT-14 Know which time level each consumer reads.** Under z*, GAD reads UPDATE_R_STAR's hFacW/S while the State's
  hFacC is one step behind theta; pairing them wrongly gave a 4e-2 heat error. In FESOM a matched restart found the bulk
  formulae reading the surface current one step late and forcing interpolated to the start rather than the middle of the
  step; a monthly read fired one step late, plus a ported "+1", so February was never read (Red Sea RMS 0.357→0.029).
  *mitjax:* for EXF, RBCS, sea ice and periodic forcing, dump record indices and time-interpolation weights each step
  and diff them against the oracle (Tasks 15a, 15c). *Sources:* [E§3] PL:179-180, 292-293. [P§3]
  PP/appendix_cport.tex:164; P2M/feedback_periodic_read_timing.md:14-28.
- **L-LIT-15 Similar routines are not the same routine.** SEAICE_ADVECTION is an older GAD_ADVECTION; reusing GAD's pass
  table left the state unchanged, and only halo-inclusive gates caught it. *mitjax:* one `.py` per `.F`; diff the master
  versions before sharing code. *Sources:* [E§3] PL:332-336.
- **L-LIT-16 An all-zero stage is not an identity.** An EXCH_XY_RS of zero controls rewrote 1538 halo values. *Sources:*
  [E§3] PL:256-257.
- **L-LIT-17 Copy halo loop bounds literally.** Halo write-loop bounds were the dominant multi-rank bug of the FESOM
  ports (5 committed instances; blow-ups at steps ~95–150 at 8/16/32/144 ranks). The worst was fixed halfway: the fluxes
  were exchanged but the Ch/Ce coefficients, read at the halo by sea-ice thermodynamics, were not, and the chain ran to
  Σssh_rhs≠0 (5.6e-6) and a dt²-scaling instability that capped dt at 500. *mitjax:* MITgcm loops run over
  `1-OLx+k..sNx+OLx-k` precisely to save exchanges; exchanging some outputs is not the same as covering every halo
  reader. *Sources:* [P§0] item 7. [P§5] P2/PORT_EXPERIENCE_REPORT.md:81-147; P2/MPI_PORT_REPORT.md:84-92.
- **L-LIT-18 Take stride and shape from the Fortran declaration, never from a loop range.** Using `nl-1` instead of `nl`
  made `sigma_xy` 1000× too large: a 21 m/s bolus velocity, S=59.69 and a CG NaN at step 2. *mitjax:* the index wrapper
  takes bounds from the declaration (`Nr` vs `Nr+1`, `1-OLx:sNx+OLx`) (Task 8). *Sources:* [P§5]
  P2/PORT_EXPERIENCE_REPORT.md:166-180.
- **L-LIT-19 Geometry and orientation are where translation errors hide.** FESOM depth is stored per node but used per
  cell; `l_e` is in radians while `d_ec` is in metres; element orientation had to be normalised (244,654 of 244,659
  swapped), and half the solver's terms depend on that sign. *mitjax:* the analogues are the C-grid metrics
  (dxC/dxG/dxF/dxV, rA/rAw/rAs/rAz, hFacC/W/S) and the exch2 face rotations (u↔v swaps and sign flips); test the
  rotations with an analytic vector field. *Sources:* [P§5] P2/PORT_EXPERIENCE_REPORT.md:208-241;
  P2M/feedback_node_vs_cell_geometry.md:7-20.
- **L-LIT-20 Literal precision and state conventions.** The Intel FESOM build used `-r8`, so `6.6` had to be ported as
  `(double)6.6f`, and replay caught it (1.445e-7); PHC initial conditions are in-situ T but the model needs potential T;
  runoff is already folded into `therm_ice` on the non-ICEPACK path, so subtracting it again double-counts. *mitjax:*
  check the oracle optfile for `-fdefault-real-8` (without it, literals without `_d 0` are default real); trace each
  surface flux from producer to consumer for each experiment's package set. *Sources:* [P§3]
  P2M/feedback_r8_literals_double.md:3,21; P2M/feedback_phc_temperature.md:7-13;
  P2M/feedback_runoff_fold_when_ice.md:12-17.
- **L-LIT-21 An absent input is not a zero input.** `forcing: null` must never become a zero forcing stack, because zero
  air temperature is forcing. *Sources:* [F§9] PL:5496-5501.
- **L-LIT-22 `np.asarray(nc_var[:])` strips netCDF masks and returns the fill value 9.97e36 as data.** *Sources:* [F§9]
  PL:5646-5650.
- **L-LIT-23 Default to "the port is wrong"; an upstream bug needs an exact signature, then a named build switch or an
  approved deviation, and an entry in an upstream-issues file.** FESOM's Fortran skipped the g2r wind rotation for the
  first forcing window at cold start: every EVP input was bit-identical except `stress_atmice`, which was off by exactly
  the local rotation angle (|Δ−θ|=0.000), self-correcting after ~6 steps. In the ECCO port the I6 meta reader is an
  approved deviation, and EXF_MONITOR reads halos that were never filled, so del2 `%MON` values depend on the tiling
  (6.49 vs 3.03). *mitjax:* `docs/ISSUES_UPSTREAM.md` for master. *Sources:* [P§3] P2/FORCING_STEP1_DIFFERENCE.md:9-37.
  [E§3] EI:3-6, 10-13, 22-30. [E§10] EI:3-4.

## L-TOL — Tolerances, comparators, localisation and long runs

- **L-TOL-1 Tolerance classes at the rounding floor, decided per kernel before porting; never loosen a tolerance to pass
  a gate.** Fix a failing gate by fixing the code or the experiment (teacher forcing, replay). ECCO classes: about 1e-15
  for pointwise maps and 1e-13 for stencils, and every ECCO kernel and a 744-step month were bitwise; fesom_jax: about
  1e-15 pointwise and about 1e-12 for scatters and reductions (do not chase a scatter below that; a structured grid
  without scatters can make more kernels bit-exact). A loose acceptance hides real bugs for decades: the FESOM sea-ice
  step accepted at 1e-4 relative hid `scale_area` (5.4×), visible only as +1.1e-6 °C/yr of deep warming over 60 years
  (10× the port-to-port spread); modules never gated at the floor were sea-ice thermodynamics, ice–ocean fluxes, bulk
  heat, sw_pene, SH winter and year boundary/leap day. *mitjax:* every routine gets one replay at the floor; any looser
  acceptance is listed with its reason and revisited before long runs. *Sources:* [E§4] KG:56-60; CAT:85; PL:101-103,
  222-224; VAL:30, 37. [P§0] item 4. [P§4] PP/appendix_cport.tex:162-164; P2M/project_longrun_drift.md:31. [F§2]
  PL:17-22. [F§7] PL:5555-5560.
- **L-TOL-2 Gate solver iteration counts, not only fields.** cg2d stops 0.8 % and LSR 0.7 % below the target residual.
  *Sources:* [E§4] PL:186, 368; EI:67-75.
- **L-TOL-3 Python's `max()` never picks a NaN: compare by element equality and require a finite oracle.** ECCO controls
  written in native byte order were read as garbage (about 415 NaN per record); the Fortran and JAX agreed on that
  garbage, and `max(|diff|)` "passed". A NaN-blind max reduction printed "PASS 0.000" for a dead ocean; `if (resid >=
  rtol)` skipped the solver on NaN, so zombie runs finished 10.8 % faster. *mitjax:* write binaries with
  `.astype('>f4')` and read the file back; invariants count NaN explicitly; `lax.while_loop(resid > tol)` exits
  immediately on NaN, so convergence reports test finiteness. *Sources:* [E§4] PL:647-652. [P§0] item 5. [P§4]
  KD/KPL:172,175,591-609; KDI/KPL:1362-1371,1431-1442,2035-2068.
- **L-TOL-4 A comparator prints what it compared (N files, N points) and fails on zero.** `diff_snap.py` given files
  instead of directories exited "clean"; a byte proof never opened the monthly files the lever wrote; a self-check that
  exchanged the halo restored what it tested; `assert()` vanished under `NDEBUG`. *mitjax:* `python -O` strips `assert`,
  so guards raise explicitly. *Sources:* [P§0] item 5. [P§4] KD/KPL:172,175,591-609;
  KDI/KPL:1362-1371,1431-1442,2035-2068.
- **L-TOL-5 A comparator refuses to run without masks, compares the whole STDOUT, and names each exemption.** The ECCO
  W1 tool ran unmasked because G00 dumps maskC rather than hFacC, and reported land values as differences;
  `tools/diffdump.py` takes its masks from hFacC/W/S and flags ZERO, MISSING and NAN. Compare the whole STDOUT with
  `%MON` as text, name each exemption with its cause, and never `cmp` sparse AD tapes by apparent size. *Sources:* [E§4]
  W1:221-224; tools/diffdump.py:5-15; PL:427-428, 682-683; PL:429-430.
- **L-TOL-6 At dry points compare values, not bits, and mask by the record's level.** Signed exchanges give -0 where the
  Fortran has +0; a surface mask counted +0/-0 at dry levels as differences. *Sources:* [E§4] PL:118-119, 680-681.
- **L-TOL-7 Gate on how many points exceed the ceiling, not only on the maximum.** The FESOM TKE gate failed on 1 of
  5,962,326 entries, and the port was correct. *mitjax:* threshold routines (convective adjustment, KPP, ridging) flip
  isolated points; report the count and the locations. *Sources:* [P§4] KD/KPL:467.
- **L-TOL-8 Report errors per field and relative.** Absolute tolerances across fields spanning nine orders of magnitude
  name the wrong field. *Sources:* [F§5] PL:5562-5567.
- **L-TOL-9 Gate prognostic fields, not kink diagnostics.** The EVP stress flipped O(0.5) on near-rigid elements while
  the u_ice it drives matched to 1e-7. *mitjax:* sea-ice viscosities and stress near the VP yield curve (M4). *Sources:*
  [F§5] PL:2643-2652.
- **L-TOL-10 A product of a huge and a tiny factor carries the huge factor's FMA noise; gate it with an absolute
  floor.** In `slope·taper` with taper→0, a result of about 1e-10 carried about 4e-10 of noise, and whether it shows
  depends on a fusion decision that varies run to run. *mitjax:* GM slope tapering in gmredi (Task 15b). *Sources:*
  [F§2] PL:1522-1528.
- **L-TOL-11 Acceptance differs by stage.** Fortran↔port is judged statistically, and bitwise holds only within one
  toolchain and at a fixed rank count. *mitjax:* gfortran↔JAX by routine replay at 1e-13..1e-16 plus climate statistics;
  JAX P=1↔P=N and backward-only switch off↔on by bitwise gates. *Sources:* [P§4] PP/methods.tex:17-19,72;
  PP/tab_ladder.tex:5-10.
- **L-TOL-12 Bit-identity is a CPU property; on GPU compare against a measured floor.** In fesom_jax the 1-device
  sharded path matched the dense path only to 7.66e-9 on GPU (about 0 on CPU), and an inert flag moved four GPU results
  by 1 ulp, because two jitted closures are different programs and autotuning depends on process state. *mitjax:* assert
  `==` on CPU; on GPU use a measured floor (relative about 1e-14 per step). *Sources:* [F§2] PL:2632-2643; PL:5771-5777.
- **L-TOL-13 Multi-GPU forced runs are nondeterministic at round-off (atomic scatter-adds), and on a stiff cold start
  fresh-vs-fresh differences grew to O(1e3).** Validate a performance change against fresh-vs-fresh noise, not against
  zero; restart bit-identity holds only on deterministic paths. *mitjax:* prefer scatter-free strip exchanges and
  tile-ordered sums. *Sources:* [F§2] PL:4522-4527.
- **L-TOL-14 Sea ice under production XLA flags can only be compared statistically.** Exact-zero branches flip (HSNOW
  -2e-58 vs 0) at about 30 points by iteration 3; the ECCO criterion is that every `%MON` value stays inside the spread
  of the Fortran 96-rank and 13-rank runs. *Sources:* [E§4] PL:417-419, 447-450; VAL:82-86.
- **L-TOL-15 Every scheme has its own floor: measure Fortran against Fortran before judging the port.** mEVP (α=β=250)
  is only approximately converged, so its C-vs-Fortran `uice` floor is ~0.95 while EVP's is ~1.0. *mitjax:* for sea-ice
  solvers and cg2d at finite tolerance, measure Fortran-vs-Fortran (e.g. another tile layout) first. *Sources:* [P§4]
  KD/KPL:486,494.
- **L-TOL-16 A control must be shown to differ, and one control is not an envelope.** At 4 ranks two "different"
  partitions coincided (rms 5.9e-7), so the control measured round-off; one control put an arm at 2× the class while
  four spanned ×2.7. FESOM's drift yardstick was a Fortran ensemble differing by fill or partition (−0.011 mK deep
  offset, no growth, ≤0.23 µK/yr), while C−F grew 4–5× faster. *mitjax:* use ≥3 Fortran runs that differ only by tile
  layout or a tiny perturbation, and quote the envelope (tier-3 twins). *Sources:* [P§4] KDI/KPL:1601-1617. [P§7]
  PP/appendix_cport.tex:162-164; P2M/project_longrun_drift.md:18-31.
- **L-TOL-17 Run at production dt, for model years, and past the old failure.** Six mis-ported FESOM terms passed dt=500
  and failed at 1800; "a 'clean 5000-step' run hid a day-110 blow"; a 2-year dt=500 run matched T to ~0.003 °C with
  momentum advection entirely missing ("Large-scale validation does NOT catch a missing term that matters for grid-scale
  stability"). *mitjax:* each verification experiment runs at its own `deltaT` for its full `nTimeSteps`; the `%MON`
  statistics in `results/output.txt` are too coarse to replace multi-year twins (tier 3). *Sources:* [P§0] item 3. [P§7]
  P2/PORTING_LESSONS.md:10-29,139-140; P2M/feedback_port_used_terms.md:20-22.
- **L-TOL-18 Localise before hypothesising: find the first diverging substep, field and global index in the dumps before
  reading code.** The failed FESOM port instrumented the Kuril trench for a session while the blow-up was at the
  Aleutian trench, 12° east. Means that worked in the ECCO port: replay intermediate dumps (a 6e-14 error was 1-ulp
  input differences amplified by the implicit solve); run gates a second time with glibc injected through a test-only
  `pure_callback` to tell a non-literal port from a libm difference; for FD, subtract outputs pointwise before
  weighting. *mitjax:* monitors print the global (tile,k,j,i) and lon/lat of every extreme. *Sources:* [P§0] item 6.
  [P§1] FPM/project_aleutian_not_kuril.md:35. [E§4] PL:169-170; PL:358-359; PL:120.
- **L-TOL-19 At a blow-up, find what leads, and sweep dt.** At the FESOM eruption node the per-step 1-ring spreads
  showed `velsprd` growing while `pgf`/`rho` stayed flat, which proved a free velocity mode; the blow-up day against dt
  (1800→d112, 1740→d155, 1500→d399, 1200 stable) was a marginal mode, not a clean CFL cliff. *mitjax:* log the 2Δx
  roughness of every tendency input at a blow-up point; deltaT is a driver argument, so a sweep needs no edit.
  *Sources:* [P§2] P2/PORTING_LESSONS.md:125-128,135-137; P2/DT1800_HANDOFF.md:153,229-236.
- **L-TOL-20 Diagnostic interventions must preserve invariants.** Clamping `fer_uv` and `fer_w` separately broke
  continuity and caused an FCT overshoot to S=59.69 at small CFL, while scaling both uniformly ran clean; removing a
  net-stabilising term moves a blow-up earlier, which is not evidence of cause. *mitjax:* to switch a term off, skip the
  whole block or scale its source; never half-apply it. *Sources:* [P§2] P2M/feedback_bolus_divergence_balance.md:9-20;
  P2/PORTING_LESSONS.md:130-133.
- **L-TOL-21 A failure step that varies between runs points to near-threshold physics or the platform, not to a fixed
  code bug.** "A real instability in the physics dies at a fixed step." Identical NG5 runs died anywhere from step 4 to
  step 291; a cudaMallocAsync pool used with CUDA-aware MPI caused random-step crashes that were read as CFL for a
  month; NG5 `dist_4096` (3-D balance 13.89×) blew up while a regenerated 1.05× partition completed, which retracted a
  recorded "physics limit shared with Fortran". *mitjax:* run every failing GPU case twice before diagnosing, and try a
  second tile layout before blaming dt or physics. *Sources:* [P§6] KM/feedback-wsplit-on-large-meshes.md:18-20;
  PP/tab_pitfalls.tex:15; KDI/KPL:1993-2034.
- **L-TOL-22 Magnitudes hide vector-direction bugs.** Examples: the missing g2r wind rotation; rotated-frame output
  compared with geographic output gave a fictitious `uice` correlation of 0.92 (0.9997 once rotated). *mitjax:* check
  vectors by magnitude ratio and by signed angle against AngleCS/AngleSN, and compare outputs in the same frame.
  *Sources:* [P§5] P2M/feedback_magnitude_invariant_masks_rotation.md:10-27; KD/KPL:461.

## L-XLA — JAX/XLA numerics

- **L-XLA-1 No FMA in bitwise CPU gates: `--xla_cpu_max_isa=AVX`.** XLA:CPU at AVX2 contracts `a*b+c` into an FMA, while
  the oracle is built with `-ffp-contract=off`. The flags are x86-only, so CPU gates stay on x86. On GPU, Triton emits
  `.rn` operations with no FMA, so Pallas kernels can be bitwise, and the Pallas interpreter gates their operation order
  on CPU. *mitjax:* `mitjax/xla_flags.py`; the fesom_jax finding that nothing stops FMA is superseded, see [L-CONF-1].
  *Sources:* [E§5] PL:96; KG:76-78; ENV:98-99; PL:463-464, 470-471.
- **L-XLA-2 No algsimp in bitwise gates.** XLA's algebraic simplifier folds `(x*c1)*c2`, `c+(y-c)` and `x/d → x*(1/d)`,
  and rewrites `A/sqrt(B) → A*rsqrt(B)` only where the sqrt has a single user; gates run with
  `--xla_disable_hlo_passes=algsimp`. "Bitwise eager, not under jit" means a rewrite: dump the HLO and grep for rsqrt
  and divide. *Sources:* [E§5] PL:97-100, 157-158.
- **L-XLA-3 fesom_jax: XLA:CPU's float64 divide is not correctly rounded.** It was 1 ULP off numpy on about 13 % of
  operands, while CUDA's `div.rn.f64` is IEEE-correct; XLA also folds `x/1000` into `x*0.001`, and fast-math flags did
  not help. fesom_jax's advice: keep divisors as runtime operands where the Fortran divides by a literal, and probe the
  platform divide inside the test. *mitjax:* open until Task 8 measures it, see [L-CONF-2]. *Sources:* [F§2]
  PL:5260-5268.
- **L-XLA-4 Float parameters reach jit as traced arguments, never closed over.** Closed-over floats changed 42 % of 1e5
  values; the ECCO port's `params_pytree` makes float fields traced leaves. No optimization barriers in kernels (but see
  [L-CONF-7]). *Sources:* [E§5] mitgcm_jax/params_io.py:77-82; KG:47-48.
- **L-XLA-5 One `XLA_FLAGS` string, set once, before the first backend initialises — in conftest and in every standalone
  gate script.** The device count must be set before JAX initialises: a fesom_jax notebook that set it later failed with
  a broadcasting error that mentioned neither devices nor partitions. An unknown `XLA_FLAGS` entry is a hard abort, so
  flag spellings can be probed offline; pass names come from a 5-line toy dump. *mitjax:* `mitjax/xla_flags.py`,
  `conftest.py`, checked by `mitjax/tests/test_env.py`. *Sources:* [E§11] PL:11-13; KG:74-78. [F§8] UR:216-220. [F§1]
  PL:5390-5395.
- **L-XLA-6 `0.0 + x` is folded even with algsimp off.** Fortran's `ZERO + x` maps -0 to +0; emulate it with `x +
  stop_gradient(where(x==0, -x, 0))`. Read `(c+t)-c` shift tricks from the bit pattern. *Sources:* [E§5] PL:348,
  697-699.
- **L-XLA-7 The reduction order is not the obvious one.** `jnp.sum` becomes a reduce-window that LLVM vectorises per
  shape. Write fixed-order sums as explicit chains of adds, and test any emulated order on random data with zeros and
  -0. *Sources:* [E§5] PL:561-571.
- **L-XLA-8 Call what the binary calls (libm).** gcc fuses SIN/COS into `sincos` (2 ulp at 70 points); glibc 2.28's exp
  is not correctly rounded (XLA differs at 14 % of arguments); gfortran 11 vectorises some loops to libmvec
  `_ZGVbN2v_exp`. Check with `nm` and `objdump` and transcribe into `ops/libm.py`. numpy's SIMD trig is not glibc, but
  Python's `math.sin`/`math.cos` are. *mitjax:* re-verify with whatever gfortran and glibc build the master oracle (Task
  7c). *Sources:* [E§3] PL:106-108, 345-348, 357-362, 376-378, 700-703. [E§5] PL:106-107.
- **L-XLA-9 Code-generation context changes bits, so two compiled programs need not agree in the last bit.** In the
  FESOM C port an uncalled helper added to `fesom_mesh.c` shifted `pgf_y` by 3e-18 at step 0 (FMA contraction is decided
  per translation unit), 1e-2 after 200 steps. XLA fusion is decided per jitted program, so adding a diagnostic output
  inside jit can change last-ULP results, and `jax.debug.print` contaminates the graph; fesom_jax compares two programs
  against the measured spread of remat on/off. *mitjax:* keep diagnostics outside the step function or prove they do not
  change bits; where the plan nevertheless requires bitwise equality of two programs, see [L-CONF-4]. *Sources:* [P§6]
  P2M/feedback_tu_codegen_bitgate.md:11-23. [F§2] mem:pj/adjoint-lessons-program.md:32-33. [F§1] PL:5377-5382.

## L-ARCH — JAX program structure

- **L-ARCH-1 Fixed-shape dense arrays plus masks, one `lax.scan` over the step, static configuration.** Ragged columns
  carried as masks let XLA fuse the whole step, and an inactive scheme is absent from the program. *mitjax:*
  `[tile,k,j,i]` padded arrays with hFac masks; every `#ifdef`/`useXXX` becomes a static Python branch (`cfg.<NAME>`).
  *Sources:* [F§1] PAPER:55-67.
- **L-ARCH-2 Step 1 runs eagerly outside the scan, and the scan carry is the State only.** `is_first_step` outside the
  scan equals the loop bit for bit; `jax.checkpoint` on/off is forward-transparent (Δ=0.0); `scan` sums the cotangents
  of closed-over params. *mitjax:* MITgcm's AB start-up cases (first-step AB bootstrap, `staggerTimeStep`) go outside
  the scan; the body stays uniform with no traced bools (Task 11). *Sources:* [F§1] PL:506-514.
- **L-ARCH-3 Static configurations are hashable NamedTuples, and the off path is bit-identical by construction.**
  `ice_cfg=None` gave a compile-time dead branch, and promoted constants kept 43+90 gates unmoved. A NamedTuple cannot
  override `__new__`, so consistency checks run in a `validate()` at the driver seam, not at construction. *mitjax:* one
  "off = identical" test per option; on defaults inside configurations see [L-CONF-11]. *Sources:* [F§1] PL:1408-1416;
  PL:4191-4198; PL:2808-2812; PL:3465-3473.
- **L-ARCH-4 Never pass a static configuration as an argument of a differentiated function.** It is flattened as a
  pytree and its bools become tracers, which surfaced as a TracerArrayConversionError deep inside the SSH solve; to
  differentiate one field use `cfg._replace(field=tracer)`. *mitjax:* drivers close over configurations; only
  params-style pytrees are arguments. *Sources:* [F§1] PL:6182-6184; PL:1960.
- **L-ARCH-5 Large arrays and the model are jit arguments, never closures.** A fesom_jax loss that closed over 240 steps
  of forcing and the targets captured 4.58 GB of constants per executable (a 49.5 GiB OOM on an 80 GB A100 with three
  jitted functions) and made XLA constant-fold scatters (about 20 s of compile); in the ECCO port, closed-over model
  data was constant-folded (275 s compiles and a CUBIN too large), and the model, Exchanger included, is a pytree jit
  argument, for the tangent-linear model too. *mitjax:* forcing, grids and observations enter jitted drivers as
  arguments; grep lowering logs for "constants were captured". *Sources:* [F§1] PL:6079-6100; PL:3840-3845. [E§11]
  PL:262-264, 323-325; AR:268-271.
- **L-ARCH-6 `lru_cache` that returns `jnp` arrays leaks tracers between traces (UnexpectedTracerError); cache numpy and
  cast per call.** *mitjax:* grid and table builders (hFac, lookup tables, exch2 maps) return numpy. *Sources:* [F§1]
  PL:1924-1931.
- **L-ARCH-7 Strong-type every State, driver and optimizer input.** Weak types from `jnp.full` survive jit and compiled
  a second program in the ECCO port; in fesom_jax a weak-typed `jnp.asarray(1.0)` init recompiled at iteration 2, the
  second CUBIN failed to load ("RESOURCE_EXHAUSTED … CUBIN"), and 8 GPU reruns blamed the allocator. *mitjax:* on any
  mid-loop OOM, count traces before touching memory settings. *Sources:* [E§5] PL:445-446. [E§11] PL:445-446. [F§1]
  PL:3801-3811.
- **L-ARCH-8 Audit the scan carry against what the step reads and writes.** Two never-read fesom_jax State fields rode
  the carry as zeros, about 145 MB per checkpointed step at CORE2. *Sources:* [F§1] CR:162-166.
- **L-ARCH-9 One shared cold-start/pickup helper: re-implemented drivers drift from the validated setup.** A missing
  `seed_ice`, a missing global coast mask and a missing forcing-year rollover were each invisible to step-level gates.
  *mitjax:* one cold-start path used by every driver; diff end-to-end runs against a known-good run (Task 11).
  *Sources:* [F§9] PL:4366-4386; PL:4388-4400.

## L-AD — Differentiability

### Masked lanes, NaN and kinks

- **L-AD-1 Masked, halo and padding lanes compute finite values: guard before the operation.** Put `jnp.where(mask, x,
  1.0)` before a division or sqrt, because a forward `where` does not stop a backward 0·inf. In fesom_jax this bit 4×
  dense and needed 7 more guards in 5 kernels once padding lanes appeared. *mitjax:* `ops/safe.py` (Task 7c); grep for
  `1/`, `sqrt`, `pow`, `segment_max/min` (±inf on empty segments) and duplicated-tail concatenations. *Sources:* [E§6]
  KG:15-17; CAT:111. [F§3] PL:482-495; PL:2559-2583.
- **L-AD-2 The sharded backward does not fold the 0·inf that single-device XLA folds silently.** A kernel can be
  NaN-clean in the dense gradient and NaN in the sharded one. *mitjax:* run the masked-NaN probe on the sharded path
  too, with blank tiles and padding lanes present. *Sources:* [F§3] PL:2559-2566.
- **L-AD-3 JAX's division JVP forms `b**-2`, which underflows for |b| < 1e-154 to 0·inf = NaN.** HSNOW reached
  -1.2e-240; the fix is a quotient-rule custom_jvp. Stacked fields share the uTrans cotangent, so one NaN lane poisons
  every field. *Sources:* [E§6] PL:337-339. [E§11] PL:337-339.
- **L-AD-4 The gradient with respect to whole initial fields (padding included) is the day-one gate; it is strictly
  stronger than a scalar-parameter gradient.** In fesom_jax `d/dk_ver` was finite and FD-correct while `d/dT0` was NaN,
  because `k_ver` enters additively and never crosses the poisoned path. Nikolay requires a model differentiable from
  day one, with a gradient check in the fast suite. *mitjax:* `grad` with respect to full theta/salt/uVel/etaN; one
  full-field gradient finiteness test per milestone in tier 1. *Sources:* [F§3] PL:497-504. [E§10]
  mem/project_differentiability_requirement.md:7-19.
- **L-AD-5 Debug NaNs with `jax_debug_nans` and a cheap probe, one trap per run.** It halts at the first NaN, harmless
  ones included; a `d/dT0` probe reaches every kernel. *Sources:* [F§3] PL:2585-2593.
- **L-AD-6 JAX splits kinks 0.5/0.5, unlike TAF, and central FD at a floor value needs h below the floor.** *mitjax:*
  see [L-CONF-12]. *Sources:* [E§6] PL:340; PL:600.
- **L-AD-7 Limiters are differentiated as written (subgradient).** A smooth relaxation changes the forward model, and
  `stop_gradient` on the limiter factor biases the gradient. *mitjax:* DST3FL/OS7MP and the other limited schemes of the
  verification experiments; test where the limiter is active (Task 14). *Sources:* [F§3] LG:36-45.

### Solvers

- **L-AD-8 Never differentiate through solver iterations, and verify a derivative solve by its residual, not against
  another run of the same solver.** fesom_jax checked its CG derivative by ‖S·g − x‖ (8.8e-14). The ECCO port checks its
  implicit derivatives by FD against a converged forward. *mitjax:* cg2d (and later cg3d, sea-ice solvers) by implicit
  derivatives with fixed iteration counts. *Sources:* [F§3] PL:1163-1177. [E§6] PL:268-271, 520-523.
- **L-AD-9 `custom_linear_solve`'s own JVP runs the loose, warm-started primal solve, so the tangent is wrong: use an
  explicit implicit-derivative rule `dx = A⁻¹(db − dA·x)` (custom_jvp) with tight tangent and transpose solves.** ECCO:
  dot test 1.9e-9 → 1.3e-11. fesom_jax: CORE2 one-step dot test 3.4e-2 → 3.1e-12, forward bitwise unchanged, about 345
  tangent CG iterations against 131 forward. *mitjax:* `ad/cg2d_rule.py` (Task 7c). *Sources:* [E§6] PL:268-271. [F§3]
  PIDX:143-145 (commit f56cb9a).
- **L-AD-10 Zero the halo lanes of every derivative output of a solve on overlapping tiles, and gate cotangents next to
  tile edges.** A sharded fesom_jax solve declared `symmetric=True` returned PCG partial sums on halo lanes; the reverse
  exchange scatter-added them onto owners, so owner cotangents next to partition boundaries were 48–63 % wrong, while
  parameter gradients stayed inside the old 5e-3 sharded-vs-dense gate for months. The fix masks the derivative solve to
  owned lanes, which gives the symmetric E S⁻¹ Eᵀ, with a lane-space dot test (4e-14 at amplitudes 1 and 1e-8).
  *mitjax:* Task 7c rule tests and the tile-edge cotangent gates of Tasks 12, 13 and 17. *Sources:* [F§3] PIDX:146-149;
  commit b51790e message.
- **L-AD-11 Differentiate the converged system, and check FD against a converged forward.** The ECCO sea-ice LSR's
  implicit derivative uses GMRES(40) with 8 fixed cycles, a line-SOR preconditioner and `linear_transpose`; FD against
  the production LSR_ERROR 2e-4 is off by 1-330 % and shows a constant 3e-3 bias. A fixed Newton count gives the
  derivative of the iterate unless it converges. *Sources:* [E§6] PL:371-375, 480-481, 520-523; AM:266-271; ADXH:42.
- **L-AD-12 Fixed iteration counts make the backward defined, not well conditioned.** Newton (5 iterations) showed no
  amplification; mEVP (120) gave σ_max 1.6e7 per ice step at α=β=250 — the iteration count is the exponent, α and β set
  the base. *mitjax:* measure the amplification of sea-ice LSR/EVP adjoints before trusting any unroll (M4). *Sources:*
  [F§3] REF:20-40.

### Backward-only switches (TAF semantics)

- **L-AD-13 "Package off in AD" is set by where TAF's STOREs sit.** For GGL90 it means `stop_gradient` on its outputs
  ("frozen"), not "recomputed without GGL90". *Sources:* [E§6] PL:210-213; AM:59-63.
- **L-AD-14 A TAF-skipped block is the identity on the variables it overwrites and zero on the ones it only reads; it is
  not a `stop_gradient`.** The ECCO port's `ops/ad_skip.py` implements it as a linear custom_jvp; `viscFacInAd` is a
  custom_jvp with the value at the arguments and the tangent at `alt`, bitwise plain AD when alt = args. *Sources:*
  [E§6] PL:412-414; PL:214-215; AM:180-183. [E§11] ad_skip.py:3-21.
- **L-AD-15 The ECCO port's TAF semantics were never checked against TAF output.** No TAF-generated code existed; the
  semantics were read from CADJ and `.flow` files (cg2d.flow: passive operator, TAF adjoint solve to 1e-7 vs 1e-13 in
  JAX), and "confirm on a TAF build" stayed open. `stop_gradient` also cuts tangents, whereas TAF's TLM keeps every
  package. *mitjax:* master's `results/output_adm*.txt` are the missing check — match them before claiming any TAF mode
  (Task 16). *Sources:* [E§6] AM:9-11, 190-202, 253-256; AM:258-260.
- **L-AD-16 Every derivative-only switch needs three tests and exactly one implementation.** The tests: the empty switch
  gives the same sha256 of the jaxpr text (addresses normalised); the forward stays byte-identical; the switch has an
  effect on a live fixture; a seam census counts each switch's equations. fesom_jax had two freeze-mixing
  implementations (a flag and a monkeypatch) that went unchecked against each other until a test showed max|Δ|=0, and a
  spelled-out monkeypatch signature silently killed `--freeze convect`. *mitjax:* TAF-compatible switches
  (`use*InAdMode`, `GMREDI_WITH_STABLE_ADJOINT`) are library flags, never monkeypatches (`ad/modes.py`, from M3).
  *Sources:* [E§6] HC:10-13; PL:215, 738-742; AM:241-242. [F§3] PL:6144-6168; PL:6208-6226.
- **L-AD-17 Option names state what is enabled or cut.** fesom_jax's `--ice rheology` meant "ice on, rheology sweeps not
  unrolled"; it was read the other way, so every rung from 60 to 731 days was measured ice-free. *mitjax:* name backward
  switches after what they cut (`freeze_X`, `X_in_ad_mode`). *Sources:* [F§3] mem:ne/ice-rheology-naming-trap.md.
- **L-AD-18 Never branch on traced parameters inside a rule; a custom_jvp cannot close over traced values either.** `if
  p.SINWAT != 0` passed every CPU test that closed over the parameters and failed only in the GPU drivers
  (TracerBoolConversionError). *Sources:* [E§6] PL:704-705, 735-737; W4:175-178. [E§11] PL:735-737.
- **L-AD-19 A trace-time cut context must also enclose the derivative trace.** Custom rules trace lazily, so the ECCO
  stage-2 result "no momentum cut matters" came from the uncut program. Diagnostic cuts passed as static arguments
  compiled 45 times; traced 0/1 weights compile once. *Sources:* [E§6] PL:799-802; W5:306-309; PL:706-708. [E§11]
  PL:799-802.
- **L-AD-20 Freeze a closure's ocean-state inputs, not its prognostic state.** The fesom TKE closure builds Kv from the
  OLD tke, and wind stress enters only through the tke update, so freezing `tke` severed wind→mixing however live the
  stress was; `uvnode` also had to be recomputed from live velocity, or the current→stress feedback died. *mitjax:*
  GGL90 has the same structure (prognostic TKE); how this relates to TAF's frozen outputs is [L-CONF-6]. *Sources:*
  [F§3] OC:781-792.

### Horizon and instability

- **L-AD-21 The exact flux-forced ECCO adjoint is clean to 112 days and first fails at 130-150 days.** The cause is one
  linearised-GGL90 event in the eastern equatorial Pacific; GGL90 frozen alone is stable for a year; cutting sigma alone
  is worse (the opposite of fesom_jax). The burst comes from Km ~ sqrt(e) near TKEmin, not from low Ri. A single cut can
  make things far worse: the shear, buoyancy and implicit-diffusion derivatives nearly cancel, and cutting one took the
  1-day growth from 976 to 1e16-1e30. *mitjax:* relevant from M3 (vermix experiments). *Sources:* [E§6] AR:330-341;
  notes/HANDOFF-20260924.md:25-28; PL:589-594.
- **L-AD-22 Calibrate a freeze on the loop, not on one variable.** A one-step self-gain freeze failed twice (GGL90 and
  the McPhee taper), and freezing more is not safer (growth went from 0.03 to 106). Calibrate on the spectral radius of
  the coupled block or the two-step gain. *Sources:* [E§6] HC:14-19; PL:604-609, 720-724; ADXS:197-210.
- **L-AD-23 Medians never see the event: screen the worst 3 consecutive chunks.** Median and log-spread go blind past
  about 100 chunks; repeats and dot tests cannot see a deterministic instability; measure growth by power iteration on
  the window map. *Sources:* [E§6] AR:342-344; PL:583-585. [F§3] PL:6228-6255.
- **L-AD-24 Window-mean losses built from few samples are near-differences, and 12-h segments alias the diurnal cycle.**
  A 1-day truncated gradient reproduced only to 3× (two samples of opposite sign). *Sources:* [F§3] OC:1015-1043.

### Gradient trust

- **L-AD-25 The gradient trust bars.** An FD plateau, TL/adjoint agreement of about 3e-13, an amplification screen
  (median ≤ 1.010/step, worst-3 ≤ 1.030), a bitwise forward across modes, and repeats. ECCO-style freezes are not free:
  6-25 % off FD outside the box, and an ecco-mode sea-ice cost has no path to the atmosphere. *mitjax:* the gradient
  trust protocol of Tasks 16 and 17. *Sources:* [E§6] PL:317-319; mitgcm_jax/adjoint/grad.py:23-30; AR:227-235, 264-267;
  PL:512-514.
- **L-AD-26 Pass-through State fields fake growth.** runoff and sIceLoad accumulate cotangent (1.0105/step); find such
  fields from the jaxpr and use a terminal seed. The chunk that ends at the AB start loses the gu/gvNm cotangent.
  *Sources:* [E§6] AR:251-262; PL:320-321. [E§11] AR:251-262.
- **L-AD-27 The norm decides what you see.** A Euclidean norm over every field is dominated by unit conversions into
  diagnostic fields (sigma 1e5 in a quiet window); mixed-unit modes mislead; the norm must cover only propagating state
  (TICES alone gave a spurious 1e4). *Sources:* [E§6] PL:586-588, 709-711, 793.
- **L-AD-28 FD plateaus exist only in smooth regimes.** Through a realistic forced fesom_jax trajectory, CORE2 N=20 FD
  swung from −15 to +222 against AD +33.5, while N=1 plateaued at 7.5e-10. An FD error that does not shrink with h is a
  switch limit: TFLUX and air-temperature footprints have many switches, TL = adjoint there, and single-column
  directions work instead. *mitjax:* small smooth verification experiments for FD gates; finiteness and subgradient
  checks for the full model. *Sources:* [F§3] PL:1141-1160. [E§6] AR:238-242; PL:515-519.
- **L-AD-29 A constant FD bias at every h is not switch noise; TL and adjoint can linearise different trajectories;
  linear ranges can be tiny.** The jvp program fuses differently, and exact-zero branches gave a 5e-10 TL/adjoint gap;
  linear ranges were 1e-9 over a day at TKEmin and about 1e-4 m of ice at the melt MAX — predict them from the forward
  state. *Sources:* [E§6] PL:520-523; PL:524-526; HC:24-26; PL:597-599, 788-789.
- **L-AD-30 The FD floor is set by the size of the loss's intermediate sums, so the plateau sits at large h: lift the
  signal rather than shrinking h; and choose an observable that depends on the parameter.** Barotropic η is physically
  insensitive to vertical viscosity (−7e-17, correct). *mitjax:* applies to grdchk comparisons against
  `output_adm*.txt`. *Sources:* [F§3] PL:517-528; PL:1163-1170.
- **L-AD-31 FD agreement is not proof of correct physics.** Classify every unexplained feature as physical, cost
  artefact, bug or instability. An unweighted mean cost is not conserved, and the Coriolis adjoint rotates λv into a λu
  "burst". *Sources:* [E§6] mem/feedback_fd_not_proof.md:10-21; PL:809-812; PL:813-814.
- **L-AD-32 The GPU program itself can be wrong: guard with vjp(0)=0 and vjp(2ct)=2·vjp(ct) on the first reverse
  chunk.** The ECCO Pallas LSR kernel mapped a zero cotangent to 6.98e6 in some chunk programs, costing 5 A100-hours.
  *Sources:* [E§6] PL:803-808; BS:468-490; W5:311-312. [E§11] PL:803-808.
- **L-AD-33 Exact-mode GPU gradients need repeats; CPU is the reference for exact gradients.** On an A100, round-off
  flips discrete switches (ice skin temperature, the convective Kv switch) within 1–6 steps, even between two runs of
  the same program; the exact 1-day TKE-constant gradient changed sign over five sweeps; the frozen `long_window()`
  configuration repeats to 2e-10; deterministic GPU ops took over 2 h to compile. ECCO repeat floors: GPU 4e-11, exact
  sea-ice modes 9e-11. *mitjax:* sharded vs single-GPU gradients are compared within the measured repeat floor (Task
  17). *Sources:* [F§2] PIDX:153-156. [E§6] PL:275; AR:884. [E§11] (checkpoint stack: repeat floors 4e-11 / 9e-11).

### Checkpointing and gradient drivers

- **L-AD-34 Scan with a checkpointed body, never a Python loop of jitted steps; build the chunked VJP API on day one.**
  In the ECCO port the step, sqrt and chunked schedules (boundaries on the host) give identical gradients, device memory
  stays flat at about 42-46 GB, and a gradient costs 4.3 forwards. In fesom_jax, in-step block remat plus a per-step
  checkpoint plus a host-parked chunked reverse kept device memory flat in window length (43.1 GB at 1/30/60 d); without
  in-step remat a single reverse step was about 34 GiB. *mitjax:* `drivers/checkpoint.py` (Task 7c). *Sources:* [E§6]
  checkpoint.py:28-31; AR:243-248; PL:272-274. [F§4] PL:5621-5627; AA:131-135.
- **L-AD-35 Jit a function that calls vjp, not the function vjp returns.** The latter captured 3 GB of residuals as
  constants. *Sources:* [E§5] PL:147. [E§11] PL:147.
- **L-AD-36 An API knob that looks like the fix but makes things worse should not be public.** In fesom_jax, segments >
  √N died at 69.24 GiB. *Sources:* [F§4] AA:131-135.

## L-PAR — Parallel and sharding

- **L-PAR-1 One code path, and the forward is bitwise at P=1 and P=N: sharded-on-1-device == dense is the invariant that
  verifies the distributed code.** In the ECCO port, probed exch2 maps, replica padding and fixed tile-order sums gave
  P=N bitwise equal to P=1 once the check_vma typing traps were handled. np=1 bit-identity is necessary, not sufficient:
  np=1 skips scatter, exchange, gather and halo initialisation, and a byte-identical owned state means the bug is in
  gather or output, not in physics. *mitjax:* compare sharded and unsharded per-tile interiors before gathered output.
  *Sources:* [F§5] PAPER:115-117. [E§7] PL:237-248, 534-537. [P§4] KD/KPL:296-318.
- **L-PAR-2 Probe the Fortran exchange instead of re-deriving it, with the halo points coded too.** Code every point,
  run each exch2 routine, decode source, component and sign; the two-pass corner update comes for free; in LLC90 1568 of
  19552 halo points are never written. MDS_FACEF_READ fills the sNx+1 row and the exch2 corner pass copies halo points,
  so a probe with zero halos missed them; a point that is never written can still be read. *mitjax:* Task 4 probes exch2
  and the eesupp (exch1) exchanges for every M1 layout. *Sources:* [E§7] PL:77-84; PL:109-112; PL:248.
- **L-PAR-3 To prove who reads a value, poison it and change nothing else.** In Kokkos, toggling syncs also removed
  fences, which confounded the test; overwriting the host copy with NaN while keeping every sync found the one real
  reader (the other three syncs were placebos), and a dead-halo poison test removed `uv_rhsAB` at no cost. In the ECCO
  probe, poisoning gave the exact dependency mask. *mitjax:* fill halo rows with NaN before a routine to prove the exact
  halo width it reads (the standing halo-poison gate). *Sources:* [P§6] KD/KPL:322,443; PP/appendix_kokkos.tex:97-110.
  [E§7] PL:109-112.
- **L-PAR-4 A halo value computed instead of received must be bitwise the owner's.** Recomputing `H0e` at halo elements
  read `eta0[-1]` and left 1334 copies wrong by up to 0.1 m, dead state until a lever made them live; a seed reduced by
  9 orders of magnitude still grew ×5.35 per step ("acceptance … is bitwise-zero drift, not small drift"). gcc
  vectorises transcendentals, so "local recompute of anything transcendental is never byte-safe". *mitjax:* MITgcm
  computes in overlaps by design and XLA may fuse or vectorise interior and halo differently: test P=1 vs P=N bitwise
  per routine; where the Fortran exchanges, the port exchanges. *Sources:* [P§0] item 7. [P§6] KDI/KPL:1403-1430;
  KD/GPU_SPEED_M7.md:1163-1167.
- **L-PAR-5 Turn index conventions into a startup census.** A Kokkos census found 0 owned-reachable `-1` references at 7
  partitions (0.08 node-hours) and the 1 violating site among 45 read sites. *mitjax:* at startup, check that every
  exch2 halo index resolves to a valid owner with the right rotation, corners included; abort otherwise. *Sources:*
  [P§6] KDI/KPL:1444-1459.
- **L-PAR-6 Padding replicates tile 1.** Keep cotangents on the padding zero in checks, because they leak through the
  exchange transpose; the padded operator is not symmetric, so zero the padding in transpose solves; replicated params
  must not carry a tile axis. *Sources:* [E§7] sharded_exchange.py:9-14; PL:244-245, 579-580. [E§11] PL:244-245,
  579-580; shard.py:22-23.
- **L-PAR-7 Only collectives with trusted transposes: `ppermute`, `psum`, `all_gather`, `all_to_all`;
  `ragged_all_to_all` is banned.** Its transpose is wrong (O(1), sign flip at P=2), still wrong in jax 0.11.1 by a
  third-party report, and no collective-transpose fixes are listed up to 0.11.2. *mitjax:* banned on every
  differentiable path, with a guard test. *Sources:* [E§7] PL:239-243; PR:24-25. [F§5] RAG:16-24. [F§8] LPA:40;
  LPA:395-397.
- **L-PAR-8 Exchanges are greedy-coloured `ppermute` rounds; global sums are a psum of a zero-padded buffer followed by
  an ordered sum; `lax.pmax` has no JVP.** ECCO at P=4: 69 permutes, 9 all-reduces, no all-gather or all-to-all. When a
  primitive is broken, re-lay the data until a trusted one fits instead of patching it: in fesom_jax coloured ppermute
  was AD-correct and the fastest at NG5-64 (543 ms against 592 and 742). *Sources:* [E§7] PL:239-243. [F§5] PAR:18-20;
  PL:5097.
- **L-PAR-9 Find the hidden cross-tile dependencies.** calc_r_star's counters and the cg2d max are global reductions;
  per-tile tables are keyed by global tile number (Grid.tile_index); shapes come from L.nTiles. *Sources:* [E§7]
  PL:246-247.
- **L-PAR-10 check_vma typing traps: keep `check_vma=True` and pcast invariants to varying.** Invariants that solves
  close over must be pcast to varying; the ECCO LSR preconditioner's scan-carry init was typed invariant, so no sea-ice
  derivative traced at P>1; a pallas_call needs `manual_axis_type`. fesom_jax needed `check_vma=False` for scans seeded
  with constant carries until `lax.pcast(..., to='varying')` fixed it. *Sources:* [E§7] PL:244, 472-473, 534-537,
  576-578. [E§11] PL:470-473, 534-537. [F§5] PL:2319-2329.
- **L-PAR-11 Fake CPU devices serve small forward gates, not sharded gradients.** They gave NaN or deadlocked for the
  ECCO gradient driver where 4 GH200s were correct; XLA:CPU `all_gather` check-fails with 32 or more in-process devices.
  *mitjax:* see [L-CONF-9]. *Sources:* [E§7] PL:541-544; AR:885. [E§11] PL:541-544. [F§8]
  mem:pj/jax-cpu-scaling-findings.md.
- **L-PAR-12 Fixed tile order makes a result independent of the process count, not of the tile layout — and the Fortran
  itself can depend on the tiling.** Tile-local LSR is layout-dependent, and 13 vs 96 tiles became bitwise only by
  emulating the 30x30 blocks. FESOM's Gauss–Seidel hole fill differed by 27.72 PSU between 8 and 16 ranks; a
  deterministic ring fill then blew up identically everywhere ("a gate that had checked only for partition-independence
  would have passed the broken version"). *mitjax:* a twin is bitwise only at the same tile size; before requiring P=1
  == P=N for an experiment, check whether its oracle is tile-count invariant (Task 3b invariance check). *Sources:*
  [E§7] PL:670-679; EI:67-75. [E§11] PL:670-673. [P§5] PP/discussion.tex:70-74; KD/CG_BLOWUPS_M13.md:1-10,86-130.
- **L-PAR-13 `jax.jit` around a `shard_map` whose body has `jax.checkpoint`, or the reverse raises "Eager evaluation of
  closed_call".** *Sources:* [F§5] PL:2618-2630.
- **L-PAR-14 Data-dependent CG trip counts are safe under `shard_map`, because the psum'd residual is identical on every
  device; normalise residual norms by the global point count; gate a distributed solver on a captured real right-hand
  side.** The iteration-count margin depends on the right-hand side's spectrum. *Sources:* [F§5] PL:2289-2296;
  PL:2286-2287; PL:2297-2304.
- **L-PAR-15 A free-running sharded-vs-dense multi-step comparison cannot be a tight gate where the paths are not
  bitwise: teacher-force each step.** Two branch-valued schemes in series (FCT, KPP) decorrelated `uv` from 5e-13 to
  2e-2 in one step; fesom_jax kept a 1e-7 budget per teacher-forced step. *Sources:* [F§5] PL:5526-5560.
- **L-PAR-16 Localise a sharded mismatch geometrically after one step.** Bin the offending points by hop distance to the
  nearest non-owned point: all at hop 0 and tracer-only means a limiter flip, while a broken exchange corrupts every
  field that crosses it. *mitjax:* bin by distance to the tile edge. *Sources:* [F§5] PL:5526-5539.
- **L-PAR-17 Keep C-grid stencils as gathers in Fortran loop order.** Serial bit-identity of the Kokkos port needed the
  C edge order, which needs ordered atomics, so threaded and GPU runs cannot be bitwise, and a gather rewrite would
  break the serial order. *mitjax:* any GPU `segment_sum` or scatter-add moves that kernel into the "climate-close"
  class. *Sources:* [P§6] KD/SCATTER_STRATEGY.md:12-50.

## L-PERF — Performance, GPU and memory

- **L-PERF-1 Exclude compile time from timings.** Subtract two warm runs, `(t_N − t_W)/(N−W)`: "JAX 10× slower than
  Kokkos" was compile time (the corrected step was 86.7–92.6 against 117 ms), and the ECCO "44x reverse/forward" ratio
  was compile time too; time warm calls on one jitted object. A jitted run compiles two executables, the cold-entry and
  the steady-state graph: a timing parser that started at chunk 2 amortised a compile into every number (about 2× too
  high). Start reductions at the first steady chunk and assert chunk_n ≈ chunk_{n+1}. *Sources:* [F§6] PL:2698-2710.
  [E§8] PL:262-264. [F§1] PL:5326-5341.
- **L-PERF-2 Keep the monitor on the device.** A per-step host monitor cost more than the GPU step. *Sources:* [E§8]
  PL:228.
- **L-PERF-3 A chunked driver reuses its executable.** A fresh `jit(shard_map(body))` closure per chunk recompiled about
  27 s every chunk, so a 96 ms/step model ran at 520–562 ms/step (125 ms/step with reuse); the benchmark harness never
  showed it, because only the production driver rebuilt the closure. *mitjax:* build the stepping executable once per
  configuration and assert the compile count in a test. *Sources:* [F§1] PL:4512-4530.
- **L-PERF-4 Compile time is a design constraint, and compile cliffs exist.** In fesom_jax TKE × multi-node hung the
  compile for 66 min while farc/1-node compiled in about 100 s; production graphs took about 80 min per executable; a
  zstar compile ran over 3 h at about 435k lanes/GPU. In the ECCO port large scanned jvp graphs compiled for over an
  hour on GPU, where a 128-core CPU node can be faster. *mitjax:* put every new physics combination through a
  `timeout`-wrapped compile canary at the target scale before a long chain. *Sources:* [F§1] PL:4449-4462; PL:5336-5341;
  JUP:521-522. [E§8] PL:718-719.
- **L-PERF-5 Do HLO and compile diagnostics on short runs with debug prints off.** The scan trip count is a constant, so
  a 5-step run gives the byte-same graph as a 150-step run. *Sources:* [F§1] PL:5377-5382.
- **L-PERF-6 Unroll literal sequential sums.** cg2d's Fortran-order tile sums were 82 % of the ECCO GPU forward;
  `sum_unroll=5` stays bitwise and cut the step from 0.312 to 0.141 s. *Sources:* [E§8] PL:265-267. [E§11] PL:265-267.
- **L-PERF-7 Launch-bound sequential solvers need a Pallas kernel.** The ECCO LSR went from 27.6 to 0.78 s/step on a
  GH200, bitwise. Measure the dependent-op floor first, then ablate. *Sources:* [E§8] PL:456-469.
- **L-PERF-8 Profile and instrument before designing or theorising.** Use `jax.profiler`; XLA:CPU splits fusions over
  threads by output size. The ECCO setup was dominated by the whole-array gather exchange (about 160 s), so the fix was
  a halo-only exchange; load npz files once. At a new scale use flushed progress (`PYTHONUNBUFFERED=1`, because a
  walltime SIGKILL loses buffered stdout) and split the time into host and device on the cheapest resource. *Sources:*
  [E§8] PL:558, 572-575; PL:258-259, 776-777. [E§11] PL:776-777. [F§6] PL:4338-4357.
- **L-PERF-9 Build global arrays on the host, per tile, and shard on load.** Building global arrays with `jnp` stages
  them on GPU 0 (`1.34 GiB jit__where`), and a global `device_put` staged 125.81 GiB on one GPU for NG5: build on the
  host with numpy and place with `make_array_from_callback`. Per-rank host setup, not the shard, capped CPU meshes,
  because every rank built the global State. *mitjax:* read MDS/netCDF per tile on the host and never form a global
  `jnp` field. *Sources:* [F§4] PL:2719-2735; PL:2758-2773. [F§5] PL:5423-5428.
- **L-PERF-10 Forcing memory decides the gradient window: build per-chunk forcing on demand and store each record
  once.** Holding all forcing cost fesom_jax 0.5 GB per simulated day; check what was left resident before blaming the
  adjoint for an OOM. The ECCO EXF window took 20 MB/step, storing each record once was 6.2x smaller, and a 365-day full
  run fits one GH200 with stride 19. *Sources:* [F§4] PL:5613-5619; PL:6819-6836. [E§6] PL:757-759; AR:345-347. [E§11]
  PL:757-759.
- **L-PERF-11 One heavy backward per process, or `jax.clear_caches()` between probes.** Three backward probes in one
  fesom_jax process each held a 26 GB graph. *Sources:* [F§4] PL:1457-1466. [E§11] (checkpoint stack: one heavy stage
  per process).
- **L-PERF-12 `memory_analysis()` under-predicts the runtime peak about 2.4×; two OOM-killed runs reporting the same
  peak show the ceiling, not the demand.** *Sources:* [F§4] PL:5416-5421; mem:pj/MEMORY.md (E1 entry).
- **L-PERF-13 vmap over step pullbacks is 75x slower than a static loop of them.** *Sources:* [E§6] PL:484.
- **L-PERF-14 On GH200, bind each process to its Grace socket, and measure cost as the second repeat in a one-GPU job.**
  Without the binding runs were 4-12x slower, and one compile took 24 min. *Sources:* [E§8] PL:529-533, 610-611. [E§11]
  PL:529-533.
- **L-PERF-15 Several processes per node: `XLA_PYTHON_CLIENT_PREALLOCATE=false` and a memory fraction of 0.8.**
  *Sources:* [E§8] PL:716-718, 794-795.
- **L-PERF-16 Small grids are the wrong regime for GPUs and do not scale.** LLC90 on 4 GPUs is slower than on 1
  (0.43-0.47 vs 0.34 s/step); CORE2 (127k nodes) is faster on 512 CPU ranks than on any GPU count, and more GPUs make it
  worse. *mitjax:* the M1 verification grids are smaller still, so P>1 there tests correctness, not speed. *Sources:*
  [E§7] VAL:72-79; PL:279-281. [F§6] mem:pj/gpu-scaling-mesh-size-tradeoff.md.
- **L-PERF-17 A speed-up goes in only if it is bit-identical, unless Nikolay approves a round-off change (as he did for
  GMRES).** *Sources:* [E§10] notes/HANDOFF-20260923.md:57; OPT:3-9.
- **L-PERF-18 Same-day, same-allocation A/B for any timing, checked against a physical floor.** An "~8 %" Kokkos win
  against a recorded baseline was 1.3 % against the same-day baseline; a 261 MB memset cannot cost 0.01 %, so the knob
  had not fired — an include-order `#ifdef` silently turned it off and every correctness gate passed. *mitjax:* both
  legs in one job with a bytes-over-bandwidth floor estimate; log every switch's resolved value at startup; unknown keys
  are errors. *Sources:* [P§8] KM/feedback-perf-same-day-baseline.md:12; KD/KPL:496-532.
- **L-PERF-19 Benchmark the real workload on a finite state.** A fesom_jax benchmark had 117.8M NaN by step 25; every
  benchmark now prints a `[bench-finite]` line. *Sources:* [F§6] PL:4963-4972.
- **L-PERF-20 Short protocols include start-up artefacts, and a faithful port inherits the reference's performance
  bugs.** Every 35-step Kokkos benchmark was contaminated because forcing coefficients were rebuilt 8× per step for
  30–60 steps (the run started before the first forcing record), so the protocol became 300 steps and is "validated per
  mesh"; the `getcoeffld` clamp never released, so the forcing was re-read every step for the first 1.5–12 h of each
  year; CG spin-up lasted until about step 30. *mitjax:* check exf record-boundary logic and the cg2d spin-up before
  timing anything, and measure after spin-up (150–300 steps). *Sources:* [P§7] KD/GPU_SPEED_M7.md:7-30; KD/KPL:1063.
  [F§6] PL:5214-5225; PL:5236-5241.
- **L-PERF-21 Host forcing cannot be hidden in the loop.** A sharded `device_put` backpressures the compute stream, and
  netCDF4/HDF5 is not thread-safe; the lever that worked was on-device interpolation (−12 %). *Sources:* [F§6]
  PL:4535-4555; mem:pj/MEMORY.md (Kokkos M7 entry).
- **L-PERF-22 CPU collectives: count them.** On gloo, psum is O(P) (0.041→0.812 ms at 2→32 ranks) while ppermute is
  flat, and MPI collectives in JAX cannot run a real model; one collective call is not one message (a dense all-to-all
  is P−1 messages per rank and was 52 % slower than Δ≈6 ppermute rounds). A Chebyshev polynomial preconditioner cut the
  CPU step 29 %, with the best degree backend-dependent (1 on GPU, ≥6 on CPU). *mitjax:* count the all-reduces in cg2d.
  *Sources:* [F§5] PL:5407-5414; PL:5446-5453; PL:5455-5460; PL:5466-5472.
- **L-PERF-23 A transport ranking measured on one machine does not transfer.** On JUPITER padded was 4× slower than
  coloured at 8–16 GPUs, and the bandwidth-bound transports were irreproducible (ragged spread 376 %). *Sources:* [F§5]
  JUP:101-120; JUP:173-200.
- **L-PERF-24 An XLA fusion decision can flip at one shard shape and cost 2.7×.** At Lmax=59637 a `multi_output_fusion`
  produced a 383 ms kernel, and `lax.optimization_barrier` gave 647→236 ms/step with bitwise outputs. Find such kernels
  with nsys top kernels by mean duration, then compare the HLO `kind=`. *mitjax:* on optimization barriers see
  [L-CONF-7]. *Sources:* [F§5] PL:5352-5375.

## L-TEST — Testing

- **L-TEST-1 Group tests by measured cost in a manifest that lists every test file, and keep tier 1 under 10 min and
  under 100 tests.** fesom_jax's "everything else" group grew to 920 tests and 3:28 of wall time without anyone deciding
  it (a 2-core CI subset ran 35 files in 320 s); the ECCO tier 1 reached exactly 100 tests and tier1x outgrew 40 min; an
  unlisted test file is a collection error. *mitjax:* `mitjax/tests/manifest.py` with per-area fragments, and a group
  per verification experiment. *Sources:* [F§7] SG:3-12. [E§9] notes/HANDOFF-20260924.md:40; PL:661. [E§11] PL:14-15;
  notes/HANDOFF-20260924.md:40; PL:661.
- **L-TEST-2 A guard has two halves, the assertion and something that runs it.** A fesom_jax manifest group ran nowhere
  for two days; a bash array copy of the group list silently dropped a group; 64 tests under `scripts/tests/` never ran
  because the runners globbed one directory. Break each assertion once on purpose and watch the runner go red. *mitjax:*
  `test_manifest.py` checks that every test directory is collected. *Sources:* [F§7] PL:6755-6817; PL:5904-5910.
- **L-TEST-3 Zero tests run is a failure, and xfail is strict.** Three sharded fesom_jax test files exited 0 having run
  zero tests: report skips loudly and fail when a check cannot run. `xfail(strict=False)` on an upstream bug XPASSes
  silently when JAX fixes it; use `strict=True`. *mitjax:* oracle-dependent tests fail without their data, see
  [L-CONF-3]. *Sources:* [F§7] UR:206; mem:ne/fail-dont-warn-when-a-check-cannot-run.md; CR:608-610.
- **L-TEST-4 An assertion needs something a dead path cannot satisfy.** "Finite", "nonzero" and "changed" are satisfied
  by a dead path, FD is not; 13 fesom_jax failures came from dead fixtures (tke=0, no seeded ice); "in bounds" does not
  show the path is switched on (global extrema agreed to 0.01 K whether or not the NN was active). Assert in the fixture
  that the subsystem is live (`count_nonzero(a_ice) > 1000`), and prefer a structural zero to a small difference.
  *Sources:* [F§7] PL:5743-5790; PL:6060-6077.
- **L-TEST-5 An equivalence test that shares a helper with the code under test cannot see a sign error.** An exact-0
  match hid a negated divergence that an analytic check (slope −0.9975) caught. *Sources:* [F§7] PL:6257-6271.
- **L-TEST-6 A behaviour-preserving refactor of array code must be bit-identical (225 values, max Δ = 0); write the
  re-run to a new path.** *Sources:* [F§7] PL:6024-6035.
- **L-TEST-7 A test that fails identically on every commit is broken until diagnosed; "pre-existing" is provenance, not
  diagnosis.** *Sources:* [F§7] PL:5517-5524.
- **L-TEST-8 XLA:CPU's JIT exhausts `vm.max_map_count` (65,530), not RAM: clear JAX caches at module boundaries.** The
  failure reads "Cannot allocate memory" at 17 GB of 300 GB; clearing caches per module kept the suite at 50 % of the
  ceiling, while `MALLOC_ARENA_MAX=4` went green at 97.5 % and was rejected — judge a mitigation by its margin on the
  measured resource. *mitjax:* `conftest.py` clears the caches per module. *Sources:* [F§7] PL:6906-6925;
  mem:ne/xla-cpu-jit-exhausts-vm-max-map-count.md.
- **L-TEST-9 An editable install plus subprocess tests silently tests another checkout: pin `PYTHONPATH`.** *mitjax:*
  worktrees; env `mitjax` never has mitjax installed. *Sources:* [F§7] PL:5510-5514.
- **L-TEST-10 The development machine cannot test its own data fetcher: run the documented command with the project
  environment scrubbed.** *Sources:* [F§7] PL:5503-5508.
- **L-TEST-11 Test the degenerate configuration first.** A fesom_jax function that frees what a callback returns had an
  aliasing contract (`a[0:len(a)]` IS `a`) and failed only in the one-chunk case. *Sources:* [F§4] PL:6819-6836.

## L-ENV — Environment and versions

- **L-ENV-1 Pin the environment with a constraints file, upgrade only through a canary that includes the gradient gates,
  and install with `pip --no-cache-dir`.** Floor pins drifted to JAX 0.11.0 on three systems (0.11.0 then ran 836 tests
  and a model year with no performance change). *mitjax:* `constraints.txt` (jax 0.10.1) and `docs/ENV.md`. *Sources:*
  [E§9] ENV:3-10, 35-36; PR:41; PL:10. [F§8] ENV:67-70; UR:210-213.
- **L-ENV-2 jax 0.10.2 deadlocked in-process CPU collectives at npes≥4; 0.10.1 works.** Re-run all sharded gates on any
  upgrade. *Sources:* [F§8] mem:pj/padded-a2a-halo-branch.md:42-44.
- **L-ENV-3 No persistent XLA compilation cache (Nikolay's decision, 2026-07-18): the staleness risk outweighs the
  wall-clock gain.** *Sources:* [F§1] mem:pj/no-persistent-xla-cache.md. [E§9] ENV:3-10, 35-36; PR:41; PL:10.
- **L-ENV-4 Login nodes run only seconds-scale smoke tests, under `taskset`, one CPU-JAX process at a time.** On fat
  login nodes XLA's compile thread pools overshoot `ulimit -u` and abort (`taskset -c 0-15` in fesom_jax; the ECCO smoke
  command uses `taskset -c 0-7`). No model runs on the login node (Nikolay: "we will get kicked out"), and never ssh to
  compute nodes. *Sources:* [F§8] ENV:94-100; PL:1418-1424. [E§9] ENV:3-10, 35-36; PR:41; PL:10. [P§9]
  KM/feedback-no-model-runs-on-login.md:3-11. [F§10] mem:pj/MEMORY.md; mem:ne/MEMORY.md:13.
- **L-ENV-5 `conda activate`, not `mamba activate`, which can exit 0 without activating; register a named Jupyter
  kernel.** *Sources:* [F§8] ENV:24; ENV:55-61.

## L-OPS — Operations

- **L-OPS-1 Never delete and never `scancel`, and never put a delete in unattended automation.** List candidates, with
  the exact command, for the owner's decision instead. In the ECCO project even an `rm -rf` on a non-existent
  path was flagged as a breach; in fesom_jax an `rm -rf $OUT` in a self-resubmitting chain fired on an inferred "cold
  start" and destroyed 27.3 model years. *Sources:* [E§9] PL:313-314; notes/DELETION_CANDIDATES.md:1-4;
  notes/HANDOFF-20260923.md:325; GCL. [F§9] mem:pj/feedback-no-silent-deletion-in-automation.md.
- **L-OPS-2 Risk concentrates in unattended shell, not in the numerics.** A review of about 10k lines of fesom_jax
  numerics found 3 majors, while the job chains had a TAG clobber, a silent cold-start fallback, non-atomic restarts and
  validated YAML keys that were never read. *Sources:* [F§9] mem:pj/code-review-20260703.md; CR:70-72.
- **L-OPS-3 A job reads its files when it reaches them, not when it was submitted: freeze the code per job.** An ECCO
  job that started after an edit ran the edited code; a jaxpr-hash reference job picked up edits made while it queued;
  the first Fortran builds came from an uncommitted tree. `cp` truncates the inode of a bash script parked in `srun`
  (use `mv`); `sbatch` snapshots the submitted script only; `GROUPS` is a bash builtin. *mitjax:* a frozen worktree or
  `git archive` per job, the commit hash in binary names; wrap a driver that another branch is editing instead of
  editing it. *Sources:* [E§1] PL:743-744; W4:182; W1:218; REF:28-30; PL:612-614. [F§9] PL:5862-5902.
- **L-OPS-4 Slurm and shell traps.** `sbatch --export` splits at commas (five duplicate builds): use `:` or `+`, or pass
  arguments after the script name; the spool copy hides `$0` (use `Command=` from `scontrol show job`); `srun` inherits
  the launcher's cpus-per-task; NFS can be stale after an edit (checksum inside `srun`, a fresh `PYTHONPYCACHEPREFIX`
  per run). `${VAR:-default}` substitutes on empty; use `${VAR-default}` where empty is meaningful. *Sources:* [E§9]
  PL:69, 477, 684-685; W1:226-227; PL:494-496; W1:219-220; PL:148-149, 249. [F§9] PL:6195-6206.
- **L-OPS-5 Judge a job by its report, not by its state.** A Slurm `COMPLETED 0:0` is not a test result (the log said `2
  failed, 209 passed … rc=1`), and tools that exit 1 on differences leave FAILED jobs with valid reports. Verdicts come
  from the JUnit report plus the exit code; a watcher must report FAILED/TIMEOUT/OOM states. *mitjax:*
  `scripts/check_pytest_report.py`. *Sources:* [E§9] PL:16-18; W1:232-233; QS:126. [E§11] PL:16-18. [F§7] PL:6882-6886;
  mem:ne/green-verdict-is-not-evidence.md.
- **L-OPS-6 Check `sacct` before quoting a job ID.** One fesom_jax job ID was invented. *Sources:* [E§9] CAT:117. [F§9]
  PL:5939-5944.
- **L-OPS-7 Never pipe a failing launcher or pytest through `tail`.** The exit code is `tail`'s and the error line is
  cut; `--durations` sorts descending, so `| tail` keeps the cheapest tests. *Sources:* [F§9] PL:5430-5434;
  PL:1421-1424. [F§7] PL:6778-6787.
- **L-OPS-8 Every job and every process gets its own run and output directory, and temp files get unique names.** The
  ECCO `make_rundir` takes the first match in sorted order, so a second data directory is shadowed; processes sharing an
  OUT directory race on creating it; interleaved logs cost hours; a fixed temp name raced once sibling jobs became
  routine. *Sources:* [E§9] PL:623-624, 655-656; PR:39-40. [P§8] P2M/feedback_unique_outdir_per_job.md:14,30. [F§9]
  PL:6321-6327.
- **L-OPS-9 Machine paths live in one stdlib-only module, and a test refuses hard-coded roots.** In the ECCO port
  `test_paths` caught a hard-coded root. *mitjax:* `mitjax/paths.py` with `MJX_*` variables (never `MITJAX_*`, the ECCO
  port's prefix). *Sources:* [E§9] PL:491-493; W5:313-317.
- **L-OPS-10 Build pitfalls: build the oracle from a conda-free script and record the optfile and compiler.** Without
  NetCDF, genmake2 silently drops `pkg/profiles`; gfortran ≥ 10 needs `-fallow-argument-mismatch` for c66g; cpp replaced
  a subroutine that had a switch macro's name; `--box=-96` needs the `=`. In FESOM, mambaforge on PATH shadowed
  compilers and netcdf, and stale objects after a checkout looked like an "intermittent bug". *mitjax:* Task 3a.
  *Sources:* [E§9] REF:28-33; W1:230-231; PL:601. [P§8] P2M/feedback_levante_build.md:7-12.
- **L-OPS-11 Downloads run as jobs on the shared partition, and a stream that ends is not a finished download.** Trust
  Content-Range, resume, and check sha512; the mambaforge CA bundle is broken. *Sources:* [E§9] PL:42-48, 57-60;
  mem/feedback_jobs_not_login.md:7-16.
- **L-OPS-12 Account limits are shared across sessions, and another session's jobs are not ours.** At most 5 running GPU
  jobs; AssocMaxJobsLimit counts all sessions, so pack processes into one `--multi` job; never submit or cancel another
  session's jobs. *Sources:* [E§9] PL:325, 602-603, 716-717; AR:272-275. [F§10] mem:pj/MEMORY.md; mem:ne/MEMORY.md:13.
- **L-OPS-13 Pin GPU jobs to `--constraint=a100_80`, and use at most 16 GPU nodes.** Levante's `gpu` partition mixes
  A100-40/80, and one slow node paces a job (+3.4 %). *Sources:* [F§6] PL:5237-5240. [P§9]
  KM/feedback-16-node-cap.md:13-17.
- **L-OPS-14 A job asserts the deltaT and protocol it intends; fatal messages print the model step; read the first
  FATAL, not the last line.** FESOM runs at the production dt instead of the cold-start ladder dt were misread as
  "unusable partitions" four times; "abort at iter 1" meant CG iteration 1 of step 71; `srun … Killed` lines hid the
  real FATAL hundreds of lines earlier. *Sources:* [P§7] KDI/KPL:1569-1600.
- **L-OPS-15 Ask before every push; run a marker scan and push only after a yes on the exact text.** The scan covers
  session links and assistant-attribution or agent labels; the agent instructions file stays untracked; MITgcm itself is
  not in the private repository, only the overlays. *Sources:* [E§9] mem/project_public_repo.md:12-23;
  mem/project_private_repo.md:16-22. [P§9] KM/feedback-ask-before-each-push.md:11.
- **L-OPS-16 Stage files explicitly, commit nothing binary, never write into shared input trees.** `git add <dir>` once
  put private plans into the public repo; ~0.9 GB of CUDA binaries were once committed; mesh copies never go to /pool.
  *mitjax:* data, builds and dumps live under `$MJX_WORK`, never in the repository. *Sources:* [F§10]
  mem:pj/never-blanket-git-add-a-directory.md. [P§8] KM/feedback-mesh-copies-never-pool.md:11;
  KM/feedback-never-commit-binaries.md:3-20.

## L-USER — Nikolay's preferences

- **L-USER-1 Decisions are Nikolay's, recorded with a date; never pre-decide for him.** An ECCO recipe README called
  option 2 "recommended" and he chose option 1; options stay options ("both as options only"); keep both research
  branches alive until data arbitrates. *Sources:* [E§1] W4:185. [E§10] AM:266, 273, 282-284; ADXS:197-205. [F§10]
  mem:ne/keep-both-options-alive.md.
- **L-USER-2 Always measure, do not guess; claims are shown by measurement.** Four reasoned answers in one Kokkos
  session were all wrong; measure his domain questions rather than answering from memory. The goal is an adjoint
  technically better than TAF, proven by measurement. *Sources:* [P§0] item 10. [P§9]
  KM/feedback-always-measure.md:10-22. [F§10] mem:ne/keep-both-options-alive.md. [E§10]
  mem/project_beat_taf_adjoint.md:7-17.
- **L-USER-3 "Whatever ECCO does": per-field adjoint monitors and point FD (grdchk), not invented norms.** In the ECCO
  port, drivers default to ECCO semantics with exact selectable (on the default in mitjax see [L-CONF-5]). *mitjax:*
  "whatever MITgcm does" — the `%MON ad_dynstat_ad*` monitor and grdchk (Task 16). *Sources:* [E§10] PL:434-436;
  AM:220-226.
- **L-USER-4 Naming and figures.** Say "Fortran 13 ranks", not "13 tiles"; scaling runs are short (a few hundred steps,
  s/step × 8760); difference maps are signed, linear and symmetric, with time series plus signed differences; reuse the
  house figure conventions (one figure was corrected twice); name non-default artefacts by variant. *Sources:* [E§10]
  mem/feedback_naming_and_scope.md:10-16. [P§9] KM/feedback-figure-conventions.md:13; KM/feedback-kpp-default.md:19.
- **L-USER-5 Stay in scope.** A docs task means docs, not long verification runs; packaging gets no unrequested tests
  (on tests in mitjax see [L-CONF-13]); no unrequested CPU speed campaigns. *Sources:* [E§10]
  mem/feedback_naming_and_scope.md:15-16; mem/project_private_repo.md:22. [P§9] KM/feedback-no-core2-cpu-scaling.md:17.
- **L-USER-6 Plots with nereus, never earthkit; reports with slidestyle.py figures and LaTeX (texlive).** When he is
  away, finish without questions and put the PDF path in the handoff. Paper numbers are generated macros; prose states
  the governing relationship, and the per-case numbers go in tables. *Sources:* [E§10]
  mem/feedback_plotting_nereus.md:10-17; mem/feedback_report_style.md:10-20. [F§10]
  mem:pp/paper-numbers-are-auto-generated-macros.md; mem:pp/prose-prefers-general-claims-over-lists.md.
- **L-USER-7 Writing and reporting.** Plain cause-and-effect sentences with no project shorthand but no
  oversimplification; PR, commit and comment text states facts and numbers with no padding and tight scope; report
  wall-clock parity unasked; ask a clarifying question when a framing is off. *Sources:* [F§10]
  mem:ne/feedback-talk-plainly.md; mem:ne/feedback-terse-written-deliverables.md. [P§9]
  KM/feedback-hpc-run-hygiene.md:15-17.
- **L-USER-8 He reads maps and catches what gates miss: show maps early and treat his pushback as data.** He found the
  `ice_strength` factor from an `m_ice` map, a fake 15 % ice gap ("the ice time series had always been
  indistinguishable") and a stale plan. *Sources:* [P§9] P2M/feedback_ice_strength_halffactor.md:17;
  P2M/feedback_nan_mask_temporal_average.md:19-23; KD/KPL:422.
- **L-USER-9 Test on small meshes before expensive ones.** *Sources:* [F§10] mem:pj/MEMORY.md; mem:ne/MEMORY.md:13.

## L-COPY — Infrastructure to copy from the ECCO port (the copy audit list)

Each module entry names the ECCO module(s) in R, the mitjax destination and plan task, what went wrong or needs care
("Found", with the ECCO citations inline), and the Fortran the module encodes, which the c66g→master audit must diff.
Fortran line numbers are at checkpoint66g.

- **L-COPY-1 Read the diff, not the file list: copy nothing before the c66g→master audit of the Fortran it encodes.** In
  the ECCO override audit five "physics" overrides turned out to be diagnostics only, and the catalog's "3 forward
  changes" were really 1. Classify every hunk as forward value, AD-only, diagnostics or I/O, as a generated table
  compared byte for byte with negative controls (`audit_overrides.py --check`). *mitjax:* `scripts/audit_upstream.py`
  and `docs/AUDIT_C66G_MASTER.md` (Task 7a) cover every module of [L-COPY-2] … [L-COPY-16]. *Sources:* [E§1] PL:22-26;
  PL:36-38.
- **L-COPY-2 exch2 maps and the single-device Exchanger.** R: `parallel/exchange.py`, `scripts/make_exch_maps.py`,
  `data/exch_maps_<layout>.npz`, `JAXDUMP_EXCH_PROBE` in `reference/jaxdump/jaxdump.F` →
  `mitjax/eesupp/{exch_maps,exchange}.py`, `scripts/make_exch_maps.py` (Task 7b; probe in Task 4). Found: the first
  version reused u as the v output's "own value" (PL:86-88); a zero-halo probe missed halo-sourced copies (PL:109-112);
  the tile code overflowed above 99 tiles (W1:225; REF:190); the code format `1e6*comp+1e4*tile+100*(j+OLy)+(i+OLx)`
  (exchange.py:4) also needs sNx+2·OLx < 100; EXCH_S3D was probed late (PL:84); -0 vs +0 (PL:118-119); there is one map
  file per layout; corner fills (FILL_CS_CORNER_*) live in the kernels, not in the exchanger (KG:39). Fortran:
  EXCH_XY_RL, EXCH_UV_XY_RL (with and without signs), EXCH_Z_3D_RL, EXCH_UV_AGRID_3D_RL, EXCH_UV_BGRID_3D_RL,
  EXCH_3D_RL, EXCH_SM_3D_RL, EXCH_UV_DGRID_3D_RL, EXCH_UV_3D_RL, EXCH_S3D_RL (jaxdump.F:1053-1109); exch2_3d_rx.template
  with its two corner passes (PL:79-81); data.exch2 and W2_MAP_PROCS (PL:425); MDS_FACEF_READ (PL:109). Only exch2 was
  probed: experiments without pkg/exch2 use the eesupp exchange, so the probe is extended to it. *Sources:* [E§11],
  citations inline.
- **L-COPY-3 Sharded exchange and driver.** R: `parallel/sharded_exchange.py`, `shard.py`, `tiles.py` →
  `mitjax/eesupp/{sharded_exchange,shard,tiles}.py` (Task 7b). Found: replica padding needs zero cotangents on the
  padding and zeroed padding in transpose solves (PL:244-245, 579-580); typing traps — pcast of invariants, the
  invariant scan-carry init, Pallas `manual_axis_type` (PL:470-473, 534-537); fake CPU devices cannot run the gradient
  driver (PL:541-544); replicated params must not carry a tile axis (shard.py:22-23); the chunked driver was not wired
  to ShardedModel at M1 (AR:299-302). Fortran: none beyond the maps and GLOBAL_SUM_TILE. *mitjax:* Task 17 wires the
  gradient driver to the sharded model. *Sources:* [E§11], citations inline.
- **L-COPY-4 Fixed-order global sums.** R: `parallel/global_sum.py`; `Exchanger.global_max` →
  `mitjax/eesupp/global_sum.py` (Task 7b). Found: per-tile partials must reproduce the callers' `DO j; DO i` loops
  (cg2d.F:181-183); psum turns -0 into +0, harmless because the sum starts from 0 (global_sum.py:30-31); the result
  depends on the tile size, though not on the process count (PL:670-673); pmax has no JVP (PL:243); padded cost sums
  differ by 1-2 ulp (PL:539-540). Fortran: eesupp/src/global_sum_tile.F:164-194 with GLOBAL_SUM_ORDER_TILES
  (CPP_EEOPTIONS.h:132; global_sum.py:3-14); _GLOBAL_MAX_RL (cg2d.F:122, 130). *Sources:* [E§11], citations inline.
- **L-COPY-5 cg2d.** R: `core/cg2d.py` → `mitjax/ad/cg2d_rule.py` (derivative rule, Task 7c) and
  `mitjax/model/src/cg2d.py` (forward, written in M1 Task 12 in the style Task 8 chooses). Found: a constant cg2dNorm
  let XLA regroup terms, giving a 1e-9 drift (PL:184-185); the stop margin is 0.8 % (PL:186); sum_order "fortran" vs
  "tile" (PL:187-188); sum_unroll=5 on GPU (PL:265-267); the custom_jvp rule replaced the custom_linear_solve JVP
  (PL:268-271); the transpose runs to 1e-13 from zero vs TAF's 1e-7 (AM:200-202); stop_coeff_grad corresponds to
  cg2d.flow's passive operator (AM:190-198); the residual `cg2d_x` is named for remat; INI_CG2D halos are never
  rewritten (PL:205); CALC_R_STAR's STOP cannot fire inside jit, so return counters (PL:189). Fortran: cg2d.F with
  DISCONNECTED_TILES, ALLOW_CG2D_NSA and ALLOW_SRCG undefined (cg2d.py:3-5); ini_cg2d.F:65-78, 90-135; update_cg2d.F;
  solve_for_pressure.F:273-311; exch2_s3d_rl.F; CG2D.h; set_defaults.F:280-286; ini_parms.F:1451-1455;
  pkg/autodiff/cg2d.flow:7-12. Check which cg2d variants master offers and which ones the verification experiments
  select. *Sources:* [E§11], citations inline.
- **L-COPY-6 Checkpoint stack and gradient drivers.** R: `adjoint/checkpoint.py`, `adjoint/grad.py` →
  `mitjax/drivers/{checkpoint,grad}.py` (Task 7c). Found: closed-over model data was constant-folded, giving 275 s
  compiles and a CUBIN too large (PL:262-264, 323-325); jit a function that calls vjp (PL:147); the model, Exchanger
  included, is a pytree jit argument, for the TL too (AR:268-271); one heavy stage per process; the AB-start chunk loses
  cotangent, and pass-through fields fake growth (AR:251-262); a linearity guard on the first reverse chunk
  (PL:803-808); repeat floors 4e-11 / 9e-11; `make_step` without a config is an error (AM:220-224); compact EXF window
  (PL:757-759); GH200 NUMA binding (PL:529-533). Fortran: the forward_step.F call order; adams_bashforth3.F:87-92;
  exf_set_fld.F records (checkpoint.py:22-25); for TAF comparisons, the_main_loop.F tapes and tamc.h nchklev_1
  (AM:50-54). *Sources:* [E§11], citations inline.
- **L-COPY-7 jaxdump shim.** R: `reference/jaxdump/{jaxdump.F, JAXDUMP.h, instrument.py, SUBSTEPS.md}` → the same paths
  in mitjax (Task 4). Found: the invisibility test (PL:64-65); myIter-1 after DYNAMICS (PL:66-68); the 72-column limit,
  `#ifdef` inside a CALL, pass suffixes, and the last-record rule for repeated keys (PL:309-311); `:` as the separator
  in JAXDUMP_STEPS (PL:69); big-endian output via `-fconvert=big-endian` (jaxdump.F header); the kinds U: (interior
  locals) and N: (scalars) (PL:305-306); stages that sit after partial updates (PL:363). Fortran: instrumented copies of
  forward_step, do_oceanic_phys, dynamics, solve_for_pressure, thermodynamics, temp_/salt_integrate,
  exf_{getforcing,bulkformulae,radiation}, seaice_{model,dynsolver,lsr,advdiff,growth}.F (instrument.py STAGES;
  C66G_DIRS at instrument.py:26); SIZE.h, EEPARAMS.h, PARAMS.h. Re-validate every anchor and its count on master; stage
  lists become per package. *Sources:* [E§11], citations inline.
- **L-COPY-8 Dump reader and diff tools.** R: `io/dump.py`, `io/llc.py`, `tools/diffdump.py`, `tools/step_vs_dump.py`,
  `reference/repro/compare_layouts.py` → `mitjax/io/dump.py`, `tools/{diffdump,step_vs_dump}.py` (Task 4). Found: read
  lazily and in parallel (PL:89, 312, 393); masks come from the first dumped hFacC/W/S (diffdump.py:5-7), and a missing
  hFac made the W1 tool run unmasked (W1:221-224), so require masks per level and for Z points; the NaN/max() trap
  (PL:647-652); the ZERO flag catches "allocated, never computed" (diffdump.py:11); load npz files once (PL:776-777).
  Fortran: the record format is tied to jaxdump.F (dump.py:3-8); io/llc.py FACET_SHAPE is LLC-only, so generalise it.
  *Sources:* [E§11], citations inline.
- **L-COPY-9 Test manifest, runner and verdict.** R: `tests/manifest.py`, `conftest.py`, `scripts/run_tier1.sbatch`,
  `scripts/check_pytest_report.py` → copied in Task 1 (the manifest split into per-area fragments). Found: an unlisted
  test file is a collection error (PL:14-15); the verdict comes from JUnit plus the exit code, with negative controls
  (PL:16-18); set the device count before backend init (PL:11-13); gate flags are set in conftest, and standalone
  scripts must set them too (KG:74-78); the tier budgets overflowed (notes/HANDOFF-20260924.md:40; PL:661);
  oracle-dependent tests skip without data (QS:112), which mitjax reverses ([L-CONF-3]). Fortran: none. Add a group per
  verification experiment so that tier 1 stays under 100 tests. *Sources:* [E§11], citations inline.
- **L-COPY-10 Safe ops and AD helpers.** R: `ops/ad_skip.py`, `ops/libm.py`, `adjoint/modes.py` (`differentiate_at`),
  `adjoint/diag_cuts.py` → `mitjax/ops/{safe,libm}.py` (Task 7c); `ad/ad_skip.py` and `ad/modes.py` are deferred to the
  first milestone with a `data.autodiff` approximation. Found: there is no `ops/safe.py` in R, although the ECCO
  project's instructions file names one; guards are written inline (KG:15-17), and the other source is fesom_jax
  `ops.py` (PLAN:28); division-JVP underflow (PL:337-339); the data-dependent zero (PL:697-699); `sg(v)+m·(v−sg(v))`
  turns -0 into +0 (PL:708); libm is transcribed for glibc 2.28 and libmvec, with XLA's exp and arccos not matching
  (PL:357-362), so re-verify on the master oracle's glibc and gfortran; ad_skip is the identity, not a stop_gradient
  (ad_skip.py:3-21); cut contexts must enclose lazily traced rules (PL:799-802). Fortran: autodiff_readparms.F:66-125;
  autodiff_inadmode_set_ad.F:33-53; autodiff_inadmode.flow; cg2d.flow; CADJ STORE sites at temp_integrate.F:488/505,
  salt_integrate.F:480/497 and dynamics.F:399-402 (AM:253-256); ZERO_ADJ_LOC under GMREDI_WITH_STABLE_ADJOINT;
  mom_calc_visc.F viscFacAdj, where V4r4 scales four fields and c66g two (AM:167-171). *Sources:* [E§11], citations
  inline.
- **L-COPY-11 Params pytree and namelist reader.** R: `params_io.py`, `io/namelist.py` → `mitjax/params_io.py`,
  `mitjax/io/namelist.py` (Task 6). Found: only fields annotated `float` are traced; int, bool and str fields are
  static, so they recompile per value and are constants to XLA (params_io.py:77-82); never branch on a traced field in a
  rule (PL:735-737); three namelist terminators (PL:53-54); KeyError without a cited default (params_io.py:30-36);
  diagnostics_is_on refuses snapshot-only lists (params_io.py:40-48); weak types (PL:445-446). Fortran: set_defaults.F,
  ini_parms.F, each package's *_readparms.F, packages_boot.F (implied useCAL, PL:653-654), diagnostics_is_on.F:47-72.
  *Sources:* [E§11], citations inline.
- **L-COPY-12 Layout and Fortran-index helpers.** R: `layout.py`, `grid/geometry.py` → `mitjax/farray.py` (Task 8;
  unused if plain slices win) and `mitjax/model/grid.py` (Task 10). Found: arrays are `[tile,(k),j,i]`, with Fortran i
  at index i-1+OLx (layout.py:1-10); unwritten points keep their prior value (KG:29-30); the tile axis was misplaced in
  3-D snapshots (PL:301-302); hFacC is not static under z* (KG:33-35). Fortran: SIZE.h (sNx, sNy, OLx, OLy, Nr); the W2
  tile numbering of data.exch2. The planned Fortran-index wrapper replaces `L.is_()`/`L.js()`; keep these semantics
  exactly, and make OLx a per-experiment setting. *Sources:* [E§11], citations inline.
- **L-COPY-13 Oracle build scripts.** R: `reference/build.sh`, `reference/optfile_levante_gfortran` → mitjax
  `reference/` (Task 3a). The ECCO oracle is gfortran -O3 with no fast-math and no FMA (`-ffp-contract=off`), invisible
  to the dump shim (PL:64-65; REF:66-67); genmake2 silently drops pkg/profiles without NetCDF, gfortran ≥ 10 needs
  `-fallow-argument-mismatch` for c66g, and cpp replaced a subroutine named like a switch macro (REF:28-33; W1:230-231);
  the first builds came from an uncommitted tree (W1:218; REF:28-30). Fortran: genmake2 and the build options file,
  audited against master's in Task 3a. *Sources:* [E§2] PL:64-65; REF:66-67. [E§9] REF:28-33; W1:230-231. [E§1] W1:218;
  REF:28-30.
- **L-COPY-14 One-routine replay harness.** R: `reference/adx_harness/` → `reference/replay/` (Task 8). A genmake2 build
  whose THE_MAIN_LOOP reads every input of the routine from a record file; it runs in seconds ([L-ORA-17]). *Sources:*
  [E§2] PL:689-693.
- **L-COPY-15 Branch coverage tool.** R: `tools/branch_coverage.py` → `tools/coverage.py` (Task 5). gcov with
  GCOV_PREFIX per run, and `cpp -traditional` line markers to map back to `.F` lines ([L-ORA-12]). *Sources:* [E§2]
  PL:350; PL:387-390; PL:70.
- **L-COPY-16 Machine paths module.** R: `paths.py` → `mitjax/paths.py` (Task 1, at 265b897), with the `MJX_*` prefix
  instead of `MITJAX_*` ([L-OPS-9]). *Sources:* [E§9] PL:491-493; W5:313-317.

## L-CONF — Conflicts between the digests

- **L-CONF-1 FMA — resolved: ECCO's `--xla_cpu_max_isa=AVX` [E§5] supersedes fesom_jax's "nothing reliably stops it"
  [F§2].** fesom_jax found that XLA FMA-contracts `a*x+y` inside jit (the EOS density moved 1e-13 under jit, beyond a
  1e-14 gate; at catastrophic-cancellation magnitudes the shift reached 1e-9 relative on 100 % of elements) and that
  `--xla_allow_excess_precision=false`, `optimization_barrier` and bitcast round-trips all failed, so its tight gates
  ran eagerly. The ECCO port found that `--xla_cpu_max_isa=AVX` stops the contraction on x86, and kept every kernel
  bitwise under jit. mitjax evidence: `mitjax/tests/test_env.py::test_gate_flags_bite` at master 265b897 on a Levante
  login node (AMD EPYC 7763, jax 0.10.1): a jitted `a*b+c` with `a = 1+2^-30`, `b = 1−2^-30`, `c = −1` (so a·b = 1−2^-60
  exactly), 1024 values, gives −2^-60 (fused) for all 1024 values without the flag and 0 (two roundings) for all of them
  with it (re-run on levante1 by this lane: 1024 fused without, 0 with). Resolution: bitwise CPU gates run under jit
  with the gate flags ([L-XLA-1], [L-XLA-5]), not eagerly; the flag is x86-only, and the GPU has no such switch (Triton
  `.rn` ops are the exception). The tier-1 run on a compute node will confirm the result on that CPU. *Sources:* [E§5]
  PL:96; KG:76-78. [F§2] PL:473-478; PL:5276-5288. Plan lines 106-108 (Development Approach, bitwise CPU gates) and 184
  (Task 1).
- **L-CONF-2 XLA:CPU float64 divide [F§2] vs the ECCO bitwise kernels [E§5] — measured in Task 8 (2026-10-01): IEEE-exact
  for normal numbers under the gate flags; subnormals are flushed to zero.** Result (lane C, `dev/prototype/divide_probe.py`,
  replay job 27826392, gates job 27826890): of 65,536 pairs (gfortran equal to numpy on all), XLA:CPU equals gfortran on
  all 49,672 pairs with normal operands and quotient, for an array divisor, a traced scalar and a Python constant; the
  optimized HLO has one divide and no multiply. All 15,864 differences involve a subnormal, and XLA returns ±0 in every
  one (flush-to-zero; gfortran keeps subnormals). With algsimp on, a scalar divisor becomes `x*(1/d)` (13 % of normal
  values differ at d = 1000, up to 36 %), a full-array divisor is not rewritten — the fesom_jax "13 %" was that rewrite.
  Original text:
  fesom_jax: XLA:CPU's float64 divide is 1 ULP off numpy on about 13 % of operands, XLA folds `x/1000` into `x*0.001`,
  and fast-math flags did not help. ECCO: with algsimp disabled, which stops `x/d → x*(1/d)`, every kernel was bitwise
  against gfortran. mitjax evidence so far (`test_env.py::test_gate_flags_bite`, same node and commit): for x = 1.1·(1,
  …, 1024) and a traced scalar d = 3, the jitted x/d equals numpy's x/d for all 1024 values with algsimp disabled, and
  differs for 339 values without it; this lane checked that without algsimp the jitted result equals x·(1/d) at every
  point, so all 339 differences are the reciprocal rewrite. Hypothesis, not measured: the fesom_jax 13 % may have been
  that rewrite rather than an inexact divide instruction; one divisor does not test the divide. Task 8 must measure,
  under the gate flags on a compute node (CPU model recorded): (1) elementwise a/b with both operands traced arrays,
  over many divisors and magnitudes, against numpy (IEEE, correctly rounded) and against gfortran's division on the
  replay harness, counting ULP differences; (2) a Fortran division by a literal (e.g. `x/1000.`) against JAX with the
  divisor as a traced runtime operand and as a Python constant; (3) the optimized HLO, grepped for divide versus
  multiply-by-reciprocal ([L-XLA-2]). If division is not bitwise under the gate flags, the per-kernel tolerance class or
  a workaround goes to Nikolay. *Sources:* [F§2] PL:5260-5268. [E§5] PL:97-100, 157-158. Plan line 303 (Task 8: "measure
  XLA:CPU float64 divide vs gfortran on the harness").
- **L-CONF-3 Oracle-dependent tests without their data: skip [E§11] vs fail [F§7] — resolved by the plan: fail.** The
  ECCO runner skipped oracle-dependent tests without data (QS:112); fesom_jax found three sharded test files that exited
  0 having run zero tests and adopted "fail, don't warn, when a check cannot run". The plan: oracle-dependent tests
  fail, not skip, when their data is missing, unless explicitly marked. *Sources:* [E§11] QS:112. [F§7] UR:206;
  mem:ne/fail-dont-warn-when-a-check-cannot-run.md. Plan lines 159-160 (Testing Strategy).
- **L-CONF-4 Bitwise equality of two different compiled programs: [F§2] "never bitwise comparable" vs [E§6]/[E§7]
  bitwise P=1 vs P=N, byte-identical forward across modes, identical gradients across checkpoint schedules — resolved by
  the plan for CPU.** fesom_jax: two different compiled programs are never bitwise comparable, compare against the
  measured spread of remat on/off; an inert flag moved four GPU results by 1 ulp. ECCO: P=N bitwise equal to P=1; every
  derivative-only switch keeps the forward byte-identical; the step, sqrt and chunked schedules give identical
  gradients; fesom_jax itself found `jax.checkpoint` on/off forward-transparent on CPU (Δ=0.0, [L-ARCH-2]). Resolution:
  on CPU under the gate flags the plan requires bitwise equality where it says so — P=1 vs P=N forward (plan line 11;
  Task 18, line 422), identical gradients across checkpoint schedules (Task 7c, line 293), backward-only switches with a
  byte-identical forward (line 7-8 with M3 line 451, "three tests per switch"). On GPU the measured floor applies (Task
  17, line 418; [L-TOL-12], [L-AD-33]), as does fesom_jax's rule for any comparison the plan does not require to be
  bitwise. *Sources:* [F§2] mem:pj/adjoint-lessons-program.md:32-33; PL:2632-2643; PL:5771-5777. [E§6]
  checkpoint.py:28-31; HC:10-13. [E§7] PL:237-248, 534-537.
- **L-CONF-5 Default gradient semantics: ECCO semantics [E§10, AM:220-226] vs exact where stable [E§10, AM:266, 273,
  282-284] — resolved by the plan: exact.** The ECCO digest records both "drivers default to ECCO semantics, with exact
  selectable" and "the exact adjoint stays the default where it is stable". The plan: the exact JAX gradient is the
  default; each `data.autodiff` approximation an experiment uses gets a TAF-compatible backward-only switch. *Sources:*
  [E§10] AM:220-226; AM:266, 273, 282-284. Plan lines 7-8 (Overview).
- **L-CONF-6 What to freeze in a TKE closure (GGL90): TAF's frozen outputs [E§6] vs fesom_jax's frozen inputs [F§3] —
  resolved for TAF-compatible switches; any other freeze is open — for Nikolay.** ECCO: TAF's "GGL90 off in AD" is
  `stop_gradient` on its outputs, set by where the STOREs sit. fesom_jax: freeze a closure's ocean-state inputs, not its
  prognostic state, because freezing tke severed wind→mixing. The plan's switches reproduce TAF's semantics (plan lines
  7-8; M3 lines 450-451), so a TAF-compatible GGL90 switch follows the STORE placement and is checked against
  `output_adm*.txt`; fesom_jax's input freeze is a different, non-TAF design that the plan does not include, and
  proposing one is Nikolay's decision. *Sources:* [E§6] PL:210-213; AM:59-63. [F§3] OC:781-792.
- **L-CONF-7 Optimization barriers: "none in kernels" [E§5] vs a measured, bitwise performance fix [F§5] — decided by
  Nikolay 2026-10-01: no barriers in physics code.** The ECCO kernel guide says no optimization barriers in kernels;
  fesom_jax's `lax.optimization_barrier` turned a 2.7× fusion cliff into 647→236 ms/step with bitwise outputs, and
  separately failed to stop FMA. Decision: `optimization_barrier` is banned in `mitjax/model/` and `mitjax/pkg/` like the
  other transforms (`test_banned_transforms.py`); a measured barrier may sit in `drivers/`, `ad/`, `eesupp/` or `ops/`,
  where the JAX-optimal route is taken. *Sources:* [E§5] KG:47-48. [F§5] PL:5352-5375. [F§2] PL:473-478. Nikolay
  2026-10-01.
- **L-CONF-8 Solver derivatives: "wrap the CG in `custom_linear_solve`" [F§3, PL:1163-1177] vs an explicit custom_jvp
  implicit-derivative rule [E§6; F§3, PIDX:143-145] — resolved by the plan: the explicit rule.** Both projects later
  found that `custom_linear_solve`'s JVP runs the loose primal solve ([L-AD-9]). The plan: `ad/cg2d_rule.py`, an
  implicit-derivative custom_jvp rule with tight tangent and transpose solves. *Sources:* [F§3] PL:1163-1177;
  PIDX:143-145. [E§6] PL:268-271. Plan line 288 (Task 7c).
- **L-CONF-9 Fake CPU devices: no stand-in for sharded gradients [E§7] vs sharded NaN probes and small fake-device gates
  [F§3, F§8] — resolved by the plan.** The plan: forward gates run at P=4 on fake CPU devices; sharded gradients are
  gated on real GPUs (tier 2); small CPU sharded VJP probes are used only after they are measured to work. *Sources:*
  [E§7] PL:541-544; AR:885. [F§3] PL:2559-2566. [F§8] mem:pj/jax-cpu-scaling-findings.md. Plan lines 103-105
  (Development Approach, one code path).
- **L-CONF-10 Tolerance class of reductions: about 1e-13 for stencils [E§4] vs about 1e-12 for scatters and reductions
  [F§2] — resolved by the plan: ~1e-13.** The plan's classes are pointwise ~1e-15 or bitwise and stencils/reductions
  ~1e-13; MITgcm's structured grid has no scatters, and fesom_jax itself expects more bit-exact kernels without them.
  *Sources:* [E§4] KG:56-60. [F§2] PL:17-22. Plan lines 153-155 (Testing Strategy).
- **L-CONF-11 Defaults in static configurations: a default equal to the `data*` value it replaces [F§1] vs no
  Python-side defaults [E§3] — resolved by the plan: no Python defaults.** The plan (Task 6): KeyError unless a cited
  Fortran default is passed. fesom_jax's off-path identity still holds: one "off = identical" test per option
  ([L-ARCH-3]). *Sources:* [F§1] PL:1408-1416; PL:4191-4198. [E§3] mitgcm_jax/params_io.py:30-36; KG:11-14. Plan line
  255 (Task 6).
- **L-CONF-12 Derivatives at an exact kink: JAX's 0.5/0.5 split, unlike TAF [E§6], vs limiters differentiated as written
  (subgradient) [F§3] — decided by Nikolay 2026-10-01: JAX's derivative by default, TAF's branch behind a switch.** The
  plan adopts the subgradient rule for limiters (Task 14). Where a gradient is compared with TAF's (admGrd ≥ 10 digits)
  and the trajectory sits exactly on a kink, such as an exact tie in a max or min, the two conventions can give
  different derivatives. Decision ("take the JAX-optimal route, but be able to compare to TAF as closely as
  possible"): the default is JAX's own derivative; where a TAF comparison lands on a kink, a TAF-compatible
  backward-only switch selects TAF's branch, with the same rules as every TAF switch (forward byte-identical, effect test
  on a live fixture). *Sources:* [E§6] PL:340. [F§3] LG:36-45. Nikolay 2026-10-01.
- **L-CONF-13 Tests for packaging work: "packaging gets no unrequested tests" [E§10] vs the plan's "every task MUST
  include new/updated tests" — resolved by the plan for mitjax tasks.** This is a digest-versus-plan tension, recorded
  so that it is not rediscovered: every mitjax plan task carries tests (plan line 111). *Sources:* [E§10]
  mem/feedback_naming_and_scope.md:15-16; mem/project_private_repo.md:22.

## Mapping: digest sections to lesson IDs

Each row lists the entries that cite the section. The ECCO digest's unnumbered "Top ten" restates bullets of its §2–§9:
item 1 → L-ORA-8; 2 → L-XLA-1, L-XLA-2, L-XLA-4; 3 → L-LIT-8, L-LIT-12, L-ORA-4, L-ORA-12; 4 → L-ORA-9;
5 → L-AD-8, L-AD-9, L-AD-11; 6 → L-AD-13, L-AD-14, L-AD-16; 7 → L-AD-25, L-AD-26, L-AD-31, L-AD-32;
8 → L-PAR-1, L-PAR-2, L-PAR-6, L-PAR-8, L-PAR-10;
9 → L-ARCH-5, L-XLA-4; 10 → L-OPS-1, L-PROC-2, L-PROC-3, L-OPS-3, L-OPS-4, L-OPS-15.

| Section | Digest heading | Lesson IDs |
|---|---|---|
| [E§1] | Process & agents | L-PROC-1, L-PROC-2, L-PROC-3, L-PROC-4, L-ORA-1, L-LIT-1, L-OPS-3, L-USER-1, L-COPY-1, L-COPY-13 |
| [E§2] | Oracle, dumps & gates | L-ORA-3, L-ORA-4, L-ORA-5, L-ORA-6, L-ORA-7, L-ORA-8, L-ORA-9, L-ORA-11, L-ORA-12, L-ORA-14, L-ORA-17, L-ORA-18, L-COPY-13, L-COPY-14, L-COPY-15 |
| [E§3] | Literal translation & configuration | L-LIT-5, L-LIT-6, L-LIT-8, L-LIT-11, L-LIT-12, L-LIT-13, L-LIT-14, L-LIT-15, L-LIT-16, L-LIT-23, L-XLA-8, L-CONF-11 |
| [E§4] | Tolerances & comparators | L-TOL-1, L-TOL-2, L-TOL-3, L-TOL-5, L-TOL-6, L-TOL-14, L-TOL-18, L-CONF-10 |
| [E§5] | JAX/XLA numerics | L-XLA-1, L-XLA-2, L-XLA-4, L-XLA-6, L-XLA-7, L-XLA-8, L-ARCH-7, L-AD-35, L-CONF-1, L-CONF-2, L-CONF-7 |
| [E§6] | Differentiability | L-AD-1, L-AD-3, L-AD-6, L-AD-8, L-AD-9, L-AD-11, L-AD-13, L-AD-14, L-AD-15, L-AD-16, L-AD-18, L-AD-19, L-AD-21, L-AD-22, L-AD-23, L-AD-25, L-AD-26, L-AD-27, L-AD-28, L-AD-29, L-AD-31, L-AD-32, L-AD-33, L-AD-34, L-PERF-10, L-PERF-13, L-CONF-4, L-CONF-6, L-CONF-8, L-CONF-12 |
| [E§7] | Parallel/sharding | L-PAR-1, L-PAR-2, L-PAR-3, L-PAR-6, L-PAR-7, L-PAR-8, L-PAR-9, L-PAR-10, L-PAR-11, L-PAR-12, L-PERF-16, L-CONF-4, L-CONF-9 |
| [E§8] | Performance & GPU | L-PERF-1, L-PERF-2, L-PERF-4, L-PERF-6, L-PERF-7, L-PERF-8, L-PERF-14, L-PERF-15 |
| [E§9] | Operations | L-TEST-1, L-ENV-1, L-ENV-3, L-ENV-4, L-OPS-1, L-OPS-4, L-OPS-5, L-OPS-6, L-OPS-8, L-OPS-9, L-OPS-10, L-OPS-11, L-OPS-12, L-OPS-15, L-COPY-13, L-COPY-16 |
| [E§10] | User preferences (Nikolay) | L-LIT-23, L-AD-4, L-PERF-17, L-USER-1, L-USER-2, L-USER-3, L-USER-4, L-USER-5, L-USER-6, L-CONF-5, L-CONF-13 |
| [E§11] | Infrastructure to copy: known pitfalls and Fortran dependencies | L-ORA-23, L-XLA-5, L-ARCH-5, L-ARCH-7, L-AD-3, L-AD-14, L-AD-18, L-AD-19, L-AD-26, L-AD-32, L-AD-33, L-AD-35, L-PAR-6, L-PAR-10, L-PAR-11, L-PAR-12, L-PERF-6, L-PERF-8, L-PERF-10, L-PERF-11, L-PERF-14, L-TEST-1, L-OPS-5, L-COPY-2, L-COPY-3, L-COPY-4, L-COPY-5, L-COPY-6, L-COPY-7, L-COPY-8, L-COPY-9, L-COPY-10, L-COPY-11, L-COPY-12, L-CONF-3 |
| [P§0] | The ten that matter most | L-PROC-5, L-ORA-1, L-ORA-10, L-LIT-9, L-LIT-17, L-TOL-1, L-TOL-3, L-TOL-4, L-TOL-17, L-TOL-18, L-PAR-4, L-USER-2 |
| [P§1] | Process and agent workflow | L-PROC-5, L-PROC-6, L-PROC-7, L-ORA-1, L-ORA-10, L-LIT-9, L-LIT-10, L-TOL-18 |
| [P§2] | Reference harness and dumps | L-ORA-2, L-ORA-8, L-ORA-15, L-ORA-16, L-TOL-19, L-TOL-20 |
| [P§3] | Literal translation and configuration | L-ORA-12, L-ORA-21, L-LIT-1, L-LIT-2, L-LIT-3, L-LIT-4, L-LIT-7, L-LIT-8, L-LIT-9, L-LIT-10, L-LIT-14, L-LIT-20, L-LIT-23 |
| [P§4] | Validation ladder, tolerances and comparators | L-ORA-13, L-ORA-21, L-ORA-22, L-TOL-1, L-TOL-3, L-TOL-4, L-TOL-7, L-TOL-11, L-TOL-15, L-TOL-16, L-PAR-1 |
| [P§5] | Bug classes that recurred | L-ORA-23, L-LIT-17, L-LIT-18, L-LIT-19, L-TOL-22, L-PAR-12 |
| [P§6] | Parallel, MPI and GPU fidelity | L-TOL-21, L-XLA-9, L-PAR-3, L-PAR-4, L-PAR-5, L-PAR-17 |
| [P§7] | Long-run and climate testing | L-ORA-20, L-TOL-16, L-TOL-17, L-PERF-20, L-OPS-14 |
| [P§8] | Operations | L-PROC-8, L-ORA-22, L-PERF-18, L-OPS-8, L-OPS-10, L-OPS-16 |
| [P§9] | User preferences (Nikolay's recorded corrections) | L-PROC-8, L-LIT-1, L-LIT-10, L-ENV-4, L-OPS-13, L-OPS-15, L-USER-2, L-USER-4, L-USER-5, L-USER-7, L-USER-8 |
| [F§1] | JAX architecture | L-XLA-5, L-XLA-9, L-ARCH-1, L-ARCH-2, L-ARCH-3, L-ARCH-4, L-ARCH-5, L-ARCH-6, L-ARCH-7, L-ARCH-8, L-PERF-1, L-PERF-3, L-PERF-4, L-PERF-5, L-ENV-3, L-CONF-11 |
| [F§2] | XLA numerics and reproducibility | L-LIT-2, L-LIT-7, L-TOL-1, L-TOL-10, L-TOL-12, L-TOL-13, L-XLA-3, L-XLA-9, L-AD-33, L-CONF-1, L-CONF-2, L-CONF-4, L-CONF-7, L-CONF-10 |
| [F§3] | Differentiability | L-AD-1, L-AD-2, L-AD-4, L-AD-5, L-AD-7, L-AD-8, L-AD-9, L-AD-10, L-AD-12, L-AD-16, L-AD-17, L-AD-20, L-AD-23, L-AD-24, L-AD-28, L-AD-30, L-CONF-6, L-CONF-8, L-CONF-9, L-CONF-12 |
| [F§4] | Checkpointing and memory | L-AD-34, L-AD-36, L-PERF-9, L-PERF-10, L-PERF-11, L-PERF-12, L-TEST-11 |
| [F§5] | Parallel and sharding | L-TOL-8, L-TOL-9, L-PAR-1, L-PAR-7, L-PAR-8, L-PAR-10, L-PAR-13, L-PAR-14, L-PAR-15, L-PAR-16, L-PERF-9, L-PERF-22, L-PERF-23, L-PERF-24, L-CONF-7 |
| [F§6] | Performance and GPU | L-PERF-1, L-PERF-8, L-PERF-16, L-PERF-19, L-PERF-20, L-PERF-21, L-OPS-13 |
| [F§7] | Testing | L-ORA-9, L-TOL-1, L-TEST-1, L-TEST-2, L-TEST-3, L-TEST-4, L-TEST-5, L-TEST-6, L-TEST-7, L-TEST-8, L-TEST-9, L-TEST-10, L-OPS-5, L-OPS-7, L-CONF-3 |
| [F§8] | Environment and JAX versions | L-XLA-5, L-PAR-7, L-PAR-11, L-ENV-1, L-ENV-2, L-ENV-4, L-ENV-5, L-CONF-9 |
| [F§9] | Operations | L-PROC-9, L-PROC-10, L-PROC-11, L-ORA-19, L-ORA-22, L-LIT-8, L-LIT-21, L-LIT-22, L-ARCH-9, L-OPS-1, L-OPS-2, L-OPS-3, L-OPS-4, L-OPS-6, L-OPS-7, L-OPS-8 |
| [F§10] | User preferences (Nikolay) | L-PROC-1, L-PROC-2, L-ENV-4, L-OPS-12, L-OPS-16, L-USER-1, L-USER-2, L-USER-6, L-USER-7, L-USER-9 |
