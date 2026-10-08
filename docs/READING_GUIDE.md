# Reading guide: JAX for MITgcm developers

This guide is for a developer who knows MITgcm's Fortran well and Python reasonably, but not JAX. It explains how to
read a routine of `mitjax` next to the `.F` file it ports: what each Python construct means in Fortran terms, why a
few things look different, and where to find the evidence that a routine computes what the Fortran computes. The
rules for writing new code are in `docs/KERNEL_GUIDE.md` and `docs/PORTING_RULES.md`; this guide only explains how to
read.

Conventions used below:
- `mitjax` ports MITgcm master at commit `63cdc0b` (checkpoint69q+9). An upstream citation is written
  `@63cdc0b <path>:<lines>`, e.g. `@63cdc0b model/src/cg2d.F:161-178`; your MITgcm checkout at that commit is
  `$MJX_UPSTREAM` ([install.md](install.md)).
- A citation of this repository is written `<path>:<lines>`, e.g. `mitjax/model/src/cg2d.py:156-166`.
- Every quoted block below is copied verbatim from the cited lines. `mitjax/tests/test_docs_examples.py` checks every
  citation and every quoted block against the repository and the upstream clone, so a quote that drifts from the code
  fails the test suite.

## 1. Where the Fortran went

The tree mirrors MITgcm. One `.py` file per `.F` file, one Python function per subroutine, with the same names in lower
case:

| MITgcm | mitjax |
|---|---|
| `model/src/*.F` | `mitjax/model/src/*.py` |
| `pkg/<pkg>/*.F` | `mitjax/pkg/<pkg>/*.py` |
| `eesupp/src` (exchanges, global sums, tiles) | `mitjax/eesupp` |
| `GRID.h` | `mitjax/model/grid.py` (class `Grid`) |
| `DYNVARS.h`, `SURFACE.h` | `mitjax/model/state.py` (class `State`) |
| `PARAMS.h` | `mitjax/model/src/ini_parms.py` (class `Params`) |
| other headers (`CG2D.h`, `GAD.h`, `GMREDI.h`, `MONITOR.h`, ...) | `<name>_h.py` next to the routines, e.g. `mitjax/model/src/cg2d_h.py`, `mitjax/pkg/generic_advdiff/gad_h.py` |
| `THE_MODEL_MAIN`, `THE_MAIN_LOOP` | `mitjax/drivers/model.py`, `mitjax/drivers/the_main_loop.py`, `mitjax/drivers/run.py` |

Everything that is about JAX rather than about the model lives outside `mitjax/model` and `mitjax/pkg`:
`mitjax/drivers` (time loop, gradient drivers, checkpointing), `mitjax/ad` (custom derivative rules),
`mitjax/eesupp` (exchanges, sums, sharding over devices), `mitjax/ops` (helpers such as safe division, `scan_k`,
Fortran `MAX`/`MIN`), `mitjax/config` (reading an experiment's `SIZE.h`, CPP options and namelists),
`mitjax/io` (dump and STDOUT readers). A test (`mitjax/tests/test_banned_transforms.py`) refuses JAX transformations
such as `jax.jit`, `jax.grad` or `lax.scan` inside `mitjax/model` and `mitjax/pkg`, so the physics code contains only
array arithmetic and the helpers listed in this guide.

Each ported routine's docstring gives the Fortran call line and the upstream citation of the routine (a few packages,
e.g. `pkg/monitor`, put the citation in the module docstring), then the routine's own header comments, then notes on
what was ported and what was not. Comments `# :NN` in the code give the
Fortran line number of the statement or block (`# :86-87` means lines 86-87 of the routine's `.F` file).

## 2. Five facts about JAX that explain most of what you will see

1. **Arrays are values.** A JAX array cannot be changed in place. `A.at[...].set(v)` returns a new array equal to `A`
   except at the indexed points, and the code rebinds the name: `KE = KE.at[i, j].set(...)`. Read it as the Fortran
   assignment `KE(i,j) = ...` inside its loops. (The compiler turns this into an in-place update where it can.)
2. **The driver compiles the step.** The drivers compile a whole time step (or a chunk of time steps) with `jax.jit`.
   To do that, JAX runs the Python function once with placeholder arrays ("tracing") and records the array
   operations into a program. Python `if` and `for` statements run during tracing, on Python values; they are not in
   the compiled program.
3. **Static and traced values.** Hence a Python `if` may only test values known before the run starts: CPP options,
   `SIZE.h`, integer and logical namelist values, a level index `k` that is a Python integer. These are called
   *static*. Arrays and REAL parameters are *traced*: their values are only known while the program runs, so a
   decision on them is written `jnp.where(cond, a, b)`, which computes both `a` and `b` at every point and selects.
4. **A Python loop is unrolled.** `for k in range(1, Nr+1):` with a body that works on arrays produces `Nr` copies of
   the body in the compiled program. This is how most `DO k` loops are written (section 7).
5. **Derivatives are computed by transforming the program.** JAX's reverse mode (the adjoint) is generated from the
   same code; there is no hand-written adjoint per routine (the exceptions are a few rules in `mitjax/ad` and
   `mitjax/ops`, such as the solver rule of section 11). Two consequences show up in the physics code: in
   `jnp.where(cond, a, b)` the derivative of the branch not taken is multiplied by zero, and `0*inf` is NaN, so both
   branches must be finite at every point (section 6); and iterative solvers are not differentiated through their
   iterations but by a rule (section 11).

## 3. Fortran-index arrays

Every array in the physics code is an `FArray` (`mitjax/farray.py`) that carries its Fortran declaration. The
declaration `_RL uFld(1-OLx:sNx+OLx,1-OLy:sNy+OLy)` becomes

```text
uFld = FArray(data, "uFld", i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
```

and a three-dimensional field such as `hFacW(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)` is declared with `k=(1, Nr)`.
Bounds always come from the declaration, so an array declared `(Nr+1)` cannot be mistaken for one declared `(Nr)`.

Loops are named index ranges, written with the bounds of the `DO` statement:
- `j = loop_j(1-OLy, sNy+OLy-1)` is `DO j=1-OLy,sNy+OLy-1`, and `i = loop_i(...)` likewise;
- `uFld[i+1, j]` reads `uFld(i+1,j)` for every `(i,j)` of the loops at once; the indices are in Fortran order (`i`
  first) and use Fortran values, never storage offsets;
- `KE.at[i, j].set(expr)` writes `KE(i,j) = expr` for every `(i,j)` of the loops; every point outside the loop range
  keeps its previous value, as a Fortran loop that does not cover the whole array leaves the other points alone;
- `A.local("name")` declares a local array with the same bounds as `A`, filled with NaN, so that reading a point the
  Fortran never wrote shows up as NaN in the gates.

Indices are checked against the declaration when the code is traced. Two real error messages:

```text
IndexError uFld(i+1, j): i+1 = -1..14 is outside the declared -2:13 of dimension 1 of uFld(-2:13,-2:13)
IndexError uFld(j, i): index 1 is a j loop, but dimension 1 of uFld(-2:13,-2:13) is i
```

The first comes from `loop_i(1-OLx, sNx+OLx)` with `OLx = 3`, `sNx = 10`: the shifted read goes past the declared
halo. The Fortran, compiled without bounds checking, would read past the array silently; here it is an error at
trace time.

**Example 1: MOM_CALC_KE**, the branch `KEscheme = -1`. Fortran, `@63cdc0b pkg/mom_common/mom_calc_ke.F:61-68`:

```fortran
      IF (KEscheme.EQ.-1) THEN
       DO j=1-OLy,sNy+OLy-1
        DO i=1-OLx,sNx+OLx-1
         KE(i,j) = 0.125*(
     &             ( uFld(i,j)+uFld(i+1, j ) )**2
     &            +( vFld(i,j)+vFld( i ,j+1) )**2 )
        ENDDO
       ENDDO
```

Python, `mitjax/pkg/mom_common/mom_calc_ke.py:36-41`:

```python
    j = loop_j(1-OLy, sNy+OLy-1)                                    # DO j=1-OLy,sNy+OLy-1 (:62, 77, ...)
    i = loop_i(1-OLx, sNx+OLx-1)
    if KEscheme == -1:                                              # :61-68
        KE = KE.at[i, j].set(0.125*(
                   (uFld[i, j]+uFld[i+1, j])**2
                  +(vFld[i, j]+vFld[i, j+1])**2))
```

The statement keeps the Fortran's operands, order and parentheses (the port never re-associates an expression: a
different association changes the last bit). `0.125` is a REAL*4 literal in the Fortran (the oracle build has no
`-fdefault-real-8`); it is exact in binary, so the Python double is the same number. A literal that is not exact in
binary (`0.1` without `_d 0`) is written through `mitjax.config.fortran.real4`, which rounds the decimal to binary32 as
gfortran does (`mitjax/tests/test_real4.py`). `KEscheme` is a static integer, so `if KEscheme == -1:` is a Python `if`.
Output arguments are also inputs: `KE` is passed in, because the points the loop does not write (the last row and
column of the halo) keep their previous values.

The `#ifdef ALLOW_AUTODIFF` block that zeroes `KE` first is a static test of the routine's own CPP view, Python,
`mitjax/pkg/mom_common/mom_calc_ke.py:31-34`:

```python
    if cfg.cpp.flag("ALLOW_AUTODIFF", _OPT):                       # :53-59
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx)
        KE = KE.at[i, j].set(0.)
```

## 4. The tile loop is implicit

Arrays are stored as `[tile, k, j, i]` (or `[tile, j, i]`): MITgcm's `(i, j, k, bi, bj)` with the tile axis first and
the Fortran order reversed, as C-order storage of a Fortran array is. The Fortran index `i` sits at storage index
`i-1+OLx`; you never see that offset in the physics code, because `FArray` applies it. Every statement acts on all
tiles at once, so the `DO bj; DO bi` loops and the `bi, bj` arguments disappear: `MOM_CALC_KE(bi,bj,k,KEscheme,...)`
becomes `mom_calc_ke(k, KEscheme, ...)`. Arrays that the Fortran declares without `bi,bj` (e.g. `recip_deepFacC(Nr)`)
are declared `tiled=False`.

A value that is a scalar per tile in the Fortran (a tile-corner coordinate, an error code) becomes an array with one
entry per tile: `solve_tridiagonal` returns `errCode` as an array `[tile]` because the Fortran calls it once per tile,
and `ini_local_grid` keeps `xG0` as `[tile, 1, 1]` so that it broadcasts over `(j, i)`.

The same code runs on one device (all tiles in one array) and on several devices (one block of tiles per device,
`mitjax/eesupp/shard.py`). The only places where the two differ are the exchanges and the global sums, which go
through `mitjax/eesupp` (section 10). Gates compare a 4-device run with a 1-device run bit for bit.

## 5. CPP options, SIZE.h and namelist values

The configuration of an experiment variant is read by `mitjax.config.params.load(experiment, variant)` in the way
`testreport` builds and runs it: `SIZE.h` and the `*_OPTIONS.h` of the experiment's `code/` (or `code_ad/`) directory,
preprocessed by the oracle build's own `cpp`, and the namelists of `input*/` read with the `NAMELIST` statements and
declarations of the preprocessed Fortran reader routines (`mitjax/config/namelists.py`). The physics code sees it
through keyword-only arguments:

- `cfg`: static and hashable. `cfg.size.sNx`, `cfg.size.Nr`, ... are the `SIZE.h` parameters. `cfg.cpp.NAME` is True
  when `NAME` is defined (an `#ifdef NAME`); `cfg.cpp.flag(NAME, "GAD_OPTIONS.h")` is the same test as seen by a
  routine that includes that header. The distinction matters because a package's options file may change what
  `CPP_OPTIONS.h` defined, and because `tutorial_global_oce_optim/code_ad/CPP_OPTIONS.h` does not include
  `PACKAGES_CONFIG.h`. A name that no compiled source tests raises `UnknownCppOption`, so a misspelt option cannot
  silently read as undefined (`mitjax/config/cpp_options.py:546-578`). `cfg.use_flag("useGMRedi")` is a package switch
  as `packages_boot.F` leaves it.
- `params`: the `PARAMS.h` values by Fortran name (`params.viscAh`, `params.usingZCoords`), built by `ini_parms_dyn` in
  `mitjax/model/src/ini_parms.py`. LOGICAL, INTEGER and CHARACTER values are static; REAL values are traced (they are
  arguments of the compiled program, so a derivative with respect to them exists and a change of value does not
  recompile).
- `grid` (`GRID.h`), `state` (`DYNVARS.h`, `SURFACE.h`) and, where a routine reads another header, an object named
  after it: `cg2dh` (`CG2D.h`), `eos` (`EOS.h`), `ff` (`FFIELDS.h`), `mon` (`MONITOR.h`), `ex` (the exchanger).

A value that the namelist files do not set is taken from the Fortran default **with its citation**. For example, the
monitor reads `monitor_stdio` from the file, else from `@63cdc0b model/src/set_defaults.F:354`, Python,
`mitjax/drivers/run.py:103-104`:

```python
            monitor_stdio=rp.get("data", "PARM03", "monitor_stdio",
                                 default=fortran_default("model/src/set_defaults.F:354", "monitor_stdio", m.exp)),
```

`fortran_default` (`mitjax/params_io.py`) reads the value from that line of `set_defaults.F` and refuses the line when
it is not compiled in this build (an `#ifdef` branch not taken), sits in a run-time `IF` block, or is assigned again
before the namelist is read. There are no defaults written in Python.

A Fortran `IF` on a REAL namelist value needs care, because a REAL is traced. The rule (`docs/KERNEL_GUIDE.md` §4):
- if the `IF` chooses between two values, it stays a `jnp.where`; e.g. `IF ( sideDragFactor.LE.0. )` in
  `@63cdc0b pkg/mom_common/mom_u_sidedrag.F:58` computes both versions and selects, Python,
  `mitjax/pkg/mom_common/mom_u_sidedrag.py:105`:

```python
    uDragTerms = uDragTerms.at[i, j].set(jnp.where(sideDragFactor <= 0., old, new))   # :58, :100, :145
```

- if it decides which code runs, the comparison is made once on the host from the namelist value and stored as a
  static flag with a name that says what was compared; e.g. `bottomDragLinear.NE.0.` in
  `@63cdc0b pkg/mom_fluxform/mom_fluxform.F:270-277` becomes `params.bottomDragLinear_ne_0`
  (`mitjax/model/src/ini_parms.py:1077`), Python, `mitjax/pkg/mom_fluxform/mom_fluxform.py:125-128`:

```python
    # :270-277 bottomDragTerms; `bottomDragLinear.NE.0.` decided on the host (params.bottomDragLinear_ne_0)
    bottomDragTerms = (params.selectImplicitDrag == 0
                       and (params.no_slip_bottom or params.selectBotDragQuadr >= 0
                            or params.bottomDragLinear_ne_0))
```

  A derivative with respect to `bottomDragLinear` must then not move it across zero, since the flag would be stale
  (`docs/KERNEL_GUIDE.md` §4).

Two kernel families still receive a flat configuration object instead of `cfg.size` and `cfg.cpp`: most
`generic_advdiff` kernels read `cfg.sNx`, `cfg.ALLOW_AUTODIFF`, ... from a `GadKernelCfg` that
`gad_kernel_cfg(cfg)` builds from `SIZE.h` and the macros after `GAD_OPTIONS.h`
(`mitjax/pkg/generic_advdiff/gad_calc_rhs.py:18-39`), and `pkg/monitor` reads a namespace built by the driver
(`mitjax/drivers/run.py:81-99`). The values are the same; only the spelling differs.

Where a ported routine reaches an option or a branch that is not ported, it raises `NotImplementedError` (naming the
routine and the option) when the code is traced, and a Fortran `STOP` raises with the Fortran message, so an
experiment that needs an unported branch stops before it runs instead of giving wrong numbers.

## 6. Pointwise IF: `jnp.where`, with both branches finite

An `IF` on array values inside a loop becomes `jnp.where(cond, a, b)`. Both `a` and `b` are computed at every point.
For the forward value that is harmless, but for the derivative it is not: the reverse pass multiplies the derivative of
the unused branch by zero, and if that derivative is infinite (a division by zero on a point where the Fortran never
divides) the result is NaN. The port therefore guards the operation itself, so that the unused branch is finite:
`safe_div(num, den, mask)` (`mitjax/ops/safe.py`) divides only where `mask` is true and uses a harmless denominator
elsewhere; where `mask` is true the value is bit for bit `num/den`.

**Example 2: GAD_DST3FL_ADV_X**, the flux limiter inside the loop over points. Fortran,
`@63cdc0b pkg/generic_advdiff/gad_dst3fl_adv_x.F:74-90`:

```fortran
        IF ( ABS(Rj)*thetaMax .LE. ABS(Rjm) ) THEN
          thetaP=SIGN(thetaMax,Rjm*Rj)
        ELSE
          thetaP=Rjm/Rj
        ENDIF
        IF ( ABS(Rj)*thetaMax .LE. ABS(Rjp) ) THEN
          thetaM=SIGN(thetaMax,Rjp*Rj)
        ELSE
          thetaM=Rjp/Rj
        ENDIF

        psiP=d0+d1*thetaP
        psiP=MAX(0. _d 0,MIN(MIN(1. _d 0,psiP),
     &                       thetaP*(1. _d 0 -uCFL)/(uCFL+1. _d -20) ))
        psiM=d0+d1*thetaM
        psiM=MAX(0. _d 0,MIN(MIN(1. _d 0,psiM),
     &                       thetaM*(1. _d 0 -uCFL)/(uCFL+1. _d -20) ))
```

Python, `mitjax/pkg/generic_advdiff/gad_dst3fl_adv_x.py:61-73`:

```python
    bigP = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjm)                     # :74-78
    thetaP = jnp.where(bigP, jnp.copysign(thetaMax, Rjm*Rj),
                       safe_div(Rjm, Rj, ~bigP))
    bigM = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjp)                     # :79-83
    thetaM = jnp.where(bigM, jnp.copysign(thetaMax, Rjp*Rj),
                       safe_div(Rjp, Rj, ~bigM))

    psiP = d0+d1*thetaP                                             # :85-90
    psiP = MAX(0., MIN(MIN(1., psiP, p="b"),                       # :86-87
                       thetaP*(1.-uCFL)/(uCFL+1.e-20), p="b"), p="b")
    psiM = d0+d1*thetaM
    psiM = MAX(0., MIN(MIN(1., psiM, p="b"),                       # :89-90
                       thetaM*(1.-uCFL)/(uCFL+1.e-20), p="b"), p="b")
```

Three things to notice:
- Each statement runs on the whole `(i, j)` range at once (the loops `j` and `i` are set at
  `mitjax/pkg/generic_advdiff/gad_dst3fl_adv_x.py:41-46`). That is equivalent to the Fortran's per-point body
  because every point reads only inputs, never a value another point of the same loop wrote; the docstring states this.
- `SIGN(a,b)` is `jnp.copysign(a, b)`: gfortran's default `-fsign-zero` gives `-|a|` for `b = -0`, as `copysign` does.
- `MAX` and `MIN` are not `jnp.maximum`/`jnp.minimum` (section 9).

**Example 3: CALC_IVDC.** A two-way `IF` that sets a value, Fortran, `@63cdc0b model/src/calc_ivdc.F:45-53`:

```fortran
       DO j=jMin,jmax
        DO i=iMin,imax
         IF ( -sigmaR(i,j,k)*gravitySign.GT.0. ) THEN
          IVDConvCount(i,j,k,bi,bj) = 1. _d 0
         ELSE
          IVDConvCount(i,j,k,bi,bj) = 0. _d 0
         ENDIF
        ENDDO
       ENDDO
```

Python, `mitjax/model/src/calc_ivdc.py:21-25`:

```python
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    # :47  IF ( -sigmaR(i,j,k)*gravitySign.GT.0. ) THEN 1. _d 0 ELSE 0. _d 0
    return state.IVDConvCount.at[i, j, K].set(
        jnp.where(-sigmaR[i, j, K]*grid.gravitySign > 0.0, 1.0, 0.0))
```

Fortran is case-insensitive; the Python keeps the spelling of the argument list (`K`, `jMax`). `IVDConvCount` is a
`DYNVARS.h` common-block variable, so the routine reads it from `state` and returns the new array; the caller stores
it back, Python, `mitjax/model/src/do_oceanic_phys.py:391-393`:

```python
        if k > 1 and calcConvect:                                               # :867-875
            IVDConvCount = calc_ivdc(iMin, iMax, jMin, jMax, k, sigmaR, myTime, myIter, grid=grid, state=state)
            state = state.replace(IVDConvCount=IVDConvCount)
```

## 7. Vertical loops

A `DO k` loop is written in one of three ways, and the docstring says which and why:

1. **Iterations independent of each other** (each level reads only inputs): the loop may be vectorised over `k`,
   with `k, j, i = loops_kji((1, Nr), (jlo, jhi), (ilo, ihi))` (`mitjax/farray.py`); every statement then acts on all
   levels at once.
2. **A routine the Fortran calls once per level** (`CALL CALC_IVDC(bi,bj,...,k,...)` inside `DO k=Nr,1,-1`): the
   routine keeps `k` as its argument and is written exactly as for one level; its caller writes the loop body once,
   as a function `level(k, c)` of the level and of the tuple `c` of what the Fortran carries from one level to the
   next, and runs it with `scan_levels` (`mitjax/ops/scan_k.py`, decided 2026-10-02, KERNEL_GUIDE §4). Inside the
   scan `k` is a traced level index (`KIdx`, `mitjax/farray.py`): array indexing with it works as with an integer,
   and a test such as `k > 1` is decided from the range of levels the scan covers -- a test that range does not
   decide raises, so the levels where the Fortran branches on `k` (here `k = 1`) run as separate calls, visibly.
   The body is compiled for two levels instead of `Nr`, and the values are bit for bit those of the loop in the
   Fortran's order. Example 3's call sits in such a body (the function starts at
   `mitjax/model/src/do_oceanic_phys.py:218`), Python, `mitjax/model/src/do_oceanic_phys.py:372-373`:

```python
    def sigma_k(k, c):
        rhoKm1, rhoKp1, sigmaX, sigmaY, sigmaR, state = c
```

   and the Fortran's `DO k=Nr,1,-1` (`@63cdc0b model/src/do_oceanic_phys.F:803`) becomes, Python,
   `mitjax/model/src/do_oceanic_phys.py:396-401`:

```python
    from mitjax.ops.scan_k import scan_levels
    c = (rhoKm1, rhoKp1, sigmaX, sigmaY, sigmaR, state)
    if sz.Nr >= 2:
        c = scan_levels(sigma_k, c, 2, sz.Nr, down=True)
    c = sigma_k(1, c)
    rhoKm1, rhoKp1, sigmaX, sigmaY, sigmaR, state = c
```

3. **A recursion in `k`** (each level reads what the previous iteration wrote: tridiagonal solves, integrations from
   the surface or the bottom): `scan_k` (`mitjax/ops/scan_k.py`). The loop is given as a Python `range` written as the
   `DO` statement (`DO k=Nr-1,1,-1` is `range(Nr-1, 0, -1)`); a function `level_k(k)` collects what iteration `k`
   reads, with Fortran level indices; the loop body `iteration(carry, x)` is one pass of the Fortran loop, where
   `carry` is what iteration `k` takes from iteration `k+1` (or `k-1`). The body is compiled once, not `Nr` times,
   and the values are bit for bit those of the loop in the Fortran's order (`mitjax/tests/test_scan_k.py`).

**Example 4: SOLVE_TRIDIAGONAL, backward sweep.** Fortran, `@63cdc0b model/src/solve_tridiagonal.F:277-295`:

```fortran
CADJ loop = sequential
C--   Backward sweep
      DO k=Nr,1,-1
       IF ( k.EQ.Nr ) THEN
        DO j=1-OLy,sNy+OLy
         DO i=1-OLx,sNx+OLx
          y3d(i,j,k) = y3d_prime(i,j,k)
         ENDDO
        ENDDO
       ELSE
        DO j=1-OLy,sNy+OLy
         DO i=1-OLx,sNx+OLx
          y3d(i,j,k) = y3d_prime(i,j,k)
     &               - c3d_prime(i,j,k)*y3d(i,j,k+1)
         ENDDO
        ENDDO
       ENDIF
C-    end k-loop
      ENDDO
```

Python, `mitjax/model/src/solve_tridiagonal.py:76-86`:

```python
    # :279-295  backward sweep
    y3d = y3d.at[i, j, Nr].set(y3d_prime[i, j, Nr])                     # :280-285  k = Nr

    def backward(ykp1, x):
        yk = x["y"].at[i, j].set(x["yp"][i, j]                          # :289-290
                                 - x["cp"][i, j]*ykp1[i, j])
        return yk, yk

    _, bw = scan_k(backward, level(y3d, Nr), range(Nr-1, 0, -1),
                   lambda k: {"y": level(y3d, k), "yp": level(y3d_prime, k), "cp": level(c3d_prime, k)})
    y3d = set_levels(y3d, bw, range(1, Nr))
```

The `IF ( k.EQ.Nr )` case runs before the scan, because the compiled loop body cannot depend on `k`. The carry
`ykp1` is `y3d(:,:,k+1)`; `level(A, k)` is the two-dimensional level `A(:,:,k)` as an `FArray`; `set_levels` writes the
results back at levels `1..Nr-1`. The forward sweep of the same routine shows the guarded division of section 6 on a
pivot test, Python, `mitjax/model/src/solve_tridiagonal.py:51-55`:

```python
    nz = b3d[i, j, 1] != 0.0                                            # IF ( b3d(i,j,1).NE.0. _d 0 ) THEN
    recVar = safe_div(1.0, b3d[i, j, 1], nz)                            # recVar = 1. _d 0 / b3d(i,j,1)
    c3d_prime = c3d_prime.at[i, j, 1].set(jnp.where(nz, c3d[i, j, 1]*recVar, 0.0))
    y3d_prime = y3d_prime.at[i, j, 1].set(jnp.where(nz, y3d_m1[i, j, 1]*recVar, 0.0))
    errCode = jnp.where(jnp.any(~nz, axis=(1, 2)), 1, errCode)
```

## 8. Recursions along i or j

A loop along `i` (or `j`) whose iteration reads what the previous iteration wrote cannot be written as one array
statement. It is a Python loop over a one-point loop index, in the Fortran's order.

**Example 5: INI_LOCAL_GRID, grid-line coordinates.** Fortran, `@63cdc0b model/src/ini_local_grid.F:153-164`:

```fortran
        DO j=1-OLy,sNy+OLy +1
         xGloc(1-OLx,j) = xG0
         DO i=1-OLx,sNx+OLx
          xGloc(i+1,j) = xGloc(i,j) + delXloc(i)
         ENDDO
        ENDDO
        DO i=1-OLx,sNx+OLx +1
         yGloc(i,1-OLy) = yG0
         DO j=1-OLy,sNy+OLy
          yGloc(i,j+1) = yGloc(i,j) + delYloc(j)
         ENDDO
        ENDDO
```

Python, `mitjax/model/src/ini_local_grid.py:99-108`:

```python
    j = loop_j(1-OLy, sNy+OLy+1)                                    # DO j=1-OLy,sNy+OLy +1
    xGloc = xGloc.at[1-OLx, j].set(xG0)                             #  xGloc(1-OLx,j) = xG0
    for ii in range(1-OLx, sNx+OLx+1):                              #  DO i=1-OLx,sNx+OLx
        i = loop_i(ii, ii)
        xGloc = xGloc.at[i+1, j].set(xGloc[i, j] + delXloc[i])
    i = loop_i(1-OLx, sNx+OLx+1)                                    # DO i=1-OLx,sNx+OLx +1
    yGloc = yGloc.at[i, 1-OLy].set(yG0)                             #  yGloc(i,1-OLy) = yG0
    for jj in range(1-OLy, sNy+OLy+1):                              #  DO j=1-OLy,sNy+OLy
        j = loop_j(jj, jj)
        yGloc = yGloc.at[i, j+1].set(yGloc[i, j] + delYloc[j])
```

The outer loop (`j` for `xGloc`) is independent and stays vectorised; only the recursive direction is a Python loop.
The grid routines run once, before the model is compiled, so the cost of the unrolled loop is paid only at set-up.

## 9. MAX and MIN

gfortran compiles each argument step of `MAX(a,b)` to one `maxsd` (or `minsd`) instruction, which returns one fixed
operand when the two are equal (+0 vs -0) or when one is NaN. Which Fortran argument that is depends on the call site
(operand order after the compiler's canonicalisation and register allocation). `jnp.maximum` gives the first argument
on a tie and propagates NaN, which matches gfortran at no site in general. So every Fortran `MAX`/`MIN` of REAL values
is written `MAX(a, b, p="a")` or `p="b"` (`mitjax/ops/fortran_minmax.py`), where `p` names the argument that wins a tie
or a NaN at that site, measured from the project's gfortran builds of the verification experiments
(`tools/fortran_minmax_sites.py`; the measured tables stay with the project's Fortran reference data, and
`mitjax/tests/test_minmax_sites.py` checks every call in `mitjax/model` and `mitjax/pkg` against them); with more
arguments, one letter per step of the left fold. At a few statements the winner depends on the build; there your own
build takes a documented default with one warning ([configurations.md](configurations.md#maxmin-at-build-dependent-statements)). In Example 2, `MAX(0. _d 0, ...)` at `@63cdc0b pkg/generic_advdiff/gad_dst3fl_adv_x.F:86` keeps the second
argument on a tie, so `MAX(0., MIN(...), p="b")` returns `-0` where the second argument is `-0`, as gfortran does. The
derivative is JAX's own (`jnp.maximum`'s); only the value on ties and NaN differs. A running maximum over a loop nest
and the tiles (`rhsMax = MAX(ABS(cg2d_b(i,j,bi,bj)),rhsMax)`) is `MAX_CHAIN` (Example 6).

## 10. Exchanges and global sums

Exchanges are called where the Fortran calls them, through the exchanger `ex` of `mitjax/eesupp`
(`ex.EXCH_XY_RL(phi)`, `ex.EXCH_UV_XY_RL(u, v, withSigns)`, ...). An exchange is a gather from a map that records,
for every halo point, the interior point (or, for exch2 vector exchanges, the combination of components) it receives;
the maps were measured by running the Fortran's own exchange routines on probe fields for every layout the experiments
use, so the halo values are those of the Fortran, signed zeros included. On several devices the same maps are applied
with `ppermute` between devices; the physics code does not change.

Global sums keep the Fortran's order of additions, because floating-point addition is not associative and a different
order changes the last bits. A MITgcm global sum has two stages: the caller forms a partial sum per tile in its own
`DO j; DO i` order, then `GLOBAL_SUM_TILE_RL` adds the tiles in tile order. In `mitjax` the first stage is
`tile_sum_fortran` (`mitjax/eesupp/global_sum.py`), an explicit chain of additions in the `DO j; DO i` order, and the
second is `ex.global_sum_tile`. `jnp.sum` is never used for a Fortran sum: it adds in a tree order.

**Example 6: CG2D, the initial residual.** Fortran, `@63cdc0b model/src/cg2d.F:161-178`:

```fortran
        DO j=1,sNy
         DO i=1,sNx
          cg2d_r(i,j,bi,bj) = cg2d_b(i,j,bi,bj) -
     &    (aW2d(i  ,j  ,bi,bj)*cg2d_x(i-1,j  ,bi,bj)
     &    +aW2d(i+1,j  ,bi,bj)*cg2d_x(i+1,j  ,bi,bj)
     &    +aS2d(i  ,j  ,bi,bj)*cg2d_x(i  ,j-1,bi,bj)
     &    +aS2d(i  ,j+1,bi,bj)*cg2d_x(i  ,j+1,bi,bj)
     &    +aC2d(i  ,j  ,bi,bj)*cg2d_x(i  ,j  ,bi,bj)
     &    )
#ifdef CG2D_SINGLECPU_SUM
          localBuf(i,j,bi,bj) = cg2d_r(i,j,bi,bj)*cg2d_r(i,j,bi,bj)
#else
          errTile(bi,bj)    = errTile(bi,bj)
     &                      + cg2d_r(i,j,bi,bj)*cg2d_r(i,j,bi,bj)
#endif
          sumRHStile(bi,bj) = sumRHStile(bi,bj) + cg2d_b(i,j,bi,bj)
         ENDDO
        ENDDO
```

Python, `mitjax/model/src/cg2d.py:165-175`:

```python
    cg2d_r = cg2d_r.at[i, j].set(cg2d_b[i, j] -
                                 (aW2d[i, j]*cg2d_x[i-1, j]
                                  + aW2d[i+1, j]*cg2d_x[i+1, j]
                                  + aS2d[i, j]*cg2d_x[i, j-1]
                                  + aS2d[i, j+1]*cg2d_x[i, j+1]
                                  + aC2d[i, j]*cg2d_x[i, j]
                                  ))                                            # :163-169
    rr = cg2d_r[i, j]*cg2d_r[i, j]                                             # :157, :170-175
    sumRHStile = tile_sum_fortran(cg2d_b[i, j])                                 # :156, :176
    cg2d_r = EXCH_S3D_RL(cg2d_r, 1, ex=ex)                                      # :181
    err_sq = gsum(rr)                                                           # :182-186
```

The Fortran loop body does three things per point; the port writes them as three statements. The residual is computed
for all points first, then each accumulator is one ordered chain over the same points. The two accumulators do not
depend on each other, so the order of additions within each is exactly the Fortran's. `gsum` is the build's sum of
the products: without `CG2D_SINGLECPU_SUM` the per-tile chain `tile_sum_fortran` (errTile) and GLOBAL_SUM_TILE_RL;
with it (lab_sea/code_ad) the products are `localBuf` and GLOBAL_SUM_SINGLECPU_RL adds them in one chain over the
global array (`mitjax/eesupp/global_sum_singlecpu.py`).

The running maximum before it, `rhsMax = MAX(ABS(cg2d_b(i,j,bi,bj)),rhsMax)` over the tiles and the `DO j; DO i` nest
(`@63cdc0b model/src/cg2d.F:105-115`), is one chain in tile order, Python, `mitjax/model/src/cg2d.py:149`:

```python
    rhsMax = MAX_CHAIN(0.0, jnp.abs(cg2d_b[i, j]), p="a", acc="b", ex=ex)      # :111
```

## 11. The pressure solver and its derivative

`cg2d` (`mitjax/model/src/cg2d.py`) is the literal CG2D: the same normalisation, stencils, sums and exchanges, and the
same stopping test (`IF ( err_sq .LT. cg2dTolerance_sq ) GOTO 11`), so the number of iterations is the Fortran's. The
iteration is a `lax.while_loop`, because the number of iterations is only known at run time (`while_loop` is not
banned in the physics code; `lax.scan` is, in favour of `scan_k`). Gates compare the solution on all points, the
iteration count and the residuals with the oracle bit for bit (`mitjax/tests/test_cg2d.py`).

Differentiating through the iterations would be wrong: the derivative of an iteration that stops on a tolerance is not
the derivative of the solution, and reverse mode through a `while_loop` is not possible anyway. `SOLVE_FOR_PRESSURE`
therefore calls `cg2d_solve`, which wraps `cg2d` in the implicit-derivative rule of `mitjax/ad/cg2d_rule.py` (with
`useNSACGSolver`, `cg2d_nsa_solve`, the same rule around CG2D_NSA). Python,
`mitjax/model/src/solve_for_pressure.py:78-84`:

```python
    solver = cg2d_solve                                                         # :306-313  CALL CG2D
    if cfg.cpp.ALLOW_CG2D_NSA and cg2d_params.useNSACGSolver:                   # :297-305 (PTRACERS lane)
        from mitjax.model.src.cg2d_nsa import cg2d_nsa_solve
        solver = cg2d_nsa_solve                                                 # CALL CG2D_NSA
    cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters, nIterMin, printed = solver(   # :308
        cg2d_b, cg2d_x, numIters, nIterMin, cfg=cfg, cg2dh=cg2dh, params=cg2d_params, ex=ex)
    cg2d_x = EXCH_XY_RL(cg2d_x, ex=ex)                                          # :315  _EXCH_XY_RL( cg2d_x )
```

What the rule does:
- **Forward value:** the output is the array `cg2d` returns, on every point including the halos, bit for bit. The rule
  changes nothing in the forward model.
- **Derivative:** the converged solution `x` satisfies `M x = b`, with `M` the five-point operator built from
  `aW2d`, `aS2d`, `aC2d` (normalised by `cg2dNorm`). Differentiating that equation gives
  `dx = M^-1 (db - dM x)`. The rule computes this with a separate preconditioned conjugate-gradient solve of the same
  operator from a zero first guess to a relative residual of `1e-13` (tighter than the forward tolerance), used for the
  tangent and, because `M` is symmetric, for the adjoint. So the tangent-linear and adjoint models are exact
  transposes of each other, and the first guess has no derivative (the solution does not depend on it).
- **Halos:** the derivative is computed on the interior points; the halos of `cg2d_x` are overwritten by the
  `_EXCH_XY_RL( cg2d_x )` that follows the call (`@63cdc0b model/src/solve_for_pressure.F:315`), so their derivative is
  that of the exchange.

TAF's adjoint of CG2D (`CG2D_MAD`, `pkg/autodiff/cg2d_mad.F`) is also a solve with the same symmetric operator, but to
the forward tolerance. Whether it adds the operator's derivative is a run-time parameter: `cg2dFullAdjoint`
(`data.autodiff`, default `.FALSE.`): `.FALSE.` leaves the operator passive (`@63cdc0b pkg/autodiff/cg2d_mad.F:220-236`), `.TRUE.`
adds the `aW2d`, `aS2d`, `aC2d` terms (`@63cdc0b pkg/autodiff/cg2d_mad.F:181-204`, only with `NONLIN_FRSURF` or
`ALLOW_DEPTH_CONTROL`). The port follows the
run's setting the same way: `mitjax/ad/modes.py` decides, from the build and `cg2dFullAdjoint`, whether the rule
includes the `dM x` term (`cg2d_implicit(..., operator_active=)`); the exact derivative (always with `dM x`) is the
option `Cg2dParams.mjx_cg2d_derivative = "exact"`. The switch acts only inside the derivative rule, so the forward model
is the same with either setting. Where no control reaches the operator (every M1 experiment) the two settings agree;
in `global_ocean.cs32x15/input_ad` (r*, the operator rebuilt every step, `cg2dFullAdjoint = .FALSE.`) they do not, and
the run's setting is the one that matches TAF.

## 12-13. Moved

How a routine is shown to be right (the Fortran reference, stage dumps, replay harnesses, whole runs, negative
controls) and how to find your way in the repository are now in [checking_a_change.md](checking_a_change.md).
