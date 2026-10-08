"""`python -m mitjax <command>`: the CLI mirror of the API (mitjax/api.py).

    python -m mitjax run EXP_DIR --variant V --out DIR [--devices N]
                                                                 forward run through the API (exp.run(DIR, devices=N),
                                                                 which writes as mitjax/drivers/run.py writes):
                                                                 DIR/rundir/output.txt (cg2d lines, %MON blocks) and
                                                                 the pickups the run's schedule asks for; prints
                                                                 make_rundir's RUNDIR line (prepare_run variants) and
                                                                 the path of output.txt
    python -m mitjax gradient EXP_DIR --variant V --out DIR [--mode run|exact] [--devices N] [--control NAME]
                                                    [--lsr-derivative run|sweeps] [--cg2d-derivative run|exact]
                                                                 dfc/dxx of the run's own cost and control: prints fc,
                                                                 writes DIR/rundir/adxx_<name>.<optimcycle>.data/.meta
    python -m mitjax grdchk EXP_DIR --variant V --out DIR [--mode run|exact] [--lsr-derivative run|sweeps]
                                                    [--cg2d-derivative run|exact]
                                                                 the gradient check at data.grdchk's points:
                                                                 DIR/rundir/output.txt as output_adm.txt prints it
    python -m mitjax compare OUTPUT EXP_DIR --variant V [--kind fwd|adm]
                                                                 testreport's digits of OUTPUT vs EXP_DIR/results
Exit codes: 0, and for compare testreport_jax's (0 pass, 1 FAIL, 3 N/O); 2 for a usage error, including --devices
above the JAX device count (run, gradient: the API's DeviceCountError text, before any run directory is made).

XLA flags: `run`, `gradient` and `grdchk` go through the API, which sets XLA_FLAGS with
mitjax.xla_flags.set_api_xla_flags (the gate flags, --xla_cpu_max_isa=AVX only on x86-64, the user's own XLA_FLAGS
entries win; docs plan 20261006 decision 12 b). On x86-64 with no user XLA_FLAGS that is the gates' string
(set_gate_xla_flags), and a CLI started by a test inherits the test's gate XLA_FLAGS unchanged; the gates themselves
(conftest.py, the gate scripts, drivers/run.run) keep the strict set_gate_xla_flags.
"""

import argparse


def cli_experiment(exp_dir, variant):
    """mitjax.api.load(exp_dir, variant), except that its run directory keeps make_rundir's own stdout: the CLI
    prints the `RUNDIR <path>` line of a prepare_run variant as it did before it went through the API (the API
    itself prints none). Everything else is the API's Experiment."""
    from mitjax import api
    from mitjax.drivers.run import make_rundir

    class CliExperiment(api.Experiment):
        def _rundir(self, out):
            return make_rundir(self.exp_dir, self.variant, out)
    return CliExperiment(exp_dir, variant)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mitjax")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="forward run of a verification experiment")
    parsers = {"run": r}
    r.add_argument("experiment_dir", help="<MITgcm tree>/verification/<experiment>")
    r.add_argument("--variant", required=True, help="input directory: input, input.<v>, ...")
    r.add_argument("--out", required=True, help="new output directory (must not exist)")
    r.add_argument("--devices", type=int, default=1, help="tile sharding over N JAX devices (same output)")
    for name, hlp in (("gradient", "adjoint of the run's cost w.r.t. its control (adxx files)"),
                      ("grdchk", "gradient check at data.grdchk's points (output_adm-like output.txt)")):
        g = parsers[name] = sub.add_parser(name, help=hlp)
        g.add_argument("experiment_dir")
        g.add_argument("--variant", required=True, help="input_ad, input_ad.<v>, ...")
        g.add_argument("--out", required=True, help="new output directory (must not exist)")
        g.add_argument("--mode", choices=("run", "exact"), default="run",
                       help="run: the run's data.autodiff switches (TAF's adjoint); exact: no backward-only switch")
        g.add_argument("--lsr-derivative", choices=("run", "sweeps"), default=None,
                       help="SEAICE_LSR derivative: sweeps = A1 (default where the build does not tape the sweeps "
                            "itself, plan decisions 14, 17); run = as the build defines it")
        g.add_argument("--cg2d-derivative", choices=("run", "exact"), default=None,
                       help="CG2D derivative (default: the mode's; mitjax/ad/modes.py)")
        if name == "gradient":
            g.add_argument("--devices", type=int, default=1, help="tile sharding over N JAX devices")
            g.add_argument("--control", default=None, help="control file name (default data.grdchk's grdchkvarname)")
    c = sub.add_parser("compare", help="testreport's comparison with the experiment's results/")
    c.add_argument("output", help="model output (output.txt)")
    c.add_argument("experiment_dir")
    c.add_argument("--variant", required=True)
    c.add_argument("--kind", choices=("fwd", "adm"), default=None, help="default: from the variant")
    a = ap.parse_args(argv)
    from mitjax import api
    if a.cmd == "run":
        exp = cli_experiment(a.experiment_dir, a.variant)
        try:
            r = exp.run(a.out, devices=a.devices)
        except api.DeviceCountError as e:
            parsers["run"].error(f"DeviceCountError: {e}")
        print(r.output)
        return 0
    exp = api.load(a.experiment_dir, a.variant)
    if a.cmd == "gradient":
        try:
            g = exp.gradient(a.out, mode=a.mode, devices=a.devices, control=a.control,
                             lsr_derivative=a.lsr_derivative, cg2d_derivative=a.cg2d_derivative)
        except api.DeviceCountError as e:
            parsers["gradient"].error(f"DeviceCountError: {e}")
        print(f"fc = {g.fc:.15E}  control {g.control}  mode {g.mode}  devices {g.devices}")
        for p in g.files:
            print(p)
        return 0
    if a.cmd == "grdchk":
        chk = exp.grdchk(a.out, mode=a.mode, lsr_derivative=a.lsr_derivative, cg2d_derivative=a.cg2d_derivative)
        print("\n".join(chk.lines))
        print(chk.output)
        return 0
    from mitjax import testreport_jax as T
    rep = api.compare(a.output, exp.results(), kind=a.kind)
    for v in rep.run.variables:
        print(f"  {v.name:8s} {'--' if v.digits == 99 else v.digits:>3}")
    print(rep.summary)
    return T.EXIT[rep.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
