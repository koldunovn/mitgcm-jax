"""COST_COPY_FILE   @63cdc0b pkg/cost/cost_copy_file.F:8-81 (host-side file copy; lane M4ADCOL)
and the host records of the package finals COST_FINAL calls (cost_final.F:86-122)."""

PKG_NAMES = ("profiles", "obsfit", "ecco", "ctrl", "obcs", "seaice", "shelfice")   # :44-51 DATA pkgNames


def cost_copy_file(rundir, optimcycle):
    """COST_COPY_FILE( oUnit, optimcycle, myThid ): for each package name (:54) whose costfunction_<pkg>.<optimcycle>
    exists (:57-61), the message `Reading cost function info from <file>` (:62-65) and every record of the file,
    trimmed to its last non-blank (at least 1) character (:69-73). Returns (records to write to oUnit, messages)."""
    copied, msgs = [], []
    for pkg in PKG_NAMES:
        cfname = f"costfunction_{pkg}.{optimcycle:04d}"                # :57-58 '(3A,I4.4)'
        f = rundir / cfname
        if f.exists():                                                 # :60-61 INQUIRE
            msgs.append(f"Reading cost function info from {cfname}")
            for rec in f.read_text().splitlines():                     # :69-73
                copied.append(rec.rstrip(" ") or " ")
    return copied, msgs


def package_final_records(pout, *, rundir, prefix):
    """The STDOUT records of ECCO_COST_FINAL, CTRL_COST_FINAL and SEAICE_COST_FINAL (cost_final.F:97, :102, :113)
    from their traced values `pout` (Model.cost_final's `out`), and their costfunction_<pkg>.0000 files (ifc .NE. -1:
    the master thread with costWriteCostFunction, cost_final.F:81-84; optimcycle 0)."""
    recs = []
    if "ecco" in pout:
        from mitjax.pkg.ecco.ecco_cost_final import ecco_cost_final_lines
        std, cf = ecco_cost_final_lines(pout["ecco"])
        recs += [prefix + ln for ln in std]
        recs.append(prefix + "Writing ecco cost function info to costfunction_ecco.0000")
        (rundir / "costfunction_ecco.0000").write_text("".join(ln + "\n" for ln in cf))
    if "ctrl" in pout and pout["ctrl"]["tim2d"] + pout["ctrl"].get("arr2d", []) + pout["ctrl"]["arr3d"]:
        from mitjax.pkg.ctrl.ctrl_cost_final import ctrl_cost_final_lines
        std, cf = ctrl_cost_final_lines(pout["ctrl"], **pout["ctrl_meta"])
        recs += [prefix + ln for ln in std]                            # ctrl_cost_final.F:137-138 PRINT_MESSAGE
        recs.append(prefix + "Writing generic ctrl cost function info to costfunction_ctrl.0000")   # :179-182
        (rundir / "costfunction_ctrl.0000").write_text("".join(ln + "\n" for ln in cf))
    if "seaice" in pout:
        from mitjax.pkg.seaice.seaice_cost_final import seaice_cost_final_lines
        std, cf = seaice_cost_final_lines(*pout["seaice"])
        recs.append(std)                                               # seaice_cost_final.F:79-80 (no PID prefix)
        (rundir / "costfunction_seaice.0000").write_text(cf + "\n")    # :85-93
    return recs
