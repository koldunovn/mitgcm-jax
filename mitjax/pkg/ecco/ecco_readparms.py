"""ECCO_READPARMS   @63cdc0b pkg/ecco/ecco_readparms.F:3-945 and ECCO_SIZE.h (lane M4ADCOL session 3)

Host-side set-up of pkg/ecco as 1D_ocean_ice_column/code_ad compiles it: ECCO_OPTIONS.h is pkg/ecco's own (the
experiment has none): ALLOW_GENCOST_CONTRIBUTION and ALLOW_GENCOST3D defined (ECCO_OPTIONS.h:30, :32); ALLOW_GENCOST_1D,
ALLOW_SSH/SST/SEAICE_COST_CONTRIBUTION, ECCO_VARIABLE_AREAVOLGLOB, ALLOW_PSBAR_STERIC undefined (:40-67).
The routine runs only with useECCO (:202-210). Strings follow Fortran CHARACTER semantics: a comparison pads with
blanks (`_eq`), a substring `s(1:n)` is the blank-padded prefix (`_sub`).

Ported: the ecco_cost_nml defaults this build's executed code reads (using_cost_altim/seaice/transp/sst :217-237,
cost_iprec :473) with the namelist itself only empty or (lane M4ADLAB) setting cost_iprec alone (any other set key
raises: none of its old-style cost terms is ported;
the retired-parameter checks :507-575 then pass), the gencost defaults (:600-656), ecco_gencost_nml (:670), the derived
gencost_flag / using_gencost / is3d / pointer3d (:677-764), the mask loads (:766-826, raising when a mask is named) and
the checks (:828-928).
"""

import dataclasses

from mitjax.io.namelist import NULL

# pkg/ecco/ECCO_SIZE.h:9-23
NGENCOST = 30
NGENCOST3D = 6          # ALLOW_GENCOST3D defined (ECCO_OPTIONS.h:32)
NGENPPROC = 10
# eesupp/inc/EEPARAMS.h: precFloat32 = 32
precFloat32 = 32

GENCOST_KEYS = ("using_gencost", "gencost_barfile", "gencost_datafile", "gencost_name", "gencost_scalefile",
                "gencost_errfile", "gencost_itracer", "gencost_kLev_select", "gencost_preproc", "gencost_preproc_c",
                "gencost_preproc_i", "gencost_preproc_r", "gencost_posproc", "gencost_posproc_c", "gencost_posproc_i",
                "gencost_posproc_r", "gencost_outputlevel", "gencost_mask", "gencost_spmin", "gencost_spmax",
                "gencost_spzero", "gencost_wei1d", "gencost_avgperiod", "gencost_nrecperiod", "gencost_startdate1",
                "gencost_startdate2", "gencost_enddate1", "gencost_enddate2", "gencost_smooth2Ddiffnbt",
                "gencost_is1d", "gencost_is3d", "gencost_msk_is3d", "gencost_useDensityMask", "gencost_refPressure",
                "gencost_sigmaLow", "gencost_sigmaHigh", "gencost_tanhScale", "gencost_timevaryweight",
                "mult_gencost")                                               # :160-199 namelist /ecco_gencost_nml/


def _sub(s, n):
    """Fortran s(1:n) of a CHARACTER variable: the first n characters, blank-padded."""
    return s.ljust(n)[:n]


def _eq(a, b):
    """Fortran character .EQ.: the shorter operand is padded with blanks."""
    return a.rstrip(" ") == b.rstrip(" ")


@dataclasses.dataclass
class Gencost:
    """ECCO.h's gencost arrays at index k (Fortran names; the NGENPPROC arrays as lists jk = 1..NGENPPROC)."""
    k: int
    using_gencost: bool = False                    # :601
    gencost_flag: int = 0                          # :602
    gencost_avgperiod: str = "     "               # :603 (CHARACTER*(5))
    gencost_startdate1: int = 0                    # :604
    gencost_startdate2: int = 0                    # :605
    gencost_enddate1: int = 0                      # :606
    gencost_enddate2: int = 0                      # :607
    gencost_datafile: str = " "                    # :608
    gencost_name: str = "gencost"                  # :609
    gencost_preproc: list = None                   # :610-619 (' ', ' ', 0, 0. _d 0 per jk)
    gencost_preproc_c: list = None
    gencost_preproc_i: list = None
    gencost_preproc_r: list = None
    gencost_posproc: list = None
    gencost_posproc_c: list = None
    gencost_posproc_i: list = None
    gencost_posproc_r: list = None
    gencost_outputlevel: int = 0                   # :620
    gencost_errfile: str = " "                     # :621
    gencost_itracer: int = 1                       # :622
    gencost_kLev_select: int = 1                   # :623
    gencost_mask: str = " "                        # :624
    gencost_barfile: str = " "                     # :625
    gencost_spmin: float = 0.0                     # :626
    gencost_spmax: float = 0.0                     # :627
    gencost_spzero: float = 9876.0                 # :628
    gencost_wei1d: float = 0.0                     # :629
    mult_gencost: float = 1.0                      # :630
    gencost_is1d: bool = False                     # :631
    gencost_is3d: bool = False                     # :632
    gencost_useDensityMask: bool = False           # :633
    gencost_refPressure: float = 0.0               # :634
    gencost_sigmaLow: float = 0.0                  # :635
    gencost_sigmaHigh: float = 1.0e3               # :636
    gencost_tanhScale: float = 1.0e5               # :637
    gencost_pointer3d: int = 0                     # :638
    gencost_msk_is3d: bool = False                 # :639
    gencost_msk_pointer3d: int = 0                 # :640
    gencost_nrecperiod: int = 0                    # :652 (deprecated)
    gencost_scalefile: str = " "                   # :653
    gencost_smooth2Ddiffnbt: int = 0               # :654
    gencost_timevaryweight: bool = False           # :655
    # set by ECCO_COST_INIT_FIXED (ecco_cost_init_fixed.py)
    gencost_barskip: bool = False
    gencost_nrec: int = 0
    gencost_period: float = 0.0
    gencost_startdate: list = None
    gencost_enddate: list = None

    def __post_init__(self):
        for n, v in (("gencost_preproc", " "), ("gencost_preproc_c", " "), ("gencost_preproc_i", 0),
                     ("gencost_preproc_r", 0.0), ("gencost_posproc", " "), ("gencost_posproc_c", " "),
                     ("gencost_posproc_i", 0), ("gencost_posproc_r", 0.0)):
            if getattr(self, n) is None:
                setattr(self, n, [v] * NGENPPROC)


@dataclasses.dataclass
class EccoParams:
    """ECCO.h's scalars this build reads, and the gencost table (index k = 1..NGENCOST as gencost[k-1])."""
    using_cost_altim: bool
    using_cost_seaice: bool
    using_cost_transp: bool
    using_cost_sst: bool
    cost_iprec: int
    gencost: list


# the ECCO_OPTIONS.h settings the port follows (pkg/ecco/ECCO_OPTIONS.h, the experiment has none): defined / undefined
ECCO_DEFINED = ("ALLOW_GENCOST_CONTRIBUTION", "ALLOW_GENCOST3D")                          # :30, :32
ECCO_UNDEFINED = ("ALLOW_GENCOST_1D", "ALLOW_GENCOST_SSTV4_OUTPUT", "ECCO_VARIABLE_AREAVOLGLOB", "ALLOW_PSBAR_STERIC",
                  "ALLOW_IB_CORR", "ALLOW_SHALLOW_ALTIMETRY", "ALLOW_HIGHLAT_ALTIMETRY", "ALLOW_ECCO_OLD_FC_PRINT",
                  "ECCO_VERBOSE", "ALLOW_ECCO_DEBUG", "ALLOW_SSH_COST_CONTRIBUTION", "ALLOW_SST_COST_CONTRIBUTION",
                  "ALLOW_SEAICE_COST_CONTRIBUTION")                                       # :40-67


def check_ecco_options(cfg):
    """Raises unless the build's ECCO_OPTIONS.h is the one the port follows (ECCO_DEFINED / ECCO_UNDEFINED); an
    option no compiled source of the build tests (cpp_options.UnknownCppOption) has no effect and is skipped."""
    from mitjax.config.cpp_options import UnknownCppOption

    def on(n):
        try:
            return cfg.cpp.flag(n, "ECCO_OPTIONS.h")
        except UnknownCppOption:
            return None
    bad = [n for n in ECCO_DEFINED if on(n) is not True]
    bad += [n for n in ECCO_UNDEFINED if on(n)]
    if bad:
        raise NotImplementedError(f"pkg/ecco: ECCO_OPTIONS.h settings not ported: {bad}")


def _nml_values(run, group):
    return {k: v.value for (f, g, k), v in run.vars.items() if f == "data.ecco" and g == group
            and v.value is not NULL}


def ecco_readparms(run, cfg):
    """ECCO_READPARMS( myThid ) with useECCO: -> EccoParams. `run`: the experiment's RunNamelists."""
    if not (cfg.cpp.ALLOW_ECCO and cfg.use_flag("useECCO")):                # :202-210
        raise ValueError("ECCO_READPARMS: called without useECCO")
    check_ecco_options(cfg)
    # :217-237 (ALLOW_SSH_COST_CONTRIBUTION, ALLOW_SEAICE_COST_CONTRIBUTION, ALLOW_SST_COST_CONTRIBUTION undefined
    # in ECCO_OPTIONS.h:65-67; ALLOW_GENCOST_CONTRIBUTION defined)
    using_cost_altim = False                                                 # :222
    using_cost_seaice = False                                                # :227
    using_cost_transp = False                                                # :231
    using_cost_sst = False                                                   # :236
    cost_iprec = precFloat32                                                 # :473
    cost_nml = dict(_nml_values(run, "ecco_cost_nml"))                       # :500
    if "cost_iprec" in cost_nml:            # lane M4ADLAB session 3 (lab_sea/input_ad: cost_iprec = 64)
        cost_iprec = int(cost_nml.pop("cost_iprec"))
        if cost_iprec not in (32, 64):
            raise NotImplementedError(f"ECCO_READPARMS: cost_iprec = {cost_iprec}: the readers take 32 or 64")
    if cost_nml:
        raise NotImplementedError(f"ECCO_READPARMS: data.ecco ecco_cost_nml sets {sorted(cost_nml)}: not ported")
    gc = [Gencost(k) for k in range(1, NGENCOST + 1)]                        # :600-656 defaults
    nml = _nml_values(run, "ecco_gencost_nml")                               # :670
    keys = {k.lower(): k for k in GENCOST_KEYS}
    for key, val in nml.items():
        if key not in keys:
            raise ValueError(f"ECCO_READPARMS: unknown ecco_gencost_nml key {key}")
        name = keys[key]
        for idx, v in dict(val).items():
            if len(idx) == 1:
                setattr(gc[idx[0] - 1], name, v.ljust(5)[:5] if name == "gencost_avgperiod" else v)
            else:                                                            # (jk, k) of the NGENPPROC arrays
                getattr(gc[idx[1] - 1], name)[idx[0] - 1] = v
    gencost_k3d = 1                                                          # :677
    gencost_msk_k3d = 1                                                      # :678
    for g in gc:                                                             # :680-764
        bf, nm = g.gencost_barfile, g.gencost_name
        if nm.rstrip() in ("sshv4-mdt", "sshv4-tp", "sshv4-ers", "sshv4-gfo", "sshv4-lsc", "sshv4-gmsl",
                           "bpv4-grace", "sstv4-amsre", "sstv4-amsre-lsc"):      # :683-694
            g.gencost_flag, g.using_gencost = -1, True
        elif _eq(_sub(bf, 9), "m_boxmean") or _eq(_sub(bf, 9), "m_horflux"):   # :698-711
            raise NotImplementedError("ECCO_READPARMS: m_boxmean / m_horflux gencost (flag -3) not ported")
        elif _eq(_sub(nm, 6), "transp"):                                      # :714-721
            raise NotImplementedError("ECCO_READPARMS: transp gencost (flag -4) not ported")
        elif _eq(_sub(nm, 3), "moc"):                                         # :724-731
            raise NotImplementedError("ECCO_READPARMS: moc gencost (flag -5) not ported")
        elif nm.rstrip() in ("siv4-conc", "siv4-deconc", "siv4-exconc"):      # :734-739
            g.gencost_flag, g.using_gencost = 2, True
        elif not _eq(g.gencost_datafile, " "):                               # :741-743
            g.gencost_flag, g.using_gencost = 1, True
        if any(_eq(_sub(bf, n), p) for n, p in ((7, "m_theta"), (6, "m_salt"), (8, "m_diffkr"), (7, "m_kapgm"),
                                                 (9, "m_kapredi"), (7, "m_trVol"), (8, "m_trHeat"),
                                                 (8, "m_trSalt"))):            # :747-756
            g.gencost_is3d = True
        if g.gencost_is3d:                                                   # :759-762
            g.gencost_pointer3d = gencost_k3d
            gencost_k3d = gencost_k3d + 1
    for g in gc:                                                             # :766-826
        if not _eq(g.gencost_mask, " ") and g.gencost_flag in (-3, -4, -5):
            raise NotImplementedError("ECCO_READPARMS: gencost_mask files are not ported")
    for g in gc:                                                             # :828-928
        if not _eq(g.gencost_barfile, " "):                                  # :832-848
            bf = g.gencost_barfile.rstrip(" ")
            if not _eq(_sub(bf, 2), "m_"):
                bf = "m_" + bf
            if _eq(_sub(bf, 8), "m_tauZon"):
                bf = "m_ustress" + bf[8:]
            if _eq(_sub(bf, 8), "m_tauMer"):
                bf = "m_vstress" + bf[8:]
            g.gencost_barfile = bf
        if g.using_gencost:
            if g.gencost_flag >= 1:  # :852-901 (flag < -1 :904-926 raised above)
                if _eq(g.gencost_name, "gencost"):                           # :854-855
                    g.gencost_name = g.gencost_datafile
                if g.gencost_avgperiod.rstrip() not in ("day", "DAY", "month", "MONTH", "step", "STEP", "const",
                                                        "CONST", "year", "YEAR"):   # :857-876
                    raise RuntimeError(f"ERROR in ECCO_READPARMS: for gencost{g.k:2d}  {g.gencost_name.rstrip()}\n"
                                       "ECCO_READPARMS: gencost_avgperiod not properly set")
                if g.gencost_spmin == 0.0 and g.gencost_spmax == 0.0:        # :878-888
                    raise RuntimeError(f"ERROR in ECCO_READPARMS: for gencost{g.k:2d}  {g.gencost_name.rstrip()}\n"
                                       "ECCO_READPARMS: gencost_spmin, gencost_spmax not set")
                if g.gencost_spzero == 9876.0:                               # :890-899
                    raise RuntimeError(f"ERROR in ECCO_READPARMS: for gencost{g.k:2d}  {g.gencost_name.rstrip()}\n"
                                       "ECCO_READPARMS: gencost_spzero not set")
            elif g.gencost_flag == -1:                                       # :902-903
                pass
    return EccoParams(using_cost_altim, using_cost_seaice, using_cost_transp, using_cost_sst, cost_iprec, gc)
