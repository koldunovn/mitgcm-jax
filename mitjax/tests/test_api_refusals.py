"""Refusals added by lane API session 2 (docs plan 20261006 Task 6 follow-up): settings the drivers cannot reproduce
raise instead of running something else. Each case is a copy of a verification experiment with one namelist edit;
each test also plants the removal of its guard (monkeypatch to a no-op) and measures that the run then goes on
silently -- the negative control: without the guard the refusal test fails.

* config: a variant directory that does not exist (mitjax.config.params.check_input_dir; testreport runs only
  existing input directories, testreport:1709, and linkdata skips a missing one, testreport:814). Planted: the
  loader returns a configuration read entirely from the base `input`.
* first guess: data.optim optimcycle = 1, or data.ctrl doInitXX = .FALSE. (1D_ocean_ice_column/input_ad). The Fortran
  reads xx_<name>.<optimcycle> then (ctrl_init_ctrlvar.F:140-151 writes zeros only with doInitXX .AND.
  optimcycle.EQ.0); the drivers' first guess is zero (Model._ctrl_first_guess_zero). Planted: the Model is built
  with the zero first guess.
* EOS: selectP_inEOS_Zc = 1 (1D_ocean_ice_column/input, eosType 'JMD95Z'): SET_REF_STATE calls FIND_HYD_PRESS_1D
  (set_ref_state.F:178-188), not ported (ini_parms.set_ref_state_hyd_press_1d). Planted: the Model is built with
  selectP_inEOS_Zc = 1 and pRef4EOS of rhoConst (no hydrostatic balance with rhoRef). Values 0, 2, 3 pass.
Costs (CPU node): four Model set-ups of the 1D column (~1-2 min each), one config load.
"""

import shutil

import numpy as np
import pytest

from mitjax import paths

VER = paths.UPSTREAM / "verification"


def _out(tag):
    from mitjax.tests import advect_gate as ag
    return ag.out_dir(f"api-refuse-{tag}")


def _copy_column(dirs, edit_file, old, new, tag):
    """<new dir>/1D_ocean_ice_column with `dirs` copied (symlinks dereferenced) and one edit of `edit_file` (a path
    relative to the experiment: the text `old`, which must occur once, replaced by `new`); its input_ad/prepare_run
    links ../../isomip/input_ad/ones_64b.bin, so that file is copied next to it."""
    root = _out(tag)
    dst = root / "1D_ocean_ice_column"
    dst.mkdir(parents=True)
    for d in dirs:
        shutil.copytree(VER / "1D_ocean_ice_column" / d, dst / d, symlinks=False)
    iso = root / "isomip" / "input_ad"
    iso.mkdir(parents=True)
    shutil.copy2(VER / "isomip" / "input_ad" / "ones_64b.bin", iso / "ones_64b.bin")
    f = dst / edit_file
    text = f.read_text()
    assert text.count(old) == 1, (f, old)
    f.write_text(text.replace(old, new))
    return dst


def _release():
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


# ------------------------------------------------------------------------------------------------------- config

def test_config_refuses_missing_variant(monkeypatch):
    from mitjax.config import params as P
    exp = VER / "tutorial_barotropic_gyre"
    assert not (exp / "input.nope").exists() and (exp / "input").is_dir()
    with pytest.raises(FileNotFoundError, match=r"no input dir .*tutorial_barotropic_gyre/input\.nope"):
        P.load("tutorial_barotropic_gyre", "input.nope", exp_dir=exp)
    # negative control: without the guard the loader reads the base input's files and returns a configuration
    monkeypatch.setattr(P, "check_input_dir", lambda exp_dir, input_dir: None)
    e = P.load("tutorial_barotropic_gyre", "input.nope", exp_dir=exp)
    dirs = {d for d, _ in e.run.files.values()}
    print("planted: input.nope loaded from", sorted(dirs))
    assert dirs == {"input"}, dirs


# -------------------------------------------------------------------------------------------------- first guess

FIRST_GUESS_EDITS = {
    "optimcycle1": ("input_ad/data.optim", " optimcycle=0,", " optimcycle=1,"),
    "doInitXX_false": ("input_ad/data.ctrl", " &CTRL_NML\n", " &CTRL_NML\n doInitXX = .FALSE.,\n"),
}


@pytest.mark.parametrize("case", sorted(FIRST_GUESS_EDITS))
def test_model_refuses_nonzero_first_guess(case, monkeypatch):
    import mitjax
    from mitjax.drivers.model import Model
    f, old, new = FIRST_GUESS_EDITS[case]
    dst = _copy_column(("code_ad", "input", "input_ad"), f, old, new, f"fg-{case}")
    exp = mitjax.load(dst, variant="input_ad")
    with pytest.raises(NotImplementedError, match=r"CTRL_INIT_CTRLVAR.*ctrl_init_ctrlvar\.F:140-151"):
        exp.model(_out(f"fg-{case}-run"))
    _release()
    # negative control: without the guard the Model is built on the zero first guess (what the Fortran would not do)
    monkeypatch.setattr(Model, "_ctrl_first_guess_zero", lambda self: None)
    m = exp.model(_out(f"fg-{case}-planted"))
    xx0 = {k: float(np.max(np.abs(np.asarray(v.data)))) for k, v in m.genarr_xx0.items()}
    print("planted:", case, "built with first guess max|xx| =", xx0)
    assert xx0 and all(v == 0.0 for v in xx0.values()), xx0
    _release()


# ---------------------------------------------------------------------------------------------------------- EOS

def test_model_refuses_selectP_inEOS_Zc_1(monkeypatch):
    import mitjax
    from mitjax.model.src import ini_parms
    for sel in (0, 2, 3):                                     # not refused (no pRef4EOS read, or no call at all)
        assert ini_parms.set_ref_state_hyd_press_1d(sel) is None
    dst = _copy_column(("code", "input"), "input/data", " eosType='JMD95Z',\n",
                       " eosType='JMD95Z',\n selectP_inEOS_Zc=1,\n", "eos")
    exp = mitjax.load(dst, variant="input")
    with pytest.raises(NotImplementedError, match=r"FIND_HYD_PRESS_1D.*set_ref_state\.F:178-188"):
        exp.model(_out("eos-run"))
    _release()
    # negative control: without the guard the Model is built with selectP_inEOS_Zc = 1 on the unbalanced pRef4EOS
    monkeypatch.setattr(ini_parms, "set_ref_state_hyd_press_1d", lambda sel: None)
    m = exp.model(_out("eos-planted"))
    print("planted: selectP_inEOS_Zc =", m.params.selectP_inEOS_Zc)
    assert m.params.selectP_inEOS_Zc == 1
    _release()
