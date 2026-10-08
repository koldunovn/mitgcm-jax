"""The public documentation pages stay true (docs plan 20261006 Task 10).

* every relative link and `#anchor` in docs/*.md, README.md and notebooks/README.md resolves (the target file exists;
  the anchor is a heading of the target, slugged as GitHub slugs it);
* the public pages hold no machine path, user name, batch job ID, host name, assistant marker or session link;
* every `python -m mitjax ...` command line the public pages show parses with the command's own argument parser
  (parsing only: nothing runs).

Negative controls, each planted in a copy of the text: a link to a missing file, a link to a missing anchor, one
sample of every forbidden class, and a command line with a misspelt option, each must be caught.
Stdlib + the mitjax CLI modules; no model run, no $MJX_REFERENCE; seconds.
"""

import argparse
import re
import shlex

import pytest

from mitjax import paths

REPO = paths.REPO

LINKED = sorted(REPO.glob("docs/*.md")) + [REPO / "README.md", REPO / "notebooks" / "README.md"]
PUBLIC = [REPO / p for p in (
    "README.md", "notebooks/README.md", "docs/install.md", "docs/running.md", "docs/gradients.md",
    "docs/configurations.md", "docs/parallel.md", "docs/checking_a_change.md", "docs/troubleshooting.md",
    "docs/status.md", "docs/fortran_map.md", "docs/READING_GUIDE.md")]

# ------------------------------------------------------------------------------------------------- links

FENCE = re.compile(r"^```.*?^```", re.S | re.M)
INLINE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)


def _prose(text):
    """The text without fenced code blocks and inline code (links there are not links)."""
    return INLINE.sub("", FENCE.sub("", text))


def slug(heading):
    """GitHub's anchor of a heading: lower case, markup and punctuation dropped, spaces to hyphens."""
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)          # a link keeps its text
    h = h.replace("`", "").lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(text):
    out, seen = set(), {}
    for _, h in HEADING.findall(FENCE.sub("", text)):
        s = slug(h)
        n = seen.get(s, 0)
        out.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    return out


def bad_links(path, text):
    bad = []
    for target in LINK.findall(_prose(text)):
        if re.match(r"^[a-z]+:", target):                           # http:, https:, mailto:
            continue
        file, _, anchor = target.partition("#")
        dest = (path.parent / file).resolve() if file else path
        if not dest.exists():
            bad.append(f"{target}: no file {dest.relative_to(REPO) if dest.is_relative_to(REPO) else dest}")
            continue
        if anchor and dest.suffix == ".md":
            if anchor not in anchors(dest.read_text(encoding="utf-8")):
                bad.append(f"{target}: no heading #{anchor} in {dest.name}")
    return bad


@pytest.mark.parametrize("path", LINKED, ids=lambda p: str(p.relative_to(REPO)))
def test_links_resolve(path):
    bad = bad_links(path, path.read_text(encoding="utf-8"))
    assert not bad, f"{path.relative_to(REPO)}: broken links {bad}"


def test_control_broken_links_are_caught():
    page = REPO / "docs" / "running.md"
    text = page.read_text(encoding="utf-8")
    assert not bad_links(page, text)
    planted = text + "\nSee [nowhere](no_such_page.md) and [gone](gradients.md#no-such-heading).\n"
    bad = bad_links(page, planted)
    assert len(bad) == 2 and "no file" in bad[0] and "no heading" in bad[1], bad


def test_slug_examples():
    assert slug("MAX/MIN at build-dependent statements") == "maxmin-at-build-dependent-statements"
    assert slug("Your own `.F` files") == "your-own-f-files"
    assert slug("Supported options (generated)") == "supported-options-generated"


# ------------------------------------------------------------------------------------------------- forbidden text

# Each literal that the public-tree scan (tools/make_public_tree.py SCAN) also forbids carries a one-letter character
# class, and the samples below are split into two adjacent strings, so that this file does not hold what it forbids;
# the patterns match exactly what they matched before.
FORBIDDEN = {
    "machine path": re.compile(r"/work/ab09[9]5|/home/[a-z]/|/scr[a]tch/[a-z]/|/sw/spack"),
    "user or project account": re.compile(r"a27[0]088|ab0995"),
    "batch job ID": re.compile(r"\bjob ?\d{6,}|\b27[89]\d{5}\b"),
    "host name": re.compile(r"levante|\bl\d{5}\b|dkrz\.de", re.I),
    "assistant marker": re.compile(r"c[l]aude|co-[a]uthored-by|generated [w]ith|anthropic", re.I),
    "session link": re.compile(r"c[l]aude\.ai|/session[s_/]|session_[0-9a-z]{6,}", re.I),
}


def forbidden_hits(text):
    return sorted({name for name, rx in FORBIDDEN.items() if rx.search(text)})


@pytest.mark.parametrize("path", PUBLIC, ids=lambda p: str(p.relative_to(REPO)))
def test_public_page_has_no_private_text(path):
    hits = {name: rx.findall(path.read_text(encoding="utf-8"))[:3] for name, rx in FORBIDDEN.items()
            if rx.search(path.read_text(encoding="utf-8"))}
    assert not hits, f"{path.relative_to(REPO)}: {hits}"


@pytest.mark.parametrize("sample, kind", [
    ("the clone at /work/ab09" "95/x/MITgcm", "machine path"),
    ("run by a27" "0088", "user or project account"),
    ("measured in job 27944829", "batch job ID"),
    ("measured in 27944829", "batch job ID"),
    ("on Levante", "host name"),
    ("node l40123", "host name"),
    ("Co-Auth" "ored-By: someone", "assistant marker"),
    ("Generated " "with a tool", "assistant marker"),
    ("https://cl" "aude.ai/code/session_01AbCdEf", "session link"),
])
def test_control_forbidden_samples_are_caught(sample, kind):
    clean = (REPO / "docs" / "install.md").read_text(encoding="utf-8")
    assert not forbidden_hits(clean)
    assert kind in forbidden_hits(clean + "\n" + sample + "\n")


# ------------------------------------------------------------------------------------------------- command lines

CMD = re.compile(r"python -m (mitjax(?:\.testreport_jax)?)\b(.*)")


def command_lines(text):
    """`python -m mitjax ...` lines of fenced code blocks (continuations joined, comments dropped) and inline code
    spans; a mention of a command (fewer than two words after it) and usage templates with `[`, `<` or `...` are skipped."""
    out = []
    for block in FENCE.findall(text):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            m = CMD.search(line.split("  #")[0])
            if m:
                out.append((m.group(1), m.group(2)))
    for span in re.findall(r"`(python -m mitjax[^`]*)`", FENCE.sub("", text)):
        m = CMD.search(" ".join(span.split()))
        if m and len(m.group(2).split()) >= 2 and not re.search(r"[\[<]|\.\.\.", m.group(2)):
            out.append((m.group(1), m.group(2)))
    return out


class _Parsed(Exception):
    pass


def parse(module, args, monkeypatch):
    """Parse `args` with the module's own parser; return the namespace (raise SystemExit on a parse error)."""
    orig = argparse.ArgumentParser.parse_args

    def parse_only(self, argv=None, namespace=None):
        ns = orig(self, argv, namespace)
        raise _Parsed(ns)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_only)
    if module == "mitjax":
        from mitjax.__main__ import main
    else:
        from mitjax.testreport_jax import main
    try:
        main(shlex.split(args))
    except _Parsed as p:
        return p.args[0]
    raise AssertionError(f"{module} {args}: main returned without parsing")


def _all_commands():
    out = []
    for path in PUBLIC:
        for module, args in command_lines(path.read_text(encoding="utf-8")):
            out.append(pytest.param(module, args, id=f"{path.name}:{module}{args[:40]}"))
    return out


def test_pages_show_commands():
    assert len(_all_commands()) >= 8


@pytest.mark.parametrize("module, args", _all_commands())
def test_command_line_parses(module, args, monkeypatch):
    parse(module, args, monkeypatch)


def test_control_bad_command_line_fails(monkeypatch):
    ok = parse("mitjax", "run EXP --variant input --out runs/x", monkeypatch)
    assert ok.cmd == "run"
    with pytest.raises(SystemExit):
        parse("mitjax", "run EXP --variant input --outt runs/x", monkeypatch)
    with pytest.raises(SystemExit):
        parse("mitjax", "gradient EXP --variant input_ad --out d --mode approximate", monkeypatch)
