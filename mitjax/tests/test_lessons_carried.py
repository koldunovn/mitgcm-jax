"""docs/LESSONS_CARRIED.md, the merged lessons list: stable IDs, sources on every entry, and a mapping table that
resolves every section of the three lessons digests (plan Task 1).

`problems(text, sections)` lists every defect of the file's text:
- an entry is a line `- **L-<TOPIC>-<n> ...` plus its two-space-indented continuation lines; its ID must be
  well-formed (a known topic, n without leading zeros) and unique, and each topic is numbered 1..N without gaps,
  because an ID is retired (`- **L-XXX-n Retired.** ...`), never deleted or renumbered;
- every entry that is not retired has a `*Sources:*` part with at least one tag [E§n], [P§n] or [F§n], and every
  tag names an existing digest section;
- every ID mentioned anywhere in the file exists;
- the mapping table has exactly one row per `## n.` section of the digests, with the digest's heading text, and its
  IDs are exactly the entries whose sources cite that section (so every section resolves to at least one ID).
The digests are read from `mitjax.paths.LESSONS`; a missing digest is an error, not a skip. The negative controls
plant each defect into the real file's text and require the checker to name it.
"""

import re
from pathlib import Path

from mitjax import paths

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "LESSONS_CARRIED.md"
DIGESTS = {"E": "ecco_port.md", "P": "port2_kokkos.md", "F": "fesom_jax.md"}
TOPICS = ("PROC", "ORA", "LIT", "TOL", "XLA", "ARCH", "AD", "PAR", "PERF", "TEST", "ENV", "OPS", "USER", "COPY",
          "CONF")

ENTRY_START = re.compile(r"^- \*\*(L-[^\s*.]+)")
ID_RE = re.compile(r"^L-([A-Z]+)-([1-9][0-9]*)$")
MENTION = re.compile(r"\bL-[A-Z]+-[0-9]+\b")
TAG = re.compile(r"\[([EPF])§([0-9]+)\]")
SOURCES = "*Sources:*"
RETIRED = re.compile(r"^- \*\*L-[A-Z]+-[0-9]+ Retired\.\*\*")
MAPPING_HEADING = "## Mapping: digest sections to lesson IDs"
ROW = re.compile(r"^\| \[([EPF])§([0-9]+)\] \| (.*?) \| (.*?) \|$")
SECTION = re.compile(r"^## ([0-9]+)\. (.+?)\s*$", re.M)


def digest_sections(directory=None):
    """{(tag, n): heading title} of the `## n. title` sections of the three digests."""
    directory = Path(directory or paths.LESSONS)
    out = {}
    for tag, name in DIGESTS.items():
        found = {(tag, int(n)): title for n, title in SECTION.findall((directory / name).read_text())}
        if not found:
            raise ValueError(f"no '## n.' sections in {directory / name}")
        out.update(found)
    return out


def entries(text):
    """[(id, joined text of the entry)] in file order."""
    out, cur = [], None
    for line in text.splitlines():
        m = ENTRY_START.match(line)
        if m:
            cur = [m.group(1), [line]]
            out.append(cur)
        elif cur is not None and line.startswith("  ") and line.strip():
            cur[1].append(line.strip())
        else:
            cur = None
    return [(i, " ".join(block)) for i, block in out]


def source_tags(entry_text):
    """The (tag, n) pairs cited after `*Sources:*` (empty when the entry has no Sources part)."""
    _, sep, after = entry_text.partition(SOURCES)
    return {(t, int(n)) for t, n in TAG.findall(after)} if sep else set()


def mapping_rows(text):
    """[((tag, n), title, [ids])] of the table under the mapping heading."""
    _, sep, after = text.partition(MAPPING_HEADING)
    rows = []
    for line in after.splitlines() if sep else []:
        m = ROW.match(line)
        if m:
            ids = [s.strip() for s in m.group(4).split(",") if s.strip()]
            rows.append(((m.group(1), int(m.group(2))), m.group(3), ids))
    return rows


def expected_mapping(text):
    """{(tag, n): [ids in file order]} implied by the entries' sources."""
    out = {}
    for i, body in entries(text):
        for key in sorted(source_tags(body)):
            out.setdefault(key, []).append(i)
    return out


def problems(text, sections):
    probs = []
    ents = entries(text)
    ids = [i for i, _ in ents]
    seen, numbers = set(), {}
    for i in ids:
        m = ID_RE.match(i)
        if not m or m.group(1) not in TOPICS:
            probs.append(f"malformed ID {i}")
            continue
        if i in seen:
            probs.append(f"duplicate ID {i}")
        seen.add(i)
        numbers.setdefault(m.group(1), set()).add(int(m.group(2)))
    for topic, nums in numbers.items():
        gaps = sorted(set(range(1, max(nums) + 1)) - nums)
        if gaps:
            probs.append(f"gap in L-{topic}: numbers {gaps} missing (retire an ID, never delete it)")
    for i, body in ents:
        if RETIRED.match(body):
            continue
        tags = source_tags(body)
        if not tags:
            probs.append(f"{i} has no source tag after {SOURCES}")
        for key in sorted(tags - set(sections)):
            probs.append(f"{i} cites unknown section [{key[0]}§{key[1]}]")
    for mention in sorted(set(MENTION.findall(text)) - seen):
        probs.append(f"unknown ID {mention} mentioned")
    rows = mapping_rows(text)
    if not rows:
        probs.append("no mapping table")
    want = expected_mapping(text)
    in_table = set()
    for key, title, row_ids in rows:
        name = f"[{key[0]}§{key[1]}]"
        if key in in_table:
            probs.append(f"mapping row {name} appears twice")
        in_table.add(key)
        if key not in sections:
            probs.append(f"mapping row {name} names no digest section")
        elif title != sections[key]:
            probs.append(f"mapping row {name} title {title!r} != digest heading {sections[key]!r}")
        if not row_ids:
            probs.append(f"mapping row {name} has no ID")
        for i in row_ids:
            if i not in seen:
                probs.append(f"mapping row {name} lists unknown ID {i}")
        if row_ids != want.get(key, []):
            probs.append(f"mapping row {name} differs from the entries citing it: table {row_ids}, "
                         f"entries {want.get(key, [])}")
    for key in sorted(set(sections) - in_table):
        probs.append(f"digest section [{key[0]}§{key[1]}] {sections[key]!r} missing from the mapping table")
    return probs


def test_lessons_carried_well_formed():
    sections = digest_sections()
    assert {t for t, _ in sections} == set(DIGESTS)
    text = DOC.read_text()
    assert problems(text, sections) == []
    ents = entries(text)
    assert len(ents) > 100, len(ents)  # the parser sees the entries (a format drift would make it see none)
    assert all(row_ids for _, _, row_ids in mapping_rows(text))


def test_lessons_checker_negative_controls():
    """Each defect planted into the real text is reported; retiring an ID (and dropping it from its mapping rows)
    leaves the file clean."""
    sections = digest_sections()
    text = DOC.read_text()
    assert problems(text, sections) == []
    lines = text.splitlines()

    def line_starting(prefix):
        return next(line for line in lines if line.startswith(prefix))

    def reported(planted, phrase):
        assert planted != text, phrase
        probs = problems(planted, sections)
        assert any(phrase in p for p in probs), (phrase, probs)

    # duplicate IDs: an entry repeated, and an entry renumbered onto another's ID
    ad1 = line_starting("- **L-AD-1 ")
    reported(text.replace(ad1, ad1 + "\n" + ad1, 1), "duplicate ID L-AD-1")
    reported(text.replace("- **L-AD-2 ", "- **L-AD-1 ", 1), "duplicate ID L-AD-1")
    # malformed IDs: leading zero, unknown topic
    reported(text.replace("- **L-AD-7 ", "- **L-AD-07 ", 1), "malformed ID L-AD-07")
    reported(text.replace("- **L-AD-7 ", "- **L-XYZ-7 ", 1), "malformed ID L-XYZ-7")
    # an entry without sources, a source tag of a section that does not exist, a mention of an unknown ID
    reported(_strip_sources(text, "L-LIT-3"), "L-LIT-3 has no source tag")
    reported(text.replace("[P§3] P2M/feedback_mfct", "[P§12] P2M/feedback_mfct", 1),
             "L-LIT-3 cites unknown section [P§12]")
    reported(text + "\nSee [L-AD-99].\n", "unknown ID L-AD-99 mentioned")
    # an ID deleted instead of retired
    reported(_drop_entry(text, "L-USER-5"), "gap in L-USER")
    # mapping table: an unmapped digest section, an unknown ID, a row out of sync with the entries, a wrong title
    row_p5 = line_starting("| [P§5] |")
    reported(text.replace(row_p5 + "\n", "", 1), "digest section [P§5]")
    row_e1 = line_starting("| [E§1] |")
    first_id = ROW.match(row_e1).group(4).split(",")[0].strip()
    reported(text.replace(row_e1, row_e1.replace(f"| {first_id},", "| L-ORA-999,", 1), 1),
             "mapping row [E§1] lists unknown ID L-ORA-999")
    reported(text.replace(row_e1, row_e1.replace(f"| {first_id}, ", "| ", 1), 1), "mapping row [E§1] differs")
    reported(text.replace(row_e1, row_e1.replace("Process & agents", "Process and agents"), 1),
             "mapping row [E§1] title")

    # positive control: retiring the last ID of a topic keeps the file clean
    last_user = max((i for i, _ in entries(text) if i.startswith("L-USER-")), key=lambda i: int(i.split("-")[2]))
    retired = _retire(text, last_user)
    assert retired != text
    assert problems(retired, sections) == [], problems(retired, sections)


def _entry_lines(text, entry_id):
    """(start, end) line indices of an entry's block."""
    lines = text.splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith(f"- **{entry_id} "))
    end = start + 1
    while end < len(lines) and lines[end].startswith("  ") and lines[end].strip():
        end += 1
    return lines, start, end


def _strip_sources(text, entry_id):
    lines, start, end = _entry_lines(text, entry_id)
    block = "\n".join(lines[start:end])
    block = block[:block.index(SOURCES)].rstrip()
    return "\n".join(lines[:start] + block.splitlines() + lines[end:]) + "\n"


def _drop_entry(text, entry_id):
    lines, start, end = _entry_lines(text, entry_id)
    return "\n".join(lines[:start] + lines[end:]) + "\n"


def _retire(text, entry_id):
    """Replace an entry by its retired form and drop it from every mapping row."""
    lines, start, end = _entry_lines(text, entry_id)
    lines = lines[:start] + [f"- **{entry_id} Retired.** 2026-10-01, negative-control test."] + lines[end:]
    out = []
    for line in lines:
        if ROW.match(line):
            line = line.replace(f", {entry_id} |", " |").replace(f", {entry_id},", ",").replace(f"| {entry_id}, ", "| ")
        out.append(line)
    return "\n".join(out) + "\n"
