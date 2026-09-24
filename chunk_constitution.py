"""
Step 2: raw text -> searchable chunks (one JSON record per line).

Written for the constituteproject.org edition of the Constitution
("Nigeria's Constitution of 1999 with Amendments through 2011").

What this PDF looks like (and how we deal with it)
--------------------------------------------------
* Every page starts with 4 header lines ("constituteproject.org", "PDF
  generated...", "Page n", "Nigeria 1999 (rev. 2011)")     -> dropped.
* Constitute adds topic tags ("Right to life", "Regional group(s)") that are
  NOT part of the law. Each tag is a line containing the DEL character
  (\\x7f) followed by the tag text                        -> removed from the
  legal text and kept separately as a `topics` list (only used to help search).
* A section is either
      "33.  "            (number alone, subsections follow), or
      "10. The Govern..."  (number + text, when it has no subsections).
  Subsections are printed "1.", "2." and items "a.", "b.", "i.", "ii.";
  we rewrite these markers as "(1)", "(a)", "(i)" as in the official text.
* Items are separated by whitespace-only lines; long items are wrapped over
  several lines, which we join back together.
* After section 320 come the Schedules, which we keep too (Exclusive
  Legislative List, Code of Conduct, oaths ...).

Legal wording is never rewritten: only whitespace and list markers change.

Run:  python chunk_constitution.py
"""

import json
import re

from config import (
    CHUNKS_FILE,
    DOCUMENT_TITLE,
    MAX_CHUNK_CHARS,
    MAX_SECTION_NUMBER,
    RAW_TEXT_FILE,
)

# --------------------------------------------------------------- patterns
PAGE_MARKER = re.compile(r"^--- PAGE (\d+) ---\s*$", re.MULTILINE)

# The 4 header lines printed at the top of every page.
PAGE_HEADER = re.compile(
    r"^(constituteproject\.org|PDF generated:.*|Page \d+|Nigeria 1999 \(rev\. 2011\))$"
)

TAG_MARKER = "\x7f"  # DEL character that precedes every Constitute topic tag

CHAPTER_HEADING = re.compile(r"^Chapter\s+([IVXLCDM]+):\s*(.*)$")           # "Chapter IV: Fundamental Rights"
PART_HEADING = re.compile(r"^Part\s+([IVXLCDM]+|\d+):\s*(.*)$")             # "Part II: State Courts"
GROUP_HEADING = re.compile(r"^([A-Z]{1,2})\.\s+([A-Z].*)$")                   # "A. The Supreme Court of Nigeria"
SECTION_LINE = re.compile(r"^(\d{1,3})([A-Z]?)\.(?:\s+(.*))?$")             # "36." / "10. The Government..."
LIST_MARKER = re.compile(r"^(\d+[A-Z]?|[a-z]|[ivx]{2,})\.\s+(?=\S)")        # "1." "2A." "a." "ii."
TOP_LEVEL_NUMBER = re.compile(r"^\((\d+)\)")                                # "(3) ..."

SCHEDULE_HEADING = re.compile(r"^Schedule\s+([IVXLC]+)(?::\s*(.*))?$")      # "Schedule II: Legislative Powers"
SCHEDULE_PART_HEADING = re.compile(r"^Part\s+([IVXLC]+|\d+)(?::\s*(.*))?$")  # "Part I: Exclusive Legislative List"
SCHEDULE_ITEM = re.compile(r"^\d+\.\s")                                      # "12. Control of capital issues."

# A section number may jump forward by at most this much (covers a stray
# missing number). Anything else is treated as ordinary text.
MAX_SKIP = 10

# Sections inserted by later amendments carry a letter (254A ... 254F).
# Everything else should be exactly 1..320.
EXPECTED_SECTIONS = [str(n) for n in range(1, MAX_SECTION_NUMBER + 1)]
EXPECTED_SECTIONS[EXPECTED_SECTIONS.index("254") + 1:EXPECTED_SECTIONS.index("254") + 1] = [
    f"254{letter}" for letter in "ABCDEF"
]


# --------------------------------------------------------- reading tokens
def clean_line(line):
    """Whitespace-only cleanup (also turns non-breaking spaces into spaces)."""
    return re.sub(r"\s+", " ", line.replace("\xa0", " ")).strip()


def read_tokens(raw_text):
    """
    Turn constitution.txt into a flat list of tokens:

        (page, "text", line)   a line of law text
        (page, "sep",  None)   a blank separator line (= paragraph break)
        (page, "tag",  label)  a Constitute topic tag (not law text)
    """
    parts = PAGE_MARKER.split(raw_text)  # [preface, "1", text1, "2", text2, ...]

    tokens = []
    for page_number, page_text in zip(parts[1::2], parts[2::2]):
        page = int(page_number)
        lines = page_text.split("\n")

        i = 0
        while i < len(lines):
            raw = lines[i]
            line = clean_line(raw.replace(TAG_MARKER, ""))

            if TAG_MARKER in raw:
                # DEL marker: the tag text is on the next line.
                if i + 1 < len(lines) and clean_line(lines[i + 1]):
                    tokens.append((page, "tag", clean_line(lines[i + 1])))
                i += 2
                continue

            if PAGE_HEADER.match(line):
                pass
            elif line:
                tokens.append((page, "text", line))
            elif raw:
                # Whitespace-only line (e.g. " \xa0") separates items.
                # Truly empty lines are layout noise and are ignored.
                tokens.append((page, "sep", None))

            i += 1

    return tokens


def find_body_range(tokens):
    """
    start = the preamble ("We the people of the Federal Republic of Nigeria").
            Everything before it is the Table of Contents.
    end   = the standalone "Schedule I" heading where the Schedules begin.
    """
    start = next(
        (i for i, (_, kind, text) in enumerate(tokens)
         if kind == "text" and text.lower().startswith("we the people of the federal republic")),
        None,
    )
    if start is None:
        raise ValueError("Could not find the preamble; is constitution.txt the right file?")

    end = next(
        (i for i in range(start, len(tokens))
         if tokens[i][1] == "text" and tokens[i][2] == "Schedule I"),
        len(tokens),
    )
    return start, end


# ------------------------------------------------------- text assembly
def join_wrapped_lines(lines):
    """
    Re-join a paragraph that was wrapped over several lines.
    A line ending in a hyphen ("Auditor-") is joined WITHOUT a space so the
    compound word survives ("Auditor-General").
    """
    text = ""
    for line in lines:
        if not text:
            text = line
        elif text.endswith("-") and len(text) > 1 and text[-2] != " ":
            text += line
        else:
            text += " " + line
    return text


def convert_marker(text):
    """'1. Every person...' -> '(1) Every person...', 'a. ...' -> '(a) ...'"""
    match = LIST_MARKER.match(text)
    if match:
        return f"({match.group(1)}) " + text[match.end():]
    return text


# ------------------------------------------------------ section detection
def match_section_start(line, last_label, last_subsection, after_separator):
    """
    If `line` starts a new section return (number, suffix, rest_of_line).

    "36."                  -> a section whose subsections follow
    "10. The Government.." -> a one-paragraph section

    Guards against mistaking a subsection ("5. Save as otherwise...") for a
    section: the number must come shortly AFTER the previous section number,
    and a line that continues the subsection count is a subsection.
    """
    match = SECTION_LINE.match(line)
    if not match:
        return None

    number = int(match.group(1))
    suffix = match.group(2)
    rest = (match.group(3) or "").strip()

    if last_label is None:
        valid = number == 1 and not suffix
    else:
        last_number, last_suffix = last_label
        if suffix:
            # Inserted sections such as 254A: same number, later letter,
            # and always printed with the number alone on its line.
            valid = not rest and number == last_number and suffix > last_suffix
        else:
            valid = last_number < number <= last_number + MAX_SKIP

    if not valid:
        return None

    # "5. text" straight after a separator, when the current section's last
    # subsection was 4, is subsection (5), not section 5.
    if rest and after_separator and number == last_subsection + 1:
        return None

    return number, suffix, rest


# ------------------------------------------------------------ body parser
def parse_body(tokens):
    """Group the body tokens into sections, tracking Chapter / Part / group."""
    sections = []
    heading = {"chapter": None, "part": None, "group": None}
    heading_target = None   # which heading is still wrapping onto the next line

    current = None          # the section being built
    paragraph = None        # lines of the paragraph being built: [(page, line)]
    last_label = None       # (number, suffix) of the previous section
    last_subsection = 0     # last top-level "(n)" seen in the current section
    after_separator = False

    def flush_paragraph():
        """Finish the paragraph being built and add it to the current section."""
        nonlocal paragraph, last_subsection
        if paragraph and current is not None:
            text = convert_marker(join_wrapped_lines([line for _, line in paragraph]))
            current["paragraphs"].append({
                "page_start": paragraph[0][0],
                "page_end": paragraph[-1][0],
                "text": text,
            })
            # Remember the subsection number, e.g. "(3)". Letter-suffixed
            # subsections such as "(2A)" don't count towards the sequence.
            top = TOP_LEVEL_NUMBER.match(text)
            if top:
                last_subsection = int(top.group(1))
        paragraph = None

    def context():
        return " > ".join(part for part in (heading["chapter"], heading["part"], heading["group"]) if part)

    for page, kind, value in tokens:

        if kind == "tag":
            if current is not None and value not in current["topics"]:
                current["topics"].append(value)
            continue

        if kind == "sep":
            flush_paragraph()
            heading_target = None
            after_separator = True
            continue

        line = value

        # --- New section?
        start = match_section_start(line, last_label, last_subsection, after_separator)
        if start:
            flush_paragraph()
            number, suffix, rest = start
            last_label = (number, suffix)
            last_subsection = 0
            heading_target = None
            current = {
                "label": f"{number}{suffix}",
                "context": context(),
                "topics": [],
                "paragraphs": [],
            }
            sections.append(current)
            if rest:
                paragraph = [(page, rest)]
            after_separator = False
            continue

        # --- Chapter / Part / group headings (kept out of section text).
        chapter = CHAPTER_HEADING.match(line)
        part = PART_HEADING.match(line)
        group = GROUP_HEADING.match(line)
        if chapter or part or group:
            flush_paragraph()
            if chapter:
                heading.update(chapter=f"Chapter {chapter.group(1)}: {chapter.group(2)}", part=None, group=None)
                heading_target = "chapter"
            elif part:
                heading.update(part=f"Part {part.group(1)}: {part.group(2)}", group=None)
                heading_target = "part"
            else:
                heading.update(group=f"{group.group(1)}. {group.group(2)}")
                heading_target = "group"
            after_separator = False
            continue

        # --- A long heading wraps onto the next line(s): keep gluing them on.
        if heading_target:
            heading[heading_target] += " " + line
            continue

        # --- Ordinary text (ignored until section 1 has started).
        if current is not None:
            if paragraph is None:
                paragraph = [(page, line)]
            else:
                paragraph.append((page, line))
        after_separator = False

    flush_paragraph()
    return sections


# ------------------------------------------------------- schedule parser
def parse_schedules(tokens):
    """
    Group the Schedules by (Schedule, Part). No section logic is needed:
    we only track headings and split items on "12." style markers.
    """
    groups = []
    current = None
    paragraph = None
    schedule = schedule_title = None

    def flush_paragraph():
        nonlocal paragraph
        if paragraph and current is not None:
            current["paragraphs"].append({
                "page_start": paragraph[0][0],
                "page_end": paragraph[-1][0],
                "text": join_wrapped_lines([line for _, line in paragraph]),
            })
        paragraph = None

    def new_group(reference, title):
        nonlocal current
        flush_paragraph()
        current = {"label": reference, "context": title, "topics": [], "paragraphs": []}
        groups.append(current)

    for page, kind, value in tokens:
        if kind == "tag":
            continue
        if kind == "sep":
            flush_paragraph()
            continue

        heading = SCHEDULE_HEADING.match(value)
        part = SCHEDULE_PART_HEADING.match(value)

        if heading:
            schedule, schedule_title = heading.group(1), (heading.group(2) or "").strip()
            new_group(f"Schedule {schedule}", schedule_title)
        elif part and schedule:
            part_title = (part.group(2) or "").strip()
            context = " - ".join(x for x in (schedule_title, part_title) if x)
            new_group(f"Schedule {schedule}, Part {part.group(1)}", context)
        elif current is not None:
            if paragraph is not None and SCHEDULE_ITEM.match(value):
                flush_paragraph()  # a new numbered item starts
            if paragraph is None:
                paragraph = [(page, value)]
            else:
                paragraph.append((page, value))

    flush_paragraph()
    return [g for g in groups if g["paragraphs"]]


# ------------------------------------------------------------- chunking
def split_oversized_text(text, max_chars):
    """Split one very long paragraph at sentence ends, then at spaces."""
    pieces = []
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if sentence:
            pieces.append(sentence)

    # Re-pack the small pieces so we don't create needlessly tiny chunks.
    packed, buffer = [], ""
    for piece in pieces:
        if buffer and len(buffer) + 1 + len(piece) > max_chars:
            packed.append(buffer)
            buffer = piece
        else:
            buffer = f"{buffer} {piece}".strip()
    if buffer:
        packed.append(buffer)
    return packed


def group_to_chunks(group, kind, max_chars=MAX_CHUNK_CHARS):
    """Pack a section's (or schedule part's) paragraphs into chunks <= max_chars."""
    # 1. Break any paragraph larger than the limit.
    items = []
    for paragraph in group["paragraphs"]:
        if len(paragraph["text"]) <= max_chars:
            items.append(paragraph)
        else:
            for piece in split_oversized_text(paragraph["text"], max_chars):
                items.append({**paragraph, "text": piece})

    # 2. Pack items in order until the limit is reached.
    packed, buffer, size = [], [], 0
    for item in items:
        length = len(item["text"]) + 1
        if buffer and size + length > max_chars:
            packed.append(buffer)
            buffer, size = [], 0
        buffer.append(item)
        size += length
    if buffer:
        packed.append(buffer)

    # 3. Turn each packed group into a record.
    slug = re.sub(r"[^a-z0-9]+", "-", group["label"].lower()).strip("-")
    reference = f"Section {group['label']}" if kind == "section" else group["label"]

    return [
        {
            "id": f"constitution-{slug}-{index}",
            "document": DOCUMENT_TITLE,
            "kind": kind,
            "reference": reference,
            "context": group["context"],
            "topics": group["topics"],
            "page_start": min(i["page_start"] for i in chunk),
            "page_end": max(i["page_end"] for i in chunk),
            "chunk_index": index,
            "chunk_count": len(packed),
            "text": "\n".join(i["text"] for i in chunk),
        }
        for index, chunk in enumerate(packed)
    ]


# ------------------------------------------------------------ reporting
def subsection_problems(sections):
    """Sections whose subsections don't run (1), (2), (3) ... in order."""
    problems = []
    for section in sections:
        numbers = []
        for paragraph in section["paragraphs"]:
            match = TOP_LEVEL_NUMBER.match(paragraph["text"])
            if match:
                numbers.append(int(match.group(1)))
        if numbers and numbers != list(range(1, len(numbers) + 1)):
            problems.append((section["label"], numbers))
    return problems


def print_report(sections, schedules, chunks, tags_seen):
    found = [s["label"] for s in sections]
    missing = [label for label in EXPECTED_SECTIONS if label not in found]
    unexpected = [label for label in found if label not in EXPECTED_SECTIONS]
    duplicates = sorted({label for label in found if found.count(label) > 1})

    print(f"Sections found  : {len(found)} (expected {len(EXPECTED_SECTIONS)})")
    print(f"Schedule parts  : {len(schedules)}")
    print(f"Chunks created  : {len(chunks)} (longest {max(len(c['text']) for c in chunks)} chars)")
    print(f"Topic tags kept : {tags_seen} (removed from the law text)")

    if missing:
        print(f"\nWARNING - sections NOT found: {missing}")
    if unexpected:
        print(f"WARNING - unexpected section labels: {unexpected}")
    if duplicates:
        print(f"WARNING - duplicated section labels: {duplicates}")

    problems = subsection_problems(sections)
    if problems:
        print(f"\nCheck these sections (subsection numbering looks off): {len(problems)}")
        for label, numbers in problems[:15]:
            print(f"    section {label}: {numbers}")
    if not (missing or unexpected or duplicates or problems):
        print("\nAll sections present, in order, with consecutive subsections.")


# ----------------------------------------------------------------- main
def main():
    raw_text = RAW_TEXT_FILE.read_text(encoding="utf-8")

    tokens = read_tokens(raw_text)
    start, end = find_body_range(tokens)
    print(f"Read {len(tokens)} tokens; body is tokens {start}-{end}.")

    sections = parse_body(tokens[start:end])
    schedules = parse_schedules(tokens[end:])

    chunks = [c for s in sections for c in group_to_chunks(s, "section")]
    chunks += [c for g in schedules for c in group_to_chunks(g, "schedule")]

    with open(CHUNKS_FILE, "w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    tags_seen = sum(len(s["topics"]) for s in sections)
    print_report(sections, schedules, chunks, tags_seen)
    print(f"\nSaved to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
