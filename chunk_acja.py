"""
Chunk the Administration of Criminal Justice Act (ACJA) 2015 into
searchable sections.

PDF quirk this script works around: marginal notes
-----------------------------------------------------
This PDF is laid out in two visual columns per page: the main body text,
and a short catchword in the right margin repeating each section's title
(e.g. "Purpose.", "Arrest generally."). pdftotext extracts both columns
as one linear stream, so the catchword ends up interleaved MID-SENTENCE
in the body text - e.g. the real text is:

    "...administration of criminal justice in Nigeria promotes..."

but the extracted text reads:

    "...administration of criminal              (marginal note:)
     justice in Nigeria promotes..."             "Purpose."

Naively concatenating lines would corrupt the law by injecting "Purpose."
into the middle of a sentence. To avoid that, this script first reads the
Arrangement of Sections (the table of contents) to learn each section's
official title, then DROPS any line in the body that exactly matches the
current section's title - that line is the marginal note, not law text.

Scope
------
Sections 1-495 (the Act itself). The PDF also has a "FIRST SCHEDULE" of
court forms after section 495 (arrest warrants, bail forms, etc.) -
templates, not substantive rules, so excluded here.

Run:  python chunk_acja.py
"""

import json
import re
import subprocess

from config import DATA_DIR, MAX_CHUNK_CHARS

PDF_FILE = DATA_DIR / "acja.pdf"
RAW_TEXT_FILE = DATA_DIR / "acja.txt"
CHUNKS_FILE = DATA_DIR / "acja_chunks.jsonl"

DOCUMENT_TITLE = "Administration of Criminal Justice Act, 2015 (ACJA)"
MAX_SECTION_NUMBER = 495

PART_HEADING = re.compile(r"^PART\s*(\d+)\s*[-–]\s*(.*)$")
SECTION_WITH_PERIOD = re.compile(r"^(\d{1,3})\.\s*(.*)$")
# The trailing period is sometimes lost in extraction (e.g. "210" instead
# of "210."), but a bare number is also how ordinary prose can start a
# wrapped line ("...punishable under section 336\nof this Act."), so this
# is accepted ONLY when the number stands entirely alone on its line -
# see the strict check in match_section_start() below.
BARE_SECTION_NUMBER = re.compile(r"^(\d{1,3})$")
BODY_START_MARKER = re.compile(r"^PART\s*1\s*[-–]\s*PRELIMINARY$")


def extract_text():
    from pypdf import PdfReader
    page_count = len(PdfReader(PDF_FILE).pages)
    with open(RAW_TEXT_FILE, "w", encoding="utf-8") as out:
        for page in range(1, page_count + 1):
            text = subprocess.run(
                ["pdftotext", "-f", str(page), "-l", str(page), str(PDF_FILE), "-"],
                capture_output=True, text=True, check=True,
            ).stdout
            out.write(f"\n--- PAGE {page} ---\n")
            out.write(text)
    return page_count


def read_lines(raw_text):
    parts = re.split(r"\n--- PAGE (\d+) ---\n", raw_text)
    lines = []
    for page_number, page_text in zip(parts[1::2], parts[2::2]):
        for line in page_text.split("\n"):
            line = line.strip()
            if line:
                lines.append((int(page_number), line))
    return lines


def find_body_range(lines):
    """The Arrangement of Sections (TOC) uses the same 'PART 1 - PRELIMINARY'
    heading as the real body - take the SECOND match as the body start.
    Body ends at 'FIRST SCHEDULE' (a set of court forms, not substantive law)."""
    part_one_hits = [i for i, (_, l) in enumerate(lines) if BODY_START_MARKER.match(l)]
    if len(part_one_hits) < 2:
        raise ValueError("Couldn't find the second 'PART 1 - PRELIMINARY' - PDF layout may differ.")

    end = next((i for i, (_, l) in enumerate(lines) if l == "FIRST SCHEDULE"), None)
    if end is None:
        raise ValueError("Couldn't find the FIRST SCHEDULE boundary after the body.")
    return part_one_hits[1], end


def parse_toc_titles(lines):
    """Build {section_number: title} from the table of contents. Entries
    are a bare number line, then the title line (a PART heading may sit
    between them, which is skipped)."""
    titles = {}
    i = 0
    while i < len(lines):
        _, line = lines[i]
        if re.match(r"^\d{1,3}$", line):
            number = int(line)
            j = i + 1
            while j < len(lines) and PART_HEADING.match(lines[j][1]):
                j += 1
            if j < len(lines):
                titles[number] = lines[j][1]
            i = j + 1
        else:
            i += 1
    return titles


def normalize(text):
    """Collapse whitespace and drop a trailing period, so marginal-note
    matching isn't thrown off by cosmetic differences between the TOC's
    copy of a title and the body's (e.g. one has a period, the other doesn't)."""
    return re.sub(r"\s+", "", text).rstrip(".").lower()


def match_section_start(line, last_number, titles):
    """Returns (number, rest) if `line` validly starts a new section, else None."""
    match = SECTION_WITH_PERIOD.match(line)
    if match:
        number, rest = int(match.group(1)), match.group(2)
        if number in titles and (last_number is None or last_number < number <= last_number + 5):
            return number, rest
        return None

    match = BARE_SECTION_NUMBER.match(line)
    if match:
        number = int(match.group(1))
        # No period AND nothing else on the line: only accept the exact
        # next number in sequence - strict, because a bare number can also
        # be ordinary prose wrapping onto its own line (e.g. "...under
        # section 336\nof this Act." - but there rest="of this Act." is
        # non-empty, so it never reaches this branch in the first place).
        if last_number is not None and number == last_number + 1 and number in titles:
            return number, ""

    return None


def parse_sections(lines, titles):
    sections = []
    current = None
    current_title_norm = None
    last_number = None
    part = None
    note_buffer = []  # lines tentatively collected as a possible marginal note

    def flush_note_buffer_as_body():
        """The buffered lines didn't complete the title after all - they're
        real body text, so hand them to the current section untouched."""
        if current is not None:
            current["lines"].extend(note_buffer)
        note_buffer.clear()

    i = 0
    while i < len(lines):
        page, line = lines[i]

        part_match = PART_HEADING.match(line)
        if part_match:
            flush_note_buffer_as_body()
            part = f"Part {part_match.group(1)}: {part_match.group(2)}"
            i += 1
            continue

        start = match_section_start(line, last_number, titles)
        if start:
            flush_note_buffer_as_body()
            number, rest = start
            last_number = number
            current_title_norm = normalize(titles[number])
            current = {
                "label": str(number),
                "context": part,
                "page_start": page,
                "page_end": page,
                "lines": [rest] if rest and normalize(rest) != current_title_norm else [],
            }
            sections.append(current)
            i += 1
            continue

        if current is not None:
            # The marginal-note catchword can be split across several short
            # lines (e.g. "Central" / "Criminal" / "Records Registry."), and
            # can repeat at every page break within a long section. Hold
            # each line in a buffer; once the buffer's squashed-together
            # text matches the section's title, discard the whole buffer as
            # a marginal note. If it stops looking like a prefix of the
            # title, the buffer was real body text all along - keep it.
            note_buffer.append(line)
            squashed = normalize("".join(note_buffer))
            if squashed == current_title_norm:
                note_buffer.clear()  # it was the marginal note - drop it
            elif not current_title_norm.startswith(squashed):
                flush_note_buffer_as_body()
            current["page_end"] = page

        i += 1

    flush_note_buffer_as_body()
    return sections


def finalize_section(section):
    return "\n".join(section["lines"])


def split_oversized_text(text, max_chars):
    pieces, buffer = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars) or max_chars
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        candidate = f"{buffer} {sentence}".strip()
        if buffer and len(candidate) > max_chars:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)
    return pieces


def section_to_chunks(section, title, max_chars=MAX_CHUNK_CHARS):
    text = finalize_section(section)
    pieces = split_oversized_text(text, max_chars) if len(text) > max_chars else [text]

    return [
        {
            "id": f"acja-{section['label']}-{index}",
            "document": DOCUMENT_TITLE,
            "reference": f"Section {section['label']}",
            "context": f"{section['context']} - {title}" if section["context"] else title,
            "topics": [],
            "page_start": section["page_start"],
            "page_end": section["page_end"],
            "chunk_index": index,
            "chunk_count": len(pieces),
            "text": f"{title}\n{piece}".strip() if title else piece,
        }
        for index, piece in enumerate(pieces)
    ]


def print_report(sections, chunks, titles):
    found = [int(s["label"]) for s in sections]
    missing = [n for n in range(1, MAX_SECTION_NUMBER + 1) if n not in found]
    duplicates = sorted({n for n in found if found.count(n) > 1})

    print(f"Sections found : {len(found)} (expected {MAX_SECTION_NUMBER})")
    print(f"TOC titles     : {len(titles)}")
    print(f"Chunks created : {len(chunks)} (longest {max(len(c['text']) for c in chunks)} chars)")
    if missing:
        print(f"\nWARNING - sections NOT found: {missing}")
    if duplicates:
        print(f"WARNING - duplicated section labels: {duplicates}")

    # Sanity check: any chunk whose text still contains a bare repeat of its
    # own title mid-text suggests the marginal-note filter missed a case.
    leaks = [c["id"] for c in chunks if c["text"].count(c["context"].split(" - ")[-1].rstrip(".")) > 1]
    if leaks:
        print(f"NOTE - possible un-filtered marginal notes in: {leaks[:5]} (of {len(leaks)})")

    if not (missing or duplicates):
        print(f"All {MAX_SECTION_NUMBER} sections present, in order.")


def main():
    print(f"Extracting text from {PDF_FILE.name} via pdftotext...")
    page_count = extract_text()
    print(f"Extracted {page_count} pages to {RAW_TEXT_FILE}")

    all_lines = read_lines(RAW_TEXT_FILE.read_text(encoding="utf-8"))
    start, end = find_body_range(all_lines)
    print(f"Body is lines {start}-{end} (pages {all_lines[start][0]}-{all_lines[end][0]}).")

    titles = parse_toc_titles(all_lines[:start])
    sections = parse_sections(all_lines[start:end], titles)
    chunks = [c for s in sections for c in section_to_chunks(s, titles.get(int(s["label"]), ""))]

    with open(CHUNKS_FILE, "w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print_report(sections, chunks, titles)
    print(f"\nSaved to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
