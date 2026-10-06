"""
Chunk the Criminal Code Act into searchable sections.

Scope
------
The uploaded PDF (C38.pdf, Chapter 77 Laws of the Federation) is actually
THREE things bound together:

  1. The "Criminal Code Act" itself - a short outer Act (about 9 sections)
     that simply enacts the Code below as its Schedule.
  2. "The Criminal Code" - the Schedule containing the actual criminal
     offences, sections 1-521 (this is what "s.327 of the Criminal Code"
     always refers to - the offence sections have their OWN numbering,
     separate from the outer Act's).
  3. A handful of 1950s "Orders" appended at the very end (banned Cold-War-
     era publications, an unlawful-societies list) - obsolete, not
     substantive criminal law.

This script chunks ONLY #2 - the 521 substantive offence sections - which
is the part any real question ("is X a crime", "what's the penalty for Y")
needs. If you also want the outer Act's procedural sections or the old
Orders, they'd need a small extension to the page range below.

PDF quirk this script works around
------------------------------------
pypdf mis-extracts this particular PDF (it splits words mid-token, e.g.
"Th\\ne Criminal Code"), so extraction here uses the `pdftotext` command
line tool instead, which renders it cleanly. There is no repeating page
header/footer to strip, unlike the Constitution PDF.

Run:  python chunk_criminal_code.py
"""

import json
import re
import subprocess

from config import DATA_DIR, MAX_CHUNK_CHARS

PDF_FILE = DATA_DIR / "criminal_code.pdf"
RAW_TEXT_FILE = DATA_DIR / "criminal_code.txt"
CHUNKS_FILE = DATA_DIR / "criminal_code_chunks.jsonl"

DOCUMENT_TITLE = "Criminal Code Act (Chapter 77 Laws of the Federation of Nigeria 2004) - the Criminal Code"
MAX_SECTION_NUMBER = 521

# The Code's substantive sections run from this page to just before the
# obsolete Orders (found by inspecting the extracted text - see the
# docstring above). If a future edition of this PDF has different pagination,
# these two markers - not the page numbers - are what actually matter, so
# regenerating from a different copy of the PDF should still work.
BODY_START_MARKER = "CHAPTER 1"   # the SECOND "CHAPTER 1" in the file (the Schedule's, not the TOC's)
BODY_END_MARKER = "SCHEDULE"      # the THIRD "SCHEDULE" (the Unlawful Societies list that follows the Code)

CHAPTER_HEADING = re.compile(r"^CHAPTER\s+(\d+)$")
PART_HEADING = re.compile(r"^PART\s+(\d+)$")
# "37. Treason" / "327 A. Offence of infanticide" / "329A. Unlawful..." /
# "426.(Repealed by ...)" - number, optional letter suffix (with or without
# a space before it), then the rest of the line (title, or for a repealed
# section, its whole one-line text).
SECTION_START = re.compile(r"^(\d{1,3})\s*([A-Z]{1,2})?\.\s*(.*)$")


def extract_text():
    """pdftotext handles this PDF far more cleanly than pypdf (see docstring)."""
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
    """Return [(page_number, line), ...] with blank lines removed.

    Two PDF quirks handled here:
    - A section sometimes starts mid-line, right after the previous
      section's last sentence, with no line break in between (e.g.
      "...excused by law. 307. When\\na child becomes..."). The regex
      below inserts a line break before any "NNN. Capitalized word" that
      follows a sentence-ending period, so it becomes its own line.
    - A few sections are prefixed with a footnote asterisk ("*453."),
      stripped per-line below.
    """
    parts = re.split(r"\n--- PAGE (\d+) ---\n", raw_text)
    lines = []
    for page_number, page_text in zip(parts[1::2], parts[2::2]):
        page_text = re.sub(
            r"(?<=\.)\s+(?=\d{1,3}\s?[A-Z]{0,2}\.\s+[A-Z])", "\n", page_text
        )
        for line in page_text.split("\n"):
            line = line.strip().lstrip("*").strip()
            if line:
                lines.append((int(page_number), line))
    return lines


def find_body_range(lines):
    """The Schedule's body is the text between the SECOND 'CHAPTER 1' and
    the THIRD 'SCHEDULE' (see the marker constants above)."""
    chapter_hits = [i for i, (_, l) in enumerate(lines) if l == BODY_START_MARKER]
    schedule_hits = [i for i, (_, l) in enumerate(lines) if l == BODY_END_MARKER]

    if len(chapter_hits) < 2 or len(schedule_hits) < 3:
        raise ValueError(
            "Couldn't find the expected boundary markers - the PDF's layout "
            "may differ from the copy this script was written against. "
            f"Found {len(chapter_hits)} 'CHAPTER 1' and {len(schedule_hits)} "
            "'SCHEDULE' headings (expected at least 2 and 3)."
        )
    return chapter_hits[1], schedule_hits[2]


def match_section_start(line, last_number):
    match = SECTION_START.match(line)
    if not match:
        return None

    number = int(match.group(1))
    suffix = match.group(2) or ""
    rest = match.group(3)

    if last_number is None:
        valid = number == 1
    else:
        last, last_suffix = last_number
        # Either the next whole number in sequence, or a lettered insertion
        # right after the same base number (e.g. 327 -> "327 A").
        valid = (last < number <= last + 5) or (number == last and suffix > last_suffix)

    return (number, suffix, rest) if valid else None


def parse_sections(lines):
    sections = []
    current = None
    last_number = None
    chapter = chapter_title = None

    i = 0
    while i < len(lines):
        page, line = lines[i]

        chapter_match = CHAPTER_HEADING.match(line)
        if chapter_match:
            chapter = f"Chapter {chapter_match.group(1)}"
            chapter_title = lines[i + 1][1] if i + 1 < len(lines) else None
            i += 2
            continue

        if PART_HEADING.match(line):
            i += 2  # the Part's name is on the next line
            continue

        start = match_section_start(line, last_number)
        if start:
            number, suffix, rest = start
            last_number = (number, suffix)
            current = {
                "label": f"{number}{suffix}",
                "context": f"{chapter}: {chapter_title}" if chapter else None,
                "page_start": page,
                "page_end": page,
                "lines": [rest] if rest else [],
            }
            sections.append(current)
        elif current is not None:
            current["lines"].append(line)
            current["page_end"] = page

        i += 1

    return sections


def finalize_section(section):
    """The first line becomes the title, UNLESS the section had no title of
    its own (a repealed section, where everything is on one line)."""
    lines = section["lines"]
    if not lines:
        return None, ""
    if len(lines) == 1 and lines[0].startswith("("):
        return None, lines[0]  # e.g. "(Repealed by Ordinance No. 20 of 1955)."
    return lines[0], "\n".join(lines[1:])


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


def section_to_chunks(section, max_chars=MAX_CHUNK_CHARS):
    title, text = finalize_section(section)
    pieces = split_oversized_text(text, max_chars) if len(text) > max_chars else [text]

    return [
        {
            "id": f"criminal-code-{section['label']}-{index}",
            "document": DOCUMENT_TITLE,
            "reference": f"Section {section['label']}",
            "context": section["context"],
            "topics": [],
            "page_start": section["page_start"],
            "page_end": section["page_end"],
            "chunk_index": index,
            "chunk_count": len(pieces),
            "text": f"{title}\n{piece}".strip() if title else piece,
        }
        for index, piece in enumerate(pieces)
    ]


def print_report(sections, chunks):
    found = [s["label"] for s in sections]
    expected = [str(n) for n in range(1, MAX_SECTION_NUMBER + 1)]
    missing = [n for n in expected if n not in found]
    duplicates = sorted({n for n in found if found.count(n) > 1})

    print(f"Sections found : {len(found)} (expected {len(expected)} base numbers, "
          f"plus any lettered insertions like 327A)")
    print(f"Chunks created : {len(chunks)} (longest {max(len(c['text']) for c in chunks)} chars)")
    if missing:
        print(f"\nWARNING - base section numbers NOT found: {missing}")
    if duplicates:
        print(f"WARNING - duplicated section labels: {duplicates}")
    if not (missing or duplicates):
        print("All 521 base sections present, in order.")


def main():
    print(f"Extracting text from {PDF_FILE.name} via pdftotext...")
    page_count = extract_text()
    print(f"Extracted {page_count} pages to {RAW_TEXT_FILE}")

    lines = read_lines(RAW_TEXT_FILE.read_text(encoding="utf-8"))
    start, end = find_body_range(lines)
    print(f"Body (the Criminal Code Schedule) is lines {start}-{end} "
          f"(pages {lines[start][0]}-{lines[end][0]}).")

    sections = parse_sections(lines[start:end])
    chunks = [c for s in sections for c in section_to_chunks(s)]

    with open(CHUNKS_FILE, "w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print_report(sections, chunks)
    print(f"\nSaved to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
