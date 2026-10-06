"""
Chunk the Labour Act into searchable sections.

IMPORTANT - source quality caveat
-----------------------------------
Unlike the Criminal Code and Evidence Act PDFs, this one (L1.pdf) is a
SCANNED document run through OCR (its PDF producer is "Adobe Acrobat
Paper Capture Plug-in"), and the OCR has real character-level errors in
the prose: "tenns" for "terms", "Rilles" for "Rules", "tbe" for "the",
"Rigbt" for "Right", and at least one paragraph marker misread entirely
("(f)" came out as "ifJ"). This script gets the STRUCTURE right (section
numbers, titles, boundaries - all 92 sections are verified present), but
it can't fix OCR errors inside the body text itself, since it has no way
to know what the original characters were. Treat the wording of any
answer sourced from this Act as needing a double-check against a clean
copy, and prefer sourcing a non-scanned Labour Act PDF if precise wording
ever matters (e.g. from lawsofnigeria.placng.org, the source used for the
Criminal Code, which had no such issues).

Scope
------
Sections 1-92 (the Act itself). The PDF also bundles subsidiary Dock
Labour regulations after the Schedule, restarting their own numbering -
excluded here, as with the other Acts' trailing subsidiary legislation.

Run:  python chunk_labour_act.py
"""

import json
import re
import subprocess

from config import DATA_DIR, MAX_CHUNK_CHARS

PDF_FILE = DATA_DIR / "labour_act.pdf"
RAW_TEXT_FILE = DATA_DIR / "labour_act.txt"
CHUNKS_FILE = DATA_DIR / "labour_act_chunks.jsonl"

DOCUMENT_TITLE = "Labour Act (Chapter L1 Laws of the Federation of Nigeria 2004) - OCR source, see caveat"
MAX_SECTION_NUMBER = 92

# Tolerant of OCR misreads of Roman numerals (I/l/1 look-alikes), e.g. "PART Ill".
PART_HEADING = re.compile(r"^PART\s+[IVXil1]{1,5}$")
# OCR sometimes inserts a stray space inside the section number ("7 4." for
# "74.") or misreads the period as a comma ("87,"). Both are handled below.
SECTION_START = re.compile(r"^(\d)\s?(\d{0,2})[.,]\s*(.*)$")


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
    """Body starts where '1.' and its title share ONE line ('1. Manner of
    payment') - in the table of contents the number and title are always on
    separate lines ('1.' / blank / 'Manner of payment.', with a period).
    Body ends at the first SCHEDULE heading after section 92."""
    start_hits = [i for i, (_, l) in enumerate(lines) if l == "1. Manner of payment"]
    if not start_hits:
        raise ValueError("Couldn't find the 'Manner of payment' body heading - PDF layout may differ.")
    start = start_hits[0]

    end = next((i for i in range(start, len(lines)) if lines[i][1] == "SCHEDULE"), None)
    if end is None:
        raise ValueError("Couldn't find the SCHEDULE boundary after the body.")
    return start, end


def match_section_start(line, last_number):
    match = SECTION_START.match(line)
    if not match:
        return None
    number = int(match.group(1) + match.group(2))
    rest = match.group(3)
    valid = number == 1 if last_number is None else last_number < number <= last_number + 3
    return (number, rest) if valid else None


def parse_sections(lines):
    sections = []
    current = None
    last_number = None
    part = None

    i = 0
    while i < len(lines):
        page, line = lines[i]

        if PART_HEADING.match(line):
            part = line
            if i + 1 < len(lines):
                part += f": {lines[i + 1][1]}"
            i += 2
            continue

        start = match_section_start(line, last_number)
        if start:
            number, rest = start
            last_number = number
            current = {
                "label": str(number),
                "context": part,
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
    lines = section["lines"]
    if not lines:
        return None, ""
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
            "id": f"labour-act-{section['label']}-{index}",
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
    found = [int(s["label"]) for s in sections]
    missing = [n for n in range(1, MAX_SECTION_NUMBER + 1) if n not in found]
    duplicates = sorted({n for n in found if found.count(n) > 1})

    print(f"Sections found : {len(found)} (expected {MAX_SECTION_NUMBER})")
    print(f"Chunks created : {len(chunks)} (longest {max(len(c['text']) for c in chunks)} chars)")
    if missing:
        print(f"\nWARNING - sections NOT found: {missing}")
    if duplicates:
        print(f"WARNING - duplicated section labels: {duplicates}")
    if not (missing or duplicates):
        print(f"All {MAX_SECTION_NUMBER} sections present, in order.")
    print("\nNOTE: this source is OCR'd and contains known character-level errors "
          "in the prose (see the module docstring) - structure is verified, wording is not.")


def main():
    print(f"Extracting text from {PDF_FILE.name} via pdftotext...")
    page_count = extract_text()
    print(f"Extracted {page_count} pages to {RAW_TEXT_FILE}")

    lines = read_lines(RAW_TEXT_FILE.read_text(encoding="utf-8"))
    start, end = find_body_range(lines)
    print(f"Body is lines {start}-{end} (pages {lines[start][0]}-{lines[end][0]}).")

    sections = parse_sections(lines[start:end])
    chunks = [c for s in sections for c in section_to_chunks(s)]

    with open(CHUNKS_FILE, "w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print_report(sections, chunks)
    print(f"\nSaved to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
