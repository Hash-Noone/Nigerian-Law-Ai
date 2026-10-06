"""
Chunk the Evidence Act into searchable sections.

Scope
------
Sections 1-230 (the Act itself). The PDF (E14.pdf) also bundles
"SUBSIDIARY LEGISLATION" after section 230 - a couple of 1990s Notices
naming specific government scientists authorised to sign drug-analysis
certificates. That's not general law (it's a personnel list), so it's
excluded here, the same way the Criminal Code's trailing Orders were.

Format
-------
Sections are grouped under PART headings (no Chapters, unlike the
Constitution or Criminal Code). A section is either:

    "230. Orders for production of prisoners"      (number + title, one line)
    "2.\\n\\nInterpretation"                          (number alone; title
                                                       follows as its own line)

Subsections are already written as "(1)", "(a)" etc, so - like the
Criminal Code - no marker conversion is needed here.

Run:  python chunk_evidence_act.py
"""

import json
import re
import subprocess

from config import DATA_DIR, MAX_CHUNK_CHARS

PDF_FILE = DATA_DIR / "evidence_act.pdf"
RAW_TEXT_FILE = DATA_DIR / "evidence_act.txt"
CHUNKS_FILE = DATA_DIR / "evidence_act_chunks.jsonl"

DOCUMENT_TITLE = "Evidence Act (Chapter E14 Laws of the Federation of Nigeria 2011)"
MAX_SECTION_NUMBER = 230

PART_HEADING = re.compile(r"^PART\s+([IVXLC]+)$")
SECTION_START = re.compile(r"^(\d{1,3})\.\s*(.*)$")


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
    """The body is the SECOND 'PART I' (the first is the Arrangement of
    Sections / table of contents), up to 'SUBSIDIARY LEGISLATION'."""
    part_one_hits = [i for i, (_, l) in enumerate(lines) if l == "PART I"]
    end_hits = [i for i, (_, l) in enumerate(lines) if l == "SUBSIDIARY LEGISLATION"]

    if len(part_one_hits) < 2 or not end_hits:
        raise ValueError(
            "Couldn't find the expected 'PART I' / 'SUBSIDIARY LEGISLATION' "
            "boundary markers - the PDF's layout may have changed."
        )
    return part_one_hits[1], end_hits[0]


def match_section_start(line, last_number):
    match = SECTION_START.match(line)
    if not match:
        return None
    number = int(match.group(1))
    rest = match.group(2)
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

        part_match = PART_HEADING.match(line)
        if part_match:
            # The Part's descriptive name is on the next one or two lines
            # (a heading, sometimes with a subheading line under it); take
            # just the first as context to keep this simple.
            part = f"Part {part_match.group(1)}"
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
            "id": f"evidence-act-{section['label']}-{index}",
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
