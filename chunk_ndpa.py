"""
Chunk the Nigeria Data Protection Act, 2023 into searchable sections.

Source: a SCANNED PDF (no text layer) - extracted here with pytesseract
OCR (400 DPI, --psm 4, one page at a time to avoid memory issues; psm 4
assumes a single column of variable-sized text, which recognised section
numbers far more reliably than the default mode - the default mode
silently DROPPED several section numbers from the table of contents).

Same marginal-note problem as the ACJA
-----------------------------------------
Like the ACJA, this PDF prints each section's title as a catchword in the
margin, which OCR/extraction interleaves into the body text mid-sentence
(e.g. "...by automated means or Application\nnot." - "Application" is the
next section's marginal note, shoved between "or" and "not."). This
script uses the same fix as chunk_acja.py: read the Arrangement of
Sections (TOC) first to learn each section's title, then drop any body
line that matches it.

OCR caveat
-----------
Even with the better settings, OCR introduces small errors the TOC-title
matching won't catch inside ordinary prose (stray punctuation, the odd
misread character). Section numbers and structure are verified below;
word-for-word wording is not guaranteed the way it is for the Acts
extracted from a real text layer - treat precise wording as needing a
double-check.

Scope
------
Sections 1-66 (the Act itself). The Schedule (supplementary council
procedure) is a separate numbered list restarting at 1 and is excluded,
consistent with how the other Acts' schedules were handled.

Run:  python chunk_ndpa.py
"""

import json
import re

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

from config import DATA_DIR, MAX_CHUNK_CHARS

PDF_FILE = DATA_DIR / "ndpa.pdf"
RAW_TEXT_FILE = DATA_DIR / "ndpa.txt"
CHUNKS_FILE = DATA_DIR / "ndpa_chunks.jsonl"

DOCUMENT_TITLE = "Nigeria Data Protection Act, 2023 - OCR source (scanned PDF), see caveat"
MAX_SECTION_NUMBER = 66

PART_HEADING = re.compile(r"^PART\s+\S+\s*[-—]+\s*(.*)$")
# Tolerant of OCR misreading the period as a comma (seen for "11," "37,").
SECTION_START = re.compile(r"^(\d{1,2})[.,]\s*(.*)$")
BODY_START_MARKER = re.compile(r"^PART\s+I\s*[-—]+\s*OBJECTIVES AND APPLICATION$")

# Known OCR digit-misreads in the SOURCE scan: (section before it, number
# printed) -> the real number. Each is a single-digit misread (1->4, 0->9,
# 5->3) that content-matches the real section when checked against its
# known TOC title - e.g. body text headed "39." is actually section 30
# ("Sensitive personal data"), confirmed by its content.
OCR_TYPO_FIXES = {(11, 42): 12, (29, 39): 30, (57, 38): 58}


def extract_text():
    """OCR one page at a time (lower memory use than converting the whole
    PDF at once, which was killed for exceeding available memory)."""
    page_count = len(PdfReader(PDF_FILE).pages)
    with open(RAW_TEXT_FILE, "w", encoding="utf-8") as out:
        for page in range(1, page_count + 1):
            image = convert_from_path(PDF_FILE, dpi=400, first_page=page, last_page=page)[0]
            text = pytesseract.image_to_string(image, config="--psm 4")
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
    part_one_hits = [i for i, (_, l) in enumerate(lines) if BODY_START_MARKER.match(l)]
    if len(part_one_hits) < 2:
        raise ValueError("Couldn't find the second 'PART I - OBJECTIVES AND APPLICATION' - OCR/layout may differ.")

    end = next((i for i, (_, l) in enumerate(lines) if l.startswith("SCHEDULE")), None)
    if end is None:
        raise ValueError("Couldn't find the SCHEDULE boundary after the body.")
    return part_one_hits[1], end


def parse_toc_titles(lines):
    """TOC entries here are 'N. Title' on one line (unlike the ACJA, where
    the number and title were on separate lines)."""
    titles = {}
    for _, line in lines:
        if PART_HEADING.match(line):
            continue
        match = SECTION_START.match(line)
        if match:
            number = int(match.group(1))
            if 1 <= number <= MAX_SECTION_NUMBER:
                titles[number] = match.group(2).strip().rstrip(".:")
    return titles


def normalize(text):
    return re.sub(r"\s+", "", text).rstrip(".").lower()


def parse_sections(lines, titles):
    sections = []
    current = None
    current_title_norm = None
    last_number = None
    part = None
    note_buffer = []

    def flush_note_buffer_as_body():
        if current is not None:
            current["lines"].extend(note_buffer)
        note_buffer.clear()

    i = 0
    while i < len(lines):
        page, line = lines[i]

        part_match = PART_HEADING.match(line)
        if part_match:
            flush_note_buffer_as_body()
            part = part_match.group(0)
            i += 1
            continue

        section_match = SECTION_START.match(line)
        number = int(section_match.group(1)) if section_match else None

        # Repair a known OCR digit-misread before validating the sequence.
        if number is not None and (last_number, number) in OCR_TYPO_FIXES:
            number = OCR_TYPO_FIXES[(last_number, number)]

        valid_start = (
            section_match
            and number in titles
            and (last_number is None or last_number < number <= last_number + 3)
        )
        if valid_start:
            flush_note_buffer_as_body()
            rest = section_match.group(2)
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
            note_buffer.append(line)
            squashed = normalize("".join(note_buffer))
            if squashed == current_title_norm:
                note_buffer.clear()
            elif not current_title_norm.startswith(squashed):
                flush_note_buffer_as_body()
            current["page_end"] = page

        i += 1

    flush_note_buffer_as_body()
    return sections


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
    text = "\n".join(section["lines"])
    pieces = split_oversized_text(text, max_chars) if len(text) > max_chars else [text]

    return [
        {
            "id": f"ndpa-{section['label']}-{index}",
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
    if not (missing or duplicates):
        print(f"All {MAX_SECTION_NUMBER} sections present, in order.")
    print("\nNOTE: this source is a scanned PDF processed with OCR - structure is "
          "verified, exact wording is not (see the module docstring).")


def main():
    print(f"OCR'ing {PDF_FILE.name} (this takes a while - one page at a time)...")
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
