"""
Chunk the Nigeria Police Act, 2020 into searchable sections.

Source: a SCANNED PDF (no text layer), laid out in THREE columns per page -
a narrow section-number column on the far left, the body paragraph in the
middle, and a marginal-note catchword column on the far right. Plain
pytesseract text extraction reads these as separate blocks (all the page's
numbers first, then all the body text, then notes), which loses the
association between each number and its paragraph entirely - worse than
the "interleaved mid-sentence" problem in the ACJA and NDPA.

How this script handles it
-----------------------------
For each page, pytesseract's word-level bounding boxes (`image_to_data`)
are used to reconstruct the real layout:
  - words in the left ~7% of the page width that look like a bare number
    ("12.", "3,") are collected as (y-position, number);
  - words in the middle (up to ~85% width) are grouped into lines and
    sorted by y-position - this reconstructs the body paragraph in correct
    reading order, with the marginal-note column (the right ~15%, where
    these words never appear) excluded entirely;
  - each number is then spliced back in just before the body line at the
    closest matching y-position.

This splicing is approximate and occasionally misplaces a number (e.g.
onto a mid-paragraph line rather than the true section start). That's
handled the same way as the ACJA's and NDPA's OCR quirks: every candidate
section start is cross-checked against the Arrangement of Sections (TOC)
and the expected next-number sequence before being accepted, so a
misplaced number is simply rejected and left as ordinary body text rather
than creating a false section.

OCR caveat
-----------
As with the Labour Act and NDPA: structure (which sections exist, their
order) is verified below; exact wording is not guaranteed the way it is
for the Acts with a real text layer.

Scope
------
Sections 1-142 (the Act itself). The Schedule (the Police hierarchy of
ranks) and the final bill-passage certification pages are excluded.

Run:  python chunk_police_act.py     (slow - OCRs 57 pages one at a time)
"""

import json
import re
from collections import defaultdict

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
from pytesseract import Output

from config import DATA_DIR, MAX_CHUNK_CHARS

PDF_FILE = DATA_DIR / "police_act.pdf"
RAW_TEXT_FILE = DATA_DIR / "police_act.txt"
CHUNKS_FILE = DATA_DIR / "police_act_chunks.jsonl"

DOCUMENT_TITLE = "Nigeria Police Act, 2020 - OCR source (scanned PDF), see caveat"
MAX_SECTION_NUMBER = 142
OCR_DPI = 250

NUMBER_ZONE_FRACTION = 0.09  # left edge where the section-number column lives
MARGIN_ZONE_FRACTION = 0.85  # everything right of this is the marginal-note column
# "1." is sometimes OCR'd as "L." or "l." (letter L, lowercase L) when
# cropped to just this narrow column - tolerated here since the section's
# own number sequence validation will reject this if it is actually wrong.
NUMBER_PATTERN = re.compile(r"^([lLI]|\d{1,3})[.,]?$")
VERTICAL_TOLERANCE = 120     # pixels (at OCR_DPI) for matching a number to its line

PART_HEADING = re.compile(r"^PART\s+\S+\s*[-–—]+\s*(.*)$")
SECTION_START = re.compile(r"^(\d{1,3})[.,]\s*(.*)$")
BODY_START_MARKER = re.compile(r"^PART\s+I\s*[-–—]+\s*PRELIMINARY$")

# Known OCR digit-misreads: (section before it, number printed) -> real
# number, confirmed by checking the body content against the TOC title.
OCR_TYPO_FIXES = {(70, 79): 71}

# Sections whose number was never recognisably detected on their page (not
# even as a misread digit) despite real effort to recover them: top-margin
# cropping, contrast enhancement, and a dedicated low-confidence sparse-text
# pass all failed to pick up a usable candidate. THEIR CONTENT IS NOT LOST -
# it is present in the database, silently merged into the end of the
# PRECEDING section's text (e.g. section 5's text currently trails onto the
# end of section 4's chunk). Only the ability to find it by searching for
# "section 5" specifically is affected; semantic search over its actual
# wording still works. Revisit with a cleaner source PDF if these are
# needed by exact citation.
KNOWN_UNRECOVERED_SECTIONS = [5, 10, 11, 41, 51, 57, 58, 71, 72, 77, 81, 109, 111, 118]


def extract_page_lines(page_image):
    """Reconstruct one page's reading order using TWO separate OCR passes
    (see the module docstring for why a single whole-page pass fails):

    1. The number column (the leftmost ~7% of the page) is cropped out and
       OCR'd on its own with --psm 6. A page-spanning pass misses some of
       these numbers entirely (they're small, isolated digits with no
       neighbouring text on the same line) - cropped and OCR'd alone,
       they're reliably picked up.
    2. The body column (7%-85% width - this also excludes the marginal
       note column on the right) is OCR'd separately, giving clean
       paragraph text with no numbers or notes mixed in.

    Each detected number is then spliced back in front of the body line at
    the closest matching vertical position.
    """
    width, height = page_image.size

    number_strip = page_image.crop((0, 0, int(NUMBER_ZONE_FRACTION * width), height))
    number_data = pytesseract.image_to_data(number_strip, config="--psm 6", output_type=Output.DICT)
    numbers = []  # [(top, number_string), ...]
    for i in range(len(number_data["text"])):
        text = number_data["text"][i].strip()
        match = NUMBER_PATTERN.match(text)
        if match:
            value = match.group(1)
            value = "1" if value in ("l", "L", "I") else value
            numbers.append((number_data["top"][i], value))

    body_strip = page_image.crop((int(NUMBER_ZONE_FRACTION * width), 0, int(MARGIN_ZONE_FRACTION * width), height))
    body_data = pytesseract.image_to_data(body_strip, config="--psm 4", output_type=Output.DICT)
    lines_by_key = defaultdict(list)
    line_top = {}
    for i in range(len(body_data["text"])):
        text = body_data["text"][i].strip()
        if not text:
            continue
        key = (body_data["block_num"][i], body_data["par_num"][i], body_data["line_num"][i])
        lines_by_key[key].append((body_data["left"][i], text))
        line_top[key] = min(line_top.get(key, body_data["top"][i]), body_data["top"][i])

    ordered_keys = sorted(lines_by_key, key=lambda k: line_top[k])
    body_lines = [(line_top[k], " ".join(w for _, w in sorted(lines_by_key[k]))) for k in ordered_keys]

    used = set()
    merged = []
    for top, text in body_lines:
        # A number never splices onto a PART heading line itself - the real
        # section start is the next body line below it, and skipping the
        # heading here lets that next line claim the match instead (this
        # was the most common misplacement: a section's number sits close
        # enough, vertically, to the Part heading above it to match it by
        # mistake).
        if PART_HEADING.match(text):
            merged.append(text)
            continue

        spliced = text
        for j, (ntop, num) in enumerate(numbers):
            if j not in used and abs(ntop - top) <= VERTICAL_TOLERANCE:
                spliced = f"{num}. {text}"
                used.add(j)
                break
        merged.append(spliced)
    return merged


TOC_LAST_PAGE = 4  # pages 1-4 are the Explanatory Memorandum and the Arrangement of Sections


def extract_text():
    """The TOC (pages 1-4) is one simple column, so it's OCR'd whole-page,
    same as a text-layer PDF would be - the layout-aware splicing in
    extract_page_lines() is neither needed nor wanted there (it's tuned to
    the body's 3-column layout, and mis-handles the TOC's dense one-entry-
    per-line format). The body (page 5 onward) uses that splicing."""
    page_count = len(PdfReader(PDF_FILE).pages)
    with open(RAW_TEXT_FILE, "w", encoding="utf-8") as out:
        for page in range(1, page_count + 1):
            image = convert_from_path(PDF_FILE, dpi=OCR_DPI, first_page=page, last_page=page)[0]
            out.write(f"\n--- PAGE {page} ---\n")
            if page <= TOC_LAST_PAGE:
                out.write(pytesseract.image_to_string(image, config="--psm 4"))
            else:
                out.write("\n".join(extract_page_lines(image)))
                out.write("\n")
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
    """The TOC's own copy of 'PART I - PRELIMINARY' is sometimes OCR'd
    badly (unlike the body's clean copy), so rather than requiring two
    matches of that heading, anchor on the enacting clause that always
    immediately precedes the real body's copy."""
    enacted_at = next((i for i, (_, l) in enumerate(lines) if l.upper().startswith("ENACTED BY")), None)
    if enacted_at is None:
        raise ValueError("Couldn't find the enacting clause ('ENACTED by the National Assembly...').")

    start = next(
        (i for i in range(enacted_at, len(lines)) if BODY_START_MARKER.match(lines[i][1])), None
    )
    if start is None:
        raise ValueError("Couldn't find 'PART I - PRELIMINARY' after the enacting clause.")

    end = next((i for i in range(start, len(lines)) if lines[i][1].upper().startswith("SCHEDULE")), None)
    if end is None:
        raise ValueError("Couldn't find the SCHEDULE boundary after the body.")
    return start, end


# A handful of TOC lines came out too garbled for SECTION_START to
# recognise at all (confirmed against the visible text): "11." became
# "il.", "42." became "4°},", "128." lost its leading digit and collided
# with the real section 28. Each is fixed by matching the exact garbled
# text pytesseract produced and substituting the correct "N. " prefix.
TOC_OCR_LINE_FIXES = [
    (re.compile(r"^il\.\s*"), "11. "),
    (re.compile(r"^4°\}," ), "42."),
    (re.compile(r"^\*\s*(\d)"), r"\1"),       # drop a stray leading footnote asterisk, e.g. "* 96."
    (re.compile(r"^28\.\s*Discipline\.?$"), "128. Discipline."),  # the 128->28 collision
]


def parse_toc_titles(lines):
    titles = {}
    for _, raw_line in lines:
        line = raw_line
        for pattern, replacement in TOC_OCR_LINE_FIXES:
            line = pattern.sub(replacement, line)

        if PART_HEADING.match(line):
            continue
        match = SECTION_START.match(line)
        if match:
            number = int(match.group(1))
            if 1 <= number <= MAX_SECTION_NUMBER:
                titles[number] = match.group(2).strip().rstrip(".:")
    return titles


def parse_sections(lines, titles):
    sections = []
    current = None
    last_number = None
    part = None

    i = 0
    while i < len(lines):
        page, line = lines[i]

        part_match = PART_HEADING.match(line)
        if part_match:
            part = part_match.group(0)
            i += 1
            continue

        section_match = SECTION_START.match(line)
        number = int(section_match.group(1)) if section_match else None
        if number is not None and (last_number, number) in OCR_TYPO_FIXES:
            number = OCR_TYPO_FIXES[(last_number, number)]

        valid_start = (
            section_match
            and number in titles
            and (last_number is None or last_number < number <= last_number + 3)
        )
        if valid_start:
            rest = section_match.group(2)
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
            "id": f"police-act-{section['label']}-{index}",
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
    unexpected_missing = [n for n in missing if n not in KNOWN_UNRECOVERED_SECTIONS]

    print(f"Sections found : {len(found)} of {MAX_SECTION_NUMBER} "
          f"({len(KNOWN_UNRECOVERED_SECTIONS)} known-unrecoverable, see module docstring)")
    print(f"TOC titles     : {len(titles)}")
    print(f"Chunks created : {len(chunks)} (longest {max(len(c['text']) for c in chunks)} chars)")
    if duplicates:
        print(f"WARNING - duplicated section labels: {duplicates}")
    if unexpected_missing:
        print(f"\nWARNING - NEWLY missing sections (not in the known list, investigate): {unexpected_missing}")
    elif missing:
        print(f"\nMissing sections match the known, documented gap exactly: {missing}")
    print("\nNOTE: this source is a scanned, 3-column PDF processed with layout-aware "
          "OCR - structure is verified for the sections found, exact wording is not "
          "(see the module docstring).")


def main():
    print(f"OCR'ing {PDF_FILE.name} (slow - layout reconstruction per page)...")
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
