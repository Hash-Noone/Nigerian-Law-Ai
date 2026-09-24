"""
Step 1: PDF -> raw text.

Writes every page to constitution.txt, each one preceded by a
"--- PAGE n ---" marker so later steps can report page numbers.

Run:  python extract_law.py
"""

from pypdf import PdfReader

from config import PDF_FILE, RAW_TEXT_FILE


def extract_pdf_text(pdf_path, output_path):
    reader = PdfReader(pdf_path)

    with open(output_path, "w", encoding="utf-8") as out:
        for page_number, page in enumerate(reader.pages, start=1):
            # extract_text() can return None for image-only pages.
            text = page.extract_text() or ""

            out.write(f"\n--- PAGE {page_number} ---\n")
            out.write(text)

    return len(reader.pages)


if __name__ == "__main__":
    pages = extract_pdf_text(PDF_FILE, RAW_TEXT_FILE)
    print(f"Extracted {pages} pages to {RAW_TEXT_FILE}")
