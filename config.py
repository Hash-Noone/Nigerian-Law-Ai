"""
Shared settings for the Nigerian Law AI project.

Every script imports its paths and tuning values from here, so a change
(a new embedding model, a different chunk size, ...) is made in ONE place.

Data flow:

    constitution.pdf
        -> extract_law.py       -> constitution.txt          (raw text, page markers)
        -> chunk_constitution.py-> constitution_chunks.jsonl (one record per section chunk)
        -> build_index.py       -> index/records.jsonl + index/embeddings.npy
                                   (constitution chunks + laws.json, searchable)
        -> main.py              (FastAPI: retrieve, then answer ONLY from what was retrieved)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read variables from a file named ".env" (note the leading dot).
load_dotenv()

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PDF_FILE = DATA_DIR / "constitution.pdf"
RAW_TEXT_FILE = DATA_DIR / "constitution.txt"
CHUNKS_FILE = DATA_DIR / "constitution_chunks.jsonl"
LAWS_FILE = DATA_DIR / "laws.json"

INDEX_DIR = DATA_DIR / "index"
RECORDS_FILE = INDEX_DIR / "records.jsonl"      # the text + metadata that gets searched
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"  # one vector per record, same order
INDEX_META_FILE = INDEX_DIR / "meta.json"       # remembers which model built the index

STATIC_DIR = BASE_DIR / "static"  # the chat UI (index.html), served by main.py at /ui

# ---------------------------------------------------------- constitution
DOCUMENT_TITLE = "Constitution of the Federal Republic of Nigeria 1999 (as amended through 2011)"
MAX_SECTION_NUMBER = 320  # the main body of the Constitution ends at section 320

# Longest chunk (in characters) we allow. all-MiniLM-L6-v2 only reads the first
# ~256 tokens (~1,000 characters, headers included) of any text, so longer chunks would be
# silently cut off and the end of a long section could never be found.
MAX_CHUNK_CHARS = 700

# Topic tags added to each chunk's embedded header (they help search, but are
# not shown to users as law).
MAX_TOPICS_EMBEDDED = 5

# ------------------------------------------------------------ retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5          # how many passages are handed to the LLM
MIN_SCORE = 0.30   # cosine similarity below this = "not relevant". TUNE THIS (see README notes)

# Section-aware expansion (see retrieval.py). A section is chunked only
# because the embedding model has a size limit -- legally it's one unit, so
# once semantic search says a section is relevant we hand the LLM the whole
# thing rather than a random one-chunk fragment of it.
EXPAND_TOP_SECTIONS = 3      # expand at most this many distinct sections/schedule parts
MAX_SECTION_CHARS = 2500     # cap a single expanded section (e.g. the long definitions
                             # section, 318) so one section can't crowd out everything else
MAX_CONTEXT_CHARS = 6000     # stop adding passages once the combined context reaches this

# ------------------------------------------------------------------ LLM
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "liquid/lfm-2.5-2.6b:free")
