"""
Step 3: build the search index (replaces embed_constitution.py).

Combines EVERY source of legal text into one list of records, embeds them,
and saves:

    data/index/records.jsonl   the text + metadata that will be shown/cited
    data/index/embeddings.npy  one vector per record, in the same order
    data/index/meta.json       which model built the index

Why one index? So the assistant searches a single place. Anything that is
not in this index can never reach the LLM, which is what keeps answers
inside your own database.

Run:  python build_index.py     (re-run whenever laws.json or the chunks change)
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    INDEX_DIR,
    INDEX_META_FILE,
    LAWS_FILE,
    MAX_TOPICS_EMBEDDED,
    RECORDS_FILE,
)


# ------------------------------------------------------- loading sources
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def constitution_record(chunk):
    """Convert a chunk from chunk_constitution.py to the common record format."""
    return {
        "id": chunk["id"],
        "source": chunk["document"],
        "reference": chunk["reference"],      # "Section 36" or "Schedule II, Part I"
        "title": chunk.get("context"),        # e.g. "Chapter IV: Fundamental Rights"
        "topics": chunk.get("topics", []),    # Constitute topic tags, used only for search
        "chapter": None,
        "pages": f"{chunk['page_start']}-{chunk['page_end']}",
        "jurisdiction": "Federal",
        "text": chunk["text"],
    }


def law_record(law):
    """Convert an entry from laws.json to the common record format."""
    return {
        "id": f"law-{law['id']}",
        "source": law["law"],
        "reference": law["section"],
        "title": law.get("title"),
        "topics": [],
        "chapter": None,
        "pages": None,
        "jurisdiction": law.get("jurisdiction"),
        "text": law["content"],
    }


def load_records():
    records = [constitution_record(c) for c in load_jsonl(CHUNKS_FILE)]

    with open(LAWS_FILE, "r", encoding="utf-8") as f:
        records += [law_record(law) for law in json.load(f)]

    return records


# ------------------------------------------------------ text to embed
def searchable_text(record):
    """
    The string that gets embedded. The header (law name, section, heading,
    topic tags) is included so a question such as "what does the Labour Act
    say about termination?" or "is there a right to life?" can match on the
    law name and topic, not just the body text.

    Only this embedded string contains the topic tags; the text shown to the
    user (record["text"]) is always the plain law.
    """
    topics = ", ".join(record.get("topics", [])[:MAX_TOPICS_EMBEDDED])
    header = [record["source"], record["reference"], record["title"], topics]
    header = ". ".join(part for part in header if part)
    return f"{header}. {record['text']}"


# ---------------------------------------------------------------- main
def main():
    records = load_records()
    print(f"Loaded {len(records)} records.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Chunks longer than the model's window are silently truncated. Warn early.
    limit_chars = model.max_seq_length * 4  # rough estimate: 1 token ~ 4 characters
    too_long = [r["id"] for r in records if len(searchable_text(r)) > limit_chars]
    if too_long:
        print(f"WARNING: {len(too_long)} records may be truncated by the model "
              f"(e.g. {too_long[:3]}). Lower MAX_CHUNK_CHARS in config.py.")

    print("Creating embeddings...")
    embeddings = model.encode(
        [searchable_text(r) for r in records],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit length -> dot product == cosine similarity
        convert_to_numpy=True,
    ).astype(np.float32)             # half the size of float64, no visible accuracy loss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Save the records exactly as they are (no text rewriting here).
    with open(RECORDS_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    np.save(EMBEDDINGS_FILE, embeddings)

    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"model": EMBEDDING_MODEL, "count": len(records)}, f)

    print(f"\nDone. Saved {len(records)} records to {INDEX_DIR}")


if __name__ == "__main__":
    main()
