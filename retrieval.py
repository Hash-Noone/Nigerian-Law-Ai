"""
Search the index built by build_index.py (replaces search_constitution.py).

Used in two ways:

    * imported by main.py:   Retriever().search("question")
    * from the terminal:     python retrieval.py "can police detain me for 48 hours?"

Section-aware retrieval
------------------------
A Constitution section is chunked only because the embedding model has a
size limit -- legally it is still ONE unit. If a question is about section
39 (freedom of expression) and semantic search happens to return only
chunk 0 (the right itself, subsection (1)), the LLM would never see chunk 1
(the limitations in subsection (3)) and could give a dangerously incomplete
answer.

So search() works in three passes:

    1. EXPLICIT   "section 36" in the question -> that whole section, no
                  matter what the embeddings say (they're poor at bare numbers).
    2. EXPAND     among the best semantic matches, the top few DISTINCT
                  sections are expanded to their full text (every chunk
                  they were split into, combined and deduplicated).
    3. FILL       remaining slots (up to top_k) are filled with the next-best
                  individual chunks, unexpanded, so a broad question doesn't
                  turn into a single giant section dump.

A running character budget (MAX_CONTEXT_CHARS) stops the LLM's prompt from
growing without bound, and MAX_SECTION_CHARS caps any one expanded section
so a single very long section can't crowd out everything else.

No index rebuild is needed for any of this: it only regroups the chunks
build_index.py already produced.
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    EXPAND_TOP_SECTIONS,
    INDEX_META_FILE,
    MAX_CONTEXT_CHARS,
    MAX_SECTION_CHARS,
    MIN_SCORE,
    RECORDS_FILE,
    TOP_K,
)

# "what does section 36 say" -> 36
SECTION_IN_QUERY = re.compile(r"\bsection\s+(\d{1,3}[a-z]?)\b", re.IGNORECASE)

# A chunk's id ends in its position within the section/schedule part, e.g.
# "constitution-39-0", "constitution-39-1" -> 0, 1. Used only to put an
# expanded section's chunks back in reading order.
CHUNK_ORDER = re.compile(r"-(\d+)$")


class Retriever:
    """Loads the index once, then answers many searches quickly."""

    def __init__(self):
        with open(INDEX_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # A query must be embedded with the SAME model that embedded the
        # records, otherwise the similarity scores are meaningless.
        if meta["model"] != EMBEDDING_MODEL:
            raise RuntimeError(
                f"Index was built with '{meta['model']}' but config uses "
                f"'{EMBEDDING_MODEL}'. Re-run build_index.py."
            )

        with open(RECORDS_FILE, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        self.embeddings = np.load(EMBEDDINGS_FILE)  # shape: (records, dimensions)

        if len(self.records) != len(self.embeddings):
            raise RuntimeError("records.jsonl and embeddings.npy are out of sync. Re-run build_index.py.")

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Group chunk indices that belong to the same section/schedule part,
        # e.g. ("Constitution ...", "Section 39") -> [12, 13]. This is what
        # lets us go from "chunk 12 scored well" to "here is all of section 39".
        self._groups = defaultdict(list)
        for i, record in enumerate(self.records):
            self._groups[self._key(record)].append(i)
        for indices in self._groups.values():
            indices.sort(key=lambda i: self._chunk_order(self.records[i]["id"]))

    @staticmethod
    def _key(record):
        """Records are grouped by (source law, reference) - e.g. two laws can
        both have a "Section 24" without being mixed together."""
        return (record["source"], record["reference"])

    @staticmethod
    def _chunk_order(record_id):
        match = CHUNK_ORDER.search(record_id)
        return int(match.group(1)) if match else 0

    # -------------------------------------------------------- merging
    def _merge_group(self, key, score, max_chars=None):
        """Combine every chunk of one section/schedule part into a single
        passage, in reading order, deduplicating identical chunk text."""
        indices = self._groups[key]
        seen_text = set()
        pieces = []
        for i in indices:
            text = self.records[i]["text"]
            if text not in seen_text:
                seen_text.add(text)
                pieces.append(text)

        combined_text = "\n\n".join(pieces)
        truncated = max_chars is not None and len(combined_text) > max_chars
        if truncated:
            combined_text = combined_text[:max_chars].rsplit(" ", 1)[0] + " ..."

        base = dict(self.records[indices[0]])  # source, reference, title, topics, jurisdiction
        pages = [self.records[i]["pages"] for i in indices if self.records[i].get("pages")]
        base.update({
            "text": combined_text,
            "score": score,
            "expanded": len(indices) > 1,
            "truncated": truncated,
            "pages": f"{pages[0].split('-')[0]}-{pages[-1].split('-')[-1]}" if pages else None,
        })
        return base, indices

    # ------------------------------------------------------------ search
    def search(
        self,
        query,
        top_k=TOP_K,
        min_score=MIN_SCORE,
        expand_top_sections=EXPAND_TOP_SECTIONS,
        max_context_chars=MAX_CONTEXT_CHARS,
    ):
        """Return up to `top_k` passages (best first), each with a "score" key."""
        hits = []
        used_keys = set()   # (source, reference) already added, so we never add it twice
        used_indices = set()
        budget = max_context_chars

        def add(record, score, indices):
            nonlocal budget
            hits.append(record)
            used_keys.add(self._key(record))
            used_indices.update(indices)
            budget -= len(record["text"])

        # --- 1. EXPLICIT section number(s) named in the question.
        # Several sources can share a reference (e.g. two different laws both
        # have a "Section 24"), so this can add more than one whole section.
        for number in SECTION_IN_QUERY.findall(query):
            reference = f"Section {number.upper()}"
            for key in list(self._groups):
                if key[1] == reference and key not in used_keys:
                    record, indices = self._merge_group(key, score=1.0, max_chars=MAX_SECTION_CHARS)
                    add(record, 1.0, indices)

        # --- rank every chunk by meaning (one matrix-vector product scores
        # them all at once; vectors are unit length so the dot product is
        # the cosine similarity).
        query_vector = self.model.encode(query, normalize_embeddings=True)
        scores = self.embeddings @ query_vector
        ranked = np.argsort(-scores)

        # --- collect the best-scoring DISTINCT sections, in the order they
        # first appear (i.e. by their best chunk's score).
        candidate_keys = []
        for i in ranked:
            if scores[i] < min_score or len(hits) + len(candidate_keys) >= top_k * 3:
                break
            key = self._key(self.records[i])
            if key not in used_keys and key not in candidate_keys:
                candidate_keys.append(key)

        # --- 2. EXPAND the top few of those into their full section text.
        for key in candidate_keys[:expand_top_sections]:
            if len(hits) >= top_k or budget <= 0:
                break
            best_score = max(float(scores[i]) for i in self._groups[key])
            record, indices = self._merge_group(key, best_score, max_chars=MAX_SECTION_CHARS)
            add(record, best_score, indices)

        # --- 3. FILL remaining slots with the next-best individual chunks,
        # left unexpanded so a broad question doesn't dump whole chapters.
        for i in ranked:
            if len(hits) >= top_k or budget <= 0 or scores[i] < min_score:
                break
            if int(i) in used_indices:
                continue
            record = {**self.records[int(i)], "score": float(scores[i]), "expanded": False, "truncated": False}
            add(record, float(scores[i]), [int(i)])

        # Highest score first, regardless of which pass added it.
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]


# ------------------------------------------------------------------ CLI
def main():
    if len(sys.argv) < 2:
        print('Usage: python retrieval.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    retriever = Retriever()

    # min_score=0 so you can SEE the scores of weak matches while tuning MIN_SCORE.
    hits = retriever.search(query, min_score=0.0)

    print(f"\nQuestion: {query}\n" + "=" * 70)
    for rank, hit in enumerate(hits, start=1):
        flags = []
        if hit.get("expanded"):
            flags.append("full section")
        if hit.get("truncated"):
            flags.append("truncated")
        flag_text = f"  [{', '.join(flags)}]" if flags else ""

        print(f"\n#{rank}  score {hit['score']:.3f}  |  {hit['source']}  |  {hit['reference']}{flag_text}", end="")
        print(f"  |  {hit['title']}" if hit["title"] else "")
        print("-" * 70)
        print(hit["text"])


if __name__ == "__main__":
    main()
