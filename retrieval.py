"""
Search the index built by build_index.py (replaces search_constitution.py).

Used in two ways:

    * imported by main.py:   Retriever().search("question")
    * from the terminal:     python retrieval.py "can police detain me for 48 hours?"
"""

import json
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    INDEX_META_FILE,
    MIN_SCORE,
    RECORDS_FILE,
    TOP_K,
)

# "what does section 36 say" -> 36
SECTION_IN_QUERY = re.compile(r"\bsection\s+(\d{1,3})\b", re.IGNORECASE)


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

    # ------------------------------------------------------------ search
    def search(self, query, top_k=TOP_K, min_score=MIN_SCORE):
        """
        Return up to `top_k` records (best first), each with a "score" key.

        1. If the question names a section ("section 36"), those records
           come first: embeddings are poor at matching bare numbers.
        2. The rest are the closest records by meaning, and only those scoring
           at least `min_score` are kept. Weak matches are dropped so the LLM
           is never handed irrelevant text.
        """
        hits = []
        seen = set()

        for number in SECTION_IN_QUERY.findall(query):
            reference = f"Section {int(number)}"
            for i, record in enumerate(self.records):
                if record["reference"] == reference and i not in seen:
                    seen.add(i)
                    hits.append({**record, "score": 1.0})

        query_vector = self.model.encode(query, normalize_embeddings=True)

        # One matrix-vector product scores every record at once
        # (vectors are unit length, so the dot product is the cosine similarity).
        scores = self.embeddings @ query_vector

        for i in np.argsort(-scores):
            if len(hits) >= top_k or scores[i] < min_score:
                break
            if int(i) not in seen:
                seen.add(int(i))
                hits.append({**self.records[i], "score": float(scores[i])})

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
        print(f"\n#{rank}  score {hit['score']:.3f}  |  {hit['source']}  |  {hit['reference']}", end="")
        print(f"  |  {hit['title']}" if hit["title"] else "")
        print("-" * 70)
        print(hit["text"])


if __name__ == "__main__":
    main()
