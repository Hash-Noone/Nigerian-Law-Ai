"""
Nigerian Law AI - FastAPI app.

How an answer is produced (and why it stays inside YOUR database):

    question
       |
       v
    1. RETRIEVE  search the index (Constitution chunks + laws.json)
       |         -> nothing relevant enough? reply "no information" and stop.
       |            The LLM is never called, so it cannot make something up.
       v
    2. GROUND    give the LLM ONLY the retrieved passages, numbered [1], [2] ...
       |         and tell it to answer from them and cite the numbers.
       v
    3. VERIFY    check the answer really cites passages we supplied.
                 Return the passages themselves so the user can read the
                 exact source text.

A small LLM can still ignore instructions, which is why steps 1 and 3 are
enforced in code rather than left to the prompt alone.

Run:  uvicorn main:app --reload
"""

import os
import re

from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

from config import LLM_MODEL, OPENROUTER_BASE_URL, TOP_K
from retrieval import Retriever

# ------------------------------------------------------------------ setup
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Create a file named '.env' (with the dot) "
        "containing:  OPENROUTER_API_KEY=your-key"
    )

app = FastAPI(title="Nigerian Law AI")
client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

# Loaded once at startup (loading the model on every request would be very slow).
retriever = Retriever()

NO_ANSWER = (
    "I don't have enough information in my legal database to answer that question."
)
DISCLAIMER = "This is legal information from a limited database, not legal advice."
CITATION = re.compile(r"\[(\d+)\]")  # matches [1], [2] ...


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


# --------------------------------------------------------------- prompting
def build_context(hits):
    """Number each retrieved passage so the LLM can cite it as [n]."""
    blocks = []
    for n, hit in enumerate(hits, start=1):
        label = f"{hit['source']}, {hit['reference']}"
        if hit["title"]:
            label += f" ({hit['title']})"
        blocks.append(f"[{n}] {label}\n{hit['text']}")
    return "\n\n".join(blocks)


def build_system_prompt(context):
    return f"""You are a Nigerian legal information assistant.

Answer ONLY from the numbered LEGAL EXCERPTS below.

Rules:
1. Use nothing from your own knowledge: no other laws, sections, cases,
   penalties or legal principles than those written in the excerpts.
2. After every statement, cite the excerpt it came from, like [1] or [2].
3. If the excerpts do not contain enough information to answer, reply with
   exactly: {NO_ANSWER}
4. Do not guess and do not fill gaps. Keep the answer short and in plain English.

LEGAL EXCERPTS:

{context}
"""


# --------------------------------------------------------------- endpoints
@app.get("/")
def home():
    return {"message": "Nigerian Law AI is running", "records": len(retriever.records)}


@app.get("/laws")
def list_laws():
    """List what is in the database (metadata only, not the full text)."""
    return [
        {k: r[k] for k in ("id", "source", "reference", "title")}
        for r in retriever.records
    ]


@app.get("/laws/search")
def search_laws(query: str, top_k: int = Query(TOP_K, ge=1, le=10)):
    """Semantic search over the database, without calling the LLM."""
    return retriever.search(query, top_k=top_k)


@app.post("/ask")
def ask_question(data: Question):
    # 1. RETRIEVE ------------------------------------------------------
    hits = retriever.search(data.question)

    if not hits:
        # Nothing relevant: do not even call the LLM.
        return {
            "question": data.question,
            "answer": NO_ANSWER,
            "grounded": True,
            "sources": [],
            "disclaimer": DISCLAIMER,
        }

    # 2. GROUND --------------------------------------------------------
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,   # deterministic: less room to improvise
            max_tokens=500,
            messages=[
                {"role": "system", "content": build_system_prompt(build_context(hits))},
                {"role": "user", "content": data.question},
            ],
        )
    except OpenAIError as error:
        raise HTTPException(status_code=502, detail=f"The language model request failed: {error}")

    answer = (response.choices[0].message.content or "").strip()

    # 3. VERIFY --------------------------------------------------------
    if NO_ANSWER.lower() in answer.lower():
        return {
            "question": data.question,
            "answer": NO_ANSWER,
            "grounded": True,
            "sources": [],
            "disclaimer": DISCLAIMER,
        }

    cited = {int(n) for n in CITATION.findall(answer)}
    valid = {n for n in cited if 1 <= n <= len(hits)}
    invalid = cited - valid  # the model cited a passage we never gave it

    grounded = bool(valid) and not invalid

    # Show the passages the answer actually cites; if it cited none, show
    # everything that was retrieved so the user can check for themselves.
    shown = sorted(valid) or list(range(1, len(hits) + 1))

    result = {
        "question": data.question,
        "answer": answer,
        "grounded": grounded,
        "sources": [
            {
                "n": n,
                "source": hits[n - 1]["source"],
                "reference": hits[n - 1]["reference"],
                "title": hits[n - 1]["title"],
                "pages": hits[n - 1]["pages"],
                "score": round(hits[n - 1]["score"], 3),
                "text": hits[n - 1]["text"],
            }
            for n in shown
        ],
        "disclaimer": DISCLAIMER,
    }

    if not grounded:
        result["warning"] = (
            "The answer did not cite the supplied sources correctly. "
            "Rely on the source passages below, not on the answer text."
        )

    return result
