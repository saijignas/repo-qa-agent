"""Ties retrieval, tools, and generation into one answer() call.

Decision policy (deliberately explicit/rule-based rather than hidden
inside a free-form LLM tool-call loop, so it's inspectable and testable
without burning an extra generation call on every query -- the free
Gemini tier's daily request quota is small):

1. Vector-search the query.
2. If the best hit's distance is above LOW_CONFIDENCE_THRESHOLD, the
   embedding match is weak (e.g. the query names an exact identifier
   like "MAX_RETRIES" that semantic search under-weights) -- fall back
   to an exact keyword search on the query's most distinctive token and
   fold any hits into the context.
3. Build a prompt instructing the model to answer ONLY from the
   provided context, cite [file:line] for claims, and say so explicitly
   if the context doesn't contain the answer -- rather than letting it
   fall back on general knowledge about RabbitMQ/Postgres/etc., which
   would defeat the point of grounding this in the actual repo.
4. Return the answer plus which sources/tools were actually used, so
   eval can check citations against ground truth instead of trusting
   the prose.
"""
import re
from pathlib import Path

from llm_client import call_gemini
from retrieve import format_hits_for_prompt, get_collection, retrieve
from tools import search_keyword

LOW_CONFIDENCE_THRESHOLD = 1.15
STOPWORDS = {"the", "a", "an", "is", "does", "do", "how", "what", "why", "when", "and", "or", "to", "of", "in", "on", "for"}

ANSWER_PROMPT_TEMPLATE = """You are answering questions about a specific codebase, using ONLY the context below. Do not use general knowledge about RabbitMQ, Postgres, Kubernetes, etc. that isn't grounded in this context -- if the context doesn't contain the answer, say exactly "I don't have enough information in the indexed context to answer that" rather than guessing.

When you do answer, cite the source for each claim in [file:line-range] form, using the labels already on each context block.

Context:
{context}

Question: {question}

Answer:"""


def _distinctive_token(question):
    """Picks the longest non-stopword token as a fallback keyword-search
    term -- crude, but the fallback path only fires on low-confidence
    vector search, so it's a second opinion, not the primary mechanism."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", question)
    candidates = [t for t in tokens if t.lower() not in STOPWORDS]
    return max(candidates, key=len) if candidates else None


def answer(question, repo_root, store_dir, collection_name="repo_qa", k=5):
    repo_root = Path(repo_root)
    collection = get_collection(store_dir, collection_name)
    hits = retrieve(collection, question, k=k)

    tools_used = []
    if not hits or hits[0]["distance"] > LOW_CONFIDENCE_THRESHOLD:
        term = _distinctive_token(question)
        if term:
            keyword_hits = search_keyword(repo_root, term)
            tools_used.append({"tool": "search_keyword", "term": term, "hit_count": len(keyword_hits)})
            for kh in keyword_hits:
                hits.append({
                    "source": kh["source"],
                    "start_line": kh["line_number"],
                    "end_line": kh["line_number"],
                    "text": kh["line_text"],
                    "distance": None,
                })

    if not hits:
        return {
            "answer": "I don't have enough information in the indexed context to answer that.",
            "sources": [],
            "tools_used": tools_used,
        }

    context = format_hits_for_prompt(hits)
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
    response_text = call_gemini(prompt)

    return {
        "answer": response_text.strip(),
        "sources": sorted({h["source"] for h in hits}),
        "tools_used": tools_used,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("repo_root")
    parser.add_argument("--store", default="./chroma_store")
    parser.add_argument("--collection", default="repo_qa")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    result = answer(args.question, Path(args.repo_root).resolve(), Path(args.store).resolve(), args.collection, args.k)
    print("\nANSWER:\n" + result["answer"])
    print("\nSOURCES:", result["sources"])
    print("TOOLS USED:", result["tools_used"])
