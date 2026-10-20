"""The same decision policy as agent.py, expressed as a LangGraph state graph.

Why this file exists: agent.py hand-rolls the policy (retrieve, check the
best hit's distance, maybe fall back to a keyword search, generate a cited
answer or decline). Several of the roles this project is meant to speak to
name orchestration frameworks such as LangGraph, so this module re-expresses
the identical policy as an explicit graph and lets the evaluation compare the
two engines on the same questions.

What is and is not framework code here:
- LangGraph owns the orchestration only: the state, the nodes, the
  conditional edges and the invocation.
- Retrieval, the keyword tool, prompt construction and the Gemini call are the
  exact same functions agent.py uses, so any difference in the evaluation
  comes from orchestration and not from a different retriever or prompt.
- No LangChain LLM or vector-store wrappers are used on purpose, for the same
  reason.

Graph:

    START -> retrieve --(weak or empty)--> keyword_search --> check_context
                      \\--(confident)--------------------------^
    check_context --(no hits)--> decline --> END
    check_context --(hits)-----> generate --> END
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import agent as hand_rolled
from retrieve import format_hits_for_prompt, get_collection, retrieve
from tools import search_keyword
from llm_client import call_gemini

DECLINE_TEXT = "I don't have enough information in the indexed context to answer that."


class QAState(TypedDict, total=False):
    question: str
    repo_root: str
    store_dir: str
    collection_name: str
    k: int
    hits: List[Dict[str, Any]]
    tools_used: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    path: List[str]  # node names visited, for tracing and for tests


def _visit(state: QAState, name: str) -> List[str]:
    return list(state.get("path", [])) + [name]


def build_graph(
    retriever: Callable = None,
    keyword_search: Callable = None,
    llm: Callable = None,
    threshold: float = None,
    collection_getter: Callable = None,
):
    """Builds and compiles the graph. The collaborators are injectable so tests
    can drive every route without Chroma or a live model; by default they are
    the same functions agent.py uses."""
    collection_getter = collection_getter or get_collection
    retriever = retriever or retrieve
    keyword_search = keyword_search or search_keyword
    llm = llm or call_gemini
    threshold = hand_rolled.LOW_CONFIDENCE_THRESHOLD if threshold is None else threshold

    def retrieve_node(state: QAState) -> QAState:
        collection = collection_getter(Path(state["store_dir"]), state.get("collection_name", "repo_qa"))
        hits = retriever(collection, state["question"], k=state.get("k", 5))
        return {"hits": list(hits), "tools_used": [], "path": _visit(state, "retrieve")}

    def route_after_retrieve(state: QAState) -> str:
        hits = state["hits"]
        if not hits or hits[0]["distance"] > threshold:
            return "keyword_search"
        return "check_context"

    def keyword_node(state: QAState) -> QAState:
        hits = list(state["hits"])
        tools_used = list(state["tools_used"])
        term = hand_rolled._distinctive_token(state["question"])
        if term:
            keyword_hits = keyword_search(Path(state["repo_root"]), term)
            tools_used.append({"tool": "search_keyword", "term": term, "hit_count": len(keyword_hits)})
            for kh in keyword_hits:
                hits.append({
                    "source": kh["source"],
                    "start_line": kh["line_number"],
                    "end_line": kh["line_number"],
                    "text": kh["line_text"],
                    "distance": None,
                })
        return {"hits": hits, "tools_used": tools_used, "path": _visit(state, "keyword_search")}

    def check_context_node(state: QAState) -> QAState:
        return {"path": _visit(state, "check_context")}

    def route_after_check(state: QAState) -> str:
        return "generate" if state["hits"] else "decline"

    def generate_node(state: QAState) -> QAState:
        context = format_hits_for_prompt(state["hits"])
        prompt = hand_rolled.ANSWER_PROMPT_TEMPLATE.format(context=context, question=state["question"])
        text = llm(prompt)
        return {
            "answer": text.strip(),
            "sources": sorted({h["source"] for h in state["hits"]}),
            "path": _visit(state, "generate"),
        }

    def decline_node(state: QAState) -> QAState:
        return {"answer": DECLINE_TEXT, "sources": [], "path": _visit(state, "decline")}

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("keyword_search", keyword_node)
    graph.add_node("check_context", check_context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("decline", decline_node)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve, {
        "keyword_search": "keyword_search",
        "check_context": "check_context",
    })
    graph.add_edge("keyword_search", "check_context")
    graph.add_conditional_edges("check_context", route_after_check, {
        "generate": "generate",
        "decline": "decline",
    })
    graph.add_edge("generate", END)
    graph.add_edge("decline", END)
    return graph.compile()


_DEFAULT_GRAPH = None


def answer(question, repo_root, store_dir, collection_name="repo_qa", k=5, graph=None):
    """Drop-in replacement for agent.answer(): same arguments, same result keys
    (answer, sources, tools_used), plus a 'path' key listing the nodes visited."""
    global _DEFAULT_GRAPH
    if graph is None:
        if _DEFAULT_GRAPH is None:
            _DEFAULT_GRAPH = build_graph()
        graph = _DEFAULT_GRAPH
    final = graph.invoke({
        "question": question,
        "repo_root": str(repo_root),
        "store_dir": str(store_dir),
        "collection_name": collection_name,
        "k": k,
    })
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "tools_used": final["tools_used"],
        "path": final["path"],
    }


def mermaid() -> str:
    """The graph as a Mermaid diagram, generated from the compiled graph itself
    (so the README diagram cannot drift from the code)."""
    return build_graph().get_graph().draw_mermaid()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("repo_root", nargs="?")
    parser.add_argument("--store", default="./chroma_store")
    parser.add_argument("--collection", default="repo_qa")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mermaid", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args()

    if args.mermaid:
        print(mermaid())
    else:
        result = answer(args.question, Path(args.repo_root).resolve(), Path(args.store).resolve(), args.collection, args.k)
        print("\nANSWER:\n" + result["answer"])
        print("\nSOURCES:", result["sources"])
        print("TOOLS USED:", result["tools_used"])
        print("PATH:", " -> ".join(result["path"]))
