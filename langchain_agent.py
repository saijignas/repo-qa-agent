"""The LangGraph agent from graph_agent.py, with its parts swapped for LangChain's own.

graph_agent.py changed only the orchestration and kept this project's retrieval,
keyword tool, prompt and Gemini call. This module goes one step further and uses
LangChain's wrappers for those parts too, so the evaluation can ask whether the
framework's components behave differently from the hand-written ones:

  hand-written part                     LangChain part used here
  ------------------------------------  ---------------------------------------------
  retrieve.py (Chroma query)            langchain_chroma.Chroma.similarity_search_with_score
  ONNX embeddings inside Chroma         an Embeddings adapter around the SAME Chroma model
  ANSWER_PROMPT_TEMPLATE.format(...)    ChatPromptTemplate built from the same template text
  llm_client.call_gemini (requests)     ChatGoogleGenerativeAI, parsed with StrOutputParser
  tools.search_keyword                  the same function wrapped as a LangChain @tool

Deliberately unchanged: the decision policy (the LangGraph graph is reused as-is),
the index (the existing Chroma collection is opened, not rebuilt), the prompt text
and the model. Sampling is left at the API default by passing temperature=None,
because the wrapper's own default is 0.7 while the hand-written client sends no
temperature at all; leaving that alone would have made the two engines differ in
something that has nothing to do with the framework.

The embedding adapter matters: a Chroma index only answers correctly in the vector
space it was built in, and the index here was built with Chroma's default ONNX
MiniLM model. Giving the LangChain wrapper a different embedding model would return
nonsense distances, so the adapter reuses Chroma's own function.
"""
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

import agent as hand_rolled
import graph_agent
from llm_client import GEMINI_MODEL
from tools import search_keyword as _search_keyword


class ChromaDefaultEmbeddings(Embeddings):
    """LangChain Embeddings interface over Chroma's default embedding function, so
    queries are embedded in the same space as the existing index."""

    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in v] for v in self._fn(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def open_vectorstore(store_dir, collection_name: str = "repo_qa") -> Chroma:
    """Opens the EXISTING collection (never creates or rebuilds one)."""
    client = chromadb.PersistentClient(path=str(Path(store_dir)))
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=ChromaDefaultEmbeddings(),
        create_collection_if_not_exists=False,
    )


_VECTORSTORES: Dict[tuple, Chroma] = {}


def collection_getter(store_dir, collection_name):
    key = (str(store_dir), collection_name)
    if key not in _VECTORSTORES:
        _VECTORSTORES[key] = open_vectorstore(store_dir, collection_name)
    return _VECTORSTORES[key]


def retriever(vectorstore: Chroma, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Same hit shape as retrieve.retrieve: {source, start_line, end_line, text, distance}."""
    hits = []
    for doc, distance in vectorstore.similarity_search_with_score(query, k=k):
        hits.append({
            "source": doc.metadata["source"],
            "start_line": doc.metadata["start_line"],
            "end_line": doc.metadata["end_line"],
            "text": doc.page_content,
            "distance": distance,
        })
    return hits


@tool
def search_keyword_tool(repo_root: str, term: str) -> list:
    """Exact, case-insensitive substring search over the indexed source files of a repository.
    Returns a list of {source, line_number, line_text}."""
    return _search_keyword(Path(repo_root), term)


def keyword_search(repo_root, term):
    return search_keyword_tool.invoke({"repo_root": str(repo_root), "term": term})


ANSWER_PROMPT = ChatPromptTemplate.from_template(hand_rolled.ANSWER_PROMPT_TEMPLATE)


def make_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=None,  # leave sampling at the API default, as the hand-written client does
        max_output_tokens=1024,
        timeout=60,
    )


def build_chain(model=None):
    """prompt | model | parser. `model` may be any runnable (tests pass a fake)."""
    return ANSWER_PROMPT | (model or make_model()) | StrOutputParser()


def build_graph(model=None, **overrides):
    chain = build_chain(model)

    def generator(context: str, question: str) -> str:
        return chain.invoke({"context": context, "question": question})

    kwargs = dict(
        retriever=retriever,
        keyword_search=keyword_search,
        collection_getter=collection_getter,
        generator=generator,
    )
    kwargs.update(overrides)
    return graph_agent.build_graph(**kwargs)


_GRAPH = None


def answer(question, repo_root, store_dir, collection_name="repo_qa", k=5, graph=None):
    """Same signature and result keys as agent.answer and graph_agent.answer."""
    global _GRAPH
    if graph is None:
        if _GRAPH is None:
            _GRAPH = build_graph()
        graph = _GRAPH
    return graph_agent.answer(question, repo_root, store_dir, collection_name, k, graph=graph)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    print("PATH:", " -> ".join(result["path"]))
