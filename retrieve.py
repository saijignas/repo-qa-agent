"""Retrieval over the Chroma index built by ingest.py. Pure retrieval,
no LLM call -- kept separate so retrieval quality can be measured and
tested independent of generation."""
import argparse
from pathlib import Path

import chromadb


def get_collection(store_dir, collection_name):
    client = chromadb.PersistentClient(path=str(store_dir))
    return client.get_collection(collection_name)


def retrieve(collection, query, k=5):
    """Returns a list of {source, start_line, end_line, text, distance},
    ordered by relevance (ascending distance = more relevant)."""
    results = collection.query(query_texts=[query], n_results=k)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "source": meta["source"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "text": doc,
            "distance": dist,
        })
    return hits


def format_hits_for_prompt(hits):
    """Renders retrieved chunks as labeled context blocks an LLM prompt
    can cite back to (e.g. "according to consumer/worker.py:40-75")."""
    blocks = []
    for hit in hits:
        blocks.append(
            f"[{hit['source']}:{hit['start_line']}-{hit['end_line']}]\n{hit['text']}"
        )
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--store", default="./chroma_store")
    parser.add_argument("--collection", default="repo_qa")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    collection = get_collection(Path(args.store).resolve(), args.collection)
    hits = retrieve(collection, args.query, k=args.k)
    for i, hit in enumerate(hits, 1):
        print(f"\n#{i} {hit['source']}:{hit['start_line']}-{hit['end_line']} (distance={hit['distance']:.4f})")
        print(hit["text"][:300])
