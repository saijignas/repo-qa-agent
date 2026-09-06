"""Tools the agent can call when retrieval alone isn't enough. Kept to
two, deliberately: a keyword search (exact-match, for identifiers/config
values that semantic embedding similarity often misses -- e.g. searching
for "MAX_RETRIES" or "SETNX") and a raw file fetch (for when the agent
needs to see a whole file, not just a 40-line indexed chunk)."""
from pathlib import Path


def search_keyword(repo_root, term, max_hits=8):
    """Plain substring search across indexed source files. Returns a
    list of {source, line_number, line_text} -- exact matches only, no
    fuzzy/semantic matching, which is exactly why this complements
    vector search instead of duplicating it."""
    repo_root = Path(repo_root)
    hits = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md", ".yml", ".yaml"}:
            continue
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        rel_path = path.relative_to(repo_root).as_posix()
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if term.lower() in line.lower():
                hits.append({"source": rel_path, "line_number": i, "line_text": line.strip()})
                if len(hits) >= max_hits:
                    return hits
    return hits


def fetch_file(repo_root, rel_path, start_line=None, end_line=None):
    """Returns raw file contents, optionally sliced to a line range.
    Raises FileNotFoundError with a clear message (not a bare
    traceback) if the agent asks for a path that doesn't exist --
    that's a legitimate outcome to hand back to the model, not a bug."""
    repo_root = Path(repo_root)
    full_path = (repo_root / rel_path).resolve()
    if repo_root.resolve() not in full_path.parents and full_path != repo_root.resolve():
        raise ValueError(f"Refusing to read outside the repo root: {rel_path}")
    if not full_path.is_file():
        raise FileNotFoundError(f"No such file in repo: {rel_path}")

    lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if start_line is not None:
        start_line = max(1, start_line)
        end_line = end_line or len(lines)
        lines = lines[start_line - 1:end_line]
    return "\n".join(lines)
