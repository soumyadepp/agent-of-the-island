# tools.py
from __future__ import annotations

import os
import json
import pickle
import requests
import pdfplumber

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder
from env import get_env

OPENAI_API_KEY = get_env("OPENAI_API_KEY")

# -----------------------
# Basic tool(s)
# -----------------------
def string_search_tool(whole_text: str, substring: str) -> int:
    """Utility to count occurrences of a substring in a text."""
    return whole_text.count(substring)


def make_summarizer_tool(llm) -> Callable[[str], str]:
    """
    Returns a tool callable: summarizer_tool(text) -> summary
    Uses the provided LLM (so no extra API/services needed beyond your LLM).
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    def summarizer_tool(text: str) -> str:
        system = (
            "Summarize the given text.\n"
            "Rules:\n"
            "- Be concise.\n"
            "- Do not invent details.\n"
            "- If the text is long, extract key points.\n"
            "- Prefer 5-10 bullet points + a 1-2 line TL;DR.\n"
        )
        msgs = [
            SystemMessage(content=system),
            HumanMessage(content=text),
        ]
        resp = llm.invoke(msgs)
        return resp.content

    summarizer_tool.__name__ = "summarizer_tool"
    summarizer_tool.__doc__ = "Summarize user-provided text."
    return summarizer_tool


# -----------------------
# RAG storage (simple global state)
# -----------------------
@dataclass
class RagState:
    faiss: Optional[FAISS] = None
    bm25: Optional[BM25Okapi] = None
    chunks: Optional[List[Document]] = None   # same order as bm25 corpus
    index_dir: Optional[str] = None
    reranker: Optional[CrossEncoder] = None


RAG = RagState()


# -----------------------
# Helpers
# -----------------------
def _tokenize(text: str) -> List[str]:
    # MVP tokenizer; replace later if you want better (spaCy, etc.)
    return [t for t in text.lower().split() if t.strip()]


def _split_docs(docs: List[Document], chunk_size=900, chunk_overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _rrf_fuse(rankings: List[List[int]], k: int = 60) -> List[int]:
    """
    Reciprocal Rank Fusion (RRF):
      score(doc) = sum_i 1 / (k + rank_i(doc))
    """
    scores: Dict[int, float] = {}
    for rlist in rankings:
        for rank, idx in enumerate(rlist):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# -----------------------
# RAG tools
# -----------------------
def rag_build_index(file_paths_json: str, index_dir: str, embeddings) -> str:
    """
    Build BOTH FAISS (dense) and BM25 (lexical) indexes.
    Persists to disk in index_dir.

    Args:
        file_paths_json: JSON list of file paths: '["/path/a.pdf", "/path/b.txt"]'
        index_dir: output directory for FAISS + bm25.pkl
        embeddings: LangChain embeddings instance
    """

    paths = json.loads(file_paths_json)

    if not isinstance(paths, list) or not paths:
        return "ERROR: file_paths_json must be a JSON list of file paths."

    # Lazy import loaders to keep requirements minimal
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    docs: List[Document] = []
    for path in paths:
        # print(path)
        if not os.path.exists(path):
            continue

        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(str(path))
            loaded = loader.load()  # page-level docs
        else:
            loader = TextLoader(str(path), encoding="utf-8")
            loaded = loader.load()

        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["source"] = d.metadata.get("source") or str(path)
        docs.extend(loaded)

    if not docs:
        return "ERROR: No documents loaded (check file paths)."

    chunks = _split_docs(docs)
    for i, c in enumerate(chunks):
        c.metadata = c.metadata or {}
        c.metadata["chunk_id"] = i

    os.makedirs(index_dir, exist_ok=True)

    # Dense index (FAISS)
    faiss = FAISS.from_documents(chunks, embeddings)
    faiss.save_local(index_dir)

    # Lexical index (BM25)
    tokenized = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # Persist BM25 + chunks
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"tokenized": tokenized, "chunks": chunks, "bm25": bm25}, f)

    # Update in-memory state
    RAG.faiss = faiss
    RAG.bm25 = bm25
    RAG.chunks = chunks
    RAG.index_dir = index_dir

    return f"Built index at '{index_dir}' with {len(chunks)} chunks (FAISS + BM25)."


def rag_load_index(index_dir: str, embeddings) -> str:
    """Load FAISS + BM25 from disk."""
    if not os.path.isdir(index_dir):
        return f"ERROR: index_dir not found: {index_dir}"

    faiss = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    bm25_path = os.path.join(index_dir, "bm25.pkl")
    if not os.path.isfile(bm25_path):
        return f"ERROR: bm25.pkl missing in {index_dir} (did you build index first?)"

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)

    RAG.faiss = faiss
    RAG.bm25 = data["bm25"]
    RAG.chunks = data["chunks"]
    RAG.index_dir = index_dir

    return f"Loaded index from '{index_dir}'."


def rag_init_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> str:
    """
    Initialize a local cross-encoder reranker (no API key).
    Recommended (all local, HF):
      - cross-encoder/ms-marco-MiniLM-L-6-v2
      - BAAI/bge-reranker-base
      - mixedbread-ai/mxbai-rerank-base-v1
    """
    RAG.reranker = CrossEncoder(model_name)
    return f"Reranker loaded: {model_name}"


def rag_search(
    query: str,
    k_dense: int = 20,
    k_lex: int = 20,
    k_final: int = 8,
    use_rerank: bool = True,
) -> str:
    """
    Hybrid retrieval:
      1) BM25 top-k_lex
      2) FAISS top-k_dense
      3) Fuse with RRF
      4) Optional cross-encoder rerank of fused candidates (if rag_init_reranker called)
    Returns JSON list: [{rank, chunk_id, content, source, page}]
    """
    if RAG.faiss is None or RAG.bm25 is None or RAG.chunks is None:
        return "ERROR: Index not loaded. Call rag_load_index or rag_build_index first."

    chunks = RAG.chunks

    # Lexical (BM25)
    qtok = _tokenize(query)
    bm25_scores = RAG.bm25.get_scores(qtok)
    lex_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_lex]

    # Dense (FAISS): returns Documents; map back to chunk_id
    dense_docs = RAG.faiss.similarity_search(query, k=k_dense)
    dense_ranked: List[int] = []
    for d in dense_docs:
        cid = d.metadata.get("chunk_id")
        if isinstance(cid, int):
            dense_ranked.append(cid)

    # Fuse
    fused = _rrf_fuse([lex_ranked, dense_ranked], k=60)
    candidates = fused[: max(k_final * 4, 20)]

    # Optional rerank
    final_ids = candidates
    if use_rerank and RAG.reranker is not None:
        pairs = [(query, chunks[i].page_content) for i in candidates]
        scores = RAG.reranker.predict(pairs)
        final_ids = [i for i, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)]

    final_ids = final_ids[:k_final]

    out = []
    for rank, i in enumerate(final_ids, start=1):
        d = chunks[i]
        meta = d.metadata or {}
        out.append({
            "rank": rank,
            "chunk_id": meta.get("chunk_id", i),
            "content": d.page_content,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
        })

    return json.dumps(out, ensure_ascii=False)


def rag_answer(query: str, llm, k_final: int = 8) -> str:
    """
    RAG answer using rag_search() for context, then generates an answer with citations [S1], [S2], ...
    """
    raw = rag_search(query, k_final=k_final, use_rerank=True)
    if raw.startswith("ERROR:"):
        return raw

    passages = json.loads(raw)

    context_blocks = []
    cites = []
    for i, p in enumerate(passages, start=1):
        page = p.get("page")
        cite = f"[S{i}] {p.get('source','unknown')}" + (f" p.{page+1}" if isinstance(page, int) else "")
        cites.append(cite)
        context_blocks.append(f"{cite}\n{p['content']}")

    system = (
        "You are a careful document analyst.\n"
        "Use ONLY the provided context.\n"
        "Do NOT speculate or infer guilt.\n"
        "If the answer is not present, say you don't have enough information.\n"
        "Every factual claim must include citations like [S1], [S2]."
    )

    from langchain_core.messages import SystemMessage, HumanMessage
    msg = [
        SystemMessage(content=system),
        HumanMessage(content=f"Question:\n{query}\n\nContext:\n\n" + "\n\n---\n\n".join(context_blocks)),
    ]
    resp = llm.invoke(msg)

    return f"{resp.content}\n\nCitations:\n" + "\n".join(cites)
