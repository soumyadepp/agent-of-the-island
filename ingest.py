# ingest.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple

from tools import rag_build_index, rag_load_index, rag_init_reranker


def collect_files(
    data_dir: str,
    exts: Tuple[str, ...] = (".pdf", ".txt"),
) -> List[str]:
    """Collects files from a directory with allowed extensions.

    Args:
        data_dir (str): Directory path containing DOJ files.
        exts (Tuple[str, ...]): File extensions to include (e.g. (".pdf", ".txt")).
        recursive (bool): If True, scans subdirectories recursively.

    Returns:
        List[str]: Sorted list of absolute/relative file paths.
    """

    p = data_dir
    print(p)

    if not os.path.exists(p):
        raise RuntimeError(f"DATA_DIR is not a valid directory: {data_dir}")

    files = [
        p + "/" + fn for fn in os.listdir(p) 
        if fn.lower().endswith(exts) and os.path.isfile(os.path.join(p, fn))
    ]
    print(files[:5])
    return sorted(files)


def ensure_index_from_dir(
    data_dir: str,
    index_dir: str,
    embeddings,
    exts: Tuple[str, ...] = (".pdf", ".txt"),
) -> str:
    """Loads index if present, otherwise builds it from files found in data_dir.

    Args:
        data_dir (str): Directory containing PDFs/TXTs.
        index_dir (str): Directory where FAISS + bm25.pkl are stored.
        embeddings: LangChain embeddings instance used for FAISS build/load.
        recursive (bool): Whether to scan subdirectories.
        exts (Tuple[str, ...]): Extensions to ingest.

    Returns:
        str: Status message from load/build operation.
    """
    idx = index_dir
    bm25_path = os.path.join(idx, "bm25.pkl")
    faiss_exists = os.path.exists(idx) and any(fn.endswith(".faiss") for fn in os.listdir(idx))

    if faiss_exists and os.path.exists(bm25_path):
        return rag_load_index(index_dir=index_dir, 
                              embeddings=embeddings)

    paths = collect_files(data_dir, 
                          exts=exts)
    if not paths:
        raise RuntimeError(f"No files found in '{data_dir}' with extensions: {exts}")

    return rag_build_index(json.dumps(paths), 
                           index_dir=index_dir, 
                           embeddings=embeddings)


def init_reranker(model_name: str) -> str:
    """Initializes local reranker model.

    Args:
        model_name (str): HuggingFace cross-encoder model id.

    Returns:
        str: Status message.
    """
    return rag_init_reranker(model_name=model_name)



def build_index(paths: List[str], index_dir: str, embeddings) -> str:
    return rag_build_index(json.dumps(paths), 
                           index_dir=index_dir, 
                           embeddings=embeddings)


def load_index(index_dir: str, embeddings) -> str:
    return rag_load_index(index_dir=index_dir,
                           embeddings=embeddings)
