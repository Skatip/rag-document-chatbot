from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from pypdf import PdfReader


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def load_documents(data_dir: str) -> List[Tuple[str, str]]:
    """Returns list of (source_name, full_text). Supports .txt and .pdf."""
    docs: List[Tuple[str, str]] = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if os.path.isdir(path):
            continue
        lower = filename.lower()
        if lower.endswith(".txt"):
            docs.append((filename, read_text_file(path)))
        elif lower.endswith(".pdf"):
            docs.append((filename, read_pdf_file(path)))
    return docs


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """Simple char-based chunking with overlap (starter-friendly)."""
    text = normalize_whitespace(text)
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm
