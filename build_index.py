"""
Build the RAG index from Refund_policy.pdf.

Run once (and again whenever the PDF changes):
    python build_index.py
"""
import re
from pathlib import Path

from pypdf import PdfReader

BASE = Path(__file__).parent
PDF_PATH = BASE / "Refund_policy.pdf"
DB_DIR = BASE / "policy_db"
COLLECTION = "luxury_bags_policy"


def load_text() -> str:
    reader = PdfReader(str(PDF_PATH))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def split_long(text: str, max_chars: int) -> list[str]:
    """Split an over-long section into sentence-based pieces."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > max_chars:
            pieces.append(current.strip())
            current = ""
        current += s + " "
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """One chunk per numbered section ("1. Time Limits ...", "2. Damaged ...")."""
    text = re.sub(r"[ \t]+", " ", text)
    parts = re.split(r"(?m)^(?=\d\.\s+[A-Z])", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) < 40:
            continue
        if len(part) <= max_chars:
            chunks.append(part)
        else:
            chunks.extend(split_long(part, max_chars))
    return chunks


def main():
    import chromadb

    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    chunks = chunk_text(load_text())
    if not chunks:
        raise SystemExit("No text could be extracted from the PDF.")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"section": c.split("\n", 1)[0][:80]} for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks from {PDF_PATH.name} into {DB_DIR}")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c.splitlines()[0][:70]}")


if __name__ == "__main__":
    main()
