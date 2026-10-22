"""Import the demo policy documents through the existing RAG upload API.

Run inside the policy-rag container after clearing the old index:
    python scripts/import_policy_docs.py
"""

import os
from pathlib import Path

import requests


DOCS_DIR = Path(__file__).resolve().parent.parent / "policy_docs"
EXPECTED_FILES = (
    "退货退款政策.txt",
    "质量问题售后规则.txt",
    "电子产品售后规则.txt",
    "耳机类商品规则.txt",
    "换货流程.txt",
    "保修政策.txt",
    "退款到账规则.txt",
)
BASE_URL = os.getenv("POLICY_RAG_IMPORT_URL", "http://127.0.0.1:8000").rstrip("/")


def get_json(session: requests.Session, path: str) -> dict:
    response = session.get(f"{BASE_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def main() -> None:
    missing = [name for name in EXPECTED_FILES if not (DOCS_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing policy documents: {missing}")

    with requests.Session() as session:
        status = get_json(session, "/rag_status")
        records = get_json(session, "/documents")["data"]
        recorded_chunks = sum(int(record["文本块数量"]) for record in records)
        existing_names = {record["文件名"] for record in records}
        if existing_names - set(EXPECTED_FILES) or status["index_count"] != recorded_chunks:
            raise RuntimeError("Existing RAG index and document records are not the expected policy set")

        for name in EXPECTED_FILES:
            path = DOCS_DIR / name
            with path.open("rb") as source:
                response = session.post(
                    f"{BASE_URL}/upload",
                    files={"file": (name, source, "text/plain")},
                    timeout=180,
                )
            response.raise_for_status()
            result = response.json()
            if not result.get("success"):
                raise RuntimeError(f"Import failed for {name}: {result.get('message')}")
            print(
                f"{name}: chunks={result.get('chunk_count', 'existing')}, "
                f"duplicate={result.get('is_duplicate', False)}"
            )

        status = get_json(session, "/rag_status")
        records = get_json(session, "/documents")["data"]
        total_chunks = sum(int(record["文本块数量"]) for record in records)
        if len(records) != len(EXPECTED_FILES) or status["index_count"] != total_chunks:
            raise RuntimeError("RAG document records and FAISS index are inconsistent after import")
        print(f"Imported documents={len(records)}, chunks={total_chunks}, index={status['index_count']}")


if __name__ == "__main__":
    main()
