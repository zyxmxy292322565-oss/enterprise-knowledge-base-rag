"""Read-only manifests and serialization helpers for reproducible RAG evaluation."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


TRACKED_PACKAGES = (
    "faiss-cpu",
    "sentence-transformers",
    "langchain-openai",
    "fastapi",
    "pymysql",
    "redis",
    "numpy",
)

EXCLUDED_SOURCE_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    "cache",
    "venv",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_SOURCE_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    for name in ("requirements.txt", "docker-compose.yml", "Dockerfile"):
        path = root / name
        if path.is_file():
            files.append(path)
    return sorted(set(files), key=lambda item: item.as_posix())


def source_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _source_files(root)
    }
    git_commit = None
    git_note = "Git metadata not found; source_files_sha256 is the reproducibility identifier"
    git_head = root / ".git" / "HEAD"
    if git_head.is_file():
        head = git_head.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref_path = root / ".git" / head.split(":", 1)[1].strip()
            if ref_path.is_file():
                git_commit = ref_path.read_text(encoding="utf-8").strip() or None
        elif head:
            git_commit = head
        if git_commit:
            git_note = "Git commit read from local metadata; uncommitted changes are represented by source_files_sha256"
    return {
        "git_commit": git_commit,
        "git_note": git_note,
        "source_files": files,
        "source_files_sha256": sha256_text(canonical_json(files)),
    }


def dataset_manifest(cases: Iterable[dict[str, Any]], dataset_version: str) -> dict[str, Any]:
    rows = []
    for case in cases:
        rows.append({
            "case_id": case.get("ID", case.get("id")),
            "question": case.get("问题", case.get("question", "")),
            "expected_answer": case.get("标准答案", case.get("expected_answer", "")),
            "expected_keywords": case.get("期望关键词", case.get("expected_keywords", "")),
        })
    rows.sort(key=lambda item: str(item.get("case_id")))
    return {
        "dataset_version": dataset_version or "unversioned",
        "case_count": len(rows),
        "dataset_manifest_sha256": sha256_text(canonical_json(rows)),
    }


def runtime_manifest(
    *,
    app_state: Any,
    prompt_callable: Any,
    source_root: Path,
    embedding_model: str | None,
    rerank_model: str | None,
    llm_model: str | None,
) -> dict[str, Any]:
    source = source_manifest(source_root)

    documents = sorted({
        (
            doc.metadata.get("source") or "",
            doc.metadata.get("file_hash") or "",
            doc.metadata.get("version", doc.metadata.get("policy_version")),
        )
        for doc in (getattr(app_state, "all_docs", None) or [])
    }, key=lambda item: tuple("" if value is None else str(value) for value in item))
    document_rows = [
        {"source": name, "file_hash": file_hash, "version": version}
        for name, file_hash, version in documents
    ]

    index_files: dict[str, dict[str, Any]] = {}
    for attribute in ("vectors_cache_file", "all_docs_cache_file", "index_cache_file"):
        raw_path = getattr(app_state, attribute, None)
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_file():
            index_files[attribute] = {
                "file_name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    prompt_code = inspect.getsource(prompt_callable)
    return {
        **source,
        "document_manifest": document_rows,
        "document_manifest_sha256": sha256_text(canonical_json(document_rows)),
        "document_count": len(document_rows),
        "chunk_count": len(getattr(app_state, "all_docs", None) or []),
        "index_files": index_files,
        "index_manifest_sha256": sha256_text(canonical_json(index_files)),
        "embedding_model": embedding_model,
        "rerank_model": rerank_model,
        "llm_model": llm_model,
        "model_revision": None,
        "model_revision_note": "Model weight revisions and remote provider model revision are not verified",
        "llm_temperature": getattr(getattr(app_state, "chat_model", None), "temperature", None),
        "prompt_code_sha256": sha256_text(prompt_code),
        "python_version": platform.python_version(),
        "package_versions": packages,
        "platform": platform.platform(),
    }


def split_trace_artifacts(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep large Context/Prompt text out of the ordinary case-run JSON."""
    compact = dict(trace or {})
    compact_candidates = []
    for candidate in compact.get("faiss_candidates", []) or []:
        item = dict(candidate)
        content = item.pop("content", None)
        if content is not None:
            item["content_sha256"] = sha256_text(content)
            item["content_chars"] = len(content)
        compact_candidates.append(item)
    if "faiss_candidates" in compact:
        compact["faiss_candidates"] = compact_candidates
    context = compact.pop("context", None)
    prompt = compact.pop("prompt", None)
    artifacts = {
        "context_text": context,
        "prompt_text": prompt,
        "context_sha256": sha256_text(context) if context is not None else None,
        "prompt_sha256": sha256_text(prompt) if prompt is not None else None,
    }
    compact["context_sha256"] = artifacts["context_sha256"]
    compact["prompt_sha256"] = artifacts["prompt_sha256"]
    return compact, artifacts


def cache_status(cache_mode: str | None, from_cache: bool | None) -> str:
    if cache_mode == "bypass":
        return "bypass"
    if cache_mode == "normal" and from_cache is True:
        return "hit"
    if cache_mode == "normal" and from_cache is False:
        return "miss"
    return "unknown"
