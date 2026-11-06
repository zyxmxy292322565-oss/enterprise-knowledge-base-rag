"""S1-A document schemas and original evidence mapping."""
from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Literal
from unicodedata import normalize

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval_center_metrics import covered_span
from knowledge_v2 import KnowledgeError, read_checked
from retrieval_engine import ChatTurn

ACTION = {"answerable": "answer", "unanswerable": "abstain", "needs_clarification": "clarify",
          "policy_conflict": "conflict"}
REVIEW = Literal["unreviewed", "proposed_unreviewed", "developer_self_reviewed",
                 "independently_reviewed", "disputed", "adjudicated"]


class Fact(BaseModel):
    fact_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    purpose: Literal["answer", "rationale"]
    evidence_options: list[list[str]] = Field(default_factory=list)


class Question(BaseModel):
    model_config = ConfigDict(extra="allow")
    question_id: str = Field(min_length=1, max_length=120)
    question_revision: int = Field(default=1, ge=1)
    question: str = Field(min_length=1, max_length=10000)
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=8)
    question_type: str = Field(min_length=1)
    answerability: Literal["answerable", "unanswerable", "needs_clarification", "policy_conflict"]
    expected_action: Literal["answer", "abstain", "clarify", "conflict"]
    required_facts: list[Fact] = Field(default_factory=list)
    forbidden_claims: list[dict[str, str]] = Field(default_factory=list)
    gold_sources: list[dict] = Field(default_factory=list)
    required_evidence_groups: list[dict] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"]
    review_status: REVIEW = "unreviewed"

    @model_validator(mode="after")
    def valid(self):
        if ACTION[self.answerability] != self.expected_action or not self.question.strip():
            raise ValueError("Invalid question or action")
        if len({f.fact_id for f in self.required_facts}) != len(self.required_facts):
            raise ValueError("Duplicate fact_id")
        for group in self.required_evidence_groups:
            if not group.get("group_id") or not isinstance(group.get("options"), list) or any(
                not isinstance(option, list) or not option or any(not isinstance(key, str) for key in option)
                for option in group["options"]
            ):
                raise ValueError("Invalid evidence group")
        return self


class Offset(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    unit: Literal["unicode_code_point"] = "unicode_code_point"
    coordinate_space: Literal["decoded_source", "extracted_page"] = "decoded_source"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")
    evidence_id: str = Field(min_length=1, max_length=120)
    evidence_revision: int = Field(default=1, ge=1)
    question_id: str = Field(min_length=1, max_length=120)
    source_id: str
    source_path: str
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_text_quote: str = Field(min_length=1, max_length=20000)
    normalized_text_quote: str | None = None
    normalization_profile: Literal["anchor-nfc-lf-v1"] = "anchor-nfc-lf-v1"
    media_type: Literal["text/plain", "application/pdf"] = "text/plain"
    source_offset: Offset | None = None
    page_number: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    occurrence_index: int | None = Field(default=None, ge=1)
    role: Literal["answer_support", "decision_rationale", "context", "hard_negative", "conflict_side"]
    necessity: Literal["required", "supplemental", "diagnostic", "alternative"] = "supplemental"
    supports_fact_ids: list[str] = Field(default_factory=list)
    exception_conditions: list[str] = Field(default_factory=list)
    negation: bool = False
    time_constraints: list[str] = Field(default_factory=list)
    applicability: str = ""
    review_status: REVIEW = "proposed_unreviewed"

    @model_validator(mode="after")
    def coordinates(self):
        if (self.media_type == "application/pdf") != (self.page_number is not None):
            raise ValueError("PDF anchor requires page_number")
        if self.source_offset and (
            self.source_offset.end <= self.source_offset.start
            or self.source_offset.coordinate_space != ("extracted_page" if self.page_number else "decoded_source")
        ):
            raise ValueError("Invalid source offset")
        if self.line_end is not None and (self.line_start is None or self.line_end < self.line_start):
            raise ValueError("Invalid line interval")
        return self


def normalized(value):
    return normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def map_anchor(store, view, anchor):
    result = {"status": "needs_review", "index_snapshot_id": view.snapshot_id,
              "evidence_revision": anchor["evidence_revision"], "bindings": []}
    matches = [m for m in view.manifest["members"] if m["source"] == anchor["source_path"]
               and m["content_hash"] == anchor["source_content_hash"]]
    if len(matches) != 1:
        return result | {"reason": "source identity/hash not present"}
    member = matches[0]
    version = member["document_version_id"]
    result.update(document_id=member["document_id"], document_version_id=version)
    try:
        read_checked(store.root / "documents" / version / "source.bin", member["content_hash"])
        artifact = store.document_artifact(version)
    except (KnowledgeError, OSError):
        return result | {"reason": "original or processed artifact failed verification"}
    page = anchor["page_number"]
    raw = next((row["text"] for row in artifact["pages"] if row["page"] == page), None)
    if raw is None:
        return result | {"reason": "page not present"}
    quote = anchor["original_text_quote"]
    offset = anchor["source_offset"]
    if offset:
        start, end = offset["start"], offset["end"]
        if end > len(raw) or normalized(raw[start:end]) != normalized(quote):
            return result | {"reason": "quote and source offset disagree"}
    else:
        occurrences = [(m.start(), m.end()) for m in re.finditer(rf"(?={re.escape(quote)})", raw)]
        occurrences = [(start, start + len(quote)) for start, _ in occurrences]
        index = anchor["occurrence_index"]
        if index:
            occurrences = occurrences[index - 1:index]
        if anchor["line_start"]:
            occurrences = [(a, b) for a, b in occurrences
                           if len(re.findall(r"\r\n|\r|\n", raw[:a])) + 1 == anchor["line_start"]]
        if len(occurrences) != 1:
            return result | {"reason": "exact quote missing or ambiguous"}
        start, end = occurrences[0]
    if anchor["normalized_text_quote"] and normalized(quote) != anchor["normalized_text_quote"]:
        return result | {"reason": "normalized quote disagrees"}
    if any(row["page"] == page and row["start"] < end and row["end"] > start for row in artifact["removed"]):
        return result | {"reason": "processing removed part of the anchor"}
    spans = []
    for row in artifact["projections"]:
        left, right = max(start, row["raw_start"]), min(end, row["raw_end"])
        if row["page"] != page or left >= right:
            continue
        if row["raw_end"] - row["raw_start"] == row["clean_end"] - row["clean_start"]:
            spans.append((row["clean_start"] + left - row["raw_start"], row["clean_start"] + right - row["raw_start"]))
        else:
            spans.append((row["clean_start"], row["clean_end"]))
    if not spans:
        return result | {"reason": "anchor removed by processing"}
    clean_start, clean_end = min(a for a, _ in spans), max(b for _, b in spans)
    bindings = [dict(child_id=c["child_id"], parent_id=c["parent_id"], chunk_id=c["chunk_id"],
                     clean_start=max(c["clean_start"], clean_start), clean_end=min(c["clean_end"], clean_end))
                for c in view.manifest["children"] if c["document_version_id"] == version and c["page"] == page
                and c["clean_start"] < clean_end and c["clean_end"] > clean_start]
    if not covered_span(clean_start, clean_end, [(b["clean_start"], b["clean_end"]) for b in bindings]):
        return result | {"reason": "Child intervals do not cover the anchor", "bindings": bindings}
    return result | {"status": "mapped", "source_offset": {"start": start, "end": end},
                     "clean_start": clean_start, "clean_end": clean_end, "bindings": bindings}


def historical_dataset(raw):
    expected = "3475e076245c6aa628e6baab42f79497ba281817339caf9758711df9ec0695fe"
    if sha256(raw).hexdigest() != expected:
        raise KnowledgeError("Historical dataset hash mismatch")
    questions, evidence = [], []
    for row in map(json.loads, raw.splitlines()):
        qid = row["case_id"]
        answerability = {"clarification_needed": "needs_clarification", "needs_clarification": "needs_clarification",
                         "policy_conflict": "policy_conflict", "unanswerable": "unanswerable"}.get(
                             row["answerability"], "answerable")
        questions.append(Question(
            question_id=qid, question=row["question"], question_type=row["primary_question_type"],
            question_tags=row["question_types"], answerability=answerability, expected_action=ACTION[answerability],
            required_facts=[Fact(fact_id=f"F{i}", text=value,
                                 purpose="answer" if answerability == "answerable" else "rationale")
                            for i, value in enumerate(row["required_facts"], 1)],
            forbidden_claims=[{"claim_id": f"FC{i}", "text": value} for i, value in enumerate(row["forbidden_claims"], 1)],
            gold_sources=[{"source_path": path, "source_content_hash": checksum, "role": "context"}
                          for path, checksum in row["reference_file_hashes"].items()],
            difficulty=row["difficulty"], expected_behavior=row["expected_behavior"],
            reference_response=row["reference_answer"], provenance={"source_type": "historical_v3_development",
                "source_dataset_version": row["dataset_version"], "source_dataset_sha256": expected,
                "source_case_id": qid, "original_answerability": row["answerability"]},
        ).model_dump())
        for i, anchor in enumerate(row["evidence"], 1):
            evidence.append(Evidence(evidence_id=f"{qid}-E{i}", question_id=qid, source_id=anchor["source_id"],
                source_path=anchor["source_file"], source_content_hash=anchor["source_sha256"],
                original_text_quote=anchor["quote"], normalized_text_quote=normalized(anchor["quote"]),
                line_start=anchor["line"], line_end=anchor["line"], role="context",
                section=anchor["section"]).model_dump())
    return {"questions": questions, "evidence": evidence, "provenance": {"source_dataset_sha256": expected}}
