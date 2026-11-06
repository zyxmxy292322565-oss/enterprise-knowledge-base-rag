"""Conservative lexical support guard, NOT an entailment or eligibility engine.

Reranker scores measure ranking, not the truth of a policy promise. Before
offering excerpts for explanation, require the question's remaining subject /
requested details to be grounded in those excerpts. Unknown wording is refused
rather than supplied from a generic returns paragraph. This deliberately trades
some recall for safety; it cannot prove semantic entailment or resolve policy
conflicts. Order/qualification checks remain exclusively in the business engine.
"""
import re
import unicodedata
from collections import defaultdict

POLICY_SUPPORT_VERSION = "lexical-support-v2"
POLICY_EVIDENCE_CACHE_VERSION = "evidence-version-v2:" + POLICY_SUPPORT_VERSION

# Only query scaffolding is removed, not arbitrary product nouns or promises.
# Keep scoped negatives and substantive conditions in the evidence for the
# grounded answer generator; this guard never declares them fulfilled.
_SCAFFOLD = (
    "无理由退货", "退货退款", "质量问题", "售后政策", "退货政策", "保修政策",
    "政策咨询", "申请期限", "适用条件", "签收后", "类商品", "疑似", "申请",
    "期限", "条件", "例外", "政策", "规则", "售后", "退货", "退款", "换货",
    "维修", "保修", "故障", "问题", "商品", "产品", "咨询", "请问", "请介绍",
    "请说明", "是什么", "分别", "多少", "多久", "如何", "怎么", "是否", "可以",
    "能否", "适用", "套用", "支持", "签收", "到货", "用了", "已经", "需要",
    "注意", "哪些", "什么", "时候", "几天", "几日", "几年", "不喜欢", "无理由", "原因", "时间", "日期",
    "说明", "介绍", "退吗", "还能", "和", "与", "及", "的", "吗", "呢", "因",
    "return_refund", "exchange", "repair", "no_reason", "quality",
)
_PROMISES = ("终身", "永久", "无条件", "必定", "百分之百", "全额", "运费", "立即到账", "秒到账")
_CODE = re.compile(r"(?<![a-z0-9-])[a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d[a-z0-9]*(?![a-z0-9-])")
_CATEGORY = re.compile(r"(?:适用于|适用范围\s*[:：]\s*)([\u4e00-\u9fff]{2,12})类商品")
_QUERY_CATEGORY = re.compile(r"([\u4e00-\u9fff]{2,12})类商品")
_CALENDAR_YEAR = re.compile(r"(?<!\d)(\d{4})\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*[日号])?)?")
_FORMATS = re.compile(r"(?<![a-z0-9])(?:txt|pdf)(?![a-z0-9])")
_COMPARE = re.compile(r"(?:txt|pdf)(?:的(?:演示)?政策)?(?:和|与|及)(?:pdf|txt)(?:的政策)?是否(?:明确)?针对同一商品(?:给出|有)?冲突规定[?？。]*")
_ARRIVAL_ASK = re.compile(r"(?:到账[^，。？?]{0,12}(?:多久|多少|几天|几小时|时间)|(?:多久|多少(?:个)?小时|几天|最迟|何时|几小时)[^，。？?]{0,12}到账)")
_AMOUNT_ASK = re.compile(r"(?:退款|赔偿|补偿)[^，。？?]{0,10}(?:多少(?:钱|元|金额)|金额(?:是|为)?多少|几元)")
_ARRIVAL_VALUE = re.compile(r"\d+(?:\.\d+)?\s*(?:个?工作日|天|小时|分钟)")
_AMOUNT_VALUE = re.compile(r"\d+(?:\.\d+)?\s*元")

def _norm(value):
    return unicodedata.normalize("NFKC", str(value)).lower()


def _wording(value):
    for source, target in (
        ("不承诺退款必定获批", "不保证获批"),
        ("refund application is prohibited", "禁止申请退款"),
        ("verified defect", "已核验缺陷"),
        ("持有购买凭证", "有购买凭证"),
    ):
        value = value.replace(source, target)
    return value


def missing_requested_value(question, evidence):
    """Treat a statement that a value is unspecified as refusal evidence."""
    question = _norm(question)
    for pattern, topic, value in (
        (_ARRIVAL_ASK, "到账", _ARRIVAL_VALUE),
        (_AMOUNT_ASK, "退款|赔偿|补偿", _AMOUNT_VALUE),
    ):
        if not pattern.search(question):
            continue
        text = "\n".join(_norm(row.get("content") or row.get("quote") or "") for row in evidence)
        clauses = [part for part in re.split(r"[。；;!?！？\n]", text) if re.search(topic, part)]
        if not any(value.search(part) and not re.search(r"未规定|未约定|不承诺|无明确|未明确", part)
                   for part in clauses):
            return "requested_value_not_in_applicable_evidence"
    return None


def select_policy_evidence(question, evidence, limit, *, product=None, category=None):
    """Select source versions using literal identifiers and scope declarations."""
    text = _norm(question).strip()
    years = set(_CALENDAR_YEAR.findall(text))
    formats = set(_FORMATS.findall(text))
    codes = set(_CODE.findall(text)) | set(_CODE.findall(_norm(product or "")))
    categories = {re.sub(r"^(?:这份|按这份|该|这款|这个)", "", item)
                  for item in _QUERY_CATEGORY.findall(_CALENDAR_YEAR.sub(" ", text))}
    if category:
        categories.add(_norm(category).removesuffix("类商品"))
    decision = {"status": "unscoped", "retained": [], "rejected": {}}
    if not (formats or codes or categories):
        return evidence[:limit], decision

    by_version = defaultdict(list)
    for row in evidence:
        if row.get("document_version_id"):
            by_version[row["document_version_id"]].append(row)
    eligible = []
    for rows in by_version.values():
        combined = "\n".join(_norm(row["content"]) for row in rows)
        file_format = _norm(rows[0].get("source") or "").rsplit(".", 1)[-1]
        declared = set(_CATEGORY.findall(combined))
        identifiers = set(_CODE.findall(combined))
        if ((not formats or file_format in formats) and codes <= identifiers and categories <= declared
                and years <= set(_CALENDAR_YEAR.findall(combined))):
            eligible.append((rows, file_format, declared, identifiers))

    selected = []
    if len(formats) == 2 and _COMPARE.fullmatch(re.sub(r"\s+", "", text)):
        # Only compare two unambiguously specified sources. Differing literal
        # labels permit reporting the lack of a demonstrated common scope.
        # They do not prove that the scopes are disjoint.
        if limit >= 2 and len(eligible) == 2 and {item[1] for item in eligible} == formats:
            anchors = []
            for rows, _, declared, identifiers in eligible:
                if not any("退款" in _wording(_norm(row["content"])) for row in rows):
                    continue
                anchor = next((row for row in rows if _CATEGORY.search(_norm(row["content"]))), None) if declared else (
                    next((row for row in rows if _CODE.search(_norm(row["content"]))), None) if identifiers else None)
                if anchor:
                    anchors.append(anchor)
            if len(anchors) == 2:
                selected = anchors
                decision["status"] = "scope_comparison"
    elif len(eligible) == 1:
        rows = eligible[0][0]
        normalized = _wording(text)
        topic = "退款" if "退款" in normalized else "退货" if "退货" in normalized else None
        if not topic or any(topic in _wording(_norm(row["content"])) for row in rows):
            selected = rows[:limit]
            if categories or codes:
                anchor = next((row for row in rows if
                               (categories and categories <= set(_CATEGORY.findall(_norm(row["content"])))) or
                               (codes and codes <= set(_CODE.findall(_norm(row["content"]))))), None)
                if anchor is not None and anchor not in selected:
                    selected = selected[:-1] + [anchor]
            decision["status"] = "scoped"

    if not selected:
        decision["status"] = "scope_unresolved"
    decision["retained"] = [row["child_id"] for row in selected]
    retained = set(decision["retained"])
    matching = {row["child_id"] for rows, *_ in eligible for row in rows}
    decision["rejected"] = {row["child_id"]: ("selection_limit" if row["child_id"] in matching and selected
                                               else "unconfirmed_scope_or_version")
                            for row in evidence if row.get("child_id") and row["child_id"] not in retained}
    return selected, decision


def assess_policy_support(question: str, evidence: list[dict], *, scope_decision=None) -> dict:
    """Return auditable lexical coverage, never a refund/eligibility decision.

    Chinese focus spans need at least half of their adjacent bigrams present;
    each span is checked independently so an earphone mention cannot hide an
    uncovered mouse/other subject. This is a *different feature* from rerank
    similarity, not a raised global rerank cutoff. Source names are not evidence.
    """
    body = _wording(_norm("\n".join(str(e.get("content") or "") for e in evidence)))
    compact = re.sub(r"\s+", "", body)
    text = _wording(_norm(question))
    status = (scope_decision or {}).get("status")
    if status == "scoped":
        # Media type and full identifiers have already been verified separately.
        text = _CALENDAR_YEAR.sub(" ", _CODE.sub(" ", _FORMATS.sub(" ", text)))
        text = re.sub(r"按这份|这份|对(?=\s)|规定", " ", text)
    if status == "scope_comparison":
        text = ""
    missing = [word for word in _PROMISES if word in text and word not in body]
    if gap := missing_requested_value(question, evidence):
        missing.append(gap)
    # ASCII model suffixes/numbers carry no category grant; the brand/family
    # still must occur in actual evidence (e.g. AirPods Pro 2 -> AirPods).
    text = re.sub(r"\d+(?:\.\d+)?\s*(?:个?工作日|天|日|年|个月|小时)", " ", text)
    for phrase in sorted(_SCAFFOLD, key=len, reverse=True):
        text = text.replace(phrase, " ")
    focus = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z][a-z0-9_]*", text)
    checks = []
    for term in focus:
        if term in {"pro", "max", "mini", "plus"}:
            continue
        if re.fullmatch(r"[a-z][a-z0-9_]*", term):
            covered = bool(re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", body))
            ratio = 1.0 if covered else 0.0
        else:
            grams = {term[i:i+2] for i in range(len(term)-1)}
            ratio = sum(g in compact for g in grams) / len(grams)
            covered = ratio >= 0.5
        checks.append({"term": term, "coverage": round(ratio, 4), "covered": covered})
        if not covered:
            missing.append(term)
    allowed = bool(body.strip()) and not missing
    return {
        "method": POLICY_SUPPORT_VERSION,
        "support_status": "CONDITIONAL_EXPLANATION" if allowed else "INSUFFICIENT",
        "sufficient_for_answer": allowed,
        "uncovered_terms": list(dict.fromkeys(missing)),
        "focus_checks": checks,
        "limitation": "Lexical coverage is not semantic entailment or approval; all stated scope/conditions still apply.",
    }
