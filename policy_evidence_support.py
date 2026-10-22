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

POLICY_SUPPORT_VERSION = "lexical-support-v1"
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

def _norm(value):
    return unicodedata.normalize("NFKC", str(value)).lower()

def assess_policy_support(question: str, evidence: list[dict]) -> dict:
    """Return auditable lexical coverage, never a refund/eligibility decision.

    Chinese focus spans need at least half of their adjacent bigrams present;
    each span is checked independently so an earphone mention cannot hide an
    uncovered mouse/other subject. This is a *different feature* from rerank
    similarity, not a raised global rerank cutoff. Source names are not evidence.
    """
    body = _norm("\n".join(str(e.get("content") or "") for e in evidence))
    compact = re.sub(r"\s+", "", body)
    text = _norm(question)
    missing = [word for word in _PROMISES if word in text and word not in body]
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
