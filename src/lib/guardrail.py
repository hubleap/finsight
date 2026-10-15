"""FinSight input guardrail.

Tiers (checked in order):
  1. HARD   always blocked: prompt injection, financial-crime how-to, partisan politics, violence/weapons
  2. SOFT   blocked only when the message has no finance context (e.g. "government", "attack")
  3. SCOPE  optional: require at least one FinSight topic (strict mode)

Tax, VAT, regulation, law, bills, duty, CBN/CAC/FIRS, scams and Ponzi checks are IN scope for an SME
finance product, so they are deliberately not blocked.
Regex covers English only; Pidgin/Yoruba/Hausa/Igbo should also be handled by the LLM system prompt.
"""
import re
import unicodedata
from dataclasses import dataclass

_F = re.IGNORECASE


def _c(*pats):
    return [re.compile(p, _F) for p in pats]


# ------------------------------------------------------------------ 1. HARD
HARD = {
    "prompt_injection": _c(
        r"\bignore (all |any )?(the )?(previous|prior|above|earlier|your) (instructions|rules|prompts?)\b",
        r"\b(reveal|show|print|repeat|leak) (me )?(your |the )?(system|hidden|initial) (prompt|instructions)\b",
        r"\b(jailbreak|developer mode|do anything now|\bDAN\b)\b",
        r"\byou are (now|no longer) (a|an|the)\b",
        r"\b(disregard|override|bypass) (your |the )?(guardrails?|safety|rules|restrictions)\b",
    ),
    "financial_crime": _c(
        r"\b(how (do|can|would) (i|we|you)|how to|help me|teach me|show me|steps to|ways to)\b.{0,50}"
        r"\b(launder|evade tax|dodge tax|forge|counterfeit|fake (an? )?(invoice|receipt|statement|alert)|"
        r"bypass (kyc|aml|verification)|hide (money|income|profit|transactions?) from|"
        r"run (a )?(ponzi|pyramid)|scam (people|customers|investors))\b",
        r"\bmake (it )?(look|appear) (like )?(legit|legitimate|real)\b.{0,30}\b(transaction|payment|invoice)\b",
    ),
    "partisan_politics": _c(
        r"\b(who|which (party|candidate)).{0,30}\b(should|to) (i |we )?(vote|support)\b",
        r"\b(election|electoral|ballot|voter|voting)\b",
        r"\b(political|presidential|gubernatorial|senatorial) (campaign|party|candidate|rally|aspirant)s?\b",
        r"\b(apc|pdp|nnpp|labour party|inec|democrat|republican)\b",
        r"\b(trump|biden|obama|clinton|bola tinubu|peter obi|atiku abubakar|muhammadu buhari)\b",
        r"\b(impeach(ment)?|coup|insurrection|sedition|treason|referendum)\b",
        r"\b(communis[mt]|fascis[mt]|marxis[mt]|nazi|jihad)\b",
    ),
    "violence": _c(
        r"\b(bomb(ing)?|terroris[mt]|assassinat\w*|kidnap\w*|massacre|genocide)\b",
        r"\b(make|build|buy|sell) (a |an )?(gun|firearm|explosive|weapon)s?\b",
    ),
}

# ------------------------------------------------------------------ 2. SOFT
SOFT = _c(
    r"\b(government|govt|politician|parliament|senate|senator|congress|policy|policies)\b",
    r"\b(war|conflict|attack|riot|protest|military|army|propaganda|conspiracy)\b",
    r"\b(poll|polls|scandal|debate|partisan|controversy)\b",
)

# Finance context rescues SOFT terms ("fuel subsidy removal hurt my margins")
FINANCE_CONTEXT = _c(
    r"\b(revenues?|incomes?|expenses?|cash ?flows?|profits?|loss(es)?|margins?|sales|transactions?|invoices?|payments?|"
    r"payroll|salar(y|ies)|rent|inventory|stock|budget|loan|debt|credit|bank|pos|ussd|transfer|naira|ngn)\b",
    r"[₦]|\bnaira\b",
    r"\b(sme|business|shop|store|customers?|suppliers?|vendors?|counterpart(y|ies)|wallet|receipts?)\b",
    r"\b(tax|vat|paye|firs|cbn|cac|kyc|aml|sec nigeria)\b",
    r"\b(fraud|scam|ponzi|phishing|anomal(y|ies)|suspicious|duplicate|chargeback|failed)\b",
)

# ------------------------------------------------------------------ 3. SCOPE (strict mode)
SCOPE = FINANCE_CONTEXT + _c(
    r"\b(forecast|insight|report|dashboard|trend|summary|compare|category|categories|month|quarter|year)\b",
    r"\b(upload|statement|document|screenshot|image|investment offer|doubling|guaranteed returns?)\b",
)

GREETING = re.compile(r"^\s*(hi|hello|hey|thanks?|thank you|ok(ay)?|yes|no|help|good (morning|afternoon|evening))\W*$", _F)


@dataclass
class GuardResult:
    allowed: bool
    category: str | None = None
    reason: str | None = None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)   # zero-width tricks
    return re.sub(r"\s+", " ", text).strip()


def check(text: str, strict: bool = False) -> GuardResult:
    t = _normalize(text)
    if not t:
        return GuardResult(False, "empty", "Empty message.")

    for cat, pats in HARD.items():
        if any(p.search(t) for p in pats):
            return GuardResult(False, cat, MESSAGES[cat])

    has_finance = any(p.search(t) for p in FINANCE_CONTEXT)
    if not has_finance and any(p.search(t) for p in SOFT):
        return GuardResult(False, "off_topic_sensitive", MESSAGES["off_topic_sensitive"])

    if strict and not GREETING.match(t) and not any(p.search(t) for p in SCOPE):
        return GuardResult(False, "out_of_scope", MESSAGES["out_of_scope"])

    return GuardResult(True)


MESSAGES = {
    "prompt_injection": "I can't change my instructions. Ask me about your business finances.",
    "financial_crime": "I can't help with evading tax, forging records or hiding money. "
                       "I can help you spot and prevent fraud instead.",
    "partisan_politics": "I don't discuss politics or elections. I can help with your cash flow, "
                         "expenses, revenue and unusual transactions.",
    "violence": "I can't help with that. I can analyse your business transactions and risks.",
    "off_topic_sensitive": "That's outside what I cover. I can help with your revenue, expenses, "
                           "cash flow, anomalies and fraud checks.",
    "out_of_scope": "I focus on SME finances: transactions, cash flow, expenses, revenue, forecasts "
                    "and scam checks. What would you like to look at?",
}