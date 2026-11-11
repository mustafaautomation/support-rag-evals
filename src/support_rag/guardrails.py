"""Rule checks for the support guardrails.

The replies below are copied verbatim from the synthetic help articles. The
pipeline returns them instead of letting the generator invent a policy.
"""

import re
from dataclasses import dataclass

from support_rag.textutil import split_sentences

PRIVACY_RESPONSE = (
    "I can't share another customer's data. Sableboard support will not reveal "
    "another customer's name, email, workspace, invoices, or tickets."
)
SECURITY_RESPONSE = (
    "I'm escalating this to a human security specialist. Chat cannot reset "
    "passwords, send backup codes, or verify an identity."
)
LEGAL_RESPONSE = (
    "I'm escalating this to a human specialist. Send legal notices to "
    "legal@sableboard.example. Chat cannot give legal advice or comment on lawsuits."
)
REFUND_EXCEPTION_RESPONSE = (
    "I can't promise a refund that is not written in the refund policy. A human "
    "billing specialist decides any exception, and the specialist can say no."
)
ACCOUNT_LOOKUP_RESPONSE = (
    "I can't see live account data. This assistant has no live account access and "
    "cannot see invoice numbers, balances, or refund status."
)
UNKNOWN_RESPONSE = "I don't know. That is not covered in the Sableboard help documents."

TEMPLATE_RESPONSES: tuple[str, ...] = (
    PRIVACY_RESPONSE,
    SECURITY_RESPONSE,
    LEGAL_RESPONSE,
    REFUND_EXCEPTION_RESPONSE,
    ACCOUNT_LOOKUP_RESPONSE,
    UNKNOWN_RESPONSE,
)

_OFFICIAL_EMAIL = re.compile(r"\b[\w.+-]+@sableboard\.example\b", re.IGNORECASE)
_OTHER_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_LEGAL = re.compile(
    r"\b(sue|lawsuit|subpoena|lawyer|attorney|court order|admit liability)\b",
    re.IGNORECASE,
)
_PRIVACY = re.compile(
    r"(another customer|other customer|someone else's|all customers|every customer|"
    r"their tickets|their workspace|other company)",
    re.IGNORECASE,
)
_SECURITY = re.compile(
    r"(hacked|compromised|attacker|set the password|disable two-factor|"
    r"send me the backup codes|email me the backup codes)",
    re.IGNORECASE,
)
_REFUND_EXCEPTION = re.compile(
    r"(ignore the policy|promise me|goodwill|partial refund|\$50|"
    r"refund me in full today)",
    re.IGNORECASE,
)
_ACCOUNT_LOOKUP = re.compile(
    r"(my invoice number|did my refund|my balance|refund already)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decision:
    category: str
    response: str | None


def template_sentences() -> frozenset[str]:
    sentences: set[str] = set()
    for response in TEMPLATE_RESPONSES:
        sentences.update(split_sentences(response))
    return frozenset(sentences)


def decide(question: str) -> Decision:
    """Classify a question. Order matters: legal beats a request for other customers' records."""
    if _LEGAL.search(question):
        return Decision("legal", LEGAL_RESPONSE)
    if _PRIVACY.search(question) or _has_other_email(question):
        return Decision("privacy", PRIVACY_RESPONSE)
    if _SECURITY.search(question):
        return Decision("security", SECURITY_RESPONSE)
    if _REFUND_EXCEPTION.search(question):
        return Decision("refund_exception", REFUND_EXCEPTION_RESPONSE)
    if _ACCOUNT_LOOKUP.search(question):
        return Decision("account_lookup", ACCOUNT_LOOKUP_RESPONSE)
    return Decision("answer", None)


def _has_other_email(question: str) -> bool:
    without_official = _OFFICIAL_EMAIL.sub(" ", question)
    return _OTHER_EMAIL.search(without_official) is not None


def classify_response(response: str) -> str:
    """Map a reply to the labelled behaviour tags used by the eval set."""
    text = response.strip()
    if text.startswith("I don't know"):
        return "unknown"
    if text.startswith("I'm escalating"):
        return "escalate"
    if text.startswith("I can't"):
        return "refuse"
    return "answer"
