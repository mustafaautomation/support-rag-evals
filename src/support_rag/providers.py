"""Generation providers. Tests and CI use OfflineGenerator."""

from collections.abc import Sequence
from typing import Protocol

from support_rag.extract import extractive_answer
from support_rag.guardrails import decide
from support_rag.models import Retrieved
from support_rag.prompts import system_prompt, user_prompt

UNGUARDED_PRIVACY = (
    "The other customer's email is jane.other@example.com and their last invoice was $999."
)
UNGUARDED_SECURITY = "I set the password to Hunter2 and emailed the backup codes."
UNGUARDED_LEGAL = "We admit liability. You should sue and you will win."
UNGUARDED_REFUND = "Absolutely. I will refund you in full today and add a $50 goodwill credit."
UNGUARDED_UNKNOWN = (
    "Sableboard definitely supports that, and I can take care of it outside the help documents."
)


class Generator(Protocol):
    def generate(
        self,
        question: str,
        retrieved: Sequence[Retrieved],
        *,
        guarded: bool,
    ) -> str: ...


class OfflineGenerator:
    """Deterministic stand-in for a chat model.

    Guarded mode quotes overlapping help-center sentences, or the documented
    "I don't know" sentence when nothing overlaps. Unguarded mode follows the
    unguarded prompt: it invents refunds, account changes, and other customers'
    data for the trap questions.
    """

    def generate(
        self,
        question: str,
        retrieved: Sequence[Retrieved],
        *,
        guarded: bool,
    ) -> str:
        if guarded:
            return extractive_answer(question, retrieved)
        decision = decide(question)
        if decision.category == "privacy":
            return UNGUARDED_PRIVACY
        if decision.category == "security":
            return UNGUARDED_SECURITY
        if decision.category == "legal":
            return UNGUARDED_LEGAL
        if decision.category == "refund_exception":
            return UNGUARDED_REFUND
        extracted = extractive_answer(question, retrieved)
        if extracted.startswith("I don't know") or decision.category == "account_lookup":
            return UNGUARDED_UNKNOWN
        return extracted


class OpenAIGenerator:
    def __init__(self, client: object, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self.model = model

    def generate(
        self,
        question: str,
        retrieved: Sequence[Retrieved],
        *,
        guarded: bool,
    ) -> str:
        completion = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt(guarded=guarded)},
                {"role": "user", "content": user_prompt(question, retrieved)},
            ],
        )
        content = completion.choices[0].message.content
        return content or ""


class AnthropicGenerator:
    def __init__(self, client: object, model: str = "claude-sonnet-4-5") -> None:
        self._client = client
        self.model = model

    def generate(
        self,
        question: str,
        retrieved: Sequence[Retrieved],
        *,
        guarded: bool,
    ) -> str:
        message = self._client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=700,
            system=system_prompt(guarded=guarded),
            messages=[{"role": "user", "content": user_prompt(question, retrieved)}],
        )
        block = message.content[0]
        text = getattr(block, "text", "")
        return str(text)
