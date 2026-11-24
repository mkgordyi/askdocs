"""Turn retrieved passages into an answer.

Two ways, behind one interface:
- Extractive: quote the best passage. No model call, cannot invent anything.
- LLM: write a short answer from the passages. The answer is accepted only if it
  cites passages that were actually retrieved. No valid citation, no answer.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .index import Hit

NOT_FOUND = "NOT_FOUND"
SYSTEM_PROMPT = (
    "You answer questions about the EU AI Act using ONLY the numbered passages given. "
    "Answer in at most four sentences. After every claim put the number of the passage "
    "it comes from in square brackets, like [2]. Do not use any other knowledge. "
    f"If the passages do not contain the answer, reply with exactly {NOT_FOUND}."
)


@dataclass(frozen=True)
class Generated:
    text: str | None          # None means "the passages do not answer this"
    cited: list[Hit]
    reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0


class Generator(Protocol):
    mode: str

    def generate(self, question: str, hits: list[Hit]) -> Generated: ...


class ExtractiveGenerator:
    mode = "extractive"

    def generate(self, question: str, hits: list[Hit]) -> Generated:
        best = hits[0]
        # Quote the best passage; list up to two more that match almost as well.
        close = [h for h in hits[1:3] if h.similarity >= best.similarity - 0.05]
        return Generated(text=_trim(best.chunk.text), cited=[best, *close])


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        """Returns (text, input tokens, output tokens)."""


class LLMGenerator:
    mode = "llm"

    def __init__(self, client: ChatClient):
        self._client = client

    def generate(self, question: str, hits: list[Hit]) -> Generated:
        passages = "\n\n".join(f"[{i}] {h.chunk.ref}: {h.chunk.text}" for i, h in enumerate(hits, 1))
        text, tokens_in, tokens_out = self._client.complete(
            SYSTEM_PROMPT, f"Passages:\n{passages}\n\nQuestion: {question}"
        )
        text = text.strip()
        if text == NOT_FOUND or NOT_FOUND in text:
            return Generated(None, [], "not_in_passages", tokens_in, tokens_out)
        numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        cited = [hits[n - 1] for n in sorted(numbers) if 1 <= n <= len(hits)]
        if not cited or any(n < 1 or n > len(hits) for n in numbers):
            # The model answered without a real source. Rule of the project: no answer.
            return Generated(None, [], "uncited", tokens_in, tokens_out)
        return Generated(text, cited, None, tokens_in, tokens_out)


class OpenAIChatClient:
    """Works with OpenAI and with Azure OpenAI (keyless, through the app's identity)."""

    def __init__(self, settings: Settings):
        if settings.llm_provider == "azure_openai":
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            from openai import AzureOpenAI

            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            self._client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=settings.azure_openai_api_version,
            )
        else:
            from openai import OpenAI

            self._client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self._model = settings.llm_model

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        response = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=1000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = response.usage
        return response.choices[0].message.content or "", usage.prompt_tokens, usage.completion_tokens


def make_generator(settings: Settings) -> Generator:
    if settings.llm_provider in ("openai", "azure_openai"):
        return LLMGenerator(OpenAIChatClient(settings))
    return ExtractiveGenerator()


def _trim(text: str, limit: int = 700) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return cut[: end + 1] if end > limit // 2 else cut.rstrip() + "..."
