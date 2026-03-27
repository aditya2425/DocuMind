"""Multi-LLM client supporting OpenAI, Anthropic (Claude), and Mistral."""

from __future__ import annotations

from typing import List, Optional

from src.config.settings import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_CHAT_MODEL,
    MISTRAL_API_KEY,
    MISTRAL_CHAT_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)


class LLMClient:
    """
    Unified wrapper around multiple LLM providers.

    Supported providers: ``openai``, ``anthropic``, ``mistral``.
    """

    SUPPORTED = ("openai", "anthropic", "mistral")

    def __init__(self, provider: str = "openai") -> None:
        provider = provider.lower()
        if provider not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: {self.SUPPORTED}"
            )
        self.provider = provider
        self._init_client()

    # ── initialisation ───────────────────────────────────────
    def _init_client(self) -> None:
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing in .env")
            from openai import OpenAI

            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_CHAT_MODEL

        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is missing in .env")
            import anthropic

            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_CHAT_MODEL

        elif self.provider == "mistral":
            if not MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY is missing in .env")
            from mistralai import Mistral

            self.client = Mistral(api_key=MISTRAL_API_KEY)
            self.model = MISTRAL_CHAT_MODEL

    # ── generation ───────────────────────────────────────────
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Send a single prompt and return the assistant's text response."""

        if self.provider == "openai":
            return self._openai_generate(
                system_prompt, user_message, temperature, max_tokens
            )
        if self.provider == "anthropic":
            return self._anthropic_generate(
                system_prompt, user_message, temperature, max_tokens
            )
        if self.provider == "mistral":
            return self._mistral_generate(
                system_prompt, user_message, temperature, max_tokens
            )

        raise RuntimeError(f"No generate path for provider: {self.provider}")

    # ── provider-specific implementations ────────────────────
    def _openai_generate(
        self, system: str, user: str, temp: float, max_tok: int
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=max_tok,
        )
        return resp.choices[0].message.content.strip()

    def _anthropic_generate(
        self, system: str, user: str, temp: float, max_tok: int
    ) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temp,
            max_tokens=max_tok,
        )
        return resp.content[0].text.strip()

    def _mistral_generate(
        self, system: str, user: str, temp: float, max_tok: int
    ) -> str:
        resp = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=max_tok,
        )
        return resp.choices[0].message.content.strip()
