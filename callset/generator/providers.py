from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Generate a text response from the LLM."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with: pip install 'callset[anthropic]'"
            )

        self._model = model
        self._client = Anthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text


DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}


def get_provider(provider_name: str, model: str | None = None, api_key: str | None = None) -> LLMProvider:
    """Factory function to create an LLM provider."""
    resolved_model = model or DEFAULT_MODELS.get(provider_name, "gpt-4o")

    if provider_name == "openai":
        return OpenAIProvider(model=resolved_model, api_key=api_key)
    elif provider_name == "anthropic":
        return AnthropicProvider(model=resolved_model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Use 'openai' or 'anthropic'.")
