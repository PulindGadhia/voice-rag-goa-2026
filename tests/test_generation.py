from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.config import Settings
from app.generation import GenerationService, parse_generated_output


def test_groq_settings_use_provider_specific_model_and_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    settings = Settings.from_env()
    assert settings.llm_provider == "groq"
    assert settings.llm_model == "openai/gpt-oss-20b"
    assert settings.active_llm_api_key == "test-groq-key"


def test_gemini_settings_use_provider_specific_model_and_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    settings = Settings.from_env()
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3.6-flash"
    assert settings.active_llm_api_key == "test-gemini-key"


def test_gemini_generation_uses_model_and_json_config_without_real_api():
    class FakeModels:
        def __init__(self):
            self.kwargs = None

        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                text='{"answer":"A corporation is a legal entity.",'
                '"grounded":true,"source_ids":["source-1"],"confidence":0.9}'
            )

    models = FakeModels()
    service = GenerationService(
        provider="gemini", model_name="gemini-3.6-flash", api_key="test-key", max_retries=0
    )
    service._client = SimpleNamespace(models=models)
    raw = service._generate_sync("prompt")
    parsed = parse_generated_output(raw, {"source-1"})
    assert parsed.grounded is True
    assert models.kwargs["model"] == "gemini-3.6-flash"
    assert models.kwargs["contents"] == "prompt"
    assert "config" in models.kwargs


def test_gemini_parser_accepts_fenced_json():
    parsed = parse_generated_output(
        '```json\n{"answer":"India","grounded":true,"source_ids":["source-1"],"confidence":0.8}\n```',
        {"source-1"},
    )
    assert parsed.answer == "India"
    assert parsed.grounded is True


def test_gemini_parser_normalizes_retrieved_identifier_aliases():
    parsed = parse_generated_output(
        '{"answer":"India","grounded":true,"source_ids":["chunk-7"],"confidence":0.8}',
        {"source-1"},
        {"chunk-7": "source-1"},
    )
    assert parsed.source_ids == ["source-1"]
    assert parsed.grounded is True


def test_gemini_generation_is_async_and_returns_latency():
    service = GenerationService(provider="gemini", model_name="test", api_key="test", max_retries=0)
    service._generate_sync = lambda prompt: '{"answer":"ok","grounded":false,"source_ids":[],"confidence":0}'

    async def run():
        raw, latency = await service.generate(question="q", context="c", passages=["c"])
        assert raw.startswith("{")
        assert latency >= 0

    asyncio.run(run())


def test_local_settings_use_provider_specific_model_and_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = Settings.from_env()
    assert settings.llm_provider == "local"
    assert settings.llm_model == "models/smollm2-135m"
    assert settings.active_llm_api_key == "local"


def test_local_generation_uses_generator():
    class FakeLocalGen:
        def generate_sync(self, prompt, max_new_tokens=64):
            return '{"answer":"Local test","grounded":true,"source_ids":["source-1"],"confidence":0.9}'

    service = GenerationService(provider="local", model_name="models/smollm2-135m", api_key="local")
    service._local_generator = FakeLocalGen()
    raw = service._generate_sync("prompt")
    parsed = parse_generated_output(raw, {"source-1"})
    assert parsed.answer == "Local test"
    assert parsed.grounded is True
