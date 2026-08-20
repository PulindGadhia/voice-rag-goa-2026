"""Local LLM generation using Apple Silicon MPS / CPU PyTorch inference."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalCausalLMGenerator:
    """Persistent local model instance kept resident in memory."""

    def __init__(
        self,
        model_path_or_id: str = "models/qwen-0.5b",
        device: str | None = None,
    ) -> None:
        self.model_path_or_id = model_path_or_id
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self) -> None:
        load_started = perf_counter()
        logger.info("Loading local model from %s on device=%s", self.model_path_or_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_or_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path_or_id,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        logger.info("Local model loaded in %.2fs", perf_counter() - load_started)

    def warmup(self) -> None:
        """Warm up PyTorch MPS kernels on small sample prompt."""
        if self.model is None or self.tokenizer is None:
            return
        inputs = self.tokenizer(["warmup prompt"], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            _ = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)

    def generate_rag_sync(self, query: str, context: str, max_new_tokens: int = 48) -> str:
        """Fast minimal RAG generation for Tier 1 local inference."""
        prompt = (
            "Answer ONLY from the supplied context.\n"
            "If the context does not contain enough information to answer the question, return:\n"
            "INSUFFICIENT_CONTEXT\n"
            "Otherwise answer in 1-3 concise sentences.\n"
            "Do not add facts from your own knowledge.\n"
            "Do not invent citations.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query}\n\n"
            "ANSWER:"
        )
        return self.generate_sync(prompt, max_new_tokens=max_new_tokens)

    def generate_sync(self, prompt: str, max_new_tokens: int = 128) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local model is not initialized")

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = output_ids[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
