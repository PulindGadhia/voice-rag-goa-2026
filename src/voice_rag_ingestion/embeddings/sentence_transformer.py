"""Sentence-Transformers adapter for multilingual-e5-small."""

from __future__ import annotations

from typing import Sequence

from .base import EmbeddingConfig, EmbeddingProvider


class SentenceTransformerEmbedder:
    """Provider-neutral wrapper around a SentenceTransformer model.

    The model is loaded lazily so unit tests and mocked integrations do not
    require PyTorch or model weights. E5's query/passage prefixes are applied
    here and are invisible to the vector store and retriever.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model: object | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for model-backed embeddings"
                ) from exc
            kwargs = {}
            if self.config.device:
                kwargs["device"] = self.config.device
            model = SentenceTransformer(self.config.model_name, **kwargs)
        self.model = model
        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("embedding model did not expose a vector dimension")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _prefix(self, text: str, input_type: str) -> str:
        if input_type == "query":
            return f"{self.config.query_prefix}{text}"
        if input_type == "passage":
            return f"{self.config.passage_prefix}{text}"
        raise ValueError("input_type must be 'query' or 'passage'")

    def embed_text(self, text: str, *, input_type: str = "passage") -> list[float]:
        vectors = self.embed_batch([text], input_type=input_type)
        return vectors[0]

    def embed_batch(
        self, texts: Sequence[str], *, input_type: str = "passage"
    ) -> list[list[float]]:
        if not texts:
            return []
        prepared = [self._prefix(text, input_type) for text in texts]
        encoded = self.model.encode(
            prepared,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        vectors = [[float(value) for value in vector] for vector in encoded]
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("model returned an unexpected embedding dimension")
        return vectors
