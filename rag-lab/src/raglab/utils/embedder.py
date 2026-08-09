"""
Embedding manager using sentence-transformers with singleton pattern.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class Embedder:
    """
    Singleton wrapper for SentenceTransformer models.
    Caches one model instance per model name to avoid redundant loading.
    """
    
    _instances: Dict[str, 'Embedder'] = {}
    
    def __new__(cls, model_name: str):
        """Singleton pattern: return existing instance if model already loaded."""
        if model_name == "none":
            raise NotImplementedError(
                "Embedder cannot be used with model='none' (sparse-only path). "
                "Use BM25 or PageIndex which do not require embeddings."
            )
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[model_name] = instance
            instance._initialized = False
        return cls._instances[model_name]
    
    def __init__(self, model_name: str):
        """
        Initialize Embedder with specified model.
        
        Args:
            model_name: Name of sentence-transformers model (e.g., 'all-MiniLM-L6-v2')
        """
        # Only initialize once per model name
        if self._initialized:
            return
        
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError("sentence-transformers is required for Embedder")

        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

        self._sanity_check()

        self._initialized = True

    def _sanity_check(self) -> None:
        """
        Skill 50E — after loading, embed a fixed test string and verify the
        output is non-empty with a non-zero norm. Catches a corrupted/partial
        model download that loads without error but produces garbage vectors
        (Failure Mode Register: "Model download corrupted").
        """
        from raglab.types import ModelCorruptedError

        try:
            vec = self.model.encode(["sanity check sentence"])[0]
        except Exception as e:
            raise ModelCorruptedError(
                f"Embedding model '{self.model_name}' failed to encode a test string: {e}"
            ) from e

        if len(vec) == 0:
            raise ModelCorruptedError(
                f"Embedding model '{self.model_name}' produced a zero-dimension vector."
            )
        if not any(abs(float(x)) > 1e-9 for x in vec):
            raise ModelCorruptedError(
                f"Embedding model '{self.model_name}' produced an all-zero vector on sanity check."
            )
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Convert numpy arrays to lists for JSON serialization
        return [emb.tolist() for emb in embeddings]
    
    def embed_one(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()

    def model_dim(self) -> int:
        """
        Get the embedding dimension for the loaded model.
        
        Returns:
            Integer dimension of the embedding vectors
        """
        return self.model.get_embedding_dimension()


class OllamaEmbedder:
    """
    Embedder backed by an Ollama-served embedding model (e.g. nomic-embed-text,
    mxbai-embed-large) via Ollama's /api/embed endpoint — Skill 47B.

    No API key needed; requires a running local Ollama server.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import requests

        response = requests.post(
            f"{self._base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class OpenAIEmbedder:
    """
    Embedder backed by OpenAI's embeddings API (text-embedding-3-small/large)
    — Skill 47B.

    Reads API key from OPENAI_API_KEY env var.
    """

    def __init__(self, model_name: str, api_key: str = None):
        import os

        self.model_name = model_name
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        from openai import OpenAI

        self._client = OpenAI(api_key=resolved_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class SIEEmbedder:
    """
    Embedder backed by a self-hosted SIE (Sentence/Semantic Inference Engine)
    server exposing 85+ models behind one HTTP endpoint — Skill 53(A).

    Model name is the part after "sie/" (e.g. "sie/BAAI/bge-large-en-v1.5"
    -> model_name="BAAI/bge-large-en-v1.5"). No API key needed; requires a
    running SIE server at `base_url`.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:8080"):
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import requests

        response = requests.post(
            f"{self._base_url}/embed",
            json={"model": self.model_name, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class QuantizedEmbedder:
    """
    Wraps any embedder and quantizes its output vectors — Skill 53(B).

    "int8": scale to the int8 range and back (simulated quantization —
        ~4x memory reduction if persisted as int8, <1% MTEB quality loss).
    "binary": sign quantization (+1.0/-1.0 per dimension) — ~32x memory
        reduction if persisted as bits, meaningful quality drop, intended
        for research/exploration rather than production defaults.
    "none": pass-through (should not be wrapped, but handled for safety).
    """

    def __init__(self, inner, quantization: str = "none"):
        self._inner = inner
        self.quantization = quantization
        # Preserve attributes callers may introspect (e.g. model_name).
        self.model_name = getattr(inner, "model_name", None)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._quantize(vec) for vec in self._inner.embed(texts)]

    def embed_one(self, text: str) -> List[float]:
        return self._quantize(self._inner.embed_one(text))

    def _quantize(self, vec: List[float]) -> List[float]:
        if self.quantization == "int8":
            return [max(-127, min(127, round(x * 127))) / 127.0 for x in vec]
        if self.quantization == "binary":
            return [1.0 if x >= 0 else -1.0 for x in vec]
        return vec


def get_embedder(embed_cfg):
    """
    Factory dispatching on the model-name prefix — Skill 47B / 53.

    "ollama/<model>"  -> OllamaEmbedder (local, no key, via Ollama server)
    "openai/<model>"  -> OpenAIEmbedder (needs OPENAI_API_KEY)
    "sie/<model>"     -> SIEEmbedder (self-hosted inference server, no key)
    anything else     -> Embedder (local sentence-transformers, default path)

    If embed_cfg.quantization is set to "int8" or "binary", the resulting
    embedder is wrapped in a QuantizedEmbedder (Skill 53B).
    """
    model_name = embed_cfg.model if hasattr(embed_cfg, "model") else embed_cfg
    quantization = getattr(embed_cfg, "quantization", "none")

    if model_name.startswith("ollama/"):
        embedder = OllamaEmbedder(model_name.split("/", 1)[1])
    elif model_name.startswith("openai/"):
        embedder = OpenAIEmbedder(model_name.split("/", 1)[1])
    elif model_name.startswith("sie/"):
        base_url = getattr(embed_cfg, "sie_base_url", "http://localhost:8080")
        embedder = SIEEmbedder(model_name.split("/", 1)[1], base_url=base_url)
    else:
        embedder = Embedder(model_name)

    if quantization != "none":
        return QuantizedEmbedder(embedder, quantization=quantization)
    return embedder

