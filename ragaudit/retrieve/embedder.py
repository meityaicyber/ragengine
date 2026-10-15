"""Embedding and reranking models.

Failure policy (D02). The previous implementation fell back silently in two
places: a failed embedding-model load swapped in a different model with a
different vector space, and a failed reranker load returned ``1.0/(rank+1)`` as
a score. Both produce plausible numbers from a broken system, so a run where
the models never loaded is indistinguishable from a healthy one.

Here a load failure raises. A retrieval system that cannot load its retriever
has nothing useful to say, and saying nothing is strictly better than
reporting invented confidence.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger("ragaudit.retrieve.embedder")


class ModelLoadError(RuntimeError):
    """Raised when a required local model cannot be loaded."""


def require_local_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise ModelLoadError(
            f"model weights not found at '{resolved}'. Stage them inside the approved "
            f"air-gapped import process; this package never downloads weights."
        )
    return resolved


class Embedder:
    """Instruction-aware dense embedder over locally staged weights.

    Qwen3-Embedding expects the retrieval instruction to be attached to the
    *query* only, never the passage. Mixing that up degrades retrieval quietly,
    so the two paths are separate methods rather than one with a flag.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        max_sequence_tokens: int = 1024,
        query_instruction: str = "",
        dimension: int | None = None,
    ) -> None:
        self.model_path = str(require_local_path(model_path))
        self.device = device
        self.batch_size = batch_size
        self.max_sequence_tokens = max_sequence_tokens
        self.query_instruction = query_instruction
        self._model = None
        self._tokenizer = None
        self._dimension = dimension

    # -- lifecycle -------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelLoadError(
                "transformers and torch are required for embedding. Install ragaudit[models]."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._model = (
                AutoModel.from_pretrained(self.model_path, local_files_only=True, torch_dtype=dtype)
                .to(self.device)
                .eval()
            )
        except Exception as exc:
            raise ModelLoadError(f"failed to load embedding model '{self.model_path}': {exc}") from exc

        probe = self._encode(["dimension probe"])
        self._dimension = int(probe.shape[1])
        logger.info(
            "embedder ready", extra={"model": self.model_path, "device": self.device, "dim": self._dimension}
        )

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self.load()
        return int(self._dimension)  # type: ignore[arg-type]

    @property
    def fingerprint(self) -> str:
        """Identity of the vector space, recorded with every index."""
        return f"{Path(self.model_path).name}:{self.dimension}"

    # -- encoding --------------------------------------------------------

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        self.load()
        return self._encode(list(texts))

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.dimension), dtype="float32")
        self.load()
        if self.query_instruction:
            queries = [f"Instruct: {self.query_instruction}\nQuery: {q}" for q in queries]
        return self._encode(list(queries))

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(  # type: ignore[misc]
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_tokens,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state  # type: ignore[misc]
            # Last-token pooling, as Qwen3-Embedding is trained for. Mean
            # pooling over a causal model mixes in positions that never
            # attended to the full sequence.
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype("float32")


#: Qwen3-Reranker's prompt, taken from the weights' own chat template. The
#: model is a causal LM that answers "yes" or "no"; relevance is the softmax
#: over those two token logits at the final position.
_QWEN_RERANK_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".'
)
_QWEN_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class CrossEncoderReranker:
    """Second-stage reranker over a candidate pool.

    Two architectures are supported and detected from the weights, because
    loading one as the other is a silent-failure trap: Qwen3-Reranker is a
    ``Qwen3ForCausalLM`` scoring through yes/no token logits, and loading it
    with ``AutoModelForSequenceClassification`` succeeds while randomly
    initialising ``score.weight``. That produces confident, meaningless
    rankings -- which is precisely the class of fault this package refuses to
    tolerate, so the mismatch raises instead.

    Scores are comparable only within one query's candidate list, which is why
    downstream code gates on rank rather than an absolute threshold (D05).
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 1024,
        instruction: str = (
            "Given a question about a cybersecurity audit report, "
            "retrieve the audit evidence that answers it"
        ),
    ) -> None:
        self.model_path = str(require_local_path(model_path))
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction
        self._model = None
        self._tokenizer = None
        self._mode = "unknown"       # 'logit_score' | 'sequence_classification'
        self._true_id: int | None = None
        self._false_id: int | None = None

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                "transformers and torch are required for reranking. Install ragaudit[models]."
            ) from exc

        try:
            config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
            architectures = list(getattr(config, "architectures", []) or [])
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

            if any("ForCausalLM" in name for name in architectures):
                self._mode = "logit_score"
                self._load_token_ids()
                from transformers import AutoModelForCausalLM

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, local_files_only=True, dtype=dtype
                )
            else:
                self._mode = "sequence_classification"
                from transformers import AutoModelForSequenceClassification

                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path, local_files_only=True, dtype=dtype
                )
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"failed to load reranker '{self.model_path}': {exc}") from exc

        # Qwen3 defines no pad token; without one, any batch larger than a
        # single row raises inside the model's attention mask handling.
        if self._tokenizer.pad_token_id is None:  # type: ignore[union-attr]
            self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore[union-attr]
        self._tokenizer.padding_side = "left"  # type: ignore[union-attr]

        self._model = self._model.to(self.device).eval()  # type: ignore[union-attr]
        logger.info(
            "reranker ready",
            extra={"model": self.model_path, "device": self.device, "mode": self._mode},
        )

    def _load_token_ids(self) -> None:
        """Read the yes/no token ids the weights ship with.

        Hardcoding them would break silently against a different tokenizer
        revision, so they are read from the model directory and only derived
        from the tokenizer as a fallback.
        """
        config_path = Path(self.model_path) / "1_LogitScore" / "config.json"
        if config_path.exists():
            import json

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self._true_id = int(payload["true_token_id"])
            self._false_id = int(payload["false_token_id"])
            return

        true_ids = self._tokenizer.encode("yes", add_special_tokens=False)  # type: ignore[union-attr]
        false_ids = self._tokenizer.encode("no", add_special_tokens=False)  # type: ignore[union-attr]
        if len(true_ids) != 1 or len(false_ids) != 1:
            raise ModelLoadError(
                f"cannot determine yes/no token ids for reranker '{self.model_path}'. "
                "Expected a 1_LogitScore/config.json in the model directory."
            )
        self._true_id, self._false_id = true_ids[0], false_ids[0]

    # -- scoring ---------------------------------------------------------

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance per passage. Higher is more relevant."""
        if not passages:
            return []
        self.load()
        import torch

        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = list(passages[start : start + self.batch_size])
            if self._mode == "logit_score":
                scores.extend(self._score_logit_batch(query, batch, torch))
            else:
                scores.extend(self._score_classifier_batch(query, batch, torch))
        return scores

    def _build_prompt(self, query: str, document: str) -> str:
        return (
            f"<|im_start|>system\n{_QWEN_RERANK_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
            f"{_QWEN_RERANK_SUFFIX}"
        )

    def _score_logit_batch(self, query: str, batch: list[str], torch) -> list[float]:
        prompts = [self._build_prompt(query, doc) for doc in batch]
        encoded = self._tokenizer(  # type: ignore[misc]
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self._model(**encoded).logits[:, -1, :]  # type: ignore[misc]
        pair = torch.stack([logits[:, self._false_id], logits[:, self._true_id]], dim=1)
        # Softmax over just the two tokens: the probability assigned to "yes"
        # relative to "no" is the relevance score the model was trained to give.
        return torch.nn.functional.log_softmax(pair.float(), dim=1)[:, 1].cpu().tolist()

    def _score_classifier_batch(self, query: str, batch: list[str], torch) -> list[float]:
        encoded = self._tokenizer(  # type: ignore[misc]
            [query] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self._model(**encoded).logits  # type: ignore[misc]
        values = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, -1]
        return values.float().cpu().tolist()
