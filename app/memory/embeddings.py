"""Local CPU embedding model for memory similarity search — decoupled
entirely from the trading vLLM instance, which is already single-GPU and
reserved for the reasoning workload. Lazy-loaded singleton so the model
loads once per process, not once per call.

Uses `onnxruntime` + a bare `tokenizers.Tokenizer` (manual mean pooling in
numpy) instead of `torch`/`transformers`. Two prior Smart App Control
incidents on this project's Windows environment forced this: first,
`sentence-transformers` (dropped 2026-0x-xx) hard-depends on scikit-learn/
scipy, whose unsigned `_linalg_pythran` native DLL got blocked; second,
`torch` itself (2026-08-05) got its own bundled `shm.dll` — the CPU
multiprocessing shared-memory allocator, a torch.multiprocessing/DataLoader
feature this module never uses even indirectly — blocked the same way, and
worse, `transformers` (even bare `AutoTokenizer`, with no `AutoModel`
involved) turned out to unconditionally import torch too, three frames deep
inside `generation.logits_process` — an assisted-decoding feature no
embedding call ever reaches. Both torch and transformers are gone from
this project entirely now; onnxruntime's own native libraries load cleanly
under the same Smart App Control policy (confirmed live 2026-08-05).
Smart App Control이 torch/transformers의 네이티브 DLL을 차단해서(2026-08-05),
onnxruntime + tokenizers 조합(순수 CPU 추론, torch 의존성 전혀 없음)으로 교체.

`settings.memory_embedding_model` (sentence-transformers/all-MiniLM-L6-v2)
is the model IDENTITY (weights) this project has always used for every
embedding stored so far — kept as-is so this isn't a silent model swap,
only a runtime-library swap. `_ONNX_REPO` below is a pre-converted ONNX
port of those exact same weights (a Xenova/transformers.js community
export, not a different model) — this avoids needing `optimum`'s
export-to-ONNX step, which itself traces through torch and would just
reintroduce the same blocked import. Verified live (2026-08-05): cosine
self-similarity 1.0, a semantically similar sentence pair ~0.49, an
unrelated pair ~0.08 — the expected shape for this model family."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from app.config import settings

# Xenova's ONNX port of sentence-transformers/all-MiniLM-L6-v2 — same
# weights/vocab, just pre-exported to ONNX (no torch involved in loading or
# running it). If `settings.memory_embedding_model` is ever changed to a
# different model family, this must be updated to a matching ONNX port —
# the two are NOT independently configurable (see module docstring).
_ONNX_REPO = "Xenova/all-MiniLM-L6-v2"
_ONNX_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "tokenizer.json"
# Matches sentence-transformers/all-MiniLM-L6-v2's OWN sentence_bert_config.
# json `max_seq_length` value (256) — the truncation length the original,
# pre-ONNX-swap sentence-transformers pipeline used (see module docstring's
# "kept as-is so this isn't a silent model swap, only a runtime-library
# swap" policy). This is NOT the underlying BERT architecture's absolute
# position-embedding limit (512, standard BERT-base) — a larger, different
# number that would silently over-truncate relative to what the pre-swap
# pipeline actually did. `situation_text` (four analyst reports
# concatenated) easily runs to thousands of tokens, so truncating at the
# wrong number is not a rounding error: it changes how much of "today's
# situation" retrieval is actually matching against.
_MAX_SEQ_LENGTH = 256

_tokenizer: Tokenizer | None = None
_session: ort.InferenceSession | None = None
# embed() runs under asyncio.to_thread, i.e. real OS threads, not asyncio
# tasks — an unguarded check-then-set here lets concurrent requests race and
# each construct their own model session.
# 실제 OS 스레드 경합이라 asyncio.Lock이 아닌 threading.Lock으로 이중 확인 잠금 적용.
_model_lock = threading.Lock()


def _get_model() -> tuple[Tokenizer, ort.InferenceSession]:
    global _tokenizer, _session
    if _session is None:
        with _model_lock:
            if _session is None:
                tokenizer_path = hf_hub_download(_ONNX_REPO, _TOKENIZER_FILE)
                tokenizer = Tokenizer.from_file(tokenizer_path)
                tokenizer.enable_truncation(max_length=_MAX_SEQ_LENGTH)
                # No fixed `length=` — pads to the longest sequence in
                # whatever batch is passed to encode_batch (a no-op for
                # embed()'s own single-text batch-of-1 calls, matching the
                # old transformers pipeline's `padding=True` on a length-1
                # list, which likewise never padded beyond that one text's
                # own length).
                tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
                _tokenizer = tokenizer
                onnx_path = hf_hub_download(_ONNX_REPO, _ONNX_FILE)
                _session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return _tokenizer, _session


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Standard mean-pooling for this model family (per the model card for
    sentence-transformers/all-MiniLM-L6-v2, the model this project uses):
    average token embeddings, weighted by the attention mask so padding
    tokens don't contribute — same computation the old torch pipeline did,
    just in numpy instead of torch ops."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def embed(text: str) -> list[float]:
    tokenizer, session = _get_model()
    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoded.type_ids], dtype=np.int64)
    (last_hidden_state,) = session.run(
        None, {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
    )
    pooled = _mean_pool(last_hidden_state, attention_mask)
    norm = np.linalg.norm(pooled, ord=2, axis=1, keepdims=True)
    normalized = pooled / np.clip(norm, 1e-12, None)
    return normalized[0].tolist()


async def embed_async(text: str) -> list[float]:
    return await asyncio.to_thread(embed, text)
