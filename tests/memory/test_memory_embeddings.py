"""app.memory.embeddings — onnxruntime InferenceSession / tokenizers.Tokenizer
mocked, no model download (F-stage). Uses onnxruntime + bare tokenizers
instead of torch/transformers (see embeddings.py's docstring: Windows Smart
App Control blocked torch's own bundled shm.dll, and transformers turned out
to unconditionally import torch too even for bare tokenization)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from app.memory import embeddings


class _FakeEncoding:
    def __init__(self, ids, attention_mask, type_ids=None):
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids or [0] * len(ids)


class _FakeTokenizer:
    def __init__(self, calls, encoding=None):
        self._calls = calls
        # One token, mask=1 — makes the mean-pooling arithmetic trivial to
        # predict in assertions below.
        self._encoding = encoding or _FakeEncoding([1], [1])
        self.truncation_max_length = None
        self.padding_enabled = False

    def enable_truncation(self, max_length):
        self.truncation_max_length = max_length

    def enable_padding(self, pad_id, pad_token):
        self.padding_enabled = True

    def encode(self, text):
        self._calls.append(text)
        return self._encoding


class _FakeSession:
    def __init__(self, hidden_values):
        # (batch=1, seq=1, dim) — matches the single-token fake encoding above.
        self._hidden = np.array([[hidden_values]], dtype=np.float32)
        self.run_calls = []

    def run(self, output_names, inputs):
        self.run_calls.append(inputs)
        return [self._hidden]


def test_embed_returns_normalized_mean_pooled_vector(monkeypatch):
    calls = []
    fake_tokenizer = _FakeTokenizer(calls)
    fake_session = _FakeSession([3.0, 4.0])  # normalizes to [0.6, 0.8]
    monkeypatch.setattr(embeddings, "_get_model", lambda: (fake_tokenizer, fake_session))

    result = embeddings.embed("some situation text")

    assert result == pytest.approx([0.6, 0.8])
    assert calls == ["some situation text"]


def test_embed_passes_ids_mask_and_type_ids_to_the_session(monkeypatch):
    calls = []
    fake_tokenizer = _FakeTokenizer(calls, encoding=_FakeEncoding([101, 42, 102], [1, 1, 1], [0, 0, 0]))
    fake_session = _FakeSession([1.0, 0.0])
    monkeypatch.setattr(embeddings, "_get_model", lambda: (fake_tokenizer, fake_session))

    embeddings.embed("some text")

    (inputs,) = fake_session.run_calls
    assert inputs["input_ids"].tolist() == [[101, 42, 102]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1]]
    assert inputs["token_type_ids"].tolist() == [[0, 0, 0]]


def test_get_model_is_a_lazy_singleton(monkeypatch):
    monkeypatch.setattr(embeddings, "_session", None)
    monkeypatch.setattr(embeddings, "_tokenizer", None)
    created = []

    def fake_hf_hub_download(repo, filename):
        return f"{repo}/{filename}"

    def fake_tokenizer_from_file(path):
        return _FakeTokenizer([])

    def fake_inference_session(path, providers=None):
        created.append(path)
        return _FakeSession([1.0])

    monkeypatch.setattr(embeddings, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(embeddings.Tokenizer, "from_file", staticmethod(fake_tokenizer_from_file))
    monkeypatch.setattr(embeddings.ort, "InferenceSession", fake_inference_session)

    embeddings.embed("first call")
    embeddings.embed("second call")
    assert len(created) == 1  # session constructed only once across both calls


def test_max_seq_length_matches_the_model_s_own_sentence_bert_config():
    # Fix 5: the pre-ONNX-swap sentence-transformers pipeline this module
    # must stay parity with truncates at 256 (the model's own
    # sentence_bert_config.json max_seq_length), not 512 (the underlying
    # BERT architecture's unrelated absolute position-embedding limit) and
    # not the old, wrong 128 that silently discarded up to half of
    # situation_text before it ever reached the model.
    assert embeddings._MAX_SEQ_LENGTH == 256


def test_get_model_configures_truncation_and_padding(monkeypatch):
    monkeypatch.setattr(embeddings, "_session", None)
    monkeypatch.setattr(embeddings, "_tokenizer", None)
    fake_tokenizer = _FakeTokenizer([])

    monkeypatch.setattr(embeddings, "hf_hub_download", lambda repo, filename: "path")
    monkeypatch.setattr(embeddings.Tokenizer, "from_file", staticmethod(lambda path: fake_tokenizer))
    monkeypatch.setattr(embeddings.ort, "InferenceSession", lambda path, providers=None: _FakeSession([1.0]))

    embeddings._get_model()

    assert fake_tokenizer.truncation_max_length == embeddings._MAX_SEQ_LENGTH
    assert fake_tokenizer.padding_enabled is True


async def test_embed_async_delegates_to_embed(monkeypatch):
    fake_tokenizer = _FakeTokenizer([])
    fake_session = _FakeSession([1.0, 0.0])
    monkeypatch.setattr(embeddings, "_get_model", lambda: (fake_tokenizer, fake_session))

    result = await embeddings.embed_async("async situation text")
    assert result == pytest.approx([1.0, 0.0])


def test_get_model_lock_serializes_concurrent_construction(monkeypatch):
    # Regression test for the race: two real OS threads (embed() runs under
    # asyncio.to_thread, not asyncio tasks) hitting `_session is None` at the
    # same time used to both construct their own model instance.
    monkeypatch.setattr(embeddings, "_session", None)
    monkeypatch.setattr(embeddings, "_tokenizer", None)
    created = []

    def fake_tokenizer_from_file(path):
        return _FakeTokenizer([])

    def fake_inference_session(path, providers=None):
        time.sleep(0.05)  # widen the race window
        created.append(path)
        return _FakeSession([1.0])

    monkeypatch.setattr(embeddings, "hf_hub_download", lambda repo, filename: "path")
    monkeypatch.setattr(embeddings.Tokenizer, "from_file", staticmethod(fake_tokenizer_from_file))
    monkeypatch.setattr(embeddings.ort, "InferenceSession", fake_inference_session)

    threads = [threading.Thread(target=embeddings._get_model) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1
