"""Synthetic benchmark input contract without subprocesses or live services."""

import importlib.util
import tracemalloc
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion

from app.llm import request_capture as recorder


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/aa_eval/capture_benchmark.py"
spec = importlib.util.spec_from_file_location("synthetic_capture_benchmark_test", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_case_matrix_covers_request_response_extras_rejections_and_queue():
    assert len(benchmark.CASES) == 10
    assert benchmark.CASES[-1] == "queue_saturation"
    for language in ("ascii", "korean"):
        for size in ("8k", "128k", "near_limit"):
            assert f"{language}_{size}" in benchmark.CASES
    for label in benchmark.CASES[:-1]:
        request, response, request_bytes, response_bytes = benchmark._payload(label)
        assert set(request) == {"model", "messages", "temperature", "extra_body",
                                "max_tokens", "response_format"}
        assert request_bytes > 0 and response_bytes > 0
        if label == "sdk_extra":
            assert isinstance(response, ChatCompletion)
            assert hasattr(response.choices[0], "model_dump")
            assert hasattr(response.choices[0].message, "model_dump")
            assert response.model_extra["top_extension"]["nested_extension"]
            assert response.choices[0].model_extra["choice_extension"]
            assert response.choices[0].message.model_extra["message_extension"]
            dumped = response.model_dump(mode="json")
            assert dumped["top_extension"]
            assert dumped["choices"][0]["choice_extension"]
            assert dumped["choices"][0]["message"]["message_extension"]
        if label == "rejected_request":
            assert request_bytes > 2 * 1024 * 1024
        if label == "rejected_response":
            assert response_bytes > 2 * 1024 * 1024
        if label.endswith("near_limit"):
            assert 1700000 < request_bytes < 2 * 1024 * 1024
            assert 1700000 < response_bytes < 2 * 1024 * 1024
        if label.endswith("_8k"):
            assert 8000 <= request_bytes <= 8192
            assert 8000 <= response_bytes <= 8192
        if label.endswith("_128k"):
            assert 131000 <= request_bytes <= 131072
            assert 131000 <= response_bytes <= 131072


def test_small_case_does_not_allocate_other_case_inputs():
    tracemalloc.start()
    try:
        _, _, request_bytes, response_bytes = benchmark._payload("ascii_8k")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert request_bytes == response_bytes == 8192
    assert peak < 1_000_000


@pytest.mark.parametrize("location", ["top", "choice", "message"])
def test_real_sdk_nested_extra_is_rejected_during_response_preflight(location):
    response = benchmark._sdk_response("{}", oversized_extra=location)
    with pytest.raises(ValueError, match="response_limit|structure_limit"):
        recorder._preflight_response(response, 2 * 1024 * 1024)


def test_real_sdk_small_nested_extras_pass_response_preflight():
    _, response, _, _ = benchmark._payload("sdk_extra")
    recorder._preflight_response(response, 2 * 1024 * 1024)
    dumped = response.model_dump(mode="json")
    assert dumped["choices"][0]["choice_extension"]
    assert dumped["choices"][0]["message"]["message_extension"]


def test_memory_probe_reports_current_and_peak_slots():
    memory = benchmark._memory()
    assert set(memory) == {"rss_bytes", "peak_rss_bytes"}
    assert all(value is None or value > 0 for value in memory.values())
    assert benchmark._memory_available(memory) is all(value is not None for value in memory.values())
    assert benchmark._memory_available({"rss_bytes": None, "peak_rss_bytes": 1}) is False
    assert benchmark._nonnegative_delta(12, 10) == 2
    assert benchmark._nonnegative_delta(8, 10) == 0
