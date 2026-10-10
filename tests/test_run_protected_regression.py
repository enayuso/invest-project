"""Tests for scripts/run_protected_regression.py pure logic (never installs the audit hook)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_protected_regression.py"
_spec = importlib.util.spec_from_file_location("run_protected_regression", _SCRIPT)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


@pytest.mark.parametrize(
    ("event", "args", "internal", "expected"),
    [
        ("socket.getaddrinfo", ("example.com", 80, 0, 0), False, "deny"),
        ("socket.connect", (object(), ("93.184.216.34", 80)), False, "deny"),
        ("socket.connect", (object(), ("127.0.0.1", 5432)), False, "deny"),  # local DB port
        ("socket.connect", (object(), ("127.0.0.1", 50000)), True, "allow_local"),
        ("socket.connect", (object(), ("::1", 50000, 0, 0)), True, "allow_local"),
        ("socket.connect", (object(), ("10.0.0.5", 50000)), True, "deny"),  # internal but not loopback
        ("socket.connect", (object(),), False, "deny"),  # malformed args fail closed
        ("subprocess.Popen", (None, ["git", "status"], None, None), False, "record"),
        ("open", ("x",), False, "ignore"),
    ],
)
def test_decide(event, args, internal, expected):
    assert tool.decide(event, args, internal) == expected


def test_format_popen_joins_and_truncates():
    assert tool.format_popen((None, ["git", "log", "-1"])) == "git log -1"
    assert tool.format_popen((None, "x" * 500)) == "x" * 160


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("....\n4870 passed, 2 skipped in 240.1s (0:04:00)\n", (4870, 0, 2)),
        ("FAILED a::b\n1 failed, 10 passed in 3s\n", (10, 1, 0)),
        ("3 passed in 0.5s", (3, 0, 0)),
        ("no summary here", (None, None, None)),
        ("", (None, None, None)),
    ],
)
def test_parse_pytest_counts(text, expected):
    got = tool.parse_pytest_counts(text)
    assert (got["passed"], got["failed"], got["skipped"]) == expected


def test_build_summary_fields_and_json_serializable():
    summary = tool.build_summary(
        exit_code=1,
        elapsed_seconds=12.345,
        python="py",
        root="r",
        git_head="abc",
        git_dirty=True,
        pytest_args=["tests/", "-q"],
        log_text="1 failed, 2 passed in 1s",
        denied=["socket.getaddrinfo", "socket.getaddrinfo", "socket.connect"],
        socketpairs=3,
        popens=[f"cmd {i}" for i in range(20)] + ["cmd 0"],
    )
    assert summary["exit_code"] == 1
    assert summary["elapsed_seconds"] == 12.3
    assert summary["failed"] == 1 and summary["passed"] == 2 and summary["skipped"] == 0
    assert summary["network_denied"] == {"socket.connect": 1, "socket.getaddrinfo": 2}
    assert summary["local_socketpairs"] == 3
    assert summary["popen_count"] == 21
    assert len(summary["popen_samples"]) == 15
    assert summary["git_dirty"] is True and summary["git_head"] == "abc"
    json.dumps(summary)


def test_build_summary_unparseable_log_does_not_raise():
    summary = tool.build_summary(
        exit_code=2, elapsed_seconds=0, python="p", root="r", git_head=None, git_dirty=None,
        pytest_args=[], log_text="garbage", denied=[], socketpairs=0, popens=[],
    )
    assert summary["passed"] is None and summary["network_denied"] == {}


def test_default_pytest_args():
    assert tool.DEFAULT_PYTEST_ARGS == ["tests/", "-q", "--tb=short"]


def test_default_out_dir_is_outside_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    out = tool.default_out_dir(root, "20261006T000000Z", tmp_base=tmp_path / "tmpbase")
    assert out == tmp_path / "tmpbase" / "invest-regression" / "repo" / "20261006T000000Z"
    assert not tool.is_inside(out, root)
    # the real default (system temp) must not be inside the checkout either
    real = tool.default_out_dir(_SCRIPT.parent.parent, "x")
    assert not tool.is_inside(real, _SCRIPT.parent.parent)


def test_is_inside_detects_in_repo_paths(tmp_path):
    assert tool.is_inside(tmp_path / "results" / "regression" / "x", tmp_path)
    assert tool.is_inside(tmp_path, tmp_path)
    assert not tool.is_inside(tmp_path.parent / "other", tmp_path)
