"""Unit tests for `server._surface_errors` -- the boundary that decides
whether a raised exception reaches the client as a labelled domain
condition or as an `내부 오류:`-prefixed bug report.

These exercise the wrapper directly against plain functions rather than
going through a real tool call, so the test doesn't need a live MCP
session/transport.
"""

from __future__ import annotations

from mcp.server.mcpserver.exceptions import ToolError
import pytest

from stock_advisor_mcp import server
from stock_advisor_mcp.freshness import StaleDataError


def test_plain_value_error_is_treated_as_a_bug():
    """A bare `ValueError` (e.g. from pandas/numpy, or from `int(nan)`) is
    not a project-defined domain error -- it must surface as a bug, never
    be mistaken for a data condition."""

    @server._surface_errors
    def boom():
        raise ValueError("boom")

    with pytest.raises(ToolError) as excinfo:
        boom()
    text = str(excinfo.value)
    assert text.startswith("내부 오류:")
    assert "ValueError" in text
    assert "boom" in text


def test_not_implemented_error_is_treated_as_a_bug():
    """`NotImplementedError` is a subclass of `RuntimeError` -- an
    unambiguous bug that must never be classified as a domain error."""

    @server._surface_errors
    def boom():
        raise NotImplementedError("not done yet")

    with pytest.raises(ToolError) as excinfo:
        boom()
    text = str(excinfo.value)
    assert text.startswith("내부 오류:")
    assert "NotImplementedError" in text


def test_domain_error_is_not_labelled_as_a_bug():
    """A real project domain error (`AdvisorError` subclass) must reach
    the client with its type name and reason, and must never carry the
    `내부 오류:` bug marker."""

    @server._surface_errors
    def boom():
        raise StaleDataError("005930.KQ: stale data")

    with pytest.raises(ToolError) as excinfo:
        boom()
    text = str(excinfo.value)
    assert "내부 오류" not in text
    assert "StaleDataError" in text
    assert "005930.KQ" in text
