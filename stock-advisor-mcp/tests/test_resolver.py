import pytest

from stock_advisor_mcp import resolver
from stock_advisor_mcp.krx_master import MasterLookupResult

_FAKE_DB = {
    "005930": MasterLookupResult(code="005930", market="KOSPI", name_ko="삼성전자"),
    "036540": MasterLookupResult(code="036540", market="KOSDAQ", name_ko="SFA반도체"),
    "379800": MasterLookupResult(code="379800", market="KOSPI", name_ko="KODEX 미국S&P500"),
    "000020": MasterLookupResult(code="000020", market="KOSPI", name_ko="동화약품"),
    "999999": MasterLookupResult(code="999999", market="KOSDAQ", name_ko="동화테스트"),
}


class FakeMaster:
    def ensure_loaded(self):
        pass

    def lookup_code(self, code):
        return _FAKE_DB.get(code)

    def search_name(self, query, limit=8):
        q = query.strip().lower()
        return [v for v in _FAKE_DB.values() if q in v.name_ko.lower()][:limit]


@pytest.fixture(autouse=True)
def fake_master(monkeypatch):
    monkeypatch.setattr(resolver, "get_master", lambda: FakeMaster())


def _fake_quote_type(mapping):
    def get_info(yf_symbol):
        return {"quoteType": mapping.get(yf_symbol)}

    return get_info


def test_resolve_by_bare_code_kospi(monkeypatch):
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({"005930.KS": "EQUITY"}))
    result = resolver.resolve_ticker("005930")
    assert result.market == "KOSPI"
    assert result.yf_symbol == "005930.KS"
    assert result.name_ko == "삼성전자"
    assert result.is_etf is False


def test_resolve_by_bare_code_kosdaq_gets_kq_suffix(monkeypatch):
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({"036540.KQ": "EQUITY"}))
    result = resolver.resolve_ticker("036540")
    assert result.market == "KOSDAQ"
    assert result.yf_symbol == "036540.KQ"


def test_resolve_ignores_caller_supplied_suffix_and_uses_master_file(monkeypatch):
    # Caller wrongly appends .KS to a KOSDAQ code -- the resolver must extract
    # the bare code and re-derive the suffix from the master file, not trust
    # the caller's guess.
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({"036540.KQ": "EQUITY"}))
    result = resolver.resolve_ticker("036540.KS")
    assert result.yf_symbol == "036540.KQ"


def test_resolve_etf_flag_set_from_yfinance_quote_type(monkeypatch):
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({"379800.KS": "ETF"}))
    result = resolver.resolve_ticker("379800")
    assert result.is_etf is True


def test_resolve_unknown_code_raises(monkeypatch):
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({}))
    with pytest.raises(resolver.TickerNotFoundError):
        resolver.resolve_ticker("123123")


def test_resolve_by_unique_name_match(monkeypatch):
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({"005930.KS": "EQUITY"}))
    result = resolver.resolve_ticker("삼성전자")
    assert result.code == "005930"


def test_resolve_ambiguous_name_raises(monkeypatch):
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({}))
    with pytest.raises(resolver.AmbiguousTickerError):
        resolver.resolve_ticker("동화")


def test_resolve_raises_when_yfinance_has_no_quote_type(monkeypatch):
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({}))
    with pytest.raises(resolver.TickerNotFoundError):
        resolver.resolve_ticker("005930")


def test_data_coverage_present_on_every_result(monkeypatch):
    monkeypatch.setattr(resolver.yf_gateway, "get_info", _fake_quote_type({"005930.KS": "EQUITY"}))
    result = resolver.resolve_ticker("005930")
    assert len(result.data_coverage) > 0
    assert any("VI" in item for item in result.data_coverage)
