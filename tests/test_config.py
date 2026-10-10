"""AIProfile axis (app/config.py) — default value and the fail-safe boot
guard blocking a non-production profile paired with the real KIS account
unless explicitly acknowledged."""

from __future__ import annotations

import pytest

from app.config import (
    AIProfile,
    Settings,
    account_mode_scope,
    normalize_scope,
    reset_scoped_lists,
    scoped_lists_snapshot,
    settings,
)
from app.user_context import user_scope


def test_ai_profile_defaults_to_production():
    s = Settings()
    assert s.ai_profile is AIProfile.PRODUCTION
    assert s.ai_profile_live_ack is False


def test_overseas_trading_enabled_defaults_true():
    s = Settings()
    assert s.overseas_trading_enabled is True


def test_test_aggressive_with_real_account_refuses_to_boot_without_ack():
    with pytest.raises(ValueError, match="ai_profile_live_ack"):
        Settings(ai_profile=AIProfile.TEST_AGGRESSIVE, kis_account_mode="real")


def test_debug_with_real_account_refuses_to_boot_without_ack():
    with pytest.raises(ValueError, match="ai_profile_live_ack"):
        Settings(ai_profile=AIProfile.DEBUG, kis_account_mode="real")


def test_test_aggressive_with_real_account_boots_when_acked():
    s = Settings(
        ai_profile=AIProfile.TEST_AGGRESSIVE, kis_account_mode="real", ai_profile_live_ack=True,
        kis_max_order_krw_real=1.0, kis_max_order_usd_real=1.0,
        kis_approval_threshold_krw_real=1.0, kis_approval_threshold_usd_real=1.0,
    )
    assert s.ai_profile is AIProfile.TEST_AGGRESSIVE
    assert s.kis_account_mode == "real"


def test_non_production_profile_with_paper_account_boots_without_ack():
    s = Settings(ai_profile=AIProfile.TEST_AGGRESSIVE, kis_account_mode="paper")
    assert s.ai_profile is AIProfile.TEST_AGGRESSIVE


# ---- LLM provider selector (openrouter-integration-idea-2026-08-27) -------
# `_env_file=None` so these don't depend on a developer's local .env.local
# (which during the gateway PoC carries LLM_PROVIDER / LLM_API_KEY).


def test_llm_provider_defaults_to_vllm():
    assert Settings(_env_file=None).llm_provider == "vllm"


def test_gateway_provider_without_api_key_refuses_to_boot():
    with pytest.raises(ValueError, match="llm_api_key"):
        Settings(_env_file=None, llm_provider="gateway")


def test_gateway_provider_boots_with_api_key():
    s = Settings(_env_file=None, llm_provider="gateway", llm_api_key="gw-test")
    assert s.llm_provider == "gateway"
    assert s.llm_api_key == "gw-test"


def test_vllm_provider_needs_no_api_key():
    assert Settings(_env_file=None, llm_provider="vllm").llm_api_key is None


# ---- DATABASE_URL boot gate (app/config.py) -------------------------------
# .env.example ships DATABASE_URL blank; an empty value would otherwise reach
# asyncpg.connect("") and silently fall back to libpq defaults instead of
# failing clearly. `_env_file=None` so a developer's real .env.local can't
# mask these.


def test_database_url_empty_refuses_to_boot():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings(_env_file=None, database_url="")


def test_database_url_whitespace_only_refuses_to_boot():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings(_env_file=None, database_url="   ")


def test_database_url_non_postgres_scheme_refuses_to_boot():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings(_env_file=None, database_url="mysql://host/db")


def test_database_url_default_boots():
    assert Settings(_env_file=None).database_url.startswith("postgresql://")


def test_database_url_postgres_scheme_alias_boots():
    s = Settings(_env_file=None, database_url="postgres://localhost:5434/trading_memory")
    assert s.database_url == "postgres://localhost:5434/trading_memory"


# ---- paper/real order-value caps & approval thresholds (app/config.py) ---
# kis_max_order_krw/usd and kis_approval_threshold_krw/usd used to be single
# global values tuned against the paper account's own equity history — a
# real account of a different size would silently inherit a mis-scaled cap.
# Each is now a _paper/_real pair, resolved through a same-named property
# picked by kis_account_mode, with a boot-time guard refusing a real-mode
# boot until all four _real values are set explicitly.

_REAL_MONEY_KWARGS = dict(
    kis_max_order_krw_real=1_000_000.0, kis_max_order_usd_real=1_000.0,
    kis_approval_threshold_krw_real=500_000.0, kis_approval_threshold_usd_real=500.0,
)


def test_real_account_refuses_to_boot_without_real_money_limits():
    with pytest.raises(ValueError, match="kis_max_order_krw_real"):
        Settings(kis_account_mode="real")


def test_real_account_refuses_to_boot_with_only_some_real_money_limits_set():
    with pytest.raises(ValueError, match="kis_approval_threshold_usd_real"):
        Settings(
            kis_account_mode="real",
            kis_max_order_krw_real=1_000_000.0, kis_max_order_usd_real=1_000.0,
            kis_approval_threshold_krw_real=500_000.0,
        )


def test_real_account_boots_once_all_four_real_money_limits_are_set():
    s = Settings(kis_account_mode="real", **_REAL_MONEY_KWARGS)
    assert s.kis_max_order_krw == 1_000_000.0
    assert s.kis_max_order_usd == 1_000.0
    assert s.kis_approval_threshold_krw == 500_000.0
    assert s.kis_approval_threshold_usd == 500.0


def test_paper_account_boots_without_any_real_money_limits():
    s = Settings(kis_account_mode="paper")
    assert s.kis_max_order_krw == s.kis_max_order_krw_paper


def test_paper_and_real_money_limits_resolve_independently():
    s = Settings(kis_account_mode="paper", kis_max_order_krw_paper=2_000_000.0, **_REAL_MONEY_KWARGS)
    assert s.kis_max_order_krw == 2_000_000.0  # paper value, not the real one set alongside it


def test_setting_kis_max_order_krw_writes_the_currently_active_mode():
    s = Settings(kis_account_mode="real", **_REAL_MONEY_KWARGS)
    s.kis_max_order_krw = 3_000_000.0
    assert s.kis_max_order_krw_real == 3_000_000.0
    assert s.kis_max_order_krw_paper != 3_000_000.0  # untouched


# ---- paper/real KIS request-rate throttle (app/config.py) ----------------
# kis_min_request_interval used to be one shared value (1.0s, tuned for the
# paper account's tighter 1 req/sec cap) — switching kis_account_mode to
# "real" left the app stuck at that same slow rate instead of using the real
# account's much higher 18 req/sec limit. Now a _paper/_real pair, resolved
# through a same-named property, same convention as the order-value caps
# above — but unlike those, there's no boot-time guard requiring an explicit
# _real value, since a wrong rate-limit guess is never a money-safety risk
# (only ever "too slow," never "too permissive").


def test_min_request_interval_defaults_to_paper_rate_in_paper_mode():
    s = Settings(kis_account_mode="paper")
    assert s.kis_min_request_interval == s.kis_min_request_interval_paper == 1.0


def test_min_request_interval_uses_the_real_rate_in_real_mode():
    s = Settings(kis_account_mode="real", **_REAL_MONEY_KWARGS)
    assert s.kis_min_request_interval == s.kis_min_request_interval_real
    assert s.kis_min_request_interval_real != s.kis_min_request_interval_paper


def test_min_request_interval_real_default_within_kis_recommended_range():
    # KIS recommends 100~150ms spacing for concurrent calls; 0.07 still
    # produced EGW00201 on the live real account (2026-10-02).
    s = Settings(kis_account_mode="real", **_REAL_MONEY_KWARGS)
    assert 0.10 <= s.kis_min_request_interval_real <= 0.15
    assert s.kis_min_request_interval_real < s.kis_min_request_interval_paper


def test_min_request_interval_paper_and_real_resolve_independently():
    s = Settings(kis_account_mode="paper", kis_min_request_interval_real=0.01, **_REAL_MONEY_KWARGS)
    # Still the paper value even though a (deliberately different) real
    # value was set alongside it — mode picks which field wins, not presence.
    assert s.kis_min_request_interval == 1.0


# ---- DB command timeout, separated from the connect timeout (app/config.py) ----
# database_command_timeout_seconds used to not exist — app.memory.db passed
# database_connect_timeout_seconds (5s) as asyncpg's command_timeout too,
# meaning every query (and the RESET run on every pool release) got the
# same 5s budget as the connect step, even though Postgres's own
# statement_timeout allows 120s. Confirmed live 2026-08-06 as the #1 cause
# of that day's kill_switch_status/daily_loss_status warning spam.


def test_command_timeout_defaults_longer_than_statement_timeout():
    s = Settings()
    assert s.database_command_timeout_seconds > s.database_statement_timeout_seconds


def test_command_timeout_is_independent_of_connect_timeout():
    s = Settings(database_connect_timeout_seconds=0.5)
    assert s.database_command_timeout_seconds != 0.5


# ---- watchlist / trade_allowlist split (app/config.py) -------------------
# trade_allowlist is split into two real fields, trade_allowlist_domestic
# and trade_allowlist_overseas (env TRADE_ALLOWLIST_DOMESTIC/_OVERSEAS),
# matching every other domestic/overseas separation in the project. watchlist
# (renamed from favorites, 2026-08-05) stays a single (non-market-split)
# superset, with trade_allowlist ⊆ watchlist.
# The transitional `legacy_watchlist_env`/`WATCHLIST` bridge (and its
# migration validator) has been fully removed (2026-08-05) — no .env file
# still sets bare WATCHLIST and no persisted DB row used the old unscoped
# key, so there was nothing left for it to migrate. Removing it also frees
# the env var name `WATCHLIST` for unrelated future use.


# The four ticker lists are per (user, account_mode) properties now, not
# pydantic fields (2026-08-06, remaining Phase 5 work). These tests used to
# construct `Settings(trade_allowlist_domestic=[...])` and assert what the
# three model_validators did at config-construction time. Both halves of that
# are gone: the kwargs are no longer fields (and `extra="ignore"` means they
# would be silently swallowed, which is why these had to be rewritten rather
# than left to fail), and the validators became `config.normalize_scope`,
# called where stored data now enters — `trade_allowlist_store.load`.
# 네 리스트가 pydantic 필드에서 (사용자, 모드) 프로퍼티로 바뀜 — 생성 시점 검증기는
# normalize_scope로 이동.


@pytest.fixture(autouse=True)
def _clean_scopes():
    reset_scoped_lists()
    yield
    reset_scoped_lists()


def test_trade_allowlist_combines_domestic_and_overseas():
    settings.trade_allowlist_domestic = ["005930.KS"]
    settings.trade_allowlist_overseas = ["AAPL", "MSFT"]
    assert settings.trade_allowlist == ["005930.KS", "AAPL", "MSFT"]


def test_trade_allowlist_for_market_returns_the_matching_half():
    settings.trade_allowlist_domestic = ["005930.KS"]
    settings.trade_allowlist_overseas = ["AAPL"]
    assert settings.trade_allowlist_for_market("domestic") == ["005930.KS"]
    assert settings.trade_allowlist_for_market("overseas") == ["AAPL"]


def test_a_fresh_scope_starts_completely_empty():
    """Deliberate change from the old fresh-install default, which seeded
    trade_allowlist_overseas with AAPL.

    Carrying that into per-scope defaults would give **every new user** a
    ticker on their trading allowlist that they never chose — and the allowlist
    is the gate kis_bridge checks, on that user's own real credentials. The old
    rationale (an empty dashboard looks broken) is also covered now by Phase
    3's explicit onboarding flow.
    신규 사용자가 고르지도 않은 종목에 매매 권한이 생기는 것을 막기 위함."""
    assert settings.trade_allowlist_domestic == []
    assert settings.trade_allowlist_overseas == []
    assert settings.watchlist == []
    assert settings.trade_allowlist_auto_added == []
    assert settings.trade_allowlist == []


def test_lists_are_isolated_per_user():
    """The point of the whole change. This list grants trading permission, so
    one user's entry must never appear in another's."""
    with user_scope("u-alice"):
        settings.trade_allowlist_domestic = ["005930.KS"]
    with user_scope("u-bob"):
        assert settings.trade_allowlist_domestic == []
    with user_scope("u-alice"):
        assert settings.trade_allowlist_domestic == ["005930.KS"]


def test_lists_are_isolated_per_account_mode():
    """A ticker approved for 모의투자 must not be tradable on real money."""
    with user_scope("u-alice"), account_mode_scope("paper"):
        settings.trade_allowlist_overseas = ["TSLA"]
    with user_scope("u-alice"), account_mode_scope("real"):
        assert settings.trade_allowlist_overseas == []


def test_writing_a_list_does_not_alias_the_callers_list():
    """The setter copies. Otherwise a caller mutating the list it passed would
    be editing another scope's state through a stale reference."""
    mine = ["AAPL"]
    with user_scope("u-alice"):
        settings.trade_allowlist_overseas = mine
        mine.append("TSLA")
        assert settings.trade_allowlist_overseas == ["AAPL"]


def test_normalize_scope_folds_allowlist_entries_into_the_watchlist():
    """Was `_watchlist_superset_of_trade_allowlist`. A ticker with trading
    permission that is missing from the lighter tracking list is incoherent, and
    at load time there is no human present to confirm a promotion."""
    with user_scope("u-alice"):
        settings.trade_allowlist_domestic = ["005930.KS"]
        settings.trade_allowlist_overseas = ["AAPL"]
        settings.watchlist = ["AAPL"]
    normalize_scope("u-alice", settings.kis_account_mode)
    with user_scope("u-alice"):
        assert set(settings.watchlist) == {"005930.KS", "AAPL"}


def test_normalize_scope_leaves_a_strict_superset_watchlist_alone():
    with user_scope("u-alice"):
        settings.trade_allowlist_overseas = ["AAPL"]
        settings.watchlist = ["AAPL", "TSLA"]
    normalize_scope("u-alice", settings.kis_account_mode)
    with user_scope("u-alice"):
        assert set(settings.watchlist) == {"AAPL", "TSLA"}


def test_normalize_scope_keeps_auto_added_entries_still_on_the_allowlist():
    with user_scope("u-alice"):
        settings.trade_allowlist_overseas = ["AAPL"]
        settings.trade_allowlist_auto_added = ["AAPL"]
    normalize_scope("u-alice", settings.kis_account_mode)
    with user_scope("u-alice"):
        assert settings.trade_allowlist_auto_added == ["AAPL"]


def test_normalize_scope_drops_stale_auto_added_entries():
    """Was `_auto_added_subset_of_trade_allowlist`. A marker for a ticker no
    longer on the allowlist is meaningless, and left alone it would make the
    ticker look auto-prunable forever."""
    with user_scope("u-alice"):
        settings.trade_allowlist_overseas = ["AAPL"]
        settings.trade_allowlist_auto_added = ["TSLA"]
    normalize_scope("u-alice", settings.kis_account_mode)
    with user_scope("u-alice"):
        assert settings.trade_allowlist_auto_added == []


def test_normalize_scope_is_a_noop_for_an_untouched_scope():
    normalize_scope("u-nobody", "paper")  # must not raise or create the scope
    assert ("u-nobody", "paper") not in scoped_lists_snapshot()


def test_interval_helpers_key_off_is_real_not_env_mode():
    # broker_from_credentials builds a broker from a credential set, and its
    # timing must follow that set's is_real — not settings.kis_account_mode.
    s = Settings(_env_file=None, kis_account_mode="paper")
    assert s.kis_min_request_interval_for(True) == s.kis_min_request_interval_real
    assert s.kis_min_request_interval_for(False) == s.kis_min_request_interval_paper
    assert s.kis_account_inquiry_min_interval_for(True) == s.kis_account_inquiry_min_interval_real
    assert s.kis_account_inquiry_min_interval_for(False) == 0.0  # paper lane disabled
    # real inquiry lane is a real, non-zero spacing
    assert s.kis_account_inquiry_min_interval_real > 0


def test_memory_krx_fallback_since_defaults_to_off_and_tolerates_blank_env(monkeypatch):
    from datetime import date

    from app.config import Settings

    monkeypatch.delenv("MEMORY_KRX_FALLBACK_SINCE", raising=False)
    assert Settings().memory_krx_fallback_since is None
    # An empty `MEMORY_KRX_FALLBACK_SINCE=` line must not crash the daemon at import time.
    assert Settings(memory_krx_fallback_since="").memory_krx_fallback_since is None
    assert Settings(memory_krx_fallback_since="   ").memory_krx_fallback_since is None
    assert Settings(memory_krx_fallback_since="2026-09-29").memory_krx_fallback_since == date(2026, 9, 29)
    monkeypatch.setenv("MEMORY_KRX_FALLBACK_SINCE", "")
    assert Settings().memory_krx_fallback_since is None
    monkeypatch.setenv("MEMORY_KRX_FALLBACK_SINCE", "2026-10-01")
    assert Settings().memory_krx_fallback_since == date(2026, 10, 1)


def test_memory_krx_fallback_since_tolerates_a_human_hand_editing_the_server_env(monkeypatch):
    # docker's --env-file does NOT strip quotes or trailing whitespace/comments -- a human
    # editing the server .env.dev by hand (e.g. a trailing space or wrapping the value in
    # quotes, both easy typos) must still boot cleanly rather than crash-loop the live daemon.
    from datetime import date

    from app.config import Settings

    for raw in ("2026-09-30 ", " 2026-09-30", '"2026-09-30"', "'2026-09-30'", "  2026-09-30  "):
        assert Settings(memory_krx_fallback_since=raw).memory_krx_fallback_since == date(2026, 9, 30), raw
    for blank in ('""', "''", "  "):
        assert Settings(memory_krx_fallback_since=blank).memory_krx_fallback_since is None, blank
