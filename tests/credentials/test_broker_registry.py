"""Per-user broker resolution and caching (Phase 2).

The cache is not a performance detail here. Every broker instance owns its own
request throttle, so "how many instances exist" IS "how many requests per
second we send KIS" — one instance too many silently bursts past the
documented limit and earns an EGW00201. app/brokers/__init__.py's comment
covers the history; these tests pin the behavior.
"""

from __future__ import annotations

import asyncio

import pytest

from app import brokers
from app.brokers.kis.const import token_cache_path_for
from app.config import settings
from app.domain.models import KISCredentials
from app.user_context import user_scope


def _creds(user_id: str, mode: str = "paper", app_key: str = "key-1") -> KISCredentials:
    return KISCredentials(
        user_id=user_id, account_mode=mode, app_key=app_key, app_secret="secret",
        account_8="12345678", account_prod="01", hts_id=None,
    )


class _FakeBroker:
    def __init__(self, tag):
        self.tag = tag
        self.closed = False

    async def aclose(self):
        self.closed = True


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(brokers, "_broker_instances", {})
    monkeypatch.setattr(settings, "kis_account_mode", "paper")


def _factory(record: list):
    async def make(user_id: str, account_mode: str):
        record.append((user_id, account_mode))
        return _FakeBroker((user_id, account_mode))

    return make


async def test_resolves_the_current_identity_with_no_arguments(monkeypatch):
    """The Phase 1 seam paying off: a route handler calls `get_broker()` bare
    and gets the logged-in person's broker."""
    calls: list = []
    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": _factory(calls)})

    with user_scope("u-alice"):
        broker = await brokers.get_broker("kis")
    assert broker.tag == ("u-alice", "paper")
    assert calls == [("u-alice", "paper")]


async def test_two_users_get_different_instances(monkeypatch):
    calls: list = []
    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": _factory(calls)})

    with user_scope("u-alice"):
        a = await brokers.get_broker("kis")
    with user_scope("u-bob"):
        b = await brokers.get_broker("kis")
    assert a is not b
    assert calls == [("u-alice", "paper"), ("u-bob", "paper")]


async def test_same_user_reuses_one_instance(monkeypatch):
    """The throttle invariant: repeat callers must share an instance, or each
    starts its own rate-limit clock from zero."""
    calls: list = []
    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": _factory(calls)})

    with user_scope("u-alice"):
        first = await brokers.get_broker("kis")
        second = await brokers.get_broker("kis")
    assert first is second
    assert len(calls) == 1


async def test_paper_and_real_are_separate_instances(monkeypatch):
    calls: list = []
    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": _factory(calls)})

    with user_scope("u-alice"):
        paper = await brokers.get_broker("kis", account_mode="paper")
        real = await brokers.get_broker("kis", account_mode="real")
    assert paper is not real
    assert calls == [("u-alice", "paper"), ("u-alice", "real")]


async def test_concurrent_cache_misses_construct_exactly_one(monkeypatch):
    """The regression this cache exists to prevent. Without the construction
    lock, N coroutines missing together each build an instance; one wins the
    dict and the losers are handed live clients with independent throttles —
    exactly the EGW00201 burst, and only under concurrency, which is when it
    actually matters."""
    calls: list = []

    async def slow_factory(user_id: str, account_mode: str):
        calls.append((user_id, account_mode))
        await asyncio.sleep(0)  # force a scheduling point mid-construction
        return _FakeBroker((user_id, account_mode))

    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": slow_factory})

    async def fetch():
        with user_scope("u-alice"):
            return await brokers.get_broker("kis")

    results = await asyncio.gather(*(fetch() for _ in range(8)))
    assert len(calls) == 1
    assert all(r is results[0] for r in results)


async def test_unconfigured_user_gets_none_and_it_is_cached(monkeypatch):
    """None means 'this user has no KIS credentials' — never 'fall back to
    someone else's'. Caching None too keeps an unconfigured user from
    re-querying Postgres on every single request."""
    calls: list = []

    async def none_factory(user_id: str, account_mode: str):
        calls.append(user_id)
        return None

    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": none_factory})
    with user_scope("u-nobody"):
        assert await brokers.get_broker("kis") is None
        assert await brokers.get_broker("kis") is None
    assert len(calls) == 1


async def test_invalidate_evicts_and_closes(monkeypatch):
    calls: list = []
    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": _factory(calls)})

    with user_scope("u-alice"):
        first = await brokers.get_broker("kis")
    await brokers.invalidate_broker_cache("u-alice")
    # Closing matters as much as evicting: the instance holds an httpx
    # connection pool, so dropping the reference alone leaks sockets on every
    # credential update.
    assert first.closed is True

    with user_scope("u-alice"):
        second = await brokers.get_broker("kis")
    assert second is not first
    assert len(calls) == 2


async def test_invalidate_leaves_other_users_alone(monkeypatch):
    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": _factory([])})

    with user_scope("u-alice"):
        alice = await brokers.get_broker("kis")
    with user_scope("u-bob"):
        bob = await brokers.get_broker("kis")

    await brokers.invalidate_broker_cache("u-alice")
    assert alice.closed is True
    assert bob.closed is False
    with user_scope("u-bob"):
        assert await brokers.get_broker("kis") is bob


async def test_invalidate_can_target_one_mode(monkeypatch):
    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": _factory([])})

    with user_scope("u-alice"):
        paper = await brokers.get_broker("kis", account_mode="paper")
        real = await brokers.get_broker("kis", account_mode="real")
    await brokers.invalidate_broker_cache("u-alice", "paper")
    assert paper.closed is True
    assert real.closed is False


async def test_invalidate_evicts_even_if_close_raises(monkeypatch):
    """Eviction must not depend on teardown succeeding — a broker whose
    aclose() throws would otherwise stay cached with stale credentials
    forever, which is the opposite of what a key rotation asked for."""
    class _BadBroker(_FakeBroker):
        async def aclose(self):
            raise RuntimeError("connection already gone")

    async def factory(user_id, account_mode):
        return _BadBroker((user_id, account_mode))

    monkeypatch.setattr(brokers, "_BROKER_FACTORIES", {"kis": factory})
    with user_scope("u-alice"):
        first = await brokers.get_broker("kis")
    await brokers.invalidate_broker_cache("u-alice")
    with user_scope("u-alice"):
        assert await brokers.get_broker("kis") is not first


async def test_unknown_broker_name_raises():
    with pytest.raises(ValueError, match="unknown broker"):
        await brokers.get_broker("not-a-broker")


# ---------------------------------------------------------------- token cache

def test_token_cache_path_differs_per_app_key():
    """Two users sharing one cache file don't merely re-fetch: KIS allows one
    token issuance per minute per app key, so they invalidate each other and
    then get rate-limited on the retry."""
    assert token_cache_path_for("key-alice") != token_cache_path_for("key-bob")


def test_token_cache_path_is_stable_for_one_app_key():
    """Stability is the point — the daemon and a CLI run under the same
    credentials must land on the same file and reuse one token, which is why
    this cache is on disk rather than in memory."""
    assert token_cache_path_for("key-alice") == token_cache_path_for("key-alice")


def test_token_cache_path_does_not_contain_the_app_key():
    """The filename shows up in directory listings, backups and tracebacks."""
    assert "supersecretappkey" not in str(token_cache_path_for("supersecretappkey"))


def test_paper_and_real_separate_via_their_different_app_keys():
    paper, real = _creds("u1", "paper", "paper-key"), _creds("u1", "real", "real-key")
    assert token_cache_path_for(paper.app_key) != token_cache_path_for(real.app_key)


def test_token_cache_dir_follows_env_var(monkeypatch, tmp_path):
    """KIS_TOKEN_CACHE_DIR relocates the cache onto a persistent volume so a
    token survives the daemon's stop/rm/run redeploy instead of being
    re-issued every deploy (KIS restricts frequent re-issuance). The path is
    resolved per call (token_cache_file), not frozen at import, so setting
    the env var mid-process actually takes effect."""
    monkeypatch.setenv("KIS_TOKEN_CACHE_DIR", str(tmp_path))
    assert token_cache_path_for("some-key").parent == tmp_path


def test_token_cache_dir_defaults_to_app_tree_when_unset(monkeypatch):
    """Unset (local dev, tests) keeps the previous location — the `app`
    package dir — so behaviour there is unchanged."""
    monkeypatch.delenv("KIS_TOKEN_CACHE_DIR", raising=False)
    assert token_cache_path_for("some-key").parent.name == "app"


def test_token_cache_dir_is_explicit_tracks_the_env_var(monkeypatch, tmp_path):
    from app.brokers.kis.const import token_cache_dir_is_explicit

    monkeypatch.delenv("KIS_TOKEN_CACHE_DIR", raising=False)
    assert token_cache_dir_is_explicit() is False
    monkeypatch.setenv("KIS_TOKEN_CACHE_DIR", str(tmp_path))
    assert token_cache_dir_is_explicit() is True
    monkeypatch.setenv("KIS_TOKEN_CACHE_DIR", "")
    assert token_cache_dir_is_explicit() is False


# ------------------------------------------------------- credentials -> broker

def test_broker_mode_comes_from_credentials_not_global_settings(monkeypatch):
    """A stale settings.kis_account_mode must not be able to drive a real
    credential set with paper TR IDs, or point it at the paper host."""
    from app.brokers.kis.broker import broker_from_credentials

    monkeypatch.setattr(settings, "kis_account_mode", "paper")
    broker = broker_from_credentials(_creds("u1", "real", "real-key"))
    assert broker._is_real is True
    assert broker._base_url == settings.kis_real_base_url.rstrip("/")


def test_credentials_repr_hides_the_secret():
    """A dataclass's default repr would put app_secret into every log line,
    traceback and pytest diff that touches one of these."""
    text = repr(_creds("u1", "paper", "very-secret-key"))
    assert "very-secret-key" not in text
    assert "secret" not in text.replace("app_secret=***", "").replace("KISCredentials", "")
    assert "u1" in text
