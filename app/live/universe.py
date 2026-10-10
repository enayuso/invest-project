"""Candidate-ticker universe for app.live.auto_trade_loop — a bounded set to
recommend/trade from instead of "every possible stock." Domestic (KRX) and
overseas (US) are kept as two entirely separate discovery feeds, matching
this project's existing domestic/overseas separation everywhere else
(app.execution.ledger's per-ticker currency-converted cash pools, the
dashboard's two distinct account blocks, and app.brokers.kis's own separate
DomesticRankingMixin/OverseasRankingMixin classes) — never pooled into one
combined ranking.
국내/해외를 하나로 합치지 않고 완전히 분리된 두 개의 발굴 소스로 유지 (다른 곳의 기존 분리 원칙과 동일).
"""

from __future__ import annotations

import logging

from app.domain.leverage_filter import is_leveraged_or_inverse
from app.domain.product_filter import is_out_of_scope_product
from app.live import ticker_names

logger = logging.getLogger("app.live.universe")


def _krx_discovery_tickers(ranked) -> list[str]:
    """The admission filter shared by every KRX discovery feed in this
    module — extracted 2026-09-02 (screening-improvement-research #3) when
    the feed count went from 2 (krx_top_n, krx_market_cap_top_n) to 4
    (krx_fluctuation_losers_n, krx_disparity_low_n added). With two copies,
    a divergence between them was survivable — easy to spot, cheap to fix.
    With four, it's a matter of time before one gets a fix the others
    don't, so this was pulled out before that happened rather than after.

    See krx_top_n's own docstring for the canonical explanation of what
    each rejection is for (non-6-digit codes, leveraged/inverse names,
    out-of-scope cash/bond/covered-call products) and the deliberate
    empty-name leniency (is_leveraged_or_inverse("")/is_out_of_scope_
    product("") both return False — "unknown isn't evidence either way").
    Pure apart from the ticker_names.remember(...) memoization side effect,
    which every caller wants anyway (display-name lookups downstream)."""
    tickers = []
    for r in ranked:
        if (not r.code or not r.code.isdigit() or len(r.code) != 6
                or is_leveraged_or_inverse(r.name) or is_out_of_scope_product(r.name)):
            continue
        ticker = f"{r.code}.KS"
        ticker_names.remember(ticker, r.name)
        tickers.append(ticker)
    return tickers


async def krx_top_n(broker, count: int, *, max_price: float | None = None) -> list[str]:
    """Top `count` KRX tickers by cumulative trading value (거래대금) right
    now, in this project's `.KS`-suffixed ticker format (RankedStock.code is
    the bare 6-digit KRX code). Trading value rather than raw share volume —
    a low-price/high-share-count name doesn't crowd out where the real money
    is trading today the way it would by share count alone. Empty list if no
    broker is configured or the KIS call fails — a discovery feed going down
    shouldn't stop the caller's whole cycle, same "log and continue"
    convention as app.live.poller/state.stop_loss_loop.

    Leveraged/inverse ETFs (KODEX 레버리지, KODEX 200선물인버스2X, ...) are
    routinely among KRX's highest-turnover tickers too, so this feed would
    surface them constantly if left unfiltered — dropped here via
    RankedStock.name (already returned by this same TR, no extra call)
    before the momentum ranking ever sees them. See app.domain.leverage_filter.

    Out-of-scope products are dropped the same way, via
    app.domain.product_filter.is_out_of_scope_product: cash-like / bond /
    bond-mix funds (CD금리·KOFR·머니마켓·국고채·종합채권·채권혼합 ETFs) AND
    covered-call / option-income funds (커버드콜·covered call·buy-write·
    premium income). Institutions park large sums in the cash/bond ones and
    trade the covered-call ones heavily, so both sit near the top of the
    거래대금 ranking, but this system's company-analysis pipeline has no
    earnings/news to work with: the cash/bond ones have a blended near-zero
    ATR that drives a misleading "100% allocation" path, and the covered-call
    ones have an option-delta/theta payoff that momentum/valuation/ATR
    analysis reads backwards. 2026-09-01: 481050.KS
    "KODEX CD1년금리플러스액티브(합성)" reached the analysts this way;
    2026-09-02: 498400.KS "KODEX 200타겟위클리커버드콜" burned a full
    ~17-LLM-call pass the same way (audit id=1170).

    Also drops any code that isn't a plain 6-digit KRX equity/ETF code —
    this TR's turnover ranking isn't limited to ordinary stocks, and a
    non-standard code (an ELW/warrant like "0156T0") sails through the
    name-based leverage filter just fine since its name looks like an
    ordinary product. Downstream (yfinance display-name/price lookups for a
    candidate ticker that made it this far) has no listing for that kind of
    code and fails every time it's rescanned — confirmed via repeated
    'possibly delisted'/404 errors for exactly such codes in practice.

    Note on empty/unresolved names (applies to all four discovery functions
    in this module): is_leveraged_or_inverse("") returns False ("unknown
    isn't evidence either way" — see its own docstring), so a ticker with an
    empty/unresolved name PASSES this discovery-stage filter instead of
    being blocked. This is deliberate, not a gap — two downstream layers
    are intentionally strict about the same condition and block on it
    before any real order: app.execution.kis_bridge.unsupported_product_block
    (the order choke point both entry points share) and
    app.web.state.trade_allowlist_add (manual/screener admission, raises
    UnresolvedTickerNameError). So an unresolved-name ticker that passes
    here can cost at most one wasted LLM analysis pass — it is never
    actually tradeable, since the trade layer refuses it downstream.

    `max_price` (added 2026-09-02, screening-improvement-research #3
    follow-up): an optional account-derived price CEILING, applied
    KIS-side via `FID_INPUT_PRICE_2` and threaded straight into
    `KISBroker.fetch_trading_value_rank` with zero interpretation in this
    module — this function computes nothing about it, only passes it
    through. Confirmed live 2026-09-02: a probe against
    `fetch_market_cap_rank` with `FID_INPUT_PRICE_2="100000"` returned a
    full, non-truncated 30 rows, 28 of them different from the unfiltered
    top-30 — KIS ranks WITHIN the price-bounded set BEFORE its own top-N
    cut, it does not just filter an already-fixed top-30 afterwards. The
    same TR/param applies identically to all four KRX discovery axes in
    this module (see app.brokers.kis.domestic.ranking for the per-TR
    wiring and the "blank" spelling each one uses for "no filter").

    The caller (`app.live.auto_trade_loop._domestic_price_ceiling`) owns
    every bit of the account math that produces this number — this module
    stays free of `app.config`/`app.execution` imports (Domain-First: an
    account's own equity/cash/risk posture is an adapter/domain concern,
    not a discovery-feed one). `max_price=None` (the default) means "no
    ceiling" and is byte-identical to this module's wire behavior before
    this parameter existed.

    What problem it solves: 2026-09-02, a ₩2,000,000 corporate account
    started raising `account_too_small_domestic` every cycle (see
    `_drop_unexecutable` in app.live.auto_trade_loop). The actual binding
    constraint turned out to be the account's OWN 1% risk cap
    (`app.domain.executability.risk_capped_shares`), not "large caps are
    simply too expensive" in the abstract — at this system's shipped risk
    defaults, even 1 share of a ticker becomes unaffordable above roughly
    `equity * 0.01 / (atr_pct * 2.5)`: about ₩400,000 for a ₩2,000,000
    account at 2% ATR, ~₩267,000 at 3% ATR, ~₩200,000 at 4% ATR. The two
    axes added earlier the same day (`krx_fluctuation_losers_n`/
    `krx_disparity_low_n`) widen WHICH names get a look, but neither is
    price-aware — a ₩900,000 name down 5% today still enters the pool and
    is still rejected by the exact same arithmetic once it reaches
    `_drop_unexecutable`. `max_price` is the structural fix: it stops
    KIS from ever handing back names the account's own risk cap could
    never afford in the first place.

    It is a heuristic PRE-filter, never the exact affordability gate: the
    ceiling assumes ONE volatility figure for the WHOLE ranking, because a
    real ticker's ATR is only knowable after it has already been picked —
    exactly the chicken-and-egg this breaks (see
    `app.domain.executability.affordable_price_ceiling`'s own docstring
    for the full derivation and the deliberate asymmetry in how generous
    it is). `app.live.auto_trade_loop._drop_unexecutable` remains the only
    EXACT per-candidate gate, using each surviving name's REAL ATR once
    it's known, and must not be removed or relaxed just because this
    upstream heuristic exists — see that function's own docstring for why."""
    if broker is None:
        return []
    try:
        ranked = await broker.fetch_trading_value_rank(count=count, max_price=max_price)
    except Exception:
        logger.warning("krx_top_n: fetch_trading_value_rank failed — returning no candidates", exc_info=True)
        return []
    return _krx_discovery_tickers(ranked)


async def krx_market_cap_top_n(broker, count: int, *, max_price: float | None = None) -> list[str]:
    """Top `count` KRX tickers by current market cap (시가총액), via
    `KISBroker.fetch_market_cap_rank` — a stable large-cap pool, distinct
    from krx_top_n's turnover-driven one. Added as a second discovery axis
    (2026-07-30) because krx_top_n's top-30 by trading value churns very
    little intraday, so once its names are all already analyzed today
    (app.live.auto_trade_loop's exclude_already_analyzed_today), a cycle
    can go a long time without a genuinely new candidate. In practice
    overlaps HEAVILY with that turnover pool (confirmed live: the top 5 by
    market cap were all already in the top-30 by trading value — large
    caps naturally top both rankings), so this only adds a handful of
    genuinely new names, not a second full 30-name pool — see project
    memory screening-method-improvement-ideas.md's 2026-07-30 update for
    the fuller discussion (this alone doesn't keep a FULL_AUTO day fully
    busy, and that may not even be the right goal). Same filtering/
    empty-on-failure conventions as krx_top_n (see its own docstring,
    including the empty/unresolved-name note).

    `max_price` — see krx_top_n's own docstring for the full account-
    affordability story; same `FID_INPUT_PRICE_2` mechanism, same
    `None`-is-a-no-op guarantee, threaded straight into
    `KISBroker.fetch_market_cap_rank` with zero interpretation here."""
    if broker is None:
        return []
    try:
        ranked = await broker.fetch_market_cap_rank(count=count, max_price=max_price)
    except Exception:
        logger.warning("krx_market_cap_top_n: fetch_market_cap_rank failed — returning no candidates", exc_info=True)
        return []
    return _krx_discovery_tickers(ranked)


async def krx_fluctuation_losers_n(
    broker, count: int, *, min_volume: int = 0, max_price: float | None = None
) -> list[str]:
    """Today's `count` biggest KRX decliners (등락률 하위), via
    `KISBroker.fetch_fluctuation_rank(rising=False, ...)` — a THIRD
    discovery axis alongside krx_top_n's turnover pool and
    krx_market_cap_top_n's market-cap pool.

    Why a third axis: those two both sort on a size/turnover quantity, and
    on KRX those are near-collinear — measured 2026-09-02, their union is
    only ~35 overlapping mega-cap/platform names every cycle, not two
    genuinely different pools. That collinearity causes two concrete live
    problems this feed addresses: (1) the downstream correlation-cluster
    dedup (`app.execution.portfolio_risk.compute_correlation_clusters` via
    `app.chat.screening.dedupe_correlated`) has little to actually dedupe
    when every candidate is a large-cap chip/platform name moving together;
    (2) 2026-09-02, the corporate account's first FULL_AUTO cycle was
    raising `account_too_small_domestic` every cycle (see
    `_drop_unexecutable` in app.live.auto_trade_loop) because a large-cap-
    only pool is, by construction, an expensive-shares-only pool — no
    amount of ranking cleverness produces an affordable candidate from a
    universe that's all six- and seven-figure share prices.

    IMPORTANT — this is an ENTRY axis, not a ranking-direction change.
    Every ticker admitted here is still scored and ranked by the existing
    `sign=+1` momentum score in app.chat.screening, completely unchanged.
    The reversal hypothesis (ranking by `sign=-1`, i.e. picking the bottom
    of the momentum ranking instead of the top) was measured separately and
    REJECTED — project memory screening-reversal-sweep-result-2026-09-02:
    sign=-1 lost at all three tested lookbacks, and the 60-day reversal
    variant lost with a 95% CI that did not even cross zero. So a name that
    enters via this axis and then survives the momentum ranking is a
    pullback that has ALREADY turned back up by the time it's picked — a
    fundamentally different thing from being picked because it's down.
    Axis-choice (which names get a look) and ranking-sign (which of those
    names get picked) are independent decisions; this feed only changes the
    former.

    `min_volume` is passed straight through to KIS's own floor
    (`fetch_fluctuation_rank`'s `min_volume` param), because a 하락률
    ranking, unlike krx_top_n's 거래대금 ranking, is not self-limiting on
    liquidity — see `fetch_disparity_rank`'s docstring for the fuller
    version of this argument (구조적으로 동일).

    Same filtering/empty-list-on-failure/ticker_names-memoization
    convention as krx_top_n — see its own docstring for the four rejection
    reasons and the deliberate empty-name leniency, not repeated here.

    `max_price` — see krx_top_n's own docstring for the full account-
    affordability story; same `FID_INPUT_PRICE_2` mechanism, same
    `None`-is-a-no-op guarantee, `blank=""` (this TR's own no-filter
    spelling — see fetch_fluctuation_rank's docstring)."""
    if broker is None:
        return []
    try:
        ranked = await broker.fetch_fluctuation_rank(
            rising=False, count=count, min_volume=min_volume, max_price=max_price
        )
    except Exception:
        logger.warning(
            "krx_fluctuation_losers_n: fetch_fluctuation_rank failed — returning no candidates",
            exc_info=True,
        )
        return []
    return _krx_discovery_tickers(ranked)


async def krx_disparity_low_n(
    broker, count: int, *, window_days: int = 20, min_volume: int = 0, max_price: float | None = None
) -> list[str]:
    """Today's `count` KRX names with the lowest 이격도 (price furthest
    BELOW their own `window_days`-day moving average), via
    `KISBroker.fetch_disparity_rank(low=True, ...)` — the FOURTH discovery
    axis in this module, and the only one of the four whose sort key has
    NEITHER a size term (unlike krx_top_n/krx_market_cap_top_n) NOR a
    same-day-move term (unlike krx_fluctuation_losers_n's 등락률, which is
    just today's single-day return). "Furthest below its own trend" is
    measured over `window_days` days, so this is structurally the
    least-correlated addition of the four — see
    krx_fluctuation_losers_n's docstring for the collinearity measurement
    (turnover/market-cap union ~35 overlapping mega-caps) this axis is
    meant to break out of from a different angle than the 등락률 axis does.

    Same IMPORTANT note as krx_fluctuation_losers_n: this is an ENTRY axis
    only. Every admitted name is still ranked by the unchanged `sign=+1`
    momentum score — the reversal hypothesis (sign=-1) was measured and
    REJECTED (screening-reversal-sweep-result-2026-09-02). A 이격도 하위
    pullback screen only survives into the final pick set if the momentum
    ranking says the pullback has already turned; this axis widens WHICH
    names get that look, it does not change how they're judged once looked
    at.

    `window_days` defaults to 20 (see `fetch_disparity_rank`'s own
    docstring for the KIS-accepted-values constraint, 5/10/20/60/120 only)
    — chosen to pair with the ranking layer's existing 14-day ATR
    normalization window without reaching into 60/120-day territory, where
    "below its mean" stops reading as a pullback and starts reading as a
    broken trend.

    `min_volume` is passed through to KIS's own floor for the same reason
    as krx_fluctuation_losers_n's: a 이격도 하위 ranking has no built-in
    liquidity floor and will readily surface 관리종목/정리매매-adjacent
    names on minimal volume — see fetch_disparity_rank's own docstring for
    the full argument.

    Same filtering/empty-list-on-failure/ticker_names-memoization
    convention as krx_top_n — see its own docstring.

    `max_price` — see krx_top_n's own docstring for the full account-
    affordability story; same `FID_INPUT_PRICE_2` mechanism, same
    `None`-is-a-no-op guarantee, `blank=""` (this TR's own no-filter
    spelling — see fetch_disparity_rank's docstring)."""
    if broker is None:
        return []
    try:
        ranked = await broker.fetch_disparity_rank(
            low=True, window_days=window_days, count=count, min_volume=min_volume, max_price=max_price
        )
    except Exception:
        logger.warning(
            "krx_disparity_low_n: fetch_disparity_rank failed — returning no candidates",
            exc_info=True,
        )
        return []
    return _krx_discovery_tickers(ranked)


async def overseas_top_n(broker, count: int, exchange: str = "NAS") -> list[str]:
    """Top `count` US tickers by cumulative trading value (거래대금) right
    now (NASDAQ by default), via
    app.brokers.kis.overseas.ranking.OverseasRankingMixin.
    fetch_overseas_trading_value_rank — a genuinely separate TR from
    fetch_overseas_volume_rank on this side, not a param switch like the
    domestic pair. RankedStock.code is already the bare US symbol (e.g.
    "AAPL"), no suffix to append (unlike krx_top_n's `.KS`). Same
    empty-list-on-failure and leveraged/inverse-filtering convention as
    krx_top_n (TQQQ/SOXL-style names), including the empty/unresolved-name
    note in krx_top_n's docstring."""
    if broker is None:
        return []
    try:
        ranked = await broker.fetch_overseas_trading_value_rank(exchange=exchange, count=count)
    except Exception:
        logger.warning(
            "overseas_top_n: fetch_overseas_trading_value_rank failed — returning no candidates", exc_info=True
        )
        return []
    return [
        r.code for r in ranked
        if r.code and not is_leveraged_or_inverse(r.name) and not is_out_of_scope_product(r.name)
    ]


async def overseas_market_cap_top_n(broker, count: int, exchange: str = "NAS") -> list[str]:
    """Top `count` US tickers on `exchange` by current market cap (시가총액),
    via `OverseasRankingMixin.fetch_overseas_market_cap_rank` — the overseas
    counterpart to krx_market_cap_top_n (see its own docstring for the
    "stable large-cap pool vs. turnover-driven pool" reasoning, added
    2026-07-30 alongside this same TR for the domestic side, but never
    actually wired into a discovery feed here until now). RankedStock.code
    is already the bare US symbol, no suffix to append (same as
    overseas_top_n). Same empty-list-on-failure and leveraged/inverse-
    filtering convention as the other discovery feeds in this module,
    including the empty/unresolved-name note in krx_top_n's docstring."""
    if broker is None:
        return []
    try:
        ranked = await broker.fetch_overseas_market_cap_rank(exchange=exchange, count=count)
    except Exception:
        logger.warning(
            "overseas_market_cap_top_n: fetch_overseas_market_cap_rank failed — returning no candidates",
            exc_info=True,
        )
        return []
    return [
        r.code for r in ranked
        if r.code and not is_leveraged_or_inverse(r.name) and not is_out_of_scope_product(r.name)
    ]
