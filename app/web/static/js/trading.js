import { LANG, LIST_CAP, MAX_HISTORY, RANK_LIST_CAP, _heldReviewBufferPct, _marketScope, _tradeAllowlistSubTab, animateNumber, cssSafe, displayTicker, escapeHtml, expanded, fmtCompact, fmtDelta, fmtKRW, fmtPrice, fmtUSD, formatTimeAgo, getRecentlyViewed, getVar, httpErrorText, isDomesticSymbol, moreButton, pushRecentlyViewed, setMarketScope, state, t } from './state.js';
import { renderMarkdown } from './assistant.js';
import { _ACTION_KO, _ACTION_PILL_CLASS, _RATING_KO, _cardReasonFor, _confidenceFrom, _executiveSummaryFrom, _isManualDecision, _lastDecisions, _ratingDiffersFromAction, _todayStr, enrichPriceFields, localizeDecisionLabels, renderMarketWidget } from './signals.js';
import { alertDialog, confirmDialog, promptDialog } from './dialog.js';

export function renderCardStats(symbol) {
  const s = state[symbol];
  if (!s) return;
  const last = s.history.length ? s.history[s.history.length - 1].price : null;
  const priceEl = document.getElementById(`price-${cssSafe(symbol)}`);
  const deltaEl = document.getElementById(`delta-${cssSafe(symbol)}`);
  if (!priceEl || !deltaEl) return;

  if (last == null) {
    priceEl.textContent = t('waiting');
    priceEl.classList.add('stale');
    return;
  }
  priceEl.classList.remove('stale');
  animateNumber(priceEl, s.cardDisplayedPrice, last, s.currency);
  s.cardDisplayedPrice = last;

  if (s.previousClose != null) {
    s.pct = (last - s.previousClose) / s.previousClose;
    const cls = s.pct > 0 ? 'up' : s.pct < 0 ? 'down' : 'flat';
    deltaEl.textContent = `${s.pct >= 0 ? '▲' : '▼'} ${fmtDelta(s.pct)}`;
    deltaEl.className = 'd ' + cls;
    deltaEl.title = t('dayChangeHint');
  } else {
    s.pct = null;
    deltaEl.textContent = '—';
    deltaEl.className = 'd flat';
  }
}

export function renderMarketStatus(symbol, isOpen) {
  const el = document.getElementById(`mkt-${cssSafe(symbol)}`);
  if (!el) return;
  el.textContent = isOpen ? t('marketOpen') : t('marketClosed');
  el.className = 'market-status ' + (isOpen ? 'open' : 'closed');
}

// Latest /api/trade-allowlist-info snapshot, kept around so the header's global
// 장중/장마감 pill (mockup-matching, scope-scoped) can be recomputed on a
// scope switch without waiting for the next poll.
let _lastTradeAllowlistInfos = [];

export function renderHeaderMarketStatus() {
  const pill = document.getElementById('header-market-status-pill');
  const relevant = _lastTradeAllowlistInfos.filter(info => (_marketScope === 'domestic') === isDomesticSymbol(info.symbol));
  if (!relevant.length) return;
  const isOpen = relevant.some(info => info.market_open);
  pill.className = 'market-status-pill ' + (isOpen ? 'open' : 'closed');
  document.getElementById('header-market-status-text').textContent = isOpen ? t('marketOpen') : t('marketClosed');
}

// Slow, unbounded recovery hook (2026-08-18) — this poller already hits the
// same endpoint buildCards() needs, every 60s, for as long as the page is
// open, so it doubles as the connection-banner's free long-term self-heal
// once ensureCardsBuilt()'s own bounded fast-retry (see its own comment)
// gives up.
export async function pollMarketStatus() {
  let infos;
  try {
    infos = await fetch('/api/trade-allowlist-info').then(r => r.json());
  } catch {
    setConnectionBanner(true);
    return;
  }
  setConnectionBanner(false);
  if (!_cardsBuilt) ensureCardsBuilt();  // fire-and-forget — don't block this poller's own cadence on it
  _lastTradeAllowlistInfos = infos;
  for (const info of infos) renderMarketStatus(info.symbol, info.market_open);
  renderHeaderMarketStatus();
  if (_tradeAllowlistSubTab === 'live') applyTradeAllowlistFilters();  // 장중 여부가 바뀌면 실시간 서브탭 목록도 갱신
}

// Real-time trading-halt/circuit-breaker overlay from the WebSocket feed
// (app.live.kis_market_status via /api/market-status) — layers on top of the
// plain open/closed status above; empty outside KRX hours, lights up
// per-ticker only when a halt/CB frame actually arrives.
export async function pollMarketOps() {
  let data = {};
  try { data = await fetch('/api/market-status').then(r => r.json()); } catch { return; }
  for (const [ticker, s] of Object.entries(data)) {
    const el = document.getElementById(`mkt-${cssSafe(ticker)}`);
    if (!el || !s.trading_halted) continue;
    const isCb = /서킷|circuit|cb/i.test(s.halt_reason || '');
    el.textContent = isCb ? t('marketStatusCb') : t('marketStatusHalt');
    el.className = 'market-status halted';
  }
}

// Connection-lost banner (2026-08-18) — buildCards()/pollMarketStatus() both
// hit /api/trade-allowlist-info; a dead backend at page-load used to reject
// buildCards() with no .catch() anywhere in main.js's boot chain, which meant
// every setInterval below it in that chain never got registered — so even
// once the backend came back, nothing ever polled it again without a manual
// reload. This banner + the retry/self-heal wiring below (ensureCardsBuilt,
// scheduleCardsRetry, and pollMarketStatus's own catch) close that gap.
let _cardsBuilt = false;
let _cardsRetryAttempt = 0;
const _CARDS_RETRY_DELAYS = [5000, 15000, 45000];

export function setConnectionBanner(show) {
  const existing = document.getElementById('connection-banner');
  if (show) {
    if (existing) return;  // idempotent — don't re-insert or rewrite text
    const bar = document.createElement('div');
    bar.id = 'connection-banner';
    bar.className = 'connection-banner';
    bar.setAttribute('role', 'status');
    bar.setAttribute('aria-live', 'polite');
    bar.textContent = t('connectionLostBanner');
    document.body.prepend(bar);
  } else {
    existing?.remove();
  }
}

function scheduleCardsRetry() {
  if (_cardsRetryAttempt >= _CARDS_RETRY_DELAYS.length) return;
  setTimeout(ensureCardsBuilt, _CARDS_RETRY_DELAYS[_cardsRetryAttempt++]);
}

// Never rejects — main.js's boot chain depends on that so the interval-
// registration block after it always runs, healthy backend or not.
export async function ensureCardsBuilt() {
  if (_cardsBuilt) return true;
  try {
    await buildCards();
  } catch {
    setConnectionBanner(true);
    scheduleCardsRetry();
    return false;
  }
  _cardsBuilt = true;
  _cardsRetryAttempt = 0;
  setConnectionBanner(false);
  return true;
}

export async function buildCards() {
  const infos = await fetch('/api/trade-allowlist-info').then(r => r.json());
  // Currently-open markets (장중) first, so the tradeable names sit at the top
  // (and inside the collapsed LIST_CAP). Stable within each group.
  infos.sort((a, b) => (b.market_open === true) - (a.market_open === true));
  const cardsEl = document.getElementById('cards');
  cardsEl.innerHTML = '';

  for (const info of infos) {
    const symbol = info.symbol;
    state[symbol] = { name: info.name, previousClose: info.previous_close, currency: info.currency, history: [], pct: null };

    const row = document.createElement('div');
    row.className = 'crow';
    row.id = `card-${cssSafe(symbol)}`;
    row.dataset.symbol = symbol;
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute('aria-label', t('openDetailAria', info.name));
    row.innerHTML = `
      <div class="nm">
        <div class="n" id="name-${cssSafe(symbol)}">${escapeHtml(info.name)}</div>
        <div class="s">${escapeHtml(symbol)} <span class="market-status closed" id="mkt-${cssSafe(symbol)}">—</span> <span class="hold-badge" id="hold-${cssSafe(symbol)}"></span></div>
      </div>
      <div class="px">
        <div class="p stale" id="price-${cssSafe(symbol)}">${t('waiting')}</div>
        <div class="d flat" id="delta-${cssSafe(symbol)}">—</div>
      </div>
    `;
    const rm = document.createElement('button');
    rm.className = 'card-remove';
    rm.textContent = '✕';
    rm.title = t('cardRemoveTitle');
    rm.addEventListener('click', async (e) => {
      e.stopPropagation();
      rm.disabled = true;
      try { await removeTradeAllowlistTicker(symbol); } finally { rm.disabled = false; }
    });
    row.appendChild(rm);

    cardsEl.appendChild(row);
    renderMarketStatus(symbol, info.market_open);
    row.addEventListener('click', () => openDetail(symbol));
    row.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDetail(symbol); }
    });
  }

  // Seed each ticker's current-price display from whatever ticks the server
  // already has (app.live.store's per-ticker buffer) — a local read, not a
  // KIS call, so all tickers are fetched concurrently instead of one-by-one
  // (one-by-one was adding up to seconds of pure round-trip latency with a
  // large trade allowlist for no reason — nothing here is rate-limited).
  await Promise.all(infos.map(async (info) => {
    const symbol = info.symbol;
    try {
      const ticks = await fetch(`/api/ticks/${encodeURIComponent(symbol)}`).then(r => r.json());
      state[symbol].history = ticks.slice(-MAX_HISTORY);
      renderCardStats(symbol);
    } catch { /* falls back to waiting for the live stream */ }
  }));

  // Rows past the cap are hidden, not un-rendered — their live tick updates
  // must survive collapse, so only display toggles.
  const extras = infos.map(i => i.symbol).slice(LIST_CAP);
  if (extras.length) {
    const btn = document.createElement('button');
    btn.className = 'more-btn';
    const applyCap = () => {
      extras.forEach(sym => {
        const row = document.getElementById(`card-${cssSafe(sym)}`);
        if (row) row.style.display = expanded.tradeAllowlist ? '' : 'none';
      });
      btn.textContent = expanded.tradeAllowlist ? t('collapse') : t('showAllTradeAllowlist', infos.length);
    };
    btn.addEventListener('click', () => { expanded.tradeAllowlist = !expanded.tradeAllowlist; applyCap(); });
    cardsEl.parentElement.appendChild(btn);
    applyCap();
  }
}

// 서버 시작 직후엔 이름/전일종가가 아직 안 채워졌을 수 있음(백엔드가 이걸
// lifespan을 막지 않는 백그라운드 작업으로 돌리도록 바꿨음 — app/web/state.py의
// _prefetch_startup_data 참조). 안 채워진 동안엔 /api/trade-allowlist-info가 이름
// 자리에 티커 코드를 그대로 반환하므로, 그 상태인 종목이 있는 동안만 재조회해서
// 채워지는 대로 반영. 카드 전체를 다시 그리지 않고 텍스트만 갱신(선택상태 유지).
export async function refreshTradeAllowlistDisplayInfo() {
  let infos;
  try { infos = await fetch('/api/trade-allowlist-info').then(r => r.json()); } catch { return true; }
  let stillLoading = false;
  for (const info of infos) {
    const symbol = info.symbol;
    if (!state[symbol]) continue;
    if (info.name === symbol) { stillLoading = true; continue; }  // backend hasn't fetched it yet
    if (state[symbol].name !== info.name) {
      state[symbol].name = info.name;
      state[symbol].previousClose = info.previous_close;
      state[symbol].currency = info.currency;
      const nameEl = document.getElementById(`name-${cssSafe(symbol)}`);
      if (nameEl) nameEl.textContent = info.name;
      renderCardStats(symbol);  // previousClose changed, so the delta% may now be computable
      if (currentDetailSymbol === symbol) {
        document.getElementById('detail-name').textContent = info.name;
        renderDetailHeader(symbol);
      }
    }
  }
  return stillLoading;
}

// buildCards() only runs on page load / manual add / manual remove — a
// ticker the backend adds and then prunes on its own (AI auto-add-on-BUY
// that never filled, see reconcile_trade_allowlist) never gets picked up
// by any of those, so its card sits forever showing stale/no data (real
// incident 2026-07-30: 459580.KS stuck as "대기 중..." long after the
// backend had already dropped it). This only prunes cards for tickers the
// backend no longer lists — additions still wait for the next full
// buildCards() (page load or a manual add), which is an acceptable gap
// since a brand-new AI-added ticker showing up late is far less confusing
// than a dead one lingering forever.
export async function pruneStaleTradeAllowlistCards() {
  let infos;
  try { infos = await fetch('/api/trade-allowlist-info').then(r => r.json()); } catch { return; }
  const fresh = new Set(infos.map(i => i.symbol));
  for (const symbol of Object.keys(state)) {
    if (fresh.has(symbol)) continue;
    delete state[symbol];
    document.getElementById(`card-${cssSafe(symbol)}`)?.remove();
  }
}

// ✕ 버튼은 "목록에서 숨기기"가 아니라 그 종목의 손절 감시를 끊고 매도까지
// 막는 동작이다(서버 쪽 /api/trade-allowlist/remove docstring 참조). 위험
// 여부 판단은 서버가 한다 — 보유 없는 종목이면 예전처럼 확인창 없이 바로
// 지워지고, 보유 중이거나 보유 여부를 확인할 수 없을 때만 409가 돌아와
// danger 확인창을 띄운 뒤 force=true로 재요청한다.
export async function removeTradeAllowlistTicker(symbol) {
  const name = state[symbol]?.name || symbol;
  try {
    let r = await fetch('/api/trade-allowlist/remove', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker: symbol, force: false }),
    });

    if (r.status === 409) {
      const e = await r.json().catch(() => ({}));
      const d = (e && e.detail) || {};
      const body = d.error_code === 'holdings_unknown'
        ? t('removeHoldingsUnknownConfirm', name)
        : t('removeHeldConfirm', name, d.shares || 0);
      const proceed = await confirmDialog({ title: t('removeHeldTitle'), body, danger: true });
      if (!proceed) return;
      r = await fetch('/api/trade-allowlist/remove', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: symbol, force: true }),
      });
    }

    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      const detail = e.detail && (e.detail.detail || e.detail);
      await alertDialog({ title: t('cardRemoveTitle'), body: t('removeFailed', detail || httpErrorText(r.status)) });
      return;
    }

    delete state[symbol];
    await buildCards();
  } catch (err) {
    console.error('trade-allowlist remove request failed', err);
    await alertDialog({ title: t('cardRemoveTitle'), body: t('removeRequestFailed') });
  }
}

function _addErrText(code, subject) {
  switch (code) {
    case 'leveraged_or_inverse_not_allowed': return t('tickerAddErrLeveraged', subject);
    case 'unsupported_product_type': return t('tickerAddErrUnsupportedType', subject);
    case 'cash_or_bond_fund': return t('tickerAddErrCashOrBond', subject);
    case 'derivative_income_fund': return t('tickerAddErrDerivativeIncome', subject);
    case 'unresolved_ticker_name': return t('tickerAddErrUnresolvedName', subject);
    case 'invalid_ticker_format': return t('tickerAddErrInvalidFormat', subject);
    default: return t('tickerAddErrNotFound', subject);
  }
}

// /api/trade-allowlist/add. 성공하면 true, 실패하면 서버의 error_code 문자열
// (없으면 false)을 반환한다. silent=false(기본)면 실패 시 사유 토스트를 띄운다 —
// 예전엔 catch{}로 전부 삼켜서 레버리지 거부든 뭐든 "종목을 찾을 수 없습니다"
// 하나로만 보였다. 순위표 +관심 버튼도 이 경로라 덤으로 사유가 표시된다.
export async function addTickerDirect(ticker, { silent = false } = {}) {
  let code = '';
  try {
    const r = await fetch('/api/trade-allowlist/add', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticker }),
    });
    if (r.ok) { await buildCards(); loadRank(); return true; }
    try { const e = await r.json(); code = (e && e.detail && e.detail.error_code) || ''; } catch { /* generic below */ }
  } catch { /* network — generic below */ }
  if (!silent) showToast(t('tickerSearchAddFailTitle'), _addErrText(code, ticker), 'rejected');
  return code || false;
}

// 타입어헤드 드롭다운에서 종목을 골랐을 때. 이미 관심종목이면 상세만 열고,
// 아니면 추가 → (필요 시)국내/해외 스코프 자동전환 → 상세 열기 → 성공 토스트.
// 스코프 전환을 빠뜨리면 해외 종목을 추가해도 카드가 안 보여 "추가 실패"로 오인된다.
export async function addTickerFromSearch(row) {
  const sym = row.symbol;
  if (state[sym]) { openDetail(sym); return; }
  const res = await addTickerDirect(sym, { silent: true });
  if (res === true) {
    const targetDomestic = isDomesticSymbol(sym);
    if (targetDomestic !== (_marketScope === 'domestic')) {
      applyTradingScope(targetDomestic ? 'domestic' : 'overseas');
    }
    if (state[sym]) openDetail(sym);
    showToast(t('tickerSearchAddedTitle'), t('tickerSearchAddedSub', row.name, displayTicker(sym)), '');
  } else {
    showToast(t('tickerSearchAddFailTitle'), _addErrText(typeof res === 'string' ? res : '', row.name), 'rejected');
  }
}

// 관심종목 패널 입력창의 Enter/버튼 폴백(main.js) — 드롭다운에 선택 가능한
// 결과가 없을 때만 여기 온다. 이미 있으면 상세로, 없으면 원문 그대로 add 시도.
// 반환값: 열린 심볼 또는 못 찾으면 null. addTickerDirect는 silent — 못 찾음
// 토스트는 main.js가 띄운다(중복 방지).
export async function handleGlobalSearch(query) {
  const q = query.trim();
  if (!q) return null;
  const upper = q.toUpperCase();
  const existing = Object.keys(state).find(s => s.toUpperCase() === upper);
  if (existing) { openDetail(existing); return existing; }
  await addTickerDirect(q, { silent: true });
  const added = Object.keys(state).find(s => s.toUpperCase() === upper);
  if (added) { openDetail(added); return added; }
  return null;
}

/* ---------------- detail modal: bigger chart, real axes, rich tooltip ---------------- */

export let currentDetailSymbol = null;

export function openDetail(symbol) {
  currentDetailSymbol = symbol;
  document.getElementById('detail-modal-empty').hidden = true;
  document.getElementById('detail-modal').hidden = false;
  pushRecentlyViewed(symbol);
  if (_tradeAllowlistSubTab === 'recent') applyTradeAllowlistFilters();  // keep the MRU order in sync
  document.querySelectorAll('#cards .crow.selected').forEach(row => row.classList.remove('selected'));
  document.getElementById(`card-${cssSafe(symbol)}`)?.classList.add('selected');
  const s = state[symbol];
  document.getElementById('detail-name').textContent = s.name;
  document.getElementById('detail-symbol').textContent = displayTicker(symbol);
  renderDetailHeader(symbol);
  renderCompactAiDiagnosis(symbol);
  renderFundamentalsSummary(symbol);
  setupDetailExtras(symbol);

  // Keep the order ticket pointed at whatever's selected — dropdown stays as
  // a manual override (not removed), just synced on selection.
  const orderSel = document.getElementById('order-ticker');
  if (orderSel && orderSel.querySelector(`option[value="${symbol}"]`)) {
    orderSel.value = symbol;
    refreshAvail();
    updateOrderSubmitButton();
  }
}

// 워치리스트 필터링 — 국내/해외 scope(헤더) AND 관심종목/실시간/최근 본 서브탭(목업
// 매칭)을 함께 적용. "실시간"은 이미 렌더된 market-status 배지(장중 여부)를 그대로
// 재사용, "최근 본"은 getRecentlyViewed()의 MRU 순서대로 화면에도 재배치한다.
export function applyTradeAllowlistFilters() {
  const cardsEl = document.getElementById('cards');
  const recent = getRecentlyViewed();
  const rows = [...cardsEl.querySelectorAll('.crow')];
  rows.forEach(row => {
    const sym = row.dataset.symbol;
    const inScope = (_marketScope === 'domestic') === isDomesticSymbol(sym);
    let matchesSubTab = true;
    if (_tradeAllowlistSubTab === 'live') {
      matchesSubTab = document.getElementById(`mkt-${cssSafe(sym)}`)?.classList.contains('open') ?? false;
    } else if (_tradeAllowlistSubTab === 'recent') {
      matchesSubTab = recent.includes(sym);
    }
    row.style.display = (inScope && matchesSubTab) ? '' : 'none';
  });
  if (_tradeAllowlistSubTab === 'recent') {
    recent.forEach(sym => {
      const row = document.getElementById(`card-${cssSafe(sym)}`);
      if (row && row.style.display !== 'none') cardsEl.appendChild(row);
    });
  }
  const firstRow = rows.find(r => r.style.display !== 'none');
  document.getElementById('trade-allowlist-subtab-empty').hidden = !(_tradeAllowlistSubTab === 'recent' && !firstRow);
  return firstRow ? firstRow.dataset.symbol : null;
}

// 국내/해외 scope (global, top header — see state.js's _marketScope) applied
// to the trade allowlist column: filters which rows show, and re-selects into
// whichever scope is active if the current selection falls outside it.
// The order ticket's own 종목 dropdown follows it too (syncOrderTickerOptions):
// server-side order routing was never the issue — a KRW-context screen offering
// USD tickers is user-facing scope confusion, and an 오주문 risk.
export function applyTradingScope(scope) {
  if (scope) setMarketScope(scope);
  // MUST stay above openDetail() below: openDetail sets #order-ticker's value,
  // and rebuilding the options after that would wipe that selection.
  // 순서 바꾸지 말 것 — openDetail이 고른 종목이 지워짐.
  syncOrderTickerOptions();
  renderHeaderMarketStatus();
  const firstVisible = applyTradeAllowlistFilters();
  const curRow = currentDetailSymbol && document.getElementById(`card-${cssSafe(currentDetailSymbol)}`);
  const curVisible = curRow && curRow.style.display !== 'none';
  if (!curVisible && firstVisible) openDetail(firstVisible);
  if (_rankMode !== 'sector') loadRank();  // re-fetch the active rank mode for the new scope (sector is KRX-only, unaffected)
  // 기타탭의 포트폴리오 상세/최근 거래도 같은 전역 국내/해외 스코프를 따름(2026-07-30) —
  // 이미 폴링으로 채워둔 캐시를 다시 그리기만 하면 되므로 재조회 없이 즉시 반영.
  renderPortfolioTable();
  renderTrades();
  // 미체결 주문도 같은 스코프를 따름 — 첫 폴링 전에는 캐시가 비어 있으므로 가드.
  if (_openOrdersCache.length) renderOpenOrders(_openOrdersCache);
}

export function setupDetailExtras(symbol) {
  // reset the analysis area for the newly opened ticker
  const analysis = document.getElementById('detail-analysis');
  analysis.hidden = true;
  analysis.innerHTML = '';
  document.getElementById('detail-analyze-status').textContent = '';
  document.getElementById('detail-analyze').disabled = false;
  const pipelineReset = document.getElementById('analyze-pipeline');
  pipelineReset.hidden = true;
  pipelineReset.querySelectorAll('.ap-step').forEach(el => el.classList.remove('done', 'active'));
  document.getElementById('analyze-pipeline-caption').textContent = '';

  // Period candlestick chart works for every ticker (KIS for .KS, yfinance for US).
  document.getElementById('detail-period-wrap').hidden = false;
  loadPeriodChart(symbol, _periodMode);

  // These are KRX-only (KIS real-time feed / Korean data providers).
  const krWrap = document.getElementById('detail-kr-insights');
  const obWrap = document.getElementById('detail-orderbook-wrap');
  // These are the yfinance-backed news/earnings-trend sections — shown only
  // for non-KRX tickers, since a .KS ticker gets the same information (and
  // more accurate) from the KR-insights "news"/"financials" tabs instead.
  // Not just hidden but never fetched for .KS, avoiding a wasted call to a
  // data source with documented weak Korean-ticker coverage.
  const newsWrap = document.getElementById('detail-news-wrap');
  const earningsWrap = document.getElementById('detail-earnings-wrap');
  if (symbol.endsWith('.KS')) {
    krWrap.hidden = false;
    document.querySelectorAll('#kr-tabs .kr-tab').forEach((tab, i) => tab.classList.toggle('active', i === 0));
    loadKrInsight(symbol, 'investor-flow');
    obWrap.hidden = false;
    loadOrderbook(symbol);
    newsWrap.hidden = true;
    earningsWrap.hidden = true;
  } else {
    krWrap.hidden = true;
    obWrap.hidden = true;
    newsWrap.hidden = false;
    renderNews(symbol);
    earningsWrap.hidden = false;
    renderEarningsTrend(symbol);
  }
}

export function renderDetailHeader(symbol) {
  const s = state[symbol];
  const last = s.history.length ? s.history[s.history.length - 1].price : null;
  const priceEl = document.getElementById('detail-price');
  animateNumber(priceEl, s.modalDisplayedPrice, last, s.currency);
  s.modalDisplayedPrice = last;
  const deltaEl = document.getElementById('detail-delta');
  if (s.pct != null) {
    deltaEl.textContent = `${s.pct >= 0 ? '▲' : '▼'} ${fmtDelta(s.pct)}`;
    deltaEl.className = 'delta ' + (s.pct > 0 ? 'up' : s.pct < 0 ? 'down' : 'flat');
  } else {
    deltaEl.textContent = '—';
    deltaEl.className = 'delta flat';
  }
}

// 목업의 상시노출 "AI 진단" 압축 라인 — 새 분석을 돌리지 않고, 이미 내려온 과거
// 결정(/api/decisions, signals.js가 폴링) 중 이 종목의 가장 최근 것만 재사용.
// AI 시그널 카드와 완전히 같은 파싱 로직(신뢰도/한줄요약)을 그대로 씀. 최근 결정이
// 없으면(한 번도 분석 안 된 종목) 카드 자체를 숨김 — 지어내지 않음.
export function renderCompactAiDiagnosis(symbol) {
  const card = document.getElementById('ai-diag-compact');
  if (!symbol) { card.hidden = true; return; }
  const decision = _lastDecisions.find(d => d.ticker === symbol);
  if (!decision) { card.hidden = true; return; }
  const ratingLabel = LANG === 'ko' ? (_RATING_KO[decision.rating] || decision.rating) : decision.rating;
  const confidence = _confidenceFrom(decision.decision_text);
  // Was _executiveSummaryFrom(decision.decision_text) — that regex-parses the
  // model's raw English/Korean prose, which is why this box showed English
  // text with the UI set to Korean. The AI 시그널 card has always preferred the
  // purpose-built card_reason_ko/card_reason_en fields; both now share
  // _cardReasonFor so they cannot drift apart again.
  const reason = _cardReasonFor(decision);
  card.hidden = false;
  const pill = document.getElementById('ai-diag-compact-pill');
  // Same action_label-first, rating-as-hint approach as the AI 시그널 카드
  // (signals.js's _buildResolvedSignalCard) — see _ACTION_KO's docstring
  // for why the two can disagree. The hint only shows when it actually
  // differs from the action (_ratingDiffersFromAction).
  const hasAction = !!decision.action_label;
  const pillClass = hasAction ? (_ACTION_PILL_CLASS[decision.action_label] || 'hold') : decision.rating.toLowerCase();
  const pillLabel = hasAction
    ? (LANG === 'ko' ? (_ACTION_KO[decision.action_label] || decision.action_label) : decision.action_label)
    : ratingLabel;
  pill.className = 'ai-diag-compact-pill ' + pillClass;
  pill.textContent = confidence ? `${pillLabel} · ${confidence}%` : pillLabel;
  const showRatingHint = _ratingDiffersFromAction(decision.rating, decision.action_label);
  const hintEl = document.getElementById('ai-diag-compact-rating-hint');
  hintEl.hidden = !showRatingHint;
  hintEl.textContent = showRatingHint ? `${t('signalAiRatingHint')}: ${ratingLabel}` : '';
  const manualEl = document.getElementById('ai-diag-compact-manual');
  if (manualEl) {
    manualEl.hidden = !_isManualDecision(decision);
    manualEl.title = t('signalManualBadgeTitle');
  }
  document.getElementById('ai-diag-compact-reason').textContent = reason;
}

// 시가총액 — 국내는 억원 단위(fetch_valuation), 해외는 원시 통화 단위
// (fetch_overseas_valuation) 그대로 내려오므로 스코프별로 다르게 축약 표기.
function _fmtMarketCap(value, domestic) {
  if (value == null) return '—';
  if (domestic) return (value / 10000).toFixed(1) + t('trillionWonSuffix');
  const abs = Math.abs(value);
  if (abs >= 1e12) return '$' + (value / 1e12).toFixed(2) + 'T';
  if (abs >= 1e9) return '$' + (value / 1e9).toFixed(2) + 'B';
  if (abs >= 1e6) return '$' + (value / 1e6).toFixed(2) + 'M';
  return '$' + value.toLocaleString('en-US');
}

// 목업의 펀더멘털 요약행 — 당일 고가/저가/거래량(이미 폴링 중인 /api/candles 마지막
// 캔들 재사용) + 시가총액/PER/PBR(+해외는 52주 고저, 신규 /api/valuation). 배당수익률·
// 외국인지분은 KIS API 자체가 없어 뺌(renderCompactAiDiagnosis와 같은 원칙 — 지어내지 않음).
export async function renderFundamentalsSummary(symbol) {
  const el = document.getElementById('fund-summary');
  const requestedSymbol = symbol;
  el.innerHTML = '';
  const domestic = isDomesticSymbol(symbol);
  const currency = domestic ? 'KRW' : 'USD';

  const [candles, valuation] = await Promise.all([
    fetch(`/api/candles/${encodeURIComponent(symbol)}?period=D`).then(r => r.json()).catch(() => []),
    fetch(`/api/valuation/${encodeURIComponent(symbol)}`).then(r => r.json()).catch(() => ({})),
  ]);
  if (currentDetailSymbol !== requestedSymbol) return;  // selection moved on while this was in flight

  const last = candles.length ? candles[candles.length - 1] : null;
  const cells = [
    { label: t('fundHigh'), value: last ? fmtPrice(last.high, currency) : '—' },
    { label: t('fundLow'), value: last ? fmtPrice(last.low, currency) : '—' },
    { label: t('fundVolume'), value: last ? fmtCompact(last.volume) : '—' },
    { label: t('fundMarketCap'), value: _fmtMarketCap(valuation.market_cap, domestic) },
    { label: t('fundPer'), value: valuation.per ? valuation.per.toFixed(1) + t('multipleSuffix') : '—' },
    { label: t('fundPbr'), value: valuation.pbr ? valuation.pbr.toFixed(2) + t('multipleSuffix') : '—' },
  ];
  if (!domestic) {
    cells.push({ label: t('fund52wHigh'), value: fmtPrice(valuation.week52_high, currency) });
    cells.push({ label: t('fund52wLow'), value: fmtPrice(valuation.week52_low, currency) });
  }
  el.innerHTML = cells.map(c => `
    <div class="fund-summary-cell">
      <div class="fund-summary-label">${escapeHtml(c.label)}</div>
      <div class="fund-summary-value">${escapeHtml(c.value)}</div>
    </div>
  `).join('');
}

/* ---------------- detail modal: recent news ---------------- */

/* ---------------- unified period chart (분/일/주/월/년 + MA10) ----------------
   원래 그래프① tick 실시간 라인차트 + 그래프③ 독립 "당일 분봉" 섹션이 따로
   있었으나, 하나의 캔들스틱 컴포넌트(탭 전환)로 통합. "분" 탭만 /api/minute를
   쓰고, 나머지는 /api/candles?period=. */

export let _periodMode = 'D';
let _periodRows = [];
let _periodRefreshTimer = null;

// Layout constants — CSS px (the price chart uses dpr + setTransform, unlike
// the EPS/revenue charts' fixed clientWidth*2 scale, so these numbers are NOT
// interchangeable with that code).
const PERIOD_PAD_X = 6;
const PERIOD_PAD_TOP = 8;
const PERIOD_PAD_BOT = 4;
const PERIOD_VOL_GAP = 8;
const PERIOD_VOL_RATIO = 0.22;

// Hover state — module scope so pointermove/leave and the 30s MIN auto-refresh
// can all reach it without threading it through every render call.
let _periodHoverIdx = null;
let _periodHoverKey = null;   // the hovered candle's day/time — re-anchor target after a refetch
let _periodPointerInside = false;
let _periodLastLayout = null; // last render's coordinate system, for hit-testing in pointermove

export async function loadPeriodChart(symbol, period) {
  if (period) _periodMode = period;
  document.querySelectorAll('#period-modes .rank-mode').forEach(b => b.classList.toggle('active', b.dataset.period === _periodMode));
  clearInterval(_periodRefreshTimer);
  _periodRefreshTimer = null;
  _periodHoverIdx = null;
  _periodHoverKey = null;

  await _fetchAndDrawPeriodChart(symbol);

  // "분" 탭은 장중 값이 계속 바뀌므로 30초마다 재조회 — 종목/탭이 바뀌면 다음
  // loadPeriodChart 호출이 위에서 이 타이머부터 지운다.
  if (_periodMode === 'MIN') {
    _periodRefreshTimer = setInterval(() => {
      if (currentDetailSymbol === symbol && _periodMode === 'MIN') _fetchAndDrawPeriodChart(symbol);
    }, 30000);
  }
}

async function _fetchAndDrawPeriodChart(symbol) {
  const labels = document.getElementById('period-labels');
  const endpoint = _periodMode === 'MIN'
    ? `/api/minute/${encodeURIComponent(symbol)}`
    : `/api/candles/${encodeURIComponent(symbol)}?period=${_periodMode}`;
  let rows = [];
  try { rows = await fetch(endpoint).then(r => r.json()); } catch { /* keep last */ }
  if (currentDetailSymbol !== symbol) return;  // selection moved on while this was in flight
  _periodRows = rows;
  // Re-anchor the hovered candle to its key (not its old index) so a MIN-tab
  // auto-refresh that appends a new bar doesn't leave the crosshair/label
  // pointing at a shifted, different candle. Lost entirely if the pointer
  // left the canvas or the key no longer exists in the new rows.
  if (_periodPointerInside && _periodHoverKey != null) {
    const idx = rows.findIndex(r => (r.time ?? r.day) === _periodHoverKey);
    _periodHoverIdx = idx >= 0 ? idx : null;
    if (idx < 0) _periodHoverKey = null;
  } else {
    _periodHoverIdx = null;
    _periodHoverKey = null;
  }
  if (!rows.length) {
    // Y(연봉)·MIN(분봉) 둘 다 해외 종목엔 KIS API 자체가 없음(확인됨, 지어내지
    // 않고 그대로 안내) — market_data.py의 candles_endpoint/minute_candles_endpoint 참조.
    const overseas = !symbol.endsWith('.KS');
    let msg = t('candleChartEmpty');
    if (_periodMode === 'Y' && overseas) msg = t('candleChartNoKisApi');
    else if (_periodMode === 'MIN' && overseas) msg = t('candleChartNoKisApiMinute');
    else if (_periodMode === 'MIN') msg = t('minuteChartEmptyOff');  // 국내인데 비면 장외/미수신
    _renderPeriodChart();
    labels.innerHTML = `<span class="mc-empty">${msg}</span>`;
    return;
  }
  _renderPeriodChart();
}

// resize 등으로 재조회 없이 다시 그리기만 할 때. 축/라벨도 같이 갱신해야
// 창 폭이 바뀐 뒤에도 눈금이 새 좌표계와 어긋나지 않는다.
export function redrawPeriodChart() {
  const canvas = document.getElementById('period-canvas');
  if (canvas) _renderPeriodChart();
}

// 캔버스·Y축·OHLC 라벨을 항상 함께 갱신하는 유일한 경로 — fetch 후, resize
// 후, 호버 중에도 전부 이 함수 하나만 부른다(각자 따로 갱신하면 축이 캔버스
// 리사이즈를 놓치는 식의 불일치가 생김).
function _renderPeriodChart() {
  const canvas = document.getElementById('period-canvas');
  const axisEl = document.getElementById('period-axis-y');
  if (!_periodRows.length) {
    drawCandles(canvas, []);
    axisEl.innerHTML = '';
    axisEl.hidden = true;
    _periodLastLayout = null;
    return;
  }
  const res = drawCandles(canvas, _periodRows, { ma: 10, hoverIdx: _periodHoverIdx });
  _periodLastLayout = res.layout;
  axisEl.hidden = false;
  _renderPeriodAxisY(axisEl, res);
  _renderPeriodLabels(_periodHoverIdx);
}

function _renderPeriodAxisY(axisEl, res) {
  const cur = state[currentDetailSymbol]?.currency || 'KRW';
  renderChartAxisY('period-axis-y', res.lo, res.hi, v => fmtPrice(v, cur), 4);
  // Align the axis's first/last label to the price plot's own top/bottom —
  // .axis-y is a space-between flex column with 6px vertical padding that
  // would otherwise throw it out of sync with the candle area.
  const labelH = parseFloat(getComputedStyle(axisEl).lineHeight) || 14;
  const { priceTop, priceBottom } = res.layout;
  axisEl.style.marginTop = (priceTop - labelH / 2) + 'px';
  axisEl.style.height = (priceBottom - priceTop + labelH) + 'px';
}

function _fmtVolumeLabel(v) {
  return v == null ? '—' : Math.round(v).toLocaleString('en-US');
}

// KIS gives minute rows a HHMMSS `time` and day rows an 8-digit `day` — this
// is display-only formatting, not a parse of a guaranteed-present field (the
// broker has been observed to send an empty string for day; see quote.py's
// own `.get(..., "")` fallback), so a missing/malformed value degrades to ''
// rather than throwing.
function _periodBarLabel(row) {
  if (row.time) {
    const digits = String(row.time).replace(/\D/g, '');
    return digits.length >= 4 ? `${digits.slice(0, 2)}:${digits.slice(2, 4)}` : '';
  }
  if (row.day) {
    const digits = String(row.day).replace(/\D/g, '');
    return digits.length === 8 ? `${digits.slice(0, 4)}-${digits.slice(4, 6)}-${digits.slice(6, 8)}` : digits;
  }
  return '';
}

function _renderPeriodLabels(idx) {
  const labels = document.getElementById('period-labels');
  const row = _periodRows[idx ?? _periodRows.length - 1];
  if (!row) return;
  const cur = state[currentDetailSymbol]?.currency || 'KRW';
  const countLabel = _periodMode === 'MIN' ? t('minuteCountLabel', _periodRows.length) : t('candleCountLabel', _periodRows.length);
  const barLabel = _periodBarLabel(row);
  labels.innerHTML = (barLabel ? `<span class="mc-period">${escapeHtml(barLabel)}</span>` : '')
    + `<span>${countLabel}</span>`
    + `<span>${t('ohlcOpen')} <b>${fmtPrice(row.open, cur)}</b></span>`
    + `<span>${t('ohlcHigh')} <b>${fmtPrice(row.high, cur)}</b></span>`
    + `<span>${t('ohlcLow')} <b>${fmtPrice(row.low, cur)}</b></span>`
    + `<span>${t('ohlcClose')} <b>${fmtPrice(row.close, cur)}</b></span>`
    + `<span>${t('ohlcVolume')} <b>${_fmtVolumeLabel(row.volume)}</b></span>`;
}

// Plot geometry — the single source of truth for where the price area ends
// and the volume area begins. Every coordinate calculation below (drawing,
// axis alignment, hit-testing) derives from this, so they can't drift apart.
function _periodLayout(w, h) {
  const plotH = h - PERIOD_PAD_TOP - PERIOD_PAD_BOT;
  const volH = Math.round(plotH * PERIOD_VOL_RATIO);
  const volBottom = h - PERIOD_PAD_BOT;
  const volTop = volBottom - volH;
  const priceTop = PERIOD_PAD_TOP;
  const priceBottom = volTop - PERIOD_VOL_GAP;
  return { padX: PERIOD_PAD_X, plotW: w - PERIOD_PAD_X * 2, priceTop, priceBottom, volTop, volBottom };
}

function _candleScale(rows, ma) {
  const maValues = ma ? _movingAverage(rows.map(r => r.close), ma) : null;
  const allLows = rows.map(r => r.low), allHighs = rows.map(r => r.high);
  if (maValues) maValues.forEach(v => { if (v != null) { allLows.push(v); allHighs.push(v); } });
  const hi = Math.max(...allHighs), lo = Math.min(...allLows);
  const span = (hi - lo) || 1;
  return { lo, hi, span, maValues };
}

export function drawCandles(canvas, rows, { ma, hoverIdx = null, volume = true } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 260;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!rows.length) return null;

  const n = rows.length;
  const layout = _periodLayout(w, h);
  const { padX, plotW, priceTop, priceBottom, volTop, volBottom } = layout;
  const { lo, hi, span, maValues } = _candleScale(rows, ma);
  const y = v => priceTop + (1 - (v - lo) / span) * (priceBottom - priceTop);
  const cw = plotW / n;              // per-candle slot width
  const cx = i => padX + cw * (i + 0.5);
  const bw = Math.max(1, Math.min(cw * 0.7, 14));  // body width
  const up = getVar('--up') || '#e34948', down = getVar('--down') || '#2a78d6';
  const volMax = Math.max(...rows.map(r => r.volume || 0)) || 1;
  const vy = v => volBottom - ((v || 0) / volMax) * (volBottom - volTop);

  // Grid — price area gets 5 lines (aligned 1:1 with the 5 Y-axis labels,
  // including the top/bottom plot boundary), volume area gets a single
  // baseline. +0.5 keeps 1px lines crisp on the dpr-scaled coordinate system.
  ctx.strokeStyle = getVar('--grid');
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gy = Math.round(priceTop + (priceBottom - priceTop) * (i / 4)) + 0.5;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
  }
  {
    const gy = Math.round(volBottom) + 0.5;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
  }

  // Hover indicator — drawn before the candles so the candle body/wick stays
  // on top of it, matching drawEpsChart's own layering.
  if (hoverIdx != null && rows[hoverIdx]) {
    ctx.fillStyle = getVar('--surface-2');
    ctx.globalAlpha = 0.55;
    ctx.fillRect(padX + cw * hoverIdx, priceTop, cw, volBottom - priceTop);
    ctx.globalAlpha = 1;

    ctx.strokeStyle = getVar('--ink-2');
    ctx.globalAlpha = 0.35;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(cx(hoverIdx), priceTop); ctx.lineTo(cx(hoverIdx), volBottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // Volume bars.
  if (volume) {
    ctx.globalAlpha = 0.45;
    rows.forEach((r, i) => {
      const rising = r.close >= r.open;
      ctx.fillStyle = rising ? up : down;
      const top = vy(r.volume);
      ctx.fillRect(cx(i) - bw / 2, top, bw, Math.max(1, volBottom - top));
    });
    ctx.globalAlpha = 1;
  }

  // Candles.
  rows.forEach((r, i) => {
    const rising = r.close >= r.open;
    ctx.strokeStyle = ctx.fillStyle = rising ? up : down;
    ctx.beginPath(); ctx.moveTo(cx(i), y(r.high)); ctx.lineTo(cx(i), y(r.low)); ctx.stroke();  // wick
    const yo = y(r.open), yc = y(r.close);
    const top = Math.min(yo, yc), bh = Math.max(1, Math.abs(yc - yo));
    ctx.fillRect(cx(i) - bw / 2, top, bw, bh);  // body
  });

  // MA line.
  if (maValues) {
    ctx.beginPath();
    let started = false;
    maValues.forEach((v, i) => {
      if (v == null) return;
      if (!started) { ctx.moveTo(cx(i), y(v)); started = true; } else { ctx.lineTo(cx(i), y(v)); }
    });
    ctx.strokeStyle = getVar('--accent');
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  return { lo, hi, layout };
}

// 단순 이동평균 — 앞의 (period-1)개는 값이 없어 null(선이 그 지점부터 시작됨).
function _movingAverage(values, period) {
  return values.map((_, i) => {
    if (i < period - 1) return null;
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += values[j];
    return sum / period;
  });
}

// Hovering the price chart — separate from wireEarningsHover() rather than
// folded into it: that one only wires canvases that exist for overseas
// tickers (earnings-trends section), while the price chart is common to
// every ticker, so the lifetime/conditions differ.
export function wirePeriodChartHover() {
  const canvas = document.getElementById('period-canvas');
  const rerender = () => _renderPeriodChart();
  canvas.addEventListener('pointerenter', () => { _periodPointerInside = true; });
  canvas.addEventListener('pointermove', (e) => {
    if (!_periodRows.length || !_periodLastLayout) return;
    _periodPointerInside = true;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const cw = _periodLastLayout.plotW / _periodRows.length;
    // floor, not round — candles occupy slots (not interpolated points like
    // the EPS chart), so the cursor's slot must match what's under it.
    let idx = Math.floor((px - _periodLastLayout.padX) / cw);
    idx = Math.max(0, Math.min(_periodRows.length - 1, idx));
    if (idx === _periodHoverIdx) return;  // same slot — skip the redraw (pointermove fires very often)
    _periodHoverIdx = idx;
    _periodHoverKey = _periodRows[idx].time ?? _periodRows[idx].day ?? null;
    rerender();
  });
  const clear = () => {
    _periodPointerInside = false;
    _periodHoverIdx = null;
    _periodHoverKey = null;
    rerender();
  };
  canvas.addEventListener('pointerleave', clear);
  canvas.addEventListener('pointercancel', clear);
}

/* ---------------- live orderbook (KRX only) ---------------- */

export async function loadOrderbook(symbol) {
  const el = document.getElementById('detail-orderbook');
  const status = document.getElementById('ob-status');
  let ob = {};
  try { ob = await fetch(`/api/orderbook/${encodeURIComponent(symbol)}`).then(r => r.json()); } catch { return; }
  if (!ob || (!ob.asks?.length && !ob.bids?.length)) {
    el.innerHTML = `<div class="ob-empty">${t('orderbookEmpty')}</div>`;
    return;
  }
  status.textContent = ob.last != null ? t('orderbookHintLive', fmtPrice(ob.last, 'KRW'), fmtDelta(ob.change_pct)) : '';
  const cur = state[symbol]?.currency || 'KRW';
  const rows = [];
  // asks high->low (top), then bids high->low (bottom)
  [...(ob.asks || [])].reverse().forEach(a => {
    rows.push(`<div class="ob-cell ob-askqty">${a.qty.toLocaleString('en-US')}</div>`
      + `<div class="ob-cell ob-price ask">${fmtPrice(a.price, cur)}</div><div class="ob-cell"></div>`);
  });
  (ob.bids || []).forEach(b => {
    rows.push(`<div class="ob-cell"></div><div class="ob-cell ob-price bid">${fmtPrice(b.price, cur)}</div>`
      + `<div class="ob-cell ob-bidqty">${b.qty.toLocaleString('en-US')}</div>`);
  });
  el.innerHTML = rows.join('');
}

/* ---------------- KR-insight panel (수급/재무/투자의견/리스크/뉴스) ----------------
   Table content (column headers, captions, notes) is Korean-only by design —
   only the tab labels/section title above go through the i18n system; see the
   krContentDisclaimer key rendered alongside every table/section here. */

export function krTableHTML(table) {
  if (!table.rows || !table.rows.length) return `<div class="kr-empty">${t('krDataEmpty')}</div>`;
  const signed = new Set(table.signed || []);
  const head = table.columns.map(c => `<th>${escapeHtml(c)}</th>`).join('');
  const body = table.rows.map(row => '<tr>' + row.map((cell, i) => {
    let cls = '';
    if (signed.has(i)) { const neg = String(cell).trim().startsWith('-') || String(cell).trim().startsWith('−'); cls = neg ? ' class="neg"' : ' class="pos"'; }
    return `<td${cls}>${escapeHtml(String(cell))}</td>`;
  }).join('') + '</tr>').join('');
  return `<table class="kr-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

export function krRender(r) {
  if (r.unsupported) return `<div class="kr-empty">${t('krUnsupported')}</div>`;
  if (r.error) return `<div class="kr-empty">${escapeHtml(r.error)}</div>`;
  const disclaimer = `<p class="kr-cap">${t('krContentDisclaimer')}</p>`;
  if (r.type === 'news') {
    if (!r.items || !r.items.length) return disclaimer + `<div class="kr-empty">${t('krNewsEmpty')}</div>`;
    return disclaimer + '<div class="kr-news">' + r.items.map(n =>
      `<div class="kr-news-row"><span class="kr-news-meta">${escapeHtml(n.time || '')} ${escapeHtml(n.source || '')}</span>`
      + `<span>${escapeHtml(n.title || '')}</span></div>`).join('') + '</div>';
  }
  if (r.type === 'sections') {
    if (!r.sections || !r.sections.length) return disclaimer + `<div class="kr-empty">${t('krDataEmpty')}</div>`;
    return disclaimer + r.sections.map(s => `<div class="kr-sec-title">${escapeHtml(s.title)}</div>${krTableHTML(s)}`).join('');
  }
  // plain table
  let html = disclaimer;
  if (r.caption) html += `<p class="kr-cap">${escapeHtml(r.caption)}</p>`;
  html += krTableHTML(r);
  if (r.note) html += `<p class="kr-note">${escapeHtml(r.note)}</p>`;
  return html;
}

export async function loadKrInsight(symbol, kind) {
  const panel = document.getElementById('kr-panel');
  panel.textContent = t('loading');
  try {
    const r = await fetch(`/api/kr/${kind}/${encodeURIComponent(symbol)}`).then(x => x.json());
    panel.innerHTML = krRender(r);
  } catch {
    panel.innerHTML = `<div class="kr-empty">${t('krLoadError')}</div>`;
  }
}

/* ---------------- one-click AI analysis ---------------- */

// Canonical 7-stage pipeline the backend's raw `event.label` strings map onto
// (see app/orchestrator/pipeline.py's 8 emission points — two of them share a
// stage: the cache-hit variant of stage 1, and "Checking memory"/"Portfolio
// Manager finalizing" both belong to stage 6). Order matters: matching walks
// this list in order and takes the first hit.
export const ANALYZE_PIPELINE_STAGES = [
  { key: 'analysts', labelKey: 'deskChainAnalysts', test: label => label.startsWith('Running Market, Fundamentals') || label.startsWith('Reusing recent analyst reports') },
  { key: 'debate', labelKey: 'deskChainDebate', test: label => label.startsWith('Bull vs. Bear debate') },
  { key: 'research', labelKey: 'pipelineStageResearch', test: label => label.startsWith('Research Manager synthesizing') },
  { key: 'trader', labelKey: 'pipelineStageTrader', test: label => label.startsWith('Trader drafting') },
  { key: 'risk', labelKey: 'deskChainRisk', test: label => label.startsWith('Risk debate') },
  { key: 'pm', labelKey: 'deskChainPm', test: label => label.startsWith('Checking memory') || label.startsWith('Portfolio Manager finalizing') },
  { key: 'execution', labelKey: 'pipelineStageExecution', test: label => label.startsWith('Recording the trade') },
];

export function matchAnalyzePipelineStage(label) {
  return ANALYZE_PIPELINE_STAGES.findIndex(stage => stage.test(label));
}

export async function runAnalyze() {
  const symbol = currentDetailSymbol;
  if (!symbol) return;
  const btn = document.getElementById('detail-analyze');
  const status = document.getElementById('detail-analyze-status');
  const analysis = document.getElementById('detail-analysis');
  const pipelineEl = document.getElementById('analyze-pipeline');
  const pipelineSteps = pipelineEl.querySelectorAll('.ap-step');
  const captionEl = document.getElementById('analyze-pipeline-caption');
  btn.disabled = true;
  status.textContent = t('analyzeStarting');
  analysis.hidden = false;
  analysis.innerHTML = `<span class="empty">${t('analyzeThinking')}</span>`;
  pipelineEl.hidden = false;
  pipelineSteps.forEach(el => el.classList.remove('done', 'active'));
  captionEl.textContent = '';
  let activeStageIndex = -1;
  let done = false;
  try {
    const res = await fetch(`/api/analyze/${encodeURIComponent(symbol)}`, { method: 'POST' });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done: streamDone, value } = await reader.read();
      if (streamDone) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split('\n\n');
      buffer = chunks.pop();
      for (const chunk of chunks) {
        if (!chunk.startsWith('data: ')) continue;
        const event = JSON.parse(chunk.slice(6));
        if (event.type === 'stage') {
          captionEl.textContent = event.label;
          const foundIndex = matchAnalyzePipelineStage(event.label);
          if (foundIndex > activeStageIndex) {
            pipelineSteps.forEach((el, i) => {
              el.classList.toggle('done', i < foundIndex);
              el.classList.toggle('active', i === foundIndex);
            });
            activeStageIndex = foundIndex;
          }
          // foundIndex === activeStageIndex (a repeat round, e.g. debate
          // round 2/2) or foundIndex === -1 (unmatched label): leave step
          // classes as-is, caption text above already carries the raw label.
        } else if (event.type === 'done') {
          done = true;
          pipelineSteps.forEach(el => el.classList.add('done'));
          pipelineSteps.forEach(el => el.classList.remove('active'));
          pipelineEl.hidden = true;
          captionEl.textContent = '';
          if (event.error_code) {
            status.textContent = '';
            analysis.textContent = event.error_code === 'no_decision'
              ? t('analyzeErrorNoDecision', event.ticker || symbol)
              : t('analyzeErrorFailed', event.ticker || symbol);
          } else {
            status.textContent = t('analyzeRatingLabel', event.rating);
            analysis.innerHTML = '';
            const body = document.createElement('div');
            // event.answer is the same render_portfolio_decision markdown as
            // the pending-approval modal's o.reason (signals.js's
            // showPendingApprovalModal) — mirror that exact pipeline
            // (enrichPriceFields BEFORE localizeDecisionLabels, then
            // renderMarkdown, which escapes before adding any tag) so the two
            // surfaces render the same payload identically and don't drift
            // apart again. Without this, the user saw literal "**Rating**:
            // Buy" asterisks instead of formatted markdown.
            body.innerHTML = renderMarkdown(localizeDecisionLabels(enrichPriceFields(event.answer, refPrice(symbol), state[symbol]?.currency)));
            analysis.appendChild(body);
            if (event.decision_id) {
              const more = document.createElement('span');
              more.className = 'see-more';
              more.style.cssText = 'display:inline-block;margin-top:0.6rem;cursor:pointer;color:var(--accent)';
              const tr = document.createElement('div');
              tr.className = 'transcript';
              tr.style.display = 'none';
              analysis.appendChild(more);
              analysis.appendChild(tr);
              wireTranscriptToggle(more, tr, event.decision_id, { open: t('hideDebate'), closed: t('seeDebate') });
            }
          }
        }
      }
    }
    if (!done) throw new Error('stream ended early');
  } catch {
    status.textContent = '';
    analysis.textContent = t('analyzeErrorGeneric');
  } finally {
    btn.disabled = false;
  }
}

export async function renderNews(symbol) {
  const el = document.getElementById('detail-news');
  el.innerHTML = `<div class="empty">${t('newsLoading')}</div>`;
  try {
    const items = await fetch(`/api/news/${encodeURIComponent(symbol)}`).then(r => r.json());
    if (currentDetailSymbol !== symbol) return;  // modal moved on while this was in flight
    if (!items.length) {
      el.innerHTML = `<div class="empty">${t('newsEmpty')}</div>`;
      return;
    }
    // KIS's news TRs (domestic/overseas) carry no article link or
    // thumbnail — a plain (non-clickable) row instead of the old <a> card,
    // rather than pointing "read more" at nowhere.
    el.innerHTML = items.map(item => `
      <div class="news-item news-item-static">
        <div class="news-item-text">
          <div class="news-title">${escapeHtml(item.title)}</div>
          <div class="news-meta">${escapeHtml(item.publisher)}${item.published_at ? ` · ${formatTimeAgo(item.published_at)}` : ''}</div>
        </div>
      </div>
    `).join('');
  } catch {
    if (currentDetailSymbol === symbol) el.innerHTML = `<div class="empty">${t('newsError')}</div>`;
  }
}

/* ---------------- detail modal: earnings trends ---------------- */

export const earningsTrendState = {};  // symbol -> { data, period: 'quarterly'|'annual' }

export const ESTIMATE_PERIOD_ORDER = ['0q', '+1q', '0y', '+1y'];
export function estimatePeriodLabel(p) { return t(`period${p}`); }

export async function renderEarningsTrend(symbol) {
  try {
    const data = await fetch(`/api/earnings-trend/${encodeURIComponent(symbol)}`).then(r => r.json());
    if (currentDetailSymbol !== symbol) return;
    earningsTrendState[symbol] = { data, period: 'quarterly' };
    document.querySelectorAll('#revenue-period-toggle button').forEach(b => b.classList.toggle('active', b.dataset.period === 'quarterly'));
    drawEpsChart(symbol);
    drawRevChart(symbol);
    renderEstimateTable('revenue-estimate-table', data.revenue_estimate, 'revenue');
    renderEstimateTable('earnings-estimate-table', data.earnings_estimate, 'eps');
  } catch {
    // Degrades quietly — an empty earnings-trends section isn't worth an error banner in a modal that's mostly about price.
  }
}

export function renderChartAxisY(elId, min, max, formatFn, steps = 5) {
  const el = document.getElementById(elId);
  let labels = '';
  for (let i = 0; i <= steps; i++) {
    labels += `<span>${formatFn(max - (i / steps) * (max - min))}</span>`;
  }
  el.innerHTML = labels;
}

// Toss-증권 스타일 마켓위젯 스파크라인 — 축/격자/점 없이 선 하나만(2026-08-19,
// 사용자와 확정). drawCandles와 같은 dpr 스케일링 규약을 따름.
export function drawSparkline(canvas, values, isUp) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 100, h = canvas.clientHeight || 32;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!values || values.length < 2) return;

  const lo = Math.min(...values), hi = Math.max(...values);
  const span = (hi - lo) || 1;
  const padY = h * 0.12;
  const x = i => (i / (values.length - 1)) * w;
  const y = v => padY + (1 - (v - lo) / span) * (h - padY * 2);

  ctx.strokeStyle = isUp ? (getVar('--up') || '#e34948') : (getVar('--down') || '#2a78d6');
  ctx.lineWidth = 1.75;
  ctx.beginPath();
  values.forEach((v, i) => { const px = x(i), py = y(v); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); });
  ctx.stroke();
}

export function drawEpsChart(symbol, hoverIdx = null) {
  const trend = earningsTrendState[symbol];
  const canvas = document.getElementById('eps-canvas');
  const axisXEl = document.getElementById('eps-axis-x');
  const history = trend?.data?.eps_history || [];
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth * 2;
  const h = canvas.height = canvas.clientHeight * 2;
  ctx.clearRect(0, 0, w, h);
  if (!history.length) {
    axisXEl.innerHTML = '';
    document.getElementById('eps-axis-y').innerHTML = '';
    document.getElementById('eps-info-line').innerHTML = '';
    return;
  }

  const currency = state[symbol]?.currency;
  const values = history.flatMap(p => [p.estimate, p.actual]).filter(v => v != null);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = (max - min) || Math.abs(max) * 0.1 || 1;
  const padTop = 20, padBot = 20, padSide = 26;
  const x = i => padSide + (history.length === 1 ? (w - padSide * 2) / 2 : (i / (history.length - 1)) * (w - padSide * 2));
  const y = v => h - padBot - ((v - min) / range) * (h - padTop - padBot);
  const activeIdx = hoverIdx != null ? hoverIdx : history.length - 1;

  renderChartAxisY('eps-axis-y', min, max, v => fmtPrice(v, currency));

  ctx.strokeStyle = getVar('--grid');
  ctx.lineWidth = 1;
  for (let i = 1; i < 3; i++) {
    const gy = padTop + ((h - padTop - padBot) / 3) * i;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
  }

  if (hoverIdx != null) {
    ctx.strokeStyle = getVar('--ink-2');
    ctx.globalAlpha = 0.35;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x(hoverIdx), 0); ctx.lineTo(x(hoverIdx), h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  history.forEach((p, i) => {
    const px = x(i);
    const hasActual = p.actual != null && p.estimate != null;
    const beat = hasActual && p.actual >= p.estimate;
    const color = hasActual ? (beat ? getVar('--up') : getVar('--down')) : getVar('--muted');
    const r = i === activeIdx ? 8 : 7;

    if (hasActual) {
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(px, y(p.estimate)); ctx.lineTo(px, y(p.actual)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    if (p.estimate != null) {
      ctx.beginPath();
      ctx.arc(px, y(p.estimate), r, 0, Math.PI * 2);
      ctx.fillStyle = getVar('--surface');
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
    if (p.actual != null) {
      ctx.beginPath();
      ctx.arc(px, y(p.actual), r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  });

  axisXEl.innerHTML = history.map((p, i) => {
    let sub = `<span style="color:var(--muted)">${t('epsPending')}</span>`;
    if (p.actual != null && p.estimate != null) {
      const diff = p.actual - p.estimate;
      const beat = diff >= 0;
      const sign = beat ? '+' : '-';
      sub = `<span style="color:${beat ? getVar('--up') : getVar('--down')}">${beat ? t('epsBeat') : t('epsMiss')} ${sign}${fmtPrice(Math.abs(diff), currency)}</span>`;
    }
    const style = i === activeIdx ? '' : 'opacity:0.6';
    return `<span style="display:flex;flex-direction:column;align-items:center;gap:0.15rem;${style}"><b style="font-weight:600;color:var(--ink)">${escapeHtml(p.quarter)}</b>${sub}</span>`;
  }).join('');

  renderEpsInfoLine(symbol, activeIdx);
}

export function renderEpsInfoLine(symbol, idx) {
  const trend = earningsTrendState[symbol];
  const history = trend?.data?.eps_history || [];
  const p = history[idx];
  const el = document.getElementById('eps-info-line');
  if (!p) { el.innerHTML = ''; return; }
  const currency = state[symbol]?.currency;
  const hasActual = p.actual != null && p.estimate != null;
  const color = hasActual ? (p.actual >= p.estimate ? getVar('--up') : getVar('--down')) : getVar('--muted');
  el.innerHTML = `
    <span class="info-period">${escapeHtml(p.quarter)}</span>
    ${p.estimate != null ? `<span><i class="legend-dot hollow"></i>${t('estimateLabel')} ${fmtPrice(p.estimate, currency)}</span>` : ''}
    ${p.actual != null ? `<span><i class="legend-dot" style="background:${color}"></i>${t('actualLabel')} ${fmtPrice(p.actual, currency)}</span>` : ''}
  `;
}

export function drawRevChart(symbol, hoverIdx = null) {
  const trend = earningsTrendState[symbol];
  const canvas = document.getElementById('rev-canvas');
  const axisXEl = document.getElementById('rev-axis-x');
  const key = trend?.period === 'annual' ? 'revenue_earnings_annual' : 'revenue_earnings_quarterly';
  const points = trend?.data?.[key] || [];
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth * 2;
  const h = canvas.height = canvas.clientHeight * 2;
  ctx.clearRect(0, 0, w, h);
  if (!points.length) {
    axisXEl.innerHTML = '';
    document.getElementById('rev-axis-y').innerHTML = '';
    document.getElementById('rev-info-line').innerHTML = '';
    return;
  }

  // Bounded to include 0 even when every value is positive, so bars always
  // have a real zero baseline to grow from — some quarters here do have
  // negative earnings, and treating 0 as just another data point (not a
  // guaranteed axis bound) would draw those bars growing the wrong way.
  const values = points.flatMap(p => [p.revenue, p.earnings]).filter(v => v != null);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const fullRange = (max - min) || 1;
  const padTop = 15, padBot = 10;
  const groupW = w / points.length;
  const barW = Math.min(groupW * 0.32, 40);
  const baseline = h - padBot;
  const y = v => baseline - ((v - min) / fullRange) * (baseline - padTop);
  const zeroY = y(0);
  const activeIdx = hoverIdx != null ? hoverIdx : points.length - 1;

  renderChartAxisY('rev-axis-y', min, max, fmtCompact);

  ctx.strokeStyle = getVar('--grid');
  ctx.lineWidth = 1;
  for (let i = 1; i < 3; i++) {
    const gy = padTop + ((baseline - padTop) / 3) * i;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
  }
  if (min < 0) {
    ctx.strokeStyle = getVar('--muted');
    ctx.globalAlpha = 0.5;
    ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (hoverIdx != null) {
    ctx.fillStyle = getVar('--surface-2');
    ctx.globalAlpha = 0.6;
    ctx.fillRect(groupW * hoverIdx, 0, groupW, h);
    ctx.globalAlpha = 1;
  }

  const bar = (cx, value, color) => {
    if (value == null) return;
    const by = y(value);
    const top = Math.min(by, zeroY);
    const barHeight = Math.abs(zeroY - by);
    ctx.fillStyle = color;
    ctx.beginPath();
    const radii = value >= 0 ? [3, 3, 0, 0] : [0, 0, 3, 3];
    if (ctx.roundRect) ctx.roundRect(cx, top, barW, barHeight, radii);
    else ctx.rect(cx, top, barW, barHeight);
    ctx.fill();
  };

  points.forEach((p, i) => {
    const centerX = groupW * i + groupW / 2;
    bar(centerX - barW - 2, p.revenue, getVar('--accent'));
    bar(centerX + 2, p.earnings, getVar('--amber'));
  });

  axisXEl.innerHTML = points.map((p, i) => {
    const style = i === activeIdx ? 'font-weight:600;color:var(--ink)' : '';
    return `<span style="${style}">${escapeHtml(p.label)}</span>`;
  }).join('');

  renderRevInfoLine(symbol, activeIdx);
}

export function renderRevInfoLine(symbol, idx) {
  const trend = earningsTrendState[symbol];
  const key = trend?.period === 'annual' ? 'revenue_earnings_annual' : 'revenue_earnings_quarterly';
  const points = trend?.data?.[key] || [];
  const p = points[idx];
  const el = document.getElementById('rev-info-line');
  if (!p) { el.innerHTML = ''; return; }
  el.innerHTML = `
    <span class="info-period">${escapeHtml(p.label)}</span>
    <span><i class="legend-dot revenue"></i>${t('revenueLabel')} ${p.revenue != null ? fmtCompact(p.revenue) : '—'}</span>
    <span><i class="legend-dot earnings"></i>${t('earningsLabel')} ${p.earnings != null ? fmtCompact(p.earnings) : '—'}</span>
  `;
}

export function wireEarningsHover() {
  const epsCanvas = document.getElementById('eps-canvas');
  epsCanvas.addEventListener('pointermove', (e) => {
    const symbol = currentDetailSymbol;
    const history = earningsTrendState[symbol]?.data?.eps_history;
    if (!history || !history.length) return;
    const rect = epsCanvas.getBoundingClientRect();
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const idx = Math.round(relX * (history.length - 1));
    drawEpsChart(symbol, idx);
  });
  epsCanvas.addEventListener('pointerleave', () => {
    if (currentDetailSymbol) drawEpsChart(currentDetailSymbol);
  });

  const revCanvas = document.getElementById('rev-canvas');
  revCanvas.addEventListener('pointermove', (e) => {
    const symbol = currentDetailSymbol;
    const trend = earningsTrendState[symbol];
    const key = trend?.period === 'annual' ? 'revenue_earnings_annual' : 'revenue_earnings_quarterly';
    const points = trend?.data?.[key];
    if (!points || !points.length) return;
    const rect = revCanvas.getBoundingClientRect();
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const idx = Math.min(points.length - 1, Math.floor(relX * points.length));
    drawRevChart(symbol, idx);
  });
  revCanvas.addEventListener('pointerleave', () => {
    if (currentDetailSymbol) drawRevChart(currentDetailSymbol);
  });
}

export function renderEstimateTable(containerId, table, kind) {
  const el = document.getElementById(containerId);
  const periods = ESTIMATE_PERIOD_ORDER.filter(p => table && table[p]);
  if (!periods.length) {
    el.innerHTML = `<div class="empty">${t('estimatesEmpty')}</div>`;
    return;
  }
  const fmtVal = v => (v == null ? '—' : kind === 'revenue' ? fmtCompact(v) : v.toFixed(2));
  const rows = [
    [t('estAvg'), p => fmtVal(table[p].avg)],
    [t('estLow'), p => fmtVal(table[p].low)],
    [t('estHigh'), p => fmtVal(table[p].high)],
    [t('estYearAgo'), p => fmtVal(table[p].year_ago)],
    [t('estGrowth'), p => (table[p].growth == null ? '—' : `${(table[p].growth * 100).toFixed(1)}%`)],
    [t('estAnalysts'), p => table[p].num_analysts ?? '—'],
  ];
  el.innerHTML = `
    <table class="estimate-table">
      <thead><tr><th></th>${periods.map(p => `<th>${estimatePeriodLabel(p)}</th>`).join('')}</tr></thead>
      <tbody>
        ${rows.map(([label, fn]) => `<tr><td>${label}</td>${periods.map(p => `<td class="num">${fn(p)}</td>`).join('')}</tr>`).join('')}
      </tbody>
    </table>
  `;
}

document.getElementById('revenue-period-toggle').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-period]');
  if (!btn || !currentDetailSymbol) return;
  const trend = earningsTrendState[currentDetailSymbol];
  if (!trend) return;
  trend.period = btn.dataset.period;
  document.querySelectorAll('#revenue-period-toggle button').forEach(b => b.classList.toggle('active', b === btn));
  drawRevChart(currentDetailSymbol);
});

// 계좌 payload가 도착할 때마다 (a) 이후 재렌더용 캐시 (b) 기타탭 설정의 KIS
// 연결 배지 (c) 사이드바 상시 요약 — 이 셋만 갱신한다. 예전엔 트레이딩 탭
// 히어로(#hero)까지 그렸으나, 총평가금액이 사이드바/히어로/포트폴리오 요약
// 3곳에서 서로 다른 스코프 규칙으로 보이는 문제 때문에 히어로를 제거함
// (2026-08-18). 국내·해외 상세 수치는 기타 → 포트폴리오 상세의
// #portfolio-summary 카드가 활성 스코프 기준으로 단독 표시.
export let _lastAccount = null, _lastHeroRows = [];
export function renderAccountSummary(account, rows) {
  _lastAccount = account; _lastHeroRows = rows || [];
  const pill = document.getElementById('kis-pill');
  if (pill) {
    pill.style.display = account ? 'inline-flex' : 'none';
    if (account) {
      pill.textContent = t(account.is_real ? 'kisConnectedReal' : 'kisConnected');
      pill.classList.toggle('real', !!account.is_real);
    }
  }
  const sidebarSummary = document.getElementById('sidebar-acct-summary');
  if (!account) { sidebarSummary.hidden = true; return; }
  sidebarSummary.hidden = false;

  // 사이드바 요약본 — 어느 탭에 있든 항상 보이는 압축 표기(국내 KRW 고정).
  // 스코프 연동은 일부러 안 함 — 라벨(sidebarAcctLabel)이 "국내(KRW)"임을
  // 명시하고, 스코프별 숫자는 #portfolio-summary가 담당.
  const pnl = account.total_pnl_amount;
  const pnlPct = account.purchase_amount > 0 ? ` (${((pnl / account.purchase_amount) * 100).toFixed(2)}%)` : '';
  document.getElementById('sidebar-acct-total').textContent = fmtKRW(account.total_assets);
  const sidebarPnl = document.getElementById('sidebar-acct-pnl');
  sidebarPnl.textContent = `${fmtKRW(pnl, { signed: true })}${pnlPct}`;
  sidebarPnl.classList.toggle('up', pnl > 0);
  sidebarPnl.classList.toggle('down', pnl < 0);
}

// KRW/USD 토글 — 기타 → 포트폴리오 상세의 요약 카드(#portfolio-summary)에
// 붙어 있음. 테마 토글처럼 localStorage에 저장, 기본값은 USD(off) —
// account.usd_krw_rate가 비어 있을 수 있으므로.
export let _fxShowKRW = localStorage.getItem('overseasFxKrw') === '1';
export function applyFxToggleButton() {
  // 이 카드는 활성 스코프 하나만 보여주므로, 국내에서는 환산 토글이 의미
  // 없어 숨긴다. 스코프 라벨 자체는 renderPortfolioSummaryCard가 씀.
  const btn = document.getElementById('fx-toggle');
  if (!btn) return;   // 마크업이 바뀌어도 모듈 로드 자체는 절대 깨지지 않게
  btn.hidden = _marketScope !== 'overseas';
  btn.textContent = t(_fxShowKRW ? 'fxToggleToUSD' : 'fxToggleToKRW');
}
document.getElementById('fx-toggle')?.addEventListener('click', () => {
  _fxShowKRW = !_fxShowKRW;
  localStorage.setItem('overseasFxKrw', _fxShowKRW ? '1' : '0');
  applyFxToggleButton();
  if (_lastAccount) renderPortfolioSummaryCard(_lastAccount);
});
applyFxToggleButton();

// 실전/모의 판정 — /api/trades 행은 전부 "현재 활성" 브로커의 체결 이력이라
// (routers/portfolio.py의 recent_executions) 행별 속성이 아니라 페이지 전역이다.
// null = 아직 모름. loadRuntimeSettings()는 initOrderTicket() 안에서 호출돼
// pollDashboardSummary()보다 늦게 끝날 수 있고(main.js:212 vs :220), 반대로
// _lastAccount는 renderTrades() 직전 같은 tick에 채워진다(:1856→:1861).
// 둘 다 없을 때 '모의'로 단정하면 실계좌를 모의로 오표기 — 그래서 tri-state.
export function isRealAccountMode() {
  if (_runtimeSettings) return _runtimeSettings.kis_account_mode === 'real';
  if (_lastAccount) return !!_lastAccount.is_real;
  return null;
}

export function venueChip(venue) {
  if (venue !== 'kis') return '—';
  const isReal = isRealAccountMode();
  if (isReal === null) return `<span class="venue-chip kis">${t('venueKisUnknown')}</span>`;
  const cls = isReal ? 'venue-chip kis real' : 'venue-chip kis';
  return `<span class="${cls}">${t(isReal ? 'venueKisReal' : 'venueKisPaper')}</span>`;
}

export function renderHoldBadges(rows) {
  for (const r of rows) {
    const el = document.getElementById(`hold-${cssSafe(r.ticker)}`);
    if (!el) continue;
    if (r.shares > 0) {
      el.textContent = t('holdingBadge', r.shares);
      el.style.display = 'inline-block';
    } else {
      el.style.display = 'none';
    }
  }
}

// "포트폴리오 상세" is meant to be "what do I actually hold," not "everything
// on my trade allowlist" — /api/portfolio returns one row per allowlist
// ticker (0-share ones included, so the AI's current rating on a
// not-yet-bought candidate is visible elsewhere), which made this table
// look like a near-duplicate of 보유종목 with a few extra all-zero rows
// (2026-07-30 user feedback). /api/portfolio only ever sets venue "kis" or
// null now (the local paper ledger's shares/cash are no longer surfaced here
// at all — see that endpoint's docstring), so this is really just
// "shares > 0", kept explicit in case that ever changes again.
function _heldOnly(rows) {
  return rows.filter(r => r.venue === 'kis' && r.shares > 0);
}

export let _portfolioTableCache = { rows: [], latestRating: {}, account: null };

// Column-sort state for "포트폴리오 상세" — client-side only, since
// pollPortfolio already caches the full row set in _portfolioTableCache
// (see its own docstring), so re-sorting never needs a network round trip.
// `key: null` (initial state) keeps the table in whatever order /api/portfolio
// returned it in, same as before this feature existed.
let _portfolioSort = { key: null, dir: 1 };

// `rating`/`weight_pct` aren't fields on the row itself — `rating` comes
// from the separate decisions list, `weight_pct` is derived from the
// account snapshot (see renderPortfolioTable) — so the caller merges both
// onto a shallow row copy before sorting/rendering, and this (and the
// render template below) can just read r.rating / r.weight_pct directly.
function _portfolioSortValue(r, key) {
  switch (key) {
    case 'name': return r.name || r.ticker;
    case 'rating': return r.rating || '';
    case 'shares': return r.shares ?? 0;
    case 'avg_price': return r.avg_price;
    case 'current_price': return r.current_price;
    case 'stop_loss': return r.stop_loss;
    case 'pnl_amount': return r.pnl_amount;
    case 'pnl_pct': return r.pnl_pct;
    case 'weight_pct': return r.weight_pct;
    case 'market_value': return r.market_value;
    default: return 0;
  }
}

// `null`/`undefined` sorts last regardless of direction (flipping `dir` on a
// missing value would otherwise yo-yo it between the top and bottom of the
// list every other click) — matches the table's own '—' placeholder for a
// value that isn't there, rather than treating it as 0.
function _comparePortfolioValues(a, b, dir) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === 'string' || typeof b === 'string') {
    return dir * String(a).localeCompare(String(b), 'ko');
  }
  return dir * (a - b);
}

export function togglePortfolioSort(key) {
  _portfolioSort = _portfolioSort.key === key
    ? { key, dir: -_portfolioSort.dir }
    : { key, dir: 1 };
  renderPortfolioTable();
}

// 시각 기호(▲▼)와 aria-sort를 항상 한 곳에서 같이 갱신 — 둘이 갈라지면
// 화면과 스크린리더가 서로 다른 정렬 상태를 말하게 된다. .sort-ind는
// aria-hidden이라 스크린리더 쪽 정보원은 aria-sort 하나뿐.
function _updatePortfolioSortIndicators() {
  const head = document.getElementById('portfolio-head');
  if (!head) return;
  for (const th of head.querySelectorAll('th.sortable')) {
    const active = th.dataset.sortKey === _portfolioSort.key;
    const ind = th.querySelector('.sort-ind');
    ind.textContent = active ? (_portfolioSort.dir > 0 ? '▲' : '▼') : '';
    // 정렬 중인 열에만 aria-sort를 둔다(ARIA 규칙: 한 번에 한 열).
    // 나머지는 'none'을 적어두는 대신 속성을 지운다 — 일부 스크린리더가
    // aria-sort="none"을 "정렬 없음"이라고 매 열마다 소리 내어 읽어서,
    // 10열짜리 헤더를 지나가는 것만으로 소음이 된다.
    if (active) th.setAttribute('aria-sort', _portfolioSort.dir > 0 ? 'ascending' : 'descending');
    else th.removeAttribute('aria-sort');
  }
}

export async function pollPortfolio() {
  // Decisions ride along so each row can show the agent's latest rating —
  // the /api/decisions list is newest-first, so first hit per ticker wins.
  // portfolio itself falls back to the last-rendered state on a transient
  // failure (network blip, slow backend) rather than throwing and leaving
  // this 30s poll cycle silently skipped with no visible change.
  const [data, decisions] = await Promise.all([
    fetch('/api/portfolio').then(r => r.json()).catch(() => null),
    fetch('/api/decisions?limit=100').then(r => r.json()).catch(() => []),
  ]);
  if (!data) return;
  const latestRating = {};
  for (const d of decisions) if (!(d.ticker in latestRating)) latestRating[d.ticker] = d.rating;
  renderAccountSummary(data.account, data.rows);
  renderHoldBadges(data.rows);
  _portfolioTableCache = { rows: _heldOnly(data.rows), latestRating, account: data.account };
  renderPortfolioTable();
}

// account.total_assets (domestic, KRW) / account.overseas_total_assets (USD)
// — see app.web.state.kis_account_snapshot's own docstring for why these
// stay two separate per-currency totals rather than one combined figure
// (mixing USD into the KRW total would need FX). Matches _marketScope's own
// domestic/overseas split, so a row's weight is always relative to the
// account total in ITS OWN currency, never mismatched.
function _portfolioTotalEquity(account) {
  if (!account) return null;
  return _marketScope === 'domestic' ? account.total_assets : account.overseas_total_assets;
}

// 포트폴리오 상세 표 위의 계좌 요약 — 활성 국내/해외 스코프 하나만 보여준다
// (아래 표가 이미 _marketScope로 같게 필터되므로 짝이 맞음). 계좌 상세 수치를
// 보여주는 화면은 2026-08-18부로 이 카드가 유일 — 트레이딩 탭 히어로는
// 스코프 규칙이 서로 다른 3중 표시를 만들어서 제거됨.
function renderPortfolioSummaryCard(account) {
  const card = document.getElementById('portfolio-summary');
  applyFxToggleButton();   // 스코프가 바뀌면 ₩/$ 버튼 노출 여부도 같이 바뀜
  if (!account) { card.style.display = 'none'; return; }
  card.style.display = 'flex';
  if (_marketScope === 'domestic') {
    document.getElementById('portfolio-summary-scope').textContent = t('scopeDomestic');
    document.getElementById('portfolio-summary-total').textContent = fmtKRW(account.total_assets);
    document.getElementById('portfolio-summary-eval').textContent = fmtKRW(account.total_eval_amount);
    document.getElementById('portfolio-summary-cash-label').textContent = t('heroCashLabel');
    document.getElementById('portfolio-summary-cash').textContent = fmtKRW(account.cash_deposit);
    document.getElementById('portfolio-summary-invested').textContent = fmtKRW(account.purchase_amount);
    document.getElementById('portfolio-summary-d2cash-label').textContent = t('heroD2CashLabel');
    document.getElementById('portfolio-summary-d2cash').textContent = fmtKRW(account.d2_cash_deposit);
    document.getElementById('portfolio-summary-positions').textContent = `${account.domestic_count ?? account.position_count}`;
    const pnl = account.total_pnl_amount;
    const pnlPct = account.purchase_amount > 0 ? ` (${((pnl / account.purchase_amount) * 100).toFixed(2)}%)` : '';
    document.getElementById('portfolio-summary-pnl-pill').textContent = `${t('pnlLabel')} ${fmtKRW(pnl, { signed: true })}${pnlPct}`;
  } else {
    // 같은 카드 헤더의 ₩/$ 토글(_fxShowKRW) 값을 그대로 따름.
    const rate = _fxShowKRW && account.usd_krw_rate ? account.usd_krw_rate : null;
    const fmtFx = (v, opts) => rate ? fmtKRW(v == null ? null : v * rate, opts) : fmtUSD(v, opts);
    document.getElementById('portfolio-summary-scope').textContent = t(_fxShowKRW ? 'scopeOverseasKrw' : 'scopeOverseas');
    document.getElementById('portfolio-summary-total').textContent = fmtFx(account.overseas_total_assets);
    document.getElementById('portfolio-summary-eval').textContent = fmtFx(account.overseas_eval_amount);
    document.getElementById('portfolio-summary-cash-label').textContent = t('heroForeignAcctCashLabel');
    document.getElementById('portfolio-summary-cash').textContent = fmtFx(account.overseas_cash_deposit);
    document.getElementById('portfolio-summary-invested').textContent = fmtFx(account.overseas_purchase_amount);
    document.getElementById('portfolio-summary-d2cash-label').textContent = t('heroForeignCashLabel');
    document.getElementById('portfolio-summary-d2cash').textContent = fmtFx(account.overseas_foreign_cash_deposit);
    document.getElementById('portfolio-summary-positions').textContent = `${account.overseas_count || 0}`;
    const uPnl = account.overseas_pnl_amount || 0;
    const uCost = account.overseas_purchase_amount || 0;
    const uPct = uCost > 0 ? ` (${((uPnl / uCost) * 100).toFixed(2)}%)` : '';
    document.getElementById('portfolio-summary-pnl-pill').textContent = `${t('pnlLabel')} ${fmtFx(uPnl, { signed: true })}${uPct}`;
  }
}

// 손절가까지 남은 여유를 현재가 대비 %로. 손절선이 숫자로만 있으면 "지금 위험한가"를
// 읽으려면 두 숫자를 암산해야 하는데, 그게 이 표에서 가장 급한 정보임.
// 음수 = 이미 손절선 아래(기계적 손절 루프가 처리할 상태), 0에 가까울수록 임박.
// 강조 기준은 지어내지 않고 서버가 준 _heldReviewBufferPct(장중 재검토를 강제하는
// 바로 그 값)를 씀 — 아직 못 받았으면 강조 없이 숫자만.
function _stopDistanceCell(r) {
  const price = r.stop_loss != null ? fmtPrice(r.stop_loss, r.currency) : '—';
  if (r.stop_loss == null || r.current_price == null || r.current_price <= 0) {
    return `<td class="num">${price}</td>`;
  }
  const frac = (r.current_price - r.stop_loss) / r.current_price;
  const cls = frac <= 0 ? 'stop-breached' : (_heldReviewBufferPct != null && frac <= _heldReviewBufferPct) ? 'stop-near' : '';
  const pct = (frac * 100).toFixed(1);
  const label = frac <= 0 ? t('stopDistBreached') : t('stopDistRemaining', pct);
  return `<td class="num"><span class="stop-cell ${cls}" title="${escapeHtml(label)}">${price}<span class="stop-dist">${escapeHtml(frac <= 0 ? t('stopDistBreachedShort') : pct + '%')}</span></span></td>`;
}

export function renderPortfolioTable() {
  const { rows: allRows, latestRating, account } = _portfolioTableCache;
  renderPortfolioSummaryCard(account);
  const totalEquity = _portfolioTotalEquity(account);
  const rows = allRows
    .filter(r => (_marketScope === 'domestic') === isDomesticSymbol(r.ticker))
    .map(r => ({
      ...r,
      rating: latestRating[r.ticker],
      weight_pct: (r.market_value != null && totalEquity) ? (r.market_value / totalEquity) * 100 : null,
    }));
  if (_portfolioSort.key) {
    const { key, dir } = _portfolioSort;
    rows.sort((a, b) => _comparePortfolioValues(_portfolioSortValue(a, key), _portfolioSortValue(b, key), dir));
  }
  const body = document.getElementById('portfolio-body');
  document.getElementById('portfolio-count').textContent = rows.length ? t('itemCount', rows.length) : '';
  _updatePortfolioSortIndicators();
  const visible = expanded.portfolio ? rows : rows.slice(0, LIST_CAP);
  body.innerHTML = visible.map(r => {
    const pnlCls = r.pnl_amount > 0 ? 'pnl-up' : r.pnl_amount < 0 ? 'pnl-down' : '';
    const rowCls = r.pnl_amount > 0 ? 'pnl-row-up' : r.pnl_amount < 0 ? 'pnl-row-down' : '';
    const pnlSign = r.pnl_amount > 0 ? '+' : '';
    return `
    <tr class="${rowCls}" role="button" tabindex="0" data-symbol="${escapeHtml(r.ticker)}">
      <td>${escapeHtml(r.name)} <span class="sym">${escapeHtml(displayTicker(r.ticker))}</span></td>
      <td>${r.rating ? `<span class="rating-chip ${r.rating.toLowerCase()}">${escapeHtml(r.rating)}</span>` : '—'}</td>
      <td class="num">${r.shares.toLocaleString('en-US')}</td>
      <td class="num">${r.avg_price != null ? fmtPrice(r.avg_price, r.currency) : '—'}</td>
      <td class="num">${r.current_price != null ? fmtPrice(r.current_price, r.currency) : '—'}</td>
      ${_stopDistanceCell(r)}
      <td class="num ${pnlCls}">${r.pnl_amount != null ? pnlSign + fmtPrice(r.pnl_amount, r.currency) : '—'}</td>
      <td class="num ${pnlCls}">${r.pnl_pct != null ? pnlSign + Number(r.pnl_pct).toFixed(2) + '%' : '—'}</td>
      <td class="num">${r.weight_pct != null ? Number(r.weight_pct).toFixed(1) + '%' : '—'}</td>
      <td class="num">${fmtPrice(r.market_value, r.currency)}</td>
    </tr>
  `;
  }).join('');
  // 행 클릭으로 상세 열기 — 보유종목 패널이 하던 역할을 이 표가 이어받음(2026-08-19).
  // [data-symbol] 선택자라 아래에서 append하는 "더보기"/합계 행에는 절대 걸리지 않는다.
  // 매매허용목록에 없는 보유 종목은 state[symbol] 자체가 없어 열 수 없으므로 가드.
  body.querySelectorAll('tr[data-symbol]').forEach(row => {
    const symbol = row.dataset.symbol;
    if (!state[symbol]) return;
    row.addEventListener('click', () => openDetail(symbol));
    row.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDetail(symbol); }
    });
  });
  if (rows.length > LIST_CAP) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 10;
    td.appendChild(moreButton('portfolio', rows.length - LIST_CAP, renderPortfolioTable));
    tr.appendChild(td);
    body.appendChild(tr);
  }
  body.appendChild(_portfolioTotalRow(rows));
}

// Sums over every row in the current market scope (not just the truncated
// `visible` slice) — a "total" that silently excludes collapsed rows would
// be wrong, not just incomplete. Overall return % is weighted by cost basis
// (market_value - pnl_amount), not a plain average of each row's own %, so
// a large position's return counts proportionally more than a token one —
// same principle as the account-level PnL% shown elsewhere.
function _portfolioTotalRow(rows) {
  const tr = document.createElement('tr');
  tr.className = 'portfolio-total';
  if (rows.length === 0) {
    tr.innerHTML = '<td colspan="10"></td>';
    return tr;
  }
  const currency = rows[0].currency;
  let marketValue = 0, pnlAmount = 0, costBasis = 0, weightPct = 0, hasPnl = false, hasWeight = false;
  for (const r of rows) {
    if (r.market_value != null) marketValue += r.market_value;
    if (r.pnl_amount != null) {
      pnlAmount += r.pnl_amount;
      hasPnl = true;
    }
    if (r.weight_pct != null) {
      weightPct += r.weight_pct;
      hasWeight = true;
    }
  }
  costBasis = marketValue - pnlAmount;
  const pnlPct = hasPnl && costBasis > 0 ? (pnlAmount / costBasis) * 100 : null;
  const pnlCls = pnlAmount > 0 ? 'pnl-up' : pnlAmount < 0 ? 'pnl-down' : '';
  const pnlSign = pnlAmount > 0 ? '+' : '';
  tr.innerHTML = `
    <td>${escapeHtml(t('portfolioTotal'))}</td>
    <td></td><td></td><td></td><td></td><td></td>
    <td class="num ${pnlCls}">${hasPnl ? pnlSign + fmtPrice(pnlAmount, currency) : '—'}</td>
    <td class="num ${pnlCls}">${pnlPct != null ? pnlSign + pnlPct.toFixed(2) + '%' : '—'}</td>
    <td class="num">${hasWeight ? weightPct.toFixed(1) + '%' : '—'}</td>
    <td class="num">${fmtPrice(marketValue, currency)}</td>
  `;
  return tr;
}

export let _tradesCache = { rows: [], decisionByKey: {} };

export async function pollTrades() {
  // Decisions ride along to link each trade back to the decision that
  // produced it — same (ticker, trade_date) key the decision history uses.
  const [rows, decisions] = await Promise.all([
    fetch('/api/trades').then(r => r.json()),
    fetch('/api/decisions?limit=100').then(r => r.json()).catch(() => []),
  ]);
  const decisionByKey = {};
  for (const d of decisions) {
    const k = `${d.ticker}|${d.trade_date}`;
    if (!(k in decisionByKey)) decisionByKey[k] = d;
  }
  _tradesCache = { rows, decisionByKey };
  renderTrades();
}

// Consolidates the fixed-interval polls that pollPortfolio/pollIndex/
// pollTrades/loadOpenOrders each did on their own separate timer into one
// request to /api/dashboard-summary — same render functions/caches as
// those four, just fed from one round trip instead of four (plus the
// /api/decisions fetch pollPortfolio and pollTrades each duplicated).
// pollPortfolio/pollIndex/pollTrades/loadOpenOrders themselves are kept
// as-is (not removed) for the event-triggered refreshes elsewhere (after
// placing/cancelling/revising an order, or a fill notification) that need
// an immediate targeted re-fetch rather than waiting for the next tick of
// this shared interval.
export async function pollDashboardSummary() {
  let data;
  try { data = await fetch('/api/dashboard-summary').then(r => r.json()); } catch { return; }

  const latestRating = {};
  const decisionByKey = {};
  for (const d of data.decisions) {
    if (!(d.ticker in latestRating)) latestRating[d.ticker] = d.rating;
    const k = `${d.ticker}|${d.trade_date}`;
    if (!(k in decisionByKey)) decisionByKey[k] = d;
  }

  renderAccountSummary(data.portfolio.account, data.portfolio.rows);
  renderHoldBadges(data.portfolio.rows);
  _portfolioTableCache = { rows: _heldOnly(data.portfolio.rows), latestRating, account: data.portfolio.account };
  renderPortfolioTable();

  _tradesCache = { rows: data.trades, decisionByKey };
  renderTrades();

  renderIndexBar(data.index);
  renderOpenOrders(data.open_orders);
}

export function renderTrades() {
  const { rows: allRows, decisionByKey } = _tradesCache;
  const rows = allRows.filter(trade => (_marketScope === 'domestic') === isDomesticSymbol(trade.ticker));
  const body = document.getElementById('trades-body');
  document.getElementById('trades-count').textContent = rows.length ? t('itemCount', rows.length) : '';
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="8"><div class="empty">${t('tradesEmpty')}</div></td></tr>`;
    return;
  }
  const visible = expanded.trades ? rows : rows.slice(0, LIST_CAP);
  body.innerHTML = visible.map((trade, i) => {
    const d = decisionByKey[`${trade.ticker}|${trade.trade_date}`];
    const seeMore = d
      ? `<span class="see-more trade-see-more" data-decision-id="${d.id}" data-idx="${i}">${t('seeReasoning')}</span> `
      : '';
    const statusLabels = { filled: t('statusFilled'), partial: t('statusPartial'), pending: t('statusPending'), recorded: t('statusRecorded') };
    return `
    <tr>
      <td class="num" style="text-align:left">${trade.trade_date}</td>
      <td><span class="side-chip ${trade.side}">${trade.side === 'buy' ? t('buy') : t('sell')}</span></td>
      <td>${escapeHtml(trade.name)} <span class="sym">${escapeHtml(displayTicker(trade.ticker))}</span></td>
      <td class="num">${trade.shares.toLocaleString('en-US')}</td>
      <td class="num">${fmtPrice(trade.price, trade.currency)}</td>
      <td><span class="status-chip ${trade.status || 'recorded'}">${statusLabels[trade.status] || t('statusRecorded')}</span></td>
      <td>${venueChip(trade.venue)}</td>
      <td>${seeMore}<div class="trade-reason" title="${escapeHtml(trade.reason)}">${escapeHtml(trade.reason)}</div></td>
    </tr>
    <tr id="trade-transcript-${i}" style="display:none"><td colspan="8"><div class="transcript"></div></td></tr>
  `;
  }).join('');
  if (rows.length > LIST_CAP) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 8;
    td.appendChild(moreButton('trades', rows.length - LIST_CAP, renderTrades));
    tr.appendChild(td);
    body.appendChild(tr);
  }
  body.querySelectorAll('.trade-see-more').forEach(el => {
    const row = document.getElementById(`trade-transcript-${el.dataset.idx}`);
    const container = row.querySelector('.transcript');
    el.addEventListener('click', async () => {
      const isOpen = row.style.display !== 'none';
      if (isOpen) { row.style.display = 'none'; el.textContent = t('seeReasoning'); return; }
      row.style.display = '';
      el.textContent = t('hideReasoning');
      if (!el.dataset.loaded) {
        container.innerHTML = `<p style="color:var(--muted)">${t('loading')}</p>`;
        try {
          const data = await fetch(`/api/decision/${el.dataset.decisionId}`).then(r => r.json());
          container.innerHTML = renderTranscriptHTML(data);
          el.dataset.loaded = '1';
        } catch {
          container.innerHTML = `<p style="color:var(--muted)">${t('transcriptError')}</p>`;
        }
      }
    });
  });
}

// 매매손익(실현손익) — KIS 체결내역 기준, 기간 설정 가능(app.execution.pnl 백엔드).
// 국내/해외 구분은 새 토글 없이 기존 전역 스코프(_marketScope)를 그대로 재사용.
let _pnlCache = { rows: [], summary: null, start_clamped: false };

function _defaultPnlRange() {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - 30);
  const iso = d => d.toISOString().slice(0, 10);
  return { start: iso(start), end: iso(end) };
}

export async function loadPnl() {
  const startInput = document.getElementById('pnl-start-input');
  const endInput = document.getElementById('pnl-end-input');
  if (!startInput.value || !endInput.value) {
    const defaults = _defaultPnlRange();
    startInput.value ||= defaults.start;
    endInput.value ||= defaults.end;
  }
  const params = new URLSearchParams({ start: startInput.value, end: endInput.value });
  try {
    const data = await fetch(`/api/pnl?${params}`).then(r => r.json());
    _pnlCache = {
      rows: data.rows, summary: data.summary, start_clamped: data.start_clamped,
      fee_unavailable_overseas: data.fee_unavailable_overseas,
    };
    startInput.value = data.start;
  } catch {
    _pnlCache = { rows: [], summary: null, start_clamped: false, fee_unavailable_overseas: false };
  }
  renderPnl();
}

export function renderPnl() {
  const { rows: allRows, summary, start_clamped, fee_unavailable_overseas } = _pnlCache;
  const rows = allRows
    .filter(row => (_marketScope === 'domestic') === isDomesticSymbol(row.ticker))
    // 최신 매도일 순 — 백엔드(compute_realized_pnl)는 오름차순으로 주고, 표시 순서는
    // UI가 정한다(그 함수는 CLI/감사 도구와 테스트가 공유하므로 건드리지 않음).
    // sell_date는 ISO "YYYY-MM-DD" 문자열이라 문자열 비교로 날짜순이 맞다.
    // .filter()가 이미 새 배열을 주므로 공유 캐시(_pnlCache.rows)는 그대로.
    .sort((a, b) => b.sell_date.localeCompare(a.sell_date));
  const body = document.getElementById('pnl-body');
  document.getElementById('pnl-clamp-banner').hidden = !start_clamped;
  // 2026-09-08 fee/tax audit: overseas fee_tax/net_pnl/win_rate are gross
  // (KIS exposes no overseas commission/tax figure at all) — only show this
  // when the user is actually looking at the overseas scope, since the
  // domestic tab's numbers are unaffected.
  document.getElementById('pnl-fee-unavailable-overseas-banner').hidden =
    !(fee_unavailable_overseas && _marketScope === 'overseas');

  const summaryEl = document.getElementById('pnl-summary');
  if (summary) {
    const scopedTotal = _marketScope === 'domestic' ? summary.total_pnl_domestic : summary.total_pnl_overseas;
    const scopedNetTotal = _marketScope === 'domestic' ? summary.total_net_pnl_domestic : summary.total_net_pnl_overseas;
    const fmtTotal = _marketScope === 'domestic' ? fmtKRW : fmtUSD;
    summaryEl.style.display = '';
    document.getElementById('pnl-summary-total').textContent = fmtTotal(scopedTotal, { signed: true });
    const scopedFeeTax = rows.reduce((sum, r) => sum + r.fee_tax, 0);
    document.getElementById('pnl-summary-fee-tax').textContent = fmtTotal(scopedFeeTax);
    document.getElementById('pnl-summary-net-total').textContent = fmtTotal(scopedNetTotal, { signed: true });
    const scopedWins = rows.filter(r => r.net_pnl > 0).length;
    const scopedLosses = rows.filter(r => r.net_pnl < 0).length;
    const scopedTotalCount = scopedWins + scopedLosses;
    document.getElementById('pnl-summary-winrate').textContent =
      scopedTotalCount ? `${(scopedWins / scopedTotalCount * 100).toFixed(1)}%` : '—';
  } else {
    summaryEl.style.display = 'none';
  }

  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="8"><div class="empty">${t('pnlEmpty')}</div></td></tr>`;
    return;
  }
  body.innerHTML = rows.map(row => {
    const fmtAmt = row.currency === 'KRW' ? fmtKRW : fmtUSD;
    // 금액과 %를 열 두 개로 나누면 거의 같은 숫자 열이 다섯 개 연달아 붙어 읽기 어려움 —
    // 보유종목 카드가 이미 쓰는 "금액 (퍼센트)" 한 칸 표기로 합침(정보 손실 없음).
    // 비용 차감 후인 순손익이 실제 결과이므로 그쪽만 색으로 강조하고, 총손익은 눌러 둔다.
    const netCls = row.net_pnl > 0 ? 'pnl-up' : row.net_pnl < 0 ? 'pnl-down' : '';
    // 승/패 행 배경 — .portfolio-table tr.pnl-row-up/-down (base.css)을 그대로 재사용.
    const rowCls = row.net_pnl > 0 ? 'pnl-row-up' : row.net_pnl < 0 ? 'pnl-row-down' : '';
    return `
    <tr class="${rowCls}">
      <td>${escapeHtml(row.name)} <span class="sym">${escapeHtml(displayTicker(row.ticker))}</span></td>
      <td>${row.sell_date}</td>
      <td class="num">${row.shares.toLocaleString('en-US')}</td>
      <td class="num">${fmtPrice(row.buy_avg_price, row.currency)}</td>
      <td class="num">${fmtPrice(row.sell_price, row.currency)}</td>
      <td class="num pnl-gross">${fmtAmt(row.realized_pnl, { signed: true })} (${fmtDelta(row.realized_pnl_pct)})</td>
      <td class="num">${fmtAmt(row.fee_tax)}</td>
      <td class="num ${netCls}">${fmtAmt(row.net_pnl, { signed: true })} (${fmtDelta(row.net_pnl_pct)})</td>
    </tr>`;
  }).join('');
}

// ---------------- 적중률(자체 감사) — /api/audit/accuracy ----------------
// Stage 2: KPI 행 + 등급별 표만 렌더. 확신도/주간추이/F7/토론영향은 index.html의
// #accuracy-stage3 에 마크업만 있고(hidden), 렌더는 Stage 3에서 추가.
//
// 백엔드는 모든 비율을 그냥 분수(0.373, -0.0142)로 준다 — 여기서만 100을 곱한다.
// 승률에 fmtDelta를 쓰면 안 됨: fmtDelta는 양수에 '+'를 붙여서 "+37.30%" 처럼
// 나오는데 승률에 부호를 붙이는 건 오독을 부른다. 그래서 _fmtRate 를 따로 둔다.
// 알파/절대수익은 반대로 부호가 정보이므로 fmtDelta 그대로 쓴다.
const ACCURACY_RATING_ORDER = [
  'Buy', 'Overweight', 'Hold (held)', 'Hold (flat)', 'Hold (held/flat unknown)', 'Underweight', 'Sell',
];
const ACCURACY_RATING_I18N = {
  'Buy': 'accuracyRatingLabelBuy',
  'Overweight': 'accuracyRatingLabelOverweight',
  'Hold (held)': 'accuracyRatingLabelHoldHeld',
  'Hold (flat)': 'accuracyRatingLabelHoldFlat',
  'Hold (held/flat unknown)': 'accuracyRatingLabelHoldUnknown',
  'Underweight': 'accuracyRatingLabelUnderweight',
  'Sell': 'accuracyRatingLabelSell',
};
const ACCURACY_WIN_CONDITION_I18N = {
  'alpha > 0': 'accuracyWinCondUp',
  'alpha < 0': 'accuracyWinCondDown',
  'unscored (needs held/flat)': 'accuracyWinCondUnscored',
};

// 필터는 이 패널만의 상태(사이드바 전역 스코프와 무관) — main.js의 클릭 핸들러가
// 프로퍼티만 바꾸므로 const 로 export 한다(모듈 간 재대입은 불가하므로).
export const _accuracyFilters = { market: 'all', scope: 'live', basis: 'variable' };
let _accuracyCache = null;
let _accuracyLoadFailed = false;

function _fmtRate(v, digits = 1) { return v == null ? '—' : `${(v * 100).toFixed(digits)}%`; }
function _fmtSignedRate(v) { return v == null ? '—' : fmtDelta(v); }

function _accuracyRatingLabel(rating) {
  const key = ACCURACY_RATING_I18N[rating];
  return key ? escapeHtml(t(key)) : escapeHtml(rating);
}
function _accuracyWinConditionLabel(cond) {
  const key = ACCURACY_WIN_CONDITION_I18N[cond];
  return key ? escapeHtml(t(key)) : escapeHtml(cond);
}

export async function loadAccuracy() {
  const params = new URLSearchParams({ scope: _accuracyFilters.scope, basis: _accuracyFilters.basis });
  // market 은 optional — 'all' 이면 아예 보내지 않는다(백엔드 기본값 None = 통합).
  if (_accuracyFilters.market !== 'all') params.set('market', _accuracyFilters.market);
  try {
    const res = await fetch(`/api/audit/accuracy?${params}`);
    if (!res.ok) throw new Error(httpErrorText(res.status));
    _accuracyCache = await res.json();
    _accuracyLoadFailed = false;
  } catch (err) {
    console.error('loadAccuracy failed', err);
    _accuracyCache = null;
    _accuracyLoadFailed = true;
  }
  renderAccuracy();
}

export function renderAccuracy() {
  // 버튼 활성 상태는 항상 _accuracyFilters 를 정본으로 다시 칠한다 — 요청이
  // 실패해도 사용자가 방금 누른 필터가 눌린 채로 남아야 한다.
  [['accuracy-market', 'market'], ['accuracy-scope', 'scope'], ['accuracy-basis', 'basis']].forEach(([id, key]) => {
    document.querySelectorAll(`#${id} .seg-btn`).forEach(btn => {
      btn.classList.toggle('active', btn.dataset.value === _accuracyFilters[key]);
    });
  });

  // Stage 3 중 확신도/토론영향은 이 응답(_accuracyCache)에서 나온다. 아래의
  // early return(데이터 없음)보다 반드시 먼저 호출해야, 로드가 실패해도 Stage 3
  // 패널이 자기 오류 문구를 그린다 — 등급별 표가 오류를 tbody 안에 그리는 것과
  // 같은 방침(패널을 숨기지 않는다).
  _revealAccuracyStage3();
  renderAccuracyConfidence();
  renderAccuracyDebate();

  const body = document.getElementById('accuracy-rating-body');
  const kpis = document.getElementById('accuracy-kpis');
  const modeBadge = document.getElementById('accuracy-account-mode');
  const thresholdHint = document.getElementById('accuracy-threshold-hint');
  const data = _accuracyCache;

  if (!data) {
    kpis.style.display = 'none';
    modeBadge.hidden = true;
    thresholdHint.textContent = '';
    body.innerHTML = `<tr><td colspan="8"><div class="empty">${escapeHtml(t(_accuracyLoadFailed ? 'accuracyLoadError' : 'loading'))}</div></td></tr>`;
    return;
  }

  const isReal = data.filters.account_mode === 'real';
  modeBadge.hidden = false;
  modeBadge.className = isReal ? 'seg-badge real' : 'seg-badge';
  modeBadge.textContent = t(isReal ? 'accuracyAccountModeReal' : 'accuracyAccountModePaper');

  const minN = data.thresholds.min_n_for_ratio;
  thresholdHint.textContent = t('accuracyThresholdHint', minN);

  kpis.style.display = '';
  document.getElementById('accuracy-kpi-n').textContent = data.headline.directional_n.toLocaleString('en-US');
  document.getElementById('accuracy-kpi-winrate').textContent = _fmtRate(data.headline.directional_win_rate);
  document.getElementById('accuracy-kpi-unscored').textContent = data.headline.unscored_hold_n.toLocaleString('en-US');
  document.getElementById('accuracy-kpi-excluded').textContent = data.headline.unscorable_n.toLocaleString('en-US');

  // 백엔드는 ORDER BY rating(알파벳순)으로 준다 — 표시 순서는 UI가 정한다
  // (renderPnl 과 같은 방침). 노출 방향 순서(매수계열→Hold→매도계열)가 읽기 쉽고,
  // 모르는 등급이 새로 생겨도 뒤에 붙어서 사라지지 않는다.
  const rows = [...data.ratings].sort(_accuracyRatingSort);

  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="8"><div class="empty">${escapeHtml(t('accuracyEmpty'))}</div></td></tr>`;
    return;
  }

  body.innerHTML = rows.map(r => {
    // 표본이 적은 승률은 회색 + 툴팁. 숨기지는 않는다(감사 화면에서 숫자를
    // 감추면 "왜 안 보이지"가 되고, CLI(app.audit)는 그대로 보여준다).
    const lowN = r.win_rate != null && r.n < minN;
    const winCell = lowN
      ? `<span class="low-n" title="${escapeHtml(t('accuracyLowNTitle', minN))}">${_fmtRate(r.win_rate)}</span>`
      : _fmtRate(r.win_rate);
    // 평균 알파/절대수익은 색상 강조를 하지 않는다 — 정답 방향이 등급마다
    // 반대라서(매도계열은 알파<0이 정답) 부호에 색을 입히면 매도계열을 정확히
    // 거꾸로 읽힌다. 2026-08-13에 백엔드에서 고친 것과 같은 종류의 버그.
    return `
    <tr>
      <td>${_accuracyRatingLabel(r.rating)}</td>
      <td class="num">${r.n.toLocaleString('en-US')}</td>
      <td class="num">${winCell}</td>
      <td class="num">${_fmtSignedRate(r.mean_alpha)}</td>
      <td class="num">${r.stdev_alpha == null ? '—' : _fmtRate(r.stdev_alpha, 2)}</td>
      <td class="num">${_fmtSignedRate(r.mean_raw_return)}</td>
      <td>${_accuracyWinConditionLabel(r.win_condition)}</td>
      <td class="num">${r.unscorable_n ? r.unscorable_n.toLocaleString('en-US') : '—'}</td>
    </tr>`;
  }).join('');
}

// ---------------- 적중률 Stage 3 — 확신도/주간추이/F7/토론영향 ----------------
// 데이터 출처가 두 엔드포인트로 갈린다:
//   확신도·토론영향 -> /api/audit/accuracy (_accuracyCache, 시장 필터 적용됨)
//   주간추이·F7    -> /api/audit/weekly   (_accuracyWeeklyCache, 시장 필터 없음)
// 후자를 별도 fetch로 두는 이유: /api/audit/weekly 는 market 파라미터를 아예 받지
// 않는다(라우터 시그니처가 week/scope/basis). 한 함수로 합치면 시장 버튼을 누를
// 때마다 완전히 같은 응답을 받으려고 무거운 쿼리(주간 승률 3회 + 확신도 + F7
// 크로스탭)를 다시 돌리게 되고, 한쪽 503이 다른 쪽 표까지 지운다.
let _accuracyWeeklyCache = null;
let _accuracyWeeklyLoadFailed = false;

// 등급 표시 순서 비교자 — Stage 2 등급별 표와 Stage 3 주간추이 표가 같은 순서를
// 써야 하므로 인라인 정렬을 함수로 뽑았다. 모르는 등급은 뒤로 밀리고 사라지지 않는다.
function _accuracyRatingSort(a, b) {
  const ia = ACCURACY_RATING_ORDER.indexOf(a.rating);
  const ib = ACCURACY_RATING_ORDER.indexOf(b.rating);
  return (ia === -1 ? ACCURACY_RATING_ORDER.length : ia) - (ib === -1 ? ACCURACY_RATING_ORDER.length : ib);
}

// index.html 에서 hidden 으로 시작하는 Stage 3 컨테이너를 연다. 데이터 성공
// 여부와 무관하게 연다(Decision 4) — 감사 화면에서 섹션이 통째로 사라지면
// "왜 없지"가 되고, 각 표는 자기 tbody 안에 로딩/오류 문구를 그린다.
function _revealAccuracyStage3() {
  document.getElementById('accuracy-stage3').hidden = false;
}

// 백엔드 라벨(app/audit/stats.py _CONFIDENCE_BUCKETS)은 구간 수식 표기라 번역하지
// 않는다 — 유일한 영어 낱말인 '(skipped)' 만 배지로 바꿔 붙인다.
function _accuracyConfidenceBandLabel(label) {
  if (label === '<0.50 (skipped)') {
    return `${escapeHtml('<0.50')} <span class="accuracy-skipped-tag">${escapeHtml(t('accuracySkippedTag'))}</span>`;
  }
  return escapeHtml(label);
}

function renderAccuracyConfidence() {
  const body = document.getElementById('accuracy-confidence-body');
  const note = document.getElementById('accuracy-confidence-footnote');
  const data = _accuracyCache;

  if (!data) {
    body.innerHTML = `<tr><td colspan="6"><div class="empty">${escapeHtml(t(_accuracyLoadFailed ? 'accuracyLoadError' : 'loading'))}</div></td></tr>`;
    note.textContent = '';
    return;
  }

  const conf = data.confidence;
  const minN = data.thresholds.min_n_for_ratio;

  // 각주는 표가 비어 있어도 항상 그린다 — "확신도 미기록이 몇 건인지"가 이 화면에서
  // 가장 자주 문제가 되는 수치다.
  const corrText = conf.signed_alpha_corr == null
    ? t('accuracyCorrNA')
    : `${conf.signed_alpha_corr >= 0 ? '+' : ''}${conf.signed_alpha_corr.toFixed(3)}`;
  note.textContent = t('accuracyConfidenceFootnote', conf.n_without_confidence, corrText);

  if (!conf.n_with_confidence) {
    body.innerHTML = `<tr><td colspan="6"><div class="empty">${escapeHtml(t('accuracyConfidenceEmpty'))}</div></td></tr>`;
    return;
  }

  body.innerHTML = conf.buckets.map(b => {
    // size_multiplier === 0 인 구간은 sizing.py 가 매매를 아예 건너뛰는 구간
    // (<0.50) — 실제로 집행된 적이 없으므로 승률은 반사실이다. 행 전체를
    // 흐리게 표시하고 각주에서 이유를 설명한다.
    const skipped = b.size_multiplier === 0;
    const lowN = b.win_rate != null && b.n < minN;
    const winCell = lowN
      ? `<span class="low-n" title="${escapeHtml(t('accuracyLowNTitle', minN))}">${_fmtRate(b.win_rate)}</span>`
      : _fmtRate(b.win_rate);
    return `
    <tr${skipped ? ' class="accuracy-counterfactual"' : ''}>
      <td>${_accuracyConfidenceBandLabel(b.label)}</td>
      <td class="num">${b.size_multiplier.toFixed(1)}x</td>
      <td class="num">${b.n.toLocaleString('en-US')}</td>
      <td class="num">${winCell}</td>
      <td class="num">${_fmtSignedRate(b.mean_signed_alpha)}</td>
      <td class="num">${b.unscorable_n ? b.unscorable_n.toLocaleString('en-US') : '—'}</td>
    </tr>`;
  }).join('');
}

function renderAccuracyDebate() {
  const nEl = document.getElementById('accuracy-debate-n');
  const changedEl = document.getElementById('accuracy-debate-changed');
  const breakdownEl = document.getElementById('accuracy-debate-breakdown');
  const note = document.getElementById('accuracy-debate-footnote');
  const data = _accuracyCache;

  if (!data) {
    nEl.textContent = '—';
    changedEl.textContent = '—';
    breakdownEl.innerHTML = '';
    note.textContent = '';
    return;
  }

  const d = data.debate;
  const minN = data.thresholds.min_n_for_ratio;
  nEl.textContent = d.n.toLocaleString('en-US');
  // stats.py: changed_pct=(changed / n) if n else 0.0 — n=0 이면 0.0 이 내려온다.
  // 그걸 "0.0%"로 그리면 "토론이 한 번도 결론을 바꾼 적 없다"로 오독되므로,
  // 표본이 임계 미만이면 건수만 보이고 비율은 감춘다(다른 표와 같은 규칙).
  if (d.n < minN) {
    changedEl.innerHTML = `${d.changed.toLocaleString('en-US')} <span class="low-n" title="${escapeHtml(t('accuracyLowNTitle', minN))}">(—)</span>`;
  } else {
    changedEl.textContent = `${d.changed.toLocaleString('en-US')} (${_fmtRate(d.changed_pct)})`;
  }

  // breakdown 키는 "{trader_action}->{final_direction}" (stats.py). 번역 대상이
  // 아니라 원문 그대로, 건수 많은 순으로.
  const entries = Object.entries(d.breakdown).sort((a, b) => b[1] - a[1]);
  breakdownEl.innerHTML = entries.map(([k, v]) =>
    `<span class="accuracy-breakdown-item">${escapeHtml(k)} <b>${v.toLocaleString('en-US')}</b></span>`
  ).join('');
  note.textContent = t('accuracyDebateFootnote');
}

// weekly.py 의 _fmt_stats_cell/_fmt_ratio 와 같은 규칙: 그 창(window)에 해당 등급이
// 아예 없으면 N/A(0건과 구분됨 — RatingRow 가 None 을 그 뜻으로 쓴다), 있으면 N 을
// 항상 먼저 보여주고 비율은 표본이 임계 이상일 때만. n < minN 이면 n 은 보이고 % 만 감춘다.
function _accuracyWeeklyCell(s, minN) {
  if (s == null) return `<span class="low-n">${escapeHtml(t('accuracyWeeklyCellNA'))}</span>`;
  const n = s.n.toLocaleString('en-US');
  if (s.win_rate == null || s.n < minN) {
    return `${n} · <span class="low-n" title="${escapeHtml(t('accuracyLowNTitle', minN))}">—</span>`;
  }
  return `${n} · ${_fmtRate(s.win_rate)}`;
}

export async function loadAccuracyWeekly() {
  // market 은 보내지 않는다 — /api/audit/weekly 가 받지 않는다(라우터 시그니처
  // week/scope/basis). week 도 보내지 않는다(기본값 = KST 이번 주).
  const params = new URLSearchParams({ scope: _accuracyFilters.scope, basis: _accuracyFilters.basis });
  try {
    const res = await fetch(`/api/audit/weekly?${params}`);
    if (!res.ok) throw new Error(httpErrorText(res.status));
    _accuracyWeeklyCache = await res.json();
    _accuracyWeeklyLoadFailed = false;
  } catch (err) {
    console.error('loadAccuracyWeekly failed', err);
    _accuracyWeeklyCache = null;
    _accuracyWeeklyLoadFailed = true;
  }
  renderAccuracyWeekly();
}

export function renderAccuracyWeekly() {
  _revealAccuracyStage3();

  const meta = document.getElementById('accuracy-weekly-meta');
  const body = document.getElementById('accuracy-weekly-body');
  const weeklyNote = document.getElementById('accuracy-weekly-footnote');
  const f7Body = document.getElementById('accuracy-f7-body');
  const f7Note = document.getElementById('accuracy-f7-footnote');
  const data = _accuracyWeeklyCache;

  if (!data) {
    const msg = escapeHtml(t(_accuracyWeeklyLoadFailed ? 'accuracyLoadError' : 'loading'));
    meta.textContent = '';
    body.innerHTML = `<tr><td colspan="4"><div class="empty">${msg}</div></td></tr>`;
    f7Body.innerHTML = `<tr><td colspan="4"><div class="empty">${msg}</div></td></tr>`;
    weeklyNote.textContent = '';
    f7Note.textContent = '';
    return;
  }

  const minN = data.thresholds.min_n_for_ratio;
  const minF7 = data.thresholds.min_n_for_f7_verdict;

  meta.textContent = t('accuracyWeeklyWindow',
    data.week_label, data.week_start, data.week_end,
    data.totals.this_week_n, data.totals.this_week_unscorable_n,
    data.totals.last_8_weeks_n, data.totals.cumulative_n);

  // 시장 필터는 이 두 표에 적용되지 않는다. 필터가 '전체'가 아닐 때만 경고를
  // 덧붙여, 국내만 보고 있는 사용자가 이 표를 국내 수치로 오독하지 않게 한다.
  const marketNote = _accuracyFilters.market === 'all' ? '' : ` ${t('accuracyWeeklyMarketIgnored')}`;
  weeklyNote.textContent = t('accuracyWeeklyFootnote', minN) + marketNote;
  f7Note.textContent = t('accuracyF7Footnote', minF7) + marketNote;

  const rows = [...data.ratings].sort(_accuracyRatingSort);
  body.innerHTML = rows.length
    ? rows.map(r => `
    <tr>
      <td>${_accuracyRatingLabel(r.rating)}</td>
      <td class="num">${_accuracyWeeklyCell(r.this_week, minN)}</td>
      <td class="num">${_accuracyWeeklyCell(r.last_8_weeks, minN)}</td>
      <td class="num">${_accuracyWeeklyCell(r.cumulative, minN)}</td>
    </tr>`).join('')
    : `<tr><td colspan="4"><div class="empty">${escapeHtml(t('accuracyWeeklyEmpty'))}</div></td></tr>`;

  if (!data.f7.total_overridden) {
    f7Body.innerHTML = `<tr><td colspan="4"><div class="empty">${escapeHtml(t('accuracyF7Empty'))}</div></td></tr>`;
    return;
  }

  // F7 row_label('Hold (held)'/'Hold (flat)'/'Hold (held/flat unknown)')은
  // ACCURACY_RATING_I18N 의 키와 정확히 일치하므로 _accuracyRatingLabel 을 그대로 쓴다.
  f7Body.innerHTML = data.f7.cells.map(c => {
    // 임계 미만/판정불가 셀의 win_rate 는 백엔드가 이미 null 로 내려준다
    // (weekly.py: n >= _MIN_N_FOR_F7_VERDICT 이고 'Hold (held/flat unknown)' 이
    // 아닐 때만 값이 있음). 프론트가 다시 계산하면 두 기준이 어긋날 수 있어 그대로 쓴다.
    const unknownRow = c.row_label === 'Hold (held/flat unknown)';
    const verdict = c.win_rate == null
      ? `<span class="low-n" title="${escapeHtml(t(unknownRow ? 'accuracyF7NoVerdictUnknownTitle' : 'accuracyF7NoVerdictTitle', minF7))}">—</span>`
      : _fmtRate(c.win_rate);
    return `
    <tr>
      <td>${_accuracyRatingLabel(c.row_label)}</td>
      <td class="num">${c.n.toLocaleString('en-US')}</td>
      <td class="num">${c.wins.toLocaleString('en-US')}</td>
      <td class="num">${verdict}</td>
    </tr>`;
  }).join('');
}

export function renderTranscriptHTML(data) {
  const transcript = data.transcript;
  if (!transcript) return `<p style="color:var(--muted)">${t('noTranscript')}</p>`;
  const parts = [];

  // 2026-08-25: a run where 1-3 of the 4 analysts failed is otherwise
  // visually identical to a full run — surface it up front so a reader
  // doesn't mistake a degraded analysis for the desk's normal standing.
  if (transcript.degraded_analysts?.length) {
    parts.push(`<div class="turn" style="color:var(--down)">${escapeHtml(t('degradedAnalystsWarning', transcript.degraded_analysts.length))}: ${transcript.degraded_analysts.map(escapeHtml).join(', ')}</div>`);
  }

  if (transcript.investment_thesis) {
    parts.push(`<h4>${t('transcriptThesis')}</h4>`);
    parts.push(`<div class="turn">${renderMarkdown(transcript.investment_thesis)}</div>`);
  }
  parts.push(`<h4>${t('transcriptAnalysts')}</h4>`);
  for (const [name, report] of Object.entries(transcript.reports || {})) {
    parts.push(`<div class="turn"><span class="speaker">${escapeHtml(name)}</span>${renderMarkdown(report)}</div>`);
  }
  if (transcript.research_debate?.length) {
    parts.push(`<h4>${t('transcriptDebate')}</h4>`);
    for (const turn of transcript.research_debate) {
      parts.push(`<div class="turn"><span class="speaker">${escapeHtml(turn.speaker)} — ${t('roundSuffix', turn.round)}</span>${renderMarkdown(turn.content)}</div>`);
    }
  }
  if (transcript.research_plan) {
    parts.push(`<h4>${t('transcriptResearchManager')}</h4>`);
    parts.push(`<div class="turn">${renderMarkdown(transcript.research_plan)}</div>`);
  }
  if (transcript.trader_proposal) {
    parts.push(`<h4>${t('transcriptTrader')}</h4>`);
    parts.push(`<div class="turn">${renderMarkdown(transcript.trader_proposal)}</div>`);
  }
  if (transcript.risk_debate?.length) {
    parts.push(`<h4>${t('transcriptRiskDebate')}</h4>`);
    for (const turn of transcript.risk_debate) {
      parts.push(`<div class="turn"><span class="speaker">${escapeHtml(turn.speaker)} — ${t('roundSuffix', turn.round)}</span>${renderMarkdown(turn.content)}</div>`);
    }
  }

  // 2026-08-25: context the Portfolio Manager actually read before deciding
  // — recorded in full_transcript from that date on (older rows simply
  // won't have these keys, and each one is independently optional since a
  // backtest leaves the two live-only strings as ""). Listed as plain
  // labeled text, not markdown — these are short machine-generated
  // narratives, not agent prose.
  const pmContextFields = [
    ['past_context', 'transcriptPastContext'],
    ['sibling_context', 'transcriptSiblingContext'],
    ['position_context', 'transcriptPositionContext'],
    ['volatility_context', 'transcriptVolatilityContext'],
    ['risk_cap_context', 'transcriptRiskCapContext'],
    ['portfolio_risk_context', 'transcriptPortfolioRiskContext'],
    ['execution_status_context', 'transcriptExecutionStatusContext'],
  ].filter(([key]) => transcript[key]);
  if (pmContextFields.length) {
    parts.push(`<h4>${t('transcriptPMContext')}</h4>`);
    for (const [key, labelKey] of pmContextFields) {
      parts.push(`<div class="turn"><span class="speaker">${escapeHtml(t(labelKey))}</span>${escapeHtml(transcript[key])}</div>`);
    }
  }
  return parts.join('');
}

// Shared by both the chat's inline toggle and the decision-history rows —
// fetches lazily (once per element) and just shows/hides on repeat clicks.
export function wireTranscriptToggle(toggleEl, containerEl, decisionId, labels) {
  let loaded = false;
  toggleEl.addEventListener('click', async () => {
    const isOpen = containerEl.style.display !== 'none';
    if (isOpen) {
      containerEl.style.display = 'none';
      toggleEl.textContent = labels.closed;
      return;
    }
    containerEl.style.display = 'block';
    toggleEl.textContent = labels.open;
    if (!loaded) {
      containerEl.innerHTML = `<p style="color:var(--muted)">${t('loading')}</p>`;
      try {
        const data = await fetch(`/api/decision/${decisionId}`).then(r => r.json());
        containerEl.innerHTML = renderTranscriptHTML(data);
        loaded = true;
      } catch {
        containerEl.innerHTML = `<p style="color:var(--muted)">${t('transcriptError')}</p>`;
      }
    }
  });
}

export const orderState = { side: 'buy', type: 'market' };

// refreshAvail()'s latest response, normalized to one shape across buy/sell
// so orderMaxShares() (the single source of truth for quick-buttons + the
// over-max warning) doesn't need to know which endpoint produced it.
// { sym, side, ok, currency, cash, maxQty, maxAmount, sellableQty, reason }
export let _availCache = null;
let _availSeq = 0;  // guards against a slow response from a since-changed ticker overwriting the cache

// KIS's own max_qty already accounts for fees — this buffer is only for the
// cash-based fallback (limit orders, overseas where max_qty is always null),
// which would otherwise compute a share count the broker rejects for
// insufficient cash. Never apply it to a KIS-supplied max_qty (double-counts).
const _BUY_CASH_BUFFER = 0.9975;

// Same KRX-detection rule used across the order ticket (currency display,
// approval-threshold check, order confirm) — one place instead of three
// copies of the same regex drifting apart.
export function isKrxTicker(sym) { return /\.KS$/i.test(sym) || /^\d{6}$/.test(sym); }
export function orderCurrency(sym) { return isKrxTicker(sym) ? 'KRW' : 'USD'; }

// The single basis for quick-buttons and the "exceeds available" warning.
// Returns null for "unknown / still loading" (buttons disabled, no claim
// either way) vs. 0 for "confirmed zero" (buttons disabled, different
// tooltip) — callers must not collapse these two into the same state.
function orderMaxShares() {
  const sym = document.getElementById('order-ticker').value;
  if (!_availCache || _availCache.sym !== sym || _availCache.side !== orderState.side || !_availCache.ok) return null;
  if (orderState.side === 'sell') return _availCache.sellableQty;
  // buy
  if (_availCache.maxQty != null) {
    // Domestic: maxQty is KIS's own market-order-basis figure. In limit
    // mode with a valid typed price, a lower limit price can legitimately
    // buy more than that market-basis max — recompute from cash instead of
    // capping at maxQty. An empty/invalid limit price falls back to maxQty.
    if (orderState.type === 'limit') {
      const lp = parseFloat(document.getElementById('order-limit-price').value);
      if (lp > 0 && _availCache.cash != null) {
        return Math.max(0, Math.floor(_availCache.cash * _BUY_CASH_BUFFER / lp));
      }
    }
    return _availCache.maxQty;
  }
  // Overseas: no per-symbol max-qty TR, so derive it from cash / reference price.
  if (_availCache.cash == null) return null;
  const price = orderState.type === 'limit' ? parseFloat(document.getElementById('order-limit-price').value) : refPrice(sym);
  if (!(price > 0)) return null;
  return Math.max(0, Math.floor(_availCache.cash * _BUY_CASH_BUFFER / price));
}

function syncOrderQtyQuick() {
  const base = orderMaxShares();
  document.querySelectorAll('#order-qty-quick .oqty').forEach(btn => {
    const disabled = base == null || base < 1;
    btn.disabled = disabled;
    if (!disabled) { btn.removeAttribute('title'); return; }
    if (base == null) btn.title = t('orderQtyQuickUnavailable');
    else btn.title = orderState.side === 'sell' ? t('orderQtyQuickNoHolding') : t('orderQtyQuickNoCash');
  });
}

// Mirrors submitOrder()'s own confirm-dialog amount calculation exactly
// (shares * (limit ? limit_price : refPrice(sym))) — the input-stage amount
// and the confirm-dialog amount must never disagree.
export function updateOrderEstimate() {
  const container = document.getElementById('order-estimate');
  const valueEl = document.getElementById('order-estimate-value');
  const sym = document.getElementById('order-ticker').value;
  container.classList.remove('over');
  valueEl.querySelector('.oest-warn')?.remove();
  if (!sym) { valueEl.textContent = '—'; return; }
  const shares = parseInt(document.getElementById('order-shares').value, 10);
  if (!shares || shares <= 0) { valueEl.textContent = t('orderEstimateEmpty'); return; }
  const price = orderState.type === 'limit' ? parseFloat(document.getElementById('order-limit-price').value) : refPrice(sym);
  if (!(price > 0)) { valueEl.textContent = t('orderEstimateUnknown'); return; }
  valueEl.textContent = fmtPrice(shares * price, orderCurrency(sym));
  const base = orderMaxShares();
  if (base != null && shares > base) {
    container.classList.add('over');
    const warn = document.createElement('span');
    warn.className = 'oest-warn';
    warn.textContent = ' ' + t('orderEstimateOver', base);
    valueEl.appendChild(warn);
  }
}

export function refPrice(sym) {
  const s = state[sym];
  if (!s) return null;
  return s.history && s.history.length ? s.history[s.history.length - 1].price : s.previousClose;
}

export function showOrderResult(msg, ok) {
  const el = document.getElementById('order-result');
  el.hidden = false;
  el.textContent = msg;
  el.className = 'order-result ' + (ok ? 'ok' : 'bad');
}

export async function loadExecMode() {
  const badge = document.getElementById('order-mode-badge');
  const m = await fetch('/api/exec-mode').then(r => r.json()).catch(() => null);
  if (!m || !m.broker_configured) { badge.textContent = t('orderModeOff'); badge.className = 'order-mode-badge off'; }
  else if (!m.enabled) { badge.textContent = t('orderModeDisabled'); badge.className = 'order-mode-badge off'; }
  else if (m.dry_run) { badge.textContent = t('orderModeDry'); badge.className = 'order-mode-badge dry'; }
  else { badge.textContent = t('orderModeLive'); badge.className = 'order-mode-badge live'; }
  loadExecModePill(m?.execution_mode);
}

// Always-visible header pill — MANUAL/SEMI_AUTO/FULL_AUTO. Changeable now
// from the 설정 tab's segmented toggle (#rt-execution-mode), but this pill
// stays read-only: the point is to always know which mode is active
// without opening a tab, not to change it from here.
// Read by notifyNewFullAutoDecisions to decide whether a new decision
// should surface as an FYI toast — set once per loadExecMode() call, which
// setRuntimeSetting() re-triggers after a successful change so this pill
// never goes stale relative to the settings tab.
export let _currentExecutionMode = null;

export function loadExecModePill(executionMode) {
  // _currentExecutionMode is read elsewhere (e.g. notifyNewFullAutoDecisions'
  // full_auto gating) regardless of whether there's still a pill anywhere —
  // the header pill is gone entirely (2026-08-19); the real control is now
  // 설정 tab's #rt-execution-mode (interactive) plus the sidebar's read-only
  // #sidebar-exec-mode-text, both synced via _renderRuntimeSegments/
  // loadRuntimeSettings instead of this function.
  _currentExecutionMode = executionMode || null;
}

// SEMI_AUTO/FULL_AUTO visibility — polled independently of loadExecMode()
// (that one only runs once at page load; this needs to actually change
// while the loop runs, so it gets its own short-interval poll below).
export async function refreshAvail() {
  const seq = ++_availSeq;
  const sym = document.getElementById('order-ticker').value;
  const side = orderState.side;
  const el = document.getElementById('order-avail');
  if (!sym) {
    _availCache = null;
    el.textContent = '—';
    syncOrderQtyQuick();
    updateOrderEstimate();
    return;
  }
  try {
    if (side === 'buy') {
      const b = await fetch(`/api/buyable/${encodeURIComponent(sym)}`).then(r => r.json());
      if (seq !== _availSeq) return;  // a since-changed ticker/side already started a newer request
      if (!b.available) {
        _availCache = { sym, side, ok: false, reason: b.reason };
        el.textContent = t('orderAvailBuyUnsupported');
      } else {
        _availCache = { sym, side, ok: true, currency: b.currency, cash: b.cash, maxQty: b.max_qty, maxAmount: b.max_amount, sellableQty: null };
        if (b.max_qty != null) {
          el.textContent = t('orderAvailBuy', fmtPrice(b.cash, b.currency), b.max_qty);
        } else {
          const price = orderState.type === 'limit' ? parseFloat(document.getElementById('order-limit-price').value) : refPrice(sym);
          if (price > 0) {
            const approxQty = Math.max(0, Math.floor(b.cash * _BUY_CASH_BUFFER / price));
            el.textContent = t('orderAvailBuyApprox', fmtPrice(b.cash, b.currency), approxQty);
          } else {
            el.textContent = t('orderAvailBuyCashOnly', fmtPrice(b.cash, b.currency));
          }
        }
      }
    } else {
      const s = await fetch(`/api/sellable/${encodeURIComponent(sym)}`).then(r => r.json());
      if (seq !== _availSeq) return;
      _availCache = { sym, side, ok: true, currency: null, cash: null, maxQty: null, maxAmount: null, sellableQty: s.sellable_qty };
      el.textContent = t('orderAvailSell', s.sellable_qty);
    }
  } catch {
    if (seq !== _availSeq) return;
    _availCache = { sym, side, ok: false, reason: null };
    el.textContent = t('orderAvailError');
  }
  syncOrderQtyQuick();
  updateOrderEstimate();
}

// /api/order's known response statuses (app.execution.order) — the raw enum
// value used to leak straight into the UI as "[submitted] ...". Anything
// outside this set still shows (via the raw-string fallback below), so an
// unexpected/new status is never silently swallowed, just unlocalized.
const _ORDER_STATUS_LABEL_KEYS = { submitted: 'orderStatusSubmitted', 'dry-run': 'orderStatusDryRun', rejected: 'orderStatusRejected', error: 'orderStatusError' };
function _orderStatusLabel(status) {
  const key = _ORDER_STATUS_LABEL_KEYS[status];
  return key ? t(key) : status;
}

export async function submitOrder() {
  const sym = document.getElementById('order-ticker').value;
  const shares = parseInt(document.getElementById('order-shares').value, 10);
  if (!sym || !shares || shares <= 0) { showOrderResult(t('orderQtyInvalid'), false); return; }
  const body = { ticker: sym, side: orderState.side, shares, order_type: orderState.type };
  if (orderState.type === 'limit') {
    const lp = parseFloat(document.getElementById('order-limit-price').value);
    if (!lp || lp <= 0) { showOrderResult(t('orderLimitPriceRequired'), false); return; }
    body.limit_price = lp;
  } else {
    body.reference_price = refPrice(sym);
  }
  const sideLabel = orderState.side === 'buy' ? t('buy') : t('sell');
  const typeLabel = orderState.type === 'limit'
    ? t('orderTypeLimitWithPrice', document.getElementById('order-limit-price').value)
    : t('orderTypeMarket');
  // Shared with the order-ticket's own currency/quantity display (isKrxTicker/
  // orderCurrency) — reused, not reinvented, so the two never drift apart.
  const isKrx = isKrxTicker(sym);
  const currency = orderCurrency(sym);
  // Estimated order amount — market orders have no submitted price yet, so
  // this uses the same last-tick/previous-close reference the order ticket
  // itself displays (refPrice), not a fabricated number. Said explicitly as
  // "unavailable" rather than showing 0/blank when there's no reference yet
  // (e.g. a ticker with no ticks at all) — the whole point of this line is
  // that the user should never submit without seeing what it costs.
  const orderPrice = orderState.type === 'limit' ? body.limit_price : refPrice(sym);
  const amountLine = orderPrice != null ? t('orderConfirmAmount', fmtPrice(shares * orderPrice, currency)) : t('orderConfirmAmountUnknown');
  const symName = state[sym]?.name;
  const confirmBody = [
    symName ? `${symName} ${displayTicker(sym)}` : displayTicker(sym),
    `${sideLabel} · ${typeLabel} · ${shares}${t('sharesUnit')}`,
    amountLine,
  ].join('\n');
  const proceed = await confirmDialog({ title: t('orderConfirmTitle'), body: confirmBody });
  if (!proceed) return;
  // Approval-threshold rule (설정 탭): if enabled and this order's value is at or
  // above the threshold, require an explicit second confirmation and mark approved.
  {
    const apprPrice = orderState.type === 'limit' ? body.limit_price : body.reference_price;
    if (_riskRules?.approval_enabled && apprPrice) {
      const thr = isKrx ? _riskRules.approval_threshold_krw : _riskRules.approval_threshold_usd;
      if (shares * apprPrice >= thr) {
        const proceedThreshold = await confirmDialog({
          title: t('orderApprovalThresholdTitle'),
          body: t('orderApprovalThresholdConfirm', fmtPrice(shares * apprPrice, currency), fmtPrice(thr, currency)),
        });
        if (!proceedThreshold) return;
        body.approved = true;
      }
    }
  }
  const btn = document.getElementById('order-submit');
  btn.disabled = true;
  try {
    const res = await fetch('/api/order', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then(r => r.json());
    const ledgerWarn = res.ledger_recorded === false ? ` · ${t('orderLedgerNotRecorded')}` : '';
    const okish = (res.status === 'submitted' || res.status === 'dry-run') && res.ledger_recorded !== false;
    showOrderResult(`${_orderStatusLabel(res.status)} · ${res.message}${ledgerWarn}`, okish);
    refreshAvail(); loadOpenOrders(); pollTrades();
  } catch (e) {
    console.error('order request failed', e);
    showOrderResult(t('orderRequestFailed'), false);
  } finally {
    btn.disabled = false;
  }
}

// Unfiltered open orders as last polled, so a 국내/해외 scope switch can
// re-render without waiting for the next poll (same pattern as
// _portfolioTableCache / _pendingOrdersCache in signals.js).
export let _openOrdersCache = [];

export function renderOpenOrders(allRows) {
  _openOrdersCache = allRows;
  const el = document.getElementById('open-orders');
  // 미체결 주문도 헤더의 국내/해외 스코프를 따름 — 원화 화면에 USD 주문이 섞여 보이던 문제.
  const rows = allRows.filter(o => (_marketScope === 'domestic') === isDomesticSymbol(o.ticker));
  if (!rows.length) { el.innerHTML = `<div class="empty">${t('openOrdersEmpty')}</div>`; return; }
  el.innerHTML = '';
  rows.forEach(o => {
    const div = document.createElement('div');
    div.className = 'open-order-row';
    const sideLabel = o.side === 'buy' ? t('buy') : t('sell');
    const label = document.createElement('span');
    label.textContent = t('openOrderRowLabel', o.name || o.ticker, sideLabel, o.remaining_qty, o.order_qty);
    div.appendChild(label);
    if (o.cancelable) {
      const rev = document.createElement('button');
      rev.textContent = t('reviseBtn');
      rev.addEventListener('click', () => reviseOrder(o.order_no, o.price, rev));
      div.appendChild(rev);
    }
    const btn = document.createElement('button');
    btn.textContent = t('cancelBtn');
    btn.disabled = !o.cancelable;
    if (!o.cancelable) btn.title = t('orderNotCancelableTitle');
    btn.addEventListener('click', () => cancelOrder(o.order_no, btn));
    div.appendChild(btn);
    el.appendChild(div);
  });
}

export async function loadOpenOrders() {
  let rows = [];
  try { rows = await fetch('/api/orders/open').then(r => r.json()); } catch { return; }
  renderOpenOrders(rows);
}

export async function reviseOrder(orderNo, curPrice, btn) {
  const input = await promptDialog({
    title: t('reviseBtn'),
    body: t('revisePricePrompt'),
    defaultValue: curPrice ? String(Math.round(curPrice)) : '',
  });
  if (input === null) return;
  const price = parseFloat(input);
  if (!price || price <= 0) { await alertDialog({ title: t('reviseBtn'), body: t('revisePriceInvalid') }); return; }
  btn.disabled = true;
  try {
    const r = await fetch('/api/order/revise', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ order_no: orderNo, new_price: price }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      const detail = e.detail && (e.detail.detail || e.detail);
      await alertDialog({ title: t('reviseBtn'), body: t('reviseFailed', detail || httpErrorText(r.status)) });
      btn.disabled = false; return;
    }
    loadOpenOrders(); pollTrades();
  } catch (e) {
    console.error('revise order request failed', e);
    await alertDialog({ title: t('reviseBtn'), body: t('reviseRequestFailed') });
    btn.disabled = false;
  }
}

export async function cancelOrder(orderNo, btn) {
  const proceed = await confirmDialog({ title: t('cancelBtn'), body: t('cancelConfirm') });
  if (!proceed) return;
  btn.disabled = true;
  try {
    const r = await fetch('/api/order/cancel', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ order_no: orderNo }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      const detail = e.detail && (e.detail.detail || e.detail);
      await alertDialog({ title: t('cancelBtn'), body: t('cancelFailed', detail || httpErrorText(r.status)) });
      btn.disabled = false;
      return;
    }
    loadOpenOrders(); pollTrades();
  } catch (e) {
    console.error('cancel order request failed', e);
    await alertDialog({ title: t('cancelBtn'), body: t('cancelRequestFailed') });
    btn.disabled = false;
  }
}

// Submit-button label/color follows the side toggle — buy is red (외화매수,
// matches the up/red color used for buy across the dashboard), sell is blue
// (현금매도, matches down/blue) — same color mapping as .oside above.
export function updateOrderSubmitButton() {
  const btn = document.getElementById('order-submit');
  const isBuy = orderState.side === 'buy';
  btn.textContent = t(isBuy ? 'orderSubmitBuy' : 'orderSubmitSell');
  btn.classList.toggle('side-buy', isBuy);
  btn.classList.toggle('side-sell', !isBuy);
}

// 주문 종목 드롭다운은 헤더의 국내/해외 스코프를 따른다 — 화면 전체가 원화 맥락인데
// 드롭다운만 USD 종목을 고를 수 있으면 오주문 위험(2026-08-19). 서버가 티커 형식으로
// 분기해 주문 자체는 정확하지만, 그건 사용자 혼동과는 별개 문제.
export function syncOrderTickerOptions() {
  const sel = document.getElementById('order-ticker');
  if (!sel) return;
  const prev = sel.value;
  const syms = Object.keys(state).filter(sym => (_marketScope === 'domestic') === isDomesticSymbol(sym));
  sel.innerHTML = syms
    // Label text only — displayTicker() is display-only, the option's own
    // `value` must stay the real ticker (it's what submitOrder reads back).
    .map(sym => `<option value="${sym}">${escapeHtml(state[sym].name || displayTicker(sym))} (${escapeHtml(displayTicker(sym))})</option>`)
    .join('');
  // Keep the current pick when it's still in scope; otherwise fall back to the
  // first in-scope option (browsers already default to it, set explicitly for clarity).
  const next = syms.includes(prev) ? prev : (syms[0] ?? '');
  sel.value = next;
  // A programmatic .value assignment does NOT fire 'change', so the dependent
  // order-ticket UI (available cash / quantity quick-buttons / estimate) would
  // keep showing the previous ticker's numbers. refreshAvail() is exactly what
  // the 'change' handler and openDetail() call, and it already chains
  // syncOrderQtyQuick()/updateOrderEstimate() itself.
  if (next !== prev) refreshAvail();
}

export function initOrderTicket() {
  const sel = document.getElementById('order-ticker');
  syncOrderTickerOptions();
  document.querySelectorAll('#order-side-toggle .oside').forEach(b => {
    b.addEventListener('click', () => {
      orderState.side = b.dataset.side;
      document.querySelectorAll('#order-side-toggle .oside').forEach(x => x.classList.toggle('active', x === b));
      updateOrderSubmitButton();
      refreshAvail();
    });
  });
  document.querySelectorAll('#order-type-toggle .otype').forEach(b => {
    b.addEventListener('click', () => {
      orderState.type = b.dataset.type;
      document.querySelectorAll('#order-type-toggle .otype').forEach(x => x.classList.toggle('active', x === b));
      document.getElementById('order-price-field').style.display = orderState.type === 'limit' ? 'flex' : 'none';
      // Prefill the limit price with the current market price at the
      // moment of switching to 지정가, instead of leaving it blank — the
      // user picked this as a starting point to adjust from, not a fixed
      // reference. Only fills when the field is empty so it doesn't
      // clobber a price the user already typed before switching back and
      // forth between market/limit.
      if (orderState.type === 'limit') {
        const priceInput = document.getElementById('order-limit-price');
        if (!priceInput.value) {
          const sym = document.getElementById('order-ticker').value;
          const price = refPrice(sym);
          if (price != null) priceInput.value = Math.round(price * 100) / 100;
        }
      }
      // After any prefill above — market<->limit switches which basis the
      // quick-buttons/estimate use (KIS max_qty vs. cash/price).
      syncOrderQtyQuick();
      updateOrderEstimate();
    });
  });
  document.querySelectorAll('#order-qty-quick .oqty').forEach(b => {
    b.addEventListener('click', () => {
      const base = orderMaxShares();
      if (base == null || base < 1) return;
      const pct = parseFloat(b.dataset.pct);
      const shares = pct >= 1 ? base : Math.max(1, Math.floor(base * pct));
      document.getElementById('order-shares').value = String(shares);
      // Programmatic .value assignment doesn't fire 'input' — call directly.
      updateOrderEstimate();
    });
  });
  document.getElementById('order-shares').addEventListener('input', updateOrderEstimate);
  document.getElementById('order-limit-price').addEventListener('input', () => {
    // A typed limit price changes the buy-side quantity basis too (see
    // orderMaxShares), not just the amount — both must refresh together.
    syncOrderQtyQuick();
    updateOrderEstimate();
  });
  sel.addEventListener('change', refreshAvail);
  document.getElementById('order-submit').addEventListener('click', submitOrder);
  updateOrderSubmitButton();
  const rrSave = document.getElementById('rr-save');
  if (rrSave) rrSave.addEventListener('click', saveRiskRules);
  const killToggle = document.getElementById('safety-kill-toggle');
  if (killToggle) killToggle.addEventListener('click', toggleKillSwitch);
  document.querySelectorAll('#rt-ai-profile .seg-btn, #rt-execution-mode .seg-btn').forEach(b => {
    b.addEventListener('click', () => setRuntimeSetting(b.dataset.key, b.dataset.value));
  });
  loadExecMode();
  loadRuntimeSettings();
  loadRiskRules();  // also primes _riskRules for the order approval prompt
  loadSafetyStatus();
  refreshAvail();
  loadOpenOrders();
}

/* ---------------- screeners (volume/fluctuation/sector rank) ---------------- */

export let _rankMode = 'volume';
// sector has no domestic/overseas split (it's a KRX-only concept), so its
// URL is fixed; every other mode gets `scope` appended for the active
// 국내/해외 header toggle (_marketScope).
const _RANK_PATH = { volume: '/api/volume-rank', value: '/api/trading-value-rank', up: '/api/fluctuation?direction=up', down: '/api/fluctuation?direction=down', sector: '/api/sectors' };

export let _rankCache = { rows: [] };

function _rankUrl(mode) {
  const path = _RANK_PATH[mode];
  if (mode === 'sector') return path;
  return path + (path.includes('?') ? '&' : '?') + `scope=${_marketScope}`;
}

export async function loadRank(mode) {
  if (mode) {
    _rankMode = mode;
    expanded.rank = false;  // switching filters starts collapsed again
  }
  document.querySelectorAll('#rank-modes .rank-mode').forEach(b => b.classList.toggle('active', b.dataset.mode === _rankMode));
  let rows = _rankCache.rows;
  try { rows = await fetch(_rankUrl(_rankMode)).then(r => r.json()); } catch { /* keep last */ }
  _rankCache = { rows };
  renderRankRows(rows);
}

export function renderRankRows(rows) {
  const body = document.getElementById('rank-body');
  if (!rows.length) { body.innerHTML = `<tr><td colspan="6"><div class="empty">${t('rankEmpty')}</div></td></tr>`; return; }
  body.innerHTML = '';
  const visible = expanded.rank ? rows : rows.slice(0, RANK_LIST_CAP);
  if (_rankMode === 'sector') {  // sector-index breadth — not a ticker, so no click-through/watch button
    visible.forEach((s, i) => {
      const up = s.change_pct >= 0;
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${i + 1}</td><td>${escapeHtml(s.name)}</td>`
        + `<td class="num">${s.value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>`
        + `<td class="num ${up ? 'up' : 'down'}">${up ? '+' : ''}${s.change_pct.toFixed(2)}%</td>`
        + `<td class="num"></td><td class="num"></td>`;
      body.appendChild(tr);
    });
  } else {
    // Overseas rows come back as bare symbols (no .KS suffix — see
    // _rank_rows on the backend), so isDomesticSymbol distinguishes the two
    // per-row rather than trusting the tab's own scope (a stale cache
    // mid-switch shouldn't mislabel currency).
    visible.forEach(r => {
      const domestic = isDomesticSymbol(r.ticker);
      const up = r.change_pct >= 0;
      const inWatch = !!state[r.ticker];
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td>${r.rank}</td>`
        + `<td><span class="rk-name">${escapeHtml(r.name)}</span> <span style="color:var(--muted)">${escapeHtml(displayTicker(r.ticker))}</span></td>`
        + `<td class="num">${fmtPrice(r.price, domestic ? 'KRW' : 'USD')}</td>`
        + `<td class="num ${up ? 'up' : 'down'}">${up ? '+' : ''}${r.change_pct.toFixed(2)}%</td>`
        + `<td class="num">${r.volume.toLocaleString('en-US')}</td>`
        + `<td class="num"></td>`;
      tr.querySelector('.rk-name').addEventListener('click', () => {
        if (state[r.ticker]) openDetail(r.ticker);
        else addTickerDirect(r.ticker);
      });
      const cell = tr.lastElementChild;
      if (!inWatch) {
        const add = document.createElement('button');
        add.className = 'rk-add'; add.textContent = t('rkAdd');
        add.addEventListener('click', () => addTickerDirect(r.ticker));
        cell.appendChild(add);
      } else {
        cell.textContent = t('rkInTradeAllowlist');
        cell.style.color = 'var(--muted)';
      }
      body.appendChild(tr);
    });
  }
  if (rows.length > RANK_LIST_CAP) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 6;
    td.appendChild(moreButton('rank', rows.length - RANK_LIST_CAP, () => renderRankRows(_rankCache.rows)));
    tr.appendChild(td);
    body.appendChild(tr);
  }
}

/* ---------------- market-regime bar (KOSPI/KOSDAQ) ---------------- */

export function renderIndexBar(idx) {
  // #market-bar was removed from the DOM (2026-08-19, header teardown) — this
  // stays a no-op until a future feature re-adds the markup. pollIndex()/
  // pollDashboardSummary() keep calling this unconditionally, so guard here
  // rather than touch either call site.
  const bar = document.getElementById('market-bar');
  if (!bar) return;
  if (!idx.length) { bar.style.display = 'none'; return; }
  bar.style.display = 'flex';
  let html = idx.map(q => {
    const up = q.change_pct >= 0;
    // circuit-breaker levels on a broad index start at -8%
    const cb = q.change_pct <= -8 ? ` <span class="cb-warn">${t('cbWarn')}</span>` : '';
    return `<div class="pulse-item ${up ? 'up' : 'down'}">`
      + `<span class="pulse-label">${escapeHtml(q.name)}</span>`
      + `<span class="pulse-value">${q.value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} `
      + `<span class="pulse-chg">${up ? '▲' : '▼'} ${Math.abs(q.change).toFixed(2)} (${q.change_pct.toFixed(2)}%)</span>${cb}</span></div>`;
  }).join('');
  // USD/KRW — no separate fetch, reuses the rate already carried on the
  // account payload (pollPortfolio → renderAccountSummary → _lastAccount) for the
  // hero's own fx-toggle. No day-over-day change is available for it, so
  // this shows the rate only, not a delta like the indices above.
  if (_lastAccount?.usd_krw_rate) {
    html += `<div class="pulse-item"><span class="pulse-label">USD/KRW</span>`
      + `<span class="pulse-value">${_lastAccount.usd_krw_rate.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span></div>`;
  }
  bar.innerHTML = html;
}

export async function pollIndex() {
  let idx = [];
  try { idx = await fetch('/api/index').then(r => r.json()); } catch { return; }
  renderIndexBar(idx);
}

// AI 시그널 탭 랜딩 히어로 아래 마켓위젯(코스피/코스닥/원달러/나스닥/VIX,
// 2026-08-19) — renderMarketWidget은 signals.js가 소유(그 탭 렌더링 전담).
export async function pollMarketWidget() {
  let items = [];
  try { items = await fetch('/api/market-widget').then(r => r.json()); } catch { return; }
  renderMarketWidget(items);
}

export let _runtimeSettings = null;

// 매매모드 값 -> 라벨. 설정 탭 #rt-execution-mode가 유일한 실제 컨트롤이라
// full_auto 는 그 버튼의 라벨(execModeFullAutoShort, "자동")과 반드시 맞춰야
// 함 — 예전엔 설정탭 사본이 execModeFullAuto("완전자동")를 따로 써서 같은
// 값의 이름이 컨트롤마다 달라 보였음(2026-08-18 제거). 사이드바의 읽기전용
// #sidebar-exec-mode-text도 같은 라벨을 씀(아래 _renderRuntimeSegments).
function _execModeLabel(mode) {
  if (mode === 'manual') return t('execModeManual');
  if (mode === 'semi_auto') return t('execModeSemiAuto');
  if (mode === 'full_auto') return t('execModeFullAutoShort');
  return t('settingsChecking');
}

export function _renderRuntimeSegments() {
  if (!_runtimeSettings) return;
  const v = _runtimeSettings.values || {};
  document.querySelectorAll('#rt-ai-profile .seg-btn, #rt-execution-mode .seg-btn').forEach(b => {
    b.classList.toggle('active', v[b.dataset.key] === b.dataset.value);
  });
  const sidebarModeText = document.getElementById('sidebar-exec-mode-text');
  if (sidebarModeText) sidebarModeText.textContent = _execModeLabel(v.execution_mode);
  const badge = document.getElementById('rt-account-mode');
  const isReal = _runtimeSettings.kis_account_mode === 'real';
  badge.textContent = isReal ? t('accountModeReal') : t('accountModePaper');
  badge.className = 'seg-badge' + (isReal ? ' real' : '');

  // Sidebar pill (compact mockup-matching version of the same account mode; moved out of the header 2026-08-19).
  const headerPill = document.getElementById('header-account-mode-pill');
  document.getElementById('header-account-mode-text').textContent = isReal ? t('accountModeReal') : t('accountModePaper');
  headerPill.className = 'account-mode-pill' + (isReal ? ' real' : '');
}

export async function loadRuntimeSettings() {
  try {
    _runtimeSettings = await fetch('/api/settings/runtime').then(r => r.json());
  } catch { return; }
  _renderRuntimeSegments();
}

export async function setRuntimeSetting(key, value) {
  const res = document.getElementById('rt-result');
  document.querySelectorAll(`.seg-btn[data-key="${key}"]`).forEach(b => b.disabled = true);
  // FULL_AUTO submits orders without a human clicking approve each time —
  // worth one extra confirmation, same reasoning as exec-mode-pill's own
  // "escalating visual weight" comment for that mode.
  if (key === 'execution_mode' && value === 'full_auto') {
    const proceed = await confirmDialog({ title: t('execModeFullAuto'), body: t('fullAutoSwitchConfirm') });
    if (!proceed) {
      document.querySelectorAll(`.seg-btn[data-key="${key}"]`).forEach(b => b.disabled = false);
      return;
    }
  }
  try {
    const r = await fetch('/api/settings/runtime', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ [key]: value }),
    });
    const data = await r.json();
    if (!r.ok) {
      const msg = t('runtimeChangeFailed', data.detail || httpErrorText(r.status));
      res.textContent = msg; res.className = 'order-result bad';
      // #rt-result lives inside the 설정 tab. ai_profile has no control
      // outside that tab, so #rt-result alone is always in view for it —
      // but execution_mode's only control is now the header (its 설정-tab
      // copy was removed, 2026-08-18), which can be clicked from any tab,
      // so ONLY that key also gets a toast; otherwise a failure triggered
      // from e.g. the AI 시그널 tab would be completely silent.
      if (key === 'execution_mode') showToast(t('settingsExecModeLabel'), msg, 'bad');
    } else {
      _runtimeSettings = data;
      _renderRuntimeSegments();
      res.textContent = t('runtimeChangeSaved'); res.className = 'order-result ok';
      loadExecMode();  // header pill + reanalyze-button gating pick up the change now
    }
  } catch (e) {
    // e itself isn't shown — it's a raw JS error object/message, not meant
    // for the user; console.error keeps it available for devtools.
    console.error('runtime setting change request failed', e);
    const msg = t('runtimeChangeRequestFailed');
    res.textContent = msg; res.className = 'order-result bad';
    if (key === 'execution_mode') showToast(t('settingsExecModeLabel'), msg, 'bad');
  } finally {
    res.hidden = false;
    document.querySelectorAll(`.seg-btn[data-key="${key}"]`).forEach(b => b.disabled = false);
  }
}

// ---------------- risk-rule settings (설정 탭) ----------------
export let _riskRules = null;

export async function loadRiskRules() {
  try {
    _riskRules = await fetch('/api/settings/risk').then(r => r.json());
  } catch { return; }
  const v = _riskRules.values || {};
  const set = (id, val) => { const el = document.getElementById(id); if (el && val != null) el.value = val; };
  const chk = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };
  set('rr-max-krw', v.kis_max_order_krw);
  set('rr-max-usd', v.kis_max_order_usd);
  chk('rr-pct-enabled', v.kis_max_position_pct_enabled);
  set('rr-pct', v.kis_max_position_pct);
  chk('rr-approval-enabled', v.kis_approval_required_enabled);
  set('rr-approval-krw', v.kis_approval_threshold_krw);
  set('rr-approval-usd', v.kis_approval_threshold_usd);
}

export async function saveRiskRules() {
  const num = id => { const n = parseFloat(document.getElementById(id).value); return Number.isFinite(n) ? n : undefined; };
  const values = {
    kis_max_order_krw: num('rr-max-krw'),
    kis_max_order_usd: num('rr-max-usd'),
    kis_max_position_pct_enabled: document.getElementById('rr-pct-enabled').checked,
    kis_max_position_pct: num('rr-pct'),
    kis_approval_required_enabled: document.getElementById('rr-approval-enabled').checked,
    kis_approval_threshold_krw: num('rr-approval-krw'),
    kis_approval_threshold_usd: num('rr-approval-usd'),
  };
  Object.keys(values).forEach(k => values[k] === undefined && delete values[k]);
  const res = document.getElementById('rr-result');
  const btn = document.getElementById('rr-save');

  // Check the cap/threshold relationship here rather than letting the server
  // 400. risk_rules._validate_cross_field rejects a threshold >= its cap
  // (above the cap an order is hard-blocked, so approval never applies and
  // the threshold would be dead config). The real-account caps ship as 0,
  // which makes the very first save of an untouched form fail this rule with
  // a message about a field the user never touched — clearer to say which
  // box to fix, before the round trip.
  // 서버 400 대신 여기서 먼저 확인 — 실전 한도 기본값이 0이라 손대지 않은 폼을
  // 그대로 저장하면 건드린 적 없는 필드 얘기를 하는 오류가 나옴.
  const showBad = (msg) => {
    res.textContent = msg;
    res.className = 'order-result bad';
    res.hidden = false;
  };
  for (const [label, capKey, thrKey] of [
    [t('scopeDomestic'), 'kis_max_order_krw', 'kis_approval_threshold_krw'],
    [t('scopeOverseas'), 'kis_max_order_usd', 'kis_approval_threshold_usd'],
  ]) {
    const cap = values[capKey];
    const thr = values[thrKey];
    if (cap === undefined || thr === undefined) continue;
    if (cap <= 0) {
      showBad(t('riskCapZero', label, cap));
      return;
    }
    if (thr >= cap) {
      showBad(t('riskThresholdTooHigh', label, thr.toLocaleString(), cap.toLocaleString()));
      return;
    }
  }

  btn.disabled = true;
  try {
    const r = await fetch('/api/settings/risk', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }),
    });
    const data = await r.json();
    if (!r.ok) { res.textContent = t('riskSaveFailed', data.detail || httpErrorText(r.status)); res.className = 'order-result bad'; }
    else { _riskRules = data; res.textContent = t('riskSaved'); res.className = 'order-result ok'; }
  } catch (e) {
    console.error('risk rules save request failed', e);
    res.textContent = t('riskSaveRequestFailed'); res.className = 'order-result bad';
  } finally {
    res.hidden = false;
    btn.disabled = false;
  }
}

// ---------------- kill switch / daily-loss safety panel (설정 탭) ----------------
// `dl.loss_fraction` is a LOSS fraction (positive = lost money, negative =
// gained) — that sign convention is what the actual circuit-breaker gate
// enforces (app.execution.safety_switch.daily_loss_block) and must not
// change. The badge shows P&L instead (flip the sign, prefix + for a gain)
// since "오늘 손익" read alongside a bare "-1.9%" looked like a 1.9% LOSS
// when the account was actually up 1.9% — see the real incident this
// fixes: a 24-hour uptrend still showed as a negative "loss" number.
export function _renderDailyLossBadge(badge, dl) {
  if (!dl || !dl.available) {
    badge.textContent = t('dailyLossUnavailable');
    badge.className = 'safety-badge';
    return;
  }
  const pnlPct = -dl.loss_fraction * 100;
  const pnlStr = (pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(1);
  const thr = (dl.threshold * 100).toFixed(1);
  badge.textContent = t('dailyLossBadge', pnlStr, thr) + (dl.breached ? t('dailyLossBreachedSuffix') : '');
  badge.className = 'safety-badge' + (dl.breached ? ' breached' : pnlPct >= 0 ? ' ok' : '');
}

export async function loadSafetyStatus() {
  const killBadge = document.getElementById('safety-kill-badge');
  const lossBadgeDomestic = document.getElementById('safety-daily-loss-domestic');
  const lossBadgeOverseas = document.getElementById('safety-daily-loss-overseas');
  const toggleBtn = document.getElementById('safety-kill-toggle');
  let data;
  try {
    data = await fetch('/api/execution/safety').then(r => r.json());
  } catch {
    killBadge.textContent = t('safetyLoadFailed');
    lossBadgeDomestic.textContent = t('safetyLoadFailed');
    lossBadgeOverseas.textContent = t('safetyLoadFailed');
    return;
  }
  if (data.kill_switch_engaged) {
    killBadge.textContent = t('killSwitchEngagedBadge', data.kill_switch_reason || t('killSwitchNoReason'));
    killBadge.className = 'safety-badge engaged';
    toggleBtn.textContent = t('killSwitchDisengageBtn');
    toggleBtn.className = 'safety-kill-btn disengage';
    toggleBtn.dataset.action = 'disengage';
  } else {
    killBadge.textContent = t('killSwitchOkBadge');
    killBadge.className = 'safety-badge ok';
    toggleBtn.textContent = t('killSwitchTitle');
    toggleBtn.className = 'safety-kill-btn';
    toggleBtn.dataset.action = 'engage';
  }
  const dl = data.daily_loss || {};
  _renderDailyLossBadge(lossBadgeDomestic, dl.domestic);
  _renderDailyLossBadge(lossBadgeOverseas, dl.overseas);
}

export async function toggleKillSwitch() {
  const btn = document.getElementById('safety-kill-toggle');
  const res = document.getElementById('safety-result');
  let body;
  if (btn.dataset.action === 'engage') {
    // inputMode: 'text' — this reason field isn't a price (promptDialog's
    // default 'decimal' inputMode was written for the revise-price prompt,
    // its original only caller; a free-text reason needs a normal keyboard).
    const reason = await promptDialog({ title: t('killSwitchTitle'), body: t('killSwitchReasonPrompt'), inputMode: 'text' });
    if (reason === null || !reason.trim()) return;
    const proceedEngage = await confirmDialog({ title: t('killSwitchTitle'), body: t('killSwitchEngageConfirm'), danger: true });
    if (!proceedEngage) return;
    body = { engaged: true, reason: reason.trim() };
  } else {
    const proceedDisengage = await confirmDialog({ title: t('killSwitchTitle'), body: t('killSwitchDisengageConfirm'), danger: true });
    if (!proceedDisengage) return;
    body = { engaged: false };
  }
  btn.disabled = true;
  try {
    const r = await fetch('/api/execution/safety/kill-switch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    // A server-side crash the router doesn't translate to an HTTPException
    // (e.g. a DB timeout — confirmed live 2026-08-06) returns a plain-text
    // body, not JSON — r.json() then throws its own SyntaxError that masked
    // the real failure. Try text first so a non-JSON body still shows
    // something readable instead of "Unexpected token...".
    const raw = await r.text();
    let data;
    try { data = JSON.parse(raw); } catch { data = null; }
    // raw is the unparsed HTTP body (see comment above) — useful for
    // debugging a non-JSON error response, but it can be an HTML error page
    // or a stack trace, so it goes to the console, never onto the page.
    if (!r.ok) {
      if (!data || !data.detail) console.error('kill-switch request failed, raw response', raw);
      res.textContent = t('killSwitchActionFailed', (data && data.detail) ? data.detail : httpErrorText(r.status));
      res.className = 'order-result bad';
    }
    else { res.textContent = data.kill_switch_engaged ? t('killSwitchEngagedResult') : t('killSwitchDisengagedResult'); res.className = 'order-result ok'; }
  } catch (e) {
    console.error('kill-switch request failed', e);
    res.textContent = t('killSwitchRequestFailed'); res.className = 'order-result bad';
  } finally {
    res.hidden = false;
    btn.disabled = false;
    loadSafetyStatus();
  }
}

// Real-time fill notifications (체결통보) -> transient toasts, except an
// actual fill (status 'filled') gets the bigger trade-fill modal instead —
// user asked to be shown what got bought/sold, at what price, and why
// (2026-07-30), not just a small corner toast easy to miss.
export const _seenFills = new Set();
export let _fillsPrimed = false;
export async function pollFills() {
  let fills = [];
  try { fills = await fetch('/api/fills').then(r => r.json()); } catch { return; }
  // prime on first load so pre-existing fills don't all toast at once
  if (!_fillsPrimed) { fills.forEach(f => _seenFills.add(f.order_no + f.status)); _fillsPrimed = true; return; }
  for (const f of fills.slice().reverse()) {
    const key = f.order_no + f.status;
    if (_seenFills.has(key)) continue;
    _seenFills.add(key);
    if (f.status === 'filled') {
      showTradeFillModal(f);
    } else {
      const sideKo = f.side === 'buy' ? (LANG === 'ko' ? '매수' : 'Buy') : (LANG === 'ko' ? '매도' : 'Sell');
      const label = f.status === 'rejected' ? (LANG === 'ko' ? '거부' : 'Rejected') : (LANG === 'ko' ? '접수' : 'Accepted');
      showToast(`${label} · ${f.name || f.ticker} ${sideKo}`,
        `${f.fill_qty || f.order_qty}주 @ ${fmtKRW(f.fill_price)} · ${f.time || ''}`, f.status);
    }
    pollPortfolio(); pollTrades();
  }
}

// AI's own reasoning for a fill, matched by ticker + today's date against
// the already-polled decisions feed (signals.js's pollDecisions) — the
// fill event itself (KIS 체결통보) carries no reasoning, only order/price
// facts, so this is the only place that reasoning is available client-side.
// Falls back to a plain "no matching decision found" line rather than
// hiding the reason section — a fill with no matched decision (a manual
// dashboard order, or a stop-loss forced exit) is itself useful to know.
async function _fillReasonText(ticker) {
  const today = _todayStr();
  let decision = _lastDecisions.find(d => d.ticker === ticker && d.trade_date === today);
  if (!decision) {
    // pollFills (5s) runs far more often than pollDecisions (60s), so a
    // decision recorded only seconds before its own trade fires can still
    // be missing from the stale _lastDecisions cache at this exact moment
    // (confirmed live 2026-08-05: 316140.KS's decision and fill landed 6
    // seconds apart, and the modal showed "no matching decision" even
    // though one existed). One on-demand refresh closes that race without
    // changing pollDecisions' own 60s baseline cadence.
    try {
      const fresh = await fetch('/api/decisions').then(r => r.json());
      _lastDecisions = fresh;
      decision = fresh.find(d => d.ticker === ticker && d.trade_date === today);
    } catch { /* network hiccup — fall through to the "not found" message below */ }
  }
  if (!decision) return LANG === 'ko' ? 'AI 판단 기록을 찾을 수 없습니다(수동 주문이거나 손절 실행일 수 있음).' : 'No matching AI decision found (may be a manual order or a stop-loss exit).';
  const summary = _executiveSummaryFrom(decision.decision_text);
  const rating = _RATING_KO[decision.rating] || decision.rating;
  return summary ? `[${rating}] ${summary}` : `[${rating}]`;
}

let _tfmDismissTimer = null;
// 열려 있는 동안만 붙어 있는 Escape 핸들러. 모듈 스코프에 보관해 두는 이유는
// hideTradeFillModal()이 (자동 10초 타이머든, 닫기 버튼이든, 배드롭 클릭이든)
// 어느 경로로 불려도 같은 함수 참조로 확실히 떼기 위함 — 체결이 연달아 나면
// showTradeFillModal이 반복 호출되므로, 매번 새 익명함수를 붙이면 리스너가
// 쌓인다.
//
// 이 모달은 포커스를 가두지도, 열 때 포커스를 옮기지도 않는다(index.html의
// 주석 참조: 알림이지 dialog가 아님). Escape만이 유일한 즉시 탈출 경로다.
let _tfmKeydown = null;
export async function showTradeFillModal(f) {
  const modal = document.getElementById('trade-fill-modal');
  if (!modal) return;
  const isBuy = f.side === 'buy';
  const sideKo = isBuy ? (LANG === 'ko' ? '매수 체결' : 'Buy Filled') : (LANG === 'ko' ? '매도 체결' : 'Sell Filled');
  document.getElementById('tfm-badge').textContent = sideKo;
  document.getElementById('tfm-badge').className = 'tfm-badge ' + (isBuy ? 'buy' : 'sell');
  document.getElementById('tfm-name').textContent = f.name || f.ticker;
  document.getElementById('tfm-symbol').textContent = displayTicker(f.ticker);
  const qty = f.fill_qty || f.order_qty;
  document.getElementById('tfm-detail').textContent = `${qty}주 @ ${fmtKRW(f.fill_price)}`;
  document.getElementById('tfm-reason-label').textContent = LANG === 'ko' ? 'AI 판단 이유' : 'AI reasoning';
  document.getElementById('tfm-reason-text').textContent = LANG === 'ko' ? '조회 중...' : 'Looking up...';
  modal.hidden = false;
  clearTimeout(_tfmDismissTimer);
  _tfmDismissTimer = setTimeout(hideTradeFillModal, 10000);
  if (!_tfmKeydown) {
    // capture 단계로 붙인다 — dialog.js와 같은 이유로, 다른 페이지 핸들러가
    // Escape를 먼저 삼켜도 여기까지 온다.
    _tfmKeydown = (e) => { if (e.key === 'Escape') { e.preventDefault(); hideTradeFillModal(); } };
    document.addEventListener('keydown', _tfmKeydown, true);
  }
  // Awaited after the modal is already shown (with a placeholder) so the
  // fill itself is never delayed by _fillReasonText's on-demand refetch —
  // only the reason line fills in a beat later on the rare race it needs to.
  document.getElementById('tfm-reason-text').textContent = await _fillReasonText(f.ticker);
}
export function hideTradeFillModal() {
  const modal = document.getElementById('trade-fill-modal');
  if (modal) modal.hidden = true;
  clearTimeout(_tfmDismissTimer);
  if (_tfmKeydown) {
    // 붙일 때와 같은 capture 플래그(true)를 반드시 넘겨야 실제로 떨어진다.
    document.removeEventListener('keydown', _tfmKeydown, true);
    _tfmKeydown = null;
  }
}
export function showToast(title, sub, kind) {
  const stack = document.getElementById('toast-stack');
  const el = document.createElement('div');
  el.className = 'toast ' + (kind || '');
  el.innerHTML = `<div class="t-title"></div><div class="t-sub"></div>`;
  el.querySelector('.t-title').textContent = title;
  el.querySelector('.t-sub').textContent = sub;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 7000);
}
