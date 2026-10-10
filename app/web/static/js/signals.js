import { LANG, LIST_CAP, MAX_HISTORY, _marketScope, dateLocale, displayTicker, escapeHtml, expanded, fmtDelta, fmtPrice, formatTimeAgo, isDomesticSymbol, moreButton, setHeldReviewBufferPct, state, t } from './state.js';
import { ANALYZE_PIPELINE_STAGES, _currentExecutionMode, currentDetailSymbol, drawSparkline, matchAnalyzePipelineStage, renderCardStats, renderCompactAiDiagnosis, renderDetailHeader, showToast, updateOrderEstimate, venueChip, wireTranscriptToggle } from './trading.js';
import { renderMarkdown } from './assistant.js';

export function startStream() {
  const es = new EventSource('/api/stream');
  const dot = document.getElementById('live-dot');
  const label = document.getElementById('live-label');

  es.onopen = () => { dot.classList.remove('off'); label.textContent = t('live'); };
  es.onerror = () => { dot.classList.add('off'); label.textContent = t('reconnecting'); };

  es.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    for (const [symbol, tick] of Object.entries(payload)) {
      if (!tick || !state[symbol]) continue;
      const h = state[symbol].history;
      h.push(tick);
      if (h.length > MAX_HISTORY) h.shift();
      renderCardStats(symbol);
      // 사이드바 미니 그래프는 없앴고, 상세화면 차트는 loadPeriodChart가 30초
      // 폴링으로 갱신하므로 여기선 현재가 숫자만 갱신.
      if (symbol === currentDetailSymbol) renderDetailHeader(symbol);
      // Market-order estimate is refPrice()-based, so a live tick for the
      // ticker currently selected in the order ticket must refresh it too —
      // otherwise the amount freezes the instant the panel is opened.
      if (symbol === document.getElementById('order-ticker')?.value) updateOrderEstimate();
    }
  };
}

export let _alertsCache = [];

export async function pollAlerts() {
  _alertsCache = await fetch('/api/alerts').then(r => r.json());
  renderAlerts();
}

export function renderAlerts() {
  const alerts = _alertsCache;
  const el = document.getElementById('alerts');
  if (!alerts.length) {
    el.innerHTML = `<div class="empty">${t('alertsEmpty')}</div>`;
    return;
  }
  el.innerHTML = '';
  for (const a of (expanded.alerts ? alerts : alerts.slice(0, LIST_CAP))) {
    const row = document.createElement('div');
    row.className = 'alert';
    // Pinned to Asia/Seoul explicitly (2026-08-14, same reasoning as
    // _decisionDateTime above) rather than relying on the browser's
    // implicit local timezone — this dashboard's whole audience is Korean.
    const time = new Date(a.triggered_at).toLocaleString(dateLocale(), { timeZone: 'Asia/Seoul' });
    const label = state[a.ticker]?.name ? `${state[a.ticker].name} (${displayTicker(a.ticker)})` : displayTicker(a.ticker);
    const isPriceMove = a.kind === 'price_move';
    const isStopLoss = a.kind === 'stop_loss';
    const isSystemFailure = a.kind === 'pipeline_failure' || a.kind === 'cycle_failure';
    const isGapRisk = a.kind === 'overnight_gap_risk';
    const isUnmirrored = a.kind === 'unmirrored_trade';
    // Both share the same dot/urgency — 'final' is just the interim alert's
    // last word after fill_monitor gives up re-checking (see
    // app.live.fill_monitor.check_pending_fills's own docstring), not a
    // different severity.
    const isUnconfirmedFill = a.kind === 'unconfirmed_fill' || a.kind === 'unconfirmed_fill_final';
    const dotCls = isStopLoss || isSystemFailure || isGapRisk || isUnmirrored || isUnconfirmedFill ? 'down' : isPriceMove ? (a.message.includes('+') ? 'up' : a.message.includes('-') ? 'down' : 'accent') : 'accent';
    const kindLabel = isStopLoss ? t('alertStopLossKind')
      : a.kind === 'pipeline_failure' ? t('alertPipelineFailureKind')
      : a.kind === 'cycle_failure' ? t('alertCycleFailureKind')
      : isGapRisk ? t('alertOvernightGapKind')
      : isUnmirrored ? t('alertUnmirroredTradeKind')
      : a.kind === 'unconfirmed_fill_final' ? t('alertUnconfirmedFillFinalKind')
      : isUnconfirmedFill ? t('alertUnconfirmedFillKind')
      : isPriceMove ? t('alertPriceKind') : t('alertVolumeKind');
    row.innerHTML = `<span class="dot ${dotCls}"></span><div><span class="kind-chip">${kindLabel}</span><span class="ticker">${escapeHtml(label)}</span> — ${escapeHtml(a.message)}<div class="meta">${time}</div></div>`;
    el.appendChild(row);
  }
  if (alerts.length > LIST_CAP) {
    el.appendChild(moreButton('alerts', alerts.length - LIST_CAP, renderAlerts));
  }
}

export let _feedCache = { decisions: [], tradeByDecisionId: {} };

// `d.trade_date` is a market-LOCAL calendar date (KST for domestic, but US
// Eastern for overseas — app.clock.local_today's deliberate definition, see
// its own docstring) while `created_at` is a real UTC timestamp. Naively
// showing "${d.trade_date} ${time-from-created_at}" mixed an ET-anchored
// date with a KST-anchored time from two different clocks — e.g. an
// overseas trade at KST 03:14 (still within the US session that started
// the evening before) rendered as "2026-08-13 03:14", a moment that never
// existed in EITHER timezone (2026-08-14 in the user's own night — see
// project memory, 2026-08-14 user report). This one Korean audience is what
// every dashboard user is (project decision, 2026-08-14), so both date and
// time are derived from `created_at` ALONE and explicitly pinned to
// Asia/Seoul — not the ambiguous market-local `trade_date`, and not the
// browser's implicit local timezone (which happens to be KST for this
// audience today, but pinning it removes that assumption entirely).
// `created_at` is a required field on every /api/decisions entry, so no
// null-guard needed here.
function _decisionDateTime(createdAt) {
  return new Date(createdAt).toLocaleString(dateLocale(), {
    timeZone: 'Asia/Seoul',
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  });
}

// 기타탭 4개 섹션(에이전트 활동/승인대기/최근거래/포트폴리오 상세) 모두 헤더의 전역
// 국내/해외 토글(_marketScope, applyTradingScope)을 그대로 따름 — 관심종목·랭킹
// 화면이 이미 쓰는 것과 같은 토글 하나로 통일(탭마다 별도 토글을 새로 안 만듦).
export function renderFeed(decisions, tradeByDecisionId) {
  _feedCache = { decisions, tradeByDecisionId };
  const el = document.getElementById('feed');
  const scoped = decisions.filter(d => (_marketScope === 'domestic') === isDomesticSymbol(d.ticker));
  if (!scoped.length) {
    el.innerHTML = `<div class="empty">${t('feedEmpty')}</div>`;
    document.getElementById('feed-count').textContent = '';
    return;
  }
  const today = new Date().toISOString().slice(0, 10);
  const todayCount = scoped.filter(d => d.trade_date === today).length;
  document.getElementById('feed-count').textContent = todayCount ? t('todayCount', todayCount) : '';

  const visible = expanded.feed ? scoped.slice(0, 20) : scoped.slice(0, LIST_CAP);
  el.innerHTML = '';
  for (const d of visible) {
    const trade = tradeByDecisionId[d.id];
    const dotCls = trade ? trade.side : 'info';
    const orderHtml = trade
      ? ` → <span class="side-chip ${trade.side}">${trade.side === 'buy' ? t('buy') : t('sell')} ${trade.shares.toLocaleString('en-US')}${t('sharesUnit')}</span> ${venueChip(trade.venue)}`
      : '';
    const outcome = d.status === 'resolved' && d.alpha_return != null
      ? t('resolvedAlpha', (d.alpha_return * 100).toFixed(1))
      : t('pendingOutcome');
    const item = document.createElement('div');
    item.className = 'feed-item';
    item.innerHTML = `
      <span class="feed-dot ${dotCls}"></span>
      <div class="feed-body">
        <div class="feed-head">
          <b>${escapeHtml(d.name)}</b> <span class="rating-chip ${d.rating.toLowerCase()}">${escapeHtml(d.rating)}</span>${orderHtml}
          <span class="feed-time">${_decisionDateTime(d.created_at)}</span>
        </div>
        <div class="feed-sub">${escapeHtml(outcome)} · <span class="see-more">${t('seeFull')}</span></div>
        <div class="transcript" style="display:none"></div>
      </div>
    `;
    el.appendChild(item);
    wireTranscriptToggle(
      item.querySelector('.see-more'),
      item.querySelector('.transcript'),
      d.id,
      { open: t('hideFull'), closed: t('seeFull') }
    );
  }
  if (scoped.length > LIST_CAP) {
    const hidden = Math.min(scoped.length, 20) - LIST_CAP;
    el.appendChild(moreButton('feed', hidden, () => renderFeed(_feedCache.decisions, _feedCache.tradeByDecisionId)));
  }
}

export async function pollDecisions() {
  // Trades are fetched alongside so each decision row can show the order it
  // actually produced — matched by decision_id (set by rebalance_to_target
  // only on the trade it itself records), NOT (ticker, trade_date): the same
  // ticker can trade more than once on the same day (e.g. a same-day
  // stop-loss exit on top of this decision's own rebalance trade), and a
  // (ticker, trade_date) key would collide and show the wrong trade next to
  // a decision. Trades with no decision_id (stop-loss exits, manual dashboard
  // orders) are simply never matched to any decision. The same map feeds
  // both the agent-activity feed (top of page) and the decision-history tab.
  const [decisions, trades] = await Promise.all([
    fetch('/api/decisions').then(r => r.json()),
    fetch('/api/trades?limit=100').then(r => r.json()).catch(() => []),
  ]);
  const tradeByDecisionId = {};
  for (const trade of trades) {
    if (trade.decision_id != null) tradeByDecisionId[trade.decision_id] = trade;
  }
  renderFeed(decisions, tradeByDecisionId);
  renderDecisionsList(decisions, tradeByDecisionId);
  notifyNewFullAutoDecisions(decisions, tradeByDecisionId);
  _lastDecisions = decisions;
  renderAnalyzingCards();
  if (currentDetailSymbol) renderCompactAiDiagnosis(currentDetailSymbol);
}

// FULL_AUTO only: an "AI just did this" FYI card per new decision — unlike
// showPendingApprovalModal's action prompt, this auto-dismisses (showToast's
// normal 7s) since there's nothing to approve, the trade (if any) already
// happened. _decisionsPrimed skips whatever's already in history at load.
export const _seenDecisionIds = new Set();
export let _decisionsPrimed = false;

export function notifyNewFullAutoDecisions(decisions, tradeByDecisionId) {
  if (!_decisionsPrimed) {
    decisions.forEach(d => _seenDecisionIds.add(d.id));
    _decisionsPrimed = true;
    return;
  }
  for (const d of decisions) {
    if (_seenDecisionIds.has(d.id)) continue;
    _seenDecisionIds.add(d.id);
    if (_currentExecutionMode !== 'full_auto') continue;
    const trade = tradeByDecisionId[d.id];
    const sub = trade
      ? `${trade.side === 'buy' ? t('buy') : t('sell')} ${trade.shares.toLocaleString('en-US')}${t('sharesUnit')} @ ${trade.fill_price.toLocaleString('en-US')}`
      : t('fullAutoNoTrade');
    showToast(`${t('fullAutoDecided')} · ${d.name || d.ticker} (${d.rating})`, sub);
  }
}

export function renderDecisionsList(decisions, tradeByDecisionId) {
  const el = document.getElementById('decisions');
  if (!decisions.length) {
    el.innerHTML = `<div class="empty">${t('decisionsEmpty')}</div>`;
    return;
  }
  const visible = expanded.decisions ? decisions : decisions.slice(0, LIST_CAP);
  el.innerHTML = '';
  for (const d of visible) {
    const row = document.createElement('div');
    row.className = 'decision-row';
    // The rating above is what was decided at the time — always known immediately.
    // This line is about whether that call turned out to be *right* in hindsight,
    // which only resolves once enough trading days have passed.
    const outcome = d.status === 'resolved' && d.alpha_return != null
      ? t('resolvedAlpha', (d.alpha_return * 100).toFixed(1)) + (d.reflection ? ': ' + d.reflection : '')
      : t('pendingOutcome');
    const trade = tradeByDecisionId[d.id];
    const orderHtml = trade
      ? ` <span class="decision-order">→ <span class="side-chip ${trade.side}">${trade.side === 'buy' ? t('buy') : t('sell')} ${trade.shares.toLocaleString('en-US')}${t('sharesUnit')}</span> ${venueChip(trade.venue)}</span>`
      : '';
    row.innerHTML = `
      <div class="row-top">
        <div><span class="ticker">${escapeHtml(d.name)} (${escapeHtml(displayTicker(d.ticker))})</span> <span class="rating-chip ${d.rating.toLowerCase()}">${escapeHtml(d.rating)}</span>${orderHtml}</div>
        <span class="see-more">${t('seeDebate')}</span>
      </div>
      <div class="meta">${_decisionDateTime(d.created_at)} — ${escapeHtml(outcome)}</div>
      <div class="transcript" style="display:none"></div>
    `;
    el.appendChild(row);
    wireTranscriptToggle(
      row.querySelector('.see-more'),
      row.querySelector('.transcript'),
      d.id,
      { open: t('hideDebate'), closed: t('seeDebate') }
    );
  }
  if (decisions.length > LIST_CAP) {
    el.appendChild(moreButton('decisions', decisions.length - LIST_CAP,
      () => renderDecisionsList(_feedCache.decisions, _feedCache.tradeByDecisionId)));
  }
}

// SEMI_AUTO mode parks every AI-proposed order here for explicit approval —
// app.execution.pending_orders on the backend; this is the only UI for it.
// SEMI_AUTO only: a new pending order pops up as a centered approval modal
// (showPendingApprovalModal) instead of requiring a trip to the tab —
// _pendingOrdersPrimed skips the first load (everything already-pending at
// page-load isn't "new"), same convention as _seenFills/_fillsPrimed above.
export const _seenPendingIds = new Set();
export let _pendingOrdersPrimed = false;

export async function loadPendingOrders() {
  let orders = [];
  try {
    orders = await fetch('/api/pending-orders').then(r => r.json());
  } catch { /* keep last rendered list on a transient fetch failure */ return; }
  renderPendingOrders(orders);
  loadPendingOrderHistory();

  if (!_pendingOrdersPrimed) {
    orders.forEach(o => _seenPendingIds.add(o.id));
    _pendingOrdersPrimed = true;
    return;
  }
  for (const o of orders) {
    if (_seenPendingIds.has(o.id)) continue;
    _seenPendingIds.add(o.id);
    showPendingApprovalModal(o);
  }
}

export function pendingOrderCardHtml(o) {
  return `
    <div class="row-top">
      <div><span class="ticker">${escapeHtml(o.name || o.ticker)}</span> <span class="side-chip ${o.side}">${o.side === 'buy' ? t('buy') : t('sell')} ${o.shares.toLocaleString('en-US')}${t('sharesUnit')} · ${t('pendingRequestPrice')} ${fmtPrice(o.price, o.currency)}</span></div>
    </div>
    <div class="meta">${escapeHtml(o.reason)}</div>
    <div class="pending-actions">
      <button type="button" class="pending-approve-btn">${t('pendingApprove')}</button>
      <button type="button" class="pending-reject-btn">${t('pendingReject')}</button>
    </div>
    <div class="order-result" hidden></div>
  `;
}

export function wirePendingOrderButtons(container, id) {
  container.querySelector('.pending-approve-btn').addEventListener('click', () => resolvePendingOrder(id, 'approve', container));
  container.querySelector('.pending-reject-btn').addEventListener('click', () => resolvePendingOrder(id, 'reject', container));
}

export let _pendingOrdersCache = [];

// 승인대기 목록(아래)과 히어로 CTA 카운트(_renderHeroPending)가 같은 국내/해외
// 스코프를 봐야 함 — CTA를 눌러 열리는 목적지가 바로 이 scoped 목록이라, 카운트만
// 다른 기준(unscoped)으로 세면 "2건"이라고 해놓고 빈 목록으로 보내는 불일치가 생김.
function _scopedByMarket(orders) {
  return orders.filter(o => (_marketScope === 'domestic') === isDomesticSymbol(o.ticker));
}

export function renderPendingOrders(orders) {
  _pendingOrdersCache = orders;
  _renderHeroPending();  // hero CTA count comes from this same cache, so it can't go stale
  const el = document.getElementById('pending-orders');
  const scoped = _scopedByMarket(orders);
  document.getElementById('pending-count').textContent = scoped.length ? t('itemCount', scoped.length) : '';
  if (!scoped.length) {
    el.innerHTML = `<div class="empty">${t('pendingOrdersEmpty')}</div>`;
    return;
  }
  el.innerHTML = '';
  for (const o of scoped) {
    const row = document.createElement('div');
    row.className = 'pending-row';
    row.innerHTML = pendingOrderCardHtml(o);
    wirePendingOrderButtons(row, o.id);
    el.appendChild(row);
  }
}

export function _pendingStatusLabel(status) {
  if (status === 'approved') return t('pendingStatusApproved');
  if (status === 'rejected') return t('pendingStatusRejected');
  return t('pendingStatusApproveFailed');  // 'approve_failed'
}

// o.price is the request-time reference price (market orders don't know
// their fill until KIS actually matches it) — o.fill_price, when the
// backend could find it (order approved + a matching KIS execution, see
// orders.py's _pending_order_payload), is the real execution price. Show
// both when they're known to differ so this history line stops looking
// like a single, final price that never changes (see the conversation
// this was added for: 한화에어로스페이스 requested @1,049,000 but filled
// @1,024,000, and this label showed only the first number forever).
function _priceLabel(o) {
  if (o.fill_price != null && o.fill_price !== o.price) {
    return `${t('pendingRequestPrice')} ${fmtPrice(o.price, o.currency)} → ${t('pendingFillPrice')} ${fmtPrice(o.fill_price, o.currency)}`;
  }
  return `${t('pendingRequestPrice')} ${fmtPrice(o.price, o.currency)}`;
}

export function pendingHistoryRowHtml(o) {
  return `
    <div class="row-top">
      <div><span class="ticker">${escapeHtml(o.name || o.ticker)}</span> <span class="side-chip ${o.side}">${o.side === 'buy' ? t('buy') : t('sell')} ${o.shares.toLocaleString('en-US')}${t('sharesUnit')} ${_priceLabel(o)}</span></div>
      <div class="pending-history-status ${o.status}">${_pendingStatusLabel(o.status)}</div>
    </div>
    <div class="meta">${escapeHtml(o.resolution || '')}</div>
    <div class="meta" style="opacity:0.7">${o.resolved_at ? new Date(o.resolved_at).toLocaleString(dateLocale(), { timeZone: 'Asia/Seoul' }) : ''}</div>
  `;
}

// What a human actually decided on each approval-modal prompt — resolved
// (approved/approve_failed/rejected) orders aren't returned by
// /api/pending-orders (status='pending' only), so this is the one place
// that history is visible at all. Refreshed on the same cadence as
// loadPendingOrders (called from there) rather than its own poll loop.
export let _pendingHistoryCache = [];

export async function loadPendingOrderHistory() {
  let orders = [];
  try {
    orders = await fetch('/api/pending-orders/history').then(r => r.json());
  } catch { return; }
  _pendingHistoryCache = orders;
  renderPendingOrderHistory();
}

export function renderPendingOrderHistory() {
  const orders = _pendingHistoryCache;
  const el = document.getElementById('pending-orders-history');
  const scoped = orders.filter(o => (_marketScope === 'domestic') === isDomesticSymbol(o.ticker));
  if (!scoped.length) {
    el.innerHTML = `<div class="empty">${t('pendingHistoryEmpty')}</div>`;
    return;
  }
  el.innerHTML = '';
  for (const o of scoped) {
    const row = document.createElement('div');
    row.className = 'pending-row';
    row.innerHTML = pendingHistoryRowHtml(o);
    el.appendChild(row);
  }
}

export async function resolvePendingOrder(id, action, row) {
  const approveBtn = row.querySelector('.pending-approve-btn');
  const rejectBtn = row.querySelector('.pending-reject-btn');
  const resultEl = row.querySelector('.order-result');
  approveBtn.disabled = true;
  rejectBtn.disabled = true;
  const busyLabel = action === 'approve' ? approveBtn : rejectBtn;
  const originalText = busyLabel.textContent;
  busyLabel.textContent = t('pendingApproving');
  try {
    const resp = await fetch(`/api/pending-orders/${id}/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(action === 'reject' ? { reason: '' } : {}),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
    resultEl.hidden = false;
    resultEl.className = 'order-result ok';
    resultEl.textContent = action === 'approve' ? t('pendingApproved', body.disposition || '') : t('pendingRejected');
    setTimeout(loadPendingOrders, 1500);  // refresh the list once the row's outcome has been shown briefly
    setTimeout(() => row.remove(), 1500);  // drop this card/row too — harmless no-op if loadPendingOrders already rebuilt it
  } catch (err) {
    resultEl.hidden = false;
    resultEl.className = 'order-result bad';
    resultEl.textContent = t('pendingActionFailed', err.message || String(err));
    approveBtn.disabled = false;
    rejectBtn.disabled = false;
    busyLabel.textContent = originalText;
  }
}

// app.state.render_portfolio_decision's field labels are hardcoded English
// regardless of the pipeline's own detected language (only the LLM-written
// content inside each field follows it) — jarring in an otherwise-Korean
// dashboard, so the modal translates just the known label set client-side
// rather than changing that shared renderer (used by chat/decisions-tab/
// pipeline reasons too, where the English labels aren't a problem today).
export const _DECISION_LABEL_KO = {
  'Rating': '등급',
  'Executive Summary': '요약',
  'Investment Thesis': '투자 논리',
  'Price Target': '목표가',
  'Time Horizon': '투자 기간',
  'Stop Loss': '손절가',
};

// PortfolioRating's fixed enum values (app/state.py) — always English
// regardless of language, unlike the free-text fields, so a plain lookup
// (not a general translator) is safe here.
export const _RATING_KO = {
  'Buy': '매수', 'Overweight': '비중 확대', 'Hold': '보유', 'Underweight': '비중 축소', 'Sell': '매도',
};

// rebalance_action_label's own fixed enum values (app/execution/sizing.py)
// — what the rebalance ACTUALLY did (current vs post-cap target shares),
// deliberately separate from _RATING_KO above (the AI's own conviction
// judgment) since the two can disagree: an Underweight rating on a
// position far below even that reduced target still nets out as a Buy
// here. Both are shown side by side, not merged.
export const _ACTION_KO = {
  'Buy': '매수', 'Sell': '매도', 'Increase': '비중 증가', 'Decrease': '비중 축소', 'Hold': '유지',
};
// Buy/Sell share the existing .signal-rating-pill buy/sell color classes;
// Increase/Decrease reuse the same up/down coloring (same direction, just
// partial), Hold reuses the amber hold styling.
export const _ACTION_PILL_CLASS = {
  Buy: 'buy', Sell: 'sell', Increase: 'buy', Decrease: 'sell', Hold: 'hold',
};

// (rating, action_label) pairs that mean the same thing — the hint below adds
// nothing for these, and showing it reads as a contradiction ("유지" vs
// "AI 등급: 보유" are both Hold). 93% of live decisions land here.
const _RATING_ACTION_SAME = new Set(['Buy|Buy', 'Sell|Sell', 'Hold|Hold', 'Overweight|Increase', 'Underweight|Decrease']);
export function _ratingDiffersFromAction(rating, actionLabel) {
  if (!actionLabel) return false;
  return !_RATING_ACTION_SAME.has(`${rating}|${actionLabel}`);
}

export function localizeDecisionLabels(text) {
  if (LANG !== 'ko') return text;
  let out = text;
  for (const [en, ko] of Object.entries(_DECISION_LABEL_KO)) out = out.split(`**${en}**`).join(`**${ko}**`);
  out = out.replace(/(\*\*등급\*\*:\s*)(Buy|Overweight|Hold|Underweight|Sell)\b/, (_, prefix, val) => prefix + _RATING_KO[val]);
  return out;
}

// Price Target / Stop Loss come through as bare numbers with no currency
// unit and no sense of how far that is from the order's own price — added
// here (currency-formatted via the existing fmtPrice, +/-% via fmtDelta)
// rather than in app.state.render_portfolio_decision, since that renderer
// has no order price to compute a percentage against. Must run BEFORE
// localizeDecisionLabels while the field labels are still the fixed English
// text this regex matches.
export function enrichPriceFields(text, price, currency) {
  const annotate = (label) => (full, prefix, num) => {
    const value = parseFloat(num);
    if (!price) return prefix + fmtPrice(value, currency);
    return `${prefix}${fmtPrice(value, currency)} (${fmtDelta(value / price - 1)})`;
  };
  return text
    .replace(/(\*\*Price Target\*\*:\s*)([\d.]+)/, annotate('target'))
    .replace(/(\*\*Stop Loss\*\*:\s*)([\d.]+)/, annotate('stop'));
}

// Centered "please decide" modal for a newly-seen pending order — the only
// popup for this event (see the markup comment above). `o.reason` is
// the full portfolio-decision markdown (rating/executive summary/price
// target/stop loss — see app.state.render_portfolio_decision) when this
// order came from the AI pipeline, or the terse approval-threshold string
// when it came from kis_bridge's value-cap gate — renderMarkdown handles
// either as plain text. `o.name`/`o.currency` come from
// state.get_display_name/_display_info (see orders.py's _pending_order_payload)
// — the ticker code alone isn't readable at a glance.
// No auto-close: an investment-decision prompt disappearing on its own after
// 30s (even mid-read, even with a multi-paragraph reason) was worse than
// leaving it open — the order stays untouched (`pending` in the DB) either
// way, so there's no urgency to auto-dismiss. Users close it explicitly via
// "나중에" or an approve/reject decision.
export function showPendingApprovalModal(o) {
  document.getElementById('pending-approval-ticker').textContent = o.name || o.ticker;
  document.getElementById('pending-approval-symbol').textContent = o.name ? displayTicker(o.ticker) : '';
  document.getElementById('pending-approval-order').innerHTML =
    `<span class="side-chip ${o.side}">${o.side === 'buy' ? t('buy') : t('sell')}</span> ` +
    `${o.shares.toLocaleString('en-US')}${t('sharesUnit')} · ${t('pendingRequestPrice')} ${fmtPrice(o.price, o.currency)}`;
  document.getElementById('pending-approval-reason').innerHTML =
    renderMarkdown(localizeDecisionLabels(enrichPriceFields(o.reason, o.price, o.currency)));

  const approveBtn = document.getElementById('pending-approval-approve-btn');
  const rejectBtn = document.getElementById('pending-approval-reject-btn');
  const laterBtn = document.getElementById('pending-approval-later-btn');
  const resultEl = document.getElementById('pending-approval-result');
  approveBtn.textContent = t('pendingApprovalConfirm');
  rejectBtn.textContent = t('pendingApprovalReject');
  laterBtn.textContent = t('pendingApprovalLater');
  approveBtn.disabled = false;
  rejectBtn.disabled = false;
  laterBtn.disabled = false;
  resultEl.hidden = true;
  approveBtn.onclick = () => resolvePendingApprovalModal(o.id, 'approve');
  rejectBtn.onclick = () => resolvePendingApprovalModal(o.id, 'reject');
  laterBtn.onclick = closePendingApprovalModal;

  document.getElementById('pending-approval-backdrop').classList.add('open');
}

export function closePendingApprovalModal() {
  document.getElementById('pending-approval-backdrop').classList.remove('open');
}

// Same approve/reject request as resolvePendingOrder, targeting the modal's
// own elements instead of a toast/tab row (the two surfaces are independent
// views of the same pending order, so the toast card and tab list still
// refresh normally via loadPendingOrders once this resolves).
export async function resolvePendingApprovalModal(id, action) {
  const approveBtn = document.getElementById('pending-approval-approve-btn');
  const rejectBtn = document.getElementById('pending-approval-reject-btn');
  const laterBtn = document.getElementById('pending-approval-later-btn');
  const resultEl = document.getElementById('pending-approval-result');
  approveBtn.disabled = true;
  rejectBtn.disabled = true;
  laterBtn.disabled = true;
  const busyBtn = action === 'approve' ? approveBtn : rejectBtn;
  const originalText = busyBtn.textContent;
  busyBtn.textContent = t('pendingApproving');
  try {
    const resp = await fetch(`/api/pending-orders/${id}/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(action === 'reject' ? { reason: '' } : {}),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
    resultEl.hidden = false;
    resultEl.className = 'order-result ok';
    resultEl.textContent = action === 'approve' ? t('pendingApproved', body.disposition || '') : t('pendingRejected');
    loadPendingOrders();
    setTimeout(closePendingApprovalModal, 1500);
  } catch (err) {
    resultEl.hidden = false;
    resultEl.className = 'order-result bad';
    resultEl.textContent = t('pendingActionFailed', err.message || String(err));
    approveBtn.disabled = false;
    rejectBtn.disabled = false;
    laterBtn.disabled = false;
    busyBtn.textContent = originalText;
  }
}

export let _lastAutoTradeStatus = null;
export let _lastDecisions = [];
// Independent from trading.js's page-wide connection banner on purpose
// (2026-08-18) — this endpoint and /api/trade-allowlist-info fail
// independently in practice (the auto-trade daemon can be down while
// quotes/balances/allowlist all serve fine), so a shared flag would either
// false-alarm the whole page over one dead endpoint or suppress this hero's
// own offline text when only this one is broken.
let _autoTradeStatusOffline = false;

export async function pollAutoTradeStatus() {
  const pill = document.getElementById('auto-trade-pill');
  const textEl = document.getElementById('auto-trade-pill-text');
  let s;
  try {
    s = await fetch('/api/auto-trade/status').then(r => r.json());
  } catch {
    // leave the pill/cards as they were — same convention as other pollers'
    // silent-fail — but the hero status line still needs to say SOMETHING
    // instead of sitting on stale or fallback text forever; that renders
    // inside _updateSignalsDashboardSub() itself (see its own check), not
    // here, so a later successful tick from a DIFFERENT caller (e.g.
    // pollDecisions' renderAnalyzingCards) can't overwrite this with a
    // stale-good status before the next auto-trade tick clears the flag.
    _autoTradeStatusOffline = true;
    _updateSignalsDashboardSub();
    return;
  }
  _autoTradeStatusOffline = false;
  _lastAutoTradeStatus = s;
  // Set before the `!enabled` return — it's a config constant, not live
  // state, and the portfolio table's stop-proximity highlight needs it in
  // manual mode too.
  setHeldReviewBufferPct(s?.held_review_buffer_pct);
  if (!s || !s.enabled) { pill.hidden = true; renderAnalyzingCards(); return; }
  pill.hidden = false;
  if (s.phase === 'analyzing' && s.current_tickers && s.current_tickers.length) {
    pill.className = 'auto-trade-pill analyzing';
    textEl.textContent = t('autoTradeAnalyzing', s.current_tickers.map(displayTicker).join(', '));
  } else if (s.phase === 'scanning') {
    pill.className = 'auto-trade-pill scanning';
    textEl.textContent = t('autoTradeScanning');
  } else {
    pill.className = 'auto-trade-pill idle';
    if (s.next_cycle_at) {
      const minsLeft = Math.max(0, Math.round((new Date(s.next_cycle_at) - new Date()) / 60000));
      textEl.textContent = t('autoTradeIdle', minsLeft);
    } else {
      textEl.textContent = t('autoTradeIdleNoTime');
    }
  }
  renderAnalyzingCards();
}

export function _todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

// Card per ticker in this cycle (queued -> analyzing -> resolved), content
// swapped in place rather than the card disappearing and a separate feed
// row appearing elsewhere — driven by the two pollers above, whichever
// fires last wins (both call this with whatever they most recently have).
// Cosmetic only — a stable-per-ticker color so repeat renders (poll every
// few seconds) don't shuffle avatar colors, same idea as any hash-based
// "pick a consistent color for this key" utility.
export const _AVATAR_COLORS = ['#2f6fed', '#d2691e', '#1a9e6a', '#7c3aed', '#c2410c', '#0891b2'];
export function _avatarColor(ticker) {
  let hash = 0;
  for (let i = 0; i < ticker.length; i++) hash = (hash * 31 + ticker.charCodeAt(i)) | 0;
  return _AVATAR_COLORS[Math.abs(hash) % _AVATAR_COLORS.length];
}

export function _signalCardName(ticker) {
  return state[ticker]?.name || ticker;
}

// decision_text is "**Rating**: ...\n\n**Executive Summary**: ...\n...", the
// same markdown render_portfolio_decision produces for the chat bubble —
// reflection (hindsight, "was this right") stays null until a LATER
// decision resolves this one, so it's almost never there yet on a card
// that just got made. Pull just the Executive Summary sentence out for the
// card's one-line reason instead.
export function _executiveSummaryFrom(decisionText) {
  if (!decisionText) return '';
  const m = decisionText.match(/\*\*Executive Summary\*\*:\s*([^\n]*)/);
  return m ? m[1].trim() : '';
}

// Short display "why" for a decision, used by both the AI 시그널 card and the
// 트레이딩 탭's compact AI 진단 box. Prefers the purpose-built card_reason in
// the current UI language, falls back to the other language (better than
// nothing), and only then to the old Executive-Summary-derived parse — that
// last one exists purely for rows that predate the card_reason_* columns.
// Returns '' when nothing is available (callers render nothing / an empty line).
export function _cardReasonFor(decision) {
  return (LANG === 'ko' ? decision.card_reason_ko : decision.card_reason_en)
    || (LANG === 'ko' ? decision.card_reason_en : decision.card_reason_ko)
    || _executiveSummaryFrom(decision.decision_text)
    || '';
}

// 'manual' = a human picked this exact ticker (the AI 종합분석 실행 button, a
// chat deep dive, a CLI run) rather than the scheduled auto-trade cycle.
// Anything that is not literally 'manual' — 'auto' and null (pre-column rows)
// — renders nothing, mirroring app.audit.stats's `IS DISTINCT FROM 'manual'`.
export function _isManualDecision(decision) {
  return decision?.decision_trigger === 'manual';
}

export function _manualBadgeHtml(decision) {
  if (!_isManualDecision(decision)) return '';
  return `<span class="signal-manual-badge" title="${escapeHtml(t('signalManualBadgeTitle'))}">${escapeHtml(t('signalManualBadge'))}</span>`;
}

// Confidence/Price Target for the card footer (목업의 "신뢰도 87% · 목표가
// 82,000원") — both already come through in decision_text's fixed-format
// fields (app.state.render_portfolio_decision), same source enrichPriceFields
// (trading.js) parses for the chat transcript's Price Target annotation.
export function _confidenceFrom(decisionText) {
  if (!decisionText) return null;
  const m = decisionText.match(/\*\*Confidence\*\*:\s*(\d+)%/);
  return m ? m[1] : null;
}
export function _priceTargetFrom(decisionText) {
  if (!decisionText) return null;
  const m = decisionText.match(/\*\*Price Target\*\*:\s*([\d.]+)/);
  return m ? parseFloat(m[1]) : null;
}

// The '[EXECUTION NOTE: ...]' lines app.memory.store.append_execution_note
// appends to decision_text at execution time — the code-level OUTCOME (a risk
// cap, a floor-to-0 target, a reversed order), NOT the AI's own reasoning
// (_cardReasonFor). Server-generated English; the card labels it with a
// localized prefix rather than translating the body (there are already
// several note shapes — cap/stop/direction-guard/rating-action-gap — and
// string-matching them in the frontend would break on the next one). No
// adjustment_note ever contains ']' (see ledger.RATING_ACTION_GAP_NOTE_PREFIX).
export function _executionNotesFrom(decisionText) {
  if (!decisionText) return '';
  const out = [];
  const re = /\[EXECUTION NOTE:\s*([^\]]*)\]/g;
  let m;
  while ((m = re.exec(decisionText)) !== null) out.push(m[1].trim());
  return out.join(' · ');
}

export let _reanalyzeCooldownUntil = 0;  // epoch ms; 0 = no cooldown active

export function renderReanalyzeButton(s) {
  const btn = document.getElementById('signal-reanalyze-btn');
  if (!btn) return;
  if (!s || s.execution_mode !== 'semi_auto') { btn.hidden = true; return; }
  btn.hidden = false;
  // Survive a page refresh: if the server says the last cycle finished
  // recently enough that the cooldown hasn't elapsed yet, reconstruct it
  // client-side instead of showing the button as clickable right away.
  if (s.cycle_finished_at && s.manual_trigger_cooldown_seconds) {
    const untilMs = new Date(s.cycle_finished_at).getTime() + s.manual_trigger_cooldown_seconds * 1000;
    if (untilMs > _reanalyzeCooldownUntil) _reanalyzeCooldownUntil = untilMs;
  }
  const remainingMs = _reanalyzeCooldownUntil - Date.now();
  const running = s.phase === 'scanning' || s.phase === 'analyzing';
  if (running) {
    btn.disabled = true;
    btn.textContent = t('signalReanalyzeRunning');
  } else if (remainingMs > 0) {
    btn.disabled = true;
    btn.textContent = t('signalReanalyzeCooldown', Math.ceil(remainingMs / 1000));
  } else {
    btn.disabled = false;
    btn.textContent = t('signalReanalyzeBtn');
  }
}

export async function triggerReanalyze() {
  try {
    const res = await fetch('/api/auto-trade/analyze-now', { method: 'POST' });
    if (res.status === 429) {
      const body = await res.json().catch(() => ({}));
      showToast(t('signalReanalyzeBtn'), body.detail || t('signalReanalyzeCooldown', 0), 'bad');
      return;
    }
    // Optimistic local cooldown — pollAutoTradeStatus() will reconcile with
    // the server's own cycle_finished_at once this run actually completes.
    if (_lastAutoTradeStatus?.manual_trigger_cooldown_seconds) {
      _reanalyzeCooldownUntil = Date.now() + _lastAutoTradeStatus.manual_trigger_cooldown_seconds * 1000;
    }
    pollAutoTradeStatus();
  } catch { /* silent — same convention as the other pollers here */ }
}

export function _buildResolvedSignalCard(decision) {
  const avatarLetter = (decision.name || decision.ticker).trim().charAt(0).toUpperCase();
  const avatarHtml = `<div class="signal-avatar" style="background:${_avatarColor(decision.ticker)}">${escapeHtml(avatarLetter)}</div>`;
  const ratingLabel = LANG === 'ko' ? (_RATING_KO[decision.rating] || decision.rating) : decision.rating;
  const confidence = _confidenceFrom(decision.decision_text);
  const priceTarget = _priceTargetFrom(decision.decision_text);
  const currency = isDomesticSymbol(decision.ticker) ? 'KRW' : 'USD';
  const footerParts = [];
  if (confidence) footerParts.push(`<div><span class="signal-card-stat-label">${escapeHtml(t('signalConfidenceLabel'))}</span><span class="signal-card-stat-value">${confidence}%</span></div>`);
  if (priceTarget) footerParts.push(`<div><span class="signal-card-stat-label">${escapeHtml(t('signalPriceTargetLabel'))}</span><span class="signal-card-stat-value">${fmtPrice(priceTarget, currency)}</span></div>`);
  // Primary pill = decision.action_label (Buy/Sell/Increase/Decrease/Hold,
  // computed from current-vs-target shares — see _ACTION_KO's docstring) —
  // what the rebalance ACTUALLY did. Falls back to the AI's own rating pill
  // for decisions from before this column existed (action_label null).
  // The rating hint underneath only shows when it says something the pill
  // doesn't (_ratingDiffersFromAction) — live data: 93% of the time rating
  // and action_label are the same underlying value just translated by two
  // different tables (Hold/유지 vs Hold/보유), so showing both read as a
  // contradiction. The remaining 7% is exactly "AI said 비중 축소 but it
  // bought" (a risk cap/gate blocked the AI's call) — that stays visible,
  // not hidden, since it's a safety-relevant mismatch.
  const hasAction = !!decision.action_label;
  const showRatingHint = _ratingDiffersFromAction(decision.rating, decision.action_label);
  // When the pill (what executed) disagrees with the AI's rating, show WHY —
  // the persisted execution note, not the AI's reasoning (2026-09-01 audit
  // #10: 5 of 6 "AI said Buy, nothing bought" decisions had no visible reason).
  const executionNote = showRatingHint ? _executionNotesFrom(decision.decision_text) : '';
  const pillClass = hasAction ? (_ACTION_PILL_CLASS[decision.action_label] || 'hold') : decision.rating.toLowerCase();
  const pillLabel = hasAction
    ? (LANG === 'ko' ? (_ACTION_KO[decision.action_label] || decision.action_label) : decision.action_label)
    : ratingLabel;
  const card = document.createElement('div');
  card.className = 'signal-card';
  card.innerHTML = `
    <div class="signal-card-top">
      <div class="signal-card-id">${avatarHtml}<div class="signal-card-id-text"><div class="name">${escapeHtml(decision.name)}</div><div class="symbol">${escapeHtml(displayTicker(decision.ticker))}</div></div></div>
      <div style="display:flex; flex-direction:column; align-items:flex-end; gap:0.2rem;">
        <span class="signal-rating-pill ${pillClass}">${escapeHtml(pillLabel)}</span>
        ${showRatingHint ? `<span class="signal-rating-hint">${escapeHtml(t('signalAiRatingHint'))}: ${escapeHtml(ratingLabel)}</span>` : ''}
        ${_manualBadgeHtml(decision)}
      </div>
    </div>
    ${executionNote ? `<div class="signal-execution-note" title="${escapeHtml(executionNote)}">${escapeHtml(t('signalExecutionNoteLabel'))}: ${escapeHtml(executionNote)}</div>` : ''}
    ${(() => {
      const blurb = _cardReasonFor(decision);
      if (!blurb) return '';
      return `<div class="reason">${escapeHtml(blurb)}</div>`;
    })()}
    ${footerParts.length ? `<div class="signal-card-stats">${footerParts.join('')}</div>` : ''}
    ${decision.total_tokens ? `<div class="signal-card-tokens">${escapeHtml(t('signalTokensUsed', decision.total_tokens))}</div>` : ''}
  `;
  return card;
}

// Only ever built for the ticker actively being analyzed right now — a
// "대기 중" (queued, not yet started) card is deliberately never shown
// (2026-07-22: the user found queued placeholders just noise), so this
// has no active/inactive variant.
export function _buildAnalyzingPlaceholderCard(ticker, stageLabel, resolvedName) {
  // `resolvedName`, when given, comes from /api/auto-trade/status's
  // current_ticker_names — covers a screened/momentum candidate that was
  // never on the trade allowlist (so state[ticker] below is empty and
  // _signalCardName would otherwise fall back to the bare ticker code for
  // the whole time it's analyzing). Falls back to the trade-allowlist-scoped
  // state cache, then the bare ticker, same as before this param existed.
  const name = resolvedName || _signalCardName(ticker);
  const avatarLetter = (name || ticker).trim().charAt(0).toUpperCase();
  const avatarHtml = `<div class="signal-avatar" style="background:${_avatarColor(ticker)}">${escapeHtml(avatarLetter)}</div>`;
  // -1 (falsy/unmatched stageLabel) degrades to just the badge above, no
  // dots lit and no caption — covers older cached status payloads that
  // predate current_stage, and a raw label the matcher doesn't recognize.
  const stageIndex = stageLabel ? matchAnalyzePipelineStage(stageLabel) : -1;
  const dotsHtml = ANALYZE_PIPELINE_STAGES.map((stage, i) => {
    const cls = stageIndex >= 0 && i < stageIndex ? 'done' : stageIndex >= 0 && i === stageIndex ? 'active' : '';
    return `<span class="card-pipeline-step ${cls}" aria-hidden="true"></span>`;
  }).join('');
  const captionText = stageIndex >= 0 ? t(ANALYZE_PIPELINE_STAGES[stageIndex].labelKey) : '';
  const card = document.createElement('div');
  card.className = 'signal-card';
  card.innerHTML = `
    <div class="signal-card-top">
      <div class="signal-card-id">${avatarHtml}<div class="signal-card-id-text"><div class="name">${escapeHtml(name)}</div><div class="symbol">${escapeHtml(displayTicker(ticker))}</div></div></div>
      <span class="signal-status-badge"><span class="dot"></span>${escapeHtml(t('autoTradeCardAnalyzing'))}</span>
    </div>
    <div class="card-pipeline"><div class="card-pipeline-dots">${dotsHtml}</div>${captionText ? `<div class="card-pipeline-caption">${escapeHtml(captionText)}</div>` : ''}</div>
  `;
  return card;
}

export const SIGNAL_CARDS_MAX = 10;
// 수동 분석(사람이 종목을 지정해 돌린 판단)은 자동 사이클 판단과 정원을 공유하지
// 않음 — 예전엔 하나의 10칸을 두고 경쟁해서, 어떤 종목이 걱정돼 한 번 눌러본
// 수동 재분석이 아직 유효한 자동 판단 카드를 조용히 밀어냈음.
export const SIGNAL_CARDS_MANUAL_MAX = 3;

// Left-to-right = most-recent-first: the ticker actively analyzing right
// now (if any) is leftmost — that's the newest thing happening — followed
// by past decisions newest-first (already that order from /api/decisions),
// deduped by ticker. Auto decisions capped at SIGNAL_CARDS_MAX, manual ones
// at SIGNAL_CARDS_MANUAL_MAX (separate row, see renderAnalyzingCards) — both
// dropped entirely past their own market's `cards_reset_at_domestic`/
// `cards_reset_at_overseas` (KRX/US open - 30min respectively, split
// 2026-08-20 — a single shared KRX-based boundary used to wipe overseas
// cards hours before the US market opened) so history doesn't show
// yesterday's leftovers forever — accumulates within a session, resets each
// market's own morning.
/* ---------------- market overview widget (Toss-증권 style, 2026-08-19) ----------------
   AI 시그널 탭 랜딩 히어로 아래 남는 공간을 채우는 5개 미니카드(코스피/코스닥/
   원달러/나스닥100/VIX) — 현재가·전일대비·스파크라인. 데이터는
   /api/market-widget(부분실패 허용, 실패한 항목은 그냥 빠짐)로 pollMarketWidget이
   채워 넣음(trading.js). */
const MARKET_WIDGET_ORDER = ['kospi', 'kosdaq', 'usdkrw', 'nasdaq', 'vix'];
const MARKET_WIDGET_LABEL_KEY = {
  kospi: 'marketWidgetKospi', kosdaq: 'marketWidgetKosdaq', usdkrw: 'marketWidgetUsdKrw',
  nasdaq: 'marketWidgetNasdaq', vix: 'marketWidgetVix',
};

export function renderMarketWidget(items) {
  const container = document.getElementById('market-widget');
  if (!container) return;
  const byKey = new Map((items || []).map(it => [it.key, it]));
  const present = MARKET_WIDGET_ORDER.filter(k => byKey.has(k));
  if (!present.length) { container.hidden = true; container.innerHTML = ''; return; }

  container.innerHTML = present.map(key => {
    const item = byKey.get(key);
    const up = item.change_pct >= 0;
    const valueStr = item.value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const chgStr = `${up ? '+' : ''}${item.change_pct.toFixed(2)}%`;
    return `<div class="mw-card">`
      + `<div class="mw-name">${escapeHtml(t(MARKET_WIDGET_LABEL_KEY[key]))}</div>`
      + `<div class="mw-value">${valueStr}</div>`
      + `<div class="mw-change ${up ? 'up' : 'down'}">${chgStr}</div>`
      + `<canvas class="mw-spark" id="mw-spark-${key}"></canvas>`
      + `</div>`;
  }).join('');
  container.hidden = false;

  // DOM부터 채운 뒤 캔버스를 그림 — clientWidth/Height를 읽으려면 이미
  // 삽입돼 있어야 함(openDetail/loadPeriodChart와 같은 2단계 순서).
  for (const key of present) {
    const item = byKey.get(key);
    const canvas = document.getElementById(`mw-spark-${key}`);
    if (canvas) drawSparkline(canvas, item.sparkline, item.change_pct >= 0);
  }
}

export function renderAnalyzingCards() {
  const container = document.getElementById('analyzing-cards');
  const section = document.getElementById('signal-section');
  const manualGroup = document.getElementById('signal-manual-group');
  const manualContainer = document.getElementById('analyzing-cards-manual');
  if (!container || !section) return;
  const s = _lastAutoTradeStatus;
  renderReanalyzeButton(s);
  if (!s || !s.enabled) {
    section.hidden = true;
    container.innerHTML = '';
    if (manualContainer) manualContainer.innerHTML = '';
    if (manualGroup) manualGroup.hidden = true;
    container.classList.remove('has-manual-group');
    _updateSignalsBadge(0);
    _updateSignalsDashboardSub();
    return;
  }
  section.hidden = false;
  // Per-market reset boundary — falls back to the legacy shared field for
  // a stale cached frontend build talking to an old backend that hasn't
  // been redeployed yet (or vice versa); either way this only ever means
  // "show extra old cards," never "hide fresh ones."
  const _mkResetAt = (v) => (v ? new Date(v) : null);
  const resetAtDomestic = _mkResetAt(s.cards_reset_at_domestic || s.cards_reset_at);
  const resetAtOverseas = _mkResetAt(s.cards_reset_at_overseas || s.cards_reset_at);
  const resetAtFor = (ticker) => (isDomesticSymbol(ticker) ? resetAtDomestic : resetAtOverseas);
  // shown/scope/reset filtering stays GLOBAL and identical for both groups —
  // only the cap is per-group. One ticker => one card, in whichever group its
  // newest decision belongs (_lastDecisions is newest-first from the backend,
  // so the first one encountered wins).
  const shown = new Set();
  const autoCards = [];
  const manualCards = [];
  // 국내/해외 scope (global, top header) — same filter the 트레이딩 tab's
  // trade allowlist applies, so a card never shows for a ticker outside scope.
  const inScope = (ticker) => (_marketScope === 'domestic') === isDomesticSymbol(ticker);
  // The analyzing placeholder is always auto (only the auto-trade loop ever
  // populates current_tickers) and therefore consumes an AUTO slot — it is an
  // in-flight auto judgment, and leaving it uncapped would let the row grow
  // past 10 whenever several tickers analyze at once.
  if (s.phase === 'analyzing' && s.current_tickers && s.current_tickers.length) {
    const today = _todayStr();
    for (const ticker of s.current_tickers) {
      if (autoCards.length >= SIGNAL_CARDS_MAX) break;
      if (!inScope(ticker) || shown.has(ticker)) continue;
      const already = _lastDecisions.find(d => d.ticker === ticker && d.trade_date === today);
      if (!already) {
        autoCards.push(_buildAnalyzingPlaceholderCard(
          ticker, s.current_stage && s.current_stage[ticker], s.current_ticker_names && s.current_ticker_names[ticker]
        ));
        shown.add(ticker);
      }
    }
  }
  for (const decision of _lastDecisions) {
    if (autoCards.length >= SIGNAL_CARDS_MAX && manualCards.length >= SIGNAL_CARDS_MANUAL_MAX) break;
    if (shown.has(decision.ticker)) continue;
    if (!inScope(decision.ticker)) continue;
    const cutoff = resetAtFor(decision.ticker);
    if (cutoff && decision.created_at && new Date(decision.created_at) < cutoff) continue;
    const bucket = _isManualDecision(decision) ? manualCards : autoCards;
    const cap = bucket === manualCards ? SIGNAL_CARDS_MANUAL_MAX : SIGNAL_CARDS_MAX;
    if (bucket.length >= cap) continue;   // continue, NOT break — the other group may still have room
    shown.add(decision.ticker);
    bucket.push(_buildResolvedSignalCard(decision));
  }
  container.innerHTML = '';
  for (const card of autoCards) container.appendChild(card);
  if (manualContainer && manualGroup) {
    manualContainer.innerHTML = '';
    for (const card of manualCards) manualContainer.appendChild(card);
    manualGroup.hidden = manualCards.length === 0;
  }
  container.classList.toggle('has-manual-group', manualCards.length > 0);
  // Both groups — this badge (and the header notification dot that reuses it)
  // means "cards visible right now"; auto-only would contradict the screen.
  _updateSignalsBadge(autoCards.length + manualCards.length);
  _updateSignalsDashboardSub();
}

// 히어로 상태줄 — 예전엔 "렌더링된 지금 시각"을 찍어서(new Date().toLocaleTimeString())
// 몇 시간째 분석이 없어도 8초 폴링마다 시계가 도는 거짓말이었음. /api/auto-trade/status가
// 실제 사이클 시각(cycle_finished_at)/분석 종목 수(tickers_this_cycle, 문자열 배열 —
// 개수 아님)를 이미 주므로 그걸 그대로 씀. renderAlerts(같은 파일)처럼 Asia/Seoul을
// 명시 고정 — 브라우저 로컬 타임존과 무관하게 항상 한국시간으로 보여야 함.
function _updateSignalsDashboardSub() {
  const el = document.getElementById('signal-dashboard-sub');
  if (!el) return;
  // Checked BEFORE the manual/enabled branch below (2026-08-18) — otherwise
  // a failed fetch would render as "수동 모드" instead of "connection lost",
  // which is actively misleading. Also guards against a later successful
  // renderAnalyzingCards() call from a DIFFERENT poller (e.g. pollDecisions)
  // overwriting this with a stale-good status before the next auto-trade
  // tick clears the offline flag.
  if (_autoTradeStatusOffline) { el.textContent = t('deskStatusOffline'); return; }
  const s = _lastAutoTradeStatus;
  if (!s || !s.enabled) { el.textContent = t('deskStatusManual'); return; }
  if (s.phase === 'analyzing') {
    el.textContent = t('deskStatusAnalyzing', (s.current_tickers || []).length);
    return;
  }
  if (s.phase === 'scanning') { el.textContent = t('deskStatusScanning'); return; }
  // idle
  let text;
  if (s.cycle_finished_at) {
    const time = new Date(s.cycle_finished_at).toLocaleTimeString(dateLocale(), { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit' });
    text = t('deskStatusDone', time, (s.tickers_this_cycle || []).length);
  } else {
    text = t('deskStatusNeverRan');
  }
  if (s.next_cycle_at && new Date(s.next_cycle_at) > new Date()) {
    const mins = Math.max(0, Math.round((new Date(s.next_cycle_at) - new Date()) / 60000));
    text += ` · ${t('deskStatusNextCycle', mins)}`;
  }
  el.textContent = text;
}

// 승인대기 건수 CTA(#desk-hero-pending) — _pendingOrdersCache는 renderPendingOrders가
// 갱신할 때마다 최신이므로, 그 갱신 직후에 호출돼서(renderPendingOrders 안에서) 카운트가
// 묵지 않게 함. 카운트는 CTA가 열어주는 목적지(승인대기 목록)와 같은 스코프여야 하므로
// unscoped 캐시 길이가 아니라 _scopedByMarket을 거친 값을 씀.
function _renderHeroPending() {
  const btn = document.getElementById('desk-hero-pending');
  if (!btn) return;
  const count = _scopedByMarket(_pendingOrdersCache).length;
  if (!count) { btn.hidden = true; return; }
  btn.hidden = false;
  btn.textContent = t('deskPendingCta', count);
}

// 사이드바 "AI 시그널" 탭의 알림뱃지 — 지금 보이는 카드 수 그대로 (필터링 이후).
// 헤더 알림벨의 안읽음 점도 같은 카운트를 재사용(별도 알림 데이터 없음).
function _updateSignalsBadge(count) {
  const badge = document.getElementById('sidebar-signals-badge');
  if (badge) {
    badge.hidden = count === 0;
    badge.textContent = String(count);
  }
  const notifDot = document.getElementById('header-notif-dot');
  if (notifDot) notifDot.hidden = count === 0;
}

