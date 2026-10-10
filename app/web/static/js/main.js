import { _tradeAllowlistSubTab, setTradeAllowlistSubTab, t } from './state.js';
import { _feedCache, _pendingOrdersCache, closePendingApprovalModal, loadPendingOrders, pollAlerts, pollAutoTradeStatus, pollDecisions, renderAnalyzingCards, renderFeed, renderPendingOrderHistory, renderPendingOrders, startStream, triggerReanalyze } from './signals.js';
import { _accuracyFilters, addTickerFromSearch, applyTradingScope, currentDetailSymbol, drawEpsChart, drawRevChart, ensureCardsBuilt, handleGlobalSearch, hideTradeFillModal, initOrderTicket, loadAccuracy, loadAccuracyWeekly, loadKrInsight, loadOrderbook, loadPeriodChart, loadPnl, loadRank, loadSafetyStatus, pollDashboardSummary, pollFills, pollMarketOps, pollMarketStatus, pollMarketWidget, pruneStaleTradeAllowlistCards, redrawPeriodChart, refreshTradeAllowlistDisplayInfo, renderPnl, runAnalyze, setConnectionBanner, showToast, togglePortfolioSort, wireEarningsHover, wirePeriodChartHover } from './trading.js';
import { attachTickerSearch } from './tickersearch.js';
import { sendChat, startNewChat } from './assistant.js';
import { ensureReady } from './onboarding.js';

document.getElementById('chat-send').addEventListener('click', sendChat);
document.getElementById('chat-new').addEventListener('click', startNewChat);
startNewChat();  // paint the suggested-question empty state before any chat happens

// AI-analysis button, Korean-data tabs, and trade allowlist add — wired once.
document.getElementById('detail-analyze').addEventListener('click', runAnalyze);
document.querySelectorAll('#kr-tabs .kr-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('#kr-tabs .kr-tab').forEach(t => t.classList.toggle('active', t === tab));
    if (currentDetailSymbol) loadKrInsight(currentDetailSymbol, tab.dataset.kr);
  });
});
document.getElementById('chat-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat(); });

// 관심종목 패널 추가 입력창 — 이름/코드 타입어헤드(tickersearch.js) + Enter/버튼 폴백.
const tradeAllowlistAddInput = document.getElementById('trade-allowlist-add-input');
const tradeAllowlistAddBtn = document.getElementById('trade-allowlist-add-btn');
// 타입어헤드를 먼저 붙인다 — 드롭다운에 활성 행이 있을 때 Enter를
// stopImmediatePropagation로 선점하므로 아래 폴백 keydown보다 리스너가 앞서야 함.
attachTickerSearch(tradeAllowlistAddInput, { onPick: (row) => addTickerFromSearch(row) });
async function submitTradeAllowlistAdd() {
  const query = tradeAllowlistAddInput.value;
  const found = await handleGlobalSearch(query);
  if (found) {
    tradeAllowlistAddInput.value = '';
  } else if (query.trim()) {
    showToast(t('searchNotFoundTitle'), t('searchNotFoundSub', query.trim()), 'rejected');
  }
}
tradeAllowlistAddInput.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  e.preventDefault();
  submitTradeAllowlistAdd();
});
tradeAllowlistAddBtn.addEventListener('click', submitTradeAllowlistAdd);
window.addEventListener('resize', () => {
  if (currentDetailSymbol) {
    redrawPeriodChart();
    drawEpsChart(currentDetailSymbol);
    drawRevChart(currentDetailSymbol);
  }
});

document.getElementById('pending-approval-backdrop').addEventListener('click', (e) => {
  if (e.target.id === 'pending-approval-backdrop') closePendingApprovalModal();
});
document.getElementById('pending-approval-modal').addEventListener('click', (e) => e.stopPropagation());
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('pending-approval-backdrop').classList.contains('open')) {
    closePendingApprovalModal();
  }
});
wireEarningsHover();
wirePeriodChartHover();

// 실적추이 섹션은 기본 접힘(<details>)인데, 접힌 동안 canvas.clientWidth가 0이라
// openDetail 시점에 그리면 0 크기로 그려지고 펼쳐도 그대로 남는다 — 펼칠 때 다시 그린다.
// 바로 위 resize 핸들러와 완전히 같은 방식으로 캐시(earningsTrendState)에서
// 재렌더하므로 재조회는 없다.
document.getElementById('detail-earnings-wrap').addEventListener('toggle', (e) => {
  if (e.currentTarget.open && currentDetailSymbol) {
    drawEpsChart(currentDetailSymbol);
    drawRevChart(currentDetailSymbol);
  }
});

// ---------------- runtime settings: AI profile / trading mode (설정 탭) ----------------
// Trading Environment (paper/real) is deliberately read-only here — see the
// .seg-badge markup's own comment and docs/safety-and-risk.md 5.1.

// Detail tabs: one panel visible at a time; renderers keep polling either
// way, so switching tabs never shows stale content.
document.querySelectorAll('#trading-subtabs .tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('#trading-subtabs .tab').forEach(t => t.classList.toggle('active', t === tab));
    document.querySelectorAll('.tab-panel').forEach(p => { p.hidden = p.id !== tab.dataset.tab; });
    // 매매손익 탭은 다른 탭들과 달리 상시 폴링 대상이 아님(기간 지정 조회라 매번
    // 자동 재조회할 이유가 없음) — 처음 열 때 1회만 기본 기간(최근 30일)으로 로드.
    if (tab.dataset.tab === 'tab-pnl' && !tab.dataset.pnlLoaded) {
      tab.dataset.pnlLoaded = '1';
      loadPnl();
    }
    // 적중률 탭도 같은 이유로 상시 폴링하지 않음 — 감사 집계는 하루 단위로도
    // 거의 안 변하고 쿼리가 무겁다. 처음 열 때 1회, 이후엔 필터를 바꿀 때만 재조회.
    if (tab.dataset.tab === 'tab-accuracy' && !tab.dataset.accuracyLoaded) {
      tab.dataset.accuracyLoaded = '1';
      loadAccuracy();
      loadAccuracyWeekly();
    }
  });
});

// Sidebar nav: same one-panel-visible-at-a-time pattern as trading-subtabs above.
// 3 content panels (nav-signals/nav-trading/nav-assistant) + nav-settings.
function switchSidebarTab(navtab) {
  document.querySelectorAll('#sidebar-nav .sidebar-tab').forEach(t => t.classList.toggle('active', t.dataset.navtab === navtab));
  document.querySelectorAll('.nav-panel').forEach(p => { p.hidden = p.id !== navtab; });
}
document.querySelectorAll('#sidebar-nav .sidebar-tab').forEach(tab => {
  tab.addEventListener('click', () => switchSidebarTab(tab.dataset.navtab));
});
// 계좌 요약의 "상세 ›" — 계좌 상세 수치(#portfolio-summary)가 있는
// 트레이딩 → 포트폴리오 상세로 이동(2026-08-18, 트레이딩 탭 히어로 제거에 따라
// 목적지 변경). 아래 desk-hero-pending과 같은 방식으로 기존 탭 클릭
// 핸들러를 재사용 — 뱃지 갱신/패널 토글을 중복 구현하지 않기 위함.
document.getElementById('sidebar-acct-detail-link').addEventListener('click', () => {
  document.querySelector('.sidebar-tab[data-navtab="nav-trading"]').click();
  document.querySelector('#trading-subtabs .tab[data-tab="tab-portfolio"]').click();
});
// 헤더 알림벨 — AI 시그널 탭으로 이동(뱃지가 그 탭의 카드 수를 그대로 반영).
document.getElementById('header-notif-btn').addEventListener('click', () => switchSidebarTab('nav-signals'));
// 히어로의 "승인 대기 N건" CTA — 트레이딩탭의 승인대기 서브탭으로 이동. 새 export를
// 만드는 대신 이미 있는 사이드바탭/서브탭 클릭 핸들러를 그대로 재사용(클릭 시
// 뱃지 갱신·서브탭 활성화 등 그 핸들러들이 이미 하는 일을 중복 구현하지 않기 위함).
document.getElementById('desk-hero-pending').addEventListener('click', () => {
  document.querySelector('.sidebar-tab[data-navtab="nav-trading"]').click();
  document.querySelector('#trading-subtabs .tab[data-tab="tab-pending"]').click();
});

// 사이드바 접기/펼치기 — 아이콘만 남는 좁은 레일로. 테마처럼 localStorage에 저장.
const sidebarNav = document.getElementById('sidebar-nav');
const sidebarCollapseBtn = document.getElementById('sidebar-collapse-btn');
function applySidebarCollapsed(collapsed) {
  sidebarNav.classList.toggle('collapsed', collapsed);
  sidebarCollapseBtn.querySelector('.sidebar-collapse-label').textContent = collapsed ? t('sidebarExpand') : t('sidebarCollapse');
  localStorage.setItem('sidebarCollapsed', collapsed ? '1' : '0');
}
sidebarCollapseBtn.addEventListener('click', () => applySidebarCollapsed(!sidebarNav.classList.contains('collapsed')));
applySidebarCollapsed(localStorage.getItem('sidebarCollapsed') === '1');
// 승인대기/최근거래/에이전트 활동/포트폴리오 상세(트레이딩·AI 시그널 탭 서브패널)도
// 같은 전역 스코프를 따름(2026-07-30) — trading.js/signals.js 사이 순환 import를
// 피하려고 이미 폴링된 캐시를 여기서 다시 그리는 자리로 둠.
function refreshOtherTabScope() {
  renderFeed(_feedCache.decisions, _feedCache.tradeByDecisionId);
  renderPendingOrders(_pendingOrdersCache);
  renderPendingOrderHistory();
  renderPnl();  // client-side re-filter only — already-fetched rows, no new request
}
document.getElementById('pnl-query-btn').addEventListener('click', loadPnl);
// 적중률 탭 필터 3종 — 위임(#rank-modes 와 같은 방식). 같은 값을 다시 누르면
// 재조회하지 않는다. 사이드바 전역 스코프(refreshOtherTabScope)에는 일부러
// 넣지 않음 — 이 패널은 자기 시장 필터를 따로 갖고 있고 기본값이 "전체"다.
[['accuracy-market', 'market'], ['accuracy-scope', 'scope'], ['accuracy-basis', 'basis']].forEach(([id, key]) => {
  document.getElementById(id).addEventListener('click', e => {
    const btn = e.target.closest('.seg-btn');
    if (!btn || _accuracyFilters[key] === btn.dataset.value) return;
    _accuracyFilters[key] = btn.dataset.value;
    loadAccuracy();
    // 주간추이/F7 은 /api/audit/weekly 에서 오고 그 엔드포인트는 market 을 받지
    // 않는다 — 시장만 바꿨을 때 재조회하면 완전히 같은 응답을 받으려고 무거운
    // 쿼리(주간 승률 3회 + 확신도 + F7)를 다시 돌리는 셈이다. 대신 각 표의
    // 각주가 "전체 시장 기준"임을 알린다(renderAccuracyWeekly).
    if (key !== 'market') loadAccuracyWeekly();
  });
});
document.querySelectorAll('#market-scope-toggle .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    applyTradingScope(btn.dataset.scope);  // sets the shared scope + refilters the trade allowlist
    renderAnalyzingCards();  // re-renders AI 시그널 cards against the new scope
    refreshOtherTabScope();
  });
});
// 관심종목/실시간/최근 본 서브탭 (목업 매칭).
document.querySelectorAll('#trade-allowlist-subtabs .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    setTradeAllowlistSubTab(btn.dataset.subtab);
    applyTradingScope();  // no scope change — just re-filters + re-selects if needed
  });
});
setTradeAllowlistSubTab(_tradeAllowlistSubTab);  // sync the active button with the saved preference on load
document.getElementById('rank-refresh').addEventListener('click', () => loadRank());
document.getElementById('rank-modes').addEventListener('click', e => { const b = e.target.closest('.rank-mode'); if (b) loadRank(b.dataset.mode); });
document.getElementById('period-modes').addEventListener('click', e => {
  const b = e.target.closest('[data-period]');
  if (b && currentDetailSymbol) loadPeriodChart(currentDetailSymbol, b.dataset.period);
});
document.getElementById('signal-reanalyze-btn').addEventListener('click', triggerReanalyze);
document.getElementById('portfolio-head').addEventListener('click', e => {
  const th = e.target.closest('th.sortable');
  if (th) togglePortfolioSort(th.dataset.sortKey);
});
document.getElementById('trade-fill-modal-close').addEventListener('click', hideTradeFillModal);
document.getElementById('trade-fill-modal-backdrop').addEventListener('click', hideTradeFillModal);

// Login + KIS credential onboarding must clear BEFORE anything below runs
// (multi-user migration Phase 3). buildCards() (via ensureCardsBuilt(), see
// trading.js) and the ~15 polls it kicks off all need an authenticated
// session with stored credentials; firing them first paints an empty
// dashboard out of 401s and 412s, which reads as a broken account rather
// than an unfinished setup. Resolves immediately, rendering nothing, when
// the visitor is already set up.
// 로그인/자격증명 설정이 끝나기 전에는 대시보드를 띄우지 않음 — 먼저 띄우면
// 폴링이 전부 실패해 '설정 미완료'가 '계좌 오류'처럼 보임.
//
// ensureCardsBuilt() never rejects (2026-08-18) — a dead backend at page-load
// used to reject buildCards() here with nothing downstream ever catching it,
// which meant every setInterval in this block never got registered at all,
// so even a backend that came back later was never polled again without a
// manual reload. The trailing .catch() below is a belt-and-braces guard for
// anything else in this block throwing synchronously (e.g. a missing DOM
// element) — it should never fire on ensureCardsBuilt()'s own account.
ensureReady().then(() => ensureCardsBuilt()).then(() => {
  startStream();
  pollAlerts();
  pollDashboardSummary();
  pollMarketWidget();
  pollDecisions();
  loadPendingOrders();
  // 트레이딩 탭 하단 순위표 서브탭 안에 있어 처음엔 숨겨져 있지만(2026-08-18), 그 아래
  // 60초 setInterval과 함께 탭 가시성과 무관하게 계속 폴링 — 탭을 열었을 때
  // 이미 최신 데이터가 준비돼 있도록 시작 시점에도 미리 로드해 둔다.
  loadRank();
  initOrderTicket();
  applyTradingScope();  // filters trade allowlist rows + auto-selects the first ticker in scope — after initOrderTicket() so the #order-ticker select already has options to sync against
  pollMarketOps();
  pollMarketStatus();  // header's 장중 pill + per-symbol badges shouldn't sit blank for the first 60s
  pollFills();
  pollAutoTradeStatus();
  setInterval(pollAlerts, 15000);
  setInterval(pollMarketStatus, 60000);
  setInterval(pollMarketOps, 20000);
  setInterval(pollFills, 5000);
  setInterval(pollAutoTradeStatus, 8000);
  // Was 4 separate timers (pollIndex/pollPortfolio/pollTrades/loadOpenOrders,
  // each fetching its own endpoint) — now one call to /api/dashboard-summary
  // feeding all four renderers. See trading.js's pollDashboardSummary.
  setInterval(pollDashboardSummary, 30000);
  setInterval(pollMarketWidget, 30000);
  setInterval(pollDecisions, 60000);
  setInterval(loadPendingOrders, 30000);
  setInterval(loadSafetyStatus, 30000);
  setInterval(pruneStaleTradeAllowlistCards, 30000);
  setInterval(() => loadRank(), 60000);
  // 서버 시작 직후엔 이름/전일종가가 아직 백그라운드에서 채워지는 중일 수 있음
  // (app/web/state.py의 _prefetch_startup_data — 이제 lifespan을 안 막음).
  // 다 채워질 때까지(또는 최대 15회=30초, 실패한 종목이 있어도 무한폴링 방지)
  // 2초 간격으로 재조회.
  let displayInfoRefreshCount = 0;
  const displayInfoRefreshTimer = setInterval(async () => {
    displayInfoRefreshCount++;
    const stillLoading = await refreshTradeAllowlistDisplayInfo();
    if (!stillLoading || displayInfoRefreshCount >= 15) clearInterval(displayInfoRefreshTimer);
  }, 2000);
  // keep the open detail modal's live orderbook fresh while it's open
  setInterval(() => {
    if (currentDetailSymbol && currentDetailSymbol.endsWith('.KS')) loadOrderbook(currentDetailSymbol);
  }, 3000);
}).catch(err => {
  console.error('dashboard boot failed', err);
  setConnectionBanner(true);
});
