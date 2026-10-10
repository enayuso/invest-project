import { _marketScope, displayTicker, escapeHtml, getRecentlyViewed, state, t } from './state.js';

// 관심종목 패널 입력창의 이름/코드 타입어헤드 (2026-08-31). 헤더 ⌘K 검색이
// 2026-08-19에 헤더째 삭제되면서 종목 추가는 정확한 코드를 그대로 쳐넣는
// 경로만 남았고(한글 이름이면 422로 죽음), 백엔드 /api/search-tickers는
// 아무도 안 쓰는 고아가 됐다. 이 모듈이 그 둘을 잇는다.
//
// dialog.js와 같은 패턴: 단일 DOM 노드를 첫 사용 때 lazy 생성해 document.body에
// append(패널이 overflow:hidden 이라 그 안에 두면 잘림), position:fixed +
// getBoundingClientRect 로 띄우고, 열려 있는 동안만 document/window 리스너를
// 붙였다가 닫을 때 정확히 같은 플래그로 뗀다.

const DEBOUNCE_MS = 180;
const MIN_CHARS = 2;
const MAX_RESULTS = 8;
const CACHE_TTL_MS = 60_000;
const LOADING_DELAY_MS = 220;   // 깜빡임 방지 — 이보다 빨리 오면 로딩행 안 보임

const _nameCache = new Map();   // symbol -> display name (최근 본 종목 이름 해석용)

export function attachTickerSearch(inputEl, { onPick }) {
  let pop = null;
  let rows = [];          // 렌더된 행: {symbol,name,market,tradable,blocked, price?, change_pct?}
  let activeIdx = -1;
  let seq = 0;            // 경쟁조건 가드 — 응답 도착 시 seq가 바뀌었으면 폐기
  let debTimer = null;
  let loadingTimer = null;
  let open = false;
  let lastSearchQuery = '';
  const searchCache = new Map();   // `${scope}\0${q}` -> {ts, results}
  let suggestCache = null;         // {scope, ts, rows}

  function buildPop() {
    if (pop) return pop;
    pop = document.createElement('div');
    pop.className = 'ticker-search-pop';
    pop.id = 'ticker-search-pop';
    pop.setAttribute('role', 'listbox');
    document.body.appendChild(pop);
    // 행 클릭 시 입력창 포커스 유지 (mousedown이 blur를 일으키기 전에 차단)
    pop.addEventListener('mousedown', (e) => e.preventDefault());
    pop.addEventListener('click', (e) => {
      const rowEl = e.target.closest('.tsr');
      if (rowEl && !rowEl.classList.contains('blocked')) { commit(Number(rowEl.dataset.idx)); return; }
      if (e.target.closest('.ticker-search-retry')) runSearch(lastSearchQuery, true);
    });
    return pop;
  }

  function position() {
    const r = inputEl.getBoundingClientRect();
    const p = buildPop();
    p.style.minWidth = `${Math.max(r.width, 250)}px`;
    // keep the pop within the viewport horizontally (250px column near the
    // left edge is fine, but a flipped/narrow layout shouldn't clip)
    p.style.left = `${Math.max(4, Math.min(r.left, window.innerWidth - 344))}px`;
    const spaceBelow = window.innerHeight - r.bottom;
    const popH = Math.min(p.scrollHeight, 320);
    if (spaceBelow < popH + 8 && r.top > spaceBelow) {
      p.style.top = 'auto';
      p.style.bottom = `${window.innerHeight - r.top + 4}px`;
    } else {
      p.style.bottom = 'auto';
      p.style.top = `${r.bottom + 4}px`;
    }
  }

  function show() {
    if (!open) {
      open = true;
      buildPop().classList.add('open');
      inputEl.setAttribute('aria-expanded', 'true');
      document.addEventListener('pointerdown', onDocPointerDown, true);
      window.addEventListener('scroll', onScrollResize, true);
      window.addEventListener('resize', onScrollResize);
    }
    position();
  }

  function close() {
    if (!open) return;
    open = false;
    activeIdx = -1;
    if (pop) pop.classList.remove('open');
    inputEl.setAttribute('aria-expanded', 'false');
    inputEl.removeAttribute('aria-activedescendant');
    document.removeEventListener('pointerdown', onDocPointerDown, true);
    window.removeEventListener('scroll', onScrollResize, true);
    window.removeEventListener('resize', onScrollResize);
  }

  function onDocPointerDown(e) {
    if (e.target === inputEl || (pop && pop.contains(e.target))) return;
    close();
  }
  function onScrollResize() { close(); }

  /* ---------------- rendering ---------------- */

  function renderMessage(kind) {
    rows = [];
    activeIdx = -1;
    const p = buildPop();
    if (kind === 'loading') {
      p.innerHTML = `<div class="ticker-search-msg">${escapeHtml(t('tickerSearchLoading'))}</div>`;
    } else if (kind === 'empty') {
      p.innerHTML = `<div class="ticker-search-msg">${escapeHtml(t('tickerSearchEmpty'))}`
        + `<div class="sub">${escapeHtml(t('tickerSearchEmptySub'))}</div></div>`;
    } else if (kind === 'error') {
      p.innerHTML = `<div class="ticker-search-msg">${escapeHtml(t('tickerSearchError'))}`
        + `<div><button type="button" class="ticker-search-retry">${escapeHtml(t('tickerSearchRetry'))}</button></div></div>`;
    }
    show();
  }

  function badgeHtml(row) {
    const key = state[row.symbol]
      ? 'tickerSearchBadgeOnList'
      : (row.market === 'overseas' ? 'tickerSearchBadgeOverseas' : 'tickerSearchBadgeDomestic');
    return `<span class="tsr-badge">${escapeHtml(t(key))}</span>`;
  }

  function subHtml(row) {
    if (row.blocked) return `<span class="tsr-note">${escapeHtml(t('tickerSearchBlockedLeveraged'))}</span>`;
    const sym = `<span>${escapeHtml(displayTicker(row.symbol))}</span>`;
    if (typeof row.change_pct === 'number') {
      const up = row.change_pct >= 0;
      return sym + `<span class="${up ? 'up' : 'down'}">${up ? '+' : ''}${row.change_pct.toFixed(2)}%</span>`;
    }
    return sym;
  }

  function rowHtml(row, idx) {
    return `<div class="tsr${row.blocked ? ' blocked' : ''}" role="option" id="tsr-opt-${idx}" data-idx="${idx}"`
      + `${row.blocked ? ' aria-disabled="true"' : ''}>`
      + `<div class="tsr-top"><span class="tsr-name">${escapeHtml(row.name)}</span>${badgeHtml(row)}</div>`
      + `<div class="tsr-sub">${subHtml(row)}</div></div>`;
  }

  // sections: [{label: string|null, items: [row]}] — 빈 섹션은 건너뜀
  function renderSections(sections) {
    rows = [];
    let html = '';
    for (const sec of sections) {
      if (!sec.items.length) continue;
      if (sec.label) html += `<div class="ticker-search-group">${escapeHtml(sec.label)}</div>`;
      for (const it of sec.items) {
        const idx = rows.length;
        rows.push({ ...it, blocked: it.tradable === false });
        html += rowHtml(rows[idx], idx);
      }
    }
    if (!rows.length) { renderMessage('empty'); return; }
    activeIdx = -1;
    buildPop().innerHTML = html;
    show();
  }

  /* ---------------- keyboard nav ---------------- */

  function selectable() { return rows.map((_, i) => i).filter(i => !rows[i].blocked); }

  function setActive(idx) {
    if (!pop) return;
    pop.querySelectorAll('.tsr.active').forEach(el => el.classList.remove('active'));
    activeIdx = idx;
    if (idx < 0) { inputEl.removeAttribute('aria-activedescendant'); return; }
    const el = pop.querySelector(`#tsr-opt-${idx}`);
    if (el) { el.classList.add('active'); el.scrollIntoView({ block: 'nearest' }); inputEl.setAttribute('aria-activedescendant', el.id); }
  }

  function move(delta) {
    const sel = selectable();
    if (!sel.length) return;
    let pos = sel.indexOf(activeIdx);
    pos = pos === -1 ? (delta > 0 ? 0 : sel.length - 1) : pos + delta;
    pos = Math.max(0, Math.min(sel.length - 1, pos));
    setActive(sel[pos]);
  }

  function commit(idx) {
    const row = rows[idx];
    if (!row || row.blocked) return;
    close();
    inputEl.value = '';           // 연달아 추가할 수 있게 비우고 포커스는 유지
    onPick(row);
  }

  function onKeydown(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!open) { if (!inputEl.value.trim()) showSuggestions(); return; }
      move(1);
    } else if (e.key === 'ArrowUp') {
      if (!open) return;
      e.preventDefault();
      const sel = selectable();
      if (sel.length && activeIdx === sel[0]) { setActive(-1); return; }
      move(-1);
    } else if (e.key === 'Enter') {
      if (!open) return;
      const sel = selectable();
      if (!sel.length) return;   // 결과 0건 — main.js의 Enter 폴백(원문 그대로 add)에 넘김
      e.preventDefault();
      e.stopImmediatePropagation();
      commit(activeIdx >= 0 ? activeIdx : sel[0]);
    } else if (e.key === 'Escape') {
      if (open) { e.stopImmediatePropagation(); close(); }
      else if (inputEl.value) inputEl.value = '';
    } else if (e.key === 'Tab') {
      close();
    }
  }

  /* ---------------- data ---------------- */

  function onInput() {
    const q = inputEl.value.trim();
    clearTimeout(debTimer);
    if (q.length < MIN_CHARS) { q ? close() : showSuggestions(); return; }
    debTimer = setTimeout(() => runSearch(q, false), DEBOUNCE_MS);
  }

  async function runSearch(q, force) {
    q = (q || '').trim();
    if (q.length < MIN_CHARS) return;
    lastSearchQuery = q;
    const scope = _marketScope;
    const key = `${scope} ${q.toLowerCase()}`;
    const cached = searchCache.get(key);
    if (cached && !force && Date.now() - cached.ts < CACHE_TTL_MS) { renderSearchResults(cached.results); return; }

    const mySeq = ++seq;
    clearTimeout(loadingTimer);
    loadingTimer = setTimeout(() => { if (mySeq === seq) renderMessage('loading'); }, LOADING_DELAY_MS);
    let results;
    try {
      const r = await fetch(`/api/search-tickers?q=${encodeURIComponent(q)}&limit=${MAX_RESULTS}&scope=${scope}`);
      if (!r.ok) throw new Error(`http ${r.status}`);
      results = await r.json();
    } catch {
      clearTimeout(loadingTimer);
      if (mySeq === seq) renderMessage('error');
      return;
    }
    clearTimeout(loadingTimer);
    if (mySeq !== seq) return;
    searchCache.set(key, { ts: Date.now(), results });
    if (searchCache.size > 40) searchCache.delete(searchCache.keys().next().value);
    renderSearchResults(results);
  }

  function renderSearchResults(results) {
    if (!results.length) { renderMessage('empty'); return; }
    renderSections([{ label: null, items: results }]);
  }

  async function resolveName(symbol) {
    if (state[symbol]?.name) return state[symbol].name;
    if (_nameCache.has(symbol)) return _nameCache.get(symbol);
    try {
      const domestic = symbol.endsWith('.KS');
      const bare = domestic ? symbol.slice(0, -3) : symbol;
      const j = await fetch(`/api/search-tickers?q=${encodeURIComponent(bare)}&limit=3&scope=${domestic ? 'domestic' : 'overseas'}`).then(r => r.json());
      const hit = j.find(x => x.symbol === symbol) || null;
      const name = hit ? hit.name : displayTicker(symbol);
      _nameCache.set(symbol, name);
      return name;
    } catch { return displayTicker(symbol); }
  }

  async function showSuggestions() {
    const scope = _marketScope;
    const mySeq = ++seq;
    const recentSyms = getRecentlyViewed()
      .filter(sym => !state[sym] && sym.endsWith('.KS') === (scope === 'domestic'))
      .slice(0, 3);

    let topRows = [];
    if (suggestCache && suggestCache.scope === scope && Date.now() - suggestCache.ts < CACHE_TTL_MS) {
      topRows = suggestCache.rows;
    } else {
      try {
        topRows = await fetch(`/api/ticker-suggestions?scope=${scope}&limit=${MAX_RESULTS}`).then(r => r.json());
        suggestCache = { scope, ts: Date.now(), rows: topRows };
      } catch {
        topRows = (suggestCache && suggestCache.scope === scope) ? suggestCache.rows : [];
      }
    }
    if (mySeq !== seq) return;

    const recent = await Promise.all(recentSyms.map(async (sym) => ({
      symbol: sym, name: await resolveName(sym), market: scope, tradable: true,
    })));
    if (mySeq !== seq) return;

    const taken = new Set(recentSyms);
    const top = topRows
      .filter(r => !state[r.symbol] && !taken.has(r.symbol))
      .slice(0, Math.max(0, MAX_RESULTS - recent.length));

    if (!recent.length && !top.length) { close(); return; }
    renderSections([
      { label: t('tickerSearchGroupRecent'), items: recent },
      { label: t('tickerSearchGroupTopValue'), items: top },
    ]);
  }

  /* ---------------- wire ---------------- */

  inputEl.addEventListener('input', onInput);
  inputEl.addEventListener('keydown', onKeydown);
  inputEl.addEventListener('focus', () => {
    const q = inputEl.value.trim();
    if (!q) showSuggestions();
    else if (q.length >= MIN_CHARS) runSearch(q, false);
  });
}
