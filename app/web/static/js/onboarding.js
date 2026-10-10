// KIS-not-configured gate (2026-08-07 single-account rollback — see
// docs/single-user-rollback-guide.md Phase 5).
//
// Through 2026-08-06 this module owned a login/registration screen and a KIS
// app-key/secret/account-number input form (the multi-user migration's
// onboarding gate). Both are gone: this deployment trades one company
// account, its KIS credentials arrive as env vars Jenkins injects at deploy
// time (app.config.Settings.kis_credentials), and there is no session to log
// into. What's left is narrower: if this deployment has no REAL KIS
// credentials configured at all, there is nothing a browser visitor can do
// about it — show an informational message and never let the dashboard boot,
// rather than painting a broken-looking page full of 412s behind it.
//
// main.js still `await`s ensureReady() before calling buildCards() and
// kicking off ~15 polling requests — that ordering is why this file exists
// at all. Do not remove that await: firing those requests before KIS is
// configured just relabels the same 412 storm the old gate existed to avoid.
// main.js가 ensureReady()를 기다린 뒤에야 대시보드를 띄움 — 이 순서를 없애면
// KIS 미설정 상태에서 폴링이 전부 412로 실패하는 게 그대로 재현됨.

const CREDENTIALS = '/api/credentials';

let overlay = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function mount() {
  if (overlay) return overlay;
  // Lock page scrolling. The overlay is position:fixed and covers the
  // viewport, but the dashboard shell behind it is wider than the window, so
  // without this the page still scrolls horizontally and slides the
  // half-rendered dashboard out from behind the gate — which is exactly what
  // the gate exists to keep out of view.
  // 뒤쪽 대시보드가 화면보다 넓어서 스크롤이 남으면 게이트 옆으로 반쯤 그려진
  // 대시보드가 드러남.
  document.documentElement.classList.add('onboard-locked');
  overlay = el('div', 'onboard-overlay');
  overlay.innerHTML = `
    <div class="onboard-card" role="dialog" aria-modal="true" aria-labelledby="onboard-title">
      <div class="onboard-brand"><span class="onboard-brand-mark">AI</span> invest-pipeline</div>
      <h1 class="onboard-title" id="onboard-title"></h1>
      <p class="onboard-sub"></p>
    </div>`;
  document.body.appendChild(overlay);
  return overlay;
}

function renderNotConfigured() {
  const card = overlay.querySelector('.onboard-card');
  card.querySelector('.onboard-title').textContent = 'KIS 자격증명 미설정';
  card.querySelector('.onboard-sub').textContent =
    '이 서버에 실전 KIS 자격증명이 설정되어 있지 않습니다. 관리자에게 문의해 ' +
    'Jenkins 환경변수(KIS_APP_KEY_REAL / KIS_APP_SECRET_REAL / KIS_ACCOUNT_8_REAL)를 ' +
    '설정한 뒤 재배포해야 합니다.';
}

/** Sticky, non-blocking banner for the "REAL credentials exist but order
 *  limits don't" state — see this state's own note in base.css. Idempotent:
 *  safe to call every time ensureReady() resolves ready, won't double-insert.
 *
 *  Requires `real` in status.configured before even looking at
 *  real_limits_configured — a paper-only deployment (every local dev
 *  instance, see docs/single-user-rollback-guide.md §8.2) has no
 *  real credentials at all, so "real order limits aren't configured" is
 *  true but irrelevant there: nobody is one step from placing a real order,
 *  so warning about it is just noise. Confirmed live (2026-08-07) — the
 *  banner fired for a pure-paper local instance with no real credentials
 *  configured at all. */
function maybeShowRealLimitsBanner(status) {
  const realConfigured = (status.configured || []).includes('real');
  if (!realConfigured || status.real_limits_configured || document.getElementById('real-limits-banner')) return;
  const bar = el(
    'div', 'real-limits-banner',
    '실전 주문 한도가 설정되지 않아 실전 계좌 주문이 모두 차단됩니다 — 관리자에게 문의하세요.',
  );
  bar.id = 'real-limits-banner';
  document.body.prepend(bar);
}

/** Resolves once this deployment has real KIS credentials configured, or
 *  never resolves otherwise — see this module's own top comment on why
 *  blocking forever (not looping, not retrying) is the right behavior here:
 *  nothing that happens inside the browser can fix a missing env var, and a
 *  reload after the actual fix (a redeploy) re-runs this check from
 *  scratch. */
export async function ensureReady() {
  let status;
  try {
    const response = await fetch(CREDENTIALS);
    if (!response.ok) return;  // 503 etc — let the dashboard show its own state
    status = await response.json();
  } catch {
    // The API is unreachable. Not a KIS-configuration problem, and this gate
    // would be a misleading thing to show — let the dashboard boot and
    // surface its own connection errors.
    return;
  }

  if (status.ready) {
    maybeShowRealLimitsBanner(status);
    return;
  }

  mount();
  renderNotConfigured();
  await new Promise(() => {});  // deliberately never resolves — see docstring
}
