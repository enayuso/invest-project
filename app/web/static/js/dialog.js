import { t } from './state.js';

// In-app replacement for native confirm()/prompt()/alert() (2026-08-18) — the
// native dialogs show an unstylable "127.0.0.1:8000 says" chrome, which made
// the confirmation step for real-money order/kill-switch actions look like
// the least-finished part of the whole screen. CSS for these classes already
// lives in css/base.css (.app-dialog-backdrop/.app-dialog/...); this file is
// only the behavior.
//
// Native confirm()'s safety property is preserved on purpose: there is
// exactly one path to the "confirmed" outcome (clicking the confirm button).
// Escape, a backdrop click, and the cancel button all resolve to the
// cancelled outcome (false / null) — never relaxed, even though it would be
// easy to, say, let Enter-anywhere confirm.
//
// One shared backdrop/dialog DOM node (built lazily on first use, appended to
// document.body — never baked into index.html) rather than one per call, so
// repeated calls can't pile up detached nodes. Only one dialog is ever open
// at a time; `_cleanup` tracks the currently-attached listener set so a
// second open (shouldn't happen since every call site awaits) can't leak the
// first one's listeners.
let _els = null;
let _cleanup = null;

function _build() {
  const backdrop = document.createElement('div');
  backdrop.className = 'app-dialog-backdrop';
  backdrop.innerHTML = `
    <div class="app-dialog" role="dialog" aria-modal="true" aria-labelledby="app-dialog-title">
      <div class="app-dialog-title" id="app-dialog-title"></div>
      <div class="app-dialog-body"></div>
      <input class="app-dialog-input" type="text" inputmode="decimal" style="display:none" />
      <div class="app-dialog-actions">
        <button type="button" class="app-dialog-cancel"></button>
        <button type="button" class="app-dialog-confirm"></button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);
  return {
    backdrop,
    title: backdrop.querySelector('.app-dialog-title'),
    body: backdrop.querySelector('.app-dialog-body'),
    input: backdrop.querySelector('.app-dialog-input'),
    cancelBtn: backdrop.querySelector('.app-dialog-cancel'),
    confirmBtn: backdrop.querySelector('.app-dialog-confirm'),
  };
}

function _dialogEls() {
  if (!_els) _els = _build();
  return _els;
}

// mode: 'confirm' | 'prompt' | 'alert'. Shared implementation for all three
// exported functions below — they differ only in whether the input/cancel
// button show and what a "cancelled" resolution value looks like.
function _open({ title, body, confirmLabel, cancelLabel, danger, mode, defaultValue, placeholder, inputMode }) {
  const els = _dialogEls();
  if (_cleanup) _cleanup();  // defensive — see module docstring; shouldn't happen in practice

  const showInput = mode === 'prompt';
  const showCancel = mode !== 'alert';
  const previouslyFocused = document.activeElement;

  // title/body are plain text ONLY (textContent, never innerHTML) — an order
  // confirmation interpolates a ticker/company name pulled from server data,
  // and letting that be parsed as HTML would defeat the whole point of
  // replacing a dialog that couldn't be exploited that way either.
  els.title.textContent = title || '';
  els.body.textContent = body || '';
  els.confirmBtn.textContent = confirmLabel || (mode === 'alert' ? t('dialogClose') : t('dialogConfirm'));
  els.confirmBtn.classList.toggle('danger', !!danger);
  els.cancelBtn.textContent = cancelLabel || t('dialogCancel');
  els.cancelBtn.style.display = showCancel ? '' : 'none';
  els.input.style.display = showInput ? '' : 'none';
  els.input.inputMode = inputMode || 'decimal';
  if (showInput) {
    els.input.value = defaultValue || '';
    els.input.placeholder = placeholder || '';
  }

  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      _cleanup();
      els.backdrop.classList.remove('open');
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') previouslyFocused.focus();
      resolve(value);
    };

    // alert() has only one real outcome (there's nothing to decline), so
    // every dismissal path — Escape, backdrop click, the one button —
    // resolves the same way, same as a native alert().
    const onConfirm = () => finish(showInput ? els.input.value : true);
    const onCancel = () => finish(mode === 'alert' ? undefined : (showInput ? null : false));
    const onBackdropClick = (e) => { if (e.target === els.backdrop) onCancel(); };
    const focusables = [...(showInput ? [els.input] : []), ...(showCancel ? [els.cancelBtn] : []), els.confirmBtn];
    const onKeydown = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); onCancel(); return; }
      if (e.key === 'Enter' && showInput && document.activeElement === els.input) { e.preventDefault(); onConfirm(); return; }
      if (e.key === 'Tab') {
        // Focus trap — this dialog only ever has 1-3 focusable elements, so
        // a plain wraparound between the first and last is enough (no need
        // for a general-purpose focusable-element query).
        //
        // Clicking the dialog's own non-focusable text (title/body div) moves
        // activeElement to <body> — neither === first nor === last matches
        // then, so Tab would otherwise escape to whatever's focusable behind
        // the backdrop. Recover by snapping back into the dialog first.
        if (!els.backdrop.contains(document.activeElement)) { e.preventDefault(); focusables[0].focus(); return; }
        const first = focusables[0], last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };

    els.confirmBtn.addEventListener('click', onConfirm);
    els.cancelBtn.addEventListener('click', onCancel);
    els.backdrop.addEventListener('click', onBackdropClick);
    // Attached to `document` (capture) rather than `els.backdrop`, and for
    // the same reason as the Tab-trap fix above: clicking the dialog's own
    // non-focusable text moves activeElement to <body>, which is outside the
    // backdrop subtree, so a listener on the backdrop would simply never see
    // the keydown — Escape would stop cancelling the dialog. Capture phase
    // guarantees we still see the event ahead of any other page handler.
    document.addEventListener('keydown', onKeydown, true);
    _cleanup = () => {
      els.confirmBtn.removeEventListener('click', onConfirm);
      els.cancelBtn.removeEventListener('click', onCancel);
      els.backdrop.removeEventListener('click', onBackdropClick);
      // Must pass the same `true` (capture) here or this never detaches.
      document.removeEventListener('keydown', onKeydown, true);
      _cleanup = null;
    };

    els.backdrop.classList.add('open');
    // Focus the safer default (cancel) on open — except promptDialog, which
    // focuses+selects the input instead (item 5 of the spec this implements),
    // and alertDialog, whose only button IS the confirm button.
    if (showInput) { els.input.focus(); els.input.select(); }
    else if (showCancel) els.cancelBtn.focus();
    else els.confirmBtn.focus();
  });
}

export function confirmDialog({ title, body, confirmLabel, cancelLabel, danger } = {}) {
  return _open({ title, body, confirmLabel, cancelLabel, danger, mode: 'confirm' });
}

// inputMode isn't in the original spec signature — added because this ended
// up with a second caller (the kill-switch's engage-reason prompt, a free-
// text reason rather than a price) beyond the price-revise prompt it was
// designed around. Defaults to 'decimal' so the documented/originally-
// intended caller (revise price) is unaffected; the reason prompt passes
// 'text' explicitly. Flagged in the handoff report as a deliberate deviation.
export function promptDialog({ title, body, defaultValue, placeholder, confirmLabel, cancelLabel, inputMode } = {}) {
  return _open({ title, body, confirmLabel, cancelLabel, mode: 'prompt', defaultValue, placeholder, inputMode });
}

export function alertDialog({ title, body, danger, confirmLabel } = {}) {
  return _open({ title, body, confirmLabel, danger, mode: 'alert' });
}
