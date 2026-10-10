import { escapeHtml, state, t } from './state.js';
import { wireTranscriptToggle } from './trading.js';

export function renderMarkdown(text) {
  // Minimal, escape-first renderer for the LLM's own **bold**-and-paragraphs
  // style (see app/state.py's render_* helpers) — not a general markdown
  // parser, just enough so those markers show as formatting instead of
  // literal asterisks. Escaping happens before any tag is introduced, so
  // there's no way for the LLM's text to inject real HTML.
  const escaped = escapeHtml(text);
  const bold = escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  return bold
    .split(/\n\n+/)
    .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`)
    .join('');
}

// Shown only on an empty chat-log — first open and after "새 대화". Picks
// questions the system can actually answer from data it has (portfolio,
// today's AI decisions, stop-loss, this week's self-audit numbers) rather
// than generic chit-chat, so a first-time viewer's first click lands on a
// real, data-backed answer instead of "AI 분석은 참고용입니다".
const SUGGESTED_QUESTIONS = ['chatSuggestRisk', 'chatSuggestReason', 'chatSuggestStopLoss', 'chatSuggestPerformance'];

function renderChatEmptyState() {
  const log = document.getElementById('chat-log');
  const chips = SUGGESTED_QUESTIONS.map(key => `<button type="button" class="chat-suggest-chip">${escapeHtml(t(key))}</button>`).join('');
  log.innerHTML = `
    <div class="chat-empty">
      <div class="chat-empty-title">${escapeHtml(t('chatEmptyTitle'))}</div>
      <div class="chat-suggest-chips">${chips}</div>
    </div>`;
  log.querySelectorAll('.chat-suggest-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.getElementById('chat-input').value = chip.textContent;
      sendChat();
    });
  });
}

export function startNewChat() {
  renderChatEmptyState();
}

export function appendChatBubble(role, text) {
  const log = document.getElementById('chat-log');
  log.querySelector('.chat-empty')?.remove();
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (role === 'bot') {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }
  msg.appendChild(bubble);
  log.appendChild(msg);
  log.scrollTop = log.scrollHeight;
  return msg;
}

export function appendTranscriptToggle(decisionId) {
  const log = document.getElementById('chat-log');
  const msg = document.createElement('div');
  msg.className = 'msg bot';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = `<span class="see-more">${t('seeDebate')}</span><div class="transcript" style="display:none"></div>`;
  msg.appendChild(bubble);
  log.appendChild(msg);
  log.scrollTop = log.scrollHeight;

  wireTranscriptToggle(
    bubble.querySelector('.see-more'),
    bubble.querySelector('.transcript'),
    decisionId,
    { open: t('hideDebate'), closed: t('seeDebate') }
  );
}

export async function sendChat() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const question = input.value.trim();
  if (!question) return;

  appendChatBubble('user', question);
  input.value = '';
  input.disabled = true;
  sendBtn.disabled = true;
  const thinkingEl = appendChatBubble('thinking', t('thinking'));
  const thinkingBubble = thinkingEl.querySelector('.bubble');
  let answered = false;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split('\n\n');
      buffer = chunks.pop();  // last chunk may be incomplete — keep it for next read
      for (const chunk of chunks) {
        if (!chunk.startsWith('data: ')) continue;
        const event = JSON.parse(chunk.slice(6));
        if (event.type === 'stage') {
          thinkingBubble.textContent = event.label;
        } else if (event.type === 'done') {
          thinkingEl.remove();
          appendChatBubble('bot', event.answer);
          if (event.decision_id) appendTranscriptToggle(event.decision_id);
          answered = true;
        }
      }
    }
    if (!answered) throw new Error('stream ended without a final answer');
  } catch (err) {
    thinkingEl.remove();
    appendChatBubble('bot', t('chatError'));
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

/* ---------------- order ticket ---------------- */

