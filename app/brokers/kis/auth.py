# app/brokers/kis/auth.py

import json
import logging
import asyncio
import contextlib
import os
import stat
from datetime import datetime, timedelta
from pathlib import Path
import httpx

from app.brokers.kis.const import (
    KISBrokerError,
    _TOKEN_PATH,
    _TOKEN_REFRESH_MARGIN,
    token_cache_dir_is_explicit,
    token_cache_file,
)
from app.clock import KR_TZ, get_clock

# 로거 생성
logger = logging.getLogger(__name__)


def _token_error_codes(resp: httpx.Response) -> tuple[str | None, str | None]:
    """Best-effort KIS error code extraction from a token-endpoint response —
    never raises. `/oauth2/tokenP` is an OAuth endpoint, not the usual
    trading-TR contract `_checked_body` (client.py) parses, so a failure body
    can carry `error_code`/`error_description` (OAuth-style) instead of
    `msg_cd`/`rt_cd` (KIS trading-TR style, still seen on some rejections) —
    both are normalized into the one (msg_cd, rt_cd) shape callers expect."""
    try:
        body = resp.json()
    except ValueError:
        return None, None
    if not isinstance(body, dict):
        return None, None
    msg_cd = body.get("msg_cd") or body.get("error_code")
    rt_cd = body.get("rt_cd")
    return (str(msg_cd) if msg_cd is not None else None, str(rt_cd) if rt_cd is not None else None)


# Token management, split out of the old monolithic kis.py.
# 토큰 발급/캐시를 전담하는 독립 클래스
class KISAuth:
    def __init__(
        self,
        app_key: str,
        app_secret: str,
        base_url: str,
        token_cache_path: Path | None = None,
        client: httpx.AsyncClient | None = None,
        throttle_func=None  # ★ 중요: 기존 클라이언트의 throttle 기능을 주입받기 위한 통로입니다.
    ) -> None:
        self._app_key = app_key
        self._app_secret = app_secret
        self._base_url = base_url.rstrip("/")
        self._token_cache_path = token_cache_path or token_cache_file()
        self._client = client or httpx.AsyncClient(timeout=15.0)
        
        self._token: str | None = None
        self._token_expires_at: datetime | None = None
        self._token_lock = asyncio.Lock()
        
        # 기존 kis.py에 있던 self._throttle() 메서드를 넘겨받아 사용하기 위함입니다.
        self._throttle_func = throttle_func

    async def _throttle(self, priority: str = "low"):
        """기존 KISBroker의 throttle 함수를 호출하거나 없으면 대기 없이 통과"""
        if self._throttle_func:
            await self._throttle_func(priority)

    async def _throttle_high(self):
        await self._throttle("high")

    # ---------------- token ----------------

    async def _get_token(self) -> str:
        async with self._token_lock:
            if self._token and self._token_expires_at:
                if get_clock().now(KR_TZ) < self._token_expires_at - _TOKEN_REFRESH_MARGIN:
                    return self._token
            cached = self._load_cached_token()
            if cached:
                self._token, self._token_expires_at = cached
                return self._token
            try:
                return await self._issue_token()
            except KISBrokerError as exc:
                # Reissue failed (KIS token endpoint down, rate-limited,
                # etc). If we still hold an in-memory token that hasn't
                # ACTUALLY expired yet (only inside the refresh margin),
                # keep serving it rather than blocking every broker call —
                # including stop-loss sells — for up to 10 minutes.
                if self._token and self._token_expires_at and get_clock().now(KR_TZ) < self._token_expires_at:
                    logger.warning(
                        "KIS token reissue failed, falling back to still-valid cached token until %s",
                        self._token_expires_at,
                    )
                    return self._token
                # Nothing usable in memory — but a disk token that is truly
                # still valid (just inside the 10-min refresh margin, so the
                # normal _load_cached_token() above rejected it) beats failing
                # every call. Matters for a fresh one-shot CLI process that
                # hits EGW00133 with an empty in-memory token.
                # 인메모리에 없어도 디스크에 갱신 마진 안쪽의 유효 토큰이 있으면 사용.
                disk = self._load_cached_token(allow_refresh_margin=True)
                if disk:
                    self._token, self._token_expires_at = disk
                    logger.warning(
                        "KIS token reissue failed (%s) — using on-disk token still valid until %s",
                        exc, self._token_expires_at,
                    )
                    return self._token
                raise

    def _load_cached_token(self, *, allow_refresh_margin: bool = False) -> tuple[str, datetime] | None:
        """Read the on-disk token, or None if missing/corrupt/stale.

        `allow_refresh_margin=True` accepts a token that is truly still valid
        but inside the 10-min pre-expiry refresh window — used only as the
        last-ditch fallback in `_get_token` when issuance itself just failed.
        """
        try:
            raw = json.loads(self._token_cache_path.read_text(encoding="utf-8"))
            expires_at = datetime.fromisoformat(raw["expires_at"])
            token = raw["token"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Cache is an optimization, never fatal (falls through to a fresh
            # issuance) — but a silent failure here previously left no trail
            # for "why did this re-issue when a cache file clearly exists".
            # TypeError is a backstop: a naive/aware datetime comparison below
            # can raise it, and a boot-path broker call must not propagate.
            logger.debug("KIS token cache unreadable/corrupt, issuing fresh: %s", exc)
            return None
        if expires_at.tzinfo is None:
            # Legacy cache written before KST-aware expiries — KST is the only
            # wall clock KIS ever used here, so tag it so the compare is sound.
            # 예전(naive) 캐시 — KST 로 간주해 tz-aware 비교.
            expires_at = expires_at.replace(tzinfo=KR_TZ)
        # Cache is keyed per app key so switching paper/real keys later can't
        # silently reuse the wrong account's token.
        if raw.get("app_key_tail") != self._app_key[-8:]:
            return None
        cutoff = expires_at if allow_refresh_margin else expires_at - _TOKEN_REFRESH_MARGIN
        if get_clock().now(KR_TZ) >= cutoff:
            return None
        return token, expires_at

    async def _issue_token(self) -> str:
        # "high": this runs while _get_token holds _token_lock, so every
        # other broker call (orders included) queues behind that lock until
        # this throttle tick is served. Registering it as "low" let a burst
        # of routine quote/rank polling push a token refresh — and every
        # order call waiting on the lock behind it — back by many ticks.
        await self._throttle_high()
        resp = await self._client.post(
            f"{self._base_url}{_TOKEN_PATH}",
            json={
                "grant_type": "client_credentials",
                "appkey": self._app_key,
                "appsecret": self._app_secret,
            },
        )
        if resp.status_code != 200:
            msg_cd, rt_cd = _token_error_codes(resp)
            raise KISBrokerError(
                f"token issuance failed: HTTP {resp.status_code} {resp.text[:200]}",
                msg_cd=msg_cd, rt_cd=rt_cd, status_code=resp.status_code,
            )
        try:
            body = resp.json()
        except ValueError as exc:
            # 200 but the body wasn't JSON at all (maintenance page/captive
            # portal HTML) — no credential-related signal was returned here,
            # so this must surface as KISBrokerError (caught by
            # app.credentials.validation / broker callers) rather than an
            # unhandled JSONDecodeError.
            raise KISBrokerError(
                f"token issuance failed: HTTP 200 but response was not JSON: {resp.text[:200]}",
                status_code=resp.status_code,
            ) from exc
        if not isinstance(body, dict):
            raise KISBrokerError(
                f"token issuance failed: HTTP 200 with unexpected body shape: {str(body)[:200]}",
                status_code=resp.status_code,
            )
        token = body.get("access_token")
        if not token:
            # Bounded like the HTTP-status branch above — an unbounded body
            # here could dump an oversized or unexpected response straight
            # into logs/exceptions.
            # 위 HTTP 상태 분기와 동일하게 응답 바디 길이를 제한 (무제한 에러 바디 방지).
            msg_cd = body.get("msg_cd") or body.get("error_code")
            raise KISBrokerError(
                f"token issuance failed: no access_token in response: {str(body)[:200]}",
                msg_cd=str(msg_cd) if msg_cd is not None else None,
                rt_cd=str(body.get("rt_cd")) if body.get("rt_cd") is not None else None,
                status_code=resp.status_code,
            )
        # access_token_token_expired is "YYYY-MM-DD HH:MM:SS" — a KST wall-clock
        # string. Tag it as KST so every comparison against get_clock().now()
        # is tz-aware: the container runs UTC, so a naive parse believed each
        # token valid ~9h past its real expiry. Fall back to expires_in
        # seconds if the format ever changes.
        # KST 벽시계 문자열 — tzinfo 를 KST 로 박아 UTC 컨테이너에서 9시간 더
        # 살아있는 것으로 오인하지 않게.
        now = get_clock().now(KR_TZ)
        expires_in = int(body.get("expires_in", 86400))
        try:
            expires_at = datetime.strptime(
                body["access_token_token_expired"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=KR_TZ)
        except (KeyError, ValueError):
            expires_at = now + timedelta(seconds=expires_in)
        # Defensive clamp: KIS tokens live ~24h, so a parsed expiry more than
        # 25h out is a malformed/garbage timestamp. Never trust it past the
        # nominal life — otherwise a dead token gets served for a whole KRX
        # session with no self-heal.
        # 방어적 상한 — 파싱된 만료가 25시간 넘게 미래면 손상된 값, 명목 수명으로 축소.
        if expires_at - now > timedelta(hours=25):
            logger.warning(
                "KIS returned an implausible token expiry %s (now %s) — clamping to now + %ds",
                expires_at, now, expires_in,
            )
            expires_at = now + timedelta(seconds=expires_in)
        self._token, self._token_expires_at = token, expires_at
        try:
            # The cache dir is normally the app tree (always present) but can
            # be pointed at a mounted volume via KIS_TOKEN_CACHE_DIR — create
            # it if missing so a first write there doesn't fail and silently
            # drop us back to re-issuing a token on every call.
            self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic publish: write + chmod a temp file in the same dir, then
            # os.replace() it over the real path (atomic on POSIX). A
            # concurrent reader — another process, or _load_cached_token on a
            # one-shot CLI run — then never sees a half-written cache; it gets
            # either the old file or the new one. chmod BEFORE the rename so
            # the token is never briefly world-readable at the final path.
            # 원자적 교체 — 반쯤 쓰인 캐시를 다른 프로세스가 읽지 않도록.
            tmp_path = self._token_cache_path.with_name(
                f"{self._token_cache_path.name}.tmp.{os.getpid()}"
            )
            try:
                tmp_path.write_text(
                    json.dumps(
                        {
                            "token": token,
                            "expires_at": expires_at.isoformat(),
                            "app_key_tail": self._app_key[-8:],
                        }
                    ),
                    encoding="utf-8",
                )
                # The token is stored in plaintext (only formatted, not
                # encrypted) — restrict to owner read/write so other local
                # accounts/processes can't read a live KIS access token off
                # disk. Best-effort: os.chmod's effect is limited on Windows
                # (mainly the read-only bit), but this is free defense-in-depth
                # and is the real protection on any POSIX deployment.
                # 파일 권한을 소유자 전용으로 제한 (평문 저장은 그대로, 접근 통제만).
                try:
                    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    pass
                os.replace(tmp_path, self._token_cache_path)
            except OSError:
                # Leave no partial temp file behind; the outer handler logs it.
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
                raise
        except OSError as exc:  # cache is an optimization, never fatal
            if token_cache_dir_is_explicit():
                # KIS_TOKEN_CACHE_DIR was set on purpose (Docker volume mount,
                # see pipeline.sh) so one token survives redeploys. An
                # unwritable cache there means every deploy re-issues a token
                # — exactly what the volume exists to prevent, and KIS
                # restricts frequent re-issuance. Distinct greppable marker.
                logger.error(
                    "kis_token_cache_unwritable: could not write KIS token cache at %s: %s",
                    self._token_cache_path, exc,
                )
            else:
                logger.warning("could not write KIS token cache: %s", exc)
        logger.info("KIS access token issued, valid until %s", expires_at)
        return token