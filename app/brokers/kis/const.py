# app/brokers/kis/const.py
# Shared constants/exception for auth.py and broker.py.
# 인증/브로커 공용 상수 및 예외 (중복 정의 제거)

import hashlib
import os
from datetime import timedelta
from pathlib import Path

from app.domain.exceptions import BrokerError

# Refresh 10 minutes before actual expiry.
# 만료 10분 전 토큰 갱신 기준
_TOKEN_REFRESH_MARGIN = timedelta(minutes=10)
_TOKEN_PATH = "/oauth2/tokenP"


def _token_cache_dir() -> Path:
    """Directory the KIS access-token cache file(s) live in.

    KIS treats the OAuth token as a once-per-day resource ("접근 토큰은 1일 1회
    발급 원칙이며, 유효기간내 잦은 토큰 발급 발생 시 이용이 제한될 수 있습니다"),
    on top of the hard one-issuance-per-minute-per-appkey cap (EGW00133). The
    on-disk cache exists so a token issued once is reused for its full ~24h
    life across every process AND every container restart.

    That last part broke silently under the Docker deploy: pipeline.sh does
    `docker stop && rm && run` with no volume, so a cache written to the
    default location (inside the image tree) is destroyed on every redeploy,
    and the daemon's boot-time `_verify_configured_kis_credentials()` then
    issues a fresh token every single deploy — exactly the "잦은 토큰 발급"
    KIS warns it will start restricting. Setting KIS_TOKEN_CACHE_DIR to a
    path backed by a persistent Docker volume (see pipeline.sh) keeps one
    token alive across deploys.

    Unset (local dev, tests) falls back to the previous location — the `app`
    package dir — so behaviour there is unchanged.
    KIS_TOKEN_CACHE_DIR 미설정 시 기존 위치 그대로 — 로컬/테스트 동작 불변.
    """
    env_dir = os.getenv("KIS_TOKEN_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[2]


def token_cache_file() -> Path:
    """The base token-cache path — resolved per call, never frozen at import.

    A module-level `_TOKEN_CACHE_FILE = _token_cache_dir() / ...` read the env
    var exactly once, at first import. Production is fine (pipeline.sh exports
    KIS_TOKEN_CACHE_DIR before anything imports this), but it made the path
    untestable — a test could `monkeypatch.setenv` all it liked and still get
    the import-time value. Now `token_cache_path_for()` calls this fresh each
    time, so the env var is honoured whenever it is actually set.
    import 시점에 얼리지 않고 매 호출 계산 — 테스트에서 env 재지정이 실제로 반영됨.
    """
    return _token_cache_dir() / ".kis_token_cache.json"


def token_cache_dir_is_explicit() -> bool:
    """True when KIS_TOKEN_CACHE_DIR is set and non-empty — the cache dir was
    deliberately relocated (Docker volume mount, see pipeline.sh). A failed
    write there means every deploy re-issues a token (KIS restricts frequent
    re-issuance), so it is an error worth surfacing rather than a shrug.
    KIS_TOKEN_CACHE_DIR 를 일부러 지정한 경우 — 쓰기 실패는 경고가 아니라 에러."""
    return bool(os.getenv("KIS_TOKEN_CACHE_DIR"))


def token_cache_path_for(app_key: str) -> Path:
    """A per-credential-set token cache file.

    The single shared token_cache_file() base was correct when one process
    meant one KIS account. With several users in one process it is a
    correctness bug, not just untidiness: KIS issues at most **one token per
    minute per app key**, so two users overwriting each other's cached token
    doesn't merely re-fetch — it makes them race, each invalidating the
    other's cache and then getting rate-limited on the re-issue.
    토큰 발급이 앱키당 분당 1회로 제한되므로, 캐시 파일을 공유하면 단순 낭비가
    아니라 서로의 토큰을 무효화하며 발급 제한에 걸림.

    Keyed on a hash of the APP KEY rather than the user_id, which gets two
    properties for free:
      - paper and real sets separate automatically, since they are different
        app keys — no need to thread account_mode in here;
      - the cross-process sharing this file exists for is preserved. The
        daemon and a CLI run under the same credentials still land on the same
        path and reuse one token, which is the whole reason it is on disk
        rather than in memory.

    The hash also keeps the app key itself out of a filename that ends up in
    directory listings, backups, and error messages.
    """
    digest = hashlib.sha256(app_key.encode("utf-8")).hexdigest()[:16]
    return token_cache_file().with_name(f".kis_token_cache.{digest}.json")


class KISBrokerError(BrokerError, RuntimeError):
    """Raised when the KIS API refuses a request or token issuance fails.

    Inherits RuntimeError too (not just BrokerError) so existing
    `except RuntimeError` call sites (e.g. dataflows/kis_provider.py) keep
    working unchanged, while new code can catch the broker-agnostic
    `app.domain.exceptions.BrokerError` instead.
    KIS API 요청/토큰 발급 실패시 발생하는 예외 (기존 RuntimeError catch와도 호환)

    `msg_cd`/`rt_cd` carry KIS's own response codes (2026-08-19) — before
    this, they only existed baked into the message string, so a catcher
    couldn't tell "permanently malformed request" (e.g. IGW00013, a bad
    ORD_QTY) from a transient network/gateway hiccup. That gap is exactly
    why a stop-loss exit for AXTI resubmitted the same doomed order 17 times
    over 43 minutes on 2026-08-19 — see is_permanent_refusal below.

    `status_code` (2026-08-26) carries the raw HTTP status when KIS gives no
    usable code body at all — a 5xx gateway/maintenance page or a 200 with a
    non-JSON body has no msg_cd to classify by, so token-issuance failure
    classification (is_transient_token_failure below) needs the status line
    as a fallback signal."""

    def __init__(
        self, message: str, *, msg_cd: str | None = None, rt_cd: str | None = None, status_code: int | None = None
    ):
        super().__init__(message)
        self.msg_cd = (msg_cd or "").strip()
        self.rt_cd = (rt_cd or "").strip()
        self.status_code = status_code


# KIS response codes confirmed to mean "this exact request is malformed and
# will be refused again no matter how many times it's resubmitted" — request
# format/validation failures, not transient gateway/network issues. Kept
# deliberately short and evidence-based (both entries below were directly
# observed on 2026-08-19's IGW00007/IGW00013 incident) rather than guessed at:
# an unlisted code defaults to "transient" (see is_permanent_refusal), so a
# wrong guess here can only ever cost an extra retry, never suppress a
# needed one — retrying a real stop-loss sell too little is much worse than
# retrying a doomed one too much.
PERMANENT_MSG_CDS = frozenset({
    "IGW00013",  # "주문 수량을 확인해주세요" — malformed ORD_QTY (2026-08-19 AXTI)
    "IGW00007",  # "MCA 전문바디 구성 중 오류" — malformed request body (2026-08-19 GOOG/GOOGL)
})


def is_permanent_refusal(exc: BaseException) -> bool:
    """True if `exc` is a KISBrokerError whose msg_cd marks the request
    itself as malformed — resubmitting the identical payload cannot help.
    Anything else (including a KISBrokerError with an unrecognized/missing
    msg_cd) is treated as transient/retryable by default — see
    PERMANENT_MSG_CDS's own comment for why that default direction is
    deliberate."""
    return isinstance(exc, KISBrokerError) and exc.msg_cd in PERMANENT_MSG_CDS


# KIS codes confirmed to mean "token issuance didn't work right now, not
# that this app key/secret is wrong" — kept separate from PERMANENT_MSG_CDS
# on purpose: that set is about order payloads being malformed (opposite
# meaning — permanent, not transient), mixing the two would corrupt the
# 2026-08-19 resubmission-suppression logic that reads it.
# EGW00133: token-issuance rate limit (one issuance per app key per minute —
# see app/credentials/validation.py's module docstring). EGW00201: the
# per-second call-rate limit already treated as retryable in client.py's own
# request loop.
TOKEN_ISSUE_TRANSIENT_MSG_CDS = frozenset({
    "EGW00133",
    "EGW00201",
})


# KIS's two DISTINCT per-second rate limits — different counters, both
# surfaced as a msg_cd in the response body (HTTP 500, or HTTP 200 with
# rt_cd != "0"), so match on the body substring, never on status code.
#   EGW00201 — the GATEWAY's per-account call-rate cap (real 18/s, paper
#     1/s; project memory kis-api-rate-limit-notice-2026-04-20). Already
#     retried once in client.py's request loop.
#   EGW00215 — "원장에서 허용 가능한 초당 거래건수를 초과하였습니다": the
#     LEDGER layer's own, tighter per-second inquiry cap. A SEPARATE counter
#     from the gateway's, so kis_min_request_interval alone can never bound
#     it. First seen 2026-09-01 on a real-account KRX cycle — the domestic
#     balance inquiry returned HTTP 500 with this code while rebalance_to_
#     target fired 3-5 balance TRs inside one market_lock at 0.07s spacing.
# EGW00201=게이트웨이 한도, EGW00215=원장 한도 — 서로 다른 카운터라 별도 처리.
RATE_LIMIT_MSG_CDS = frozenset({
    "EGW00201",
    "EGW00215",
})


def is_rate_limited(body_text: str) -> str | None:
    """The rate-limit msg_cd present in a raw KIS response body, or None.
    Body-substring match (same as client.py's existing `"EGW00201" in
    resp.text`) — these arrive on HTTP 500 and on HTTP 200 alike."""
    for code in RATE_LIMIT_MSG_CDS:
        if code in body_text:
            return code
    return None


def is_transient_token_failure(exc: BaseException) -> bool:
    """True if `exc` (from KIS token issuance) reflects KIS being
    unavailable/rate-limited/unparseable right now, not the app key/secret
    itself being wrong.

    Deliberately the opposite default direction from is_permanent_refusal:
    there, an unrecognized code defaults to "transient" because a missed
    stop-loss retry is worse than an extra one. Here, this module's entire
    purpose is catching a genuinely typo'd secret at boot — defaulting an
    unrecognized 4xx to "transient" would let a real bad-credentials error
    get logged as a mere WARNING and re-surface later as the "wall of KIS
    call failed warnings" this module exists to prevent. The known-transient
    cases (rate limit, 5xx, no parseable body) are instead enumerated
    explicitly below."""
    if not isinstance(exc, KISBrokerError):
        return False
    if exc.msg_cd in TOKEN_ISSUE_TRANSIENT_MSG_CDS:
        return True
    status = exc.status_code
    if status is None:
        return False
    # 5xx: KIS-side outage/maintenance gateway. 429: rate limited outside the
    # msg_cd-carrying path. 200: HTTP succeeded but the body carried no
    # usable access_token/code (maintenance HTML, captive portal) — no
    # credential-related signal was actually returned.
    return status >= 500 or status == 429 or status == 200
