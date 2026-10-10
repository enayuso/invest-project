#!/usr/bin/env bash
set -euo pipefail

# Default (no module arg) deploys a persistent, always-on container running
# app/daemon.py — now the live dashboard (FastAPI + background poller), not
# just a background loop. Matches be-fastapi/api-rag's `docker run -d
# --restart` pattern otherwise, so this shows up in `docker ps` as "Up" like
# the other services on the host. Explicit module args instead run a one-shot CLI
# command (--rm, exits when done) for manual/local use — this project's
# other entrypoints (app.main, app.backtest, app.memory, app.execution) were
# all designed as batch scripts, not servers, so only the no-arg/daemon path
# runs the dashboard or is meant to be left running.
#
# Usage: ./pipeline.sh <env> [module] [module args...]
#   ./pipeline.sh dev                                                          deploy persistent dashboard
#   ./pipeline.sh dev main --ticker AAPL --date 2026-07-08                     one-shot manual decision
#   ./pipeline.sh dev backtest --ticker AAPL --start 2026-05-01 --end 2026-07-01 --cadence-days 7
#   ./pipeline.sh dev memory resolve --ticker AAPL
#   ./pipeline.sh dev execution check-stops --ticker AAPL
#
# Env file = .env.<env>, falling back to .env if that doesn't exist — <env>
# is just a label picking which env file to load (no GPU/memory tiers here,
# unlike the sibling repos' dev|prd|gpu-dev|gpu-prd split).
#
# Daemon deploy uses --network host instead of `-p`: app.config's
# dashboard_host is 127.0.0.1 on purpose (POST /api/order has no auth, so it
# must not be LAN/internet-reachable). With Docker's default bridge network,
# `-p PORT:PORT` NATs to the container's bridge IP, not its loopback, so a
# 127.0.0.1-bound process inside the container is unreachable through it —
# even from the host itself. --network host makes the container's 127.0.0.1
# the same as the host's, so an SSH tunnel to the host's 127.0.0.1:PORT
# actually reaches it, while staying just as unreachable from the LAN/internet
# as before.

SERVICE="invest-project"
TARGET_ENV="${1:-dev}"
[ $# -gt 0 ] && shift
MODULE="${1:-daemon}"
[ $# -gt 0 ] && shift

ROOT_DIR="${WORKSPACE:-$(pwd)}"
ENV_FILE="$ROOT_DIR/.env.${TARGET_ENV}"
[ -f "$ENV_FILE" ] || ENV_FILE="$ROOT_DIR/.env"
[ -f "$ENV_FILE" ] || { echo "Error: no env file: .env.${TARGET_ENV} or .env"; exit 1; }

CONTAINER_NAME="${SERVICE}-${TARGET_ENV}"
IMAGE_NAME="$CONTAINER_NAME"
DASHBOARD_PORT="$(grep -E '^DASHBOARD_PORT=' "$ENV_FILE" | cut -d= -f2 || true)"
DASHBOARD_PORT="${DASHBOARD_PORT:-8094}"

# Parse capture keys once, rejecting aliases and duplicates before Docker runs.
# 비밀 env 파일을 실행·출력하지 않고 수집 키의 별칭과 중복을 먼저 거부한다.
if ! iconv -f UTF-8 -t UTF-8 -- "$ENV_FILE" > /dev/null; then
  echo "Error: invalid env file encoding" >&2
  exit 1
fi
CAPTURE_VALUES="$(LC_ALL=C awk '
  function unicode_leading_space(value, pair, triple, code) {
    pair = substr(value, 1, 2)
    triple = substr(value, 1, 3)
    if (pair == sprintf("%c%c", 194, 133) || pair == sprintf("%c%c", 194, 160)) return 1
    if (triple == sprintf("%c%c%c", 225, 154, 128)) return 1
    for (code = 128; code <= 138; code++) {
      if (triple == sprintf("%c%c%c", 226, 128, code)) return 1
    }
    return triple == sprintf("%c%c%c", 226, 128, 168) \
        || triple == sprintf("%c%c%c", 226, 128, 169) \
        || triple == sprintf("%c%c%c", 226, 128, 175) \
        || triple == sprintf("%c%c%c", 226, 129, 159) \
        || triple == sprintf("%c%c%c", 227, 128, 128)
  }
  {
    line = $0
    if (NR == 1 && substr(line, 1, 3) == sprintf("%c%c%c", 239, 187, 191)) {
      invalid = 1
      next
    }
    sub(/\r$/, "", line)
    candidate = line
    sub(/^[[:space:]]*/, "", candidate)
    # Reject Unicode leading whitespace that Docker would trim before key parsing.
    # Docker가 키 해석 전에 제거하는 유니코드 앞 공백은 먼저 거부한다.
    if (unicode_leading_space(candidate)) { invalid = 1; next }
    if (candidate ~ /^#/ || candidate == "") next
    if (toupper(candidate) ~ /^EXPORT[[:space:]]+RM_REQUEST_CAPTURE_/) {
      invalid = 1
      next
    }
    if (index(toupper(candidate), "RM_REQUEST_CAPTURE_") != 1) next
    if (candidate != line) { invalid = 1; next }
    equals = index(candidate, "=")
    if (!equals) { invalid = 1; next }
    key = substr(candidate, 1, equals - 1)
    value = substr(candidate, equals + 1)
    if (key !~ /^RM_REQUEST_CAPTURE_(ENABLED|DIRECTORY|HOST_DIRECTORY|EVENT_MAX_BYTES|QUEUE_MAX_BYTES|SESSION_MAX_BYTES)$/ || key in seen || value == "") { invalid = 1; next }
    seen[key] = 1
    print key "=" value
  }
  END { if (invalid) exit 2 }
' "$ENV_FILE")" || { echo "Error: invalid capture configuration" >&2; exit 1; }
declare -A CAPTURE_ENV=()
if [ -n "$CAPTURE_VALUES" ]; then
  while IFS='=' read -r key value; do
    CAPTURE_ENV["$key"]="$value"
  done <<< "$CAPTURE_VALUES"
fi

CAPTURE_ENABLED="${CAPTURE_ENV[RM_REQUEST_CAPTURE_ENABLED]-false}"
if [ "$CAPTURE_ENABLED" != "true" ] && [ "$CAPTURE_ENABLED" != "false" ]; then
  echo "Error: invalid capture_enabled value" >&2
  exit 1
fi
for CAPTURE_LIMIT in EVENT_MAX_BYTES QUEUE_MAX_BYTES SESSION_MAX_BYTES; do
  CAPTURE_KEY="RM_REQUEST_CAPTURE_${CAPTURE_LIMIT}"
  CAPTURE_VALUE="${CAPTURE_ENV[$CAPTURE_KEY]-}"
  if [ -n "$CAPTURE_VALUE" ] && ! [[ "$CAPTURE_VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: invalid capture numeric setting" >&2
    exit 1
  fi
done
CAPTURE_MOUNT_ARGS=()
if [ "$CAPTURE_ENABLED" = "true" ]; then
  if [ "$MODULE" != "daemon" ]; then
    echo "Error: capture is only supported for the persistent daemon" >&2
    exit 1
  fi
  CAPTURE_DIR="${CAPTURE_ENV[RM_REQUEST_CAPTURE_DIRECTORY]-}"
  CAPTURE_HOST_DIR="${CAPTURE_ENV[RM_REQUEST_CAPTURE_HOST_DIRECTORY]-}"
  if [ "${CAPTURE_ENV[RM_REQUEST_CAPTURE_EVENT_MAX_BYTES]-2097152}" != "2097152" ] \
     || [ "${CAPTURE_ENV[RM_REQUEST_CAPTURE_QUEUE_MAX_BYTES]-67108864}" != "67108864" ] \
     || [ "${CAPTURE_ENV[RM_REQUEST_CAPTURE_SESSION_MAX_BYTES]-536870912}" != "536870912" ]; then
    echo "Error: unapproved capture limits" >&2
    exit 1
  fi
  CAPTURE_DEST="/var/lib/invest-eval/rm-capture/current"
  case "$CAPTURE_HOST_DIR" in
    /var/lib/invest-eval/rm-capture/*) ;;
    *) echo "Error: capture host directory is outside the approved root" >&2; exit 1 ;;
  esac
  if [ "$CAPTURE_DIR" != "$CAPTURE_DEST" ] || [ ! -d "$CAPTURE_HOST_DIR" ] \
     || [ -L "$CAPTURE_HOST_DIR" ] || [[ "$CAPTURE_HOST_DIR" == *','* ]] \
     || [[ "$CAPTURE_HOST_DIR" == *':'* ]] \
     || [ "$(readlink -e -- "$CAPTURE_HOST_DIR")" != "$CAPTURE_HOST_DIR" ]; then
    echo "Error: capture mount is invalid or missing" >&2
    exit 1
  fi
  if [ "$(stat -c %a -- "$CAPTURE_HOST_DIR")" != "700" ] \
     || [ "$(stat -c %u -- "$CAPTURE_HOST_DIR")" != "0" ]; then
    echo "Error: capture host directory owner or mode is invalid" >&2
    exit 1
  fi
  CAPTURE_ROOT="/var/lib/invest-eval/rm-capture"
  if [ "$(findmnt -n -o TARGET --target "$CAPTURE_HOST_DIR")" != "$CAPTURE_ROOT" ]; then
    echo "Error: capture root is not a dedicated filesystem" >&2
    exit 1
  fi
  CAPTURE_TOTAL="$(df -B1 --output=size -- "$CAPTURE_HOST_DIR" | tail -n 1 | tr -d ' ')"
  CAPTURE_USED="$(du -sb -- "$CAPTURE_ROOT" | awk '{print $1}')"
  CAPTURE_AVAILABLE="$(df -B1 --output=avail -- "$CAPTURE_HOST_DIR" | tail -n 1 | tr -d ' ')"
  if ! [[ "$CAPTURE_TOTAL" =~ ^[0-9]+$ && "$CAPTURE_USED" =~ ^[0-9]+$ && "$CAPTURE_AVAILABLE" =~ ^[0-9]+$ ]] \
     || [ "$CAPTURE_TOTAL" -gt 1073741824 ] \
     || [ "$CAPTURE_USED" -gt 444596224 ] || [ "$CAPTURE_AVAILABLE" -lt 629145600 ]; then
    echo "Error: capture storage capacity is insufficient" >&2
    exit 1
  fi
  CAPTURE_MOUNT_ARGS=(--mount "type=bind,src=$CAPTURE_HOST_DIR,dst=$CAPTURE_DEST")
fi

cd "$ROOT_DIR"

# KIS issues its OAuth access token as a once-per-day resource and warns it
# will restrict an app key that re-issues repeatedly within a token's ~24h
# life. The app caches the token on disk (app/brokers/kis/auth.py) so one
# token serves every process and survives restarts — but only if that file
# outlives the container. The daemon does `docker stop && rm && run`, and a
# one-shot run is `--rm`, so without a volume every deploy and every manual
# run boots cache-less and issues a brand-new token. A named volume (not a
# host path — survives `docker rm`, no host layout assumptions) mounted at a
# fixed dir, with KIS_TOKEN_CACHE_DIR pointing the app there, keeps a single
# token alive across all of them.
# KIS 토큰은 하루 1회 발급 원칙 — 볼륨으로 캐시를 컨테이너 밖에 유지해 재배포마다
# 재발급되는 것을 막는다.
# Volume name is deliberately keyed to $SERVICE, NOT $CONTAINER_NAME — do not
# "fix" it to ${CONTAINER_NAME}-kis-token. The on-disk cache is keyed per KIS
# app key (token_cache_path_for), and the same app key should share one token
# across every env (dev/prd/...) rather than issue a separate one per env —
# KIS restricts frequent re-issuance. One volume per service, shared by envs.
# 볼륨명은 env가 아니라 서비스 단위 — 같은 앱키는 env 무관하게 토큰 1개 공유.
KIS_TOKEN_VOLUME="${SERVICE}-kis-token"
KIS_TOKEN_CACHE_DIR_IN_CONTAINER="/var/cache/kis"

# Stamps the built image with the commit it was built from, so every decision
# and backtest run it records carries its code version (see the Dockerfile's
# own ARG GIT_COMMIT comment for why this can't be resolved inside the
# container). Resolved HERE, on the build host, because that is the only place
# a git checkout exists — Jenkins runs this script straight after its own
# `git` checkout step, so it picks this up with no Jenkins Job change needed.
# Empty on a non-checkout build: the tagging degrades to NULL exactly as
# before rather than failing the deploy.
# 빌드 호스트에서만 git을 쓸 수 있으므로 여기서 커밋을 뽑아 이미지에 굽는다.
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || true)"
echo "$CONTAINER_NAME | building image | git_commit:${GIT_COMMIT:-(unavailable)}"
docker build --build-arg GIT_COMMIT="$GIT_COMMIT" -t "$IMAGE_NAME" .

if [ "$CAPTURE_ENABLED" = "true" ]; then
  IMAGE_USER="$(docker image inspect --format '{{.Config.User}}' "$IMAGE_NAME")"
  if [ -n "$IMAGE_USER" ] && [ "$IMAGE_USER" != "0" ] && [ "$IMAGE_USER" != "root" ]; then
    echo "Error: capture directory owner does not match the image user" >&2
    exit 1
  fi
fi

if [ "$MODULE" = "daemon" ]; then
  echo "$CONTAINER_NAME | env:$ENV_FILE | port:$DASHBOARD_PORT | deploying persistent dashboard"
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  docker rm   "$CONTAINER_NAME" 2>/dev/null || true
  docker run -d \
    --name "$CONTAINER_NAME" \
    --env-file "$ENV_FILE" \
    --network host \
    --restart unless-stopped \
    -v "${KIS_TOKEN_VOLUME}:${KIS_TOKEN_CACHE_DIR_IN_CONTAINER}" \
    -e KIS_TOKEN_CACHE_DIR="$KIS_TOKEN_CACHE_DIR_IN_CONTAINER" \
    "${CAPTURE_MOUNT_ARGS[@]}" \
    "$IMAGE_NAME"
  echo "done - $CONTAINER_NAME (running on port $DASHBOARD_PORT — 'docker logs -f $CONTAINER_NAME' to follow)"
else
  echo "$IMAGE_NAME | env:$ENV_FILE | module:app.$MODULE $*"
  docker run --rm \
    --env-file "$ENV_FILE" \
    -v "${KIS_TOKEN_VOLUME}:${KIS_TOKEN_CACHE_DIR_IN_CONTAINER}" \
    -e KIS_TOKEN_CACHE_DIR="$KIS_TOKEN_CACHE_DIR_IN_CONTAINER" \
    "$IMAGE_NAME" \
    uv run python -m "app.$MODULE" "$@"
  echo "done - $IMAGE_NAME"
fi
