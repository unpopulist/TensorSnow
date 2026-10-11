# Sourced by start.sh, stop.sh and docker/common.sh: reads the site settings from deploy/redsnow/.env (or the file
# named by REDSNOW_ENV), if there is one. Copy .env.example to .env and fill it in. A variable already set in the
# environment wins over the file, so `CONTEXT=131072 ./start.sh` still works with a .env in place.
_redsnow_env="${REDSNOW_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env}"
if [[ -f "$_redsnow_env" ]]; then
  _redsnow_saved=$(export -p | grep -v -E '^declare -[a-zA-Z]*r')
  set -a
  # shellcheck source=/dev/null
  source "$_redsnow_env"
  set +a
  eval "$_redsnow_saved"
  unset _redsnow_saved
elif [[ -n "${REDSNOW_ENV:-}" ]]; then
  printf 'REDSNOW_ENV=%s does not exist\n' "$REDSNOW_ENV" >&2
  exit 1
fi
unset _redsnow_env
