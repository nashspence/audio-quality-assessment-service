#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspaces/audio-segment-quality-assessment

cd "${PROJECT_ROOT}"

for cmd in docker jq curl python3; do
  command -v "${cmd}" >/dev/null
done

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

python3 scripts/generate_sample_audio.py

# shellcheck disable=SC1091
source .env
API_BASE_URL="${API_BASE_URL:-http://host.docker.internal:${SERVICE_PORT:-8000}}"

cleanup() {
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker compose down -v --remove-orphans >/dev/null 2>&1 || true
docker compose up --build -d

container_id="$(docker compose ps -q api)"
if [[ -z "${container_id}" ]]; then
  echo "API container did not start." >&2
  exit 1
fi

for _ in $(seq 1 60); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health}" == "healthy" ]]; then
    break
  fi
  sleep 2
done

health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
if [[ "${health}" != "healthy" ]]; then
  docker compose logs api >&2
  echo "API container did not become healthy." >&2
  exit 1
fi

curl --retry 10 --retry-delay 1 --retry-connrefused -fsS "${API_BASE_URL}/healthz" | jq -e '
  .ready == true and
  (.service | type) == "string" and
  (.version | type) == "string"
' >/dev/null

response_file="$(mktemp)"
schema_file="$(mktemp)"
request_json="$(jq -c . examples/smoke-request.json)"

curl --retry 10 --retry-delay 1 --retry-connrefused -fsS "${API_BASE_URL}/v1/schema" > "${schema_file}"
jq -e '
  .transport.file_field == "file" and
  .request_payload_json_schema.type == "object" and
  .response_json_schema.type == "object"
' "${schema_file}" >/dev/null

curl --retry 10 --retry-delay 1 --retry-connrefused -fsS \
  -F "file=@examples/smoke-sample.wav;type=audio/wav" \
  -F "request_json=${request_json}" \
  "${API_BASE_URL}/v1/analyze" \
  > "${response_file}"

jq -e '
  .analysis_target == "file" and
  (.file_summary | type) == "object" and
  (.segments | length) == 2 and
  (.segments[0].quality_metrics | type) == "object" and
  (.segments[0].quality_assessment | type) == "object" and
  (.segments[0].confidence | type) == "object" and
  (.segments[0].quality_metrics.speech_ratio | type) == "number" and
  (.segments[0].quality_assessment.overlap_risk.score | type) == "number" and
  (.segments[0].confidence.overall_confidence | type) == "number" and
  (.segments[0].confidence.reason_codes | type) == "array" and
  (.segments[1].confidence.contributing_factors | length) > 0
' "${response_file}" >/dev/null

docker compose down -v --remove-orphans >/dev/null
trap - EXIT

echo "Smoke test passed."
