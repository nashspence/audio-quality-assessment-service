#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspaces/titanet-speaker-id-service

cd "${PROJECT_ROOT}"

for cmd in docker jq python3 git gh nvcc nvidia-smi; do
  command -v "${cmd}" >/dev/null
done

python3 --version
gh --version | head -n 1
nvcc --version | tail -n 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker compose version
test -S /var/run/docker.sock
docker ps >/dev/null
docker run --rm alpine:3.22 true
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 \
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

# shellcheck disable=SC1091
source .env
API_BASE_URL="${API_BASE_URL:-http://host.docker.internal:${SERVICE_PORT}}"

cleanup() {
  docker compose down -v --remove-orphans
}

trap cleanup EXIT

docker compose down -v --remove-orphans >/dev/null 2>&1 || true
docker compose up --build -d

container_id="$(docker compose ps -q api)"
if [[ -z "${container_id}" ]]; then
  echo "API container did not start." >&2
  exit 1
fi

for _ in $(seq 1 120); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health}" == "healthy" ]]; then
    break
  fi
  sleep 5
done

health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
if [[ "${health}" != "healthy" ]]; then
  docker compose logs api >&2
  echo "API container did not become healthy." >&2
  exit 1
fi

curl --retry 12 --retry-delay 2 --retry-connrefused -fsS "${API_BASE_URL}/healthz" | jq -e '
  .ready == true and
  .embedding_dimension > 0 and
  (.device | startswith("cuda"))
' >/dev/null

curl -fsS \
  -F label=test_person \
  -F archive=@test_person.zip \
  "${API_BASE_URL}/v1/enroll" \
  | tee /tmp/titanet-enroll.json \
  | jq -e '
    .speaker.label == "test_person" and
    .speaker.embedding_count > 0 and
    .speaker.centroid_dimension > 0
  ' >/dev/null

curl -fsS \
  -F file=@test.opus \
  "${API_BASE_URL}/v1/identify" \
  | tee /tmp/titanet-identify.json \
  | jq -e '
    (.predicted_label | type) == "string" and
    .predicted_label != "" and
    (.scores | length) > 0 and
    (.scores[0].label | type) == "string" and
    (.scores[0].score | type) == "number"
  ' >/dev/null

curl -fsS \
  -F label=test_person \
  -F file=@test.opus \
  "${API_BASE_URL}/v1/verify" \
  | tee /tmp/titanet-verify.json \
  | jq -e '
    .speaker_label == "test_person" and
    (.score | type) == "number" and
    (.match | type) == "boolean"
  ' >/dev/null

docker compose down -v --remove-orphans
trap - EXIT
