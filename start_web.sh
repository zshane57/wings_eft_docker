#!/bin/bash
# --------------------------------------------------------------
# start_web.sh — start the browser frontend (flywing-web container)
#
# Builds the small web-server image (cached after the first time),
# starts it with the Docker socket and this package directory
# mounted, and opens the page in your browser:
#
#     http://localhost:8080
#
# The server keeps running until you stop it:
#     docker rm -f flywing-web
# (the SAM3 weights volume and the build caches are kept)
# --------------------------------------------------------------
set -e
cd "$(dirname "$0")"

IMAGE="flywing-web"
PORT=8080

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker was not found on this machine."; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker is not running. Start Docker Desktop first, then re-run."; exit 1
fi

echo "=== Building the web frontend image (cached layers keep this quick) ==="
docker build -f Dockerfile.web -t "$IMAGE" .

docker rm -f "$IMAGE" >/dev/null 2>&1 || true
docker run -d --name "$IMAGE" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD:/app/work" \
  -e "WORK_HOST_PATH=$PWD" \
  -p "127.0.0.1:$PORT:$PORT" \
  "$IMAGE"

echo
echo "Web frontend running at http://localhost:$PORT"
echo "Stop it with:   docker rm -f $IMAGE"

# open the browser (macOS: open, Linux: xdg-open, WSL: wslview)
sleep 1
if command -v open >/dev/null 2>&1; then
  open "http://localhost:$PORT" >/dev/null 2>&1 || true
elif command -v wslview >/dev/null 2>&1; then
  wslview "http://localhost:$PORT" >/dev/null 2>&1 || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || true
fi
