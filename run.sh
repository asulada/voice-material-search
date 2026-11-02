#!/usr/bin/env bash
# 启动后端（方案 §6.7）。只监听 127.0.0.1 —— 这个服务能读本地已入库音频。
#
#   ./run.sh                 # 用默认库根目录（项目根 / voice_library）
#   VOICE_LIBRARY_ROOT=... ./run.sh
#   PORT=8000 ./run.sh
set -euo pipefail

cd "$(dirname "$0")/backend"

PY="${PY:-python3}"
PORT="${PORT:-8000}"

if ! "$PY" -c "import fastapi" >/dev/null 2>&1; then
  echo "缺少依赖。先执行：$PY -m pip install -r requirements.txt" >&2
  exit 1
fi

exec "$PY" -m uvicorn server:app --host 127.0.0.1 --port "$PORT" "$@"
