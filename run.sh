#!/usr/bin/env bash
# 启动后端（方案 §6.7）。只监听 127.0.0.1 —— 这个服务能读本地已入库音频。
#
#   ./run.sh                 # 用默认库根目录（项目根 / voice_library）
#   VOICE_LIBRARY_ROOT=... ./run.sh
#   PORT=8000 ./run.sh
set -euo pipefail

cd "$(dirname "$0")/backend"

# 优先用 PATH 里的 python（conda / venv 激活后就是环境里的那个），
# 再退回 python3。显式设置 PY=... 可覆盖。
if [ -n "${PY:-}" ]; then
  :
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi
PORT="${PORT:-8000}"

if ! "$PY" -c "import fastapi" >/dev/null 2>&1; then
  echo "缺少依赖。先激活环境（conda activate / source .venv/bin/activate），再执行：" >&2
  echo "  $PY -m pip install -r requirements.txt" >&2
  exit 1
fi

exec "$PY" -m uvicorn server:app --host 127.0.0.1 --port "$PORT" "$@"
