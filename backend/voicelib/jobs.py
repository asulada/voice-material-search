"""后台长任务队列（方案 §6.3）。

* ThreadPoolExecutor(max_workers=1)：单任务并发，保证检索始终可响应
* 进度帧：{status, done, total, message, errors, result}
* 任务快照落盘到 SQLite jobs 表，重启后仍能看到历史
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .db import Database

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # running | done | failed | cancelled
    done: int = 0
    total: int = 0
    message: str = ""
    errors: list[str] = field(default_factory=list)
    result: Any = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "message": self.message,
            "errors": self.errors[-20:],
            "error_count": len(self.errors),
            "result": self.result,
            "elapsed_sec": round(
                (self.finished_at or time.time()) - self.created_at, 1
            ),
        }


class JobManager:
    def __init__(self, db: Database, max_workers: int = 1):
        self.db = db
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load_history()

    def _load_history(self) -> None:
        try:
            for rec in self.db.load_jobs()[:50]:
                job = Job(id=rec["id"], kind="import", status=rec["status"])
                job.done = rec.get("done") or 0
                job.total = rec.get("total") or 0
                job.errors = rec.get("errors") or []
                job.result = rec.get("result")
                if job.status == "running":
                    # 上次进程退出时未完成的任务
                    job.status = "failed"
                    job.message = "服务重启，任务中断"
                self._jobs[job.id] = job
        except Exception as exc:  # noqa: BLE001
            log.warning("读取任务历史失败: %s", exc)

    # ------------------------------------------------------------------ API

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.snapshot() for j in jobs]

    def submit(
        self,
        kind: str,
        fn: Callable[[Job], Any],
        *,
        total: int = 0,
        message: str = "",
    ) -> Job:
        """立即返回 job；fn 在单线程池里跑，通过 job 更新进度。"""
        job = Job(id=uuid.uuid4().hex, kind=kind, total=total, message=message)
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)
        self.executor.submit(self._wrap, job, fn)
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status != "running":
            return False
        job.cancel.set()
        job.message = "正在取消…"
        return True

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------- internal

    def _persist(self, job: Job) -> None:
        try:
            self.db.save_job(
                {
                    "id": job.id,
                    "status": job.status,
                    "done": job.done,
                    "total": job.total,
                    "errors": job.errors,
                    "result": job.result,
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("任务快照落盘失败: %s", exc)

    def _wrap(self, job: Job, fn: Callable[[Job], Any]) -> None:
        last_persist = 0.0
        try:
            result = fn(job)
            if job.cancel.is_set():
                job.status = "cancelled"
                job.message = job.message or "已取消"
            else:
                job.status = "done"
                job.result = result
                job.message = job.message or "完成"
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.message = f"{type(exc).__name__}: {exc}"
            job.errors.append(traceback.format_exc(limit=5))
            log.exception("任务 %s 失败", job.id)
        finally:
            job.finished_at = time.time()
            if job.total and job.done < job.total and job.status == "done":
                job.done = job.total
            now = time.time()
            if now - last_persist > 0.5 or job.status != "running":
                self._persist(job)


# --------------------------------------------------------------- 进度推送


class JobProgress:
    """给执行函数用的进度句柄：限制刷新频率，避免刷爆 WebSocket。"""

    def __init__(self, job: Job, min_interval: float = 0.2):
        self.job = job
        self.min_interval = min_interval
        self._last = 0.0

    def set_total(self, total: int) -> None:
        self.job.total = total

    def update(self, done: int, total: int | None = None, message: str | None = None) -> None:
        self.job.done = done
        if total is not None:
            self.job.total = total
        if message is not None:
            self.job.message = message
        now = time.time()
        if now - self._last >= self.min_interval:
            self._last = now

    def error(self, msg: str) -> None:
        self.job.errors.append(msg)


async def stream_job(job_getter: Callable[[], Job | None], interval: float = 0.3):
    """WebSocket 帧生成器（§6.3）：done 变化或状态终结时推送。"""
    last_done = -1
    last_status = ""
    while True:
        job = job_getter()
        if job is None:
            yield {"error": "unknown job"}
            return
        snap = job.snapshot()
        if snap["done"] != last_done or snap["status"] != last_status:
            yield snap
            last_done = snap["done"]
            last_status = snap["status"]
        if snap["status"] in ("done", "failed", "cancelled"):
            return
        await asyncio.sleep(interval)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
