"""SQLite 数据层：建表、CRUD、统计。

Schema 严格按方案 §2.2。为了后续能平滑加列，启动时会做一次轻量迁移
（PRAGMA table_info + ALTER TABLE ADD COLUMN）。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id              INTEGER PRIMARY KEY,
    filepath        TEXT NOT NULL UNIQUE,
    checksum        TEXT NOT NULL,
    source_file     TEXT,
    start_sec       REAL,

    target_use      TEXT NOT NULL,
    clip_kind       TEXT NOT NULL,

    category        TEXT NOT NULL,
    category_raw    TEXT,
    intensity       INTEGER,

    speaker_id      TEXT,
    gender          TEXT,
    language        TEXT,

    duration_sec    REAL NOT NULL,
    sample_rate     INTEGER,
    loudness_lufs   REAL,
    gain_db         REAL,
    has_background  INTEGER DEFAULT 0,
    studio_quality  INTEGER DEFAULT 0,

    transcript      TEXT,
    description     TEXT NOT NULL,
    embedding       BLOB,

    license         TEXT NOT NULL,
    commercial_ok   INTEGER NOT NULL,

    verified        INTEGER DEFAULT 0,
    notes           TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_use       ON clips(target_use, clip_kind);
CREATE INDEX IF NOT EXISTS idx_filter    ON clips(category, gender, commercial_ok);
CREATE INDEX IF NOT EXISTS idx_duration  ON clips(duration_sec);
CREATE INDEX IF NOT EXISTS idx_verified  ON clips(verified, target_use);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    done        INTEGER DEFAULT 0,
    total       INTEGER DEFAULT 0,
    errors      TEXT DEFAULT '[]',
    result      TEXT,
    started_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# 方案 §2.2 的首版列；后续版本新增的列放这里做兜底迁移
_MIGRATIONS: dict[str, str] = {}


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    # ---------------- 连接 ----------------
    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init(self) -> None:
        with self.tx() as conn:
            conn.executescript(SCHEMA)
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(clips)")}
            for name, ddl in _MIGRATIONS.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE clips ADD COLUMN {name} {ddl}")

    # ---------------- clips ----------------
    def get_clip(self, clip_id: int) -> sqlite3.Row | None:
        with self.read() as conn:
            return conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()

    def get_by_filepath(self, filepath: str) -> sqlite3.Row | None:
        with self.read() as conn:
            return conn.execute(
                "SELECT * FROM clips WHERE filepath = ?", (filepath,)
            ).fetchone()

    @staticmethod
    def _insert_sql(fields: Sequence[str]) -> str:
        cols = ", ".join(fields)
        qs = ", ".join("?" for _ in fields)
        return f"INSERT INTO clips ({cols}) VALUES ({qs})"

    # 人工精修的列：重复导入时保留库里的值，不被模板重新生成的内容覆盖。
    # 这与方案 §2.3 / §7 阶段四的意图一致——人工描述是检索质量的上限，
    # 且一旦被模板覆盖就不可逆。
    HUMAN_FIELDS = frozenset({"verified", "intensity", "description", "notes"})

    def upsert_clip(
        self,
        row: dict[str, Any],
        conn: sqlite3.Connection | None = None,
        preserve_human: bool = True,
    ) -> int:
        """按 filepath 幂等写入；已存在则更新。返回 clip id。"""
        fields = [k for k in row]
        values = [row[k] for k in fields]
        sql = self._insert_sql(fields)

        def assign(key: str) -> str:
            if preserve_human and key in self.HUMAN_FIELDS:
                return f"{key} = clips.{key}"
            return f"{key} = excluded.{key}"

        updates = ", ".join(assign(k) for k in fields if k != "filepath")
        sql += f" ON CONFLICT(filepath) DO UPDATE SET {updates}"

        own = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(sql, values)
            got = conn.execute(
                "SELECT id FROM clips WHERE filepath = ?", (row["filepath"],)
            ).fetchone()
            if own:
                conn.commit()
            return int(got["id"]) if got else 0
        finally:
            if own:
                conn.close()

    def update_clip(self, clip_id: int, patch: dict[str, Any]) -> bool:
        allowed = {
            "verified",
            "intensity",
            "description",
            "notes",
            "category",
            "gender",
            "has_background",
            "studio_quality",
            "commercial_ok",
            "license",
            "transcript",
            "target_use",
        }
        sets = {k: v for k, v in patch.items() if k in allowed}
        if not sets:
            return False
        sql = "UPDATE clips SET " + ", ".join(f"{k} = ?" for k in sets) + " WHERE id = ?"
        with self.tx() as conn:
            cur = conn.execute(sql, [*sets.values(), clip_id])
            return cur.rowcount > 0

    def set_embedding(
        self, clip_id: int, embedding: bytes, description: str | None = None
    ) -> bool:
        with self.tx() as conn:
            if description is None:
                cur = conn.execute(
                    "UPDATE clips SET embedding = ? WHERE id = ?", (embedding, clip_id)
                )
            else:
                cur = conn.execute(
                    "UPDATE clips SET embedding = ?, description = ? WHERE id = ?",
                    (embedding, description, clip_id),
                )
            return cur.rowcount > 0

    def set_embedding_many(self, pairs: Iterable[tuple[int, bytes]]) -> None:
        with self.tx() as conn:
            conn.executemany(
                "UPDATE clips SET embedding = ? WHERE id = ?",
                [(emb, cid) for cid, emb in pairs],
            )

    def delete_clip(self, clip_id: int) -> sqlite3.Row | None:
        row = self.get_clip(clip_id)
        if row is None:
            return None
        with self.tx() as conn:
            conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
        return row

    def iter_clips(self, **where: Any) -> list[sqlite3.Row]:
        sql = "SELECT * FROM clips"
        params: list[Any] = []
        if where:
            sql += " WHERE " + " AND ".join(f"{k} = ?" for k in where)
            params = list(where.values())
        with self.read() as conn:
            return conn.execute(sql + " ORDER BY id", params).fetchall()

    def count(self) -> int:
        with self.read() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM clips").fetchone()
            return int(row["n"])

    def stats(self) -> dict[str, Any]:
        with self.read() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM clips").fetchone()["n"]
            verified = conn.execute(
                "SELECT COUNT(*) AS n FROM clips WHERE verified = 1"
            ).fetchone()["n"]
            dur = conn.execute(
                "SELECT COALESCE(SUM(duration_sec), 0) AS s FROM clips"
            ).fetchone()["s"]

            def group(col: str) -> dict[str, int]:
                rows = conn.execute(
                    f"SELECT {col} AS k, COUNT(*) AS n FROM clips "
                    f"GROUP BY {col} ORDER BY n DESC"
                ).fetchall()
                return {(str(r["k"]) if r["k"] is not None else "(未知)"): r["n"] for r in rows}

            use_rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT target_use, clip_kind, COUNT(*) AS n FROM clips "
                    "GROUP BY target_use, clip_kind"
                ).fetchall()
            ]
            embedded = conn.execute(
                "SELECT COUNT(*) AS n FROM clips WHERE embedding IS NOT NULL"
            ).fetchone()["n"]
            commercial = conn.execute(
                "SELECT COUNT(*) AS n FROM clips WHERE commercial_ok = 1"
            ).fetchone()["n"]
            # 注意：分组必须在 with 块内算完 —— 闭包里的 conn 出了块就关掉了
            by_category = group("category")
            by_gender = group("gender")
            by_license = group("license")
            by_language = group("language")

        return {
            "total": total,
            "verified": verified,
            "embedded": embedded,
            "commercial_ok": commercial,
            "total_duration_sec": round(float(dur), 1),
            "by_use": [
                {
                    "target_use": r["target_use"],
                    "clip_kind": r["clip_kind"],
                    "count": r["n"],
                }
                for r in use_rows
            ],
            "by_category": by_category,
            "by_gender": by_gender,
            "by_license": by_license,
            "by_language": by_language,
        }

    # ---------------- jobs ----------------
    def save_job(self, job: dict[str, Any]) -> None:
        import json

        with self.tx() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, done, total, errors, result) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
                "done=excluded.done, total=excluded.total, "
                "errors=excluded.errors, result=excluded.result",
                (
                    job["id"],
                    job["status"],
                    job.get("done", 0),
                    job.get("total", 0),
                    json.dumps(job.get("errors", []), ensure_ascii=False),
                    json.dumps(job.get("result"), ensure_ascii=False)
                    if job.get("result") is not None
                    else None,
                ),
            )

    def load_jobs(self) -> list[dict[str, Any]]:
        import json

        with self.read() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY started_at DESC").fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "status": r["status"],
                    "done": r["done"],
                    "total": r["total"],
                    "errors": json.loads(r["errors"] or "[]"),
                    "result": json.loads(r["result"]) if r["result"] else None,
                }
            )
        return out
