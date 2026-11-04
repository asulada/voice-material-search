"""SQLite 数据层：建表、CRUD、统计。

Schema 严格按方案 §2.2。为了后续能平滑加列，启动时会做一次轻量迁移
（PRAGMA table_info + ALTER TABLE ADD COLUMN）。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

log = logging.getLogger(__name__)

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

-- 用户标注：与 clips 里的「程序标注」分开存，读取时用户值优先。
--
-- 为什么单独一张表而不是给每个字段加 *_user 列：
--   1. 导入只写 clips，完全不碰这张表 —— 「重复导入不覆盖用户标注」
--      是**结构上成立**的，不依赖 HUMAN_FIELDS 白名单列举得全不全。
--   2. 加字段不用 ALTER TABLE 加列，insert 一行即可。
--   3. 能一眼看出「哪些字段被人改过」，供界面打标。
--
-- clip_id + field 唯一：同一字段只保留最后一次人工输入。
CREATE TABLE IF NOT EXISTS annotations (
    clip_id     INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    field       TEXT NOT NULL,
    value       TEXT,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (clip_id, field)
);

CREATE INDEX IF NOT EXISTS idx_anno_field ON annotations(field);

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

# ---------------------------------------------------------------- 用户标注合并
#
# 用户标注存在 annotations(field, value) 里，读取时**用户值优先**。
# 为什么不在读取后于 Python 里覆盖：L1 结构化过滤必须在 SQL 里
# （否则 top_k 会被悄悄缩小，见 test_top_k_not_silently_shrunk_after_filtering）。
# 若筛选查 clips 原值而显示用用户值，就会出现「按中文筛出来的却是英文」。
#
# 类型必须显式 CAST：annotations.value 是 TEXT，不转的话
# `intensity >= 3` 会按 TEXT 比较，'10' < '3'。
ANNOTATABLE_FIELDS: dict[str, str] = {
    "category": "TEXT",
    "gender": "TEXT",
    "language": "TEXT",
    "transcript": "TEXT",
    "description": "TEXT",
    "notes": "TEXT",
    "license": "TEXT",
    "intensity": "INTEGER",
    "verified": "INTEGER",
    "has_background": "INTEGER",
    "studio_quality": "INTEGER",
    "commercial_ok": "INTEGER",
}

MERGED_VIEW = "clips_merged"


def _merged_view_sql(columns: Sequence[str]) -> str:
    """生成 clips_merged 视图：每个可标注字段用 LEFT JOIN 取用户值，缺省回落到程序值。

    列清单从 PRAGMA 读，这样以后给 clips 加列不必同步改这里
    （新列自动按「不可标注」处理，直接取程序值）。
    """
    select = []
    joins = []
    for i, col in enumerate(columns):
        kind = ANNOTATABLE_FIELDS.get(col)
        if kind is None:
            select.append(f"c.{col} AS {col}")
            continue
        alias = f"a{i}"
        taken = f"{alias}.value"
        if kind == "INTEGER":
            taken = f"CAST({alias}.value AS INTEGER)"
        select.append(f"COALESCE({taken}, c.{col}) AS {col}")
        joins.append(
            f"LEFT JOIN annotations {alias} "
            f"ON {alias}.clip_id = c.id AND {alias}.field = '{col}'"
        )
    return (
        f"CREATE VIEW {MERGED_VIEW} AS SELECT " + ", ".join(select) +
        # 已人工标注的字段名，逗号分隔（空串 = 无）。
        # 放在视图里而不是逐条再查：检索一页 20 条，
        # 每条单独查一次标注就是 N+1，列表页会明显变慢。
        # ORDER BY 不能省：GROUP_CONCAT 的顺序本来不保证，
        # 不排序的话同一批数据每次刷新列出来的顺序都可能变。
        ", COALESCE((SELECT GROUP_CONCAT(f.field, ',') FROM ("
        "SELECT field FROM annotations WHERE clip_id = c.id ORDER BY field) f"
        "), '') AS annotated_fields"
        " FROM clips c " + " ".join(joins)
    )


# SQLite 的锁等待与重试参数。
# 背景：WAL 模式下写锁是「库级」的，同一时刻只能有一个写事务；
# 导入任务会开一个长写事务（批量 upsert），此时任何写请求都必须排队。
# timeout 只控制单次等待；长事务超过它就会抛 "database is locked"，
# 所以 write() 里还要有有限次重试，把偶发冲突变成「慢一点」而不是「报错」。
BUSY_TIMEOUT_MS = 30_000
WRITE_RETRIES = 5
WRITE_RETRY_BASE_DELAY = 0.05


def _is_locked(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wal_ok = False
        self._init()

    # ---------------- 连接 ----------------
    def connect(self, retry_wal: bool = True) -> sqlite3.Connection:
        """新建连接。

        重要：这里**不能**执行 `PRAGMA journal_mode=WAL`。
        journal_mode 是持久化到数据库文件的属性，只需设置一次；
        而每次连接都执行它，等价于要求一次独占写锁——
        只要另一处正在写（导入任务），新建连接就会立刻抛
        "database is locked"。实测：持锁期间新连接设 WAL 会失败，
        但直接读、写（带等待）都正常。见 tests/test_db_lock.py。
        """
        conn = sqlite3.connect(
            self.path,
            timeout=BUSY_TIMEOUT_MS / 1000,
            check_same_thread=False,
            isolation_level=None,  # 自己管事务，避免隐式 BEGIN 把读也变成写
        )
        conn.row_factory = sqlite3.Row
        # 这两个是**连接级** pragma，必须每连接设置
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        # 上次没切成 WAL 的话，借这次机会再试（此时大概率已无锁）。
        # retry_wal=False 供 _init/_try_enable_wal 自己使用，避免递归。
        if retry_wal:
            self._ensure_wal()
        return conn

    @contextmanager
    def tx(self, retries: int = 0) -> Iterator[sqlite3.Connection]:
        """写事务。retries>0 时对 "database is locked" 做有限重试。"""
        attempt = 0
        while True:
            conn = self.connect()
            try:
                conn.execute("BEGIN IMMEDIATE")  # 立刻要写锁，避免升级时死锁
                yield conn
                conn.execute("COMMIT")
                return
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # noqa: BLE001 - 连接可能已不可用
                    pass
                if attempt < retries and _is_locked(exc):
                    attempt += 1
                    time.sleep(WRITE_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                    continue
                raise
            finally:
                conn.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """供所有写操作使用：带重试的写事务。"""
        with self.tx(retries=WRITE_RETRIES) as conn:
            yield conn

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """读事务。结束前必须 commit/rollback，否则 WAL 的读快照会一直占着，
        挡住写方的 checkpoint（表现为 WAL 文件无限增长 + 写越来越慢）。"""
        conn = self.connect()
        try:
            conn.execute("BEGIN")
            yield conn
        finally:
            try:
                conn.execute("COMMIT")
            except Exception:  # noqa: BLE001
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
            conn.close()

    def _init(self) -> None:
        # 1) journal_mode 是**持久化**属性，只需设一次（不是每个连接）。
        #    注意：它不能在事务内执行（会被忽略 / 报错），也不能和 executescript
        #    混用（后者会隐式 COMMIT），所以单独用一条裸连接完成。
        #
        #    这一步在「库还不是 WAL」（首次运行 / 旧库）时**需要写锁**，
        #    而导入任务可能正持有写锁。此时 PRAGMA 会立即返回 locked，
        #    busy_timeout 对它不起作用 —— 所以要手动等待重试。
        if not self._try_enable_wal():
            # 拿不到写锁就先按当前模式跑，**不永久降级**：
            # 下次有人调 connect() 时会再试一次（见 _ensure_wal）。
            log.warning("暂时无法启用 WAL，先以当前模式运行，稍后自动重试")

        # 2) 建表（幂等）
        conn = self.connect(retry_wal=False)
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

        # 3) 轻量迁移
        with self.tx(retries=WRITE_RETRIES) as conn:
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(clips)")}
            for name, ddl in _MIGRATIONS.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE clips ADD COLUMN {name} {ddl}")

        # 4) 重建用户标注合并视图。
        #    必须每次启动都重建：视图的列清单来自 clips 的当前结构，
        #    加过列（迁移）后旧视图会少列，查询直接报错。
        #    DROP + CREATE 放在同一个事务里，避免中间态被读到。
        with self.tx(retries=WRITE_RETRIES) as conn:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(clips)")]
            conn.execute(f"DROP VIEW IF EXISTS {MERGED_VIEW}")
            conn.execute(_merged_view_sql(cols))

    def _try_enable_wal(self) -> bool:
        """尝试把库切到 WAL。返回是否成功；持锁时重试有限次后返回 False。"""
        for attempt in range(WRITE_RETRIES):
            conn = self.connect(retry_wal=False)
            try:
                mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
                if mode == "wal":
                    self._wal_ok = True
                    return True
                return False
            except sqlite3.OperationalError as exc:
                if not _is_locked(exc):
                    log.warning("切换到 WAL 失败：%s", exc)
                    return False
                time.sleep(WRITE_RETRY_BASE_DELAY * (2**attempt))
            finally:
                conn.close()
        return False

    def _ensure_wal(self) -> None:
        """若还没切到 WAL，借一次连接的机会再试（持锁窗口过去之后就会成功）。"""
        # 测试里会用 Database.__new__ 绕过 __init__，这里容错
        if getattr(self, "_wal_ok", False):
            return
        if self._try_enable_wal():
            log.info("已启用 WAL 模式")

    # ---------------- clips ----------------
    def get_clip(self, clip_id: int) -> sqlite3.Row | None:
        with self.read() as conn:
            return conn.execute("SELECT * FROM clips_merged WHERE id = ?", (clip_id,)).fetchone()

    def get_clip_program(self, clip_id: int) -> sqlite3.Row | None:
        """只取程序值（clips 原表，不含用户标注）。

        重建描述时要用它：用户清空某字段的标注后，描述应按**程序值**重拼，
        若用合并值就等于把清空前的标注又写回去了。
        """
        with self.read() as conn:
            return conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()

    def get_by_filepath(self, filepath: str) -> sqlite3.Row | None:
        with self.read() as conn:
            return conn.execute(
                "SELECT * FROM clips_merged WHERE filepath = ?", (filepath,)
            ).fetchone()

    @staticmethod
    def _insert_sql(fields: Sequence[str]) -> str:
        cols = ", ".join(fields)
        qs = ", ".join("?" for _ in fields)
        return f"INSERT INTO clips ({cols}) VALUES ({qs})"

    # 人工标注不存在这里，而在 annotations 表（见文件开头的 SCHEMA）。
    # 导入只写本表，所以「重复导入覆盖程序标注、不覆盖用户标注」
    # 是结构性成立的，不需要白名单。
    def upsert_clip(
        self,
        row: dict[str, Any],
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """按 filepath 幂等写入；已存在则更新。返回 clip id。

        冲突时**所有列都取新值**（`excluded.*`）：用户要的是
        「重复导入覆盖程序标注」。人工值在 annotations 里，不受影响。
        """
        fields = [k for k in row]
        values = [row[k] for k in fields]
        sql = self._insert_sql(fields)

        updates = ", ".join(f"{k} = excluded.{k}" for k in fields if k != "filepath")
        sql += f" ON CONFLICT(filepath) DO UPDATE SET {updates}"

        own = conn is None
        conn = conn or self.connect()
        began = False
        if own:
            # 显式立刻拿写锁，而不是等第一条 INSERT 时才隐式升级
            conn.execute("BEGIN IMMEDIATE")
            began = True
        try:
            conn.execute(sql, values)
            got = conn.execute(
                "SELECT id FROM clips WHERE filepath = ?", (row["filepath"],)
            ).fetchone()
            if own:
                conn.execute("COMMIT")
            return int(got["id"]) if got else 0
        except Exception:
            if began:
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            if own:
                conn.close()

    # 这些字段写进 clips（程序标注，导入时会用新值覆盖）。
    # 注意 target_use 不在 ANNOTATABLE_FIELDS 里：它是线路归属，
    # 人工改了就该直接改 clips，导入时按数据源重判。
    PATCHABLE_FIELDS = frozenset({
        "target_use",
        *ANNOTATABLE_FIELDS,
    })

    def update_clip(
        self,
        clip_id: int,
        patch: dict[str, Any],
        *,
        program_keys: Iterable[str] = (),
    ) -> bool:
        """人工修改。**可标注字段写 annotations，其余写 clips。**

        为什么写到两张表：用户要求「重复导入覆盖程序标注，但不覆盖用户标注」。
        把人工值放进 annotations，导入只 Update clips，这条规则就是结构性的，
        不依赖白名单列举得全不全（旧实现靠 HUMAN_FIELDS 四字段，
        改 category 之类会被下次导入冲掉，描述与字段还会互相矛盾）。

        `program_keys` 里的字段**强制写 clips**，即使它本来可标注。
        用于「程序重建出来的值」：例如用户改了类别，描述被模板重新拼出来，
        这份描述是程序产物、不是用户手写的，写进 annotations 的话
        以后导入就再也刷不新它了。
        """
        forced = set(program_keys)
        wanted = {k: v for k, v in patch.items() if k in self.PATCHABLE_FIELDS}
        if not wanted:
            return False

        annotated = {
            k: v for k, v in wanted.items()
            if k in ANNOTATABLE_FIELDS and k not in forced
        }
        program = {
            k: v for k, v in wanted.items()
            if k not in ANNOTATABLE_FIELDS or k in forced
        }
        # 程序列大多是 NOT NULL（target_use 等），把它们设成 NULL 会直接
        # 抛 IntegrityError（表现为 500）。这里只允许清空**可标注**字段，
        # 程序列传 null 视为「没传」。
        program = {k: v for k, v in program.items() if v is not None}

        changed = False
        # 同一事务里两张表一起写，避免读到「字段改了但标注还没落」的中间态
        with self.write() as conn:
            if program:
                sets = ", ".join(f"{k} = ?" for k in program)
                cur = conn.execute(
                    f"UPDATE clips SET {sets} WHERE id = ?",
                    [*program.values(), clip_id],
                )
                changed = changed or cur.rowcount > 0
            for k, v in annotated.items():
                # None 表示「清掉人工标注、回落到程序值」，不是「把字段设成 NULL」
                if v is None:
                    cur = conn.execute(
                        "DELETE FROM annotations WHERE clip_id = ? AND field = ?",
                        (clip_id, k),
                    )
                    changed = changed or cur.rowcount > 0
                else:
                    conn.execute(
                        "INSERT INTO annotations(clip_id, field, value) VALUES (?,?,?) "
                        "ON CONFLICT(clip_id, field) DO UPDATE SET "
                        "value = excluded.value, updated_at = CURRENT_TIMESTAMP",
                        (clip_id, k, str(v)),
                    )
                    changed = True
        return changed

    def annotations_of(self, clip_id: int) -> dict[str, str]:
        """某条素材当前有哪些人工标注，供界面打「已手改」标记。"""
        with self.read() as conn:
            rows = conn.execute(
                "SELECT field, value FROM annotations WHERE clip_id = ?", (clip_id,)
            ).fetchall()
        return {r["field"]: r["value"] for r in rows}

    def clear_annotations(
        self,
        clip_ids: Sequence[int] | None = None,
        *,
        where_sql: str = "",
        where_params: Sequence[Any] = (),
        fields: Sequence[str] | None = None,
    ) -> int:
        """清空人工标注，返回清掉的字段条数。

        `clip_ids` 为空时用 `where_sql` 选一批（「清空当前筛选下的全部人工标注」）。
        **两个都为空时直接返回 0**：不提供「清空全库标注」这种无差别入口，
        误调用一次就把所有人工成果抹掉，代价不可逆。

        `fields` 非空时只清这几项（例如只撤销语种标注、保留手写描述）。
        """
        if not clip_ids and not where_sql:
            return 0
        fields = [f for f in (fields or []) if f in ANNOTATABLE_FIELDS]
        if fields:
            field_sql = " AND field IN ({})".format(",".join("?" * len(fields)))
            field_params: list[Any] = list(fields)
        else:
            field_sql = ""
            field_params = []

        with self.write() as conn:
            if clip_ids:
                marks = ",".join("?" * len(clip_ids))
                cur = conn.execute(
                    f"DELETE FROM annotations WHERE clip_id IN ({marks}){field_sql}",
                    [*clip_ids, *field_params],
                )
            else:
                cur = conn.execute(
                    f"DELETE FROM annotations WHERE clip_id IN "
                    f"(SELECT id FROM clips_merged WHERE {where_sql}){field_sql}",
                    [*where_params, *field_params],
                )
            return cur.rowcount

    def set_embedding(
        self, clip_id: int, embedding: bytes, description: str | None = None
    ) -> bool:
        with self.write() as conn:
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
        with self.write() as conn:
            conn.executemany(
                "UPDATE clips SET embedding = ? WHERE id = ?",
                [(emb, cid) for cid, emb in pairs],
            )

    def delete_clip(self, clip_id: int) -> sqlite3.Row | None:
        row = self.get_clip(clip_id)
        if row is None:
            return None
        with self.write() as conn:
            conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
        return row

    def iter_clips(self, **where: Any) -> list[sqlite3.Row]:
        sql = "SELECT * FROM clips_merged"
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
        # 统计也走合并视图：用户把某条改成「中文」后，
        # 统计里的语种分布应当跟着变，否则界面两处数字打架。
        with self.read() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM clips_merged").fetchone()["n"]
            verified = conn.execute(
                "SELECT COUNT(*) AS n FROM clips_merged WHERE verified = 1"
            ).fetchone()["n"]
            dur = conn.execute(
                "SELECT COALESCE(SUM(duration_sec), 0) AS s FROM clips_merged"
            ).fetchone()["s"]

            def group(col: str) -> dict[str, int]:
                rows = conn.execute(
                    f"SELECT {col} AS k, COUNT(*) AS n FROM clips_merged "
                    f"GROUP BY {col} ORDER BY n DESC"
                ).fetchall()
                return {(str(r["k"]) if r["k"] is not None else "(未知)"): r["n"] for r in rows}

            use_rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT target_use, clip_kind, COUNT(*) AS n FROM clips_merged "
                    "GROUP BY target_use, clip_kind"
                ).fetchall()
            ]
            embedded = conn.execute(
                "SELECT COUNT(*) AS n FROM clips WHERE embedding IS NOT NULL"
            ).fetchone()["n"]
            commercial = conn.execute(
                "SELECT COUNT(*) AS n FROM clips_merged WHERE commercial_ok = 1"
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

        with self.write() as conn:
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
