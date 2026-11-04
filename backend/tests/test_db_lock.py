"""SQLite 锁行为测试（对应 "database is locked" 的三个成因）。

这些用例都是**先复现问题**再修的，用来防止回退：
  1. connect() 里执行 PRAGMA journal_mode=WAL → 建连接就要求独占写锁
  2. 读事务不结束 → WAL 读快照一直占着，挡住 checkpoint
  3. 长写事务（导入）里做 CPU 计算 → 写锁被持有多秒，其他写请求超时
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from voicelib.db import Database


def _seed(db: Database, n: int = 3) -> None:
    for i in range(1, n + 1):
        db.upsert_clip(
            {
                "filepath": f"audio/x/{i}.wav",
                "checksum": "x",
                "target_use": "indextts",
                "clip_kind": "speech",
                "category": "Angry",
                "duration_sec": 5.0,
                "description": f"d{i}",
                "license": "apache-2.0",
                "commercial_ok": 1,
            }
        )


@pytest.fixture
def db(tmp_lib) -> Database:
    d = Database(tmp_lib / "index" / "library.db")
    _seed(d)
    return d


class _LockHolder:
    """在另一个连接上持有写锁，模拟导入任务正在写。"""

    def __init__(self, path: Path, seconds: float = 2.0):
        self.path = path
        self.seconds = seconds
        self.ready = threading.Event()
        self.release = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        assert self.ready.wait(timeout=10), "持锁线程没能及时进入事务"
        return self

    def _run(self) -> None:
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO clips (filepath, checksum, target_use, clip_kind, category, "
            "duration_sec, description, license, commercial_ok) "
            "VALUES ('audio/lock/x.wav','x','indextts','speech','Angry',5.0,'d',"
            "'apache-2.0',1)"
        )
        self.ready.set()
        self.release.wait(timeout=self.seconds)
        conn.execute("ROLLBACK")
        conn.close()

    def __exit__(self, *exc):
        self.release.set()
        self._thread.join(timeout=10)
        return False


def test_connect_does_not_take_a_write_lock(db):
    """新建连接不能因为别处持锁而失败（journal_mode 只在初始化时设一次）。"""
    with _LockHolder(db.path):
        conn = db.connect()          # 之前这里会因 PRAGMA journal_mode=WAL 报 locked
        try:
            # 读是允许的
            assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] >= 1
        finally:
            conn.close()


def test_fresh_nonwal_database_initialises_while_another_writer_holds_lock(tmp_path):
    """回归测试：库文件还是非 WAL 时，导入持锁 + 新连接 不应报 locked。

    这是原始的崩溃路径：`Database()` 在 `_init()` 里把库切成 WAL，
    而旧 `connect()` 每个连接都重设一次 WAL。当库尚未处于 WAL
    （首次运行、旧库、删掉 index/ 重建）而导入正在写时，任何新连接
    执行该 PRAGMA 都会立即抛 "database is locked"。

    修复把 WAL 收敛到 `_init()` 里的一次性设置。这里既验证
    `db.connect()` 本身不触碰 journal_mode，也验证在持锁期间
    新建 Database 仍能成功初始化。
    """
    import voicelib.db as mod

    path = tmp_path / "fresh.db"
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("PRAGMA journal_mode=DELETE")
    raw.executescript(mod.SCHEMA)
    raw.close()

    with _LockHolder(path):
        # 1) connect() 不应要求写锁：能够建立连接并读
        db = Database.__new__(Database)          # 跳过 __init__ 的初始化动作
        db.path = path
        conn = db.connect()
        try:
            assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal"
            assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 0
        finally:
            conn.close()

        # 2) 持锁期间新建 Database：_init 会等锁，但不应立刻崩
        db2 = Database(path)
        assert db2.count() >= 0

    # 初始化完成后库应已切到 WAL
    conn = sqlite3.connect(path)
    try:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
    finally:
        conn.close()


def test_wal_pragma_on_fresh_nonwal_db_under_lock_is_the_real_trap(tmp_path):
    """精确复现原始故障，并说明它为何只在特定条件下出现。

    实测条件（tests/probe_wal_conditions.py）：
      * 库文件**还不是 WAL**（首次运行 / 旧库 / 刚删除 index/）
      * 且此刻有连接持有写锁（导入任务正在写）
    ⇒ 新连接执行 `PRAGMA journal_mode=WAL` 会**立刻**抛
      "database is locked"（不等待 busy_timeout），
      所以表现是「一开始就报错」而不是「等一会儿变慢」。

    这正是旧 connect() 的写法：每个连接都设 WAL。
    修复后 WAL 只在 Database._init() 里设一次，connect() 不再碰它。
    """
    import voicelib.db as mod

    path = tmp_path / "fresh.db"
    # 造一个「非 WAL」的库，但表结构完整（模拟「把 journal_mode 切回 DELETE 的旧库」）
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("PRAGMA journal_mode=DELETE")
    raw.executescript(mod.SCHEMA)
    raw.close()
    assert str(
        sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0]
    ).lower() != "wal"

    # 坏写法：连接级设 WAL
    def bad_connect() -> sqlite3.Connection:
        c = sqlite3.connect(path, timeout=0.5)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    # 好写法：与生产代码一致（只设 busy_timeout，不设 WAL）
    def good_connect() -> sqlite3.Connection:
        c = sqlite3.connect(path, timeout=0.5)
        c.execute(f"PRAGMA busy_timeout={mod.BUSY_TIMEOUT_MS}")
        return c

    with _LockHolder(path):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            bad_connect()
        # 修复后的写法在同样的持锁条件下可以建立连接并读取
        c = good_connect()
        try:
            assert c.execute("SELECT COUNT(*) FROM clips").fetchone()[0] >= 0
        finally:
            c.close()


def test_journal_mode_is_wal_and_persisted(db):
    with db.read() as conn:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
    # 换一个全新连接也仍是 WAL（持久化属性）
    conn = db.connect()
    try:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
    finally:
        conn.close()


def test_read_does_not_leave_open_transaction(db):
    """读结束后不能残留事务，否则 WAL checkpoint 被挡住。"""
    with db.read() as conn:
        conn.execute("SELECT COUNT(*) FROM clips").fetchone()
    conn = db.connect()
    try:
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_write_retries_through_short_lock(db, monkeypatch):
    """短锁应在等待/重试后成功，而不是直接抛 locked。"""
    import voicelib.db as mod

    monkeypatch.setattr(mod, "WRITE_RETRY_BASE_DELAY", 0.01)
    holder = _LockHolder(db.path, seconds=0.6)
    with holder:
        t0 = time.time()
        ok = db.update_clip(1, {"intensity": 5})
        elapsed = time.time() - t0
    assert ok is True
    assert elapsed >= 0.5, "应当等待持锁方释放后才写入"
    assert db.get_clip(1)["intensity"] == 5


def test_concurrent_writers_all_land(db):
    """多线程同时写：全部成功且不丢数据。"""
    errors: list[str] = []

    def work(n: int) -> None:
        try:
            for i in range(10):
                db.update_clip((i % 3) + 1, {"intensity": ((n + i) % 5) + 1})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    assert db.count() == 3


def test_import_flush_does_not_hold_lock_while_encoding(tmp_lib, tmp_path, monkeypatch):
    """导入时建向量必须在写事务之外：编码期间别的写请求不应被卡住。"""
    from voicelib import audio as A
    from voicelib.config import Config
    from voicelib.embedder import HashingEncoder
    from voicelib.ingest import ImportOptions, import_into

    from conftest import synth_speech

    # 造 40 条素材，够触发一次 COMMIT_EVERY=100 之外的批次（用 monkeypatch 调小）
    src = tmp_path / "src"
    (src / "S01" / "Angry").mkdir(parents=True)
    for i in range(4):
        A.write_wav(src / "S01" / "Angry" / f"{i}.wav", synth_speech(5.0, seed=i), 22050)

    class SlowEncoder(HashingEncoder):
        def encode(self, texts):
            time.sleep(0.6)  # 模拟 BGE 编码耗时
            return super().encode(texts)

    import voicelib.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "COMMIT_EVERY", 2)

    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)

    # 先插一条，供并发写用
    db.upsert_clip(
        {
            "filepath": "audio/x/probe.wav",
            "checksum": "x",
            "target_use": "indextts",
            "clip_kind": "speech",
            "category": "Angry",
            "duration_sec": 5.0,
            "description": "probe",
            "license": "apache-2.0",
            "commercial_ok": 1,
        }
    )
    probe_id = db.iter_clips()[0]["id"]

    latencies: list[float] = []

    def poke_writes() -> None:
        """导入进行中，反复尝试写：记录每次耗时。"""
        deadline = time.time() + 3.0
        while time.time() < deadline:
            t0 = time.time()
            db.update_clip(probe_id, {"intensity": 3})
            latencies.append(time.time() - t0)
            time.sleep(0.01)

    t = threading.Thread(target=poke_writes, daemon=True)
    t.start()
    res = import_into(
        cfg,
        db,
        SlowEncoder(),
        ImportOptions(
            kind="csemotions", path=str(src), target_use="indextts",
            gender_map={"S01": "male"},
        ),
    )
    t.join(timeout=10)

    assert res.imported == 4
    assert latencies, "探针写入没有执行"
    worst = max(latencies)
    # 编码期若持锁，单次写会等 0.6s 以上；修好后应当都是毫秒级
    assert worst < 0.5, f"编码期间写请求被阻塞了 {worst:.2f}s，说明写锁被长时间持有"
