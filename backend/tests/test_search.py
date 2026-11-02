"""检索测试（§3.2 / §3.3）。

覆盖方案里最容易被写错的两点：
  * L1 过滤必须在 SQL 里，不能返回后过滤（否则 top_k 悄悄变少）
  * 两条线路的 kinds / 时长约束互斥
"""

from __future__ import annotations

from voicelib.db import Database
from voicelib.embedder import HashingEncoder
from voicelib.search import USAGE_RULES, search, search_for


def make_db(tmp_lib) -> Database:
    db = Database(tmp_lib / "index" / "library.db")
    enc = HashingEncoder()
    rows = [
        # id 由 SQLite 分配：1..N
        dict(filepath="audio/csemotions/a.wav", checksum="x", target_use="indextts",
             clip_kind="speech", category="Angry", gender="male", duration_sec=6.0,
             transcript="你怎么能这样对我", description="一位男性用愤怒的语气说：你怎么能这样对我。",
             license="apache-2.0", commercial_ok=1, verified=1, intensity=4,
             studio_quality=1, speaker_id="S03"),
        dict(filepath="audio/csemotions/b.wav", checksum="x", target_use="indextts",
             clip_kind="speech", category="Sad", gender="female", duration_sec=5.0,
             transcript="我很难过", description="一位女性用悲伤的语气说：我很难过。",
             license="apache-2.0", commercial_ok=1, verified=0, intensity=3,
             studio_quality=0, speaker_id="S07"),
        dict(filepath="audio/csemotions/c.wav", checksum="x", target_use="indextts",
             clip_kind="speech", category="Angry", gender="male", duration_sec=2.0,
             transcript="太短了", description="一位男性用愤怒的语气说：太短了。",
             license="apache-2.0", commercial_ok=1, verified=1, intensity=5,
             studio_quality=0, speaker_id="S03"),  # 时长 2s < 3s，必须被 usage 规则排除
        dict(filepath="audio/csemotions/d.wav", checksum="x", target_use="indextts",
             clip_kind="speech", category="Angry", gender="male", duration_sec=8.0,
             transcript="NC 素材", description="一位男性用愤怒的语气说：NC 素材。",
             license="cc-by-nc-sa-4.0", commercial_ok=0, verified=1, intensity=4,
             studio_quality=1, speaker_id="S09"),
        dict(filepath="audio/vocal_burst/e.wav", checksum="x", target_use="seedvc",
             clip_kind="vocal_burst", category="laugh", gender="male", duration_sec=1.8,
             description="一段强烈的男性笑声，单次发声，1.8秒。",
             license="apache-2.0", commercial_ok=1, verified=1, intensity=4),
        dict(filepath="audio/vocal_burst/f.wav", checksum="x", target_use="seedvc",
             clip_kind="vocal_burst", category="cry", gender="female", duration_sec=2.5,
             description="一段中等的女性哭声，单次发声，2.5秒。",
             license="apache-2.0", commercial_ok=1, verified=1, intensity=3),
        dict(filepath="audio/vocal_burst/g.wav", checksum="x", target_use="seedvc",
             clip_kind="vocal_burst", category="cough", gender="male", duration_sec=0.4,
             description="一段轻微的男性咳嗽，单次发声，0.4秒。",
             license="apache-2.0", commercial_ok=1, verified=1, intensity=2),
    ]
    with db.tx() as conn:
        for r in rows:
            emb = enc.encode([r["description"]])[0]
            r["embedding"] = enc.to_blob(emb)
            db.upsert_clip(r, conn=conn)
    return db


def test_indextts_rules_exclude_short_and_bursts(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    total, scored = search_for(db, enc, "indextts", top_k=50)
    got_kinds = {r["clip_kind"] for _, r in scored}
    assert got_kinds == {"speech"}
    durs = [r["duration_sec"] for _, r in scored]
    assert all(3.0 <= d <= 15.0 for d in durs), durs
    # 2 秒那条不在结果里
    assert all(r["category"] != "Angry" or r["duration_sec"] >= 3 for _, r in scored)


def test_seedvc_rules_only_bursts(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    total, scored = search_for(db, enc, "seedvc", top_k=50)
    assert scored
    assert {r["clip_kind"] for _, r in scored} == {"vocal_burst"}
    assert all(0.3 <= r["duration_sec"] <= 5.0 for _, r in scored)


def test_top_k_not_silently_shrunk_after_filtering(tmp_lib):
    """L1 过滤在 SQL 里做，所以 top_k=3 就应返回 3 条（库里有 4 条 speech）。"""
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    total, scored = search(db, enc, use="indextts", kinds=["speech"], top_k=3)
    assert total == 4
    assert len(scored) == 3


def test_commercial_only(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    total, scored = search_for(db, enc, "indextts", top_k=50, commercial_only=True)
    assert all(r["commercial_ok"] == 1 for _, r in scored)
    assert any(r["license"].startswith("cc-by-nc") for _, r in
               search_for(db, enc, "indextts", top_k=50)[1])


def test_verified_only(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    _, scored = search_for(db, enc, "indextts", top_k=50, verified_only=True)
    assert all(r["verified"] == 1 for _, r in scored)


def test_gender_filter(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    _, scored = search_for(db, enc, "seedvc", top_k=50, gender="female")
    assert scored
    assert all(r["gender"] == "female" for _, r in scored)


def test_semantic_query_orders_by_similarity(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    _, scored = search_for(db, enc, "indextts", q="愤怒的男人", top_k=10)
    assert scored
    # 哈希嵌入下，含"愤怒"字面的条目分数应更高
    assert scored[0][0] > 0
    assert "愤怒" in scored[0][1]["description"]


def test_query_without_candidates_returns_empty(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    total, scored = search_for(db, enc, "indextts", top_k=10, categories=["Happy"])
    assert total == 0
    assert scored == []


def test_usage_rules_are_the_documented_values():
    assert USAGE_RULES["indextts"]["kinds"] == ["speech"]
    assert USAGE_RULES["indextts"]["duration"] == (3.0, 15.0)
    assert USAGE_RULES["seedvc"]["kinds"] == ["vocal_burst"]
    assert USAGE_RULES["seedvc"]["duration"] == (0.3, 5.0)


def test_pagination(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    _, page1 = search_for(db, enc, "indextts", top_k=2, offset=0)
    _, page2 = search_for(db, enc, "indextts", top_k=2, offset=2)
    ids1 = {r["id"] for _, r in page1}
    ids2 = {r["id"] for _, r in page2}
    assert ids1 and ids2 and not (ids1 & ids2)


def test_sort_random_returns_all(tmp_lib):
    db = make_db(tmp_lib)
    enc = HashingEncoder()
    total, scored = search_for(db, enc, "seedvc", top_k=50, sort="random")
    assert len(scored) == total == 3
