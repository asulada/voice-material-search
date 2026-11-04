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


# ---------------------------------------------------------------- 类别中文显示名
#
# 用户反馈：筛选面板里全是 Adoration / cough / panting 这种英文。
# 原因是 category 存的是数据集原始标签，前端直接渲染。
# 决定（用户选择）：**显示层翻译**，存储层保留原值 —— 原始标签是数据的一部分，
# 改掉就再也无法按 `Adoration` 精确回溯；而映射表会随数据集扩充，
# 放在读取层才能随时调整。


def test_categories_include_chinese_name(tmp_lib):
    """每个类别都要同时给出原值和中文名。"""
    from voicelib.search import categories_for

    db = make_db(tmp_lib)
    cats = categories_for("seedvc", db)

    assert cats, "seedvc 线路应有类别"
    for c in cats:
        assert "category" in c and "category_cn" in c
        assert c["count"] > 0

    by_cat = {c["category"]: c["category_cn"] for c in cats}
    assert by_cat["laugh"] == "笑声"
    assert by_cat["cry"] == "哭声"
    assert by_cat["cough"] == "咳嗽"


def test_categories_indextts_uses_emotion_mapping(tmp_lib):
    """说话片段走 emotion 映射表，不是 burst 表。"""
    from voicelib.search import categories_for

    db = make_db(tmp_lib)
    cats = categories_for("indextts", db)

    by_cat = {c["category"]: c["category_cn"] for c in cats}
    assert by_cat.get("Angry") == "愤怒"


def test_categories_original_value_is_unchanged(tmp_lib):
    """关键约束：category 必须保持原值，筛选才能精确匹配。"""
    from voicelib.search import categories_for

    db = make_db(tmp_lib)
    cats = categories_for("seedvc", db)
    originals = {c["category"] for c in cats}

    # 原值仍是英文，没有被翻译污染
    assert originals == {"laugh", "cry", "cough"}


def test_unknown_category_falls_back_to_itself(tmp_lib):
    """映射表没有的类别，中文名回退成原值，不能变成空串。"""
    from voicelib import tags as T

    assert T.burst_kind_cn("some_brand_new_label") == "some_brand_new_label"
    assert T.emotion_cn("some_brand_new_emotion") == "some_brand_new_emotion"


# ---------------------------------------------------------------- 语种筛选
#
# 用户反馈：筛选里没有语种（中文 / 英文）可选。
# 后端 search() 一直有 language 参数，但前端没接，且没人处理 NULL：
# SQL 里 `language = ''` 永远不成立，所以「未标注」必须显式翻成 IS NULL。
# 库里大量素材是 NULL（副语言子集刻意不标语种），这条分支不能省。


def make_db_with_language(tmp_lib) -> Database:
    """在 make_db 的基础上给部分行标语种，留几条 NULL。"""
    db = make_db(tmp_lib)
    with db.tx() as conn:
        conn.execute("UPDATE clips SET language = 'en' WHERE filepath LIKE '%/a.wav'")
        conn.execute("UPDATE clips SET language = 'zh' WHERE filepath LIKE '%/b.wav'")
        # c / d / e / f / g 保持 NULL
    return db


def test_language_filter_exact_match(tmp_lib):
    db = make_db_with_language(tmp_lib)
    _, scored = search_for(db, "x", "indextts", top_k=50, language="zh")
    assert [r["filepath"] for _, r in scored] == ["audio/csemotions/b.wav"]


def test_language_filter_unlabeled_is_null_not_empty_string(tmp_lib):
    """language="" 要筛出 NULL 的行，不能是 0 条。"""
    db = make_db_with_language(tmp_lib)
    _, scored = search_for(db, "x", "seedvc", top_k=50, language="")
    assert scored, "未标注分支必须命中 NULL 行"
    assert all(r["language"] is None for _, r in scored)


def test_language_none_means_no_filter(tmp_lib):
    """language=None 不加 SQL 条件；总数应等于该线路的全部候选（3s 以下那条本来就被排除）。"""
    db = make_db_with_language(tmp_lib)
    total, _ = search_for(db, "x", "indextts", top_k=50)
    assert total == 3


def test_languages_for_counts_each_line(tmp_lib):
    from voicelib.search import languages_for

    db = make_db_with_language(tmp_lib)
    seed = {l["language"]: l["count"] for l in languages_for("seedvc", db)}
    assert seed == {"": 3}, seed  # 三条副语言都没有语种
    speech = {l["language"]: l["count"] for l in languages_for("indextts", db)}
    assert speech == {"en": 1, "zh": 1, "": 2}, speech


def test_language_cn_display_names(tmp_lib):
    """和类别一样：原值存码，显示层给中文；映射不到原样返回。"""
    from voicelib import tags as T

    assert T.language_cn("zh") == "中文"
    assert T.language_cn("en") == "英文"
    assert T.language_cn("zh-CN") == "中文"
    assert T.language_cn("xx") == "xx"
    assert T.language_cn(None) == "未标注"


# ---------------------------------------------------------------- 用户标注优先
#
# 用户要求：所有业务字段支持人工标注，显示与筛选都以人工值为准。
# 存储上是 annotations(clip_id, field, value) 与 clips 分开，
# 读取走 clips_merged 视图（LEFT JOIN + COALESCE）。
#
# 这里钉住的重点是**筛选也认人工值**：若只在返回前于 Python 里覆盖，
# 按「中文」筛出来的条目会显示成「英文」，自相矛盾。


def _annotate(db: Database, clip_id: int, **fields) -> None:
    with db.write() as conn:
        for k, v in fields.items():
            conn.execute(
                "INSERT INTO annotations(clip_id, field, value) VALUES (?,?,?) "
                "ON CONFLICT(clip_id, field) DO UPDATE SET value = excluded.value",
                (clip_id, k, None if v is None else str(v)),
            )


def test_annotation_overrides_program_value(tmp_lib):
    db = make_db(tmp_lib)
    cid = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    _annotate(db, cid, category="愤怒", language="zh")

    row = db.get_clip(cid)
    assert row["category"] == "愤怒"
    assert row["language"] == "zh"


def test_annotation_participates_in_filtering(tmp_lib):
    """按人工值筛选必须命中，按原程序值则不该再命中。"""
    db = make_db(tmp_lib)
    cid = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    _annotate(db, cid, category="愤怒")

    _, hit = search_for(db, HashingEncoder(), "indextts", top_k=50, categories=["愤怒"])
    assert [r["id"] for _, r in hit] == [cid]

    _, miss = search_for(db, HashingEncoder(), "indextts", top_k=50, categories=["Angry"])
    assert cid not in [r["id"] for _, r in miss]


def test_annotation_numeric_compare_is_not_textual(tmp_lib):
    """annotations.value 是 TEXT，比较必须 CAST，否则 '10' < '4'。"""
    db = make_db(tmp_lib)
    cid = db.get_by_filepath("audio/csemotions/b.wav")["id"]
    _annotate(db, cid, intensity=10)

    _, hit = search_for(db, HashingEncoder(), "indextts", top_k=50, min_intensity=9)
    assert [r["id"] for _, r in hit] == [cid]
    row = db.get_clip(cid)
    assert isinstance(row["intensity"], int) and row["intensity"] == 10


def test_annotation_reflected_in_stats_and_facets(tmp_lib):
    """统计与筛选面板的计数也要跟着人工值走。"""
    from voicelib.search import categories_for, languages_for

    db = make_db(tmp_lib)
    cid = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    _annotate(db, cid, category="愤怒", language="zh")

    cats = {c["category"]: c["count"] for c in categories_for("indextts", db)}
    # 造数里 Angry 有 a/c/d 三条；a.wav 被人工改成「愤怒」后，
    # Angry 应剩两条（c、d）。
    # 注意 categories_for 只按 target_use + clip_kind 分组，**不套时长规则**，
    # 所以 2 秒的 c.wav 仍然计入 —— 不要按检索的时长过滤去推这里的数。
    assert cats["愤怒"] == 1
    assert cats.get("Angry", 0) == 2

    langs = {l["language"]: l["count"] for l in languages_for("indextts", db)}
    assert langs["zh"] == 1


def test_missing_annotation_falls_back_to_program_value(tmp_lib):
    """没标注的字段仍用程序值，不能因为 join 变成 NULL。"""
    db = make_db(tmp_lib)
    cid = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    _annotate(db, cid, category="愤怒")  # 只改类别

    row = db.get_clip(cid)
    assert row["category"] == "愤怒"
    assert row["gender"] == "male"        # 程序值
    assert row["duration_sec"] == 6.0     # 不可标注，恒为程序值


# ---------------------------------------------------------------- 人工标注写入与清空


def test_update_clip_writes_annotations_not_clips(tmp_lib):
    """人工改可标注字段 → 写 annotations；clips 里的程序值保持不动。"""
    db = make_db(tmp_lib)
    cid = db.get_by_filepath("audio/csemotions/a.wav")["id"]

    assert db.update_clip(cid, {"category": "愤怒", "intensity": 5}) is True

    # 读取层看到的是人工值
    assert db.get_clip(cid)["category"] == "愤怒"
    # 存储层 clips 仍是程序值 —— 这是「导入能覆盖程序值、但动不了人工值」的前提
    with db.read() as conn:
        raw = conn.execute("SELECT category, intensity FROM clips WHERE id = ?", (cid,)).fetchone()
    assert raw["category"] == "Angry"
    assert raw["intensity"] == 4
    assert db.annotations_of(cid)["category"] == "愤怒"


def test_update_clip_none_clears_annotation(tmp_lib):
    """传 None = 清掉人工标注、回落到程序值（不是把字段设成 NULL）。"""
    db = make_db(tmp_lib)
    cid = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    db.update_clip(cid, {"category": "愤怒"})
    assert db.get_clip(cid)["category"] == "愤怒"

    db.update_clip(cid, {"category": None})
    assert db.get_clip(cid)["category"] == "Angry"
    assert "category" not in db.annotations_of(cid)


def test_update_clip_target_use_goes_to_clips(tmp_lib):
    """target_use 不是可标注字段，应直接改 clips。"""
    db = make_db(tmp_lib)
    cid = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    db.update_clip(cid, {"target_use": "both"})
    with db.read() as conn:
        assert conn.execute(
            "SELECT target_use FROM clips WHERE id = ?", (cid,)
        ).fetchone()["target_use"] == "both"


def test_clear_annotations_by_ids(tmp_lib):
    db = make_db(tmp_lib)
    a = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    b = db.get_by_filepath("audio/csemotions/b.wav")["id"]
    db.update_clip(a, {"category": "愤怒", "language": "zh"})
    db.update_clip(b, {"category": "悲伤"})

    cleared = db.clear_annotations([a])
    assert cleared == 2                      # a 的两个字段
    assert db.annotations_of(a) == {}
    assert db.annotations_of(b)              # b 不受影响
    assert db.get_clip(a)["category"] == "Angry"


def test_clear_annotations_refuses_empty_selection(tmp_lib):
    """两个条件都不给时返回 0：不提供「清空全库标注」这种不可逆入口。"""
    db = make_db(tmp_lib)
    a = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    db.update_clip(a, {"category": "愤怒"})

    assert db.clear_annotations() == 0
    assert db.annotations_of(a) == {"category": "愤怒"}


def test_clear_annotations_by_condition(tmp_lib):
    """按筛选条件清：只清条件命中的那些。"""
    db = make_db(tmp_lib)
    a = db.get_by_filepath("audio/csemotions/a.wav")["id"]
    b = db.get_by_filepath("audio/csemotions/b.wav")["id"]
    db.update_clip(a, {"category": "愤怒"})
    db.update_clip(b, {"category": "悲伤"})

    cleared = db.clear_annotations(where_sql="gender = ?", where_params=["female"])
    assert cleared == 1                     # 只有 b 是 female
    assert db.annotations_of(b) == {}
    assert db.annotations_of(a)
