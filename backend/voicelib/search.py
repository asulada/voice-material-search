"""检索核心（方案 §3.2 / §3.3）。

L1 SQLite 结构化过滤 → L2 候选池内语义排序 → L3 top-K + 元数据。
两条线路的互斥约束（kinds / 时长）**必须进 SQL**，不能返回后再过滤。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .db import Database
from . import tags as T

# §3.3 两条线路的必要约束
USAGE_RULES: dict[str, dict[str, Any]] = {
    "indextts": {
        "kinds": ["speech"],
        "duration": (3.0, 15.0),  # 15s 是 IndexTTS 的硬截断
        "require": ["transcript"],
        "label": "IndexTTS · 情绪参考",
    },
    "seedvc": {
        "kinds": ["vocal_burst"],
        "duration": (0.3, 5.0),  # 副语言越短越纯
        "require": [],
        "label": "SeedVC · 副语言",
    },
}


@dataclass
class SearchQuery:
    use: str = "indextts"
    q: str | None = None
    kinds: list[str] | None = None
    categories: list[str] | None = None
    gender: str | None = None
    min_intensity: int | None = None
    duration: tuple[float, float] | None = None
    commercial_only: bool = False
    verified_only: bool = False
    top_k: int = 20
    sort: str = "relevance"  # relevance | quality | duration | random
    offset: int = 0
    # 不套用 use 的默认约束（浏览全库时用）
    ignore_use_rules: bool = False
    language: str | None = None


def _score_results(
    rows: Sequence[Any], query_vec: np.ndarray | None
) -> list[tuple[float, Any]]:
    scored: list[tuple[float, Any]] = []
    for r in rows:
        if query_vec is None:
            scored.append((0.0, r))
            continue
        blob = r["embedding"] if "embedding" in r.keys() else None
        if not blob:
            scored.append((0.0, r))
            continue
        ev = np.frombuffer(blob, dtype=np.float32)
        if ev.shape != query_vec.shape:
            scored.append((0.0, r))
            continue
        scored.append((float(np.dot(query_vec, ev)), r))
    return scored


def _quality_key(r: Any) -> tuple:
    """无查询词时的排序：棚录 / 高强度 / 已验证 / 商用优先。"""
    keys = r.keys()
    studio = r["studio_quality"] if "studio_quality" in keys else 0
    intensity = r["intensity"] if "intensity" in keys else 0
    verified = r["verified"] if "verified" in keys else 0
    commercial = r["commercial_ok"] if "commercial_ok" in keys else 0
    return (int(studio or 0), int(verified or 0), int(intensity or 0), int(commercial or 0))


def build_filter_clause(
    *,
    use: str = "indextts",
    kinds: list[str] | None = None,
    categories: list[str] | None = None,
    gender: str | None = None,
    min_intensity: int | None = None,
    duration: tuple[float, float] | None = None,
    commercial_only: bool = False,
    verified_only: bool = False,
    language: str | None = None,
) -> tuple[str, list[Any]]:
    """L1 结构化条件的唯一来源，返回 (WHERE 之后的片段, 参数)。

    抽出来是因为「按当前筛选条件清空人工标注」必须与检索**用同一套条件**：
    两处各写一份，早晚会漂移，出现「界面筛出 30 条，却清掉另外 200 条」。

    注意 `use` 会带进 `target_use IN (?, 'both')`；
    传 'all' 的调用方由 server 层先归一成具体线路。
    """
    cond = "target_use IN (?, 'both')"
    params: list[Any] = [use]

    if kinds:
        cond += " AND clip_kind IN ({})".format(",".join("?" * len(kinds)))
        params += list(kinds)
    if verified_only:
        cond += " AND verified = 1"
    if commercial_only:
        cond += " AND commercial_ok = 1"
    if categories:
        cond += " AND category IN ({})".format(",".join("?" * len(categories)))
        params += list(categories)
    if gender:
        cond += " AND gender = ?"
        params.append(gender)
    if language is not None:
        # 空串是前端「未标注」选项的取值：SQL 里 NULL != '' 永远不成立，
        # 必须显式拼 IS NULL，否则「未标注」会筛出 0 条。
        if language == "":
            cond += " AND language IS NULL"
        else:
            cond += " AND language = ?"
            params.append(language)
    if min_intensity is not None:
        cond += " AND intensity >= ?"
        params.append(int(min_intensity))
    if duration:
        cond += " AND duration_sec BETWEEN ? AND ?"
        params += [float(duration[0]), float(duration[1])]
    return cond, params


def ids_matching(
    db: Database,
    *,
    use: str = "indextts",
    kinds: list[str] | None = None,
    categories: list[str] | None = None,
    gender: str | None = None,
    min_intensity: int | None = None,
    duration: tuple[float, float] | None = None,
    commercial_only: bool = False,
    verified_only: bool = False,
    language: str | None = None,
) -> list[int]:
    """当前筛选条件下的全部 id（**不受 top_k 限制**）。

    用于「按筛选条件批量清空标注」：界面只显示一页，但要清的是整批。
    """
    cond, params = build_filter_clause(
        use=use, kinds=kinds, categories=categories, gender=gender,
        min_intensity=min_intensity, duration=duration,
        commercial_only=commercial_only, verified_only=verified_only,
        language=language,
    )
    with db.read() as conn:
        rows = conn.execute(
            f"SELECT id FROM clips_merged WHERE {cond}", params
        ).fetchall()
    return [int(r["id"]) for r in rows]


def search(
    db: Database,
    encoder,
    *,
    use: str = "indextts",
    q: str | None = None,
    kinds: list[str] | None = None,
    categories: list[str] | None = None,
    gender: str | None = None,
    min_intensity: int | None = None,
    duration: tuple[float, float] | None = None,
    commercial_only: bool = False,
    verified_only: bool = False,
    top_k: int = 20,
    sort: str = "relevance",
    offset: int = 0,
    # 语种过滤：None = 不限；"" = 只要未标注（NULL）的；其余按值精确匹配
    language: str | None = None,
) -> tuple[int, list[tuple[float, Any]]]:
    """返回 (命中总数, [(score, row), ...])。

    rows 含 embedding 列，调用方负责剥离后返回给前端。
    """
    # ---- L1 结构化过滤 ----
    # 查 clips_merged（用户标注优先），不是 clips。
    # 若筛选查程序值而响应里给用户值，用户按某值筛出来的条目
    # 会显示成另一个值，自相矛盾。
    cond, params = build_filter_clause(
        use=use, kinds=kinds, categories=categories, gender=gender,
        min_intensity=min_intensity, duration=duration,
        commercial_only=commercial_only, verified_only=verified_only,
        language=language,
    )
    sql = (
        "SELECT id, filepath, description, embedding, category, intensity, gender, "
        "duration_sec, speaker_id, target_use, clip_kind, verified, commercial_ok, "
        "license, transcript, gain_db, loudness_lufs, language, studio_quality, "
        "has_background, notes, sample_rate, source_file, start_sec, annotated_fields "
        f"FROM clips_merged WHERE {cond}"
    )

    with db.read() as conn:
        rows = conn.execute(sql, params).fetchall()

    total = len(rows)
    if not rows:
        return 0, []

    # ---- L2 语义排序（只对候选池）----
    if q and q.strip():
        qv = encoder.encode_query(q.strip())
        scored = _score_results(rows, np.asarray(qv, dtype=np.float32))
        if sort in ("relevance", ""):
            scored.sort(key=lambda x: -x[0])
        if sort == "quality":
            # 质量优先、相关度次之
            scored.sort(key=lambda x: (_quality_key(x[1]), x[0]), reverse=True)
        elif sort == "duration":
            scored.sort(key=lambda x: x[1]["duration_sec"])
        elif sort == "random":
            rng = np.random.default_rng()
            rng.shuffle(scored)
    else:
        if sort == "duration":
            rows = sorted(rows, key=lambda r: r["duration_sec"])
        elif sort == "random":
            rows = list(rows)
            np.random.default_rng().shuffle(rows)
        elif sort == "quality":
            rows = sorted(rows, key=_quality_key, reverse=True)
        scored = [(0.0, r) for r in rows]

    return total, scored[offset : offset + top_k]


def search_for(db: Database, encoder, use: str, *, q: str | None = None, top_k: int = 20, **kw):
    """把 usage 规则落到 SQL 条件里（§3.3）。"""
    rule = USAGE_RULES.get(use)
    if rule is None:
        raise ValueError(f"未知用途: {use}")
    kinds = kw.pop("kinds", None) or rule["kinds"]
    duration = kw.pop("duration", None)
    if duration is None:
        duration = rule["duration"]
    return search(
        db,
        encoder,
        use=use,
        q=q,
        kinds=kinds,
        duration=duration,
        top_k=top_k,
        **kw,
    )


def categories_for(use: str, db: Database | None = None) -> list[dict[str, Any]]:
    """某条线路下可用的类别（带计数），供前端筛选面板。

    同时返回两套名字：
      * `category`    —— 库里的原始标签（Adoration / cough），用于筛选时精确匹配
      * `category_cn` —— 中文显示名（倾慕 / 咳嗽），只用于界面

    为什么不在存储层直接存中文：原始标签是数据集的一部分，
    改掉就再也无法按 `Adoration` 精确回溯。中文映射是**展示用**的，
    而且映射表会随数据集扩充，放在读取层才能随时调整。
    """
    rule = USAGE_RULES.get(use)
    if db is None or rule is None:
        return []
    kinds = rule["kinds"]
    sql = (
        "SELECT category, COUNT(*) AS n FROM clips_merged "
        "WHERE target_use IN (?, 'both') AND clip_kind IN ({}) "
        "GROUP BY category ORDER BY n DESC"
    ).format(",".join("?" * len(kinds)))
    with db.read() as conn:
        rows = conn.execute(sql, [use, *kinds]).fetchall()
    # 副语言走 burst 映射，说话片段走 emotion 映射；
    # 两套表都覆盖不到时回退成原值（category_cn == category）。
    to_cn = T.burst_kind_cn if "vocal_burst" in kinds else T.emotion_cn
    return [
        {"category": r["category"], "category_cn": to_cn(r["category"]), "count": r["n"]}
        for r in rows
    ]


def languages_for(use: str, db: Database | None = None) -> list[dict[str, Any]]:
    """某条线路下出现过的语种（带计数），供筛选面板。

    与 `categories_for` 同一套口径（该线路的 clip_kind 内），
    同样返回原值 + 中文名：`language` / `language_cn`。
    NULL 归一成空串，前端据此给出「未标注」按钮，`language=""` 传回
    检索层时会被翻译成 `IS NULL`。
    """
    rule = USAGE_RULES.get(use)
    if db is None or rule is None:
        return []
    kinds = rule["kinds"]
    sql = (
        "SELECT language, COUNT(*) AS n FROM clips_merged "
        "WHERE target_use IN (?, 'both') AND clip_kind IN ({}) "
        "GROUP BY language ORDER BY n DESC"
    ).format(",".join("?" * len(kinds)))
    with db.read() as conn:
        rows = conn.execute(sql, [use, *kinds]).fetchall()
    return [
        {
            "language": r["language"] or "",
            "language_cn": T.language_cn(r["language"]),
            "count": r["n"],
        }
        for r in rows
    ]


def row_to_dict(row: Any, *, score: float = 0.0, keep_embedding: bool = False) -> dict[str, Any]:
    d = dict(row)
    if not keep_embedding:
        d.pop("embedding", None)
    d["score"] = round(float(score), 4)
    d["audio_url"] = f"/api/clips/{d['id']}/audio"
    return d


def build_request_query(payload: dict[str, Any]) -> SearchQuery:
    dur = payload.get("duration")
    return SearchQuery(
        use=payload.get("use") or "indextts",
        q=payload.get("q"),
        kinds=payload.get("kinds"),
        categories=payload.get("categories"),
        gender=payload.get("gender"),
        min_intensity=payload.get("min_intensity"),
        duration=(float(dur[0]), float(dur[1])) if dur else None,
        commercial_only=bool(payload.get("commercial_only")),
        verified_only=bool(payload.get("verified_only")),
        top_k=int(payload.get("top_k") or 20),
        sort=payload.get("sort") or "relevance",
        offset=int(payload.get("offset") or 0),
        ignore_use_rules=bool(payload.get("ignore_use_rules")),
        language=payload.get("language"),
    )
