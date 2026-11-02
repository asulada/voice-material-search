"""检索核心（方案 §3.2 / §3.3）。

L1 SQLite 结构化过滤 → L2 候选池内语义排序 → L3 top-K + 元数据。
两条线路的互斥约束（kinds / 时长）**必须进 SQL**，不能返回后再过滤。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .db import Database

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
    language: str | None = None,
) -> tuple[int, list[tuple[float, Any]]]:
    """返回 (命中总数, [(score, row), ...])。

    rows 含 embedding 列，调用方负责剥离后返回给前端。
    """
    # ---- L1 结构化过滤 ----
    sql = (
        "SELECT id, filepath, description, embedding, category, intensity, gender, "
        "duration_sec, speaker_id, target_use, clip_kind, verified, commercial_ok, "
        "license, transcript, gain_db, loudness_lufs, language, studio_quality, "
        "has_background, notes, sample_rate, source_file, start_sec "
        "FROM clips WHERE target_use IN (?, 'both')"
    )
    params: list[Any] = [use]

    if kinds:
        sql += " AND clip_kind IN ({})".format(",".join("?" * len(kinds)))
        params += list(kinds)
    if verified_only:
        sql += " AND verified = 1"
    if commercial_only:
        sql += " AND commercial_ok = 1"
    if categories:
        sql += " AND category IN ({})".format(",".join("?" * len(categories)))
        params += list(categories)
    if gender:
        sql += " AND gender = ?"
        params.append(gender)
    if language:
        sql += " AND language = ?"
        params.append(language)
    if min_intensity is not None:
        sql += " AND intensity >= ?"
        params.append(int(min_intensity))
    if duration:
        sql += " AND duration_sec BETWEEN ? AND ?"
        params += [float(duration[0]), float(duration[1])]

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
    """某条线路下可用的类别（带计数），供前端筛选面板。"""
    rule = USAGE_RULES.get(use)
    if db is None or rule is None:
        return []
    kinds = rule["kinds"]
    sql = (
        "SELECT category, COUNT(*) AS n FROM clips "
        "WHERE target_use IN (?, 'both') AND clip_kind IN ({}) "
        "GROUP BY category ORDER BY n DESC"
    ).format(",".join("?" * len(kinds)))
    with db.read() as conn:
        rows = conn.execute(sql, [use, *kinds]).fetchall()
    return [{"category": r["category"], "count": r["n"]} for r in rows]


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
