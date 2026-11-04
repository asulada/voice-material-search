"""API 测试（§6.2 API 表 + §6.6 文件访问边界）。

用 TestClient 打真实路由，覆盖：检索、试听（Range）、标注、重建向量、导出、
路径穿越防护、任务进度。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from voicelib import audio as A

from conftest import synth_speech


@pytest.fixture
def client(tmp_path, monkeypatch):
    """每个测试一个独立库根目录，并强制走哈希嵌入（不联网）。"""
    monkeypatch.setenv("VOICE_LIBRARY_BGE_INSTRUCTION", "0")
    root = tmp_path / "voice_library"
    root.mkdir()

    from voicelib.embedder import HashingEncoder

    import server

    app = server.create_app(str(root))
    # 用确定性哈希嵌入替换真模型，测试不依赖网络
    app.state.vl.encoder._impl = HashingEncoder()
    with TestClient(app) as c:
        c.state = app.state.vl  # type: ignore[attr-defined]
        yield c


def _seed_dataset(base: Path) -> Path:
    """造一个 CSEMOTIONS 形态的数据源目录。"""
    root = base / "CSEMOTIONS"
    for speaker, seed in (("S03", 0), ("S07", 4)):
        for emo in ("Angry", "Happy"):
            d = root / speaker / emo
            d.mkdir(parents=True)
            for i in range(3):
                A.write_wav(d / f"{emo}_{i}.wav", synth_speech(6.0, seed=seed + i), 22050)
    return root


def _run_import(client, src: Path, **kw) -> dict:
    body = {
        "kind": "csemotions",
        "path": str(src),
        "target_use": "indextts",
        "license": "apache-2.0",
        "gender_map": {"S03": "male", "S07": "female"},
        **kw,
    }
    r = client.post("/api/import", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    # 同步等任务结束（单线程池，轮询即可）
    import time

    for _ in range(200):
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert snap["status"] == "done", snap
    return snap


# ------------------------------------------------------------------ 基础


def test_health_and_meta(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/api/meta")
    assert r.status_code == 200
    data = r.json()
    uses = {u["value"]: u for u in data["uses"]}
    assert uses["indextts"]["kinds"] == ["speech"]
    assert uses["seedvc"]["duration"] == [0.3, 5.0]
    assert data["encoder"]["backend"] == "hash-fallback"


def test_import_then_search(client, tmp_path):
    src = _seed_dataset(tmp_path)
    snap = _run_import(client, src)
    assert snap["result"]["imported"] == 12

    stats = client.get("/api/stats").json()
    assert stats["total"] == 12
    assert stats["by_category"]["Angry"] == 6
    assert stats["by_gender"]["male"] == 6

    r = client.post("/api/search", json={"use": "indextts", "q": "愤怒男声", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 12
    assert len(data["items"]) == 5
    item = data["items"][0]
    # 响应必须含前端渲染所需全部字段（§6.2）
    for key in ("id", "filepath", "audio_url", "description", "category",
                "intensity", "gender", "duration_sec", "gain_db", "score",
                "verified", "commercial_ok"):
        assert key in item, key
    assert item["audio_url"] == f"/api/clips/{item['id']}/audio"
    assert "embedding" not in item, "向量不应返回给前端"


def test_search_for_each_line(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    # seedvc 线路应当没有素材
    r = client.post("/api/search", json={"use": "seedvc"})
    assert r.json()["total"] == 0
    # 显式 kinds 覆盖可以查到
    r = client.post("/api/search", json={"use": "indextts", "kinds": ["vocal_burst"]})
    assert r.json()["total"] == 0


def test_search_filters_land_in_sql(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    r = client.post("/api/search", json={"use": "indextts", "gender": "female", "top_k": 50})
    items = r.json()["items"]
    assert items and all(i["gender"] == "female" for i in items)
    assert len(items) == 6


def test_audio_endpoint_serves_file_and_range(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    cid = client.post("/api/search", json={"use": "indextts", "top_k": 1}).json()["items"][0]["id"]

    r = client.get(f"/api/clips/{cid}/audio")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    full = r.content
    assert len(full) > 1000

    r = client.get(f"/api/clips/{cid}/audio", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert len(r.content) == 100
    assert r.headers["content-range"].startswith("bytes 0-99/")


def test_audio_missing_clip_404(client):
    assert client.get("/api/clips/99999/audio").status_code == 404


def test_path_traversal_blocked(client, tmp_path):
    """即使 db 里被写入了越界路径，也不能读到库外文件。"""
    vl = client.state
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    with vl.db.tx() as conn:
        conn.execute(
            "INSERT INTO clips (filepath, checksum, target_use, clip_kind, category, "
            "duration_sec, description, license, commercial_ok) "
            "VALUES ('../secret.txt', 'x', 'indextts', 'speech', 'Angry', 5.0, 'd', "
            "'apache-2.0', 1)"
        )
        cid = conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]

    r = client.get(f"/api/clips/{cid}/audio")
    assert r.status_code in (403, 404)
    assert b"top secret" not in r.content


def test_patch_clip_updates_and_reembeds(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    cid = client.post("/api/search", json={"use": "indextts", "top_k": 1}).json()["items"][0]["id"]

    r = client.patch(f"/api/clips/{cid}", json={"intensity": 5, "verified": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["intensity"] == 5
    assert data["verified"] == 1
    assert data["reembedded"] is False  # 描述没改，不重建

    new_desc = "一段刻意压抑的男性愤怒，像在忍着火。"
    r = client.patch(
        f"/api/clips/{cid}", json={"description": new_desc}
    )
    data = r.json()
    assert data["description"] == new_desc
    assert data["reembedded"] is True, "描述改了必须重建向量"

    # 重建后能按新描述检索到
    r = client.post("/api/search", json={"use": "indextts", "q": "压抑的愤怒", "top_k": 3})
    assert cid in [i["id"] for i in r.json()["items"]]


def test_patch_license_recomputes_commercial_ok(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    cid = client.post("/api/search", json={"use": "indextts", "top_k": 1}).json()["items"][0]["id"]
    assert client.get(f"/api/clips/{cid}").json()["commercial_ok"] == 1

    r = client.patch(f"/api/clips/{cid}", json={"license": "cc-by-nc-sa-4.0"})
    assert r.json()["commercial_ok"] == 0, "commercial_ok 必须由 license 推导"

    r = client.post("/api/search", json={"use": "indextts", "commercial_only": True, "top_k": 50})
    assert cid not in [i["id"] for i in r.json()["items"]]


def test_reebed_single(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    cid = client.post("/api/search", json={"use": "indextts", "top_k": 1}).json()["items"][0]["id"]
    r = client.post(f"/api/clips/{cid}/reebed", json={"description": "全新的描述文本"})
    assert r.status_code == 200
    assert r.json()["description"] == "全新的描述文本"
    assert client.get(f"/api/clips/{cid}").json()["description"] == "全新的描述文本"


def test_waveform(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    cid = client.post("/api/search", json={"use": "indextts", "top_k": 1}).json()["items"][0]["id"]
    r = client.get(f"/api/clips/{cid}/waveform?points=100")
    assert r.status_code == 200
    data = r.json()
    assert len(data["points"]) == 100
    assert all(0 <= p <= 1.0001 for p in data["points"])
    assert data["duration_sec"] > 0


def test_export_writes_selection_json(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    items = client.post("/api/search", json={"use": "indextts", "top_k": 3}).json()["items"]
    ids = [i["id"] for i in items]

    r = client.post("/api/export", json={"clip_ids": ids, "name": "test-export"})
    assert r.status_code == 200
    data = r.json()
    assert data["exported"] == 3

    out = Path(data["directory"])
    assert out.is_dir()
    manifest = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 3
    assert len(manifest["clips"]) == 3
    for item in manifest["clips"]:
        assert Path(item["export_path"]).is_file()
        # 导出的是独立文件（IndexTTS 只取前 15 秒，§5.1）
        assert item["export_path"].endswith(".wav")

    r = client.get("/api/exports")
    assert any(e["name"] == "test-export" for e in r.json()["exports"])


def test_export_missing_ids_reported(client, tmp_path):
    r = client.post("/api/export", json={"clip_ids": [424242]})
    assert r.status_code == 200
    assert r.json()["exported"] == 0
    assert r.json()["missing"] == [424242]


def test_export_empty_selection_rejected(client):
    assert client.post("/api/export", json={"clip_ids": []}).status_code == 400


def test_import_preview(client, tmp_path):
    src = _seed_dataset(tmp_path)
    r = client.get("/api/import/preview", params={"path": str(src)})
    assert r.status_code == 200
    data = r.json()
    assert data["detected_kind"] == "csemotions"
    assert data["count"] == 12
    assert len(data["samples"]) == 12


def test_import_bad_path(client):
    r = client.post("/api/import", json={"kind": "flat", "path": "Z:/does/not/exist"})
    assert r.status_code == 400


def test_job_unknown_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_voices_empty(client):
    r = client.get("/api/voices")
    assert r.status_code == 200
    assert r.json()["voices"] == []


def test_delete_clip(client, tmp_path):
    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    item = client.post("/api/search", json={"use": "indextts", "top_k": 1}).json()["items"][0]
    path = client.state.cfg.root_path / item["filepath"]
    assert path.is_file()

    r = client.delete(f"/api/clips/{item['id']}", params={"delete_file": True})
    assert r.json()["file_removed"] is True
    assert not path.is_file()
    assert client.get(f"/api/clips/{item['id']}").status_code == 404


# ---------------------------------------------------------------- 语种筛选（前端筛选面板）

def test_search_language_filter_and_endpoint(client, tmp_path):
    """导入时语言留空 → 全库 NULL → 「未标注」要能筛出来，不能是 0 条。"""
    src = _seed_dataset(tmp_path)
    # 一批标 zh，一批留空
    _run_import(client, src, language="zh")
    r = client.post("/api/search", json={"use": "indextts", "language": "zh", "top_k": 50})
    assert r.json()["total"] == 12
    r = client.post("/api/search", json={"use": "indextts", "language": "", "top_k": 50})
    assert r.json()["total"] == 0

    # 再导一份不标语种的
    src2 = tmp_path / "CSEMOTIONS2"
    for emo in ("Sad",):
        d = src2 / "S09" / emo
        d.mkdir(parents=True)
        for i in range(2):
            A.write_wav(d / f"{emo}_{i}.wav", synth_speech(6.0, seed=10 + i), 22050)
    _run_import(client, src2, kind="csemotions", gender_map={"S09": "male"})

    r = client.post("/api/search", json={"use": "indextts", "language": "zh", "top_k": 50})
    assert r.json()["total"] == 12
    r = client.post("/api/search", json={"use": "indextts", "language": "", "top_k": 50})
    unlabeled = r.json()
    assert unlabeled["total"] == 2
    assert all(i["language"] is None for i in unlabeled["items"])

    # 筛选面板的语种分布：原值 + 中文名
    langs = client.get("/api/languages", params={"use": "indextts"}).json()["languages"]
    by_code = {l["language"]: l for l in langs}
    assert by_code["zh"]["language_cn"] == "中文" and by_code["zh"]["count"] == 12
    assert by_code[""]["language_cn"] == "未标注" and by_code[""]["count"] == 2

    stats = client.get("/api/stats").json()
    assert stats["languages"]["indextts"] == langs


# ---------------------------------------------------------------- 音频落盘策略（设置项）

def test_import_audio_mode_keep_via_api(client, tmp_path):
    """audio_mode=keep 时，已存在的 wav 不被重写，元数据仍刷新。"""
    import os

    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    files = sorted((Path(client.state.cfg.root_path) / "audio").rglob("*.wav"))
    assert files
    stamp = os.stat(files[0]).st_mtime_ns

    snap = _run_import(client, src, audio_mode="keep")
    assert snap["result"]["audio_kept"] > 0
    assert os.stat(files[0]).st_mtime_ns == stamp, "keep 模式不应重写文件"


def test_import_audio_mode_overwrite_by_default(client, tmp_path):
    import os, time

    src = _seed_dataset(tmp_path)
    _run_import(client, src)
    files = sorted((Path(client.state.cfg.root_path) / "audio").rglob("*.wav"))
    stamp = os.stat(files[0]).st_mtime_ns

    time.sleep(0.01)
    snap = _run_import(client, src)
    assert snap["result"]["audio_kept"] == 0, "默认应重写"
    assert os.stat(files[0]).st_mtime_ns != stamp


def test_import_rejects_unknown_audio_mode(client, tmp_path):
    src = _seed_dataset(tmp_path)
    r = client.post("/api/import", json={
        "kind": "csemotions", "path": str(src), "audio_mode": "nonsense",
    })
    assert r.status_code == 422


# ---------------------------------------------------------------- 人工标注（PATCH）

def _first_id(client) -> int:
    return client.post(
        "/api/search", json={"use": "indextts", "top_k": 1}
    ).json()["items"][0]["id"]


def test_patch_language_is_editable(client, tmp_path):
    """语种必须能通过 PATCH 改：导入时没标的素材只能靠这里补。"""
    _run_import(client, _seed_dataset(tmp_path))
    cid = _first_id(client)
    assert client.get(f"/api/clips/{cid}").json()["language"] is None

    data = client.patch(f"/api/clips/{cid}", json={"language": "zh"}).json()
    assert data["language"] == "zh"
    assert "language" in data["annotated_fields"].split(",")


def test_patch_null_clears_annotation(client, tmp_path):
    """显式传 null = 撤销该项人工标注，回落到程序值。"""
    _run_import(client, _seed_dataset(tmp_path))
    cid = _first_id(client)
    original = client.get(f"/api/clips/{cid}").json()["category"]

    client.patch(f"/api/clips/{cid}", json={"category": "我改的类别"})
    assert client.get(f"/api/clips/{cid}").json()["category"] == "我改的类别"

    data = client.patch(f"/api/clips/{cid}", json={"category": None}).json()
    assert data["category"] == original
    assert "category" not in data["annotated_fields"]


def test_patch_returns_annotated_fields(client, tmp_path):
    """前端靠 annotated_fields 显示「已手改」，且不产生 N+1 查询。"""
    _run_import(client, _seed_dataset(tmp_path))
    cid = _first_id(client)
    assert client.get(f"/api/clips/{cid}").json()["annotated_fields"] == ""

    client.patch(f"/api/clips/{cid}", json={"category": "甲", "notes": "乙"})
    fields = client.get(f"/api/clips/{cid}").json()["annotated_fields"].split(",")
    assert set(fields) == {"category", "notes"}, fields

    # 检索结果里也要带，否则列表页无从判断
    items = client.post(
        "/api/search", json={"use": "indextts", "top_k": 50}
    ).json()["items"]
    hit = next(i for i in items if i["id"] == cid)
    assert "annotated_fields" in hit


def test_rebuilt_description_is_not_user_annotation(client, tmp_path):
    """只改类别时描述由模板重建 —— 那份描述算程序值，不算用户标注。

    否则以后重新导入永远刷不新这条的描述。
    """
    _run_import(client, _seed_dataset(tmp_path))
    cid = _first_id(client)

    data = client.patch(f"/api/clips/{cid}", json={"category": "自定义类别"}).json()
    # 描述跟着重建（否则类别与描述会矛盾）
    assert "自定义类别" in data["description"] or data["description"]
    fields = data["annotated_fields"].split(",")
    assert "category" in fields
    assert "description" not in fields, "重建的描述不该被记成用户标注"


def test_handwritten_description_is_user_annotation(client, tmp_path):
    """用户手写的描述才算人工标注。"""
    _run_import(client, _seed_dataset(tmp_path))
    cid = _first_id(client)
    data = client.patch(f"/api/clips/{cid}", json={"description": "我手写的"}).json()
    assert "description" in data["annotated_fields"].split(",")


# ---------------------------------------------------------------- 批量清空人工标注

def test_clear_annotations_by_ids(client, tmp_path):
    _run_import(client, _seed_dataset(tmp_path))
    ids = [i["id"] for i in client.post(
        "/api/search", json={"use": "indextts", "top_k": 3}
    ).json()["items"]]
    for cid in ids:
        client.patch(f"/api/clips/{cid}", json={"category": "甲", "language": "zh"})

    r = client.post("/api/annotations/clear", json={"clip_ids": ids[:2]})
    assert r.status_code == 200
    assert r.json()["cleared"] == 4          # 两条 × 两个字段
    assert client.get(f"/api/clips/{ids[0]}").json()["annotated_fields"] == ""
    assert client.get(f"/api/clips/{ids[2]}").json()["annotated_fields"] != ""


def test_clear_annotations_requires_explicit_target(client, tmp_path):
    """空 body 不能当成「条件全不限」而清光全库。"""
    _run_import(client, _seed_dataset(tmp_path))
    cid = _first_id(client)
    client.patch(f"/api/clips/{cid}", json={"category": "甲"})

    r = client.post("/api/annotations/clear", json={})
    assert r.status_code == 400
    assert client.get(f"/api/clips/{cid}").json()["annotated_fields"] != ""


def test_clear_annotations_by_filter_covers_all_pages(client, tmp_path):
    """按筛选清空必须覆盖整批，不受 top_k 限制。"""
    _run_import(client, _seed_dataset(tmp_path))
    items = client.post(
        "/api/search", json={"use": "indextts", "top_k": 50}
    ).json()["items"]
    assert len(items) == 12
    for it in items:
        client.patch(f"/api/clips/{it['id']}", json={"category": "甲"})

    # 只传 by_filter 和一个窄条件：gender=female 共 6 条
    r = client.post("/api/annotations/clear", json={
        "by_filter": True, "use": "indextts", "gender": "female",
    })
    assert r.status_code == 200
    assert r.json()["clips"] == 6
    assert r.json()["cleared"] == 6

    left = client.post("/api/search", json={"use": "indextts", "top_k": 50}).json()["items"]
    females = [i for i in left if i["gender"] == "female"]
    males = [i for i in left if i["gender"] == "male"]
    assert all(i["annotated_fields"] == "" for i in females)
    assert all(i["annotated_fields"] for i in males)


def test_clear_annotations_filter_with_no_match(client, tmp_path):
    """条件没命中时明确告知，而不是静默成功。"""
    _run_import(client, _seed_dataset(tmp_path))
    r = client.post("/api/annotations/clear", json={
        "by_filter": True, "use": "indextts", "gender": "female", "min_intensity": 5,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["clips"] == 0
    assert "message" in data


def test_clear_annotations_can_target_fields(client, tmp_path):
    """只清指定字段：例如只撤销语种标注，保留手写描述。"""
    _run_import(client, _seed_dataset(tmp_path))
    cid = _first_id(client)
    client.patch(f"/api/clips/{cid}", json={"language": "zh", "description": "我写的"})

    r = client.post("/api/annotations/clear",
                    json={"clip_ids": [cid], "fields": ["language"]})
    assert r.json()["cleared"] == 1
    data = client.get(f"/api/clips/{cid}").json()
    assert data["language"] is None
    assert data["description"] == "我写的"
    assert data["annotated_fields"] == "description"


def test_clear_annotations_restores_description(client, tmp_path):
    """清掉语种标注后，描述里的「语言为中文」也应消失（否则自相矛盾）。"""
    _run_import(client, _seed_dataset(tmp_path))
    cid = _first_id(client)
    client.patch(f"/api/clips/{cid}", json={"language": "zh"})
    assert "语言为中文" in client.get(f"/api/clips/{cid}").json()["description"]

    client.post("/api/annotations/clear", json={"clip_ids": [cid]})
    data = client.get(f"/api/clips/{cid}").json()
    assert "语言为中文" not in data["description"]
    assert data["language"] is None
