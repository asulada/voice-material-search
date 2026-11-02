# 决策记录

只记录重大修改或重要结论，小修（typo、格式、局部重构、调试）不记。

---

## 2026-11-02 · 按 `语音素材检索方案.md` 落地双线路素材库（FastAPI + Vue）

**背景**
方案文档已完整定义检索架构（§1）、数据模型（§2）、导入流程（§4）、Web 界面（§6），但仓库里只有两份设计文档，没有可运行的实现。需要把方案变成能实际导入 / 检索 / 试听 / 标注 / 导出的应用。

**决定**
按方案逐节实现，不引入方案明确排除的东西：

| 方案要求 | 实现落点 | 说明 |
|---|---|---|
| 不做 CLAP，检索描述文本（§1.1） | `voicelib/embedder.py` + `search.py` | 只对 `description` 建向量 |
| BGE 中文嵌入（§3.1） | `SentenceTransformerEncoder` | 不可用时降级哈希嵌入，界面标注 |
| 两条线路约束进 SQL（§3.3） | `search.py: USAGE_RULES` + `search_for()` | 不在返回后过滤，`top_k` 不会悄悄变少 |
| 只裁数字静音（§4.3） | `audio.py: trim_digital_silence` | 阈值 −90 dBFS，**不用** `librosa.effects.trim` |
| 发声段只做标记（§4.2） | `audio.py: count_segments` + `tags.segment_note` | `n==0` 丢弃，`n>=2` 写 `notes` |
| 响度只算增益（§4.4） | `audio.py: gain_to_target` | 写 `gain_db`，前端 `GainNode` 临时应用 |
| `commercial_ok` 由 license 推导（§4.5） | `tags.commercial_ok_of` | 含 `nc` 即 0；PATCH 改 license 时重算 |
| 单进程 + 单任务并发（§6.1） | `jobs.py: ThreadPoolExecutor(max_workers=1)` | |
| 音频只按 id 取、防目录穿越（§6.6） | `server.py: _safe_audio_path` | `Path.resolve()` + `relative_to` |

**实现中修正的两处方案细节**

1. **`trim_dead_edges` 不能整段重裁。** 方案 §4.3 的示意是「检测到 >1 秒死静音就按 −60 dBFS 重裁整段」。实测这样做会**把另一侧的低电平渐弱尾一起切掉**——正是该节要避免的损失。改为只在「确实死掉的那一侧」收边，另一侧保持不动。已有测试覆盖（`test_trim_preserves_low_level_tail_through_ingest`）。

2. **重复导入必须保留人工字段。** 方案 §2.3 说「描述改了必须重建 embedding」，§7 阶段四说人工描述决定检索质量上限。若重导入时用模板覆盖 `description`，人工成果不可逆丢失。`upsert_clip` 因此对 `verified` / `intensity` / `description` / `notes` 采用「保留已有值」策略，其余列正常刷新。

**影响**
- 库根目录默认取项目根 `voice_library/`，可用 `VOICE_LIBRARY_ROOT` 覆盖（方案默认是 Linux 部署路径）。
- 嵌入降级是**能力降级**而非故障：哈希后端只按字面重合，界面顶部会显式提示。
- 性别信息不在 CSEMOTIONS / EmoGator 目录里，导入向导提供「说话人性别映射」手动补；不补则该列为空，性别筛选会漏。
- 新增 `GET /api/clips/{id}/waveform`、`POST /api/clips/{id}/delete`、`GET /api/import/preview`、`POST /api/reebed`（全库重建），均在方案 §6.2 表之外但被界面 §6.4 / §6.5 需要。

---

## 2026-11-02 · 请求模型必须定义在模块级（FastAPI + `from __future__ import annotations`）

**背景**
`server.py` 最初把 Pydantic 请求模型定义在 `create_app()` 内部。所有 POST 接口返回 422，detail 为 `{"type":"missing","loc":["query","body"]}`。

**决定**
把 `SearchBody` / `ClipPatch` / `ImportBody` / `ExportBody` / `ReembedBody` 提到模块级。原因：`server.py` 开启了 `from __future__ import annotations`，函数内局部定义的模型在运行时无法被 FastAPI 解析其注解，于是请求体被当作 query 参数。

**影响**
后续在本文件新增请求模型时必须写在模块级，不能写进 `create_app()`。

---

## 2026-11-02 · 目录类型识别不能先看 `manifest.json`

**背景**
`POST /api/import/preview` 把 `<类别>/*.mp3` 形态的 EmoGator 目录识别成 `flat`，报「0 条素材」。原因是 `guess_kind` 先判断根目录有没有 `manifest.json`，而该目录恰好带了一个。

**决定**
`guess_kind` 改为先看目录**形状**（`<speaker>/<emotion>/` → csemotions，`<label>/` 含音频 → emogator，根目录平铺 → 视 manifest 定 flat），最后才看 manifest。

**影响**
- 前端预览时**不传** `kind`，让后端自动识别；识别结果回填表单后再允许用户手改。
- 若某数据集同时符合两种形状，以「更深的、带情绪目录名的那一层」为 csemotions。
