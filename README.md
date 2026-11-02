# 语音素材库

为 **IndexTTS**（`emo_audio_prompt`）与 **SeedVC**（`--target`）两条下游线路提供参考音频的本地素材库。
实现依据：`语音素材检索方案.md`。

- **检索的是「描述文本」，不是音频**（不用 CLAP，中文查询不会静默失效）
- **两条线路的约束进 SQL**：IndexTTS 只取 speech / 3–15 秒，SeedVC 只取 vocal_burst / 0.3–5 秒
- **只有一份音频**：入库时就裁好、切好，可直接传给下游
- **响度只算增益不改文件**：`gain_db` 记库，试听时临时应用

## 快速开始

```bash
# 1. 后端依赖
cd backend
python -m pip install -r requirements.txt

# 2. 前端构建（产物由 FastAPI 直接托管）
cd ../frontend
npm install
npm run build
cd ..

# 3. 启动
./run.sh            # Linux / macOS
run.bat             # Windows
```

打开 <http://127.0.0.1:8000/>。只监听 127.0.0.1（这个服务能读本地任意已入库音频）。

### 开发模式（前端热更新）

```bash
cd backend && python -m uvicorn server:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev          # http://127.0.0.1:5173，Vite 代理 /api 到 8000
```

### 关于嵌入模型

首选 `BAAI/bge-small-zh-v1.5`（512 维，CPU 毫秒级，约 95MB）：

```bash
python -m pip install sentence-transformers
```

首次启动会下载模型。**未安装时自动降级为哈希嵌入**，界面顶栏会标注
「哈希兜底（无语义）」——此时只能按字面重合检索（"愤怒"能匹配"愤怒"，
但"火冒三丈"匹配不到），应用不会因此起不来。

## 目录结构

```
voice_library/                # 库根目录（VOICE_LIBRARY_ROOT 可改）
├── audio/
│   ├── csemotions/           # 说话片段（IndexTTS）
│   └── vocal_burst/          # 副语言（SeedVC）
├── index/library.db          # SQLite
├── exports/<时间戳>/         # 导出的成品 + selection.json
├── voices/                   # 放 spk_audio_prompt 的候选（可选）
└── config.yaml               # 可选，见 backend/config.example.yaml
```

## 界面

| 页面 | 能做什么 |
|---|---|
| **检索** | 按线路/类别/性别/强度/时长/许可筛选；中文描述语义排序；**行内试听**（按 `gain_db` 统一音量）；星标改强度、✓ 改已验证；点描述直接编辑（保存后自动重建向量）；多选导出 |
| **导入** | 填目录 → 预览（自动识别 CSEMOTIONS / EmoGator / 自定义目录 + 样本条数）→ 后台任务 + WebSocket 进度；可试跑、可取消 |
| **统计** | 总量/时长/已验证/已建向量；各线路可用类别；类别与许可分布；批量补建/全库重建向量；导出历史 |

## 导入的数据源形态

| `kind` | 排布 | 典型数据 |
|---|---|---|
| `csemotions` | `<根>/<说话人>/<情绪>/*.wav` | CSEMOTIONS（Apache-2.0） |
| `emogator` | `<根>/<类别>/*.mp3` 或 `*.wav` | EmoGator（Apache-2.0）、自制副语言 |
| `flat` | 任意层级 + `manifest.json` | 自定义素材 |

`manifest.json` 每条至少要有 `filepath` 与 `category`：

```json
[
  {"filepath": "laugh/001.wav", "category": "laugh", "gender": "male", "intensity": 4}
]
```

可选字段：`speaker_id` / `transcript` / `language` / `notes`。
文件名里的 `NNNNNN-EE-I.mp3`（EmoGator 命名）会被解析出说话人编号。

## 入库时发生了什么

```
读音频 → 只裁数字静音（−90 dBFS）→ 时长筛选 → 写 audio/ → 中文描述 → 向量 → 入库
```

几个刻意的选择，都有方案里的实测依据：

- **不用 `librosa.effects.trim`**：它以峰值为基准，会在渐弱收尾中途下刀（−42 dBFS 是发声，不是静音）。
  只有首/尾出现 **>1 秒、且整段都低于 −60 dBFS** 的死静音时才收拾那一侧，另一侧绝不碰。
- **副语言的发声段计数只写 `notes`**：`n==0`（静音/损坏）丢弃，`n>=2` 入库并标记建议人工核对。
  抽泣天然是多段，不是缺陷，不丢弃。
- **`commercial_ok` 由 `license` 推导**，不允许手填；含 `nc` 即 0。
- **重复导入保留人工成果**：`verified` / `intensity` / `description` / `notes` 不被模板覆盖。

## API

完整交互式文档：<http://127.0.0.1:8000/docs>

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/stats` | 库总量、各来源/类别/性别分布 |
| POST | `/api/search` | 检索（L1 SQL 过滤 + L2 语义排序） |
| GET | `/api/clips/{id}/audio` | 流式音频（支持 Range） |
| GET | `/api/clips/{id}/waveform` | 波形点（前端画图 + 显示播放进度） |
| PATCH | `/api/clips/{id}` | 改 `verified` / `intensity` / `description` / `notes` … |
| POST | `/api/clips/{id}/reebed` | 描述改动后重建该条向量 |
| POST | `/api/import` | 启动导入任务，返回 `job_id` |
| GET | `/api/jobs/{id}` | 任务状态 |
| WS | `/api/jobs/{id}/ws` | 进度推送 |
| POST | `/api/export` | 导出选中片段 + 写 `selection.json` |
| GET | `/api/voices` | 列出 `<根>/voices/` 下的音色候选 |

**检索请求示例**

```json
{
  "use": "indextts",
  "q": "压抑但克制的愤怒",
  "categories": ["Angry"],
  "gender": "male",
  "min_intensity": 3,
  "commercial_only": true,
  "top_k": 20
}
```

`use="all"` 可跨线路浏览全库（不套用途约束）。

## 下游怎么用

```python
# IndexTTS —— 不要同时传 emo_vector 或 use_emo_text，否则 emo_audio_prompt 被静默丢弃
tts.infer(
    spk_audio_prompt=voice_path,
    emo_audio_prompt=exported_clip_path,
    emo_alpha=alpha,
    text=text,
    lang="ZH",
)
```

```bash
# SeedVC —— --target 同时给音色和风格，无法分离
python inference_v2.py --source <副语言片段> --target <音色样本> --convert-style true
```

导出的是**独立文件**，满足 IndexTTS 只取前 15 秒的行为。

## 测试

```bash
cd backend
python -m pytest tests/          # 62 条：音频处理 / 标签 / 检索 / 导入 / API
```

前端构建校验：`cd frontend && npm run build`。

## 已知边界

- **语义检索依赖 BGE**。哈希兜底只按字面匹配，不是语义能力。
- **性别信息不在数据集里**。CSEMOTIONS / EmoGator 的目录没带性别，
  导入页的「高级选项 → 说话人性别映射」用于补上；不补则该字段为空，`gender` 筛选会漏。
- **Vue 前端不含 TS**。用 JSDoc 标注类型，换取零构建复杂度；类型错误不会在构建期被发现。
- **导入是单任务队列**（并发 1）。这是刻意的：BGE 和 SQLite 写入都不适合并发，
  且要保证检索始终可响应。
