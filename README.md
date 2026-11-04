# 语音素材库

为 **IndexTTS**（`emo_audio_prompt`）与 **SeedVC**（`--target`）两条下游线路提供参考音频的本地素材库。
实现依据：`语音素材检索方案.md`。

- **检索的是「描述文本」，不是音频**（不用 CLAP，中文查询不会静默失效）
- **两条线路的约束进 SQL**：IndexTTS 只取 speech / 3–15 秒，SeedVC 只取 vocal_burst / 0.3–5 秒
- **只有一份音频**：入库时就裁好、切好，可直接传给下游
- **响度只算增益不改文件**：`gain_db` 记库，试听时临时应用

## 环境准备

三个先决条件：

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | **3.9+**，推荐 3.11 / 3.12 | 3.12 为实测版本；源码的 `str \| None` 写法由 `from __future__ import annotations` 兜住，3.9 也能导入 |
| Node.js | **20+** | 仅用于构建前端（Vite 6） |
| npm | 随 Node 附带 | |

下面三种方式**任选其一**，都在项目根目录（本 README 所在目录）执行。
装好依赖后，`run.sh` / `run.bat` / `pytest` 都要在**同一个激活状态**下运行
——它们都取 PATH 里的 Python（Windows 的 `run.bat` 另外会优先用 `%CONDA_PREFIX%`），
所以换了终端就要重新激活。

### 安装体积与优化

这是自动安装 PyTorch 时最需要注意的一点：**PyPI 上的默认 PyTorch 包通常包含了 CUDA（GPU 支持）依赖，体积非常庞大，可能达到数 GB**。

如果你的电脑没有 NVIDIA GPU，或者不需要 GPU 加速，直接安装默认包会浪费大量磁盘空间和下载时间。

**一个常见的优化做法是**：在安装 `sentence-transformers` **之前**，先手动安装 **CPU 版本的 PyTorch**。这样当 `pip` 后续解析依赖时，会发现 `torch` 已存在且版本满足要求，从而跳过下载庞大的 CUDA 版本。

bash

```
# 先安装 CPU 版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 再安装 sentence-transformers，此时它会复用已安装的 CPU 版 PyTorch
pip install sentence-transformers
```

### 方式 A：conda

```bash
conda create -n voice-library python=3.11 -y
conda activate voice-library
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

不想激活、每次显式指定环境也可以：

```bash
conda run -n voice-library python -m pip install -r backend/requirements.txt
conda run -n voice-library ./run.sh            # Linux / macOS
```

> **Windows + conda 注意**：`conda activate voice-library` 之后可以直接运行 `run.bat`
> ——它优先使用 `%CONDA_PREFIX%\python.exe`。这一点必须显式处理：Windows 的
> `py` launcher 只认 `VIRTUAL_ENV`，**不认** `CONDA_PREFIX`，若默认走 `py`
> 会静默落到 base 环境（装的依赖不在那儿）。
> 若 `conda activate` 完全没生效（提示 `CommandNotFoundError`），先跑一次
> `conda init cmd.exe`（或 `conda init powershell`）再重开终端。

退出与删除环境：

```bash
conda deactivate
conda env remove -n voice-library
```

### 方式 B：venv（Linux / macOS）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

### 方式 C：venv（Windows）

PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

命令提示符（cmd）：

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

> PowerShell 若提示「禁止运行脚本」，先执行一次：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
>
> venv 激活后 `VIRTUAL_ENV` 被设置，`python` 与 `py` 都指向 `.venv`（已实测），
> `run.bat` 的默认值可以直接用。

退出环境（两种 shell 相同）：`deactivate`

### 只想要能跑起来（暂时跳过语义模型）

`requirements.txt` 里的 `sentence-transformers` 会连带安装 PyTorch，下载量大、耗时长。
只想先验证界面与检索流程时，可以只装核心依赖：

```bash
python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" numpy soundfile scipy librosa pyyaml python-multipart
python -m pip install pytest httpx          # 需要跑测试时
```

此时界面顶栏会标「哈希兜底（无语义）」，只能按字面匹配（见下文「关于嵌入模型」）。

## 快速开始（依赖已装好之后）

```bash
# 1. 确认环境已激活（命令行前缀出现 (voice-library) 或 (.venv)）
python -c "import fastapi, librosa; print('依赖就绪')"

# 2. 前端构建（产物由 FastAPI 直接托管）
cd frontend
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
| **检索** | 按线路/类别/性别/语种/强度/时长/许可筛选（类别框可输中文名或英文原名过滤）；中文描述语义排序；**行内试听**（按 `gain_db` 统一音量）；星标改强度、✓ 改已验证；点描述直接编辑（保存后自动重建向量）；多选导出 |
| **导入** | **只需填目录**：预览会自动拆分目录——一个根目录里混着 CSEMOTIONS / EmoGator / 自定义目录也能一次导完，每个数据源按**自己的**排布解读类型与线路；许可与条数一并列出，确认后后台运行 + WebSocket 进度，可取消 |
| **统计** | 总量/时长/已验证/已建向量；各线路可用类别；类别与许可分布；批量补建/全库重建向量；导出历史 |

## 导入的数据源形态

支持五种排布。**不必先知道自己的目录属于哪种**：预览会逐个子目录识别，
一个根目录里混着多种排布也能一次导完，各自按自己的排布入库。

| `kind` | 排布 | 典型数据 | 自动判定为 |
|---|---|---|---|
| `csemotions` | `<根>/<说话人>/<情绪>/*.wav` | CSEMOTIONS（Apache-2.0） | IndexTTS · 说话片段 |
| `emogator` | `<根>/<类别>/*.mp3` 或 `*.wav` | EmoGator（Apache-2.0）、自制副语言 | SeedVC · 副语言 |
| `flat` | 任意层级 + 清单文件，或平铺 | 自定义素材 | 有台词 → IndexTTS；否则按时长推断 |
| `parquet` | `<根>/train-*-of-*.parquet` | HF datasets（`audio` 列内嵌音频） | IndexTTS · 说话片段 |
| `webdataset` | `<根>/<分片>/<key>.flac` + `<key>.json` | speech-utterances 聚合包 | 按子集，见下 |

### webdataset 聚合包（speech-utterances）

`gijs/speech-utterances` 不是一个数据集，而是 **6 个子集的聚合**，每个子集
**许可各不相同**。直接填根目录即可，程序会按子集拆开、各自按真实许可入库：

| 子集 | 许可 | 可商用 | 线路 |
|---|---|---|---|
| EmoGator | Apache-2.0 | ✅ | SeedVC |
| VIVAE | CC-BY-4.0 | ✅ | SeedVC |
| VocalSound | CC-BY-SA-4.0 | ✅（有 ShareAlike） | SeedVC |
| Expresso | CC-BY-NC-4.0 | ❌ | IndexTTS |
| NonSpeech7k | CC-BY-NC-SA-4.0 | ❌ | SeedVC |
| DeeplyNonverbalVocalization | CC-BY-NC-ND-4.0 | ❌ | SeedVC |

**不要用整包 README 里的 `license: cc-by-nc-sa-4.0`**：那只代表最严的那个子集。
整包统一落库会把 EmoGator / VIVAE 里本来可商用的素材永久标成不可商用。

其它要点：

- 支持已解压目录与未解压 `.tar` 两种形态；两者都在时只读目录，不重复导入
- `NonSpeech7k/train/`、`/test/` 这种 split 层会被正确归到同一子集
- VIVAE 自带 `intensity`（low/moderate/strong/peak），直接映射为 2–5 星，免手工打分
- Expresso：`text` 恰为 `"Expressive speech in a neutral style"` 的条目是整段双人对话，
  按用户听过音频后的确认**整条跳过**；同子集里其余条目正常，
  `text` 是真台词，照常入库并保留。
  （不要用 `style == "default"` 当判据：实测有 825 条，大多是真台词句子，会误伤。）
- 类别取 `emotion` → `label` → `style`，说话人取 `speaker` → `speaker_id` → `contributor_id`

### parquet 数据集

从 Hugging Face 下载的 CSEMOTIONS 这类数据集，拿到的是 `train-0000X-of-00008.parquet`，
音频**内嵌在文件里**，不是散落的 wav。这种目录可以直接填进来：

- 音频从 `audio.bytes` 就地解码，**不落中间 wav**，磁盘不会翻倍
- 类别取 `emotion` 列，台词取 `text` 列，性别从 `speaker` 前缀推断（`female*` / `male*`）
- 需要 `pyarrow`：`pip install pyarrow`（已在 `requirements.txt` 里）

**文件名不可信**：实测数据里 `female003_jolliest_01.wav` 的 `emotion` 列是 `playfulness`。
所以类别只认列值，不解析文件名。

清单文件按顺序找：`manifest.json` / `manifest.jsonl` / `index.json` /
`metadata.jsonl` / `metadata.json` / `metadata.csv`（后三个是 HF parquet 导出工具的默认产物）。
**清单不必和音频同层**：在导入目录及其上 4 层内会向上查找，所以填总目录、填 `wav/`、
填 `<说话人>/` 都能匹配上。

每条记录指向音频的键可以是 `filepath` / `path` / `file` / `wav_path` /
`audio_path` / `original_filename` 等（导出工具命名不统一）。

```json
[
  {"filepath": "laugh/001.wav", "category": "laugh", "gender": "male", "intensity": 4}
]
```

可选字段：`speaker_id`（或 `speaker`）/ `text`（或 `transcript` / `sentence` / `content`）/
`language` / `notes`。台词会写进库里，并在描述中替代「（无台词）」。
性别没填时会从说话人名推断（`female001` → `female`）。
文件名里的 `NNNNNN-EE-I.mp3`（EmoGator 命名）会被解析出说话人编号。

### 有台词就走 IndexTTS，不看目录形状

线路判定**先看内容，再看形状**：

| 情况 | 判定 |
|---|---|
| 清单里有非空台词 | IndexTTS · 说话片段（`speech`） |
| 没有台词 | 按目录形状 / 时长推断（副语言 → SeedVC） |

这条很关键。`<说话人>/<情绪>/*.wav` 的形状恰好符合 `<类别>/*.wav`，
只看形状会被判成副语言，8.5 秒的台词就会被塞进 SeedVC 的 0.3–5 秒窗口，
然后全部标成「超出范围」。副语言不可能有台词，所以「有台词」比形状和时长都更本质。

### 选哪一层做分类名

默认情况下类别取自**直接父目录**或台账。但目录结构不总是能表达意图，
所以预览卡片顶部会列出「分类层级」：从导入目录开始向上，直到盘符或用户主目录。

以 `a/b/c/{1,2}.wav`、`a/b/d/{2,4}.wav`、`a/5.wav` 为例，在 `a/b/c` 上预览会列出：

| 选这个 | 扫描范围 | 类别名 |
|---|---|---|
| `c` | 只有 `c` 下的 2 条 | `c` |
| `b` | `b` 全子树 4 条（含 `c` 与 `d`） | 统一为 `b` |
| `a` | 整棵 5 条 | 统一为 `a` |

规则是**选定层级后不再做形状识别**：该层以下所有音频统一挂该层目录名。
因为你已经明确表达了「这些属于一组」，再按子目录名拆开就违背意图。

同名文件不会互相覆盖：落盘名是 `<说话人>_<类别>_<内容hash>.wav`，
`c/2.wav` 与 `d/2.wav` 内容不同 → hash 不同。当前的 4 字节 hash 在几千条规模下
碰撞概率极低；真撞了会走 `ON CONFLICT(filepath)` 覆盖，届时再上编号即可。

命令行与 API 也能用：

```bash
# 页面上的用法
POST /api/import  {"path": "...", "category_level": "b"}
GET  /api/import/preview?path=...&category_level=b

# 已有库批量重分类（不动音频文件，只改库里的类别/线路/描述）
cd backend
py -m tools.reclassify --root <库根> --level b          # 先预览
py -m tools.reclassify --root <库根> --level b --apply  # 再执行
```

**一个副作用要知道**：类别名会进描述，例如「一位女性用 b 的语气说：……」。
对中文情绪词（angry / 悲伤）读起来自然，对纯目录名（`b`、`c`）会别扭。
如果这一层不是情绪维度的名字，导入后建议人工改 `description`。

## 入库时发生了什么

```
读音频 → 只裁数字静音（−90 dBFS）→ 时长越界只标 notes → 写 audio/ → 中文描述 → 向量 → 入库
```

几个刻意的选择，都有方案里的实测依据：

- **时长越界不丢弃，只标 `notes`**：过短（IndexTTS <3s、SeedVC <0.3s）与超长
  （IndexTTS >15s、SeedVC >5s）都入库，分别在 `notes` 写
  「低于 Ns 下限，作参考偏短」/「超出 A–B s 范围，使用前需人工截取」。
  原因：实测 CSEMOTIONS 有 74 条 <3s，其中 **73 条是 angry 的短促爆发**
  （「滚！别让我再看到你！」1.75s）——恰恰是该情绪最典型、最干净的样本，
  按硬阈值丢弃等于把愤怒类的高质量短句全部排掉。
  检索侧仍按线路时长过滤，所以它们不会污染默认结果，放宽时长后可见。
  导入结果里 `imported_short` / `imported_over_long` 分别计数，页面会给出提示。
- **不用 `librosa.effects.trim`**：它以峰值为基准，会在渐弱收尾中途下刀（−42 dBFS 是发声，不是静音）。
  只有首/尾出现 **>1 秒、且整段都低于 −60 dBFS** 的死静音时才收拾那一侧，另一侧绝不碰。
- **副语言的发声段计数只写 `notes`**：`n==0`（静音/损坏）丢弃，`n>=2` 入库并标记建议人工核对。
  抽泣天然是多段，不是缺陷，不丢弃。
- **`commercial_ok` 由 `license` 推导**，不允许手填。只有明确的可商用许可才是 1；
  含 `nc` 条款、空值、以及 `unknown` 这类不明确取值一律记为 0（保守优先）。
  导入页会按**拆分结果**预填许可：所有数据源都是 CSEMOTIONS / EmoGator 时才给
  Apache-2.0，只要有一个是自定义目录就按「未知」处理；预览卡上始终显示
  「许可 X · 可商用/不可商用」。
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

确保环境已激活（`conda activate voice-library` 或 `source .venv/bin/activate`），再：

```bash
cd backend
python -m pytest tests/          # 62 条：音频处理 / 标签 / 检索 / 导入 / API
```

Windows 下若用 `py` 显式指定版本（不走激活环境），依赖要装在同一个解释器里：

```bat
cd backend
py -3.12 -m pytest tests/
```

前端构建校验：`cd frontend && npm run build`。

## 已知边界

- **语义检索依赖 BGE**。哈希兜底只按字面匹配，不是语义能力。
- **性别信息不在数据集里**。CSEMOTIONS / EmoGator 的目录没带性别，
  导入页的「高级选项 → 说话人性别映射」用于补上；不补则该字段为空，`gender` 筛选会漏。
- **Vue 前端不含 TS**。用 JSDoc 标注类型，换取零构建复杂度；类型错误不会在构建期被发现。
- **导入是单任务队列**（并发 1）。这是刻意的：BGE 和 SQLite 写入都不适合并发，
  且要保证检索始终可响应。
