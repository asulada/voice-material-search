"""配置：voice_library/config.yaml 的读写与默认值。

约定（见方案 §2.1）：
  voice_library/
  ├── audio/csemotions/       说话片段
  ├── audio/vocal_burst/      副语言
  ├── index/library.db
  └── config.yaml

为了让后端在 **Windows 与 Linux** 上都能跑，默认库根目录取「项目根 / voice_library」，
并允许用环境变量 VOICE_LIBRARY_ROOT 或 --root 覆盖（§6.7 的 Linux 部署只改这里）。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_TARGET_LUFS = -20.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScanRoot:
    """一个「数据源目录」搜索路径。

    kind:
      csemotions  —— <root>/<speaker>/<emotion>/*.wav（CSEMOTIONS 原始排布）
      emogator    —— <root>/<label>/*.wav（EmoGator 30 类排布）
      flat        —— 扁平目录，需 manifest.json 提供每条素材的标注
      huggingface —— HF datasets 数据集名（见 slug），通过 load_dataset 读取
    """

    kind: str
    path: str = ""
    slug: str = ""
    license: str = "apache-2.0"
    target_use: str = "indextts"  # 'indextts' | 'seedvc'
    gender_map: dict[str, str] = field(default_factory=dict)
    language: str = ""
    label_map: dict[str, str] | None = None
    # 覆盖自动推导的 clip_kind（bad case 用，比如把某个子集强制当副语言）
    clip_kind: str | None = None


@dataclass
class Config:
    root: str
    embed_model: str = DEFAULT_EMBED_MODEL
    target_lufs: float = DEFAULT_TARGET_LUFS
    scan_roots: list[ScanRoot] = field(default_factory=list)
    # 导入时的时长筛选，由 USAGE_RULES 默认值给出；这里留空表示「跟随 usage 规则」
    indextts_duration: tuple[float, float] = (3.0, 15.0)
    seedvc_duration: tuple[float, float] = (0.3, 5.0)

    # ---- 派生路径 ----
    @property
    def root_path(self) -> Path:
        return Path(self.root).resolve()

    @property
    def audio_dir(self) -> Path:
        return self.root_path / "audio"

    @property
    def index_dir(self) -> Path:
        return self.root_path / "index"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "library.db"

    @property
    def export_dir(self) -> Path:
        return self.root_path / "exports"

    @property
    def config_path(self) -> Path:
        return self.root_path / "config.yaml"

    def ensure_dirs(self) -> None:
        for p in (
            self.root_path,
            self.audio_dir / "csemotions",
            self.audio_dir / "vocal_burst",
            self.index_dir,
            self.export_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    # ---- 读写 ----
    def save(self) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "embed_model": self.embed_model,
            "target_lufs": self.target_lufs,
            "indextts_duration": list(self.indextts_duration),
            "seedvc_duration": list(self.seedvc_duration),
            "scan_roots": [asdict(r) for r in self.scan_roots],
        }
        self.config_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, root: str | os.PathLike | None = None) -> "Config":
        root = root or os.environ.get("VOICE_LIBRARY_ROOT") or (PROJECT_ROOT / "voice_library")
        cfg = cls(root=str(Path(root)))
        if cfg.config_path.is_file():
            raw = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8")) or {}
            cfg.embed_model = raw.get("embed_model", cfg.embed_model)
            cfg.target_lufs = float(raw.get("target_lufs", cfg.target_lufs))
            for key in ("indextts_duration", "seedvc_duration"):
                if key in raw and raw[key]:
                    setattr(cfg, key, tuple(float(x) for x in raw[key]))
            for item in raw.get("scan_roots") or []:
                cfg.scan_roots.append(
                    ScanRoot(
                        kind=item.get("kind", "flat"),
                        path=item.get("path") or "",
                        slug=item.get("slug") or "",
                        license=item.get("license") or "apache-2.0",
                        target_use=item.get("target_use") or "indextts",
                        gender_map=item.get("gender_map") or {},
                        language=item.get("language") or "",
                        label_map=item.get("label_map") or None,
                        clip_kind=item.get("clip_kind") or None,
                    )
                )
        return cfg
