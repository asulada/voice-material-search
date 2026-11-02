"""中文文本嵌入（方案 §3.1）。

首选 BAAI/bge-small-zh-v1.5（512 维，CPU 毫秒级）。
模型不可用时自动降级为 **确定性哈希嵌入**：检索仍能按字面/字重叠工作，
不会因为没装 sentence-transformers 就让整个应用起不来。

降级只是兜底，语义能力（"火冒三丈" ≈ "愤怒"）必须真模型才有。
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading

import numpy as np

log = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BaseEncoder:
    backend = "base"
    dim = 0

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # 序列化（所有后端共用）
    def to_blob(self, vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def from_blob(blob: bytes | None) -> np.ndarray | None:
        if not blob:
            return None
        arr = np.frombuffer(blob, dtype=np.float32)
        return arr if arr.size else None


class HashingEncoder(BaseEncoder):
    """零依赖兜底：字 + 2-gram 哈希到固定维度，L2 归一化。

    它只对字面重合敏感（同义改写搜不到），但保证：
      * 每条素材都能建向量
      * 检索接口在模型缺失时仍可用
    """

    backend = "hash-fallback"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = (text or "").strip()
        if not t:
            return v
        grams: list[str] = [ch for ch in t if not ch.isspace()]
        grams += [t[i : i + 2] for i in range(len(t) - 1)]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[idx] += sign
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)


class SentenceTransformerEncoder(BaseEncoder):
    backend = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.model_name = model_name
        lowered = model_name.lower()
        self.zh_model = any(k in lowered for k in ("zh", "bge-m3"))
        self.use_instruction = bool(
            int(os.environ.get("VOICE_LIBRARY_BGE_INSTRUCTION", "1"))
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return v.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        if self.zh_model and self.use_instruction:
            text = BGE_QUERY_INSTRUCTION + text
        return self.encode([text])[0]


class Encoder:
    """Facade：懒加载 + 单例 + 失败降级。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._impl: BaseEncoder | None = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self.get().dim

    @property
    def backend(self) -> str:
        return self.get().backend

    def get(self) -> BaseEncoder:
        if self._impl is not None:
            return self._impl
        with self._lock:
            if self._impl is not None:
                return self._impl
            try:
                self._impl = SentenceTransformerEncoder(self.model_name)
                log.info("已加载嵌入模型 %s (%d 维)", self.model_name, self._impl.dim)
            except Exception as exc:  # noqa: BLE001 - 任何异常都降级
                log.warning(
                    "嵌入模型 %s 不可用（%s），降级为哈希嵌入；语义检索能力受限",
                    self.model_name,
                    exc,
                )
                self._impl = HashingEncoder()
            return self._impl

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.get().encode(texts)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def encode_query(self, text: str) -> np.ndarray:
        return self.get().encode_query(text)

    def to_blob(self, vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def from_blob(blob: bytes | None) -> np.ndarray | None:
        if not blob:
            return None
        arr = np.frombuffer(blob, dtype=np.float32)
        return arr if arr.size else None
