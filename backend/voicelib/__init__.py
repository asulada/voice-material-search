"""语音素材库后端（IndexTTS + SeedVC 双线路）。

模块划分：
    config    voice_library/config.yaml 与路径
    db        SQLite schema / CRUD / 统计
    audio     静音裁剪、响度近似、发声段计数
    tags      标签映射、许可推导、描述模板
    embedder  中文文本嵌入（BGE，可降级）
    search    L1 结构化过滤 + L2 语义排序
    sources   数据源目录适配
    ingest    导入管线
    jobs      后台任务 + WebSocket 进度
    export    导出到 exports/ 并写 selection.json
"""

from .config import Config

__all__ = ["Config"]
__version__ = "1.0.0"
