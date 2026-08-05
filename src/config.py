from pathlib import Path
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """全局应用配置对象"""
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    pdf_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "pdf")
    pages_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "pages")
    index_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "index.pt")
    evidence_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "evidence")

    model_name: str = "athrael-soju/colqwen3.5-4.5B-v3"
    qwen_vl_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_vl_backend: str = "local"  # "local" 或 "api"
    dpi: int = 150
    batch_size: int | None = None  # None 时自动判别 (CUDA=8, MPS/CPU=4)

    # 工业 API 部署配置 (Milestone 5)
    gea_api_key: str = Field(
        default_factory=lambda: __import__("os").getenv("GEA_API_KEY", "gea-industrial-secret-key")
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def ensure_directories(self) -> None:
        """确保数据与证据相关目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()
