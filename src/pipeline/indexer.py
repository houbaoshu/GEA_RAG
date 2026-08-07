from pathlib import Path
import torch
from PIL import Image
from src.config import config, AppConfig
from src.utils.device import get_device_and_dtype
from src.utils.model_loader import load_colqwen_model
from src.pipeline.pdf_processor import convert_pdf_to_images


class DocumentIndexer:
    """Document Embedding & Multi-vector Index Builder"""

    def __init__(self, app_config: AppConfig = config):
        self.cfg = app_config

    def build_index(
        self,
        output_index_path: Path | None = None,
        batch_size: int | None = None,
        force_rebuild: bool = False,
    ) -> Path:
        """
        根据 data/pages/ 下的图片提取 ColQwen 多向量特征并保存索引 index.pt。
        若 pages/ 目录为空，则自动触发 PDF 转换。
        """
        self.cfg.ensure_directories()
        index_file = output_index_path or self.cfg.index_path

        if index_file.exists() and not force_rebuild:
            print(f"✅ 索引文件已存在: {index_file.resolve()}")
            return index_file

        device, dtype = get_device_and_dtype()

        # 检查待处理的页面图片
        image_paths = sorted(
            [p for p in self.cfg.pages_dir.glob("*.png") if not p.name.startswith(".")]
        )

        if not image_paths:
            print(f"⚠️ 目录 '{self.cfg.pages_dir}' 下未找到图片，正在检查 PDF 文件并触发转换...")
            try:
                image_paths = convert_pdf_to_images(
                    pdf_dir=self.cfg.pdf_dir,
                    output_dir=self.cfg.pages_dir,
                    dpi=self.cfg.dpi,
                    allow_empty=True,
                )
            except Exception as e:
                print(f"⚠️ PDF 转换遇到异常: {e}")
                image_paths = []

        if not image_paths:
            print(f"⚠️ 提示: 当前未找到任何可用的页面图片或 PDF 文件，将创建空索引。")
            index_data = {
                "image_paths": [],
                "embeddings": [],
                "model_name": self.cfg.model_name,
            }
            index_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(index_data, index_file)
            print(f"✅ 空索引文件已建立: {index_file.resolve()}")
            return index_file

        processor, model = load_colqwen_model(self.cfg.model_name, device, dtype)

        print(f"🖼️ 共 {len(image_paths)} 张页面图片，开始提取多向量特征...")

        effective_batch_size = (
            batch_size
            or self.cfg.batch_size
            or (8 if device == "cuda" else 4)
        )
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(image_paths), effective_batch_size):
                batch_paths = image_paths[i : i + effective_batch_size]

                images = []
                for p in batch_paths:
                    with Image.open(p) as img:
                        images.append(img.convert("RGB"))

                inputs = processor.process_images(images).to(device)
                page_embeddings = model(**inputs)

                for emb in page_embeddings:
                    all_embeddings.append(emb.cpu())

                done = min(i + effective_batch_size, len(image_paths))
                print(
                    f"已处理 [{done}/{len(image_paths)}] 页 ({(done / len(image_paths)) * 100:.1f}%)"
                )

        index_data = {
            "image_paths": [p.name for p in image_paths],
            "embeddings": all_embeddings,
            "model_name": self.cfg.model_name,
        }

        index_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(index_data, index_file)
        print(f"\n🎉 索引构建完成！包含 {len(all_embeddings)} 页特征向量，保存至: {index_file.resolve()}")
        return index_file

    def ensure_index_exists(self) -> Path:
        """确保索引文件 index.pt 存在；若不存在则自动生成。"""
        if not self.cfg.index_path.exists():
            print(f"⚠️ 索引文件 {self.cfg.index_path} 不存在，启动自动索引构建...")
            return self.build_index()
        return self.cfg.index_path
