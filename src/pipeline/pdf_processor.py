import sys
import re
from pathlib import Path
import tempfile
import shutil
from pdf2image import convert_from_path
from src.config import config


def parse_page_metadata(image_path: Path | str) -> tuple[str, int]:
    """
    从图片文件名解析 document_id 与 page_number。
    示例:
        '1_page_0025.png' -> ('1', 25)
        'page_0003.png'   -> ('doc', 3)
    """
    path_obj = Path(image_path)
    stem = path_obj.stem
    match = re.match(r"^(.*?)_page_(\d+)$", stem)
    if match:
        doc_id = match.group(1)
        page_num = int(match.group(2))
    elif stem.startswith("page_"):
        doc_id = "doc"
        page_num = int(stem.replace("page_", ""))
    else:
        doc_id = stem
        page_num = 1
    return doc_id, page_num


def resolve_pdf_files(
    pdf_target: Path | str | list[Path | str] | None = None,
    pdf_dir: Path | None = None,
    allow_empty: bool = False,
) -> list[Path]:
    """解析 PDF 文件列表"""
    target_pdf_dir = pdf_dir or config.pdf_dir

    if pdf_target is not None:
        if isinstance(pdf_target, (list, tuple)):
            pdf_files = []
            for p in pdf_target:
                path_item = Path(p)
                if path_item.is_dir():
                    pdf_files.extend(sorted(list(path_item.glob("*.pdf"))))
                else:
                    pdf_files.append(path_item)
        else:
            target_path = Path(pdf_target)
            if target_path.is_dir():
                pdf_files = sorted(list(target_path.glob("*.pdf")))
            elif target_path.is_file():
                pdf_files = [target_path]
            elif not target_path.exists():
                if allow_empty:
                    return []
                raise FileNotFoundError(f"❌ 指定的路径不存在: {target_path}")
            else:
                pdf_files = []
    else:
        pdf_files = sorted(list(target_pdf_dir.glob("*.pdf"))) if target_pdf_dir.exists() else []

    if not pdf_files:
        if allow_empty:
            return []
        raise FileNotFoundError(
            f"❌ 未找到任何可处理的 PDF 文件！\n"
            f"💡 请将 PDF 文件放入 {target_pdf_dir} 目录或指定 pdf_target。"
        )

    for pdf_file in pdf_files:
        if not pdf_file.exists():
            if allow_empty:
                continue
            raise FileNotFoundError(f"❌ PDF 文件不存在: {pdf_file}")

    return pdf_files


def convert_pdf_to_images(
    pdf_target: Path | str | list[Path | str] | None = None,
    pdf_dir: Path | None = None,
    output_dir: Path | None = None,
    dpi: int = 150,
    allow_empty: bool = False,
) -> list[Path]:
    """
    将 PDF 文件转换为 PNG 页面图片。
    使用临时目录转换，成功后再更新目标目录。
    """
    target_output_dir = output_dir or config.pages_dir
    target_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_dir = target_output_dir.resolve()

    target = pdf_target or pdf_dir
    pdf_files = resolve_pdf_files(target, pdf_dir=pdf_dir, allow_empty=allow_empty)
    if not pdf_files:
        print(f"⚠️ 未找到任何 PDF 文件，跳过转换图片操作。")
        return []

    print(f"📄 共找到 {len(pdf_files)} 个 PDF 文件，准备导出至: {resolved_output_dir}")

    is_multi_pdf = len(pdf_files) > 1

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_files = []

            for pdf_idx, pdf_file in enumerate(pdf_files, 1):
                print(f"  [{pdf_idx}/{len(pdf_files)}] 正在导出: {pdf_file.name} ...")
                pages = convert_from_path(pdf_file, dpi=dpi)
                num_digits = max(4, len(str(len(pages))))

                for i, page in enumerate(pages):
                    if is_multi_pdf:
                        filename = f"{pdf_file.stem}_page_{i + 1:0{num_digits}d}.png"
                    else:
                        filename = f"page_{i + 1:0{num_digits}d}.png"

                    out_file = temp_path / filename
                    page.save(out_file)
                    temp_files.append(filename)

            # 清理旧的 PNG 图片并移动新图片
            for old_img in target_output_dir.glob("*.png"):
                if not old_img.name.startswith("."):
                    old_img.unlink()

            new_file_paths = []
            for filename in temp_files:
                target_file = target_output_dir / filename
                shutil.move(str(temp_path / filename), str(target_file))
                new_file_paths.append(target_file)

        print(
            f"🎉 成功将 {len(pdf_files)} 个 PDF (共 {len(new_file_paths)} 页) 导出至: {resolved_output_dir}"
        )
        return sorted(new_file_paths)

    except Exception as e:
        print(f"\n❌ PDF 转换失败: {e}")
        raise RuntimeError(f"PDF 转换失败: {e}") from e
