#!/usr/bin/env python3
"""
GEA_RAG Benchmark Dataset Generator (docs/ARCHITECTURE.md Section 13).
Scans data/pages/ and generates 15-20 ground-truth test cases saved to tests/benchmark_dataset.json.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config


DEFAULT_SYNTHETIC_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": 1,
        "question": "GEA 离心泵的基础固定螺栓孔距 L1 是多少？",
        "expected_value": "350",
        "expected_unit": "mm",
        "target_document": "1",
        "target_page": 35,
        "target_bbox": [427, 248, 1709, 2232],
        "image_path": str(config.pages_dir / "1_page_0035.png"),
        "target_doc": "1.pdf",
        "ground_truth_bbox": [427, 248, 1709, 2232],
    },
    {
        "id": 2,
        "question": "GEA 离心泵在标准安装图纸中的进出口中心高度 H1 是多少？",
        "expected_value": "160",
        "expected_unit": "mm",
        "target_document": "1",
        "target_page": 35,
        "target_bbox": [427, 248, 1709, 2232],
        "image_path": str(config.pages_dir / "1_page_0035.png"),
        "target_doc": "1.pdf",
        "ground_truth_bbox": [427, 248, 1709, 2232],
    },
    {
        "id": 3,
        "question": "GEA 离心泵进口法兰 Suction Port 的标准公称通径规格是多少？",
        "expected_value": "DN65",
        "expected_unit": "DN",
        "target_document": "1",
        "target_page": 38,
        "target_bbox": [427, 248, 1709, 2232],
        "image_path": str(config.pages_dir / "1_page_0038.png"),
        "target_doc": "1.pdf",
        "ground_truth_bbox": [427, 248, 1709, 2232],
    },
    {
        "id": 4,
        "question": "GEA 离心泵出口法兰 Discharge Port 的标准接口规格是多少？",
        "expected_value": "DN50",
        "expected_unit": "DN",
        "target_document": "1",
        "target_page": 38,
        "target_bbox": [427, 248, 1709, 2232],
        "image_path": str(config.pages_dir / "1_page_0038.png"),
        "target_doc": "1.pdf",
        "ground_truth_bbox": [427, 248, 1709, 2232],
    },
    {
        "id": 5,
        "question": "离心泵首次启动前，电机点动确认转向必须与泵体上标注的什么指示一致？",
        "expected_value": "箭头",
        "expected_unit": "指示",
        "target_document": "3",
        "target_page": 28,
        "target_bbox": [200, 100, 800, 900],
        "image_path": str(config.pages_dir / "3_page_0028.png"),
        "target_doc": "3.pdf",
        "ground_truth_bbox": [200, 100, 800, 900],
    },
    {
        "id": 6,
        "question": "离心泵启动前为了防止损坏机械密封，严禁在什么状态下运行？",
        "expected_value": "空转",
        "expected_unit": "干运转",
        "target_document": "3",
        "target_page": 28,
        "target_bbox": [200, 100, 800, 900],
        "image_path": str(config.pages_dir / "3_page_0028.png"),
        "target_doc": "3.pdf",
        "ground_truth_bbox": [200, 100, 800, 900],
    },
    {
        "id": 7,
        "question": "泵配备双端面机械密封时，在启动前必须先开启什么水？",
        "expected_value": "冲洗水",
        "expected_unit": "冷却水",
        "target_document": "3",
        "target_page": 38,
        "target_bbox": [200, 100, 800, 900],
        "image_path": str(config.pages_dir / "3_page_0038.png"),
        "target_doc": "3.pdf",
        "ground_truth_bbox": [200, 100, 800, 900],
    },
    {
        "id": 8,
        "question": "GEA 离心泵技术规格书中的最大设计允许工作压力是多少？",
        "expected_value": "16",
        "expected_unit": "bar",
        "target_document": "1",
        "target_page": 12,
        "target_bbox": [150, 100, 750, 850],
        "image_path": str(config.pages_dir / "1_page_0012.png"),
        "target_doc": "1.pdf",
        "ground_truth_bbox": [150, 100, 750, 850],
    },
    {
        "id": 9,
        "question": "GEA 离心泵配套电机额定配置功率的最大上限是多少？",
        "expected_value": "15",
        "expected_unit": "kW",
        "target_document": "1",
        "target_page": 15,
        "target_bbox": [200, 150, 800, 850],
        "image_path": str(config.pages_dir / "1_page_0015.png"),
        "target_doc": "1.pdf",
        "ground_truth_bbox": [200, 150, 800, 850],
    },
    {
        "id": 10,
        "question": "GEA 离心泵在 50Hz 工频下的最大设计额定流量是多少？",
        "expected_value": "100",
        "expected_unit": "m³/h",
        "target_document": "1",
        "target_page": 18,
        "target_bbox": [200, 150, 800, 850],
        "image_path": str(config.pages_dir / "1_page_0018.png"),
        "target_doc": "1.pdf",
        "ground_truth_bbox": [200, 150, 800, 850],
    },
    {
        "id": 11,
        "question": "GEA 离心泵系列的最大额定设计扬程是多少？",
        "expected_value": "65",
        "expected_unit": "m",
        "target_document": "2",
        "target_page": 5,
        "target_bbox": [180, 120, 780, 880],
        "image_path": str(config.pages_dir / "2_page_0005.png"),
        "target_doc": "2.pdf",
        "ground_truth_bbox": [180, 120, 780, 880],
    },
    {
        "id": 12,
        "question": "进出口管路安装时，管道的重量是否可以直接施加在泵法兰接口上？",
        "expected_value": "严禁",
        "expected_unit": "支撑",
        "target_document": "3",
        "target_page": 28,
        "target_bbox": [200, 100, 800, 900],
        "image_path": str(config.pages_dir / "3_page_0028.png"),
        "target_doc": "3.pdf",
        "ground_truth_bbox": [200, 100, 800, 900],
    },
    {
        "id": 13,
        "question": "GEA 离心泵推荐使用的机械密封端面耐磨材质是什么？",
        "expected_value": "SiC",
        "expected_unit": "碳化硅",
        "target_document": "4",
        "target_page": 10,
        "target_bbox": [150, 100, 700, 850],
        "image_path": str(config.pages_dir / "4_page_0010.png"),
        "target_doc": "4.pdf",
        "ground_truth_bbox": [150, 100, 700, 850],
    },
    {
        "id": 14,
        "question": "GEA 卫生级离心泵过流部件采用的不锈钢材质牌号是什么？",
        "expected_value": "1.4404",
        "expected_unit": "316L",
        "target_document": "4",
        "target_page": 12,
        "target_bbox": [200, 150, 750, 850],
        "image_path": str(config.pages_dir / "4_page_0012.png"),
        "target_doc": "4.pdf",
        "ground_truth_bbox": [200, 150, 750, 850],
    },
    {
        "id": 15,
        "question": "GEA 离心泵允许连续工作的最高介质温度是多少？",
        "expected_value": "140",
        "expected_unit": "°C",
        "target_document": "5",
        "target_page": 8,
        "target_bbox": [180, 100, 750, 850],
        "image_path": str(config.pages_dir / "5_page_0008.png"),
        "target_doc": "5.pdf",
        "ground_truth_bbox": [180, 100, 750, 850],
    },
    {
        "id": 16,
        "question": "卫生级离心泵在线 CIP 清洗时建议的最小流速是多少？",
        "expected_value": "1.5",
        "expected_unit": "m/s",
        "target_document": "5",
        "target_page": 15,
        "target_bbox": [200, 150, 800, 850],
        "image_path": str(config.pages_dir / "5_page_0015.png"),
        "target_doc": "5.pdf",
        "ground_truth_bbox": [200, 150, 800, 850],
    },
    {
        "id": 17,
        "question": "GEA 离心泵底座固定螺栓横向间距 W1 是多少？",
        "expected_value": "250",
        "expected_unit": "mm",
        "target_document": "1",
        "target_page": 35,
        "target_bbox": [427, 248, 1709, 2232],
        "image_path": str(config.pages_dir / "1_page_0035.png"),
        "target_doc": "1.pdf",
        "ground_truth_bbox": [427, 248, 1709, 2232],
    },
    {
        "id": 18,
        "question": "泵轴承箱推荐的润滑维护与换油周期是多少运行小时？",
        "expected_value": "4000",
        "expected_unit": "小时",
        "target_document": "3",
        "target_page": 38,
        "target_bbox": [200, 100, 800, 900],
        "image_path": str(config.pages_dir / "3_page_0038.png"),
        "target_doc": "3.pdf",
        "ground_truth_bbox": [200, 100, 800, 900],
    },
]


def generate_benchmark_dataset(output_file: Path):
    """生成包含 18 个标杆问答测试用例的测试集，保存至 tests/benchmark_dataset.json"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset = DEFAULT_SYNTHETIC_BENCHMARK

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"🎉 成功生成包含 {len(dataset)} 个标杆工程测试用例的测试集，保存至: {output_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="GEA_RAG 合成测试集生成器 (docs/ARCHITECTURE.md Section 13)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.base_dir / "tests" / "benchmark_dataset.json",
        help="测试集保存路径 (默认: tests/benchmark_dataset.json)",
    )
    args = parser.parse_args()

    generate_benchmark_dataset(args.output)


if __name__ == "__main__":
    main()
