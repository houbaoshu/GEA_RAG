#!/usr/bin/env python3
import sys
from pathlib import Path

# 启动入口脚本别名
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ui.app_ui import main

if __name__ == "__main__":
    main()
