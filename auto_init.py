#!/usr/bin/env python3
"""
auto_init.py
遍历当前目录树，为每个缺少 __init__.py 的文件夹新建一个空文件。
用法：
    python auto_init.py          # 默认从脚本所在目录开始
    python auto_init.py /path    # 从指定根目录开始
"""

import os
import sys
from pathlib import Path

def ensure_init_py(root: Path) -> None:
    """递归补建 __init__.py"""
    for dirpath, dirnames, filenames in os.walk(root):
        folder = Path(dirpath)
        init_file = folder / "__init__.py"

        # 跳过已存在 __init__.py 的目录
        if init_file.exists():
            continue

        # 跳过 Python 缓存目录
        if folder.name in {"__pycache__", ".git", ".venv", "venv", ".nox", ".tox"}:
            continue

        # 创建空文件
        init_file.touch()
        print(f"created: {init_file}")

def main() -> None:
    # 支持命令行传入根目录，默认用脚本所在目录
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    if not root.is_dir():
        sys.exit(f"Error: {root} 不是有效目录")
    ensure_init_py(root)

if __name__ == "__main__":
    main()
