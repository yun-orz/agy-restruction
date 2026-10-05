"""
Jeon 2019 核心物理与仿真模块包
"""
import sys
from pathlib import Path

# 将项目根目录添加至 sys.path，保证在任意位置及 PyCharm 中直接运行子脚本时模块导入畅通
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
