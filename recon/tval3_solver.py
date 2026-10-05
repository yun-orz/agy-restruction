"""
TVAL3 / 增广拉格朗日 TV 优化对比求解器 (TVAL3 Baseline Solver)

对应 Jeon 2019 论文第 7.2 节 Table 2 与 Figure 8 中作为对比基准的 TVAL3 (Li et al. 2009) 算法。
TVAL3 采用交替方向乘子法 (ADMM) / 增广拉格朗日乘子法求解带全变分 (TV) 正则化的压缩感知反卷积：
min_I || D*I ||_1 + (mu / 2) * || J - Φ*I ||_2^2
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from typing import Tuple, List
from core.forward_model import SnapshotForwardModel
from recon.hqs_solver import soft_threshold


def solve_tval3_reconstruction(
    measurement: np.ndarray,
    forward_model: SnapshotForwardModel,
    iterations: int = 12,
    mu: float = 2.0,
    beta: float = 4.0,
    verbose: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """
    简化版 TVAL3 / ADMM 光谱重建求解器。
    
    参数:
        measurement: 探测器捕获图像 J (H, W, C)
        forward_model: 前向模型 Φ
        iterations: 迭代轮数
        mu: 数据项拉格朗日乘子惩罚权重
        beta: 梯度分裂惩罚权重
        verbose: 是否打印日志
    """
    # 伴随算子初值
    I_0 = forward_model.adjoint(measurement)
    fwd_I0, _ = forward_model.forward(I_0, add_noise=False)
    scale = np.sum(fwd_I0 * measurement) / (np.sum(fwd_I0 ** 2) + 1e-8)
    I_curr = np.clip(I_0 * scale, 0.0, 1.0)

    # 辅助梯度变量与乘子
    wx = np.zeros_like(I_curr)
    wy = np.zeros_like(I_curr)
    bx = np.zeros_like(I_curr)
    by = np.zeros_like(I_curr)

    step_size = 1.0 / (0.10 * mu + 4.0 * beta)
    loss_history = []

    for it in range(iterations):
        # 1. 更新图像 I
        meas_pred, _ = forward_model.forward(I_curr, add_noise=False)
        meas_res = meas_pred - measurement
        loss = 0.5 * np.sum(meas_res ** 2)
        loss_history.append(float(loss))

        grad_data = mu * forward_model.adjoint(meas_res)

        # 梯度项反投影
        dx = np.roll(I_curr, -1, axis=1) - I_curr
        dy = np.roll(I_curr, -1, axis=0) - I_curr
        div_w = (dx - wx + bx) - np.roll(dx - wx + bx, 1, axis=1) + \
                (dy - wy + by) - np.roll(dy - wy + by, 1, axis=0)

        I_curr = np.clip(I_curr - step_size * (grad_data + beta * div_w), 0.0, 1.0)

        # 2. 更新分裂辅助变量 w (软阈值)
        dx = np.roll(I_curr, -1, axis=1) - I_curr
        dy = np.roll(I_curr, -1, axis=0) - I_curr
        wx = soft_threshold(dx + bx, 1.0 / beta)
        wy = soft_threshold(dy + by, 1.0 / beta)

        # 3. 更新乘子 b
        bx = bx + (dx - wx)
        by = by + (dy - wy)

        if verbose and ((it + 1) % 4 == 0 or it == iterations - 1):
            print(f"  [TVAL3 迭代 {it+1:02d}/{iterations:02d}] Loss: {loss:.4e}")

    return I_curr, loss_history
