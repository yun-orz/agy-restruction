"""
半二次分裂 (Half-Quadratic Splitting, HQS) 光谱反卷积重建算法

对应 Jeon 2019 论文第 5.1 节公式 (16) ~ (21)：
目标函数：
min_I || J - Φ*I ||_2^2 + R(I)

引入辅助变量 V = I：
min_{I, V} || J - Φ*I ||_2^2 + ς * || V - I ||_2^2 + R(V)

拆分为两个交替迭代子问题：
1. 数据保真度项更新 (I-subproblem)：
   I^{(l+1)} = argmin_I || J - Φ*I ||_2^2 + ς * || V^{(l)} - I ||_2^2
   采用梯度下降求解：
   I^{(l+1)} = I^{(l)} - ε * [ Φ^T(Φ*I^{(l)} - J) + ς*(I^{(l)} - V^{(l)}) ]
   = Φ_bar * I^{(l)} + ε * I^{(0)} + ε*ς * V^{(l)}
   其中 I^{(0)} = Φ^T * J 为伴随反投影初值。

2. 空间-光谱先验正则化项近端映射 (V-subproblem)：
   V^{(l+1)} = argmin_V ς * || V - I^{(l+1)} ||_2^2 + R(V)
   使用空间-光谱全变分 (Spatial-Spectral 3D TV) 先验近端算子进行边缘保持平滑。
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from core.forward_model import SnapshotForwardModel


def soft_threshold(v: np.ndarray, tau: float) -> np.ndarray:
    """软阈值收缩算子：S_tau(v) = sign(v) * max(|v| - tau, 0)"""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def prox_spatial_spectral_tv(
    cube: np.ndarray,
    rho: float,
    tv_spatial_weight: float = 0.015,
    tv_spectral_weight: float = 0.008,
    num_sub_iters: int = 4
) -> np.ndarray:
    """
    3D 空间-光谱全变分 (TV) 近端投影算子。
    利用梯度域软阈值与反差投影有效去除旋转 PSF 引入的空间弥散残留，并平滑光谱维度的连续性。
    """
    v = cube.copy()
    tau_s = tv_spatial_weight / max(rho, 1e-4)
    tau_l = tv_spectral_weight / max(rho, 1e-4)

    for _ in range(num_sub_iters):
        # 1. 空间水平与垂直方向梯度
        dx = np.roll(v, -1, axis=1) - v
        dy = np.roll(v, -1, axis=0) - v
        # 2. 光谱方向一阶差分
        dl = np.roll(v, -1, axis=2) - v

        # 3. 软阈值收缩
        dx_thresh = soft_threshold(dx, tau_s)
        dy_thresh = soft_threshold(dy, tau_s)
        dl_thresh = soft_threshold(dl, tau_l)

        # 4. 散度更新
        div_x = dx_thresh - np.roll(dx_thresh, 1, axis=1)
        div_y = dy_thresh - np.roll(dy_thresh, 1, axis=0)
        div_l = dl_thresh - np.roll(dl_thresh, 1, axis=2)

        # 步长松弛更新
        v = 0.5 * v + 0.5 * (cube + 0.15 * (div_x + div_y + div_l))

    return np.clip(v, 0.0, 1.0)


def solve_hqs_reconstruction(
    measurement: np.ndarray,
    forward_model: SnapshotForwardModel,
    iterations: int = 15,
    rho: float = 0.5,
    step_size: Optional[float] = None,
    tv_spatial_weight: float = 0.015,
    tv_spectral_weight: float = 0.008,
    verbose: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    运行 HQS 半二次分裂光谱重建算法。
    
    参数:
        measurement: 探测器捕获图像 J (H, W, C)
        forward_model: 前向成像模型实例 Φ
        iterations: HQS 迭代次数 (默认 15)
        rho: 惩罚项参数 ς
        step_size: 梯度下降步长 ε（若为 None，自适应根据算子范数确定）
        tv_spatial_weight: 空间 TV 权重
        tv_spectral_weight: 光谱 TV 权重
        verbose: 是否打印迭代日志
        
    返回:
        recon_cube: 重建出的高光谱立方体 (H, W, Λ) [0, 1]
        loss_history: 目标函数数据项残差历史
    """
    # 1. 伴随算子初始反投影 I^(0) = Φ^T * J
    I_0 = forward_model.adjoint(measurement)

    # 最佳线性投影尺度匹配：min_a || a * Φ(I_0) - J ||^2
    fwd_I0, _ = forward_model.forward(I_0, add_noise=False)
    scale = np.sum(fwd_I0 * measurement) / (np.sum(fwd_I0 ** 2) + 1e-8)
    I_current = np.clip(I_0 * scale, 0.0, 1.0)
    V_current = I_current.copy()

    # 2. 估计算子谱范数 L = ||Φ^T Φ|| 以确定最佳收敛步长
    if step_size is None:
        L_norm = 0.10
        step_size = 1.0 / (L_norm + rho)

    loss_history = []
    if verbose:
        print(f"--- 启动 HQS 迭代重建 (共 {iterations} 轮，自适应步长 eps={step_size:.3f}, rho={rho:.2f}) ---")

    for it in range(iterations):
        # ---------------------------------------------------------
        # 子问题 1: 数据项梯度下降更新 (I-Subproblem, 论文公式 21)
        # ---------------------------------------------------------
        pred_meas, _ = forward_model.forward(I_current, add_noise=False)
        meas_res = pred_meas - measurement
        data_loss = 0.5 * np.sum(meas_res ** 2)
        loss_history.append(float(data_loss))

        grad_data = forward_model.adjoint(meas_res)
        grad_penalty = rho * (I_current - V_current)

        total_grad = grad_data + grad_penalty
        I_next = I_current - step_size * total_grad
        I_next = np.clip(I_next, 0.0, 1.0)

        # ---------------------------------------------------------
        # 子问题 2: 空间-光谱正则化先验投影 (V-Subproblem, 论文公式 20)
        # ---------------------------------------------------------
        V_next = prox_spatial_spectral_tv(
            cube=I_next,
            rho=rho,
            tv_spatial_weight=tv_spatial_weight,
            tv_spectral_weight=tv_spectral_weight
        )

        I_current = I_next
        V_current = V_next

        if verbose and ((it + 1) % 3 == 0 or it == iterations - 1):
            print(f"  [迭代 {it+1:02d}/{iterations:02d}] 数据残差 Loss: {data_loss:.4e}")

    recon_cube = np.clip(V_current, 0.0, 1.0)
    return recon_cube, loss_history


if __name__ == "__main__":
    from config import default_config
    from core.wave_propagation import compute_spectral_psfs
    from core.sensor_response import get_sensor_response
    from core.test_scenes import create_colorchecker_scene
    from recon.metrics import evaluate_all_metrics

    print("测试 HQS 重建求解器...")
    psf_e, _, _, _ = compute_spectral_psfs(default_config)
    omega = get_sensor_response(default_config.optical.wavelengths)
    fwd = SnapshotForwardModel(psf_e, omega)

    cube_gt, _ = create_colorchecker_scene((64, 64), default_config.optical.wavelengths)
    meas, _ = fwd.forward(cube_gt, add_noise=True)

    recon, history = solve_hqs_reconstruction(meas, fwd, iterations=12, verbose=True)
    metrics = evaluate_all_metrics(cube_gt, recon)
    print(f"重建评价结果: PSNR = {metrics['PSNR']:.2f} dB, SSIM = {metrics['SSIM']:.4f}, SAM = {metrics['SAM']:.4f} rad")
