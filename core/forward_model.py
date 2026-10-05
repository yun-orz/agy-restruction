"""
非相干快照光谱成像前向模型模块 (Forward Imaging Model)

对应 Jeon 2019 论文第 4.2 节公式 (13) ~ (15)：
J_c(x, y) = ∫ Ω_c(λ) * (I_λ * p_λ)(x, y) dλ
离散矩阵模型：J = Φ * I = Ω * P * I

物理机制详解（供几何光学背景同学理解）：
1. 非相干光强叠加：
   自然光（日光、室内照明）属于空间与时间完全非相干光。不同波长、不同视场点发出的光波之间相位无恒定关系，不发生干涉。
   因此，像面总光强等于各光谱通道单独成像光强的代数相加（线性强度叠加）。
2. 空间点扩散卷积：
   根据光学线性系统理论，在等晕区内，物方点经过系统在像方形成弥散斑 p_λ(x, y)。
   物面连续光谱分布 I_λ(x, y) 在像方的响应为空间卷积：B_λ(x, y) = (I_λ * p_λ)(x, y)。
3. 传感器积分与色彩滤光：
   探测器像元表面覆盖有 Bayer 滤色片（红绿蓝通道响应函数 Ω_c(λ)）。
   各波长卷积后的模糊图像被探测器量子效率积分，最终压缩形成一幅 2D 的 RGB 快照 J_c(x, y)（从 25 维高度压缩为 3 维）。
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import scipy.signal
from typing import Tuple, Optional
from core.sensor_response import get_sensor_response


class SnapshotForwardModel:
    """
    快照高光谱成像前向算子 Φ 与伴随算子 Φ^T。
    """
    def __init__(
        self,
        psf_tensor: np.ndarray,
        sensor_response: np.ndarray,
        noise_std: float = 0.005
    ):
        """
        参数:
            psf_tensor: (Λ, K, K) 各波长在统一探测器网格上的能量归一化 PSF 卷积核
            sensor_response: (C, Λ) 探测器光谱灵敏度矩阵 (例如 3x25 或 1x25)
            noise_std: 探测噪声标准差 (默认 0.005)
        """
        self.psf_tensor = psf_tensor.astype(np.float32)
        self.sensor_response = sensor_response.astype(np.float32)
        self.noise_std = noise_std

        self.num_wl = psf_tensor.shape[0]
        self.kernel_size = psf_tensor.shape[1]
        self.num_channels = sensor_response.shape[0]

        # 预先计算伴随算子所需的翻转核 p_λ(-x, -y)
        self.psf_flipped = np.ascontiguousarray(self.psf_tensor[:, ::-1, ::-1])

        # 归一化响应矩阵，使得白平衡平衡
        norm_factor = np.sum(self.sensor_response, axis=1, keepdims=True)
        norm_factor[norm_factor == 0] = 1.0
        self.omega_normalized = self.sensor_response / norm_factor

    def forward(self, cube: np.ndarray, add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向成像仿真：J = Φ * I = Ω * (P * I)
        
        参数:
            cube: 输入清晰高光谱数据立方体 I，形状为 (H, W, Λ)
            add_noise: 是否添加探测器加性高斯噪声
            
        返回:
            measurement: 探测器捕获的 2D 压缩图像 J，形状为 (H, W, C)（C=3 为 RGB，C=1 为单色）
            blurred_cube: 经 PSF 空间卷积后、但在传感器光谱积分前的模糊中间立方体 (H, W, Λ)
        """
        H, W, Λ = cube.shape
        assert Λ == self.num_wl, f"输入数据波段数 {Λ} 与 PSF 核波段数 {self.num_wl} 不匹配"

        # 1. 空间 2D FFT 卷积：对每个光谱波段分别与对应波长的旋转 PSF 进行卷积
        # B_λ(x, y) = (I_λ * p_λ)(x, y)
        blurred_cube = np.zeros_like(cube)
        for i in range(Λ):
            blurred_cube[:, :, i] = scipy.signal.fftconvolve(
                cube[:, :, i],
                self.psf_tensor[i],
                mode='same'
            )

        # 2. 光谱积分投影：J_c(x, y) = Σ_λ Ω_{c, λ} * B_λ(x, y)
        # 矩阵乘法：(H, W, Λ) @ (Λ, C) -> (H, W, C)
        measurement = np.matmul(blurred_cube, self.omega_normalized.T)

        # 3. 添加传感器探测噪声
        if add_noise and self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=measurement.shape).astype(np.float32)
            measurement = measurement + noise

        # 截断在物理合法光强区间 [0, 1]
        measurement = np.clip(measurement, 0.0, 1.0)
        return measurement, blurred_cube

    def adjoint(self, measurement: np.ndarray) -> np.ndarray:
        """
        前向成像算子的伴随算子（转置运算）：I_adj = Φ^T * J
        
        在反问题优化（如 HQS 梯度下降 step）中极其关键：
        Φ^T = P^T * Ω^T
        1. 先将 3 通道 RGB 测量按 Ω^T 逆投影回 25 个波段空间
        2. 再将每个波段与中心对称翻转的核 p_λ(-x, -y) 进行卷积（相关运算）
        
        返回:
            cube_adj: 伴随反投影高光谱立方体 (H, W, Λ)
        """
        # 1. Ω^T 反投影：(H, W, C) @ (C, Λ) -> (H, W, Λ)
        projected = np.matmul(measurement, self.omega_normalized)

        # 2. P^T 翻转核空间卷积
        H, W, Λ = projected.shape
        cube_adj = np.zeros_like(projected)
        for i in range(Λ):
            cube_adj[:, :, i] = scipy.signal.fftconvolve(
                projected[:, :, i],
                self.psf_flipped[i],
                mode='same'
            )

        return cube_adj

    def test_adjoint_property(self, shape: Tuple[int, int] = (64, 64)) -> float:
        """
        严格检验伴随算子的数学正确性（Inner Product Test）：
        <Φ(x), y> 必须严格等于 <x, Φ^T(y)>
        这是优化算法收敛的基石。
        """
        H, W = shape
        x = np.random.rand(H, W, self.num_wl).astype(np.float32)
        y = np.random.rand(H, W, self.num_channels).astype(np.float32)

        # Φ(x) 不加噪
        Ax, _ = self.forward(x, add_noise=False)
        # Φ^T(y)
        ATy = self.adjoint(y)

        inner_1 = np.sum(Ax * y)
        inner_2 = np.sum(x * ATy)
        rel_diff = abs(inner_1 - inner_2) / (abs(inner_1) + abs(inner_2))
        return float(rel_diff)


if __name__ == "__main__":
    from config import default_config
    from core.wave_propagation import compute_spectral_psfs
    from core.test_scenes import create_colorchecker_scene

    print("测试前向成像模型与伴随算子...")
    psf_e, _, _, _ = compute_spectral_psfs(default_config)
    omega = get_sensor_response(default_config.optical.wavelengths)
    forward_model = SnapshotForwardModel(psf_e, omega)

    cube_gt, _ = create_colorchecker_scene((64, 64), default_config.optical.wavelengths)
    meas, blurred = forward_model.forward(cube_gt)

    adj_err = forward_model.test_adjoint_property((64, 64))
    print(f"测量图尺寸: {meas.shape}, 模糊立方体尺寸: {blurred.shape}")
    print(f"伴随算子内积测试相对误差 <Φx, y> vs <x, Φ^T y>: {adj_err:.2e} (应接近 0)")
