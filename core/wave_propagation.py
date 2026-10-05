"""
标量衍射波动光学传播引擎模块

基于标量衍射理论（Scalar Diffraction Theory）与菲涅尔衍射积分（Fresnel Diffraction Integral），
计算波长依赖的复振幅光场从 DOE 平面到探测器焦平面的空间传播，并记录光强分布（PSF）。

物理原理与公式解析（供几何光学背景同学理解）：
1. 波动光学基本假设：
   在标量衍射理论下，光被视为标量单色电磁波。复振幅 u(x, y) = A(x, y) * exp(i * ϕ(x, y)) 描述了空间各处的电场大小与相位。
   探测器（CMOS/CCD）无法直接响应高达 10^14 Hz 的光波相位，只能记录其时间平均光强（Poynting 矢量大小）：I(x, y) = |u(x, y)|^2。
2. 菲涅尔衍射与单次傅里叶变换（SFT）关系：
   根据 Huygens-Fresnel 原理及近轴菲涅尔近似（Fresnel Approximation），当传播距离 z 远大于波长 λ 时，
   焦平面处的复振幅 u2(x, y) 为：
   u2(x, y) = (exp(ikz) / (iλz)) * exp(i * (k/(2z)) * (x^2+y^2)) * 
              ∬ u1(x', y') * exp(i * (k/(2z)) * (x'^2+y'^2)) * exp(-i * (2π/(λz)) * (xx' + yy')) dx'dy'
   其中积分项正是 [u1(x', y') * exp(i * (π/(λz)) * (x'^2+y'^2))] 的二维连续傅里叶变换，在空间频率 fx = x/(λz), fy = y/(λz) 处的取值！
   这正是 Jeon 2019 论文第 4 节公式 (6) 的直接来源！
3. 统一探测器物理坐标映射（核心难点）：
   由于单次傅里叶变换输出的物理采样间距与波长成正比：dx_raw(λ) = (λ * f) / L_DOE，
   不同波长 λ 计算出的像面像素物理大小各不相同。
   为满足相机探测器真实的固定物理像元尺寸（如 6.22 um），必须将各个波长的 PSF 精确插值并重采样到统一的探测器物理网格上！
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import scipy.ndimage
from typing import Tuple, Dict, Any, List
from core.doe_design import (
    generate_rotating_doe_height,
    compute_doe_phase_delay
)
from core.materials import get_refractive_index


def propagate_fresnel_sft(
    u1: np.ndarray,
    wavelength: float,
    z: float,
    dx_doe: float
) -> Tuple[np.ndarray, float]:
    """
    单次傅里叶变换（SFT）菲涅尔衍射积分传播。
    
    参数:
        u1: DOE 表面紧贴处的出射复振幅光场 (N, N)
        wavelength: 入射单色光波长 λ (m)
        z: 传播距离/焦距 f (m)
        dx_doe: DOE 平面的空间采样间隔 (m)
        
    返回:
        u2: 焦平面处的复振幅光场 (N, N)
        dx_sensor_raw: 焦平面处对应的原生空间采样间隔 (m)
    """
    N = u1.shape[0]
    L_doe = N * dx_doe
    k = (2.0 * np.pi) / wavelength

    # 1. DOE 平面坐标
    coords_doe = (np.arange(N) - N // 2) * dx_doe
    X_doe, Y_doe = np.meshgrid(coords_doe, coords_doe)
    r2_doe = X_doe ** 2 + Y_doe ** 2

    # 2. 菲涅尔二次相位因子 exp(i * π / (λz) * (x'^2 + y'^2))
    # 物理意义：由球面波菲涅尔近似展开引入的传播倾斜二次相位
    fresnel_quadratic_doe = np.exp(1j * (np.pi / (wavelength * z)) * r2_doe)

    # 3. 2D FFT 计算傅里叶变换
    # 注意：使用 ifftshift/fftshift 确保零频分量严格对准中心
    integrand = u1 * fresnel_quadratic_doe
    fft_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(integrand)))

    # 4. 前置振幅与相位缩放系数 (exp(ikz) / (iλz)) * dx_doe^2
    scale_factor = (np.exp(1j * k * z) / (1j * wavelength * z)) * (dx_doe ** 2)
    u2 = fft_result * scale_factor

    # 5. 焦平面处的原生空间像元物理尺寸
    dx_sensor_raw = (wavelength * z) / L_doe

    return u2, dx_sensor_raw


def resample_psf_to_detector_grid(
    psf_raw: np.ndarray,
    dx_raw: float,
    target_pixel_pitch: float,
    target_kernel_size: int
) -> np.ndarray:
    """
    将原生波长依赖坐标系下的 PSF 重采样到相机探测器的统一固定物理像元网格上。
    
    参数:
        psf_raw: 原生网格上的 PSF 光强图像 (N, N)
        dx_raw: 原生网格的空间像素尺寸 (m)
        target_pixel_pitch: 目标探测器像元尺寸 (m)，例如 6.22 um
        target_kernel_size: 目标核大小 (像素，如 65x65)
        
    返回:
        psf_uniform: 统一探测器网格上的 PSF 矩阵 (K, K)
    """
    N = psf_raw.shape[0]
    center_raw = N // 2

    # 目标探测器的物理坐标范围
    coords_target = (np.arange(target_kernel_size) - target_kernel_size // 2) * target_pixel_pitch

    # 映射目标物理坐标到原生阵列的浮点索引
    # target_x = (idx - center_raw) * dx_raw  ==>  idx = target_x / dx_raw + center_raw
    idx_target_y = (coords_target / dx_raw) + center_raw
    idx_target_x = (coords_target / dx_raw) + center_raw
    Target_Y, Target_X = np.meshgrid(idx_target_y, idx_target_x, indexing='ij')

    # 高精度三次双样条插值（Bicubic Spline Interpolation）
    psf_uniform = scipy.ndimage.map_coordinates(
        psf_raw,
        [Target_Y, Target_X],
        order=3,
        mode='constant',
        cval=0.0,
        prefilter=True
    )
    # 保证非负物理性质
    psf_uniform = np.maximum(psf_uniform, 0.0)
    return psf_uniform


def compute_spectral_psfs(
    config
) -> Tuple[np.ndarray, np.ndarray, List[float], Dict[str, Any]]:
    """
    计算完整多光谱通道（如 25 个波长）在统一探测器坐标系下的 PSF 矩阵立方体。
    
    物理与复现关键逻辑：
    1. 生成一次固定的物理 DOE 高度剖面 Δh(x, y)。
    2. 对每个波长 λ_i：
       - 计算该波长下的复振幅相位延迟 ϕ_h(λ_i) = (2π / λ_i) * Δη(λ_i) * Δh
       - 组合孔径掩膜形成出射光场 u1 = Aperture * exp(i * ϕ_h)
       - 进行菲涅尔衍射积分传播，得到焦面复振幅 u2 并计算强度 |u2|^2
       - 严谨插值重采样至统一探测器物理网格
    3. 分别计算：
       - psf_energy_normalized: 总能量归一化（用于前向成像模型中的光线追迹与强度卷积）
       - psf_vis_normalized: 峰值归一化为 1.0（用于直观对比各波长 PSF 的几何形状与旋转角度）
    
    返回:
        psf_energy_tensor: (num_wavelengths, K, K) 能量守恒归一化 PSF
        psf_vis_tensor: (num_wavelengths, K, K) 峰值归一化 PSF（可视化展示用）
        energies: 各波长在有效核内的相对通光能量
        info: 辅助信息字典
    """
    opt = config.optical
    det = config.detector
    grid = config.grid

    # 1. 生成物理加工的固定 DOE 高度图
    delta_h, aperture_mask, doe_meta = generate_rotating_doe_height(
        grid_size=grid.doe_grid_size,
        window_size=grid.doe_window_size,
        aperture_diameter=opt.aperture_diameter,
        focal_length=opt.focal_length,
        wavelength_min=opt.wavelength_min,
        wavelength_max=opt.wavelength_max,
        num_wings=opt.num_wings,
        quantization_levels=opt.quantization_levels,
        quantization_step=opt.quantization_step,
        substrate_material=opt.substrate
    )

    dx_doe = grid.doe_window_size / grid.doe_grid_size
    aperture_area = np.pi * (opt.aperture_radius ** 2)

    psf_energy_list = []
    psf_vis_list = []
    energies = []

    for wl in opt.wavelengths:
        # 2. 计算固定高度下该波长的相位延迟
        phi_h = compute_doe_phase_delay(delta_h, wl, substrate_material=opt.substrate)
        u1 = aperture_mask * np.exp(1j * phi_h)

        # 3. 菲涅尔 SFT 衍射传播
        u2, dx_raw = propagate_fresnel_sft(u1, wl, opt.focal_length, dx_doe)
        psf_raw = np.abs(u2) ** 2

        # 4. 重采样到统一探测器物理像元网格
        psf_det = resample_psf_to_detector_grid(
            psf_raw=psf_raw,
            dx_raw=dx_raw,
            target_pixel_pitch=det.effective_pixel_pitch,
            target_kernel_size=det.psf_kernel_size
        )

        total_energy = np.sum(psf_det) * (det.effective_pixel_pitch ** 2)
        energies.append(float(total_energy))

        # 能量卷积归一化：保证核积分之和为 1.0（非相干点扩散函数通光守恒）
        sum_val = np.sum(psf_det)
        if sum_val > 1e-12:
            psf_energy_norm = psf_det / sum_val
        else:
            psf_energy_norm = psf_det

        # 峰值归一化：最大值为 1.0（用于观察旋转三叶草形态）
        max_val = np.max(psf_det)
        if max_val > 1e-12:
            psf_vis_norm = psf_det / max_val
        else:
            psf_vis_norm = psf_det

        psf_energy_list.append(psf_energy_norm)
        psf_vis_list.append(psf_vis_norm)

    psf_energy_tensor = np.array(psf_energy_list)  # (Λ, K, K)
    psf_vis_tensor = np.array(psf_vis_list)        # (Λ, K, K)

    coords_um = (np.arange(det.psf_kernel_size) - det.psf_kernel_size // 2) * (det.effective_pixel_pitch * 1e6)

    info = {
        "wavelengths_nm": [float(w * 1e9) for w in opt.wavelengths],
        "detector_pixel_pitch_um": det.effective_pixel_pitch * 1e6,
        "kernel_size": det.psf_kernel_size,
        "coords_um": coords_um,
        "delta_h": delta_h,
        "doe_meta": doe_meta
    }

    return psf_energy_tensor, psf_vis_tensor, energies, info


if __name__ == "__main__":
    from config import default_config
    print("开始测试光谱 PSF 衍射波动光学计算...")
    psf_e, psf_v, energies, info = compute_spectral_psfs(default_config)
    print(f"PSF 立方体形状: {psf_e.shape}")
    print(f"有效波段数: {len(info['wavelengths_nm'])}, 首末波段: {info['wavelengths_nm'][0]} nm ~ {info['wavelengths_nm'][-1]} nm")
    print(f"能量波动范围: {min(energies):.3e} ~ {max(energies):.3e}")
    print(f"统一物理视场: [{info['coords_um'][0]:.1f}, {info['coords_um'][-1]:.1f}] um")
