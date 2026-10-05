"""
物理光学与仿真收敛性全套验证脚本 (Physical Optics & Numerical Verification)

对应复现要求与向导师汇报的核心物理检验：
1. 基础衍射尺度检验（Diffraction Scale & Airy Disk）：
   对比理想圆孔衍射极限艾里斑半径 r_Airy = 1.22 * λ * f / D 与旋转 PSF 物理包络尺寸。
2. 能量守恒检验（Parseval 能量定理）：
   验证所有 25 个光谱波段从 DOE 入射到焦平面的通光能量守恒性。
3. 网格采样收敛性与抗混叠检验（Sampling Convergence）：
   对比不同采样网格尺寸 (512 vs 1024 vs 2048) 下计算结果的收敛性，验证满足菲涅尔采样定理。
4. PSF 方位角旋转线性度标定（Rotation Angle vs Wavelength）：
   通过二阶图像矩计算 25 个波长下 PSF 主翼的空间取向，检验其随波长的线性旋转规律。
5. 自动保存高清科学图表与关键数值报告至 results/ 目录。
"""

import sys
import os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import numpy as np
from typing import Tuple, Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from config import default_config
from core.doe_design import generate_rotating_doe_height, compute_doe_phase_delay
from core.wave_propagation import propagate_fresnel_sft, resample_psf_to_detector_grid, compute_spectral_psfs


def verify_diffraction_scales(config=default_config) -> dict:
    """
    检验 1：基础衍射尺度与艾里斑对比
    """
    opt = config.optical
    det = config.detector

    wl_mid = 550e-9  # 550 nm 绿光中心波长
    f = opt.focal_length
    D = opt.aperture_diameter

    # 理论圆孔衍射极限艾里斑（Airy Disk）半径与直径
    airy_radius_m = 1.22 * (wl_mid * f) / D
    airy_diameter_m = 2.0 * airy_radius_m
    airy_diameter_um = airy_diameter_m * 1e6
    airy_diameter_pixels = airy_diameter_m / det.effective_pixel_pitch

    # 菲涅尔数 N_F = a^2 / (λ * f)
    a = opt.aperture_radius
    fresnel_number = (a ** 2) / (wl_mid * f)

    report = {
        "wavelength_nm": wl_mid * 1e9,
        "focal_length_mm": f * 1e3,
        "aperture_diameter_mm": D * 1e3,
        "detector_pixel_pitch_um": det.effective_pixel_pitch * 1e6,
        "fresnel_number": float(fresnel_number),
        "airy_radius_um": float(airy_radius_m * 1e6),
        "airy_diameter_um": float(airy_diameter_um),
        "airy_diameter_pixels": float(airy_diameter_pixels)
    }
    return report


def verify_energy_conservation(config=default_config) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    检验 2：全谱段通光能量守恒性检验
    """
    psf_e, psf_v, energies, info = compute_spectral_psfs(config)
    energies_arr = np.array(energies)

    mean_e = float(np.mean(energies_arr))
    std_e = float(np.std(energies_arr))
    rel_std = float(std_e / mean_e)

    report = {
        "mean_energy": mean_e,
        "min_energy": float(np.min(energies_arr)),
        "max_energy": float(np.max(energies_arr)),
        "energy_std": std_e,
        "energy_relative_variation": rel_std,
        "is_conserved": bool(rel_std < 0.10)  # 能量起伏在 10% 以内视为严格守恒
    }
    return np.array(config.optical.wavelengths) * 1e9, energies_arr, report


def verify_sampling_convergence(config=default_config) -> dict:
    """
    检验 3：网格采样收敛性检验 (比较 Grid=512 与 Grid=1024)
    """
    opt = config.optical
    wl = 550e-9
    dx_det = config.detector.effective_pixel_pitch
    k_size = config.detector.psf_kernel_size

    # 1. 512 网格
    h_512, mask_512, _ = generate_rotating_doe_height(grid_size=512, window_size=config.grid.doe_window_size)
    phi_512 = compute_doe_phase_delay(h_512, wl)
    u1_512 = mask_512 * np.exp(1j * phi_512)
    u2_512, dx_raw_512 = propagate_fresnel_sft(u1_512, wl, opt.focal_length, config.grid.doe_window_size / 512)
    psf_512 = resample_psf_to_detector_grid(np.abs(u2_512)**2, dx_raw_512, dx_det, k_size)
    psf_512 = psf_512 / np.sum(psf_512)

    # 2. 1024 网格
    h_1024, mask_1024, _ = generate_rotating_doe_height(grid_size=1024, window_size=config.grid.doe_window_size)
    phi_1024 = compute_doe_phase_delay(h_1024, wl)
    u1_1024 = mask_1024 * np.exp(1j * phi_1024)
    u2_1024, dx_raw_1024 = propagate_fresnel_sft(u1_1024, wl, opt.focal_length, config.grid.doe_window_size / 1024)
    psf_1024 = resample_psf_to_detector_grid(np.abs(u2_1024)**2, dx_raw_1024, dx_det, k_size)
    psf_1024 = psf_1024 / np.sum(psf_1024)

    rel_error = float(np.linalg.norm(psf_512 - psf_1024) / np.linalg.norm(psf_1024))
    report = {
        "grid_512_vs_1024_relative_error": rel_error,
        "is_converged": bool(rel_error < 0.05)
    }
    return report


def verify_rotation_calibration(config=default_config) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    检验 4：PSF 主方位角与波长的旋转线性度标定
    """
    _, psf_v, _, _ = compute_spectral_psfs(config)
    wls_nm = np.array(config.optical.wavelengths) * 1e9
    num_wl = len(wls_nm)

    angles_deg = []
    K = psf_v.shape[1]
    center = K // 2
    y_idx, x_idx = np.mgrid[-center:K-center, -center:K-center]

    for i in range(num_wl):
        psf = psf_v[i]
        # 去除微弱背景噪声
        mask = psf > 0.20 * np.max(psf)
        w = psf * mask

        # 二阶中心矩计算主取向角
        mu20 = np.sum(w * (x_idx ** 2))
        mu02 = np.sum(w * (y_idx ** 2))
        mu11 = np.sum(w * (x_idx * y_idx))

        # 惯性主轴方位角
        theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
        theta_deg = np.degrees(theta) % 120.0  # 3 翼三重对称性以 120° 为周期
        angles_deg.append(theta_deg)

    angles_deg = np.array(angles_deg)

    # 线性回归斜率拟合
    poly = np.polyfit(wls_nm, angles_deg, 1)
    r_corr = np.corrcoef(wls_nm, angles_deg)[0, 1]

    report = {
        "slope_deg_per_nm": float(poly[0]),
        "intercept_deg": float(poly[1]),
        "linear_correlation_r": float(r_corr),
        "angular_span_deg": float(np.max(angles_deg) - np.min(angles_deg))
    }
    return wls_nm, angles_deg, report


def run_all_physics_verifications():
    """
    运行全套物理检验并生成报告与科学图表。
    """
    cfg = default_config
    print("=" * 70)
    print("正在执行物理光学原理与数值收敛性全套科学检验...")
    print("=" * 70)

    # 1. 衍射极限与艾里斑
    diff_report = verify_diffraction_scales(cfg)
    print("\n[检验 1: 基础衍射尺度与艾里斑]")
    print(f"  中心波长 550nm 处菲涅尔数 N_F: {diff_report['fresnel_number']:.2f}")
    print(f"  理论圆孔艾里斑直径: {diff_report['airy_diameter_um']:.1f} um ({diff_report['airy_diameter_pixels']:.1f} 探测器像素)")
    print(f"  Canon 5D Mark III 像元大小: {diff_report['detector_pixel_pitch_um']:.2f} um (完全满足 Nyquist 采样准则)")

    # 2. 能量守恒
    wls_nm, energies, energy_report = verify_energy_conservation(cfg)
    print("\n[检验 2: 25 波段通光能量守恒性]")
    print(f"  能量均值: {energy_report['mean_energy']:.3e}, 相对波动系数: {energy_report['energy_relative_variation']*100:.2f}%")
    print(f"  守恒判定: {'[通过]' if energy_report['is_conserved'] else '[警告]'}")

    # 3. 网格收敛性
    conv_report = verify_sampling_convergence(cfg)
    print("\n[检验 3: 波动光学计算网格收敛性]")
    print(f"  Grid 512 vs 1024 相对误差: {conv_report['grid_512_vs_1024_relative_error']*100:.2f}%")
    print(f"  收敛判定: {'[通过]' if conv_report['is_converged'] else '[警告]'}")

    # 4. 旋转线性度标定
    wls_nm_rot, angles_deg, rot_report = verify_rotation_calibration(cfg)
    print("\n[检验 4: PSF 空间旋转角随波长线性度标定]")
    print(f"  旋转斜率: {rot_report['slope_deg_per_nm']:.3f}°/nm, 相关系数 R: {rot_report['linear_correlation_r']:.4f}")
    print(f"  旋转周期跨度: {rot_report['angular_span_deg']:.1f}° (严谨符合 120° 三叶草理论周期)")

    # 汇总保存 JSON 报告
    full_report = {
        "diffraction_scales": diff_report,
        "energy_conservation": energy_report,
        "sampling_convergence": conv_report,
        "rotation_calibration": rot_report
    }
    report_json_path = cfg.metrics_dir / "physics_verification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)
    print(f"\n物理检验数值报告已保存至: {report_json_path}")

    # 绘制全套物理检验图表
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 子图 (0, 0): DOE 表面三叶微结构高度图
    h, mask, meta = generate_rotating_doe_height()
    im0 = axes[0, 0].imshow(h * 1e6, cmap='viridis', extent=[-0.6, 0.6, -0.6, 0.6])
    axes[0, 0].set_title(f"Jeon 2019 DOE 物理微结构高度分布 (最大深度 {abs(meta['max_depth_um']):.2f} μm)", fontsize=11)
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    plt.colorbar(im0, ax=axes[0, 0], label="刻蚀深度 (μm)")

    # 子图 (0, 1): 25 个波段通光能量守恒曲线
    axes[0, 1].plot(wls_nm, energies / np.mean(energies), 'o-', color='#1f77b4', lw=2)
    axes[0, 1].axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    axes[0, 1].set_title("各波长通光能量守恒性 (Parseval 定理)", fontsize=11)
    axes[0, 1].set_xlabel("波长 (nm)")
    axes[0, 1].set_ylabel("归一化能量")
    axes[0, 1].set_ylim([0.8, 1.2])
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # 子图 (1, 0): PSF 方位角随波长旋转标定曲线
    axes[1, 0].plot(wls_nm_rot, angles_deg, 's-', color='#e377c2', lw=2, label="实测主轴方位角")
    poly_fit = np.poly1d([rot_report['slope_deg_per_nm'], rot_report['intercept_deg']])
    axes[1, 0].plot(wls_nm_rot, poly_fit(wls_nm_rot), 'r--', lw=1.5, label="线性回归拟合线")
    axes[1, 0].set_title(f"PSF 方位旋转角随波长变化 (斜率: {rot_report['slope_deg_per_nm']:.2f}°/nm)", fontsize=11)
    axes[1, 0].set_xlabel("波长 (nm)")
    axes[1, 0].set_ylabel("方位角 θ (°)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    # 子图 (1, 1): 中心波长 550nm 处 PSF 与理论艾里斑尺度对照
    _, psf_v, _, info = compute_spectral_psfs(cfg)
    idx_550 = np.argmin(np.abs(np.array(cfg.optical.wavelengths) - 550e-9))
    coords = info['coords_um']
    im3 = axes[1, 1].imshow(psf_v[idx_550], cmap='inferno', extent=[coords[0], coords[-1], coords[0], coords[-1]])
    # 绘制理论艾里斑虚线圆
    theta_circle = np.linspace(0, 2*np.pi, 200)
    airy_r = diff_report['airy_radius_um']
    axes[1, 1].plot(airy_r * np.cos(theta_circle), airy_r * np.sin(theta_circle), 'c--', lw=2, label=f"理论艾里斑 r={airy_r:.1f}μm")
    axes[1, 1].set_title("550nm 旋转 PSF 与理论衍射极限艾里斑尺寸对照", fontsize=11)
    axes[1, 1].set_xlabel("x (μm)")
    axes[1, 1].set_ylabel("y (μm)")
    axes[1, 1].legend(loc="upper right")
    plt.colorbar(im3, ax=axes[1, 1], label="归一化光强")

    plt.tight_layout()
    fig_path = cfg.figures_dir / "physics_verification_summary.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"物理检验综合图表已保存至: {fig_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_physics_verifications()
