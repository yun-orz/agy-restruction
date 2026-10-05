"""
Jeon et al. 2019 《Compact Snapshot Hyperspectral Imaging with Diffracted Rotation》
主程序入口 (Main Pipeline)

本脚本为复现项目的主运行入口，可在 Windows 电脑、PyCharm 中直接点击“运行 (Run)”启动。
全流程涵盖：
1. 光学系统初始化与 DOE 微结构设计（Jeon 2019 N=3 翼三叶草单螺旋高度图）
2. 波动光学标量衍射传播（菲涅尔 SFT 积分，全谱 25 波段，统一探测器物理像元映射）
3. 传统菲涅尔透镜 vs 旋转 DOE 对比（复现论文 Figure 3 与 Figure 11）
4. 非相干快照成像前向模型仿真（25 波段 PSF 空间卷积 + Canon 5D Mark III 传感器响应积分 + 噪声）
5. 半二次分裂 (HQS) 逆问题光谱反卷积重建
6. 科学指标评价（PSNR, SSIM, SAM, MTF）与色块光谱曲线对比（复现论文 Figure 9 & Figure 10）
7. 自动生成并保存全套科研级高清图表与量化指标报告至 results/ 目录。
"""

import sys
import os
import json
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 控制台编码自适应（防止 Windows GBK 控制台乱码）
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from config import default_config, ProjectConfig
from core.doe_design import (
    generate_rotating_doe_height,
    generate_conventional_fresnel_height,
    compute_doe_phase_delay
)
from core.wave_propagation import (
    compute_spectral_psfs,
    propagate_fresnel_sft,
    resample_psf_to_detector_grid
)
from core.sensor_response import get_sensor_response
from core.test_scenes import create_colorchecker_scene, get_macbeth_24_reflectances
from core.forward_model import SnapshotForwardModel
from recon.hqs_solver import solve_hqs_reconstruction
from recon.metrics import evaluate_all_metrics, compute_mtf_edge


def plot_fig1_doe_and_rotating_psfs(
    delta_h: np.ndarray,
    psf_vis_tensor: np.ndarray,
    wavelengths: np.ndarray,
    coords_um: np.ndarray,
    output_path: Path
):
    """
    绘制图 1：DOE 表面高度剖面与多波长旋转 PSF 序列（复现 Figure 2 & Figure 3）
    """
    wls_nm = [int(w * 1e9) for w in wavelengths]
    # 选取 5 个代表性波长（从紫蓝到深红）
    selected_indices = [0, 6, 12, 18, 24]  # 420, 480, 540, 600, 660 nm

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.2, 1, 1, 1, 1, 1])

    # 1. DOE 高度分布
    ax_doe = fig.add_subplot(gs[0])
    im_doe = ax_doe.imshow(delta_h * 1e6, cmap='viridis', extent=[-0.6, 0.6, -0.6, 0.6])
    ax_doe.set_title("DOE 高度分布 h(x,y)\n(N=3 翼单螺旋)", fontsize=11, fontweight='bold')
    ax_doe.set_xlabel("x (mm)")
    ax_doe.set_ylabel("y (mm)")
    cbar = plt.colorbar(im_doe, ax=ax_doe, fraction=0.046, pad=0.04)
    cbar.set_label("深度 (μm)", fontsize=9)

    # 2. 5 个代表性波长的旋转 PSF
    extent = [coords_um[0], coords_um[-1], coords_um[0], coords_um[-1]]
    for idx, wl_idx in enumerate(selected_indices):
        ax = fig.add_subplot(gs[idx + 1])
        psf = psf_vis_tensor[wl_idx]
        im = ax.imshow(psf, cmap='inferno', origin='lower', extent=extent)
        ax.set_title(f"λ = {wls_nm[wl_idx]} nm\n(旋转角 θ_{idx+1})", fontsize=11, fontweight='bold')
        ax.set_xlabel("x (μm)")
        if idx == 0:
            ax.set_ylabel("y (μm)")
        else:
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  [图表保存] Figure 1: DOE 与波长旋转 PSF 序列已保存至 -> {output_path.name}")


def plot_fig2_fresnel_vs_rotating_psf(
    config: ProjectConfig,
    psf_rotating_vis: np.ndarray,
    output_path: Path
):
    """
    绘制图 2：传统菲涅尔透镜（各向同性色差弥散）vs Jeon 2019 旋转 DOE 对比（复现 Figure 3 & 11）
    """
    opt = config.optical
    det = config.detector
    grid = config.grid

    # 生成传统菲涅尔透镜（设计波长 550 nm）
    h_fresnel, mask_fresnel, _ = generate_conventional_fresnel_height(
        grid_size=grid.doe_grid_size,
        window_size=grid.doe_window_size,
        aperture_diameter=opt.aperture_diameter,
        focal_length=opt.focal_length,
        design_wavelength=550e-9
    )

    dx_doe = grid.doe_window_size / grid.doe_grid_size
    test_wls = [450e-9, 500e-9, 550e-9, 600e-9, 650e-9]

    psf_fresnel_list = []
    for wl in test_wls:
        phi = compute_doe_phase_delay(h_fresnel, wl)
        u1 = mask_fresnel * np.exp(1j * phi)
        u2, dx_raw = propagate_fresnel_sft(u1, wl, opt.focal_length, dx_doe)
        psf_det = resample_psf_to_detector_grid(np.abs(u2)**2, dx_raw, det.effective_pixel_pitch, det.psf_kernel_size)
        psf_norm = psf_det / (np.max(psf_det) + 1e-12)
        psf_fresnel_list.append(psf_norm)

    # 对应旋转 PSF 索引
    rot_indices = [
        np.argmin(np.abs(opt.wavelengths - wl)) for wl in test_wls
    ]

    coords_um = (np.arange(det.psf_kernel_size) - det.psf_kernel_size // 2) * (det.effective_pixel_pitch * 1e6)
    extent = [coords_um[0], coords_um[-1], coords_um[0], coords_um[-1]]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for col, (wl, rot_idx) in enumerate(zip(test_wls, rot_indices)):
        wl_nm = int(wl * 1e9)
        # 第一行：传统菲涅尔透镜
        axes[0, col].imshow(psf_fresnel_list[col], cmap='hot', origin='lower', extent=extent)
        axes[0, col].set_title(f"传统菲涅尔 ({wl_nm} nm)", fontsize=10)
        axes[0, col].set_xlabel("x (μm)")
        if col == 0:
            axes[0, col].set_ylabel("传统菲涅尔\n(弥散斑大小剧变)", fontsize=11, fontweight='bold')

        # 第二行：Jeon 2019 旋转 DOE
        axes[1, col].imshow(psf_rotating_vis[rot_idx], cmap='hot', origin='lower', extent=extent)
        axes[1, col].set_title(f"本设计 DOE ({wl_nm} nm)", fontsize=10)
        axes[1, col].set_xlabel("x (μm)")
        if col == 0:
            axes[1, col].set_ylabel("Jeon 2019 旋转 DOE\n(大小恒定，角度旋转)", fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  [图表保存] Figure 2: 传统菲涅尔 vs 旋转 DOE 对比已保存至 -> {output_path.name}")


def plot_fig3_imaging_and_reconstruction(
    rgb_gt: np.ndarray,
    measurement_rgb: np.ndarray,
    rgb_recon: np.ndarray,
    metrics: dict,
    output_path: Path
):
    """
    绘制图 3：成像与重建总览（真实场景、相机模糊快照、重建结果及误差图）
    """
    error_map = np.abs(rgb_gt - rgb_recon)
    error_rgb = np.mean(error_map, axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. 真实场景 sRGB
    axes[0].imshow(rgb_gt)
    axes[0].set_title("Ground Truth\n(真实高光谱 sRGB 映射)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # 2. 探测器快照输入
    axes[1].imshow(measurement_rgb)
    axes[1].set_title("Sensor Snapshot J\n(旋转 PSF 编码模糊快照)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # 3. HQS 重建结果
    axes[2].imshow(rgb_recon)
    axes[2].set_title(f"HQS 重建结果\nPSNR: {metrics['PSNR']:.2f}dB, SSIM: {metrics['SSIM']:.3f}", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # 4. 重建绝对误差热力图
    im_err = axes[3].imshow(error_rgb, cmap='magma', vmin=0.0, vmax=0.25)
    axes[3].set_title(f"重建绝对误差图\nSAM: {metrics['SAM']:.3f} rad", fontsize=11, fontweight='bold')
    axes[3].axis('off')
    cbar = plt.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("绝对误差", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  [图表保存] Figure 3: 成像与重建对比图已保存至 -> {output_path.name}")


def plot_fig4_spectral_channels_slices(
    cube_gt: np.ndarray,
    cube_recon: np.ndarray,
    wavelengths: np.ndarray,
    output_path: Path
):
    """
    绘制图 4：单波段二维切片精细对比（复现 Figure 9 & Figure 10）
    """
    wls_nm = [int(w * 1e9) for w in wavelengths]
    selected_indices = [3, 9, 15, 21]  # 450nm, 510nm, 570nm, 630nm

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))

    for col, idx in enumerate(selected_indices):
        wl_val = wls_nm[idx]
        # 第一行：Ground Truth
        axes[0, col].imshow(cube_gt[:, :, idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f"真实波段: {wl_val} nm", fontsize=11)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].text(-0.25, 0.5, "Ground Truth", transform=axes[0, col].transAxes,
                              rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')

        # 第二行：Reconstruction
        axes[1, col].imshow(cube_recon[:, :, idx], cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f"重建波段: {wl_val} nm", fontsize=11)
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].text(-0.25, 0.5, "Reconstruction", transform=axes[1, col].transAxes,
                              rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  [图表保存] Figure 4: 单波段切片对比图已保存至 -> {output_path.name}")


def plot_fig5_spectral_curves_comparison(
    cube_gt: np.ndarray,
    cube_recon: np.ndarray,
    wavelengths: np.ndarray,
    output_path: Path
):
    """
    绘制图 5：经典色块实测光谱曲线 vs 重建光谱曲线对比（复现 Figure 9 & Figure 10(a)）
    """
    wls_nm = wavelengths * 1e9
    H, W, _ = cube_gt.shape

    # 4 行 6 列排布定位色块中心像素
    rows, cols = 4, 6
    margin_y = int(H * 0.08)
    margin_x = int(W * 0.08)
    patch_h = (H - 2 * margin_y) // rows
    patch_w = (W - 2 * margin_x) // cols

    # 取 4 个代表性色块：
    # 纯红 (Row 2, Col 2), 纯绿 (Row 2, Col 1), 纯蓝 (Row 2, Col 0), 灰度 (Row 3, Col 2)
    sample_patches = [
        ("红色彩块 (Red Patch)", 2, 2, '#d62728'),
        ("绿色彩块 (Green Patch)", 2, 1, '#2ca02c'),
        ("蓝色彩块 (Blue Patch)", 2, 0, '#1f77b4'),
        ("中灰彩块 (Neutral Gray)", 3, 2, '#7f7f7f'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for idx, (title, r, c, color) in enumerate(sample_patches):
        ax = axes[idx // 2, idx % 2]
        cy = margin_y + r * patch_h + patch_h // 2
        cx = margin_x + c * patch_w + patch_w // 2

        spec_gt = cube_gt[cy, cx, :]
        spec_recon = cube_recon[cy, cx, :]

        ax.plot(wls_nm, spec_gt, 'o-', color=color, lw=2.2, label="Ground Truth (真实值)")
        ax.plot(wls_nm, spec_recon, 's--', color='black', lw=1.8, alpha=0.85, label="Reconstructed (本方法重建)")

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("波长 (nm)", fontsize=10)
        ax.set_ylabel("光谱反射率 / 归一化强度", fontsize=10)
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  [图表保存] Figure 5: 色块光谱反射率曲线对比已保存至 -> {output_path.name}")


def main():
    """
    主程序执行逻辑
    """
    t_start = time.time()
    cfg = default_config

    print("=" * 75)
    print("  Jeon et al. 2019 《Compact Snapshot Hyperspectral Imaging with Diffracted Rotation》")
    print("  全流程复现仿真与光谱重建系统")
    print("=" * 75)
    print(f"[*] 项目根目录: {cfg.project_root}")
    print(f"[*] 输出结果目录: {cfg.results_dir}")
    print(f"[*] 系统参数: 焦距 f = {cfg.optical.focal_length*1e3:.1f} mm, 通光孔径 D = {cfg.optical.aperture_diameter*1e3:.1f} mm")
    print(f"[*] 光谱通道: {cfg.optical.num_wavelengths} 波段 ({cfg.optical.wavelength_min*1e9:.0f} ~ {cfg.optical.wavelength_max*1e9:.0f} nm)")
    print(f"[*] 探测器像元尺寸: {cfg.detector.effective_pixel_pitch*1e6:.2f} μm, 类型: {cfg.detector.sensor_type.upper()}")

    # -------------------------------------------------------------
    # 步骤 1: 生成旋转 DOE 微结构并计算全光谱 PSF
    # -------------------------------------------------------------
    print("\n[阶段 1/5] 正在进行波动光学标量衍射计算 (菲涅尔 SFT 积分与统一像元映射)...")
    psf_energy, psf_vis, energies, info = compute_spectral_psfs(cfg)
    delta_h = info['delta_h']
    coords_um = info['coords_um']
    print(f"  -> PSF 计算完成，形状: {psf_energy.shape}，统一探测器覆盖视场: [{coords_um[0]:.1f}, {coords_um[-1]:.1f}] μm")

    # 保存图 1: DOE 与多波长旋转 PSF 序列
    plot_fig1_doe_and_rotating_psfs(
        delta_h=delta_h,
        psf_vis_tensor=psf_vis,
        wavelengths=cfg.optical.wavelengths,
        coords_um=coords_um,
        output_path=cfg.figures_dir / "fig1_doe_and_rotating_psfs.png"
    )

    # 保存图 2: 传统菲涅尔透镜 vs 旋转 DOE 对比
    plot_fig2_fresnel_vs_rotating_psf(
        config=cfg,
        psf_rotating_vis=psf_vis,
        output_path=cfg.figures_dir / "fig2_fresnel_vs_rotating_psf.png"
    )

    # -------------------------------------------------------------
    # 步骤 2: 构建测试场景与探测器响应
    # -------------------------------------------------------------
    print("\n[阶段 2/5] 正在构建 Macbeth 24 色卡高光谱场景与 Canon 5D Mark III 传感器响应...")
    cube_gt, rgb_gt = create_colorchecker_scene(
        image_size=cfg.recon.test_image_size,
        wavelengths=cfg.optical.wavelengths
    )
    sensor_response = get_sensor_response(
        wavelengths=cfg.optical.wavelengths,
        sensor_type=cfg.detector.sensor_type
    )
    print(f"  -> 测试场景数据立方体尺寸: {cube_gt.shape}")
    print(f"  -> 传感器响应矩阵尺寸: {sensor_response.shape}")

    # -------------------------------------------------------------
    # 步骤 3: 运行快照前向成像仿真 (J = Φ * I + 噪声)
    # -------------------------------------------------------------
    print("\n[阶段 3/5] 正在执行非相干前向成像仿真 (强度叠加卷积 + 滤色片积分 + 噪声)...")
    fwd_model = SnapshotForwardModel(
        psf_tensor=psf_energy,
        sensor_response=sensor_response,
        noise_std=cfg.detector.noise_std
    )
    measurement, blurred_cube = fwd_model.forward(cube_gt, add_noise=True)
    print(f"  -> 探测器拍摄的 2D 压缩测量图 J 形状: {measurement.shape}，噪声标准差: {cfg.detector.noise_std}")

    # -------------------------------------------------------------
    # 步骤 4: 运行半二次分裂 (HQS) 光谱反卷积重建
    # -------------------------------------------------------------
    print(f"\n[阶段 4/5] 正在执行 HQS 光谱反卷积重建算法 (共 {cfg.recon.hqs_iterations} 轮迭代)...")
    t0_recon = time.time()
    cube_recon, loss_history = solve_hqs_reconstruction(
        measurement=measurement,
        forward_model=fwd_model,
        iterations=cfg.recon.hqs_iterations,
        rho=cfg.recon.hqs_rho,
        tv_spatial_weight=cfg.recon.tv_spatial_weight,
        tv_spectral_weight=cfg.recon.tv_spectral_weight,
        verbose=True
    )
    t_recon = time.time() - t0_recon
    print(f"  -> HQS 重建完成，耗时: {t_recon:.2f} 秒")

    # -------------------------------------------------------------
    # 步骤 5: 科学量化评价与结果呈现
    # -------------------------------------------------------------
    print("\n[阶段 5/5] 正在计算科学评价指标 (PSNR, SSIM, SAM) 并绘制科研图表...")
    metrics = evaluate_all_metrics(cube_gt, cube_recon)
    print(f"  [最终评价] PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  [最终评价] SSIM = {metrics['SSIM']:.4f}")
    print(f"  [最终评价] SAM  = {metrics['SAM']:.4f} rad ({np.degrees(metrics['SAM']):.2f}°)")

    # 生成重建 sRGB 图像
    idx_r = np.argmin(np.abs(cfg.optical.wavelengths - 610e-9))
    idx_g = np.argmin(np.abs(cfg.optical.wavelengths - 530e-9))
    idx_b = np.argmin(np.abs(cfg.optical.wavelengths - 460e-9))
    rgb_recon = np.stack([cube_recon[:, :, idx_r], cube_recon[:, :, idx_g], cube_recon[:, :, idx_b]], axis=-1)
    rgb_recon = np.clip(rgb_recon, 0.0, 1.0)

    # 绘制图 3: 成像与重建总览
    plot_fig3_imaging_and_reconstruction(
        rgb_gt=rgb_gt,
        measurement_rgb=measurement,
        rgb_recon=rgb_recon,
        metrics=metrics,
        output_path=cfg.figures_dir / "fig3_imaging_and_reconstruction.png"
    )

    # 绘制图 4: 单波段二维切片精细对比
    plot_fig4_spectral_channels_slices(
        cube_gt=cube_gt,
        cube_recon=cube_recon,
        wavelengths=cfg.optical.wavelengths,
        output_path=cfg.figures_dir / "fig4_spectral_channels_slices.png"
    )

    # 绘制图 5: 色块光谱曲线对比
    plot_fig5_spectral_curves_comparison(
        cube_gt=cube_gt,
        cube_recon=cube_recon,
        wavelengths=cfg.optical.wavelengths,
        output_path=cfg.figures_dir / "fig5_spectral_curves_comparison.png"
    )

    # 保存量化报告至 JSON 与 TXT
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "optical_parameters": {
            "focal_length_mm": cfg.optical.focal_length * 1e3,
            "aperture_diameter_mm": cfg.optical.aperture_diameter * 1e3,
            "wavelength_min_nm": cfg.optical.wavelength_min * 1e9,
            "wavelength_max_nm": cfg.optical.wavelength_max * 1e9,
            "num_wavelengths": cfg.optical.num_wavelengths,
            "num_wings": cfg.optical.num_wings,
            "quantization_levels": cfg.optical.quantization_levels
        },
        "detector_parameters": {
            "pixel_pitch_um": cfg.detector.effective_pixel_pitch * 1e6,
            "sensor_type": cfg.detector.sensor_type,
            "noise_std": cfg.detector.noise_std
        },
        "reconstruction_metrics": {
            "PSNR_dB": float(metrics["PSNR"]),
            "SSIM": float(metrics["SSIM"]),
            "SAM_rad": float(metrics["SAM"]),
            "reconstruction_time_sec": float(t_recon)
        }
    }

    report_json_path = cfg.metrics_dir / "reconstruction_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)

    summary_txt_path = cfg.metrics_dir / "summary_report.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Jeon et al. 2019 复现系统数值指标汇总报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试时间: {report_dict['timestamp']}\n")
        f.write(f"PSNR (峰值信噪比): {metrics['PSNR']:.2f} dB\n")
        f.write(f"SSIM (结构相似度): {metrics['SSIM']:.4f}\n")
        f.write(f"SAM  (光谱角距离): {metrics['SAM']:.4f} rad ({np.degrees(metrics['SAM']):.2f}°)\n")
        f.write(f"重建耗时: {t_recon:.2f} s\n")
        f.write("=" * 60 + "\n")

    t_total = time.time() - t_start
    print(f"\n[√] 仿真与重建全流程执行圆满完成！总耗时: {t_total:.2f} 秒")
    print(f"[√] 评测报告已保存至: {report_json_path}")
    print(f"[√] 所有高清科学图表已保存至: {cfg.figures_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
