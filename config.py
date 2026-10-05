"""
Jeon et al. 2019 《Compact Snapshot Hyperspectral Imaging with Diffracted Rotation》
集中系统配置文件

本文件包含复现系统的所有光学参数、DOE设计参数、探测器参数、仿真网格参数以及重建算法参数。
所有路径均相对于当前项目根目录动态解析，支持在任意目录和 PyCharm 中直接运行。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np


@dataclass
class OpticalConfig:
    """光学系统与 DOE 设计参数"""
    # 焦距 f = 50 mm (0.05 m)
    focal_length: float = 0.05
    # DOE 通光孔径直径 D = 1.0 mm (0.001 m)
    aperture_diameter: float = 0.001
    # DOE 通光孔径半径 R = D / 2
    aperture_radius: float = field(init=False)
    
    # 工作光谱范围：420 nm ~ 660 nm，25 个谱段，间隔 10 nm
    wavelength_min: float = 420e-9
    wavelength_max: float = 660e-9
    num_wavelengths: int = 25
    wavelengths: np.ndarray = field(init=False)
    
    # DOE 螺旋翼数 N（Jeon 2019 论文推荐 N=3，即 120° 周期对称）
    num_wings: int = 3
    
    # 基底材料：熔融石英 (Fused Silica, SiO2)
    substrate: str = "fused_silica"
    
    # DOE 微结构量化阶数：16 台阶（每阶 100 nm，总深度 ~1.5 um），设为 0 表示连续高度剖面
    quantization_levels: int = 16
    quantization_step: float = 100e-9  # 100 nm 台阶高度
    
    def __post_init__(self):
        self.aperture_radius = self.aperture_diameter / 2.0
        self.wavelengths = np.linspace(self.wavelength_min, self.wavelength_max, self.num_wavelengths)


@dataclass
class DetectorConfig:
    """探测器传感器参数 (Canon EOS 5D Mark III)"""
    # 真实像元尺寸 6.22 um
    pixel_pitch: float = 6.22e-6
    # 模拟时可选的降采样倍率（论文第 6 节中提到计算网络使用 2 倍下采样以减小显存与计算开销，即 12.44 um）
    # 在本仿真中，默认使用真实 6.22 um，也可配置为 2x
    binning_factor: int = 1
    effective_pixel_pitch: float = field(init=False)
    
    # 探测器类型：'rgb' (三通道彩色相机) 或 'mono' (单色全色相机)
    sensor_type: str = "rgb"
    
    # PSF 提取与卷积时的核尺寸（像元数，奇数），默认 65x65 像元，覆盖 ~404 um 的物理视场，完全覆盖旋转 PSF
    psf_kernel_size: int = 65
    
    # 模拟探测噪声（高斯白噪声标准差，论文 Section 6 设置为 0.005）
    noise_std: float = 0.005
    
    def __post_init__(self):
        self.effective_pixel_pitch = self.pixel_pitch * self.binning_factor


@dataclass
class SimulationGridConfig:
    """波动光学传播网格参数"""
    # DOE 平面采样点数（默认 1024x1024，兼顾高精度与计算速度）
    doe_grid_size: int = 1024
    # DOE 平面物理计算视场宽度（1.2 mm，大于 1.0 mm 孔径以留出边界）
    doe_window_size: float = 1.2e-3
    
    # 衍射计算方法：'fresnel_sft' (菲涅尔单次傅里叶变换，焦平面严谨解析)
    propagation_method: str = "fresnel_sft"


@dataclass
class ReconstructionConfig:
    """光谱图像重建算法超参数"""
    # 优化迭代算法：'hqs' (Half-Quadratic Splitting 半二次分裂法) 或 'tval3'
    algorithm: str = "hqs"
    
    # HQS 外层迭代次数（论文设置 10~15 次）
    hqs_iterations: int = 15
    # 惩罚项参数 ς (penalty parameter)
    hqs_rho: float = 1.0
    # 梯度下降步长 ε (gradient descent step size)
    hqs_step_size: float = 0.5
    
    # 空间-光谱正则化权重 (TV 权重)
    tv_spatial_weight: float = 0.015
    tv_spectral_weight: float = 0.008
    
    # 测试图像尺寸 (H, W)，默认 128x128 适合 CPU 快速运行与验证；也可设置为 256x256
    test_image_size: tuple = (128, 128)


@dataclass
class ProjectConfig:
    """全局项目配置整合对象"""
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    grid: SimulationGridConfig = field(default_factory=SimulationGridConfig)
    recon: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    
    # 路径管理（统一使用相对项目根目录的绝对解析路径）
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.resolve())
    data_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.figures_dir = self.results_dir / "figures"
        self.metrics_dir = self.results_dir / "metrics"
        
        # 确保目录存在
        for p in [self.data_dir, self.results_dir, self.figures_dir, self.metrics_dir]:
            p.mkdir(parents=True, exist_ok=True)


# 全局默认单例配置
default_config = ProjectConfig()

if __name__ == "__main__":
    cfg = ProjectConfig()
    print("=" * 60)
    print("Jeon 2019 复现系统配置测试：")
    print(f"项目根目录: {cfg.project_root}")
    print(f"焦距: {cfg.optical.focal_length * 1e3:.1f} mm, 孔径: {cfg.optical.aperture_diameter * 1e3:.1f} mm")
    print(f"光谱通道数: {cfg.optical.num_wavelengths} ({cfg.optical.wavelength_min*1e9:.0f} ~ {cfg.optical.wavelength_max*1e9:.0f} nm)")
    print(f"探测器像元尺寸: {cfg.detector.effective_pixel_pitch * 1e6:.2f} um")
    print(f"PSF 核尺寸: {cfg.detector.psf_kernel_size}x{cfg.detector.psf_kernel_size}")
    print("=" * 60)
