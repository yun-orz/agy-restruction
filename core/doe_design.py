"""
DOE (衍射光学元件) 高度剖面设计模块

基于 Jeon et al. 2019 《Compact Snapshot Hyperspectral Imaging with Diffracted Rotation》
论文第 4 节公式 (7) ~ (12) 实现单螺旋/多翼旋转 PSF 的 DOE 表面微结构设计。

物理机制核心说明（供几何光学背景同学理解）：
1. 几何光学的等光程原理：平行光经透镜汇聚到焦点，中心光线经过较厚介质，边缘光线经过较薄介质，使得所有光线光程相等、同相干涉。
2. 菲涅尔透镜（Fresnel Lens）：将光程差以一个波长（对应 2π 相位）为周期进行“折叠（Wrapping）”，将毫米级厚度的宏观透镜压扁成微米级的薄膜结构。
3. 传统菲涅尔透镜的色差缺陷：其环带是按照某单一设计波长 λ0 计算的，其他波长的光无法在同一焦距 f 处同相干涉，导致巨大的弥散斑（模糊圆大小随波长剧烈变化）。
4. Jeon 2019 的发明（衍射旋转）：
   将圆孔径划分为 N 个扇区（默认 N=3 翼），在每个扇区内，不同方位角 θ 对应的半径切线分别针对不同的波长 λ(θ) 进行等光程设计。
   从而将传统色差从“弥散斑大小变化”转化为“各向异性 PSF 沿方位角的顺时针旋转”！
"""

import numpy as np
from typing import Tuple, Optional
from core.materials import get_delta_n


def generate_doe_mesh(grid_size: int, window_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 DOE 平面的笛卡尔坐标与极坐标网格。
    
    参数:
        grid_size: 网格离散采样点数 (如 1024)
        window_size: 物理计算窗口尺寸 (m)，如 1.2 mm
        
    返回:
        X: x 坐标二维网格 (m)
        Y: y 坐标二维网格 (m)
        R: 极坐标半径 r = sqrt(X^2 + Y^2) (m)
        Theta: 极坐标方位角 θ in [0, 2π) (rad)
    """
    coords = (np.arange(grid_size) - grid_size // 2) * (window_size / grid_size)
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X) % (2.0 * np.pi)
    return X, Y, R, Theta


def generate_rotating_doe_height(
    grid_size: int = 1024,
    window_size: float = 1.2e-3,
    aperture_diameter: float = 1.0e-3,
    focal_length: float = 0.05,
    wavelength_min: float = 420e-9,
    wavelength_max: float = 660e-9,
    num_wings: int = 3,
    quantization_levels: int = 16,
    quantization_step: float = 100e-9,
    substrate_material: str = "fused_silica"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    生成 Jeon 2019 论文设计的衍射旋转 DOE 物理高度分布图 h(x, y)。
    
    对应论文公式 (9) ~ (12)：
    λ(θ) = λ_min + (λ_max - λ_min) * (θ mod (2π/N)) / (2π/N)
    Δh(r, θ) = - [(sqrt(r^2 + f^2) - f) mod λ(θ)] / Δη(λ(θ))
    
    参数:
        grid_size: 网格点数
        window_size: 物理窗口大小 (m)
        aperture_diameter: 通光孔径直径 D (m)
        focal_length: 设计焦距 f (m)
        wavelength_min: 最小设计波长 (m)
        wavelength_max: 最大设计波长 (m)
        num_wings: 翼数 N (论文推荐 3)
        quantization_levels: 量化阶数 (16 或 0 表示连续)
        quantization_step: 台阶高度 (m)
        substrate_material: 基底材料
        
    返回:
        delta_h: DOE 相对中心的高程分布 Δh(x, y)，单位：米 (m)
        aperture_mask: 通光孔径掩膜（圆形孔径内为 1.0，外部为 0.0）
        metadata: 元数据字典
    """
    aperture_radius = aperture_diameter / 2.0
    X, Y, R, Theta = generate_doe_mesh(grid_size, window_size)

    # 1. 划分扇区，计算方位角对应的设计波长 λ(θ)
    wing_period = (2.0 * np.pi) / num_wings
    theta_in_wing = Theta % wing_period
    lambda_theta = wavelength_min + (wavelength_max - wavelength_min) * (theta_in_wing / wing_period)

    # 2. 计算各方位角设计波长处的材料折射率差 Δη(λ(θ))
    delta_n_theta = get_delta_n(lambda_theta, material=substrate_material)

    # 3. 几何光程差（从 DOE 上的 (r, θ) 点到焦点的光程减去沿光轴中心的光程）
    geometric_opd = np.sqrt(R ** 2 + focal_length ** 2) - focal_length

    # 4. 相位折叠（Modulo Wrapping）：以设计波长 λ(θ) 为模折叠
    wrapped_opd = geometric_opd % lambda_theta

    # 5. 计算刻蚀深度 Δh(r, θ) = - wrapped_opd / Δη
    # 高度范围约束在 [-λ(θ) / Δη, 0] 之间
    delta_h = - wrapped_opd / delta_n_theta

    # 6. 孔径边界处理：圆形通光孔径之外高度归零
    aperture_mask = (R <= aperture_radius).astype(np.float64)
    delta_h = delta_h * aperture_mask

    # 7. 微纳制造多阶台阶量化（16-level 制造工艺）
    if quantization_levels > 0 and quantization_step > 0:
        # 将连续深度按 100 nm 台阶离散量化
        discrete_steps = np.round(delta_h / quantization_step)
        # 截断在 16 个阶梯范围内 [-(quantization_levels - 1), 0]
        discrete_steps = np.clip(discrete_steps, -(quantization_levels - 1), 0)
        delta_h = discrete_steps * quantization_step * aperture_mask

    metadata = {
        "grid_size": grid_size,
        "window_size": window_size,
        "aperture_diameter": aperture_diameter,
        "focal_length": focal_length,
        "num_wings": num_wings,
        "wavelength_min": wavelength_min,
        "wavelength_max": wavelength_max,
        "quantization_levels": quantization_levels,
        "quantization_step": quantization_step,
        "max_depth_um": float(np.min(delta_h) * 1e6)
    }

    return delta_h, aperture_mask, metadata


def generate_conventional_fresnel_height(
    grid_size: int = 1024,
    window_size: float = 1.2e-3,
    aperture_diameter: float = 1.0e-3,
    focal_length: float = 0.05,
    design_wavelength: float = 550e-9,
    quantization_levels: int = 16,
    quantization_step: float = 100e-9,
    substrate_material: str = "fused_silica"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    生成传统同心环形菲涅尔透镜（各向同性，作为对照组实验）。
    针对单色设计波长（通常取中心波长 550 nm）设计。
    """
    aperture_radius = aperture_diameter / 2.0
    X, Y, R, _ = generate_doe_mesh(grid_size, window_size)

    delta_n = get_delta_n(design_wavelength, material=substrate_material)
    geometric_opd = np.sqrt(R ** 2 + focal_length ** 2) - focal_length
    wrapped_opd = geometric_opd % design_wavelength
    delta_h = - wrapped_opd / delta_n

    aperture_mask = (R <= aperture_radius).astype(np.float64)
    delta_h = delta_h * aperture_mask

    if quantization_levels > 0 and quantization_step > 0:
        discrete_steps = np.round(delta_h / quantization_step)
        discrete_steps = np.clip(discrete_steps, -(quantization_levels - 1), 0)
        delta_h = discrete_steps * quantization_step * aperture_mask

    metadata = {
        "type": "conventional_fresnel",
        "design_wavelength": design_wavelength,
        "max_depth_um": float(np.min(delta_h) * 1e6)
    }

    return delta_h, aperture_mask, metadata


def compute_doe_phase_delay(
    delta_h: np.ndarray,
    wavelength: float,
    substrate_material: str = "fused_silica"
) -> np.ndarray:
    """
    计算特定波长 λ 的光波穿过固定 DOE 结构时所产生的相位延迟：
    ϕ_h(x, y; λ) = (2π / λ) * Δη(λ) * Δh(x, y)
    
    物理要求严格遵守：
    DOE 的物理高度 Δh(x, y) 是固定的（加工成型后不可变），
    当不同波长入射时，由于波长 λ 与材料折射率差 Δη(λ) 的不同，其引起的相位调制发生变化。
    """
    delta_n = get_delta_n(wavelength, material=substrate_material)
    k = (2.0 * np.pi) / wavelength
    return k * delta_n * delta_h


if __name__ == "__main__":
    h, mask, meta = generate_rotating_doe_height()
    print("DOE 高度分布生成测试完成：")
    print(f"网格形状: {h.shape}, 最大刻蚀深度: {abs(meta['max_depth_um']):.3f} um")
    print(f"通光孔径有效像素数: {np.sum(mask):.0f} / {mask.size}")
