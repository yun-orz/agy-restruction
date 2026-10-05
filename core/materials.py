"""
光学材料色散模型模块

本模块提供 DOE 基底材料在不同波长下的折射率计算。
在 Jeon 2019 论文中，DOE 制作在 0.5 mm 厚的熔融石英 (Fused Silica, SiO2) 晶圆上。
光的波长不同，材料折射率不同（色散现象），从而引起相位差变化。
"""

import numpy as np


def get_refractive_index(wavelengths, material: str = "fused_silica") -> np.ndarray:
    """
    计算给定材料在指定波长处的折射率 n(λ)。
    
    参数:
        wavelengths: 波长，单位：米 (m)，支持标量浮点数或 numpy 数组
        material: 材料名称，默认为 'fused_silica' (熔融石英)
        
    返回:
        对应波长处的绝对折射率 n
    """
    wl_array = np.atleast_1d(wavelengths)
    # 转换为微米 (um)，用于代入标准 Sellmeier 色散公式
    wl_um = wl_array * 1e6
    w2 = wl_um ** 2

    if material.lower() in ["fused_silica", "sio2", "silica"]:
        # 熔融石英常温 (20°C) 标准三项 Sellmeier 色散公式系数
        # 参考文献：Malitson, I. H. (1965). Interspecimen comparison of the refractive index of fused silica. JOSA.
        b1, c1 = 0.6961663, 0.0684043 ** 2
        b2, c2 = 0.4079426, 0.1162414 ** 2
        b3, c3 = 0.8974794, 9.896161 ** 2

        n2 = 1.0 + (b1 * w2) / (w2 - c1) + (b2 * w2) / (w2 - c2) + (b3 * w2) / (w2 - c3)
        n = np.sqrt(np.maximum(n2, 1.0))
    elif material.lower() == "pmma":
        # 聚甲基丙烯酸甲酯 (PMMA 树脂) 色散公式（备选微纳压印材料）
        # 柯西公式近似
        n = 1.489 + 0.0035 / w2
    elif material.lower() == "constant":
        # 恒定折射率（用于消色差/对比实验测试）
        n = np.full_like(wl_array, 1.46)
    else:
        raise ValueError(f"未知的材料类型: {material}")

    if np.isscalar(wavelengths):
        return float(n[0])
    return n


def get_delta_n(wavelengths, material: str = "fused_silica", ambient_index: float = 1.0) -> np.ndarray:
    """
    计算 DOE 材料与周围介质（通常为空气，n_air ≈ 1.0）之间的相对折射率差：
    Δη(λ) = n_substrate(λ) - n_ambient
    """
    return get_refractive_index(wavelengths, material=material) - ambient_index


if __name__ == "__main__":
    wls = np.array([420e-9, 550e-9, 660e-9])
    n = get_refractive_index(wls)
    dn = get_delta_n(wls)
    print("材料色散测试 (熔融石英 Fused Silica)：")
    for wl, n_val, dn_val in zip(wls, n, dn):
        print(f"波长: {wl*1e9:.1f} nm -> 折射率 n: {n_val:.5f}, 折射率差 Δη: {dn_val:.5f}")
