"""
高光谱成像质量评价指标模块 (Hyperspectral Image Quality Metrics)

提供学术界与 Jeon 2019 论文标准的四项核心评价指标：
1. PSNR (Peak Signal-to-Noise Ratio，峰值信噪比，单位 dB)：衡量整体灰度重建误差。
2. SSIM (Structural Similarity Index，结构相似度 [0, 1])：衡量空间边缘与纹理保持能力。
3. SAM (Spectral Angle Mapper，光谱角距离，单位弧度 rad)：衡量每个空间像素的光谱形状保真度，SAM 越小说明光谱保真度越高。
4. MTF (Modulation Transfer Function，调制传递函数)：衡量光学系统对空间频率的反差传递能力（空间分辨率）。
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from typing import Tuple, Dict, Any, List


def compute_psnr(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算三维高光谱立方体的平均峰值信噪比 (PSNR)。
    
    参数:
        gt: 真实高光谱数据立方体 (H, W, Λ)
        pred: 重建高光谱数据立方体 (H, W, Λ)
        max_val: 信号最大动态范围 (默认 1.0)
    """
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse <= 1e-15:
        return 100.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim_2d(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算单张二维切片图像的结构相似度 (SSIM)。
    采用标准均值、方差及协方差公式。
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim_val = ((2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_val)


def compute_ssim(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算高光谱立方体各波段二维切片 SSIM 的全谱平均值。
    """
    H, W, Λ = gt.shape
    ssim_list = []
    for i in range(Λ):
        s = compute_ssim_2d(gt[:, :, i], pred[:, :, i], max_val=max_val)
        ssim_list.append(s)
    return float(np.mean(ssim_list))


def compute_sam(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算光谱角距离 (Spectral Angle Mapper, SAM)。
    对应论文 Table 1 & Table 2 中的 SAM 指标（以弧度 rad 为单位，越小越优）。
    
    公式：
    对于每个像元 (x, y)，真实光谱向量 s 与重建光谱向量 s_hat 夹角为：
    θ = arccos( <s, s_hat> / (||s||_2 * ||s_hat||_2) )
    全图 SAM 为所有有效像元 θ 的均值。
    """
    # 展平成 (N_pixels, Λ)
    s_gt = gt.reshape(-1, gt.shape[-1]).astype(np.float64)
    s_pred = pred.reshape(-1, pred.shape[-1]).astype(np.float64)

    dot_product = np.sum(s_gt * s_pred, axis=-1)
    norm_gt = np.linalg.norm(s_gt, axis=-1)
    norm_pred = np.linalg.norm(s_pred, axis=-1)

    # 避免分母为 0（如全黑背景像元）
    valid_mask = (norm_gt > eps) & (norm_pred > eps)
    if np.sum(valid_mask) == 0:
        return 0.0

    cos_theta = dot_product[valid_mask] / (norm_gt[valid_mask] * norm_pred[valid_mask])
    # 限制在 [-1, 1] 内以防数值截断误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    return float(np.mean(angles))


def compute_mtf_edge(image_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从刀边/台阶区域计算空间调制传递函数 (MTF 曲线)，对应论文 Figure 10(b)。
    
    参数:
        image_slice: 二维单通道切片 (H, W)
        
    返回:
        freqs: 归一化空间频率 [0, 0.5] (cycles/pixel)
        mtf: 对应频率处的对比度调制值 [0, 1]
    """
    # 提取中心横截面边缘响应 (ESF: Edge Spread Function)
    H, W = image_slice.shape
    center_y = H // 2
    esf = np.mean(image_slice[center_y - 5:center_y + 5, :], axis=0)

    # 线扩散函数 (LSF: Line Spread Function) 为 ESF 的导数
    lsf = np.diff(esf)
    lsf = lsf - np.min(lsf)

    # MTF 为 LSF 的傅里叶变换幅度并归一化
    mtf = np.abs(np.fft.rfft(lsf))
    if mtf[0] > 0:
        mtf = mtf / mtf[0]

    freqs = np.fft.rfftfreq(len(lsf), d=1.0)
    return freqs, mtf


def evaluate_all_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    一站式计算所有指标并返回字典。
    """
    psnr = compute_psnr(gt, pred)
    ssim = compute_ssim(gt, pred)
    sam = compute_sam(gt, pred)
    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "SAM": sam
    }


if __name__ == "__main__":
    gt = np.random.rand(64, 64, 25).astype(np.float32)
    noise = np.random.normal(0, 0.02, size=gt.shape).astype(np.float32)
    pred = np.clip(gt + noise, 0, 1)

    res = evaluate_all_metrics(gt, pred)
    print("科学评价指标测试完成：")
    print(f"PSNR: {res['PSNR']:.2f} dB, SSIM: {res['SSIM']:.4f}, SAM: {res['SAM']:.4f} rad")
