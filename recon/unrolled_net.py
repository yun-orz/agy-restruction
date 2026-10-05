"""
深度展开优化网络模块 (Optimization-based Unrolled Network)

基于 Jeon et al. 2019 《Compact Snapshot Hyperspectral Imaging with Diffracted Rotation》
第 5.2 节与第 5.3 节架构实现：
1. 优化展开框架（Unrolled HQS / ISTA-Net 思想）：
   将传统的 L 次半二次分裂优化迭代展开为端到端的前馈深度神经网络。
   每一阶段包含两个核心操作：
   (a) 数据一致性梯度下降步：
       I^{(l+1)} = I^{(l)} - ε_l * [ Φ^T(Φ*I^{(l)} - J) + ς_l * (I^{(l)} - V^{(l)}) ]
       其中步长 ε_l 和惩罚因子 ς_l 为该阶段可学习的参数。
   (b) 空间-光谱先验 U-Net 网络（Spatial-Spectral Prior Network）：
       V^{(l+1)} = S_θ_l(I^{(l+1)})
       包含多尺度收缩路径、跳跃连接（带软阈值收缩 Soft-thresholding）与扩张路径。
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SoftThreshold(nn.Module):
    """
    可学习阈值的软阈值收缩层 (Soft-Thresholding)
    用于跳跃连接中提取高频梯度稀疏性（遵循 ISTA-Net 设计）。
    """
    def __init__(self, num_channels: int, init_thresh: float = 0.01):
        super().__init__()
        self.threshold = nn.Parameter(torch.full((1, num_channels, 1, 1), init_thresh))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thresh = F.relu(self.threshold)
        return torch.sign(x) * F.relu(torch.abs(x) - thresh)


class ConvBlock(nn.Module):
    """基础空间-光谱双卷积块：Conv(3x3) -> LeakyReLU -> Conv(3x3) -> LeakyReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialSpectralUNetPrior(nn.Module):
    """
    空间-光谱先验 U-Net 子网络 (Figure 5)
    
    输入: 包含 25 个波段的高光谱图像 (B, 25, H, W)
    特征通道: Γ = 64
    输出: 先验滤波后的高光谱特征立方体 (B, 25, H, W)
    """
    def __init__(self, num_wavelengths: int = 25, feature_size: int = 64):
        super().__init__()
        self.in_conv = nn.Conv2d(num_wavelengths, feature_size, kernel_size=3, padding=1)

        # 编码器 (收缩路径)
        self.enc1 = ConvBlock(feature_size, feature_size)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(feature_size, feature_size * 2)
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = ConvBlock(feature_size * 2, feature_size * 2)

        # 跳跃连接软阈值收缩
        self.thresh1 = SoftThreshold(feature_size)
        self.thresh2 = SoftThreshold(feature_size * 2)

        # 解码器 (扩张路径)
        self.up2 = nn.ConvTranspose2d(feature_size * 2, feature_size * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(feature_size * 4, feature_size)

        self.up1 = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(feature_size * 2, feature_size)

        # 输出卷积层
        self.out_conv = nn.Conv2d(feature_size, num_wavelengths, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.in_conv(x)

        # 编码
        e1 = self.enc1(x_in)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # 瓶颈
        b = self.bottleneck(p2)

        # 解码 + 跳跃软阈值特征融合
        u2 = self.up2(b)
        e2_thresh = self.thresh2(e2)
        d2 = self.dec2(torch.cat([u2, e2_thresh], dim=1))

        u1 = self.up1(d2)
        e1_thresh = self.thresh1(e1)
        d1 = self.dec1(torch.cat([u1, e1_thresh], dim=1))

        residual = self.out_conv(d1)
        # 残差连接：输入加上 U-Net 学习的去模糊去噪残差
        return x + residual


class DiffractedRotationUnrolledNet(nn.Module):
    """
    Jeon 2019 优化展开端到端光谱重建网络 (Figure 4)
    
    参数:
        num_stages: 展开优化迭代阶段数 L (例如 5 或 8 阶)
        num_wavelengths: 光谱波段数 (25)
        feature_size: 特征通道数 Γ (64)
    """
    def __init__(
        self,
        num_stages: int = 5,
        num_wavelengths: int = 25,
        feature_size: int = 64
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_wavelengths = num_wavelengths

        # 每一阶段独立的可学习优化超参数（梯度步长 ε 与惩罚因子 ς）
        self.step_sizes = nn.Parameter(torch.full((num_stages,), 1.0))
        self.penalties = nn.Parameter(torch.full((num_stages,), 0.5))

        # 每一阶段独立的空间-光谱 U-Net 先验网络
        self.prior_stages = nn.ModuleList([
            SpatialSpectralUNetPrior(num_wavelengths=num_wavelengths, feature_size=feature_size)
            for _ in range(num_stages)
        ])

    def forward(
        self,
        measurement: torch.Tensor,
        initial_cube: torch.Tensor,
        forward_fn,
        adjoint_fn
    ) -> torch.Tensor:
        """
        前向重建推理。
        
        参数:
            measurement: 探测器输入图像 J, 形状 (B, C, H, W)
            initial_cube: 初始估计 I^(0) = Φ^T * J, 形状 (B, Λ, H, W)
            forward_fn: 前向算子函数 Φ(I) -> J
            adjoint_fn: 伴随算子函数 Φ^T(J) -> I
            
        返回:
            recon_cube: 重建后的高光谱数据立方体 (B, Λ, H, W)
        """
        I_curr = initial_cube
        V_curr = initial_cube

        for stage in range(self.num_stages):
            eps = F.softplus(self.step_sizes[stage])
            rho = F.softplus(self.penalties[stage])
            prior_net = self.prior_stages[stage]

            # 1. 数据一致性更新 (Data Consistency Step)
            meas_pred = forward_fn(I_curr)
            meas_residual = meas_pred - measurement
            grad_data = adjoint_fn(meas_residual)
            grad_penalty = rho * (I_curr - V_curr)

            total_grad = grad_data + grad_penalty
            I_next = I_curr - eps * total_grad

            # 2. 空间-光谱先验滤波更新 (Prior Network Step)
            V_next = prior_net(I_next)

            I_curr = I_next
            V_curr = V_next

        return torch.clamp(V_curr, 0.0, 1.0)


if __name__ == "__main__":
    print("测试 Jeon 2019 深度展开网络架构...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffractedRotationUnrolledNet(num_stages=3, num_wavelengths=25, feature_size=32).to(device)

    # 伪数据测试前向传递
    B, C, H, W = 1, 3, 64, 64
    dummy_meas = torch.rand(B, C, H, W).to(device)
    dummy_init = torch.rand(B, 25, H, W).to(device)

    # 简化的线性测试算子
    dummy_forward = lambda x: torch.mean(x, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    dummy_adjoint = lambda y: torch.mean(y, dim=1, keepdim=True).repeat(1, 25, 1, 1)

    out = model(dummy_meas, dummy_init, dummy_forward, dummy_adjoint)
    print(f"网络输出尺寸: {out.shape}, 极值: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"总可学习参数量: {sum(p.numel() for p in model.parameters()):,}")
