import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareContrastiveLoss(nn.Module):
    """
    对应论文 Eq. 12 & 13: Edge-Aware Contrastive Learning
    计算 z_c 和 z_d 之间的 MSE，并在边缘处加权。
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, z_c, z_d, edges):
        """
        Args:
            z_c, z_d: (B, C, H, W) 特征图
            edges: (B, 1, H, W) Canny 边缘图 (0 or 1)
        """
        # 1. 计算基础 MSE (Pixel-wise)
        loss_mse = F.mse_loss(z_c, z_d, reduction='none') # (B, C, H, W)
        
        # 2. 对齐空间尺寸 (如果 edges 分辨率不同)
        if edges.shape[-2:] != z_c.shape[-2:]:
            edges = F.interpolate(edges, size=z_c.shape[-2:], mode='nearest')
            
        # 3. 生成权重图: 平坦区域=1.0, 边缘区域=(1+alpha)
        # 论文 Eq. 13: alpha 平衡边缘贡献
        weight_map = 1.0 + self.alpha * edges
        
        # 4. 加权平均
        loss_weighted = (loss_mse * weight_map).mean()
        return loss_weighted

class AffinityRegularizationLoss(nn.Module):
    """
    对应论文 Eq. 17: Spatial Smoothness / Affinity Loss
    L_reg = sum ||c_i - c_j||^2 * exp(-||rgb_i - rgb_j||^2)
    这里我们在特征图 z_c 上应用该约束，使用原图 RGB 作为引导。
    """
    def __init__(self, sigma_rgb=0.1, radius=1):
        super().__init__()
        self.sigma_rgb = sigma_rgb
        self.radius = radius

    def forward(self, features, image):
        """
        Args:
            features: (B, C, H, W) -> z_c (语义特征)
            image: (B, 3, H, W) -> 原始 RGB 图像 (用于引导)
        """
        # 确保图像和特征图尺寸一致 (通常 z_c 是下采样过的，需要把 image 缩放或者把 feature 插值)
        # 为了效率，我们将 image 下采样到 feature 的尺寸
        if image.shape[-2:] != features.shape[-2:]:
            image_small = F.interpolate(image, size=features.shape[-2:], mode='bilinear', align_corners=False)
        else:
            image_small = image

        loss_reg = 0.0
        
        # 计算 4 邻域 (上下左右) 的差异
        # 这是一个高效的向量化实现
        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up
        
        for dx, dy in shifts:
            # Shift features and image
            # 使用 torch.roll 实现位移，虽然边界会有 wrap-around 但影响极小且效率高
            # 或者更严谨地使用 slicing
            
            # Slicing 方法 (更严谨，忽略边界)
            if dx == 0: # Vertical shift
                if dy > 0: # Right
                    feat_diff = features[..., :, :-dy] - features[..., :, dy:]
                    img_diff = image_small[..., :, :-dy] - image_small[..., :, dy:]
                else: # Left
                    feat_diff = features[..., :, -dy:] - features[..., :, :dy]
                    img_diff = image_small[..., :, -dy:] - image_small[..., :, :dy]
            else: # Horizontal shift
                if dx > 0: # Down
                    feat_diff = features[..., :-dx, :] - features[..., dx:, :]
                    img_diff = image_small[..., :-dx, :] - image_small[..., dx:, :]
                else: # Up
                    feat_diff = features[..., -dx:, :] - features[..., :dx, :]
                    img_diff = image_small[..., -dx:, :] - image_small[..., :dx, :]
            
            # 1. 特征差异 (L2 squared)
            feat_dist_sq = (feat_diff ** 2).sum(dim=1) # (B, H', W')
            
            # 2. 颜色差异 (RGB L2 squared)
            img_dist_sq = (img_diff ** 2).sum(dim=1) # (B, H', W')
            
            # 3. 亲和力权重: exp(-||dRGB||^2 / sigma)
            # 颜色越接近，权重越大，强迫特征也接近
            affinity_weight = torch.exp(-img_dist_sq / (self.sigma_rgb ** 2))
            
            # 4. 加权损失
            term = (feat_dist_sq * affinity_weight).mean()
            loss_reg += term
            
        return loss_reg / len(shifts)
