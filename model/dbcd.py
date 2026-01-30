import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. 基础组件 (Basic Modules)
# ==========================================

class SinusoidalPosEmb(nn.Module):
    """时间步编码 (Time Embedding)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    """轻量级 ResBlock"""

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """支持时间嵌入的 ResBlock"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """标准的 Self-Attention (用于 Bottleneck)"""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.view(b, self.heads, -1, h * w).permute(0, 1, 3, 2), qkv
        )
        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)
        return self.to_out(out) + x


# ==========================================
# 2. 新增组件 (New Modules for Dual-Branch)
# ==========================================

class ConditionalEncoder(nn.Module):
    """
    条件分支编码器 (f_c): 处理 SpPre 特征
    将输入 (B, C_sp, H, W) 编码为与 U-Net Bottleneck 相同的尺寸 (B, Mid_Dim, H/8, W/8)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 假设 U-Net 下采样 3 次 (dim_mults=1,2,4,8 -> /2, /4, /8)
        # 我们需要对应的 3 次下采样
        self.net = nn.Sequential(
            # Layer 1: H -> H/2
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            # Layer 2: H/2 -> H/4
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            # Layer 3: H/4 -> H/8
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),

            # Mapping to output dimension
            nn.Conv2d(256, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    """
    投影头: 将特征映射到对比学习空间 (Eq. 2)
    保持空间维度，用于 pixel-wise (or patch-wise) contrastive learning
    """

    def __init__(self, dim_in, dim_out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),  # 1x1 Conv
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_out, 1)  # Project to latent dim (e.g., 64)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. 主网络架构 (DBCD Dual-Branch Network)
# ==========================================

class DBCD(nn.Module):
    def __init__(
            self,
            dim=64,
            channels=3,
            out_dim=1,
            dim_mults=(1, 2, 4, 8),
            with_time_emb=True,
            sp_channels=5  # [NEW] SpPre 特征维度 (RGB(3) + Ent(1) + Edge(1) = 5)
    ):
        super().__init__()
        self.channels = channels

        # 1. 时间步编码
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # 2. Diffusion Branch Input
        # Input = Image(3) + Noisy Mask(1) = 4 channels
        input_channels = channels + 1
        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding=3)

        # 3. Encoder (Downsampling)
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        # Downsample
                        nn.Conv2d(dim_in, dim_out, 4, 2, 1) if not is_last else nn.Conv2d(dim_in, dim_out, 3, 1, 1),
                    ]
                )
            )

        # 4. Middle (Bottleneck)
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 5. [NEW] Conditional Branch & Projections
        # Conditional Encoder 应该输出与 mid_dim 相同的维度
        self.cond_encoder = ConditionalEncoder(sp_channels, mid_dim)

        # Projection Heads for z_c and z_d
        self.proj_c = ProjectionHead(mid_dim, 64)
        self.proj_d = ProjectionHead(mid_dim, 64)

        # 6. Decoder (Upsampling)
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out + dim_in, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in * 2, dim_in, time_emb_dim=time_dim),
                        nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Conv2d(dim_in, dim_in, 3, 1,
                                                                                                  1),
                    ]
                )
            )

        # 7. Final Output
        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, t=None, cond_features=None):
        # x: (Batch, 4, H, W) -> [Image, Noisy_Mask]
        # cond_features: (Batch, C_sp, H, W) -> SpPre 特征图

        # 1. Time Embedding
        if self.time_mlp is not None:
            if t is None:
                t = torch.tensor([50] * x.shape[0], device=x.device)
            t = self.time_mlp(t)

        # 2. Diffusion Branch Encoder
        h = self.init_conv(x)
        skips = [h]
        for block1, block2, downsample in self.downs:
            h = block1(h, t)
            skips.append(h)
            h = block2(h, t)
            skips.append(h)
            h = downsample(h)

        # 3. Middle Block (Bottleneck)
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)

        # [NEW] Extract Diffusion Feature (z_d)
        # h is currently at bottleneck resolution (e.g., 32x32 for 256 input)
        z_d = self.proj_d(h)

        # [NEW] Conditional Branch (z_c)
        z_c = None
        if cond_features is not None:
            # Check dimensions logic
            # 如果输入特征图尺寸与原图一致 (256x256), cond_encoder 会下采样到 32x32
            feat_c = self.cond_encoder(cond_features)

            # Align spatial dimensions strictly (in case of padding issues)
            if feat_c.shape[-2:] != h.shape[-2:]:
                feat_c = F.interpolate(feat_c, size=h.shape[-2:], mode='bilinear', align_corners=False)

            z_c = self.proj_c(feat_c)

        # 4. Decoder
        for block1, block2, upsample in self.ups:
            h = torch.cat((h, skips.pop()), dim=1)
            h = block1(h, t)
            h = torch.cat((h, skips.pop()), dim=1)
            h = block2(h, t)
            h = upsample(h)

        # 5. Final Output
        h = torch.cat((h, skips.pop()), dim=1)
        h = self.final_res_block(h, t)
        pred_noise = self.final_conv(h)

        # 返回: 预测噪声, 条件特征(z_c), 扩散特征(z_d)
        return pred_noise, z_c, z_d


if __name__ == "__main__":
    try:
        # Test Dual-Branch
        model = DBCD(dim=64, sp_channels=5).cuda()

        # Batch=2, Image+Mask=4ch, H=256, W=256
        inp = torch.randn(2, 4, 256, 256).cuda()
        time = torch.randint(0, 1000, (2,)).cuda()

        # SpPre Features (5ch)
        cond = torch.randn(2, 5, 256, 256).cuda()

        pred, zc, zd = model(inp, time, cond_features=cond)

        print(f"Test Pass!")
        print(f"Pred shape: {pred.shape} (Expect B,1,256,256)")
        print(f"z_c shape:  {zc.shape}  (Expect B,64,32,32)")
        print(f"z_d shape:  {zd.shape}  (Expect B,64,32,32)")

    except Exception as e:
        print(f"Test Failed: {e}")