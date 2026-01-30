import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import numpy as np

# 引入项目模块
from model.dbcd import DBCD
from model.gaussian_diffusion import GaussianDiffusion
from data.dataset import DBCDDataset, get_train_transform
from utils import seed_everything, get_canny_edge


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ==========================================
# Dynamic Weight Scheduler (Paper Eq. 14)
# ==========================================
def get_dynamic_weights(current_iter, total_iter, config):
    """
    Curriculum Learning Schedule:
    lambda1 (Contrastive): Starts high, decays exponentially.
    lambda2 (Diffusion): Starts low, grows linearly.
    """
    w_conf = config['train']['loss_weights']
    l_max = w_conf['lambda_max']
    l_min = w_conf['lambda_min']

    # 论文公式依赖的是当前迭代进度
    progress = current_iter / total_iter

    # Lambda 1: Exponential Decay (Semantic Alignment phase)
    # 衰减系数 -5.0 确保在训练后期权重接近 0
    lambda1 = l_max * math.exp(-5.0 * progress)

    # Lambda 2: Linear Growth (Geometric Refinement phase)
    lambda2 = l_min + (l_max - l_min) * progress

    return lambda1, lambda2


def train(config):
    # 1. Setup Environment
    seed_everything(config['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Directories
    out_dir = config['inference']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.dirname(config['inference']['checkpoint_path'])
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. Data Loading
    # 注意: Dataset 现在返回 (image, mask, features)
    train_dataset = DBCDDataset(
        data_dir=config['dataset']['processed_dir'],
        list_file=os.path.join(config['dataset']['processed_dir'], 'train_list.txt'),
        transform=get_train_transform(config),
        is_train=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 3. Model Initialization
    # 注意: channels=3, 但 DBCD 内部 init_conv 会处理 channels+1 的输入
    dbcd_model = DBCD(
        dim=config['model']['dim'],
        channels=config['model']['channels'],
        out_dim=config['model']['out_dim'],
        with_time_emb=config['model']['with_time_emb'],
        sp_channels=7
    ).to(device)

    # Diffusion 包装器 (主要用于 q_sample 和 betas 管理)
    diffusion = GaussianDiffusion(
        model=dbcd_model,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule']
    ).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        dbcd_model.parameters(),
        lr=float(config['train']['learning_rate']),
        weight_decay=float(config['train']['weight_decay'])
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['train']['num_iterations']
    )

    # 5. Training Loop
    total_iters = config['train']['num_iterations']
    alpha_edge = config['train']['loss_weights']['edge_weight']

    print(f"Start training DBCD for {total_iters} iterations on {device}...")
    dbcd_model.train()

    train_iter = iter(train_loader)
    pbar = tqdm(range(total_iters))

    for step in pbar:
        # --- Data Fetching ---
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # 解包数据: Image (B,3,H,W), Mask (B,1,H,W), SpFeats (B,C,H,W)
        images, masks, sp_feats = batch
        images = images.to(device)
        masks = masks.to(device)
        sp_feats = sp_feats.to(device)

        # --- A. Dynamic Weights ---
        lambda1, lambda2 = get_dynamic_weights(step, total_iters, config)

        optimizer.zero_grad()

        # --- B. Diffusion Process ---
        # 1. Sample t and noise
        t = torch.randint(0, config['diffusion']['timesteps'], (images.shape[0],), device=device).long()
        noise = torch.randn_like(masks)

        # 2. Add noise (Forward Diffusion) -> x_t
        x_noisy = diffusion.q_sample(x_start=masks, t=t, noise=noise)

        # 3. Construct Model Input
        # Diffusion Branch Input: Image (Condition) + Noisy Mask (Target)
        # 拼接顺序必须与 dbcd.py 中 init_conv 的预期一致 (通常是 dim=1)
        model_input = torch.cat([images, x_noisy], dim=1)

        # --- C. Model Forward (Dual Branch) ---
        # 传入 model_input 和 cond_features (Superpixel Features)
        # 返回: 预测噪声, z_c (条件特征), z_d (扩散特征)
        pred_noise, z_c, z_d = dbcd_model(model_input, t, cond_features=sp_feats)

        # --- D. Loss Calculation ---

        # 1. Edge-Aware Diffusion Loss (L_diff)
        # 计算 Canny 边缘用于加权
        edges = get_canny_edge(images)  # (B, 1, H, W)
        weight_map = 1.0 + alpha_edge * edges

        # Pixel-wise MSE
        loss_mse = F.mse_loss(pred_noise, noise, reduction='none')
        # 加权平均
        loss_diff = (loss_mse * weight_map).mean()

        # 2. Contrastive Loss (L_contrast)
        if z_c is not None and z_d is not None:
            # 这里的 z_c 和 z_d 是投影后的特征图 (B, 64, H_mid, W_mid)
            # 计算特征空间的对齐误差
            loss_contrast_raw = F.mse_loss(z_c, z_d, reduction='none')

            # 同样应用 Edge-Aware 约束 (论文 Eq. 12 & 13)
            # 需要将边缘图下采样到特征图尺寸
            weight_map_small = F.interpolate(weight_map, size=z_c.shape[-2:], mode='nearest')
            loss_contrast = (loss_contrast_raw * weight_map_small).mean()
        else:
            # 如果是单分支消融实验或未启用双分支
            loss_contrast = torch.tensor(0.0, device=device)

        # 3. Total Loss (Eq. 15)
        loss_total = lambda2 * loss_diff + lambda1 * loss_contrast

        # --- E. Backprop ---
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # --- F. Logging ---
        pbar.set_description(
            f"Loss: {loss_total.item():.4f} | "
            f"Diff: {loss_diff.item():.4f} | "
            f"Cont: {loss_contrast.item():.4f} | "
            f"L1: {lambda1:.3f} L2: {lambda2:.3f}"
        )

        # Checkpointing
        if step % 5000 == 0 and step > 0:
            save_path = os.path.join(out_dir, f"ckpt_{step}.pth")
            torch.save(dbcd_model.state_dict(), save_path)

    # Save Final Model
    final_path = config['inference']['checkpoint_path']
    torch.save(dbcd_model.state_dict(), final_path)
    print(f"Training finished. Best model saved to {final_path}")


if __name__ == "__main__":
    if os.path.exists("configs/default.yaml"):
        config = load_config("configs/default.yaml")
        train(config)
    else:
        print("Error: configs/default.yaml not found.")