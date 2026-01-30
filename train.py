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
from model.loss import EdgeAwareContrastiveLoss, AffinityRegularizationLoss # [NEW]
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
    
    progress = current_iter / total_iter

    # Lambda 1: Exponential Decay
    lambda1 = l_max * math.exp(-5.0 * progress)

    # Lambda 2: Linear Growth
    lambda2 = l_min + (l_max - l_min) * progress

    return lambda1, lambda2


def train(config):
    # 1. Setup Environment
    seed_everything(config['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    out_dir = config['inference']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.dirname(config['inference']['checkpoint_path'])
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. Data Loading
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
    # [FIX] sp_channels=7 (RGB(3)+Ent(1)+Edge(1)+XY(2)) based on sppre.py
    dbcd_model = DBCD(
        dim=config['model']['dim'],
        channels=config['model']['channels'],
        out_dim=config['model']['out_dim'],
        with_time_emb=config['model']['with_time_emb'],
        sp_channels=7 
    ).to(device)

    diffusion = GaussianDiffusion(
        model=dbcd_model,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule']
    ).to(device)

    # 4. Losses & Optimizer
    # [NEW] Initialize Loss Modules
    alpha_edge = config['train']['loss_weights']['edge_weight']
    criterion_contrast = EdgeAwareContrastiveLoss(alpha=alpha_edge).to(device)
    criterion_reg = AffinityRegularizationLoss(sigma_rgb=0.1).to(device)
    
    # Lambda 3 (Regularization Weight) - Paper doesn't specify schedule, usually constant
    lambda3 = config['train']['loss_weights'].get('lambda_reg', 0.1) 

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
    
    print(f"Start training DBCD for {total_iters} iterations...")
    print(f"Config: L_reg weight (lambda3) = {lambda3}")
    
    dbcd_model.train()

    train_iter = iter(train_loader)
    pbar = tqdm(range(total_iters))
    
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        images, masks, sp_feats = batch
        images = images.to(device)
        masks = masks.to(device)
        sp_feats = sp_feats.to(device)

        # --- A. Dynamic Weights ---
        lambda1, lambda2 = get_dynamic_weights(step, total_iters, config)

        optimizer.zero_grad()

        # --- B. Diffusion Forward ---
        t = torch.randint(0, config['diffusion']['timesteps'], (images.shape[0],), device=device).long()
        noise = torch.randn_like(masks)
        x_noisy = diffusion.q_sample(x_start=masks, t=t, noise=noise)

        # Input: Image + Noisy Mask
        model_input = torch.cat([images, x_noisy], dim=1) 

        # --- C. Model Forward ---
        # Returns: pred_noise, z_c (Conditional Feat), z_d (Diffusion Feat)
        pred_noise, z_c, z_d = dbcd_model(model_input, t, cond_features=sp_feats)

        # --- D. Loss Calculation ---
        
        # 1. Diffusion Loss (Edge-Aware weighting is now optional or handled implicitly, 
        # but let's keep the edge-weighting on diffusion loss as per previous code logic for robustness)
        edges = get_canny_edge(images) # (B, 1, H, W)
        
        # Pixel-wise MSE for Diffusion
        loss_mse = F.mse_loss(pred_noise, noise, reduction='none')
        weight_map_diff = 1.0 + alpha_edge * edges
        loss_diff = (loss_mse * weight_map_diff).mean()

        # 2. Contrastive Loss (Eq. 13)
        if z_c is not None and z_d is not None:
            # 使用 loss.py 中的 EdgeAwareContrastiveLoss
            loss_contrast = criterion_contrast(z_c, z_d, edges)
            
            # 3. [NEW] Regularization Loss (Eq. 17)
            # Enforce spatial smoothness on z_c guided by image
            loss_reg = criterion_reg(z_c, images)
        else:
            loss_contrast = torch.tensor(0.0, device=device)
            loss_reg = torch.tensor(0.0, device=device)

        # 4. Total Loss (Eq. 16)
        loss_total = lambda1 * loss_contrast + lambda2 * loss_diff + lambda3 * loss_reg

        # --- E. Backprop ---
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # --- F. Logging ---
        pbar.set_description(
            f"L_tot:{loss_total.item():.3f} | "
            f"Diff:{loss_diff.item():.3f} | "
            f"Con:{loss_contrast.item():.3f} | "
            f"Reg:{loss_reg.item():.3f} | "
            f"w1:{lambda1:.2f} w2:{lambda2:.2f}"
        )

        if step % 5000 == 0 and step > 0:
            save_path = os.path.join(out_dir, f"ckpt_{step}.pth")
            torch.save(dbcd_model.state_dict(), save_path)

    final_path = config['inference']['checkpoint_path']
    torch.save(dbcd_model.state_dict(), final_path)
    print(f"Training finished. Model saved to {final_path}")


if __name__ == "__main__":
    if os.path.exists("configs/default.yaml"):
        config = load_config("configs/default.yaml")
        train(config)
    else:
        print("Error: configs/default.yaml not found.")
