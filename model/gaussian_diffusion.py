import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size=256, timesteps=1000, sampling_steps=100,
                 beta_schedule="linear", beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.sampling_steps = sampling_steps  # For DDIM

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Helper function to register buffer (automatically move to device)
        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, cond_feats=None, noise=None):
        """
        Calculate loss for a training step.
        x_start: The target clean mask (pseudo-label)
        cond_feats: The encoded image features or the image itself (Condition)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict noise
        # DBCD model forward: model(x, t) -> pred_noise (usually) or pred_x0
        # Assuming the model outputs the predicted noise epsilon
        # Note: In DBCD, model takes (x_noisy, t) AND we likely concat conditional image
        # The model defined in dbcd.py handles the concat of image+noise internally in init_conv?
        # Check dbcd.py: "h = self.init_conv(torch.cat([x, noise], dim=1))"
        # Wait, standard diffusion usually concats conditioning input (image) with noisy mask.
        # In dbcd.py, forward(x, t) expects x. It concats random noise inside? No.
        # Let's align with standard implementation:
        # We pass `x_noisy` (1 channel) and `cond_image` (3 channels) concatenated.

        # Important: The dbcd.py provided seems to act as the U-Net.
        # We need to adapt input logic here.
        # If DBCD.forward takes x and t, and does `torch.cat([x, noise], dim=1)`,
        # it implies x is the Condition (Image) and noise is the X_t.
        # Let's adjust usage: model(cond_image, t) is not right.
        # Standard: model(torch.cat([x_noisy, cond_image], dim=1), t)

        # Based on dbcd.py structure provided earlier:
        # `noise = torch.randn(...)` inside forward suggests it might be generating noise internally?
        # Actually, looking at `dbcd.py`:
        # `h = self.init_conv(torch.cat([x, noise], dim=1))`
        # This looks like it treats 'x' as image and 'noise' as standard Gaussian noise concat.
        # But for diffusion training, we need to pass X_t (Noisy Mask) as input.
        # Let's assume we modify dbcd.py slightly or wrap inputs to fit standard diffusion.
        # We will feed [Image, X_t] to the model.

        # Correct logic for Segmentation Diffusion:
        # Input to UNet: Concat(Image, Noisy_Mask_t)
        model_input = torch.cat([cond_feats, x_noisy], dim=1)
        # NOTE: You might need to adjust dbcd.py input channels to 3+1=4 if not already set.

        model_out = self.model(model_input, t)

        loss = F.mse_loss(model_out, noise)
        return loss, model_out

    @torch.no_grad()
    def ddim_sample(self, cond_image):
        """
        DDIM Accelerated Sampling (Paper IV.F: 100 steps)
        """
        batch_size = cond_image.shape[0]
        device = cond_image.device

        # 1. Start from pure noise
        img = torch.randn((batch_size, 1, self.image_size, self.image_size), device=device)

        # 2. Setup DDIM time steps (e.g., [0, 10, 20, ...])
        step_ratio = self.timesteps // self.sampling_steps
        timesteps = torch.arange(0, self.timesteps, step_ratio).flip(0).to(device)

        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            # Construct input
            # Again, assuming model takes Concat(Image, Mask)
            model_input = torch.cat([cond_image, img], dim=1)

            # Predict noise
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.model(model_input, t_batch)

            # DDIM update step
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else \
            self.alphas_cumprod[0]

            sigma = 0  # Deterministic DDIM

            # Predict x0
            pred_x0 = (img - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
            pred_x0 = torch.clamp(pred_x0, -1., 1.)

            # Direction to x_t-1
            dir_xt = torch.sqrt(1. - alpha_cumprod_prev - sigma ** 2) * pred_noise

            # x_t-1
            img = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt

        return img

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)