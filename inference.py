import os
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
import math
from pathlib import Path

from model.dbcd import DBCD
from model.gaussian_diffusion import GaussianDiffusion
from utils import seed_everything, MetricTracker


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class SlidingWindowInferer:
    def __init__(self, model, diffusion, device, patch_size=256, overlap=0.5):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap))

    def predict(self, full_image):
        """
        Perform sliding window inference on a large image.
        Args:
            full_image: (H, W, 3) numpy array, RGB, [0, 255]
        Returns:
            full_mask: (H, W) numpy array, [0, 1]
        """
        h, w = full_image.shape[:2]

        # Padding to fit patch size
        pad_h = (self.patch_size - h % self.stride) % self.stride
        pad_w = (self.patch_size - w % self.stride) % self.stride

        # If image is smaller than patch size, pad to patch size
        if h < self.patch_size: pad_h = self.patch_size - h
        if w < self.patch_size: pad_w = self.patch_size - w

        img_padded = np.pad(full_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        h_pad, w_pad = img_padded.shape[:2]

        # Result accumulators
        prob_map = np.zeros((h_pad, w_pad), dtype=np.float32)
        count_map = np.zeros((h_pad, w_pad), dtype=np.float32)

        # Sliding Window Loop
        for y in range(0, h_pad - self.patch_size + 1, self.stride):
            for x in range(0, w_pad - self.patch_size + 1, self.stride):
                patch = img_padded[y:y + self.patch_size, x:x + self.patch_size]

                # Preprocess: Normalize & Tensor
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
                patch_tensor = patch_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

                # Inference (DDIM Sampling)
                with torch.no_grad():
                    # ddim_sample returns (1, 1, H, W) in range [-1, 1] or [0, 1] depending on implementation
                    # Usually diffusion operates in [-1, 1], we need to check dbcd/gaussian logic.
                    # Assuming output is sigmoid-activated or raw.
                    # Standard DDPM predicts x_0 directly or via noise.
                    # Let's assume ddim_sample output is clamped [-1, 1].
                    pred_patch = self.diffusion.ddim_sample(patch_tensor)

                    # Map [-1, 1] to [0, 1] if necessary, or simple sigmoid if it's logits
                    # Since we usually clamp x0 to [-1, 1], we map it to [0, 1] for probability
                    pred_patch = (pred_patch + 1) * 0.5
                    pred_patch = torch.clamp(pred_patch, 0, 1)

                pred_np = pred_patch.squeeze().cpu().numpy()

                # Accumulate
                prob_map[y:y + self.patch_size, x:x + self.patch_size] += pred_np
                count_map[y:y + self.patch_size, x:x + self.patch_size] += 1.0

        # Average
        avg_map = prob_map / count_map

        # Crop back to original size
        final_mask = avg_map[:h, :w]

        return final_mask


def run_inference(config):
    # 1. Setup
    seed_everything(config['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Checkpoint
    ckpt_path = config['inference']['checkpoint_path']
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    out_dir = config['inference']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # 2. Load Model
    print(f"Loading model from {ckpt_path}...")
    dbcd_model = DBCD(
        dim=config['model']['dim'],
        channels=config['model']['channels'],
        out_dim=config['model']['out_dim'],
        with_time_emb=config['model']['with_time_emb'],
        sp_channels=7  # [FIX] 必须与训练时保持一致
    ).to(device)

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device)
    dbcd_model.load_state_dict(checkpoint)
    dbcd_model.eval()

    # Init Diffusion Wrapper for Sampling
    diffusion = GaussianDiffusion(
        model=dbcd_model,
        timesteps=config['diffusion']['timesteps'],
        sampling_steps=config['diffusion']['sampling_steps'],  # 100 steps
        beta_schedule=config['diffusion']['beta_schedule']
    ).to(device)

    # 3. Setup Inferer
    inferer = SlidingWindowInferer(
        model=dbcd_model,
        diffusion=diffusion,
        device=device,
        patch_size=config['dataset']['patch_size'],
        overlap=config['inference']['window_overlap']
    )

    # 4. Prepare Test Data
    # Paper Protocol: "17 tiles for testing on Vaihingen, 14 for testing on Potsdam"
    # We assume 'raw_data_dir' contains all tifs, and we filter by a test_list if available.
    # Otherwise, for simplicity in this script, we infer on ALL tifs in raw_dir
    # (Users should place test images there or modify list loading).

    raw_dir = config['dataset']['raw_data_dir']

    # Try loading specific test split if it exists
    split_list_path = os.path.join(config['dataset']['processed_dir'], 'test_list.txt')
    if os.path.exists(split_list_path):
        with open(split_list_path, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        # Note: test_list usually contains patch names, but here we need Tile names.
        # So we scan raw_dir directly.
        print("Note: Performing inference on full tiles in raw_data_dir.")

    # Scan for TIFs
    # Adjust extension for Potsdam (usually .tif)
    image_files = [f for f in os.listdir(raw_dir) if f.endswith('.tif') and "label" not in f]

    print(f"Found {len(image_files)} tiles for inference.")

    tracker = MetricTracker()

    # 5. Inference Loop
    for fname in tqdm(image_files, desc="Inference"):
        img_path = os.path.join(raw_dir, fname)

        # Construct GT path (Adapting to ISPRS naming conventions)
        # Vaihingen: top_mosaic_09cm_area1.tif -> top_mosaic_09cm_area1_label.tif (Check your data!)
        # Potsdam: top_potsdam_2_10_RGB.tif -> top_potsdam_2_10_label.tif
        if "RGB" in fname:
            gt_name = fname.replace("_RGB.tif", "_label.tif")
        elif "IRRG" in fname:
            gt_name = fname.replace("_IRRG.tif", "_label.tif")
        else:
            gt_name = fname.replace(".tif", "_label.tif")  # Generic fallback

        gt_path = os.path.join(raw_dir, gt_name)
        # Or check a separate GT dir if structure differs
        if not os.path.exists(gt_path):
            # Try ISPRS GT folder structure if needed
            # For this script, we assume GT is alongside or provided.
            # If no GT, skip metric calc.
            has_gt = False
        else:
            has_gt = True

        # Read Image
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB for inference

        # Predict
        pred_map = inferer.predict(image)

        # Binarize
        binary_pred = (pred_map > config['inference']['threshold']).astype(np.uint8)

        # Save Result
        save_name = fname.replace(".tif", "_pred.png")
        cv2.imwrite(os.path.join(vis_dir, save_name), binary_pred * 255)

        # Metrics
        if has_gt:
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # Resize if dimensions differ (rare in ISPRS but possible)
            if gt.shape != binary_pred.shape:
                gt = cv2.resize(gt, (binary_pred.shape[1], binary_pred.shape[0]), interpolation=cv2.INTER_NEAREST)

            # ISPRS Labels: 255 is White (Impervious/Building?).
            # Usually: Building is Blue (0,0,255) in RGB GT or 255 in binary.
            # We assume GT is already binary 0/255 or 0/1.
            # If RGB GT, need color conversion logic similar to make_patches.
            # Assuming pre-processed binary GT for simplicity here.

            # If standard ISPRS RGB Label:
            # Building is (0, 0, 255) (Blue) -> Read as BGR -> (255, 0, 0)
            if len(gt.shape) == 3 or (cv2.imread(gt_path).ndim == 3):
                gt_color = cv2.imread(gt_path)  # BGR
                # Building is Blue: B=255, G=0, R=0
                # Mask where channel 0 is > 200 (Blue) and others low
                gt_binary = np.zeros(gt_color.shape[:2], dtype=np.uint8)
                # Check ISPRS color: Building = (0, 0, 255) (RGB) -> OpenCV reads (255, 0, 0)
                # Strict check
                is_building = (gt_color[:, :, 0] == 255) & (gt_color[:, :, 1] == 0) & (gt_color[:, :, 2] == 0)
                gt_binary[is_building] = 1
            else:
                gt_binary = (gt > 127).astype(np.uint8)

            tracker.update(binary_pred, gt_binary)

    # 6. Report
    scores = tracker.get_scores()
    print("\n" + "=" * 50)
    print(f"Inference Completed on {len(image_files)} tiles.")
    print("=" * 50)
    print(f"mIoU    : {scores['mIoU']:.2f}%")
    print(f"F1-Score: {scores['F1']:.2f}%")
    print(f"BoundF  : {scores['BoundF']:.2f}%")
    print("=" * 50)
    print(f"Results saved to {vis_dir}")


if __name__ == "__main__":
    # Ensure config exists
    if not os.path.exists("configs/default.yaml"):
        print("Config file not found. Please generate configs/default.yaml first.")
    else:
        config = load_config("configs/default.yaml")
        run_inference(config)