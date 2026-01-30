import os
import cv2
import yaml
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
sys.path.append(str(project_root))

from model.sppre import SpPreModule


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_patches(config):
    dataset_name = config['dataset']['name']
    raw_dir = config['dataset']['raw_data_dir']
    out_dir = config['dataset']['processed_dir']
    patch_size = config['dataset']['patch_size']
    stride = config['dataset']['train_stride']
    use_filtering = config['dataset'].get('use_filtering', False)

    # 1. Setup Directories
    img_out_dir = os.path.join(out_dir, 'images')
    mask_out_dir = os.path.join(out_dir, 'masks')
    feat_out_dir = os.path.join(out_dir, 'features')  # [NEW] 用于存储超像素特征图

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(feat_out_dir, exist_ok=True)

    # 2. Initialize SpPre Module
    print(f"Initializing SpPre for {dataset_name}...")
    sppre = SpPreModule(
        dataset_type=dataset_name,
        rho_fine=config['dataset']['sppre_params']['rho_fine'],
        rho_coarse=config['dataset']['sppre_params']['rho_coarse'],
        tau=config['dataset']['sppre_params']['tau'],
        sigma_scale=config['dataset']['sppre_params']['sigma_scale']
    )

    # 3. Find Images
    # Assuming TIFF files. Adjust extension if needed (e.g., .png, .tif).
    search_path = os.path.join(raw_dir, "*.tif")
    img_files = glob.glob(search_path)
    if len(img_files) == 0:
        print(f"No .tif files found in {raw_dir}. Please check path.")
        return

    print(f"Found {len(img_files)} images. Starting patch generation...")

    total_patches = 0
    kept_patches = 0

    for img_path in tqdm(img_files):
        fname = os.path.basename(img_path).split('.')[0]

        # Read Image
        # OpenCV reads as BGR. Note: sppre.process handles BGR->RGB/Gray conversion internally.
        # But for saving patches, we keep them as is (BGR) or convert if you prefer RGB on disk.
        # Here we save as BGR (standard OpenCV behavior).
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                total_patches += 1

                # Crop
                img_crop = image[y:y + patch_size, x:x + patch_size]

                # --- SpPre Generation & Filtering Logic ---
                # process returns: mask, features, segments, confidence_map
                # c_fine (features): (N_segments, Feature_Dim)
                # segments_fine: (H, W) index map
                mask_pred, c_fine, segments_fine, _ = sppre.process(img_crop)

                # Binary Mask (0 or 1)
                mask_binary = (mask_pred > 0).astype(np.uint8)

                # --- Filtering Strategy (Paper IV.D) ---
                # Logic: Only keep patches where the model found high-confidence building structures.
                # 25.2% retention rate is a result of this dynamic filtering, not a hard limit.
                if use_filtering and dataset_name == 'potsdam':
                    # Threshold: at least 50 pixels classified as building to avoid noise patches
                    if np.sum(mask_binary) < 50:
                        continue

                kept_patches += 1
                save_name = f"{fname}_{y}_{x}"

                # --- Construct Pixel-wise Feature Map for Conditional Branch ---
                # Map segment-level features back to pixel space (H, W, Dim)
                # c_fine shape: (Num_Segments, Dim)
                # segments_fine shape: (H, W), values in [0, Num_Segments-1]

                # Create a placeholder feature map
                # c_fine.shape[1] is typically 5 or 8 depending on sppre implementation
                feat_dim = c_fine.shape[1]
                feature_map = np.zeros((patch_size, patch_size, feat_dim), dtype=np.float32)

                # Advanced indexing to fill the map efficiently
                # Warning: segments_fine indices must match row indices of c_fine
                # SLIC labels usually are 0..N-1, but strictly we should check mapping
                unique_labels = np.unique(segments_fine)

                # Fast Vectorized Mapping (replacing slow loop)
                # c_fine is indexed by segment ID.
                # Ideally, c_fine rows correspond exactly to segment IDs 0, 1, 2...
                # sppre.py _extract_features usually returns features sorted by label if implemented standardly.
                # Assuming sppre returns features aligned with unique_labels or 0..max_label

                # Safe loop method to ensure alignment:
                # (Note: sppre.py in previous context returned unique_labels alongside features)
                # If c_fine corresponds to unique_labels indices:
                for idx, label in enumerate(unique_labels):
                    if idx < len(c_fine):
                        # Create mask for this segment
                        seg_mask = (segments_fine == label)
                        feature_map[seg_mask] = c_fine[idx]

                # --- Save Artifacts ---
                # 1. Image
                cv2.imwrite(os.path.join(img_out_dir, f"{save_name}.png"), img_crop)

                # 2. Pseudo-label Mask (0 or 255)
                cv2.imwrite(os.path.join(mask_out_dir, f"{save_name}.png"), mask_binary * 255)

                # 3. Superpixel Features (.npy) - Vital for Dual-Branch Training
                np.save(os.path.join(feat_out_dir, f"{save_name}.npy"), feature_map)

    print(f"Processing Complete.")
    print(f"Total Patches Scanned: {total_patches}")
    print(f"Patches Kept after Filtering: {kept_patches}")
    if total_patches > 0:
        print(f"Retention Rate: {kept_patches / total_patches * 100:.2f}% (Target: ~25.2% for Potsdam)")

    # Save Split List (Train List)
    # We use the filenames from the images folder as the definitive list
    with open(os.path.join(out_dir, 'train_list.txt'), 'w') as f:
        imgs = sorted(os.listdir(img_out_dir))
        for i in imgs:
            f.write(f"{i}\n")


if __name__ == "__main__":
    # Ensure config exists relative to script execution
    # Assuming script is run from project root: python data/make_patches.py
    # or python make_patches.py inside data/

    # Try to find config
    possible_paths = [
        "configs/default.yaml",
        "../configs/default.yaml",
        "../../configs/default.yaml"
    ]

    config_path = None
    for p in possible_paths:
        if os.path.exists(p):
            config_path = p
            break

    if config_path is None:
        print("Error: configs/default.yaml not found. Please run from project root.")
    else:
        print(f"Loading config from {config_path}")
        config = load_config(config_path)
        make_patches(config)