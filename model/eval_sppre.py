import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 1. 粘贴您的 SpPreModule 类定义 =================

# 这里我先粘贴您刚才提供的类，确保环境闭环
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.morphology import closing, square, opening
from scipy.spatial.distance import cdist
import warnings
from sppre import SpPreModule

warnings.filterwarnings("ignore", category=UserWarning)



# ================= 2. 评估代码 (Potsdam IRRG) =================

class MetricTracker:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred, gt):
        self.tp += np.sum((pred == 1) & (gt == 1))
        self.fp += np.sum((pred == 1) & (gt == 0))
        self.fn += np.sum((pred == 0) & (gt == 1))

    def get_scores(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        return iou * 100, f1 * 100


def main():
    # 配置路径
    IMG_DIR = r"D:\Python Project\datasets\Potsdam_IRRG_Sliced\images"
    MASK_DIR = r"D:\Python Project\datasets\Potsdam_IRRG_Sliced\masks"

    # 初始化 SpPre 模块
    # !!! 关键 !!!
    # 我们故意设为 'vaihingen'，以触发 class 内部的 NDVI 物理先验逻辑
    sppre_module = SpPreModule(dataset_type='vaihingen', rho_fine=100, rho_coarse=400, tau=0.35)

    tracker = MetricTracker()
    files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    files = files[:100]
    print(f"Running sppre on Potsdam IRRG (Total: {len(files)})...")

    for fname in tqdm(files):
        img_path = os.path.join(IMG_DIR, fname)
        mask_name = fname.replace("irrg_", "building_mask_")
        gt_path = os.path.join(MASK_DIR, mask_name)

        if not os.path.exists(gt_path): continue

        # 读取图片 (OpenCV default is BGR)
        # 我们的切片: Ch0=Green, Ch1=Red, Ch2=IR
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt > 127).astype(np.uint8)

        # === 通道重排 (Channel Swapping) ===
        # SpPreModule 在 'vaihingen' 模式下期望: Ch0=NIR, Ch1=Red
        # 当前 image_bgr: Ch0=Green, Ch1=Red, Ch2=IR
        # 所以我们需要把 Ch2 移到 Ch0，Ch1 保持，Ch0 移到后面
        # 目标: [IR, Red, Green]
        image_input = image_bgr.copy()
        image_input[:, :, 0] = image_bgr[:, :, 2]  # NIR -> Ch0
        image_input[:, :, 1] = image_bgr[:, :, 1]  # Red -> Ch1
        image_input[:, :, 2] = image_bgr[:, :, 0]  # Green -> Ch2

        # 运行 SpPre 完整模块
        # 返回值: mask, features, segments, confidence_map
        pred_mask_float, _, _, _ = sppre_module.process(image_input)

        pred_binary = (pred_mask_float > 0.5).astype(np.uint8)

        tracker.update(pred_binary, gt_binary)

    iou, f1 = tracker.get_scores()
    print("\n" + "=" * 50)
    print("Full SpPreModule Evaluation (Potsdam IRRG -> Vaihingen Logic)")
    print("=" * 50)
    print(f"mIoU    : {iou:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()