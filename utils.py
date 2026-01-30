import os
import random
import numpy as np
import torch
import cv2
from scipy.ndimage import binary_dilation


def seed_everything(seed=42):
    """Ensure deterministic reproducibility as stated in Paper IV.C"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. Edge Extraction (Paper III.E)
# ==========================================
def get_canny_edge(img_tensor, low_threshold=100, high_threshold=200):
    """
    Extract Canny edges from image tensor for Edge-Aware Loss.
    Args:
        img_tensor: (B, 3, H, W) normalized float tensor [0, 1]
    Returns:
        edge_tensor: (B, 1, H, W) binary edge map {0, 1}
    """
    device = img_tensor.device
    edges = []

    # Convert to numpy for OpenCV
    imgs_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)

    for i in range(imgs_np.shape[0]):
        # Transpose to (H, W, 3) and RGB -> Gray
        img = np.transpose(imgs_np[i], (1, 2, 0))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Canny
        edge = cv2.Canny(gray, low_threshold, high_threshold)
        edge = (edge > 0).astype(np.float32)
        edges.append(edge)

    edges = np.stack(edges, axis=0)  # (B, H, W)
    edges = torch.from_numpy(edges).unsqueeze(1).to(device)  # (B, 1, H, W)
    return edges


# ==========================================
# 2. Metrics: mIoU, F1, BoundF (Paper IV.B)
# ==========================================
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        # For BoundF
        self.bound_tp = 0
        self.bound_fp = 0
        self.bound_fn = 0

    def update(self, pred_mask, gt_mask):
        """
        pred_mask, gt_mask: (H, W) numpy arrays, 0 or 1
        """
        # Semantic Metrics
        self.tp += np.sum((pred_mask == 1) & (gt_mask == 1))
        self.fp += np.sum((pred_mask == 1) & (gt_mask == 0))
        self.fn += np.sum((pred_mask == 0) & (gt_mask == 1))

        # Boundary Metrics (Paper Eq. 20)
        # "Boundaries are extracted using a 3-pixel morphological dilation"
        # "matching within a 5-pixel tolerance zone"

        pred_bound = self._get_boundary(pred_mask)
        gt_bound = self._get_boundary(gt_mask)

        # Calculate matching with tolerance
        # Use distance transform to find distance from pred boundary to nearest gt boundary
        if np.sum(gt_bound) > 0:
            gt_dist = cv2.distanceTransform(1 - gt_bound, cv2.DIST_L2, 5)
            # Precision: pixels in pred_bound that are close to gt_bound
            tp_precision = np.sum(pred_bound & (gt_dist < 5))  # 5-pixel tolerance
        else:
            tp_precision = 0

        if np.sum(pred_bound) > 0:
            pred_dist = cv2.distanceTransform(1 - pred_bound, cv2.DIST_L2, 5)
            # Recall: pixels in gt_bound that are close to pred_bound
            tp_recall = np.sum(gt_bound & (pred_dist < 5))
        else:
            tp_recall = 0

        self.bound_tp += tp_precision  # Note: This is an approximation for accumulation
        # For strict F1, we usually sum Precision and Recall separately
        self._accumulate_boundary(pred_bound, gt_bound)

    def _get_boundary(self, mask):
        """Extract boundary via 3-pixel dilation as stated in paper"""
        mask = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(mask, kernel)
        boundary = dilated - mask
        return boundary

    def _accumulate_boundary(self, pred_bound, gt_bound):
        # We store raw counts for final calculation
        # Precision = (Pred_bound & Near_GT) / Pred_bound
        # Recall = (GT_bound & Near_Pred) / GT_bound

        if np.sum(gt_bound) == 0 and np.sum(pred_bound) == 0:
            return

        # Tolerance checking logic
        # 1. Precision Part
        if np.sum(gt_bound) > 0:
            gt_dist = cv2.distanceTransform(1 - gt_bound, cv2.DIST_L2, 5)
            self.bound_tp_prec = getattr(self, 'bound_tp_prec', 0) + np.sum(pred_bound & (gt_dist < 5))
        self.bound_total_pred = getattr(self, 'bound_total_pred', 0) + np.sum(pred_bound)

        # 2. Recall Part
        if np.sum(pred_bound) > 0:
            pred_dist = cv2.distanceTransform(1 - pred_bound, cv2.DIST_L2, 5)
            self.bound_tp_recall = getattr(self, 'bound_tp_recall', 0) + np.sum(gt_bound & (pred_dist < 5))
        self.bound_total_gt = getattr(self, 'bound_total_gt', 0) + np.sum(gt_bound)

    def get_scores(self):
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-8)

        # Boundary F1
        b_prec = getattr(self, 'bound_tp_prec', 0) / (getattr(self, 'bound_total_pred', 0) + 1e-8)
        b_rec = getattr(self, 'bound_tp_recall', 0) / (getattr(self, 'bound_total_gt', 0) + 1e-8)
        bound_f1 = 2 * b_prec * b_rec / (b_prec + b_rec + 1e-8)

        return {
            "mIoU": iou * 100,
            "F1": f1 * 100,
            "BoundF": bound_f1 * 100
        }