import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.morphology import closing, square, opening
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SpPreModule:
    def __init__(self, dataset_type='potsdam', rho_fine=100, rho_coarse=400, tau=0.35, compactness=20.0,
                 sigma_scale=0.2):
        self.dataset_type = dataset_type.lower()
        self.rho_fine = rho_fine
        self.rho_coarse = rho_coarse
        self.tau = tau  # Initial tau, will be adaptive for Potsdam
        self.compactness = compactness
        self.sigma_scale = sigma_scale

    def _get_num_segments(self, height, width, rho):
        return max(2, int(np.floor((height * width) / rho)))

    def _find_valley_threshold(self, values, bins=64, sigma=2.0, fallback=None):
        """
        Implements the Histogram Valley Detection Method.
        1. Computes histogram.
        2. Smooths it with a Gaussian kernel.
        3. Finds peaks and identifies the deepest valley between the two dominant peaks.
        """
        # Remove NaNs and flatten
        v_flat = values.flatten()
        v_flat = v_flat[~np.isnan(v_flat)]

        if len(v_flat) == 0:
            return fallback if fallback is not None else 0.5

        # Compute histogram
        hist, bin_edges = np.histogram(v_flat, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth histogram to remove noise
        hist_smooth = gaussian_filter1d(hist, sigma=sigma)

        # Find peaks
        peaks, properties = find_peaks(hist_smooth, prominence=0.01)

        # If we have at least 2 peaks (Bimodal distribution)
        if len(peaks) >= 2:
            # Sort peaks by height and take top 2
            sorted_indices = np.argsort(hist_smooth[peaks])[::-1]
            peak1_idx = peaks[sorted_indices[0]]
            peak2_idx = peaks[sorted_indices[1]]

            # Ensure peak1 < peak2 for slicing
            idx_min = min(peak1_idx, peak2_idx)
            idx_max = max(peak1_idx, peak2_idx)

            # Find the minimum (valley) between the two peaks
            valley_idx = np.argmin(hist_smooth[idx_min:idx_max]) + idx_min
            threshold = bin_centers[valley_idx]
            return threshold

        # Fallback: If distribution is unimodal or irregular, use Otsu's approximation
        # or the provided fallback (e.g., mean)
        if fallback is not None:
            return fallback
        else:
            # Simple Otsu-like approximation using mean
            return np.mean(v_flat)

    def _extract_features(self, image, segments):
        features = []
        if self.dataset_type == 'vaihingen':
            gray = image[:, :, 0]  # NIR as intensity
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gray_uint8 = (gray * 255).astype(np.uint8)
        # Ensure image is uint8 for LBP
        if gray_uint8.max() <= 1:
            gray_uint8 = (gray * 255).astype(np.uint8)

        lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')
        edge_map = sobel(gray)

        h, w = image.shape[:2]
        y_grid, x_grid = np.mgrid[:h, :w]
        x_grid, y_grid = x_grid.astype(np.float32) / w, y_grid.astype(np.float32) / h

        unique_labels = np.unique(segments)

        for label in unique_labels:
            mask = (segments == label)
            if not np.any(mask): continue

            mean_rgb = np.mean(image[mask], axis=0)
            lbp_region = lbp[mask]
            hist, _ = np.histogram(lbp_region, bins=np.arange(0, 11), density=True)
            hist = hist[hist > 0]
            texture_entropy = -np.sum(hist * np.log2(hist))
            mean_edge = np.mean(edge_map[mask])
            cx, cy = np.mean(x_grid[mask]), np.mean(y_grid[mask])

            feat = np.concatenate([mean_rgb, [texture_entropy, mean_edge, cx, cy]])
            features.append(feat)

        return np.array(features, dtype=np.float32), unique_labels

    def process(self, image_input):
        # 1. Normalize
        if image_input.dtype == np.uint8:
            image = image_input.astype(np.float32) / 255.0
        else:
            image = image_input.astype(np.float32)
            if image.max() > 1.1: image /= 255.0

        h, w = image.shape[:2]

        # 2. SLIC
        n_fine = self._get_num_segments(h, w, self.rho_fine)
        n_coarse = self._get_num_segments(h, w, self.rho_coarse)
        segments_fine = slic(image, n_segments=n_fine, compactness=self.compactness, start_label=0,
                             enforce_connectivity=True)
        segments_coarse = slic(image, n_segments=n_coarse, compactness=self.compactness, start_label=0,
                               enforce_connectivity=True)

        if len(np.unique(segments_fine)) < 2:
            return np.zeros((h, w), dtype=np.float32), None, None, np.zeros((h, w), dtype=np.float32)

        # 3. Features & Voting
        c_fine, l_fine = self._extract_features(image, segments_fine)
        c_coarse, l_coarse = self._extract_features(image, segments_coarse)

        dists = cdist(c_fine, c_coarse, metric='sqeuclidean')
        min_dists = np.min(dists, axis=1)
        sigma2 = np.median(min_dists) * self.sigma_scale
        if sigma2 <= 1e-8: sigma2 = 1e-5

        W = np.exp(-dists / sigma2)
        scores_fine = np.max(W / (np.sum(W, axis=1, keepdims=True) + 1e-8), axis=1)

        score_lookup = np.zeros(np.max(segments_fine) + 1)
        score_lookup[l_fine] = scores_fine

        # === Confidence Map (Consensus) ===
        w_p_map = score_lookup[segments_fine]

        # 4. Domain-Specific Thresholding (Sensor-Adaptive)

        if self.dataset_type == 'vaihingen':
            # === Physical Prior Branch (NDVI) ===
            # Input convention: Ch0=NIR, Ch1=Red, Ch2=Green
            NIR, Red = image[:, :, 0], image[:, :, 1]
            ndvi = (NIR - Red) / (NIR + Red + 1e-8)

            # Use fixed physical thresholds for vegetation
            is_vegetation = ndvi > 0.1
            is_shadow = NIR < 0.15

            # Consensus using fixed tau
            consensus_mask = w_p_map >= self.tau

            final_mask = consensus_mask & (~is_vegetation) & (~is_shadow)

        elif self.dataset_type == 'potsdam':
            # === Statistical Prior Branch (Histogram Valley) ===
            # Input convention: RGB
            R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

            # A. Dynamic Texture Confidence Threshold
            # Find valley in the confidence map (w_p_map) distribution
            # Fallback to self.tau if valley not found
            tau_dynamic = self._find_valley_threshold(w_p_map, bins=50, sigma=1.5, fallback=self.tau)
            consensus_mask = w_p_map >= tau_dynamic

            # B. Dynamic Saturation Threshold
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            Saturation = hsv[:, :, 1]

            # Find valley in Saturation histogram
            # Impervious surfaces (Roads/Buildings) usually have different saturation than Vegetation
            # But specifically, gray roads vs red roofs have different saturation.
            # Low saturation = Gray/White (Roads/Concrete)
            # High saturation = Red Roofs / Vegetation
            s_thresh = self._find_valley_threshold(Saturation, bins=50, sigma=2.0, fallback=0.12)

            # Assuming roads are the low saturation component
            is_road_gray = Saturation < s_thresh

            # C. Other Priors (Keep ExG for Vegetation as it's robust)
            ExG = 2 * G - R - B
            is_vegetation = ExG > 0.05

            intensity = np.mean(image, axis=2)
            is_shadow = intensity < 0.12

            # D. Red Roof Boosting (Heuristic based on color ratio, relatively stable)
            is_red_roof = (R > G * 1.1) & (R > B * 1.05) & (R > 0.2)

            # Combine
            refined_base = consensus_mask & (~is_vegetation) & (~is_shadow) & (~is_road_gray)
            final_mask = refined_base | (is_red_roof & (~is_vegetation))

        else:
            # Default / Fallback
            consensus_mask = w_p_map >= self.tau
            final_mask = consensus_mask

        # 5. Morphology
        final_mask = opening(final_mask, square(2))
        final_mask = closing(final_mask, square(3))

        # Return: Mask, Features, Segments, ConfidenceMap
        return final_mask.astype(np.float32), c_fine, segments_fine, w_p_map