import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _apply_transform(geo, norm, to_t, image, mask, features):
    """
    辅助函数：分离几何变换与光度变换，防止特征图被错误归一化
    """
    # 1. 第一步：几何变换 (同步应用)
    # features 被注册为 'image' 类型以获得相同的几何变换
    augmented = geo(image=image, mask=mask, features=features)
    img_aug = augmented['image']
    mask_aug = augmented['mask']
    feat_aug = augmented['features']

    # 2. 第二步：分别处理
    # A. Image: Normalize + ToTensor (HWC -> CHW, 0-1 float)
    img_final = norm(image=img_aug)['image']

    # B. Features: 仅 ToTensor (HWC -> CHW, 保持原值), 不做 Normalize
    # 我们借用 'image' 接口来转换 tensor
    feat_final = to_t(image=feat_aug)['image']

    return {'image': img_final, 'mask': mask_aug, 'features': feat_final}


def get_train_transform(config):
    """
    构建训练增强管道：
    Paper IV.C: "random 90-degree rotations, vertical and horizontal flips, and Gaussian blur"
    """
    aug_conf = config['train']['aug']

    # 1. 几何变换配置 (Geometric)
    geometric_aug = A.Compose([
        A.HorizontalFlip(p=aug_conf['prob_flip']),
        A.VerticalFlip(p=aug_conf['prob_flip']),
        A.RandomRotate90(p=aug_conf['prob_rotate']),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=aug_conf['gaussian_blur_sigma'], p=0.5),
    ], additional_targets={'features': 'image'})  # 关键：让 features 跟随 image 变换

    # 2. 光度变换配置 (Photometric) - 仅 RGB
    normalize = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 3. 特征转 Tensor 配置
    to_tensor = ToTensorV2()

    # 返回一个 lambda，将这三个步骤串联起来
    return lambda image, mask, features: _apply_transform(
        geometric_aug, normalize, to_tensor, image, mask, features
    )


class DBCDDataset(Dataset):
    def __init__(self, data_dir, list_file, transform=None, is_train=True):
        """
        Args:
            data_dir: Path to 'processed_data' containing 'images', 'masks', and 'features'
            list_file: Path to 'train_list.txt'
            transform: Transform function (usually from get_train_transform)
        """
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.feat_dir = os.path.join(data_dir, 'features')

        with open(list_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]

        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # 假设特征文件以 .npy 结尾 (make_patches.py 生成的)
        # 如果 fname 已经是 "image_x_y.png"，则寻找 "image_x_y.png.npy"
        # 或者 "image_x_y.npy"，视 make_patches 实现而定。
        # 这里为了兼容性，假设是 fname + ".npy"
        feat_path = os.path.join(self.feat_dir, fname + ".npy")

        # 1. Read Image (BGR -> RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Read Mask (0-255 -> 0-1)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        # 3. Read Features (H, W, C)
        if os.path.exists(feat_path):
            feat_map = np.load(feat_path).astype(np.float32)
        else:
            # 如果特征文件缺失 (例如推理阶段或数据未准备好)，生成零矩阵占位
            # 假设 SpPre 特征维度为 5 (RGB mean + texture + edge + coords)
            # 这里的维度需要与 make_patches.py 中保存的一致
            feat_map = np.zeros((image.shape[0], image.shape[1], 5), dtype=np.float32)

        # 4. Apply Transform
        if self.transform:
            # 调用我们自定义的 transform lambda
            augmented = self.transform(image=image, mask=mask, features=feat_map)
            image = augmented['image']  # Tensor (C, H, W)
            mask = augmented['mask']  # Numpy (H, W) or Tensor
            feat_map = augmented['features']  # Tensor (C, H, W)
        else:
            # 如果没有 transform (例如验证集)，手动转换为 Tensor
            # Image: Normalize + ToTensor
            norm = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            image = norm(image=image)['image']

            # Features: ToTensor only
            to_t = ToTensorV2()
            feat_map = to_t(image=feat_map)['image']

        # 5. Finalize Mask (Ensure Tensor 1xHxW)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)

        return image, mask, feat_map