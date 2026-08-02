import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MoNuSegDataset(Dataset):
    def __init__(
        self, 
        img_dir, 
        mask_dir, 
        target_size=(512, 512), 
        augment=False, 
        augment_factor=1,
        augment_prob=0.5, # <--- 1. Added argument with a default value (50% chance)
        rotation_range=(-15, 15), 
        scale_range=(0.9, 1.1)
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.augment = augment
        self.augment_factor = augment_factor
        self.augment_probability = augment_prob # <--- Store probability
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        
        valid_extensions = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

        self.samples = []
        for img_name in self.images:
            self.samples.append({"img_name": img_name, "augment": False})
            if self.augment:
                for _ in range(self.augment_factor):
                    self.samples.append({"img_name": img_name, "augment": True})

        if self.augment:
            np.random.shuffle(self.samples)

        print(
            f"Dataset initialized with {len(self.images)} original images from {img_dir} "
            f"and {len(self.samples) - len(self.images)} augmented entries, total entries={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def _augment_pair(self, img, mask):
        angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)

        img = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        mask = cv2.warpAffine(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return img, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample["img_name"]
        should_augment = sample["augment"]
        
        # 1. Image Loading
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Mask Loading
        base_name = os.path.splitext(img_name)[0] 
        mask_name = f"{base_name}_bin_mask.png" 
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, 0)
        
        if mask is None:
            raise FileNotFoundError(f"Could not find mask at {mask_path}")

        # 3. Resize to target size
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 4. Apply augmentation based on probability
        # <--- 2. Modified condition to check against random probability
        if should_augment and (np.random.rand() < self.augment_probability):
            img, mask = self._augment_pair(img, mask)

        # 5. Processing
        mask = (mask > 0).astype("float32")

        # Convert to Tensors: (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask