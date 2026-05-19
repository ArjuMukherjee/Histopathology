import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MoNuSegDataset(Dataset):
    # Inside dataset.py
    def __init__(self, img_dir, mask_dir, target_size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        
        # Updated to find both .tif (common for MoNuSeg images) and .png
        valid_extensions = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
        
        print(f"Dataset initialized with {len(self.images)} images from {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
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

        # 3. CRITICAL FIX: Resize to multiple of 32
        # This prevents the 'tensor a (250) must match tensor b (252)' error
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 4. Processing
        mask = (mask > 0).astype("float32")

        # Convert to Tensors: (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask