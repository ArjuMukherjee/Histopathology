import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def list_image_files(img_dir, extensions=None):
    if extensions is None:
        extensions = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
    return sorted(
        [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith(extensions)
        ]
    )


def split_image_list(image_files, test_size=0.1, val_size=0.1, random_state=None):
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1.0:
        raise ValueError("val_size and test_size must be non-negative and sum to less than 1.0")

    train_val, test_files = train_test_split(
        image_files,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    if val_size == 0.0:
        return train_val, [], test_files

    val_fraction = val_size / (1.0 - test_size)
    train_files, val_files = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True,
    )

    return train_files, val_files, test_files


class MoNuSegDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        target_size=(512, 512),
        augment=False,
        augment_factor=1,
        augment_prob=0.5,
        rotation_range=(-15, 15),
        scale_range=(0.9, 1.1),
        image_list=None,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = tuple(target_size)
        self.augment = augment
        self.augment_factor = augment_factor
        self.augment_probability = augment_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range

        valid_extensions = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
        if image_list is None:
            self.images = list_image_files(img_dir, extensions=valid_extensions)
        else:
            self.images = [f for f in image_list if f.lower().endswith(valid_extensions)]

        self.samples = []
        for img_name in self.images:
            self.samples.append({"img_name": img_name, "augment": False})
            if self.augment:
                for _ in range(self.augment_factor):
                    self.samples.append({"img_name": img_name, "augment": True})

        if self.augment:
            np.random.shuffle(self.samples)

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

        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_bin_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                raise FileNotFoundError(f"Could not find mask for {img_name} at {self.mask_dir}")

        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        if should_augment and (np.random.rand() < self.augment_probability):
            img, mask = self._augment_pair(img, mask)

        # Ensure contiguous memory before tensor conversion
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        mask = (mask > 0).astype("float32")
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return img_tensor, mask_tensor