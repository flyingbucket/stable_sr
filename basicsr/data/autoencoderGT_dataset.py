import os
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class AutoencoderImageDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt['gt_path']
        self.image_type = opt.get('image_type', 'png')
        self.color_mode = opt.get('color', 'gray')  # 'gray' or 'rgb'
        self.paths = sorted(Path(self.root).glob(f'*.{self.image_type}'))
        assert len(self.paths) > 0, f"No images found in {self.root} with type {self.image_type}"

    def __getitem__(self, index):
        path = str(self.paths[index])
        if self.color_mode == 'gray':
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape [H, W]
            img = img[..., None]  # [H, W, 1]
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR → RGB

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [C, H, W]
        return {'gt': img, 'gt_path': path}

    def __len__(self):
        return len(self.paths)
