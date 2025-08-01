import os
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


# class AutoencoderImageDataset(Dataset):
#     def __init__(self, opt):
#         self.opt = opt
#         self.root = opt['gt_path']
#         self.image_type = opt.get('image_type', 'png')
#         self.color_mode = opt.get('color', 'gray')  # 'gray' or 'rgb'
#         self.paths = sorted(Path(self.root).glob(f'*.{self.image_type}'))
#         assert len(self.paths) > 0, f"No images found in {self.root} with type {self.image_type}"

#     def __getitem__(self, index):
#         path = str(self.paths[index])
#         if self.color_mode == 'gray':
#             img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape [H, W]
#             img = img[..., None]  # [H, W, 1]
#         else:
#             img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR → RGB

#         img = img.astype(np.float32) / 255.0
#         img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [C, H, W]
#         return {'gt': img, 'gt_path': path}

#     def __len__(self):
#         return len(self.paths)

# class AutoencoderImageDataset(Dataset):
#     def __init__(self, opt):
#         self.opt = opt
#         # self.image_type = opt.get('image_type', 'png')
#         self.color_mode = opt.get('color', 'gray')  # 'gray' or 'rgb'
#         self.resolution = opt.get('resolution', None)

#         suffixes = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']
#         gt_paths = opt['gt_path']
#         if isinstance(gt_paths, (str, Path)):
#             gt_paths = [gt_paths]  # 单路径也包装成列表

#         all_paths = []
#         for p in gt_paths:
#             p = Path(p)
#             for suffix in suffixes:
#                 all_paths.extend(sorted(p.glob(f'*.{suffix}')))
        
#         assert len(all_paths) > 0, f"No images found in {gt_paths} with type {self.image_type}"

#         self.paths = all_paths

#     def __getitem__(self, index):
#         path = str(self.paths[index])
#         if self.color_mode == 'gray':
#             img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape [H, W]
#             img = img[..., None]  # [H, W, 1]
#         else:
#             img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR → RGB

#         img = img.astype(np.float32) / 255.0
#         img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [C, H, W]
#         return {'gt': img, 'gt_path': path}

#     def __len__(self):
#         return len(self.paths)
class AutoencoderImageDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root_paths = opt['gt_path']
        if isinstance(self.root_paths, (str, Path)):
            self.root_paths = [self.root_paths]
        self.image_type = opt.get('image_type', None)
        self.color_mode = opt.get('color', 'gray')
        self.resolution = opt.get('resolution', None)

        # 扫描所有图片（支持多路径）
        self.paths = []
        for root in self.root_paths:
            p = Path(root)
            if self.image_type:
                self.paths += sorted(p.rglob(f'*.{self.image_type}'))
            else:
                self.paths += sorted(p.rglob('*'))
        # 过滤非图像文件
        self.paths = [x for x in self.paths if x.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']]
        assert len(self.paths) > 0, f"No images found in {self.root_paths}"

    def __len__(self):
        return len(self.paths)

    def center_crop(self, img, res):
        h, w = img.shape[:2]
        top = (h - res) // 2
        left = (w - res) // 2
        return img[top:top+res, left:left+res]

    def __getitem__(self, index):
        tries = 0
        while tries < 10:
            path = str(self.paths[index])
            try:
                if self.color_mode == 'gray':
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError("Image is None")
                    img = img[..., None]
                else:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1]
                    if img is None:
                        raise ValueError("Image is None")

                h, w = img.shape[:2]
                if self.resolution is not None:
                    if h < self.resolution or w < self.resolution:
                        # skip too small
                        raise ValueError(f"Image too small: {img.shape}")
                    if h > self.resolution or w > self.resolution:
                        img = self.center_crop(img, self.resolution)

                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
                return {'gt': img, 'gt_path': path}

            except Exception as e:
                # 读图失败或不符合要求，跳下一个
                tries += 1
                index = (index + 1) % len(self.paths)
        raise RuntimeError(f"Too many bad images starting from index {index}")
