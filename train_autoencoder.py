#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Autoencoder/VAE trainer, compatible with PyTorch-Lightning 1.4.x
No dependency on ldm.data.*  -- we roll our own ImageFolderDataModule.
"""

import argparse, os, glob, random
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from omegaconf import OmegaConf

# StableSR / latent-diffusion util (for AutoencoderKLPlus etc.)
from ldm.util import instantiate_from_config

# ---------- Dataset ---------- #
class ImageFolderDataset(Dataset):
    def __init__(self,
                 root: str,
                 size: int = 256,
                 random_crop: bool = True,
                 exts: Optional[List[str]] = None):
        exts = exts or ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
        self.paths = []
        for e in exts:
            self.paths.extend(glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images with extensions {exts} found in {root}")

        self.random_crop = random_crop
        self.size = size
        self.transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size) if not random_crop else T.RandomCrop(size),
            T.ToTensor(),            # [0,1]
            # normalize to [-1,1] (ldm/StableSR uses this)
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {"image": img}

# ---------- DataModule ---------- #
class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_root: str,
                 val_root: str,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 size: int = 256,
                 random_crop: bool = True):
        super().__init__()
        self.save_hyperparameters()

    # PL 1.4.x hooks
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = ImageFolderDataset(
                self.hparams.train_root,
                size=self.hparams.size,
                random_crop=self.hparams.random_crop,
            )
            self.val_ds = ImageFolderDataset(
                self.hparams.val_root,
                size=self.hparams.size,
                random_crop=False,
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=min(8, self.hparams.batch_size),
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)

# ---------- Helpers ---------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="YAML path for AutoencoderKLPlus")
    ap.add_argument("-n", "--name", default="autoencoder")
    ap.add_argument("--train_dir", required=False, help="Override train image directory")
    ap.add_argument("--val_dir", required=False, help="Override val image directory")
    ap.add_argument("--resume", default="", help="Checkpoint")
    return ap.parse_args()

def map_trainer_cfg(trainer_cfg: dict) -> dict:
    out = trainer_cfg.copy()
    if "devices" in out and "gpus" not in out:
        dev = out.pop("devices")
        out["gpus"] = dev if isinstance(dev, (list, tuple)) else int(dev)
    if out.get("accelerator", None) == "gpu":
        out.pop("accelerator")
    if "strategy" in out:
        out.pop("strategy")
    return out

def build_trainer(trainer_cfg, callbacks_cfg, resume_ckpt: Optional[str], find_unused=False):
    callbacks = []
    if callbacks_cfg:
        for _, cb_conf in callbacks_cfg.items():
            callbacks.append(instantiate_from_config(cb_conf))

    plugin = None
    if trainer_cfg.get("gpus", 0):
        ngpu = trainer_cfg["gpus"] if isinstance(trainer_cfg["gpus"], int) else len(trainer_cfg["gpus"])
        if ngpu > 1:
            plugin = DDPPlugin(find_unused_parameters=find_unused)

    return pl.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        plugins=[plugin] if plugin else None,
        resume_from_checkpoint=resume_ckpt
    )

# ---------- Main ---------- #
def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    print(f"✔ YAML loaded: {args.config}")

    # ---------- Model ----------
    model = instantiate_from_config(cfg.model)

    # ---------- Data ----------
    dat_cfg = cfg.data.params  # get inner params
    train_root = args.train_dir or dat_cfg.train.params.paths
    val_root   = args.val_dir   or dat_cfg.validation.params.paths
    size       = dat_cfg.train.params.size
    batch_size = cfg.data.params.batch_size
    num_workers= cfg.data.params.num_workers
    random_crop= dat_cfg.train.params.random_crop

    datamodule = ImageFolderDataModule(
        train_root=train_root,
        val_root=val_root,
        batch_size=batch_size,
        num_workers=num_workers,
        size=size,
        random_crop=random_crop,
    )

    # ---------- Trainer ----------
    trainer_cfg = map_trainer_cfg(OmegaConf.to_container(cfg.lightning.trainer, resolve=True))
    callbacks_cfg = cfg.lightning.get("callbacks", {})
    logdir = Path("logs") / args.name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"✔ Logs → {logdir}")

    trainer = build_trainer(trainer_cfg,
                            callbacks_cfg,
                            resume_ckpt=args.resume or None,
                            find_unused=cfg.lightning.get("find_unused_parameters", False))

    # ---------- Fit ----------
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
