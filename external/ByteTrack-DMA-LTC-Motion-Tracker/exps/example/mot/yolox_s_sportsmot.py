# encoding: utf-8
"""YOLOX-s trained and evaluated on SportsMOT using Table 2 hyperparameters."""

import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from yolox_s_mot17_half import Exp as SHalfExp  # noqa: E402


class Exp(SHalfExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = os.environ.get("SPORTSMOT_DIR") or os.path.join(
            get_yolox_datadir(), "sportmot"
        )
        self.train_ann = os.environ.get("SPORTSMOT_TRAIN_ANN") or "train_filtered.json"
        self.val_ann = os.environ.get("SPORTSMOT_VAL_ANN") or "val_filtered.json"

        max_epoch_env = os.environ.get("YOLOX_MAX_EPOCH")
        self.max_epoch = int(max_epoch_env) if max_epoch_env and str(max_epoch_env).strip() else 100

        workers_env = os.environ.get("YOLOX_WORKERS")
        if workers_env and str(workers_env).strip():
            self.data_num_workers = int(workers_env)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MOTDataset,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler,
        )

        mosaic_prob = self.mosaic_prob

        class ProbMosaicDetection(MosaicDetection):
            """MosaicDetection that applies mosaic with probability `mosaic_prob`."""

            def __getitem__(self, index):
                if isinstance(index, (tuple, list)) and index[2] and random.random() >= mosaic_prob:
                    img, label, _, _ = MosaicDetection.__getitem__(self, (index[0], index[1], False))
                    return img, label, (img.shape[1], img.shape[0]), np.array([int(index[1])])
                return MosaicDetection.__getitem__(self, index)

        dataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = ProbMosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
        }
        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        split_name = "test" if self.val_ann == "test.json" else "val"
        valdataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name=split_name,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
