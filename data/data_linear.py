# --------------------------------------------------------
# Feature Distillation
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# Modified by Yixuan Wei
# --------------------------------------------------------

import os
import math
import torch
import torch.distributed as dist
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension
from timm.data.transforms import _pil_interp

import numpy as np
from .cached_image_folder import CachedImageFolder
from .utils import SubsetRandomSampler
IMAGENET_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_loader(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, logger=logger)
    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config, logger=logger)
    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE:
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.DATA.ZIP_MODE:
        indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
        sampler_val = SubsetRandomSampler(indices)
    else:
        sampler_val = DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    mixup_fn = None

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def is_valid_file(x: str) -> bool:
    unvalid_file_list = """
        n01678043_6448.JPEG
        n01896844_997.JPEG
        n02368116_318.JPEG
        n02428089_710.JPEG
        n02487347_1956.JPEG
        n02597972_5463.JPEG
        n03957420_30695.JPEG
        n03957420_33553.JPEG
        n03957420_8296.JPEG
        n04135315_8814.JPEG
        n04135315_9318.JPEG
        n04257684_9033.JPEG
        n04427559_2974.JPEG
        n06470073_47249.JPEG
        n07930062_4147.JPEG
        n09224725_3995.JPEG
        n09359803_8155.JPEG
        n09620794_5529.JPEG
        n09789566_3522.JPEG
        n09894445_7463.JPEG
        n10175248_583.JPEG
        n10316360_4246.JPEG
        n10368624_12550.JPEG
        n10585217_8484.JPEG
        n10721819_1131.JPEG
        n12353203_3849.JPEG
        n12630763_8018.JPEG
    """
    unvalid_file_list = tuple([i.strip() for i in unvalid_file_list.split('\n') if len(i.strip()) > 0])
    assert len(unvalid_file_list) == 27
    
    return has_file_allowed_extension(x, IMG_EXTENSIONS) and not x.endswith(unvalid_file_list)


def build_dataset(is_train, config, logger):
    transform = build_transform(is_train, config)
    logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')
    
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode='part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F._get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # linear probe: weak augmentation
        transform = transforms.Compose([
            RandomResizedCrop(config.DATA.IMG_SIZE, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform

    t = []
    if resize_im:
        if True: # config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_CLIP_MEAN, IMAGENET_CLIP_STD))
    return transforms.Compose(t)