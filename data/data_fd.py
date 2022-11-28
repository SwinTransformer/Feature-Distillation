# --------------------------------------------------------
# Feature Distillation
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# Modified by Yixuan Wei
# --------------------------------------------------------

import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .utils import SubsetRandomSampler
from .cached_image_folder import CachedImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FDTransform:
    def __init__(self, config):
        self.config = config
        
        crop_size = config.DATA.IMG_SIZE
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(crop_size, scale=(config.AUG.MIN_SCALE, config.AUG.MAX_SCALE), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
        
    def __call__(self, img):
        img = self.transform_img(img)
        return img


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


def build_loader_fd(config, logger):
    transform = FDTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    if config.DATA.DATASET == 'imagenet':
        prefix = 'train'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode='part')
        else:
            dataset = ImageFolder(config.DATA.DATA_PATH, transform, is_valid_file=is_valid_file)
    elif config.DATA.DATASET == 'imagenet22k':
        dataset = ImageFolder(config.DATA.DATA_PATH, transform, is_valid_file=is_valid_file)

    if config.DATA.DATASET == 'imagenet' and config.DATA.ZIP_MODE:
        indices = np.arange(dist.get_rank(), len(dataset), dist.get_world_size())
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    return dataloader