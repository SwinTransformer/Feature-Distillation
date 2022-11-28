# --------------------------------------------------------
# Feature Distillation
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# Modified by Yixuan Wei
# --------------------------------------------------------

from .swin_transformer import build_swin
from .swin_transformer_v2 import build_swin_v2
from .vision_transformer import build_vit
from .feature_distillation import build_fd


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_fd(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin_v2':
            model = build_swin_v2(config)
        elif model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
