# --------------------------------------------------------
# Feature Distillation
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yixuan Wei
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision.transforms import Resize
from timm.models.layers import trunc_normal_

# from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .vision_transformer import VisionTransformer
from .clip import load as load_clip
from .dino import load_dino
from .deit import load_deit
from .esvit import load_esvit

import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
mean = torch.tensor(IMAGENET_DEFAULT_MEAN)
std = torch.tensor(IMAGENET_DEFAULT_STD)
normalize = T.Normalize(mean=mean, std=std)
unnormalize = T.Normalize(mean=-mean / std, std=1.0 / std)
normalize_clip = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class VisionTransformerForFD(VisionTransformer):
    def __init__(self, use_checkpoint=False, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        self.use_checkpoint = use_checkpoint

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x):
        x = self.patch_embed(x)

        B, L, _ = x.shape
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        return x


class SwinV2ForFD(SwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        assert self.num_classes == 0
        
    def forward(self, x):
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay()


class FD(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.encoder = encoder

        self.pred_feat = config.DEV.PRED_FEAT
        self.feat_after_norm = config.DEV.PRED_FEAT_AFTERNORM

        # pred target is feature
        if config.DEV.PRED_FEAT == 'CLIP_400M':
            self.feature_model, _ = load_clip("ViT-B/16", image_size=config.DATA.IMG_SIZE)
            self.resize_func = None
            if config.DATA.IMG_SIZE != 224:
                self.resize_func = Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
        elif config.DEV.PRED_FEAT == 'CLIP_400M_Large':
            self.feature_model, _ = load_clip("ViT-L/14", image_size=224)
            self.resize_func = None
            if config.DATA.IMG_SIZE != 224:
                self.resize_func = Resize((224, 224))
        elif config.DEV.PRED_FEAT == 'DINO':
            self.feature_model = load_dino(config.DEV.PRED_FEAT)
        elif config.DEV.PRED_FEAT == 'DEIT':
            self.feature_model = load_deit(config.DEV.PRED_FEAT)
        elif config.DEV.PRED_FEAT == 'ESVIT':
            self.feature_model = load_esvit(config.DEV.PRED_FEAT, return_s3=config.DEV.PRED_FEAT_S3)
        else:
            raise NotImplementedError
        for name, params in self.feature_model.named_parameters():
            params.requires_grad = False

        if 'Large' in config.DEV.PRED_FEAT:
            embed_dim = 1024
        elif 'ESVIT' in config.DEV.PRED_FEAT:
            embed_dim = self.feature_model.embed_dim * 4 if config.DEV.PRED_FEAT_S3 else self.feature_model.embed_dim * 8        
        else:
            embed_dim = 768

        self.loss_feat = nn.SmoothL1Loss(beta=2.0)
        self.ln_tgt = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.encoder.num_features,
                out_channels=embed_dim, kernel_size=1),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x):
        z = self.encoder(x)

        x_rec = self.decoder(z.permute(0,2,1)).permute(0,2,1)
        self.feature_model.eval()
        with torch.no_grad():
            # DINO & DeiT don't unnormalize
            if 'CLIP' in self.pred_feat:
                x = normalize_clip(unnormalize(x))
            if self.pred_feat == 'CLIP_400M_Large' or self.pred_feat == 'CLIP_400M':
                # large as teacher: student: 256/p16 or 224/p14; teacher 224/p14
                if self.resize_func is not None:
                    x = self.resize_func(x)

            x_tgt = self.feature_model.encode_image_featuremap(x)

            if self.feat_after_norm:
                if 'CLIP' in self.pred_feat:
                    x_tgt = self.feature_model.visual.ln_post(x_tgt)
                elif 'DINO' in self.pred_feat or 'DEIT' in self.pred_feat or 'ESVIT' in self.pred_feat:
                    x_tgt = self.feature_model.norm(x_tgt)
                else:
                    raise NotImplementedError
            x_tgt = x_tgt.detach()
            x_tgt = self.ln_tgt(x_tgt)

        loss = self.loss_feat(x_rec, x_tgt)
        loss = loss.mean()
        return {'loss': loss}

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return self.encoder.no_weight_decay_keywords()
        return {}


def build_fd(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin_v2':
        encoder = SwinV2ForFD(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_shared_rel_pos_bias=config.MODEL.SWIN.USE_SHARED_RPB,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'vit':
        encoder = VisionTransformerForFD(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            with_k_bias=config.DEV.VIT_WITHKBIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING,
            with_cls_token=config.MODEL.VIT.WITH_CLS_TOKEN,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = FD(config=config, encoder=encoder)

    return model
