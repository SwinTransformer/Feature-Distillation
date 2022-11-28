# --------------------------------------------------------
# Feature Distillation
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# Modified by Yixuan Wei
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import numpy as np
import torchvision.transforms as T
from scipy import interpolate
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

mean = torch.tensor(IMAGENET_DEFAULT_MEAN)
std = torch.tensor(IMAGENET_DEFAULT_STD)
normalize = T.Normalize(mean=mean, std=std)
unnormalize = T.Normalize(mean=-mean / std, std=1.0 / std)


def load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger):
    logger.info(f">>>>>>>>>> Resuming from {config.MODEL.RESUME} ..........")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)

    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'scaler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info('Load Lr Scheduler')
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])

        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()

        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        else:
            max_accuracy = 0.0

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = []
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        # total_norm += param_norm.item() ** norm_type
        total_norm.append(param_norm)
    # total_norm = total_norm ** (1. / norm_type)
    total_norm = torch.stack(total_norm).norm(norm_type).item()
    return total_norm


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_pretrained(config, model, logger):
    logger.info(f">>>>>>>>>> Fine-tuned from {config.PRETRAINED} ..........")
    checkpoint = torch.load(config.PRETRAINED, map_location='cpu')
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    checkpoint_model = checkpoint_model['student'] if 'student' in checkpoint_model else checkpoint_model  # for esvit

    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        logger.info('Detect pre-trained model, remove [encoder.] prefix.')
    elif any([True if 'module.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint_model.items() if k.startswith('module.')}
        logger.info('Detect pre-trained model, remove [module.] prefix.')
    else:
        logger.info('Detect non-pre-trained model, pass without doing anything.')

    if config.DEV.FT_SKIP_REMAP:
        logger.info(f">>>>>>>>>> Skip remapping when loading pre-trained model")
    else:
        if config.MODEL.TYPE in ['swin', 'swin_v2']:
            logger.info(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
            checkpoint_model = remap_pretrained_keys_swin(model, checkpoint_model, logger)
        elif config.MODEL.TYPE in ['vit']:
            logger.info(f">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
            checkpoint_model = remap_pretrained_keys_vit(model, checkpoint_model, logger)
        else:
            raise NotImplementedError

    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)

    del checkpoint
    del checkpoint_model
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{config.PRETRAINED}'")


def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    # Duplicate shared rel_pos_bias to each layer
    if "layers.0.rel_pos_bias.relative_coords_table" in checkpoint_model:
        # only support swinv2
        logger.info("Expand the shared relative position embedding to each transformer block.")
        for l in range(model.num_layers):
            # relative_coords_table = checkpoint_model.pop(f"layers.{l}.rel_pos_bias.relative_coords_table")
            # relative_position_index = checkpoint_model.pop(f"layers.{l}.rel_pos_bias.relative_position_index")
            mlp0weight = checkpoint_model.pop(f"layers.{l}.rel_pos_bias.rpe_mlp.0.weight")
            mlp0bias = checkpoint_model.pop(f"layers.{l}.rel_pos_bias.rpe_mlp.0.bias")
            mlp2bias = checkpoint_model.pop(f"layers.{l}.rel_pos_bias.rpe_mlp.2.weight")
            for i in range(model.depths[l]):
                # checkpoint_model[f"layers.{l}.blocks.{i}.attn.relative_coords_table"] = relative_coords_table.clone()
                # checkpoint_model[f"layers.{l}.blocks.{i}.attn.relative_position_index"] = relative_position_index.clone()
                checkpoint_model[f"layers.{l}.blocks.{i}.attn.rpe_mlp.0.weight"] = mlp0weight.clone()
                checkpoint_model[f"layers.{l}.blocks.{i}.attn.rpe_mlp.0.bias"] = mlp0bias.clone()
                checkpoint_model[f"layers.{l}.blocks.{i}.attn.rpe_mlp.2.weight"] = mlp2bias.clone()

    state_dict = model.state_dict()

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def remap_pretrained_keys_vit(model, checkpoint_model, logger, rpe_method=None):
    # Duplicate shared rel_pos_bias to each layer
    if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        logger.info("Expand the shared relative position embedding to each transformer block.")
    num_layers = model.get_num_layers()
    if "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            if key not in model.state_dict():
                # case for additional encoder block
                continue
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                logger.info("Position interpolate for %s from %dx%d to %dx%d" % (key, src_size, src_size, dst_size, dst_size))
                if rpe_method == 'outer_mask':
                    pad_size = (dst_size - src_size) // 2
                    padding = (pad_size, pad_size, pad_size, pad_size)

                    all_rel_pos_bias = []
                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size)
                        all_rel_pos_bias.append(
                            torch.nn.functional.pad(z, padding, "constant", z.min().item() - 3).view(dst_num_pos, 1))
                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    checkpoint_model[key] = new_rel_pos_bias
                else:
                    if num_extra_tokens > 0:
                        extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                        rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
                    else:
                        extra_tokens = rel_pos_bias.new_zeros((0, num_attn_heads))

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias

    if 'pos_embed' in all_keys and getattr(model, 'pos_embed', None) is not None:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            logger.info("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            if num_extra_tokens > 0:
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            else:
                new_pos_embed = pos_tokens
            checkpoint_model['pos_embed'] = new_pos_embed

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    return checkpoint_model


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# LARS optimizer, implementation from MoCo v3:
# https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])