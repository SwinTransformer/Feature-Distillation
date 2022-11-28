import os
from typing import Union
import torch
from .clip.clip import _download
from .swin_transformer import SwinTransformer
import torch.distributed as dist
_MODELS = {
    "ESVIT": "https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_nodes4_gpu16_bs8_multicrop_epoch300_dino_aug_window14_lv/continued_from_epoch0200_dense_norm_true/checkpoint_best.pth",
}

class WarpperSwinTransformer(SwinTransformer):
    def __init__(self, **kwargs):
        super(WarpperSwinTransformer, self).__init__(**kwargs)
    
    @property
    def dtype(self):
        return self.norm.weight.dtype

    def forward_featuremap(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        # x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x
        
    def encode_image_featuremap(self, image):
        return self.forward_featuremap(image.type(self.dtype))

def load_esvit(name: str, return_s3=False, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
    if name in _MODELS:
        if ((dist.is_initialized() or dist.is_available()) and int(dist.get_rank()) % torch.cuda.device_count() == 0) or not dist.is_available():
            model_path = _download(_MODELS[name], sha_check=False)
        dist.barrier()
        model_path = _download(_MODELS[name], sha_check=False)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; ")


    state_dict = torch.load(model_path, map_location="cpu")['student']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_kwargs = dict(patch_size=4, embed_dim=128, depths=[2,2,18] if return_s3 else [2,2,18,2], num_heads=[4,8,16] if return_s3 else [4,8,16,32], window_size=14, num_classes=0)
    model = WarpperSwinTransformer(**model_kwargs)
    if return_s3:
        state_dict.pop('norm.weight')
        state_dict.pop('norm.bias')
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    return model.to(device)
