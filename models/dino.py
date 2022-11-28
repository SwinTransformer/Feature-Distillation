import os
from typing import Union, List
import torch
from .clip.clip import _download
from .clip.vit import VisionTransformer
import torch.distributed as dist
_MODELS = {
    "DINO": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    "DINO_T": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth"
}

class WarpperVisionTransformer(VisionTransformer):
    def __init__(self, **kwargs):
        super(WarpperVisionTransformer, self).__init__(**kwargs)
    
    @property
    def dtype(self):
        return self.norm.weight.dtype

    def encode_image_featuremap(self, image):
        return self.forward_featuremap(image.type(self.dtype))

def load_dino(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
    if name in _MODELS:
        if ((dist.is_initialized() or dist.is_available()) and int(dist.get_rank()) % torch.cuda.device_count() == 0) or not dist.is_available():
            model_path = _download(_MODELS[name], sha_check=False)
        dist.barrier()
        model_path = _download(_MODELS[name], sha_check=False)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; ")


    state_dict = torch.load(model_path, map_location="cpu")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0)
    model = WarpperVisionTransformer(**model_kwargs)
    msg = model.load_state_dict(state_dict)
    print(msg)
    return model.to(device)