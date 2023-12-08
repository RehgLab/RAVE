import os
import torch
import numpy as np
from einops import rearrange
from annotator.pidinet.model import pidinet
from annotator.util import safe_step
from annotator.annotator_path import models_path, DEVICE
import safetensors.torch
# from modules.safe import unsafe_torch_load

def get_state_dict(d):
    return d.get("state_dict", d)

def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = torch.load(ckpt_path, map_location=torch.device(location))
    state_dict = get_state_dict(state_dict)
    return state_dict

netNetwork = None
remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
modeldir = os.path.join(models_path, "pidinet")
old_modeldir = os.path.dirname(os.path.realpath(__file__))

def apply_pidinet(input_image, is_safe=False, apply_fliter=False):
    global netNetwork
    if netNetwork is None:
        modelpath = os.path.join(modeldir, "table5_pidinet.pth")
        old_modelpath = os.path.join(old_modeldir, "table5_pidinet.pth")
        if os.path.exists(old_modelpath):
            modelpath = old_modelpath
        elif not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=modeldir)
        netNetwork = pidinet()
        ckp = load_state_dict(modelpath)
        netNetwork.load_state_dict({k.replace('module.',''):v for k, v in ckp.items()})
        
    netNetwork = netNetwork.to(DEVICE)
    netNetwork.eval()
    assert input_image.ndim == 3
    input_image = input_image[:, :, ::-1].copy()
    with torch.no_grad():
        image_pidi = torch.from_numpy(input_image).float().to(DEVICE)
        image_pidi = image_pidi / 255.0
        image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
        edge = netNetwork(image_pidi)[-1]
        edge = edge.cpu().numpy()
        if apply_fliter:
            edge = edge > 0.5 
        if is_safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        
    return edge[0][0] 

def unload_pid_model():
    global netNetwork
    if netNetwork is not None:
        netNetwork.cpu()