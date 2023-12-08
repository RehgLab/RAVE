import cv2
import yaml

import numpy as np
from annotator.lineart import LineartDetector
from annotator.zoe import ZoeDetector
from annotator.manga_line import MangaLineExtration
from annotator.lineart_anime import LineartAnimeDetector
from annotator.hed import apply_hed
from annotator.canny import apply_canny
from annotator.pidinet import apply_pidinet
from annotator.leres import apply_leres
from annotator.midas import apply_midas


def yaml_load(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def yaml_dump(path, data):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad



def lineart_standard(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    x = img.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), 6.0)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 8]))
    intensity *= 127
    result = intensity.clip(0, 255).astype(np.uint8)
    return remove_pad(result), True


def lineart(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)    
    model_lineart = LineartDetector('sk_model.pth')

    # applied auto inversion
    result = 255 - model_lineart(img)
    return remove_pad(result), True


def lineart_coarse(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_lineart_coarse = LineartDetector('sk_model2.pth')

    # applied auto inversion
    result = 255 - model_lineart_coarse(img)
    return remove_pad(result), True

def lineart_anime(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_lineart_anime = LineartAnimeDetector()

    # applied auto inversion
    result = 255 - model_lineart_anime(img)
    return remove_pad(result), True


def lineart_anime_denoise(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_manga_line = MangaLineExtration()

    # applied auto inversion
    result = model_manga_line(img)
    return remove_pad(result), True


def canny(img, res=512, thr_a=100, thr_b=200, **kwargs):
    l, h = thr_a, thr_b
    img, remove_pad = resize_image_with_pad(img, res)        
    model_canny = apply_canny
    result = model_canny(img, l, h)
    return remove_pad(result), True



def hed(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_hed = apply_hed
    result = model_hed(img)
    return remove_pad(result), True


def hed_safe(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_hed = apply_hed
    result = model_hed(img, is_safe=True)
    return remove_pad(result), True

def midas(img, res=512, a=np.pi * 2.0, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_midas = apply_midas
    result, _ = model_midas(img, a)
    return remove_pad(result), True


def leres(img, res=512, thr_a=0, thr_b=0, boost=False, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_leres = apply_leres
    result = model_leres(img, thr_a, thr_b, boost=boost)
    return remove_pad(result), True

def lerespp(img, res=512, thr_a=0, thr_b=0, boost=True, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_leres = apply_leres
    result = model_leres(img, thr_a, thr_b, boost=boost)
    return remove_pad(result), True


def pidinet(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_pidinet = apply_pidinet
    result = model_pidinet(img)
    return remove_pad(result), True


def pidinet_ts(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_pidinet = apply_pidinet
    result = model_pidinet(img, apply_fliter=True)
    return remove_pad(result), True


def pidinet_safe(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_pidinet = apply_pidinet
    result = model_pidinet(img, is_safe=True)
    return remove_pad(result), True



def zoe_depth(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    model_zoe_depth = ZoeDetector()
    result = model_zoe_depth(img)
    return remove_pad(result), True


preprocessors_dict = {
    'lineart_realistic': lineart,
    'lineart_coarse': lineart_coarse,
    'lineart_standard': lineart_standard,
    'lineart_anime': lineart_anime,
    'lineart_anime_denoise': lineart_anime_denoise,
    'softedge_hed': hed,
    'softedge_hedsafe': hed_safe,
    'softedge_pidinet': pidinet,
    'softedge_pidsafe': pidinet_safe,
    'canny': canny,
    'depth_leres': leres,
    'depth_leres++': lerespp,
    'depth_midas': midas,
    'depth_zoe': zoe_depth,
}

def pixel_perfect_process(input_image, p_name):
    raw_H, raw_W, _ = input_image.shape
    preprocessor_resolution = raw_H
    detected_map, _ = preprocessors_dict[p_name](input_image, res=preprocessor_resolution)
    return detected_map
