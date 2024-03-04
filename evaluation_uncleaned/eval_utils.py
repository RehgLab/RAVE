from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torch.nn.functional as F
import cv2
import imageio
import argparse
import sys
import torch
import clip
import warnings
import numpy as np
sys.path.append('/coc/flash6/okara7/codes/video-editing/hf-controlnet/RAFT/RAFT-master')
sys.path.append('/coc/flash6/okara7/codes/video-editing/hf-controlnet/RAFT/RAFT-master/core')
from core.raft import RAFT
from core.utils.utils import InputPadder
from skimage.metrics import structural_similarity

def video_to_pil_list(video_path):
    if video_path.endswith('.mp4'):
        vidcap = cv2.VideoCapture(video_path)
        pil_list = []
        while True:
            success, image = vidcap.read()
            if success:
                pil_list.append(Image.fromarray(image))
            else:
                break

        return pil_list
    elif video_path.endswith('.gif'):
        gif = imageio.get_reader(video_path)
        pil_list = []

        for frame in gif:
            pil_list.append(Image.fromarray(frame))

        return pil_list


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img,
                    sample_coords,
                    mode='bilinear',
                    padding_mode='zeros',
                    return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img,
                        grid,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (
            y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp_rerender(feature,
              flow,
              mask=False,
              mode='bilinear',
              padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature,
                           grid,
                           mode=mode,
                           padding_mode=padding_mode,
                           return_mask=mask)


def clip_text(pil_list, text_prompt, preprocess, device, model):
    text = clip.tokenize([text_prompt]).to(device)

    scores = []
    images = []
    with torch.no_grad():
        text_features = model.encode_text(text)
        for pil in pil_list:
            image = preprocess(pil).unsqueeze(0).to(device)
            images.append(image)
        image_features = model.encode_image(torch.cat(images))
        scores = [torch.cosine_similarity(text_features, image_feature).item() for image_feature in image_features]

    score = sum(scores) / len(scores)
    
    return score

def clip_frame(pil_list, preprocess, device, model):
    image_features = []
    images = []
    with torch.no_grad():
        for pil in pil_list:
            image = preprocess(pil).unsqueeze(0).to(device)
            images.append(image)
        
        image_features = model.encode_image(torch.cat(images))
        
    image_features = image_features.cpu().numpy()
    cosine_sim_matrix = cosine_similarity(image_features)
    np.fill_diagonal(cosine_sim_matrix, 0)  # set diagonal elements to 0
    score = cosine_sim_matrix.sum() / (len(pil_list) * (len(pil_list)-1))

    return score

def pick_score_func(frames, prompt, model, processor, device):
    image_inputs = processor(images=frames, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    text_inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        score_per_image = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        score_per_image = score_per_image.detach().cpu().numpy()
        score = score_per_image.mean()

    return score

def prepare_raft_model(device):
    raft_dict = {
        'model': '/coc/flash6/okara7/codes/kurtkaya/RAFT/models/raft-things.pth',
        'small': False,
        'mixed_precision': False,
        'alternate_corr': False
    }

    args = argparse.Namespace(**raft_dict)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
    model.eval()

    return model

def flow_warp(img: np.ndarray,
              flow: np.ndarray,
              filling_value: int = 0,
              interpolate_mode: str = 'nearest'):
    '''Use flow to warp img.

    Args:
        img (ndarray): Image to be warped.
        flow (ndarray): Optical Flow.
        filling_value (int): The missing pixels will be set with filling_value.
        interpolate_mode (str): bilinear -> Bilinear Interpolation;
                                nearest -> Nearest Neighbor.

    Returns:
        ndarray: Warped image with the same shape of img
    '''
    warnings.warn('This function is just for prototyping and cannot '
                  'guarantee the computational efficiency.')
    assert flow.ndim == 3, 'Flow must be in 3D arrays.'
    height = flow.shape[0]
    width = flow.shape[1]
    channels = img.shape[2]

    output = np.ones(
        (height, width, channels), dtype=img.dtype) * filling_value

    grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    dx = grid[:, :, 0] + flow[:, :, 1]
    dy = grid[:, :, 1] + flow[:, :, 0]
    sx = np.floor(dx).astype(int)
    sy = np.floor(dy).astype(int)
    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)

    if interpolate_mode == 'nearest':
        output[valid, :] = img[dx[valid].round().astype(int),
                               dy[valid].round().astype(int), :]
    elif interpolate_mode == 'bilinear':
        # dirty walkround for integer positions
        eps_ = 1e-6
        dx, dy = dx + eps_, dy + eps_
        left_top_ = img[np.floor(dx[valid]).astype(int),
                        np.floor(dy[valid]).astype(int), :] * (
                            np.ceil(dx[valid]) - dx[valid])[:, None] * (
                                np.ceil(dy[valid]) - dy[valid])[:, None]
        left_down_ = img[np.ceil(dx[valid]).astype(int),
                         np.floor(dy[valid]).astype(int), :] * (
                             dx[valid] - np.floor(dx[valid]))[:, None] * (
                                 np.ceil(dy[valid]) - dy[valid])[:, None]
        right_top_ = img[np.floor(dx[valid]).astype(int),
                         np.ceil(dy[valid]).astype(int), :] * (
                             np.ceil(dx[valid]) - dx[valid])[:, None] * (
                                 dy[valid] - np.floor(dy[valid]))[:, None]
        right_down_ = img[np.ceil(dx[valid]).astype(int),
                          np.ceil(dy[valid]).astype(int), :] * (
                              dx[valid] - np.floor(dx[valid]))[:, None] * (
                                  dy[valid] - np.floor(dy[valid]))[:, None]
        output[valid, :] = left_top_ + left_down_ + right_top_ + right_down_
    else:
        raise NotImplementedError(
            'We only support interpolation modes of nearest and bilinear, '
            f'but got {interpolate_mode}.')
    return output.astype(img.dtype)

def calculate_flow(pil_list, model, DEVICE):
    def load_image(imfile, DEVICE):
        img = np.array(imfile).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    flow_up_list = []
    with torch.no_grad():
        images = pil_list.copy()
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1, DEVICE)
            image2 = load_image(imfile2, DEVICE)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=20, test_mode=True)

            flow_up_list.append(flow_up.detach().squeeze().permute(1,2,0).cpu().numpy())
    return flow_up_list

def rerender_warp(img, flow, mode='bilinear'):
    expand = False
    if len(img.shape) == 2:
        expand = True
        img = np.expand_dims(img, 2)

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    dtype = img.dtype
    img = img.to(torch.float)
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
    res = flow_warp_rerender(img, flow, mode=mode)
    res = res.to(dtype)
    res = res[0].cpu().permute(1, 2, 0).numpy()
    if expand:
        res = res[:, :, 0]
    return res

def opencv_warp(img, flow):

    h, w = flow.shape[:2]
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    warped_img = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return warped_img

rearrange = lambda x: (np.array(x)/255).reshape(-1,1)

def warp_video(edit_pil_list, source_pil_list, raft_model, device, distance_func):
    # print('source size', source_pil_list[0].size)
    flow_up_list = calculate_flow(source_pil_list, raft_model, device)

    res_list = [edit_pil_list[0]]
    for i,pil_img in enumerate(edit_pil_list[:-1]):
        warped = opencv_warp(np.array(pil_img), flow_up_list[i])
        pil_warped = Image.fromarray(warped)
        # pil_warped.save(f'warped_{i}.png')
        res_list.append(pil_warped)
    # res_list[0].save('warped.gif', save_all=True, append_images=res_list[1:], duration=100, loop=0)
    # print('size of video', res_list[0].size)
    if distance_func == structural_similarity:
        return np.mean(np.array([distance_func(np.array(edit_pil_list[i]), np.array(res_list[i]), channel_axis=2) for i in range(len(res_list))]))
    else:
        return np.mean(np.array([distance_func(edit_pil_list[i], res_list[i]) for i in range(len(res_list))]))
