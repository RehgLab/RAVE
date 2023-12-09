import os
import cv2 as cv
import numpy as np
import torch
import imageio
import glob

from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image


def prepare_video_to_grid(path, grid_count, grid_size, pad):

    video = cv.VideoCapture(path)
    if grid_count == -1:
        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    else:
        frame_count = min(grid_count * pad * grid_size**2, int(video.get(cv.CAP_PROP_FRAME_COUNT)))

    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
    ])
    success = True

    max_grid_area = 512*512* grid_size**2
    grids = []
    frames = []

    total_grid = grid_size**2
    for idx in range(frame_count):
        success,image = video.read()
        assert success, 'Video read failed'
        if idx % pad == 0:
            rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb_img = np.transpose(rgb_img, (2, 0, 1))
            frames.append(transform(torch.from_numpy(rgb_img)))
            
            if len(frames) == total_grid:
                grid = make_grid(frames, nrow=grid_size, padding=0)
                pil_image = (to_pil_image(grid))
                w,h = pil_image.size
                a = float(np.sqrt((w*h/max_grid_area)))
                w1 = int((w//a)//(grid_size*8))*grid_size*8
                h1 = int((h//a)//(grid_size*8))*grid_size*8
                pil_image= pil_image.resize((w1, h1))
                grids.append(pil_image)

                frames = []

    return grids # list of frames

def prepare_video_to_frames(path, grid_count, grid_size, pad, format='gif'):
    video = cv.VideoCapture(path)
    
    if grid_count == -1:
        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        
    else:
        frame_count = min(grid_count * pad * grid_size**2, int(video.get(cv.CAP_PROP_FRAME_COUNT)))
        
    frame_idx = 0
    frames = []
    frames_grid = []

    dir_path = os.path.dirname(path)
    video_name = path.split('/')[-1].split('.')[0]
    os.makedirs(os.path.join(dir_path, 'frames/', video_name), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'video/', video_name), exist_ok=True)

    for idx in range(frame_count):
        success,image = video.read()
        assert success, 'Video read failed'
        if idx % pad == 0:
            frames.append(image)

    for frame in frames[:(len(frames)//(grid_size**2)*(grid_size**2))]:
        frames_grid.append(frame)
        cv.imwrite(os.path.join(dir_path, 'frames/', video_name, f'{str(frame_idx).zfill(5)}.png'), frame)
        frame_idx += 1


    if format == 'gif':
        with imageio.get_writer(os.path.join(dir_path, 'video/', f'{video_name}_fc{frame_idx}_pad{pad}_grid{grid_size}.gif'), mode='I') as writer:
            for frame in frames_grid:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                writer.append_data(frame)
    elif format == 'mp4':
        image_files = sorted(glob.glob(os.path.join(dir_path, 'frames/', video_name, '*.png')))
        images = [imageio.imread(image_file) for image_file in image_files]
        save_file_path = os.path.join(dir_path, 'video/', f'{video_name}_fc{frame_idx}_pad{pad}_grid{grid_size}.mp4')
        imageio.mimsave(save_file_path, images, fps=20)

    return frame_idx # number of frames

