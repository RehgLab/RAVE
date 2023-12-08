from PIL import Image
import torch
import numpy as np
import cv2 as cv

def load_pil_img(load_path):
    img = Image.open(load_path).convert('RGB')
    return img

def load_img_as_torch_batch(load_path):
    pil_img = load_pil_img(load_path)
    return pil_img_to_torch_tensor(pil_img).unsqueeze(0)

def pil_img_to_torch_tensor_grayscale(img_pil):
    '''
    Takes a PIL image and returns a torch tensor of shape (1, 1, H, W) with values in [0, 1]
    '''
    return torch.tensor(np.array(img_pil).transpose(0, 1)/255, dtype=torch.float).unsqueeze(0).unsqueeze(0)

def pil_img_to_torch_tensor(img_pil):
    '''
    Takes a PIL image and returns a torch tensor of shape (1, 3, H, W) with values in [0, 1]
    '''
    return torch.tensor(np.array(img_pil).transpose(2, 0, 1)/255, dtype=torch.float).unsqueeze(0)

def torch_to_pil_img(img_torch):
    '''
    Takes a torch tensor of shape (1, 3, H, W) with values in [0, 1] and returns a PIL image
    '''
    return Image.fromarray((img_torch.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)*255).astype('uint8'))

def torch_to_pil_img_batch(img_torch):
    '''
    Takes a torch tensor of shape (1, 3, H, W) with values in [0, 1] and returns a PIL image
    '''
    return [torch_to_pil_img(img_torch[i]) for i in range(img_torch.shape[0])]


def pil_to_cv_gray(pil_img):
    return cv.cvtColor(cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR), cv.COLOR_RGB2GRAY)

def np_to_pil(np_img):
    return Image.fromarray((np_img/255).astype(np.float32).transpose(2,0,1), 'RGB')

def cv_to_pil(np_img):
    return Image.fromarray((np_img/255).astype(np.float32), 'RGB')


def create_grid_from_numpy(np_img, grid_size=[2,2]):

    _, h,w = np_img.shape
    w_grid = w * grid_size[1]
    h_grid = h * grid_size[0]
    grid = np.zeros((h_grid, w_grid))
    img_idx = 0
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            grid[i*h:(i+1)*h, j*w:(j+1)*w] = np_img[img_idx]
            img_idx += 1
    return grid