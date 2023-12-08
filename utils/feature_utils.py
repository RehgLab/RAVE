import torch

def flatten_grid(x, grid_size=[2, 2]):
    '''
    x: B x C x H x W
    '''
    B, C, H, W = x.size()

    hs, ws = grid_size

    img_h = H // hs

    flattened = torch.cat(torch.split(x, img_h, dim=2), dim=-1)

    return flattened

def unflatten_grid(x, grid_size=[2,2]):
    ''' 
    x: B x C x H x W
    '''
    B, C, H, W = x.size()
    hs, ws = grid_size
    img_w = W // (ws)

    unflattened = torch.cat(torch.split(x, img_w, dim=3), dim=-2)
        
    return unflattened
    
def prepare_key_grid_latents(latents_video, latent_grid_size=[2,2], key_grid_size=[3,3], rand_indices=None):

    T = latents_video.size(0)
    img_h, img_w = latents_video.size(-2) // latent_grid_size[0], latents_video.size(-1) // latent_grid_size[1]
    list_of_flattens = [flatten_grid(el.unsqueeze(0), latent_grid_size) for el in latents_video]
    long_flatten = torch.cat(list_of_flattens, dim=-1)
    
    keyframe_grid = unflatten_grid(torch.cat([long_flatten[:,:,:,ind*(img_w):(ind+1)*(img_w)] for ind in rand_indices], dim=-1), key_grid_size)
    return keyframe_grid, rand_indices

    
def pil_grid_to_frames(pil_grid, grid_size=[2,2]):
    w,h = pil_grid.size
    img_w = w // grid_size[1]
    img_h = h // grid_size[0]
    list_of_pil = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            list_of_pil.append(pil_grid.crop((j*img_w, i*img_h, (j+1)*img_w, (i+1)*img_h)))
    return list_of_pil
    

if __name__ == '__main__':
    a = torch.randint(0,5,(1,3), dtype=torch.float)

    
    
