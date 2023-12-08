import torch
import argparse
import os
import itertools
import sys
import yaml
import datetime
sys.path.append(os.getcwd())
from pipelines.sd_controlnet_rave import RAVE
from pipelines.sd_multicontrolnet_rave import RAVE_MultiControlNet

import utils.constants as const
import utils.video_grid_utils as vgu

import warnings

warnings.filterwarnings("ignore")

import numpy as np


def init_device():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    return device

def init_paths(input_ns):
    if input_ns.save_folder == None or input_ns.save_folder == '':
        input_ns.save_folder = input_ns.video_name.replace('.mp4', '').replace('.gif', '')
    else:
        input_ns.save_folder += f"/{input_ns.video_name.replace('.mp4', '').replace('.gif', '')}"
    save_dir = f'{const.OUTPUT_PATH}/{input_ns.save_folder}'
    os.makedirs(save_dir, exist_ok=True)
    save_idx = max([int(x[-5:]) for x in os.listdir(save_dir)])+1 if os.listdir(save_dir) != [] else 0
    input_ns.save_path = f'{save_dir}/{input_ns.positive_prompts}-{str(save_idx).zfill(5)}'
    

    input_ns.video_path = f'{const.MP4_PATH}/{input_ns.video_name}.mp4'
    
    if '-' in input_ns.preprocess_name:
        input_ns.hf_cn_path = [const.PREPROCESSOR_DICT[i] for i in input_ns.preprocess_name.split('-')]
    else:
        input_ns.hf_cn_path = const.PREPROCESSOR_DICT[input_ns.preprocess_name]
    input_ns.hf_path = "runwayml/stable-diffusion-v1-5"
    
    input_ns.inverse_path = f'{const.GENERATED_DATA_PATH}/inverses/{input_ns.video_name}/{input_ns.preprocess_name}_{input_ns.model_id}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}'
    input_ns.control_path = f'{const.GENERATED_DATA_PATH}/controls/{input_ns.video_name}/{input_ns.preprocess_name}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}'
    os.makedirs(input_ns.control_path, exist_ok=True)
    os.makedirs(input_ns.inverse_path, exist_ok=True)
    os.makedirs(input_ns.save_path, exist_ok=True)
    
    return input_ns
    
def run(input_ns):

    if 'model_id' not in list(input_ns.__dict__.keys()):
        input_ns.model_id = "None"
    device = init_device()
    input_ns = init_paths(input_ns)

    input_ns.image_pil_list = vgu.prepare_video_to_grid(input_ns.video_path, input_ns.sample_size, input_ns.grid_size, input_ns.pad)
    input_ns.sample_size = len(input_ns.image_pil_list)
    print(f'Frame count: {len(input_ns.image_pil_list)}')

    controlnet_class = RAVE_MultiControlNet if '-' in str(input_ns.controlnet_conditioning_scale) else RAVE
    

    CN = controlnet_class(device)


    CN.init_models(input_ns.hf_cn_path, input_ns.hf_path, input_ns.preprocess_name, input_ns.model_id)
    
    input_dict = vars(input_ns)
    yaml_dict = {k:v for k,v in input_dict.items() if k != 'image_pil_list'}

    start_time = datetime.datetime.now()
    if '-' in str(input_ns.controlnet_conditioning_scale):
        res_vid, control_vid_1, control_vid_2 = CN(input_dict)
    else: 
        res_vid, control_vid = CN(input_dict)
    end_time = datetime.datetime.now()
    save_name = f"{'-'.join(input_ns.positive_prompts.split())}_cstart-{input_ns.controlnet_guidance_start}_gs-{input_ns.guidance_scale}_pre-{'-'.join((input_ns.preprocess_name.replace('-','+').split('_')))}_cscale-{input_ns.controlnet_conditioning_scale}_grid-{input_ns.grid_size}_pad-{input_ns.pad}_model-{input_ns.model_id.split('/')[-1]}"
    res_vid[0].save(f"{input_ns.save_path}/{save_name}.gif", save_all=True, append_images=res_vid[1:], optimize=False, loop=10000)
    # control_vid[0].save(f"{input_ns.save_path}/control_{save_name}.gif", save_all=True, append_images=control_vid[1:], optimize=False, loop=10000)

    yaml_dict['total_time'] = (end_time - start_time).total_seconds()
    yaml_dict['total_number_of_frames'] = len(res_vid)
    yaml_dict['sec_per_frame'] = yaml_dict['total_time']/yaml_dict['total_number_of_frames']
    with open(f'{input_ns.save_path}/config.yaml', 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file)
        

if __name__ == '__main__':
    config_path = sys.argv[1]
    input_dict_list = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    list_vals = []
    list_keys = []
    for key in input_dict_list.keys():
        if type(input_dict_list[key]) is list:
            list_vals.append(input_dict_list[key])
            list_keys.append(key)

    input_dict_list_temp = {k:v for k,v in input_dict_list.items() if k not in list_keys}        
    for item in list(itertools.product(*list_vals)):
        input_dict_list_temp.update({list_keys[i]:item[i] for i in range(len(list_keys))})

        input_ns = argparse.Namespace(**input_dict_list_temp)
        run(input_ns)
    