import gradio as gr
import cv2
import os
import torch
import argparse
import os
import sys
import yaml
import datetime
sys.path.append(os.path.dirname(os.getcwd()))
from pipelines.sd_controlnet_rave import RAVE
from pipelines.sd_multicontrolnet_rave import RAVE_MultiControlNet
import shutil
import subprocess
import utils.constants as const
import utils.video_grid_utils as vgu
import warnings
warnings.filterwarnings("ignore")
import pprint 
import glob


def init_device():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    return device

def init_paths(input_ns):
    if input_ns.save_folder == None or input_ns.save_folder == '':
        input_ns.save_folder = input_ns.video_name
    else:
        input_ns.save_folder = os.path.join(input_ns.save_folder, input_ns.video_name)
    save_dir = os.path.join(const.OUTPUT_PATH, input_ns.save_folder)
    os.makedirs(save_dir, exist_ok=True)
    save_idx = max([int(x[-5:]) for x in os.listdir(save_dir)])+1 if os.listdir(save_dir) != [] else 0
    input_ns.save_path = os.path.join(save_dir, f'{input_ns.positive_prompts}-{str(save_idx).zfill(5)}')
    
    
    if '-' in input_ns.preprocess_name:
        input_ns.hf_cn_path = [const.PREPROCESSOR_DICT[i] for i in input_ns.preprocess_name.split('-')]
    else:
        input_ns.hf_cn_path = const.PREPROCESSOR_DICT[input_ns.preprocess_name]
    input_ns.hf_path = "runwayml/stable-diffusion-v1-5"
    
    input_ns.inverse_path = os.path.join(const.GENERATED_DATA_PATH, 'inverses', input_ns.video_name, f'{input_ns.preprocess_name}_{input_ns.model_id}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}')
    input_ns.control_path = os.path.join(const.GENERATED_DATA_PATH, 'controls', input_ns.video_name, f'{input_ns.preprocess_name}_{input_ns.grid_size}x{input_ns.grid_size}_{input_ns.pad}')
    os.makedirs(input_ns.control_path, exist_ok=True)
    os.makedirs(input_ns.inverse_path, exist_ok=True)
    os.makedirs(input_ns.save_path, exist_ok=True)
    return input_ns
    
def install_civitai_model(model_id):
    full_path = os.path.join(const.CWD, 'CIVIT_AI', 'diffusers_models', model_id, '*')
    if len(glob.glob(full_path)) > 0:
        full_path = glob.glob(full_path)[0]
        return full_path
    install_path = os.path.join(const.CWD, 'CIVIT_AI', 'safetensors')
    install_path_model = os.path.join(const.CWD, 'CIVIT_AI', 'safetensors', model_id)
    diffusers_path = os.path.join(const.CWD, 'CIVIT_AI', 'diffusers_models', model_id)
    convert_py_path = os.path.join(const.CWD, 'CIVIT_AI', 'convert.py')
    os.makedirs(install_path, exist_ok=True)
    os.makedirs(diffusers_path, exist_ok=True)
    subprocess.run(f'wget https://civitai.com/api/download/models/{model_id} --content-disposition --directory {install_path_model}'.split())
    model_name = glob.glob(os.path.join(install_path, model_id, '*'))[0]
    model_name2 = os.path.basename(glob.glob(os.path.join(install_path, model_id, '*'))[0]).replace('.safetensors', '')
    diffusers_path_model_name = os.path.join(const.CWD, 'CIVIT_AI', 'diffusers_models', model_id, model_name2)
    print(model_name)
    subprocess.run(f'python {convert_py_path} --checkpoint_path {model_name}  --dump_path {diffusers_path_model_name}  --from_safetensors'.split())
    subprocess.run(f'rm -rf {install_path}'.split())
    return diffusers_path_model_name

def run(*args):
    list_of_inputs = [x for x in args]
    input_ns = argparse.Namespace(**{})
    input_ns.video_path = list_of_inputs[0] # video_path 
    input_ns.video_name = os.path.basename(input_ns.video_path).replace('.mp4', '').replace('.gif', '') 
    input_ns.preprocess_name = list_of_inputs[1]

    input_ns.batch_size = list_of_inputs[2]  
    input_ns.batch_size_vae = list_of_inputs[3]

    input_ns.cond_step_start = list_of_inputs[4]
    input_ns.controlnet_conditioning_scale = list_of_inputs[5]  
    input_ns.controlnet_guidance_end = list_of_inputs[6]  
    input_ns.controlnet_guidance_start = list_of_inputs[7]  

    input_ns.give_control_inversion = list_of_inputs[8]  

    input_ns.grid_size = list_of_inputs[9]  
    input_ns.sample_size = list_of_inputs[10]  
    input_ns.pad = list_of_inputs[11]
    input_ns.guidance_scale = list_of_inputs[12]
    input_ns.inversion_prompt = list_of_inputs[13]

    input_ns.is_ddim_inversion = list_of_inputs[14]  
    input_ns.is_shuffle = list_of_inputs[15]  

    input_ns.negative_prompts = list_of_inputs[16]  
    input_ns.num_inference_steps = list_of_inputs[17]  
    input_ns.num_inversion_step = list_of_inputs[18]  
    input_ns.positive_prompts = list_of_inputs[19] 
    input_ns.save_folder = list_of_inputs[20]  

    input_ns.seed = list_of_inputs[21]  
    input_ns.model_id = const.MODEL_IDS[list_of_inputs[22]] 
    # input_ns.width = list_of_inputs[23] 
    # input_ns.height = list_of_inputs[24] 
    # input_ns.original_size = list_of_inputs[25]
    diffusers_model_path = os.path.join(const.CWD, 'CIVIT_AI', 'diffusers_models')
    os.makedirs(diffusers_model_path, exist_ok=True)
    if 'model_id' not in list(input_ns.__dict__.keys()):
        input_ns.model_id = "None"
    
    if str(input_ns.model_id) != 'None':
        input_ns.model_id = install_civitai_model(input_ns.model_id)

    
    device = init_device()
    input_ns = init_paths(input_ns)

    input_ns.image_pil_list = vgu.prepare_video_to_grid(input_ns.video_path, input_ns.sample_size, input_ns.grid_size, input_ns.pad)

    print(input_ns.video_path)
    input_ns.sample_size = len(input_ns.image_pil_list)
    print(f'Frame count: {len(input_ns.image_pil_list)}')

    controlnet_class = RAVE_MultiControlNet if '-' in str(input_ns.controlnet_conditioning_scale) else RAVE
    

    CN = controlnet_class(device)

    CN.init_models(input_ns.hf_cn_path, input_ns.hf_path, input_ns.preprocess_name, input_ns.model_id)
    
    input_dict = vars(input_ns)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(input_dict)
    yaml_dict = {k:v for k,v in input_dict.items() if k != 'image_pil_list'}

    start_time = datetime.datetime.now()
    if '-' in str(input_ns.controlnet_conditioning_scale):
        res_vid, control_vid_1, control_vid_2 = CN(input_dict)
    else: 
        res_vid, control_vid = CN(input_dict)
    end_time = datetime.datetime.now()
    save_name = f"{'-'.join(input_ns.positive_prompts.split())}_cstart-{input_ns.controlnet_guidance_start}_gs-{input_ns.guidance_scale}_pre-{'-'.join((input_ns.preprocess_name.replace('-','+').split('_')))}_cscale-{input_ns.controlnet_conditioning_scale}_grid-{input_ns.grid_size}_pad-{input_ns.pad}_model-{os.path.basename(input_ns.model_id)}"
    res_vid[0].save(os.path.join(input_ns.save_path, f'{save_name}.gif'), save_all=True, append_images=res_vid[1:], loop=10000)
    control_vid[0].save(os.path.join(input_ns.save_path, f'control_{save_name}.gif'), save_all=True, append_images=control_vid[1:], optimize=False, loop=10000)

    yaml_dict['total_time'] = (end_time - start_time).total_seconds()
    yaml_dict['total_number_of_frames'] = len(res_vid)
    yaml_dict['sec_per_frame'] = yaml_dict['total_time']/yaml_dict['total_number_of_frames']
    with open(os.path.join(input_ns.save_path, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file)
    
    return os.path.join(input_ns.save_path, f'{save_name}.gif'), os.path.join(input_ns.save_path, f'control_{save_name}.gif')



block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown('## RAVE')
    with gr.Row():
        with gr.Column():
            # input_path = gr.Video(label='Input Video',
            #                       sources='upload',
            #                       format='mp4',
            #                       visible=True)
            with gr.Row():
                input_path = gr.File(label='Upload Input Video', file_types=['.mp4'], scale=1)
                
                inputs = gr.Video(label='Input Video', 
                                    format='mp4',
                                    visible=True,
                                    interactive=False,
                                    scale=5)
                input_path.upload(lambda x:x, inputs=[input_path], outputs=[inputs])
                
            with gr.Row():
                positive_prompts = gr.Textbox(label='Positive prompts')
                negative_prompts = gr.Textbox(label='Negative prompts')
            with gr.Row():  
                preprocess_name = gr.Dropdown(const.PREPROCESSOR_DICT.keys(),
                                            label='Control type',
                                            value='depth_zoe')
                guidance_scale = gr.Slider(label='Guidance scale',
                                minimum=0,
                                maximum=40,
                                step=0.1,
                                value=7.5)

            with gr.Row():
                inversion_prompt = gr.Textbox(label='Inversion prompt')
                seed = gr.Slider(label='Seed',
                                minimum=0,
                                maximum=2147483647,
                                step=1,
                                value=0,
                                randomize=True)
            
            with gr.Row():
                model_id = gr.Dropdown(const.MODEL_IDS,
                                    label='Model id',
                                    value='SD 1.5')
                save_folder = gr.Textbox(label='Save folder')
            
            run_button = gr.Button(value='Run All')
            with gr.Accordion('Configuration',
                              open=False):
                with gr.Row():
                    batch_size = gr.Slider(label='Batch size',
                              minimum=1,
                              maximum=36,
                              value=4,
                              step=1)
                    batch_size_vae = gr.Slider(label='Batch size of VAE',
                              minimum=1,
                              maximum=36,
                              value=1,
                              step=1)
                
                with gr.Row():
                    is_ddim_inversion = gr.Checkbox(
                        label='Use DDIM Inversion',
                        value=True)
                    is_shuffle = gr.Checkbox(
                        label='Shuffle',
                        value=True)
                
                with gr.Row():
                    num_inference_steps = gr.Slider(label='Number of inference steps',
                              minimum=1,
                              maximum=100,
                              value=20,
                              step=1)
                    num_inversion_step = gr.Slider(label='Number of inversion steps',
                              minimum=1,
                              maximum=100,
                              value=20,
                              step=1)
                    cond_step_start = gr.Slider(label='Conditioning step start',
                                minimum=0,
                                maximum=1.0,
                                value=0.0,
                                step=0.1)
                
                with gr.Row():
                    controlnet_conditioning_scale = gr.Slider(label='ControlNet conditioning scale',
                              minimum=0.0,
                              maximum=1.0,
                              value=1.0,
                              step=0.01)
                    controlnet_guidance_end = gr.Slider(label='ControlNet guidance end',
                              minimum=0.0,
                              maximum=1.0,
                              value=1.0,
                              step=0.01)
                    controlnet_guidance_start = gr.Slider(label='ControlNet guidance start',
                              minimum=0.0,
                              maximum=1.0,
                              value=0.0,
                              step=0.01)
                give_control_inversion = gr.Checkbox(
                    label='Give control during inversion',
                    value=True)
                    
                with gr.Row():
                    grid_size = gr.Slider(label='Grid size',
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1)
                    sample_size = gr.Slider(label='Sample size',
                                minimum=-1,
                                maximum=100,
                                value=-1,
                                step=1)
                    pad = gr.Slider(label='Pad',
                                minimum=1,
                                maximum=10,
                                value=1,
                                step=1)

        with gr.Column():
            with gr.Row():
                result_video = gr.Image(label='Edited Video',
                                        interactive=False)
                control_video = gr.Image(label='Control Video',
                                        interactive=False)

    inputs = [input_path, preprocess_name, batch_size, batch_size_vae, cond_step_start, controlnet_conditioning_scale, controlnet_guidance_end, controlnet_guidance_start, give_control_inversion, grid_size, sample_size, pad, guidance_scale, inversion_prompt, is_ddim_inversion, is_shuffle, negative_prompts, num_inference_steps, num_inversion_step, positive_prompts, save_folder, seed, model_id]
    
    run_button.click(fn=run,
                     inputs=inputs,
                     outputs=[result_video, control_video])


block.launch(share=True)