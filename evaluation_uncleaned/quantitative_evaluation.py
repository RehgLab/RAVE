import torch
import clip

import sys

import shutil
import os
import glob

import numpy as np


from collections import defaultdict
from transformers import AutoProcessor, AutoModel

from skimage.metrics import structural_similarity

import utils.eval_utils as eu
import utils.preprocesser_utils as pu


if __name__ == '__main__':
    typ = sys.argv[1]
    if typ == 'style':
        dataset_path = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/data/rave_dataset_prepared_512'
        style_prompts_dict = pu.yaml_load(f'{dataset_path}/style_prompts.yaml')
        prev_methods_path = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/PREV_OUTPUTS/outputs_512'
        rave_dataset_path = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/res_automate/11-01-2023_rave_512_style'
        no_shuffle_path = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/res_automate/11-04-2023/no-shuffle-style'
        output_dir = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/FINAL_PREPARED/evaluation_set_512_style'
    elif typ == 'shape':
        dataset_path = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/data/rave_dataset_prepared_512'
        style_prompts_dict = pu.yaml_load(f'{dataset_path}/shape_prompts.yaml')
        prev_methods_path = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/PREV_OUTPUTS/outputs_shape_512'
        rave_dataset_path = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/res_automate/11-02-2023-shape_512'
        no_shuffle_path = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/res_automate/11-04-2023/no-shuffle-shape'
        output_dir = '/coc/flash6/okara7/codes/video-editing/hf-controlnet/FINAL_PREPARED/evaluation_set_512_shape'
    frame_count = int(sys.argv[2])
    st = 50
    prepare = False
    output_dir = f'{output_dir}/{frame_count}-frames'
    frame_prompt_dict = style_prompts_dict[f'{frame_count}-frames']

    if prepare:
        for key in frame_prompt_dict:
            for prompt in frame_prompt_dict[key]:
                output_save_dir = f'{output_dir}/{key}/{prompt}'
                os.makedirs(output_save_dir, exist_ok=True)
                
                # Prepare Rerender Data
                for i in range(1,4):
                    rerender_path = f'{prev_methods_path}/st-{st}_fr-{frame_count}/rerender/{key}_pad-{i}/{prompt.replace(" ", "-")}/res.gif'
                    if os.path.exists(rerender_path):
                        shutil.copy(rerender_path, f'{output_dir}/{key}/{prompt}/rerender.gif')
                        break
                    
                # Prepare Tokenflow Data
                for i in range(1,4):
                    tokenflow_path = f'{prev_methods_path}/st-{st}_fr-{frame_count}/tokenflow/pnp_SD_1.5/{key}_pad-{i}/{prompt}'
                    if os.path.exists(tokenflow_path):
                        tokenflow_path = glob.glob(f'{tokenflow_path}/**/*.gif', recursive=True)  
                        if len(tokenflow_path) > 0:
                            tokenflow_path = tokenflow_path[0]
                            shutil.copy(tokenflow_path, f'{output_dir}/{key}/{prompt}/tokenflow.gif')
                            break
                
                # Prepare Pix2Video Data
                    pix2video_path = f'{prev_methods_path}/st-{st}_fr-{frame_count}/pix2video/{key}/{prompt.replace(" ","+")}/samples'
                    if os.path.exists(pix2video_path):
                        try:
                        
                            pix2video_path = glob.glob(f'{pix2video_path}/sample/*.gif', recursive=True)[0]  
                            shutil.copy(pix2video_path, f'{output_dir}/{key}/{prompt}/pix2video.gif')
                            break
                        except:
                            print(pix2video_path)
                            break
                
                # Prepare Text2Video-Zero Data
                for i in range(1,4):
                    text2video_path = f'{prev_methods_path}/st-{st}_fr-{frame_count}/text2video/{key}_pad-{i}/{prompt}'
                    if os.path.exists(text2video_path):
                        text2video_path = glob.glob(f'{text2video_path}/**/*.gif', recursive=True)[0]  
                        shutil.copy(text2video_path, f'{output_dir}/{key}/{prompt}/text2video.gif')
                        break

                # Prepare Rave Data
                rave_path = glob.glob(f'{rave_dataset_path}/{key}*/{prompt}*/*.gif', recursive=True)
                if len(rave_path) > 0:

                    rave_path = rave_path[0]
                    shutil.copy(rave_path, f'{output_dir}/{key}/{prompt}/rave.gif')
                        
                # Prepare No-Shuffle Data
                    

                no_shuffle = glob.glob(f'{no_shuffle_path}/*{key}*/*{prompt}*/*.gif', recursive=True)

                if len(no_shuffle) > 0:

                    no_shuffle = no_shuffle[0]
                    shutil.copy(no_shuffle, f'{output_dir}/{key}/{prompt}/no-shuffle.gif')
                    
                # Prepare Source Video
                source_video_path = glob.glob(f'{dataset_path}/{frame_count}-frames/{key}*.mp4', recursive=True)[0]
                shutil.copy(source_video_path, f'{output_dir}/{key}/{prompt}/source.mp4')
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        pick_model = AutoModel.from_pretrained("pickapic-anonymous/PickScore_v1").to(device)
        pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        raft_model = eu.prepare_raft_model(device)

        rearrange = lambda x: (np.array(x)/255).reshape(-1,1)
        l2_norm = lambda x,y: np.linalg.norm(rearrange(x)-rearrange(y))/rearrange(x).shape[0]
        l1_norm = lambda x,y: np.linalg.norm(rearrange(x)-rearrange(y), ord=1)/rearrange(x).shape[0]
        
        main_dict = {
            'rerender': {},
            'tokenflow': {},
            'text2video': {},
            'rave': {},
            'no-shuffle': {},
            'pix2video': {},
        }
        
        scores_main = defaultdict(float)

        for video_name in frame_prompt_dict:
            
            for prompt in frame_prompt_dict[video_name]:

                for k in main_dict.keys(): 

                    main_dict[k][video_name] = {}
                    scores = scores_main.copy()
                    video_path = f'{output_dir}/{video_name}/{prompt}/{k}.gif'
                    source_video_path = f'{output_dir}/{video_name}/{prompt}/source.mp4'
                    if os.path.exists(video_path):
                        pil_list = eu.video_to_pil_list(video_path)
                        source_pil_list = eu.video_to_pil_list(source_video_path)
                        
                        scores['clip-frame'] = eu.clip_frame(pil_list, preprocess, device, model)
                        scores['clip-text'] = eu.clip_text(pil_list, prompt, preprocess, device, model)
                        
                        scores['pick-score'] = eu.pick_score_func(pil_list, prompt, pick_model, pick_processor, device)
                        if k == 'rerender':
                            # scores['warp-error-l1'] = eu.warp_video(pil_list, source_pil_list[1:-1], raft_model, device, l2_norm)
                            # scores['warp-error-l2'] = eu.warp_video(pil_list, source_pil_list[1:-1], raft_model, device, l1_norm)
                            scores['warp-error-ssim'] = eu.warp_video(pil_list, source_pil_list[1:-1], raft_model, device, structural_similarity)
                        else:
                            # scores['warp-error-l1'] = eu.warp_video(pil_list, source_pil_list, raft_model, device, l2_norm)
                            # scores['warp-error-l2'] = eu.warp_video(pil_list, source_pil_list, raft_model, device, l1_norm)
                            scores['warp-error-ssim'] = eu.warp_video(pil_list, source_pil_list, raft_model, device, structural_similarity)
                        # print(f'{video_name} - {prompt} - {k} - ', end='\n')

                    main_dict[k][video_name][prompt] = scores.copy()
                print(f'{video_name} - {prompt} - ', end='\n')
                for k in main_dict.keys():
                    print(f'\t{k}: ', end='')
                    for s in sorted(main_dict[k][video_name][prompt].keys()):
                        if 'warp-error-l1' in s:
                            print(f'{(main_dict[k][video_name][prompt][s]*100000):.2f}', end=', ')
                        elif 'warp-error-l2' in s or 'warp-error-ssim' in s:
                            print(f'{(main_dict[k][video_name][prompt][s]*100):.2f}', end=', ')
                        else:
                            print(f'{main_dict[k][video_name][prompt][s]:.4f}', end=', ')
                    print()
                print()

        for k in main_dict.keys():
            samp_num = 0
            scores = scores_main.copy()
            for video_name in main_dict[k]:
                for prompt in main_dict[k][video_name]:
                    for score in main_dict[k][video_name][prompt]:
                        scores[score] += main_dict[k][video_name][prompt][score]
                    samp_num += 1
            for score in scores:
                scores[score] /= samp_num
            print(k,scores)




    