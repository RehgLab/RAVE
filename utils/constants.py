
import os
from datetime import datetime 

date_time = datetime.now().strftime("%m-%d-%Y")

CWD = os.getcwd()
MP4_PATH = f'{CWD}/data/mp4_videos'
OUTPUT_PATH = f'{CWD}/results/{date_time}'

GENERATED_DATA_PATH = os.path.join(CWD, 'generated', 'data')
PREPROCESSOR_DICT = {
    'lineart_realistic': "lllyasviel/control_v11p_sd15_lineart",
    'lineart_coarse': "lllyasviel/control_v11p_sd15_lineart",
    'lineart_standard': "lllyasviel/control_v11p_sd15_lineart",
    'lineart_anime': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'lineart_anime_denoise': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'softedge_hed': 'lllyasviel/control_v11p_sd15_softedge',
    'softedge_hedsafe': 'lllyasviel/control_v11p_sd15_softedge',
    'softedge_pidinet': 'lllyasviel/control_v11p_sd15_softedge',
    'softedge_pidsafe': 'lllyasviel/control_v11p_sd15_softedge',
    'canny': 'lllyasviel/control_v11p_sd15_canny',
    'depth_leres': 'lllyasviel/control_v11f1p_sd15_depth',
    'depth_leres++': 'lllyasviel/control_v11f1p_sd15_depth',
    'depth_midas': 'lllyasviel/control_v11f1p_sd15_depth',
    'depth_zoe': 'lllyasviel/control_v11f1p_sd15_depth',
}

MODEL_IDS = {
    # 'Realistic Vision': '130072',
    'SD 1.5': 'None'
}

