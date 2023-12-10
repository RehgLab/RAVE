
import os
from datetime import datetime 

date_time = datetime.now().strftime("%m-%d-%Y")

CWD = os.getcwd()
MP4_PATH = os.path.join(CWD, 'data', 'mp4_videos')
OUTPUT_PATH = os.path.join(CWD, 'results', date_time) 

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
    'Realistic Vision V5.1': '130072',
    'Realistic Vision V6.0' : '245598',
    'MajicMIXRealisticV7' : '176425',
    'DreamShaper' : '128713',
    'EpicPhotoGasm' : '223670',
    'DivineEleganceMix (Anime)': '238656',
    'GhostMix (Anime)': '76907',
    'CetusMix (Anime)': '105924',
    'Counterfeit (Anime)': '57618',
    'SD 1.5': 'None'
}

