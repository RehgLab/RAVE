import os
import torch
import utils.constants as const

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_path = f'{const.CWD}/pretrained_models'

clip_vision_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clip_vision')
# clip vision is always inside controlnet "extensions\sd-webui-controlnet"
# and any problem can be solved by removing controlnet and reinstall

models_path = os.path.realpath(models_path)
os.makedirs(models_path, exist_ok=True)
print(f'ControlNet preprocessor location: {models_path}')
# Make sure that the default location is inside controlnet "extensions\sd-webui-controlnet"
# so that any problem can be solved by removing controlnet and reinstall
# if users do not change configs on their own (otherwise users will know what is wrong)
