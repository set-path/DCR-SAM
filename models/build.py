import torch
from sam3 import build_sam3_image_model

def build_model(model_info, use_bg=True):
    model_name = model_info['name']
    if model_name == "SAM3":
        model = build_sam3_image_model(bpe_path=model_info['bpe_path'], checkpoint_path=model_info['checkpoint_path'], use_bg=use_bg)
        return model