import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
from utils.data_transforms import *
from custom_datasets.basic_dataset import BasicDataset
from glob import glob

class VaihingenDataset(BasicDataset):
    def __init__(self, root, split='val', class_name_path=None, bg_idx=0, ignore_index=255, reduce_zero_label=False, mode='train'):
        super().__init__(class_name_path, bg_idx=bg_idx, ignore_index=ignore_index, reduce_zero_label=reduce_zero_label, mode=mode)
        self.root = root
        self.split = split
        self.img_dir = os.path.join(self.root, 'img_dir', self.split)

        
        self.img_paths = glob(os.path.join(self.img_dir, '*.png'))
        self.img_paths.sort()


        self.ann_paths = [f.replace('img_dir', 'ann_dir') for f in self.img_paths]