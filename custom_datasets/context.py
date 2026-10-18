import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from custom_datasets.basic_dataset import BasicDataset
from glob import glob

class ContextDataset(BasicDataset):
    def __init__(self, root, split='training', class_name_path=None, bg_idx=0, ignore_index=255, reduce_zero_label=False, mode='train'):
        super().__init__(class_name_path, bg_idx=bg_idx, ignore_index=ignore_index, reduce_zero_label=reduce_zero_label, mode=mode)
        self.root = root
        self.split = split


        self.img_names = [f.strip() for f in open(os.path.join(self.root, 'ImageSets', 'SegmentationContext', f'{self.split}.txt')).read().splitlines()]
        
        self.img_paths = [os.path.join(self.root, 'JPEGImages', f'{name}.jpg') for name in self.img_names]
        self.img_paths.sort()


        self.ann_paths = [os.path.join(self.root, 'SegmentationClassContext', f'{name}.png') for name in self.img_names]
