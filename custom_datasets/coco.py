import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from custom_datasets.basic_dataset import BasicDataset
from glob import glob

class COCODataset(BasicDataset):
    def __init__(self, root, split='training', class_name_path=None, bg_idx=0, ignore_index=255, reduce_zero_label=False, mode='train'):
        super().__init__(class_name_path, bg_idx=bg_idx, ignore_index=ignore_index, reduce_zero_label=reduce_zero_label, mode=mode)
        self.root = root
        self.split = split


        self.img_paths = glob(os.path.join(self.root, 'images', self.split, '*.jpg'))
        self.img_paths.sort()


        suffix = '_labelTrainIds.png' if len(self.raw_class_names) == 171 else '_instanceTrainIds.png'
        self.ann_paths = [os.path.join(self.root, 'annotations', self.split, os.path.basename(p).replace('.jpg', suffix)) for p in self.img_paths]