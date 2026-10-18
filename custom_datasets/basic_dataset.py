import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.data_transforms import *
import json
from tqdm import tqdm
from sam3.train.transforms.basic import get_random_resize_scales, get_random_resize_max_size

class BasicDataset(Dataset):
    def __init__(
        self, 
        class_name_path: str,
        ignore_index=255,
        bg_idx=0,
        reduce_zero_label=False,
        mode='train'
    ):
        super().__init__()
        self.class_name_path = class_name_path
        self.ignore_index = ignore_index
        self.bg_idx = bg_idx
        self.reduce_zero_label = reduce_zero_label
        self.mode = mode
        
        with open(self.class_name_path, 'r', encoding='utf-8') as f:
            self.raw_class_names = [line.strip().lower().split(",")[0] for line in f.readlines() if line.strip()]
            
        self.fg_classes = []
        self.fg_class_indices = []
        
        for idx, class_name in enumerate(self.raw_class_names):
            if self.bg_idx is not None and idx == self.bg_idx:
                continue
            
            self.fg_classes.append(class_name)
            self.fg_class_indices.append(idx)

        self.K = len(self.fg_classes)

        if self.mode == 'train':
            self.transforms = ComposeAPI([
                RandomResizeAPI(sizes=get_random_resize_scales(size=1008, min_size=480, rounded=False), max_size=get_random_resize_max_size(size=1008), square=True),
                PadToSizeAPI(size=1008),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.mode == 'eval':
            self.transforms = ComposeAPI([
                RandomResizeAPI(sizes=1008, max_size=1008, square=True),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.__train_getitem__(index)
        elif self.mode == 'eval':
            return self.__eval_getitem__(index)

    def __train_getitem__(self, idx):
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]
        
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(ann_path)
        
        # 1. Reduce Zero Label
        if self.reduce_zero_label:
            mask_np = np.array(mask_pil).copy()
            mask_np[mask_np == 0] = self.ignore_index
            valid_pixels = mask_np != self.ignore_index
            mask_np[valid_pixels] = mask_np[valid_pixels] - 1
            mask_pil = Image.fromarray(mask_np)

        original_w, original_h = img_pil.size
        resized_w, resized_h = 1008, 1008
        
        if self.transforms is not None:
            img_tensor, mask_tensor = self.transforms(img_pil, mask_pil)
        else:
            img_tensor, mask_tensor = img_pil, mask_pil

        gt_masks = torch.full((mask_tensor.shape[0], mask_tensor.shape[1]), 
                              fill_value=self.ignore_index, 
                              dtype=torch.long)
        
        valid_masks = (mask_tensor != self.ignore_index)
        presence_labels = torch.zeros(self.K, dtype=torch.float32)

        gt_masks[valid_masks] = self.K

        for k, class_idx in enumerate(self.fg_class_indices):
            mask_k = (mask_tensor == class_idx)
            gt_masks[mask_k] = k 
            presence_labels[k] = 1.0 if mask_k.sum() > 0 else 0.0

        metadatas = {
            "img_path": img_path,
            "ann_path": ann_path,
            "original_h": original_h, "original_w": original_w,
            "resized_h": resized_h, "resized_w": resized_w,
        }
        
        return {
            "metadatas": metadatas,
            "image": img_tensor,                
            "gt_masks": gt_masks,               
            "presence_labels": presence_labels, 
            "class_names": self.fg_classes
        }
    
    def __eval_getitem__(self, idx):
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(ann_path)
        mask_np = np.array(mask_pil).copy()
        
        if self.reduce_zero_label:
            mask_np[mask_np == 0] = self.ignore_index
            valid_pixels = mask_np != self.ignore_index
            mask_np[valid_pixels] = mask_np[valid_pixels] - 1

        original_w, original_h = img_pil.size
        resized_w, resized_h = 1008, 1008
        
        if self.transforms is not None:
            img_tensor, _ = self.transforms(img_pil)
        
        gt_masks = torch.full((mask_np.shape[0], mask_np.shape[1]), 
                              fill_value=self.ignore_index, 
                              dtype=torch.long)
        
        valid_masks = (mask_np != self.ignore_index)
        presence_labels = torch.zeros(self.K, dtype=torch.float32)

        gt_masks[valid_masks] = self.K
        
        for k, class_idx in enumerate(self.fg_class_indices):
            mask_k = (mask_np == class_idx)
            gt_masks[mask_k] = k 
            presence_labels[k] = 1.0 if mask_k.sum() > 0 else 0.0

        metadatas = {
            "img_path": img_path,
            "ann_path": ann_path,
            "original_h": original_h, "original_w": original_w,
            "resized_h": resized_h, "resized_w": resized_w,
        }
        
        return {
            "metadatas": metadatas,
            "image": img_tensor,                
            "gt_masks": gt_masks,               
            "presence_labels": presence_labels, 
            "class_names": self.fg_classes
        }