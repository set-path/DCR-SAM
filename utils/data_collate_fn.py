import torch
from sam3.model.data_misc import FindStage, BatchedDatapoint, BatchedFindTarget, BatchedInferenceMetadata
from dataclasses import dataclass
from typing import List, Optional, Any
import json
from utils.data_transforms import *
from sam3.model.data_misc import FindStage, BatchedFindTarget, BatchedInferenceMetadata

from typing import Union

MyTensor = Union[torch.Tensor, List[Any]]

@dataclass
class MulticlassBatchedDatapoint:
    img_batch: torch.Tensor
    find_text_batch: List[str]
    find_inputs: List[FindStage]
    find_targets: Optional[List[BatchedFindTarget]] = None
    find_metadatas: Optional[List[BatchedInferenceMetadata]] = None
    raw_images: Optional[List[Any]] = None


@dataclass
class MulticlassFindStage:
    img_ids: MyTensor
    img_ids__type = torch.long

    input_boxes: MyTensor
    input_boxes__type = torch.float
    input_boxes_mask: MyTensor
    input_boxes_mask__type = torch.bool
    input_boxes_label: MyTensor
    input_boxes_label__type = torch.long

    input_points: MyTensor
    input_points__type = torch.float
    input_points_mask: MyTensor
    input_points_mask__type = torch.bool

    # We track the object ids referred to by this query.
    # This is beneficial for tracking in videos without the need for pointers.
    object_ids: Optional[List[List]] = None  # List of objects per query

def collate_fn(batch):
    metadatas = [item["metadatas"] for item in batch]
    B = len(metadatas)
    
    images = [item["image"] for item in batch]
    img_batch = torch.stack(images, dim=0)  # [B, 3, H, W]
    
    gt_masks = [item["gt_masks"] for item in batch]
    gt_mask_batch = torch.stack(gt_masks, dim=0)  # [B, H, W]

    presence_label = [item['presence_labels'] for item in batch]
    presence_label_batch = torch.stack(presence_label, dim=0) # [B, K]

    img_ids = torch.arange(B, dtype=torch.long)  # [B]
    
    
    input_boxes = torch.empty((0, B, 4), dtype=torch.float32)
    input_boxes_mask = torch.empty((B, 0), dtype=torch.bool)
    input_boxes_label = torch.empty((0, B), dtype=torch.int64)
    input_points = torch.empty((B, 0, 257), dtype=torch.float32)
    input_points_mask = torch.empty((B, 0), dtype=torch.bool)
    object_ids = [[] for _ in range(B)]
    
    find_stage = MulticlassFindStage(
        img_ids=img_ids,
        input_boxes=input_boxes,
        input_boxes_mask=input_boxes_mask,
        input_boxes_label=input_boxes_label,
        input_points=input_points,
        input_points_mask=input_points_mask,
        object_ids=object_ids
    )

    batched = MulticlassBatchedDatapoint(
        img_batch=img_batch,
        find_text_batch=batch[0]["class_names"],
        find_inputs=[find_stage],
    )
    
    return metadatas, batched, gt_mask_batch, presence_label_batch
