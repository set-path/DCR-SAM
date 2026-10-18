import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time

class SAM3SemanticSegWrapper(nn.Module):
    def __init__(self, backbone, prob_thresh=0.1, device='cuda'):
        super().__init__()
        self.prob_thresh = prob_thresh
        self.device = device
        
        self.backbone = backbone

    @torch.no_grad()
    def forward(self, metadatas, input):
        outputs = self.backbone(input)
        outputs = outputs[0]

        img_path = metadatas[0]['img_path']
        h, w = metadatas[0]['original_h'], metadatas[0]['original_w']

        dense_logits = outputs['semantic_seg']
        presence_logits = outputs['presence_logit']
        K = dense_logits.shape[1] - 1

        dense_logits = F.interpolate(
            dense_logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        dense_probs = F.softmax(dense_logits, dim=1).squeeze(0)

        seg_pred = torch.argmax(dense_probs, dim=0)

        if hasattr(self, 'prob_thresh'):
            max_vals = dense_probs.max(dim=0)[0]
            seg_pred[max_vals < self.prob_thresh] = K

        return seg_pred.unsqueeze(0)