import torch
import torch.distributed as dist
from .distributed import is_dist_avail_and_initialized

class DistributedConfusionMatrix:
    def __init__(self, num_classes, ignore_index=255, device='cuda'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    def update(self, preds, targets):
        n = self.num_classes
        valid_mask = (targets >= 0) & (targets < n)
        inds = n * targets[valid_mask] + preds[valid_mask]
        
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reduce_from_all_processes(self):
        if is_dist_avail_and_initialized():
            dist.all_reduce(self.mat, op=dist.ReduceOp.SUM)

    def compute_iou(self):
        h = self.mat.float()
        tp = torch.diag(h)
        fp = h.sum(dim=0) - tp
        fn = h.sum(dim=1) - tp
        
        union = tp + fp + fn
        iou = tp / (union + 1e-6)
        
        return iou, tp, union