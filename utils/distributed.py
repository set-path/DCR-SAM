import os
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize a torchrun (DDP) distributed environment."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        
        torch.cuda.set_device(local_rank)
        
        try:
            dist.init_process_group(
                backend="nccl", 
                device_id=torch.device(f"cuda:{local_rank}")
            )
        except TypeError:
            dist.init_process_group(backend="nccl")
            
        return local_rank, rank, world_size
    else:
        return 0, 0, 1

def is_dist_avail_and_initialized():
    """Check whether distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def cleanup_distributed():
    """Clean up the distributed environment."""
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()