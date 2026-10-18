import os
import datetime
import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from custom_datasets.build import build_dataset
from utils.data_collate_fn import collate_fn
from utils.distributed import *
from sam3 import build_sam3_image_model
from sam3.model.utils.misc import copy_data_to_device
from utils.checkpoint_utils import *

import warnings
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()` was not necessary")


def calculate_multiclass_ce_dice_loss(pred_logits, target_masks, ignore_index=255, eps=1e-5):
    B, K, H, W = pred_logits.shape
    
    ce_loss = F.cross_entropy(pred_logits, target_masks, ignore_index=ignore_index)
    
    valid_masks = (target_masks != ignore_index)  # [B, H, W]
    if not valid_masks.any():
        return ce_loss, torch.tensor(0.0, dtype=pred_logits.dtype, device=pred_logits.device)
    
    pred_probs = F.softmax(pred_logits, dim=1)  # [B, K, H, W]
    
    pred_probs_flat = pred_probs.permute(0, 2, 3, 1).reshape(-1, K)  # [B*H*W, K]
    target_flat = target_masks.reshape(-1)  # [B*H*W]
    valid_flat = valid_masks.reshape(-1)    # [B*H*W]
    
    pred_probs_valid = pred_probs_flat[valid_flat]  # [N, K], where N is the number of valid pixels
    target_valid = target_flat[valid_flat]          # [N]
    target_one_hot_valid = F.one_hot(target_valid, num_classes=K).float()  # [N, K]
    
    intersection = torch.sum(pred_probs_valid * target_one_hot_valid, dim=0)  # [K]
    cardinality = torch.sum(pred_probs_valid + target_one_hot_valid, dim=0)   # [K]
    
    dice_coeff = torch.where(
        cardinality > 0,
        (2. * intersection + eps) / (cardinality + eps),
        torch.tensor(1.0, device=pred_logits.device)
    )
    dice_loss = (1.0 - dice_coeff).mean()
    
    return ce_loss, dice_loss

def main(config):
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        if config['model'].get('adapter_resume_path') is not None:
            output_dir = os.path.dirname(config['model']['adapter_resume_path'])
            print(f"🔄 Detected a resume configuration, restoring the log directory and continuing TensorBoard: {output_dir}")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{config['project']['name']}_{timestamp}"
            output_dir = os.path.join(config['project']['output_dir'], run_name)
            os.makedirs(output_dir, exist_ok=True)
            
        writer = SummaryWriter(log_dir=output_dir)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    else:
        output_dir = None
        writer = None

    dataset = build_dataset(config['data'])
    
    sampler = DistributedSampler(dataset, shuffle=config['data']['shuffle']) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['data']['pin_memory'],
        drop_last=config['data']['drop_last']
    )

    model = build_sam3_image_model(bpe_path=config['model']['bpe_path'], checkpoint_path=config['model']['sam3_checkpoint'], use_bg=True if config['data']['bg_idx'] is not None else False)
    global_step = 0
    if config['model'].get('adapter_resume_path') is not None:
        resume_path = config['model']['adapter_resume_path']
        print(f"🔄 Loading adapter checkpoint from {resume_path}...")
        model, _, resumed_step = load_adapter_checkpoint(model, None, resume_path, device='cpu')
        if resumed_step is not None:
            global_step = resumed_step
            print(f"✅ Successfully restored global_step to: {global_step}")
    model = set_trainable_params(model, use_bg=True if config['data']['bg_idx'] is not None else False)
    
    model.to(device)

    model_without_ddp = model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=config['train']['lr'], 
        weight_decay=config['train']['weight_decay']
    )
    if config['model'].get('adapter_resume_path') is not None:
        resume_path = config['model']['adapter_resume_path']
        print(f"🔄 Loading optimizer state from {resume_path}...")
        _, optimizer, _ = load_adapter_checkpoint(None, optimizer, resume_path, device='cpu')
        print("✅ Optimizer momentum state has been restored")
    
    steps_per_epoch = len(dataloader)
    epochs = config['train'].get('epochs', 5)
    total_steps = config['train'].get('total_steps', epochs * steps_per_epoch)

    epoch = global_step // steps_per_epoch
    batches_to_skip = global_step % steps_per_epoch

    loss_weights = config['train']['loss_weights']
    save_steps = config['train'].get('save_steps', 100)
    accumulation_steps = config['train'].get('accumulation_steps', 4)

    while global_step < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for step_in_epoch, (metadatas, batched, gt_masks, presence_labels) in enumerate(dataloader):
            if batches_to_skip > 0:
                batches_to_skip -= 1
                continue

            if global_step >= total_steps:
                break

            batched = copy_data_to_device(batched, device, non_blocking=True)
            gt_masks = gt_masks.to(device) # [K, H, W]
            presence_labels = presence_labels.to(device) # [K] 

            outputs = model(batched)

            dense_logits = outputs[0]["semantic_seg"]    # [B, K+1, H_pred, W_pred]
            presence_logits = outputs[0]["presence_logit"] # [B, K]
            
            h, w = gt_masks.shape[-2:]
            
            if dense_logits.shape[-2:] != (h, w):
                dense_logits = F.interpolate(dense_logits, size=(h, w), mode="bilinear", align_corners=False)
            
            ce_loss, dice_loss = calculate_multiclass_ce_dice_loss(dense_logits, gt_masks)
            
            presence_loss = F.binary_cross_entropy_with_logits(presence_logits, presence_labels)

            raw_loss = loss_weights['ce'] * ce_loss + loss_weights['dice'] * dice_loss + loss_weights['presence'] * presence_loss

            loss_scaled = raw_loss / accumulation_steps
            loss_scaled.backward()
            
            if (step_in_epoch + 1) % accumulation_steps == 0 or (step_in_epoch + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            if is_main_process() and global_step % config['train']['log_interval'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Step [{global_step}/{total_steps}]) LR: {current_lr:.2e} | "
                      f"Loss: {raw_loss.item():.4f} (CE: {ce_loss.item():.4f}, Dice: {dice_loss.item():.4f}, Presence: {presence_loss.item():.4f})")
                
                writer.add_scalar('Loss/Total', raw_loss.item(), global_step)
                writer.add_scalar('Loss/CE', ce_loss.item(), global_step)
                writer.add_scalar('Loss/Dice', dice_loss.item(), global_step)
                writer.add_scalar('Loss/Presence', presence_loss.item(), global_step)
                writer.add_scalar('Train/LR', current_lr, global_step)

            if is_main_process() and global_step % save_steps == 0:
                save_path = os.path.join(output_dir, f"cdca_step_{global_step}.pth")
                save_adapter_checkpoint(model_without_ddp, optimizer, global_step, save_path)

        epoch += 1

    if is_main_process():
        final_save_path = os.path.join(output_dir, "cdca_final.pth")
        save_adapter_checkpoint(model_without_ddp, optimizer, global_step, final_save_path)
        writer.close()
        
    cleanup_distributed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DCR-SAM Finetuning")
    parser.add_argument('--config', type=str, default='', help='path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)