import os
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from sam3.model.utils.misc import copy_data_to_device
from custom_datasets.build import build_dataset
from models.build import build_model
from models.sam3_wrapper import SAM3SemanticSegWrapper
from utils.config import load_config
from utils.logger import setup_logger
from utils.distributed import setup_distributed, is_main_process, cleanup_distributed
from utils.metrics import DistributedConfusionMatrix
from utils.data_collate_fn import collate_fn
from utils.checkpoint_utils import load_adapter_checkpoint
from vis.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description="DCR-SAM Segmentation Eval")
    parser.add_argument("--config", type=str, default="", help="Path to the yaml config file")
    return parser.parse_args()

def main():
    args = parse_args()

    local_rank, global_rank, world_size = setup_distributed()
    device = f'cuda:{local_rank}'

    cfg = load_config(args.config)
    
    logger = setup_logger("SAM3_OV", save_dir="logs", distributed_rank=global_rank)
    
    if is_main_process():
        logger.info("========== Start Evaluation ==========")
        logger.info(f"Config: {cfg}")
        logger.info(f"World Size: {world_size}")


    if is_main_process():
        logger.info("Building {} dataset...".format(cfg['dataset']['name']))
    dataset = build_dataset(cfg['dataset'])
    
    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['eval']['batch_size'], 
        sampler=sampler, 
        num_workers=cfg['eval']['num_workers'], 
        pin_memory=cfg['dataset']['pin_memory'],
        drop_last=cfg['dataset']['drop_last'],
        collate_fn=collate_fn
    )

    if is_main_process():
        logger.info("Building {} model...".format(cfg['model']['name']))

    backbone = build_model(cfg['model'], use_bg=True if cfg['dataset']['bg_idx'] is not None else False)
    backbone, _, _ = load_adapter_checkpoint(backbone, None, cfg['model']['adapter_path'], device='cpu')
    model = SAM3SemanticSegWrapper(backbone, prob_thresh=cfg['model']['prob_thresh'], device=device)

    conf_mat = DistributedConfusionMatrix(
        num_classes=cfg['dataset']['num_classes'], 
        ignore_index=cfg['dataset']['ignore_index'], 
        device=device
    )

    visualizer = Visualizer(save_dir=os.path.join(os.path.dirname(cfg['model']['adapter_path']), "vis_img"), num_classes=cfg['dataset']['num_classes'], bg_idx=cfg['dataset']['bg_idx']) if cfg['eval'].get('vis', False) else None
    if is_main_process():
        logger.info("Starting inference loop...")
        pbar = tqdm(total=len(dataloader), desc="Evaluating")

    with torch.no_grad():
        for batch_idx, (metadatas, batched, gt_masks, _) in enumerate(dataloader):
            batched = copy_data_to_device(batched, device)
            gt_masks = gt_masks.to(device)

            pred_masks = model(metadatas, batched)
            if visualizer is not None:
                visualizer.visual(metadatas, gt_masks, pred_masks)
            
            conf_mat.update(pred_masks, gt_masks)

            if is_main_process():
                pbar.update(1)

    if is_main_process():
        pbar.close()

    if world_size > 1:
        torch.distributed.barrier()
    
    if is_main_process():
        logger.info("Reducing metrics across all GPUs...")
    conf_mat.reduce_from_all_processes()

    if is_main_process():
        text_prompts = dataset.raw_class_names
        iou_per_class, tp, union = conf_mat.compute_iou()
        
        valid_classes = union > 0
        mIoU = iou_per_class[valid_classes].mean().item() * 100
        
        logger.info("\n" + "="*50)
        logger.info(f"Evaluation Completed! Global mIoU: {mIoU:.2f}%")
        logger.info("="*50)
        
        logger.info("--- Sample Class IoU ---")
        for i in range(cfg['dataset']['num_classes']):
            if valid_classes[i]:
                logger.info(f"Class [{text_prompts[i]}]: IoU = {iou_per_class[i].item()*100:.2f}%")
        
        with open("logs/eval_summary.txt", "a") as f:
            f.write(f"mIoU: {mIoU:.2f}%\n")
            for i in range(cfg['dataset']['num_classes']):
                if valid_classes[i]:
                    f.write(f"{text_prompts[i]} = {iou_per_class[i].item()*100:.2f}%\n")

    cleanup_distributed()

if __name__ == "__main__":
    main()