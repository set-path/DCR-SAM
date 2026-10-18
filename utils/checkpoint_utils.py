import torch

def set_trainable_params(model, use_bg=True):
    trainable_keys = [
        'task_prompt_embed',
        'segmentation_head.vgtp_attn_proj.0.weight',
        'segmentation_head.vgtp_attn_proj.0.bias',
        'segmentation_head.vgtp_attn_proj.2.weight',
        'segmentation_head.vgtp_attn_proj.2.bias'
    ]
    if use_bg:
        trainable_keys.append('segmentation_head.bg_token')

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        for target_key in trainable_keys:
            if name == target_key:
                param.requires_grad = True
                break

    for name, param in model.vctr.named_parameters():
        param.requires_grad = True

    return model

def save_adapter_checkpoint(model, optimizer, global_step, save_path):
    trainable_params = {
        name: param.state_dict() if hasattr(param, 'state_dict') else param.data
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    torch.save({
        'step': global_step,
        'adapter_parms': trainable_params,
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)


def load_adapter_checkpoint(model, optimizer, load_path, device='cuda'):
    """
    Load a checkpoint containing only trainable parameters.
    Args:
        model: Model instance after freezing parameters.
        optimizer: Optional optimizer instance.
        load_path: Checkpoint path.
        device: Load device.
    Returns:
        model: Model with adapter parameters loaded.
        optimizer: Optimizer with restored state, if provided.
        global_step: Restored training step.
    """
    checkpoint = torch.load(load_path, map_location=device)
    global_step = checkpoint['step']
    adapter_state_dict = checkpoint['adapter_parms']

    if model is not None:
        model_state_dict = model.state_dict()
        load_dict = {
            k: v for k, v in adapter_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(load_dict)
        model.load_state_dict(model_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    return model, optimizer, global_step