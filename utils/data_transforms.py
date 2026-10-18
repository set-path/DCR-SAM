import random
import torch
import torchvision.transforms.v2.functional as Fv2
import torchvision.transforms.functional as F
from typing import Iterable
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors

def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)

def resize(image, mask, size, max_size=None, square=False, v2=False):

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = image.size
        size = get_size(cur_size, size, max_size)

    if v2:
        image = Fv2.resize(
            image, size, antialias=True
        )
    else:
        image = F.resize(image, size)

    if mask is not None:
        mask = F.resize(mask, size, interpolation=F.InterpolationMode.NEAREST)
    return image, mask

def pad(image, mask, padding, v2=False):
    old_w, old_h = image.size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        if v2:
            image = Fv2.pad(
                image, (0, 0, padding[0], padding[1])
            )
        else:
            image = F.pad(
                image, (0, 0, padding[0], padding[1])
            )
        h += padding[1]
        w += padding[0]
    else:
        if v2:
            # left, top, right, bottom
            image = Fv2.pad(
                image,
                (padding[0], padding[1], padding[2], padding[3]),
            )
        else:
            # left, top, right, bottom
            image = F.pad(
                image,
                (padding[0], padding[1], padding[2], padding[3]),
            )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]
        if mask is not None:
            if v2:
                if len(padding) == 2:
                    mask = Fv2.pad(
                        mask,
                        (0, 0, padding[0], padding[1]),
                        fill=255
                    )
                else:
                    mask = Fv2.pad(
                        mask, tuple(padding), fill=255
                    )
            else:
                if len(padding) == 2:
                    mask = F.pad(
                        mask,
                        (0, 0, padding[0], padding[1]), fill=255
                    )
                else:
                    mask = F.pad(
                        mask, tuple(padding), fill=255
                    )
    return image, mask

class RandomResizeAPI:
    def __init__(
        self, sizes, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.v2 = v2

    def __call__(self, image, mask = None):
        size = random.choice(self.sizes)
        image, mask = resize(
            image, mask, size, self.max_size, square=self.square, v2=self.v2
        )
        return image, mask

class PadToSizeAPI:
    def __init__(self, size, bottom_right=False, v2=False):
        self.size = size
        self.v2 = v2
        self.bottom_right = bottom_right

    def _sample_pad(self, w, h):
        pad_x = self.size - w
        pad_y = self.size - h
        assert pad_x >= 0 and pad_y >= 0
        pad_left = random.randint(0, pad_x)
        pad_right = pad_x - pad_left
        pad_top = random.randint(0, pad_y)
        pad_bottom = pad_y - pad_top
        return pad_left, pad_top, pad_right, pad_bottom

    def __call__(self, image, mask = None):
        w, h = image.size
        if self.bottom_right:
            pad_right = self.size - w
            pad_bottom = self.size - h
            padding = (pad_right, pad_bottom)
        else:
            padding = self._sample_pad(w, h)
        datapoint = pad(image, mask, padding, v2=self.v2)
        return datapoint


class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, image, mask = None):
        if self.v2:
            image = Fv2.to_image_tensor(image)
            mask = Fv2.to_image_tensor(mask) if mask is not None else None
        else:
            image = F.to_tensor(image)
            mask = torch.from_numpy(np.array(mask)).long() if mask is not None else None
        return image, mask

class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, image, mask = None):
        if self.v2:
            image = Fv2.convert_image_dtype(image, torch.float32)
            image = Fv2.normalize(image, mean=self.mean, std=self.std)
        else:
            image = F.normalize(image, mean=self.mean, std=self.std)

        return image, mask
    

class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask = None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def reverse_transform(tensor_img: torch.Tensor) -> Image.Image:
    # 1. Denormalize back to [0, 1].
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(tensor_img.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(tensor_img.device)
    img_denorm = tensor_img * std + mean
    
    # 2. Clamp to [0, 1].
    img_denorm = torch.clamp(img_denorm, 0.0, 1.0)
    
    # 3. Convert tensor to HWC numpy.
    img_np = img_denorm.cpu().detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    
    img_pil = Image.fromarray(img_np)
    
    return img_pil


def visualize_mask(
    mask: torch.Tensor,
    num_classes: int,
    original_img: Image.Image = None,
    alpha: float = 0.5,
    colormap: dict = None,
    return_type: str = "pil"  # "pil" / "numpy" / "tensor"
) -> Image.Image | np.ndarray | torch.Tensor:
    # Step 1: Convert the mask to a numpy array.
    mask_np = mask.cpu().detach().numpy()
    
    # Step 2: Build a color map.
    if colormap is None:
        unique_classes = np.unique(mask_np)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        tab_colors_rgb = [mcolors.hex2color(c) for c in tab_colors]
        tab_colors_rgb = [(int(r*255), int(g*255), int(b*255)) for r, g, b in tab_colors_rgb]
        
        colormap = {}
        for i, cls in enumerate(unique_classes):
            if i < len(tab_colors_rgb):
                colormap[cls] = tab_colors_rgb[i]
            else:
                np.random.seed(cls)
                colormap[cls] = tuple(np.random.randint(0, 256, size=3))
        
        if 255 in colormap:
            colormap[255] = (128, 128, 128)
        colormap[num_classes - 1] = (0, 0, 0)
    
    # Step 3: Map the mask to RGB.
    h, w = mask_np.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in colormap.items():
        rgb_mask[mask_np == cls_idx] = rgb
    
    # Step 4: Blend with the original image if provided.
    if original_img is not None:
        original_img = original_img.resize((w, h), Image.Resampling.BILINEAR)
        original_np = np.array(original_img, dtype=np.float32)
        rgb_mask_float = rgb_mask.astype(np.float32)
        
        fused_np = (original_np * (1 - alpha) + rgb_mask_float * alpha).astype(np.uint8)
        rgb_mask = fused_np
    
    # Step 5: Convert to the requested output type.
    if return_type == "pil":
        result = Image.fromarray(rgb_mask)
    elif return_type == "numpy":
        result = rgb_mask
    elif return_type == "tensor":
        result = torch.from_numpy(rgb_mask).permute(2, 0, 1).float() / 255.0
    else:
        raise ValueError(f"return_type 'pil'/'numpy'/'tensor'")
    
    return result