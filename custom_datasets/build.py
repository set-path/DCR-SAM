from .ade20k import ADE20KDataset
from .city_scapes import CityscapesDataset
from .context import ContextDataset
from .coco import COCODataset
from .loveda import LoveDADataset
from .isaid import iSAIDDataset
from .potsdam import PotsdamDataset
from .vaihingen import VaihingenDataset

_dataset_maps = {
    "ADE20K": ADE20KDataset,
    "Cityscapes": CityscapesDataset,
    "Context60": ContextDataset,
    "Context59": ContextDataset,
    "COCO_obj": COCODataset,
    "COCO_stuff": COCODataset,
    "LoveDA": LoveDADataset,
    "iSAID": iSAIDDataset,
    "Potsdam": PotsdamDataset,
    "Vaihingen": VaihingenDataset
}

def build_dataset(dataset_info):
    """Dataset factory."""
    dataset_name = dataset_info['name']
    dataset_cls = _dataset_maps.get(dataset_name)
    if dataset_cls is None:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported yet.")
    return dataset_cls(
        root=dataset_info['root'],
        split=dataset_info['split'],
        mode=dataset_info['mode'],
        class_name_path=dataset_info['class_name_path'],
        bg_idx=dataset_info['bg_idx'],
        ignore_index=dataset_info['ignore_index'],
        reduce_zero_label=dataset_info['reduce_zero_label']
    )
