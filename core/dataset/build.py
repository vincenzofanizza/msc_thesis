'''
Code taken from the SPEED+ baseline repository: https://github.com/tpark94/speedplusbaseline.

'''

import torch

from objects import ObjDetDataset, KeyDetDataset, SpeedplusAugmentCfg


def build_dataset(cfg, is_train = True, load_labels = True):

    augment_cfg = SpeedplusAugmentCfg(cfg)
    transforms = augment_cfg.build_transforms(is_train = is_train, load_labels = load_labels)

    if cfg.MODEL.TYPE == 'KD':
        dataset = KeyDetDataset(cfg, transforms = transforms, is_train = is_train)
    elif cfg.MODEL.TYPE == 'OD':
        dataset = ObjDetDataset(cfg, transforms = transforms, is_train = is_train)

    return dataset

def build_dataloader(cfg, is_train = True, is_source = True, load_labels = True):
    if is_train:
        # TODO: include batch size and number of workers in configuration
        images_per_gpu = cfg.batch_size
        shuffle = True
        num_workers = cfg.num_workers
    else:
        images_per_gpu = 1
        shuffle = False
        num_workers = 1

    dataset = build_dataset(cfg, is_train, is_source, load_labels)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_gpu,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return data_loader