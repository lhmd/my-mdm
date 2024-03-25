from . import samplers
from .dataset_catalog import DatasetCatalog
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config.config import cfg
from torch.utils.data import DataLoader, ConcatDataset


torch.multiprocessing.set_sharing_strategy('file_system')

def _dataset_factory(is_train, is_val):
    if is_val:
        module = cfg.val_dataset_module
        path = cfg.val_dataset_path
    elif is_train:
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
    dataset = imp.load_source(module, path).Dataset
    return dataset

def get_dataset_class(name):
    if name == "amass":
        pass
    elif name == "uestc":
        from .mdm.a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .mdm.a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from lib.datasets.mdm.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from lib.datasets.mdm.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def make_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter,
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_distributed=False, max_iter=-1, hml_mode='train'):
    batch_size = cfg.batch_size
    name = cfg.dataset
    num_frames = cfg.num_frames
    
    dataset = make_dataset(name, num_frames, hml_mode=hml_mode)
    collator = make_collator(name, hml_mode=hml_mode)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=8,
                             drop_last=True,
                             collate_fn=collator)

    return data_loader

def get_data_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = make_dataset(name, num_frames, split, hml_mode)
    collate = make_collator(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader