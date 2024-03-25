from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
from lib.config import cfg
from lib.datasets.mdm.tensors import collate as all_collate
from lib.datasets.mdm.tensors import t2m_collate

_collators = {}

def make_collator(name, hml_mode='train'):
    
    if hml_mode == 'gt':
        from lib.datasets.mdm.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate
