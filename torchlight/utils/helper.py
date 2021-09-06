import json
import time
from pathlib import Path
from collections import OrderedDict
from functools import partial

import torch
import numpy as np


def get_obj(info, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in info, and returns the
    instance initialized with corresponding arguments given.

    `object = get_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    module_name = info['type']
    module_args = dict(info['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def get_ftn(info, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in info, and returns the
    function with given arguments fixed with functools.partial.

    `function = get_ftn('name', module, a, b=1)`
    is equivalent to
    `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
    """
    module_name = info['type']
    module_args = dict(info['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return partial(getattr(module, module_name), *args, **module_args)


def setup_mannul_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def adjust_learning_rate(optimizer, lr):    
    print('Adjust Learning Rate => %.4e' %lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['initial_lr'] = lr

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device).float()
    if isinstance(data, (list, tuple)):
        return map(partial(to_device, device=device), data)
    if isinstance(data, dict):
        return {k: to_device(v, device=device) for k, v in data.items()}
    return data

def load_checkpoint(model, ckpt_path):
    model.load_state_dict(torch.load(ckpt_path)['module']['model'])

class timer:
    _start_time = 0
    
    @classmethod
    def tic(cls):
        cls._start_time = time.time()

    @classmethod
    def tok(cls):
        now = time.time()
        used = int(now - cls._start_time)
        second = used % 60
        used = used // 60
        minutes = used % 60
        used = used // 60
        hours = used
        return "{}:{}:{}".format(hours, minutes, second)