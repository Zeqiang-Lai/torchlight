import json
from logging import log
from os import stat
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from numpy import inf
import numpy as np

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

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

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


class MetricTracker:
    def __init__(self, writer=None):
        self.writer = writer
        self._data = {}
        self.reset()

    def reset(self):
        self._data = {}
        
    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        if key not in self._data.keys():
            self._data[key] = {'total': 0, 'count': 0}
        self._data[key]['total'] += value * n
        self._data[key]['count'] += n

    def avg(self, key):
        return self._data[key]['total'] / self._data[key]['count']

    def result(self):
        return {k: self._data[k]['total'] / self._data[k]['count'] for k in self._data.keys()}
    
    def summary(self):
        items = ['{}: {:.8f}'.format(k, v) for k, v in self.result().items()]
        return ' '.join(items)

        
        
class PerformanceMonitor:
    def __init__(self, mnt_mode, early_stop_threshold):
        self.mnt_mode = mnt_mode
        self.early_stop_threshold = early_stop_threshold
        
        assert self.early_stop_threshold > 0, 'early_stop_threshold should be greater than 0'
        assert self.mnt_mode in ['min', 'max']
        
        self.reset()
        
    def update(self, metric):
        improved = (self.mnt_mode == 'min' and metric <= self.mnt_best) or \
                   (self.mnt_mode == 'max' and metric >= self.mnt_best)
        self.best = False
        if improved:
            self.mnt_best = metric
            self.not_improved_count = 0
            self.best = True
        else:
            self.not_improved_count += 1    
                    
    def is_best(self):
        return self.best == True
    
    def should_early_stop(self):
        return self.not_improved_count > self.early_stop_threshold
    
    def reset(self):
        self.not_improved_count = 0
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.best = False
        
    def state_dict(self):
        return {'not_improved_count': self.not_improved_count, 'mnt_best': self.mnt_best, 'best': self.best}
    
    def load_state_dict(self, states):
        self.not_improved_count = states['not_improved_count']
        self.mnt_best = states['mnt_best']
        self.best = states['best']