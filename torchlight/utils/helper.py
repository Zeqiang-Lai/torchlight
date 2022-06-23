import json
import os
import sys
from functools import partial
from operator import attrgetter
from pathlib import Path
from collections import OrderedDict

__all__ = [
    'get_cmd',
    'instantiate',
    'auto_rename'
]


def get_cmd():
    """ get cmd that starts current script"""
    args = ' '.join(sys.argv)
    return f'python {args}'


def instantiate(module, name, *args, **kwargs):
    return attrgetter(name)(module)(*args, **kwargs)


def get_obj(info, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type@' in info, and returns the
    instance initialized with corresponding arguments given.

    `object = get_obj(info['type@'], module)`
    is equivalent to
    `object = module.info['type@'](info.pop('type@'))`
    """
    module_name = info['type@']
    module_args = dict(info)
    module_args.pop('type@')
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return attrgetter(module_name)(module)(*args, **module_args)


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


def auto_rename(path, ignore_ext=False):
    count = 1
    new_path = path
    while True:
        if not os.path.exists(new_path):
            return new_path
        file_name = os.path.basename(path)
        try:
            name, ext = file_name.split('.')
            new_name = f'{name}_{count}.{ext}'
        except:
            new_name = f'{file_name}_{count}'
        if ignore_ext:
            new_name = f'{file_name}_{count}'
        new_path = os.path.join(os.path.dirname(path), new_name)
        count += 1


class ConfigurableLossCalculator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.loss_fns = {}

    def register(self, fn, name):
        self.loss_fns[name] = fn

    def compute(self):
        loss = 0
        loss_dict = {}
        for name, weight in self.cfg.items():
            l = self.loss_fns[name]() * weight
            loss_dict[name] = l.item()
            loss += l
        return loss, loss_dict


def convert_ckpt(src, dst, src_keys, dst_keys):
    import torch
    ckpt = torch.load(src)
    print('src', ckpt.keys())
    ckpt['net'].pop('norm.body.weight')
    ckpt['net'].pop('norm.body.bias')
    module = {}
    for ksrc, kdst in zip(src_keys, dst_keys):
        module[kdst] = ckpt[ksrc]
    torch.save({'module': module}, dst)
