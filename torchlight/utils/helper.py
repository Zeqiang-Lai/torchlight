import json
from pathlib import Path
from collections import OrderedDict
from functools import partial
import os
from operator import attrgetter


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


def auto_rename(path):
    count = 1
    new_path = path
    while True:
        if not os.path.exists(new_path):
            return new_path
        file_name = os.path.basename(path)
        name, ext = file_name.split('.')
        new_file_name = '{}_{}.{}'.format(name, count, ext)
        new_path = os.path.join(os.path.dirname(path), new_file_name)
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
