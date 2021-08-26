import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
import argparse

from ..logging import setup_logging
from . import backend


class Config:
    _backend = 'yaml'
    _ext = 'yaml'
    _read = getattr(backend, 'read_{}'.format(_backend))
    _write = getattr(backend, 'write_{}'.format(_backend))
    
    @classmethod
    def backend(cls, another=None, ext=''):
        if another is None:
            return cls._backend
        if another not in backend.Choices:
            raise ValueError('Invalid backend, choose from ' + str(backend.Choices))
        cls._backend = another
        cls._ext = ext
        cls._read = getattr(backend, 'read_{}'.format(another))
        cls._write = getattr(backend, 'write_{}'.format(another))
        return cls._backend

    @classmethod
    def basic_args(cls, description):
        args = argparse.ArgumentParser(description=description)
        args.add_argument('mode', type=str, help='running mode', choices=['train', 'test'])
        args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
        args.add_argument('-n', '--name', default=None, type=str,
                        help='name of log directory (default: None)')
        return args
    
    
    @classmethod
    def default(cls, description):
        args = cls.basic_args(description)
        return cls.from_args(args)
    
    @classmethod
    def from_args(cls, args=None, options=[]):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """ 
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        train = args.mode == 'train'
         
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.{}'.format(cls._ext)
            run_id = args.name
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.yaml', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
            run_id = None
        
        config = cls._read(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(cls._read(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, train, resume, modification, run_id)

    def __init__(self, config, train, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self.train = train

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        else:
            run_id += '_' + datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / exper_name / run_id  / 'models'
        self._log_dir = save_dir / exper_name / run_id  / 'log'
        
        self.config['engine']['log_dir'] = self.log_dir
        self.config['engine']['save_dir'] = self.save_dir
        
        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        Config._write(self.config, self.save_dir / 'config.{}'.format(Config._ext))
        
    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
    

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
