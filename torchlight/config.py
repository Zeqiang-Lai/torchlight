import argparse
from pathlib import Path
import json
import yaml
import sys
import os

from munch import Munch

def basic_args(description=''):
    """
    fresh new training:
        python run.py train -c [config.yaml] -s [save_dir]
    resume training:
        python run.py train -s [save_dir] -r latest
    resume training with overrided config
        python run.py train -c [override.yaml] -s [save_dir] -r latest  # this will override the original config
        python run.py train -c [config.yaml] -s [save_dir] -r latest -n [new_save_dir]
    test
        python run.py test -s [save_dir] -r best
    test with overrided config
        python run.py test -s [save_dir] -r best -c config.yaml # test won't override the original config, but save the override config in test directory
        python run.py test -s [save_dir] -r best -n new_save_dir # save in a new place
    """

    args = argparse.ArgumentParser(description=description)
    args.add_argument('mode', type=str, help='running mode',
                      choices=['train', 'test'])
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='resume to # checkpoint (default: None), e.g. best | latest | epoch# or a complete path')
    args.add_argument('-d', '--device', default='cuda', type=str,
                      help='indices of GPUs to enable (default: cuda)')
    args.add_argument('-s', '--save_dir', default='saved', type=str, required=True,
                      help='path of log directory (default: saved)')
    args.add_argument('-n', '--new_save_dir', default=None, type=str,
                      help='path of new log directory (default: new_saved)')
    args = args.parse_args()

    vars(args)['resume_dir'] = args.save_dir
    if args.new_save_dir is not None:
        args.save_dir = args.new_save_dir

    if args.mode == 'test':
        assert args.resume is not None, 'resume cannot be None in test mode'

    resume_config_path = Path(args.resume_dir) / 'config.yaml'
    if resume_config_path.exists():
        cfg = read_yaml(resume_config_path)
        if args.config:
            cfg.update(read_yaml(args.config))
    else:
        if args.config is None:
            print('warning: default config not founded, forgot to specify a configuration file?')
            cfg = {'engine': {}}
        else:
            cfg = read_yaml(args.config)
    
    os.makedirs(os.path.join(args.save_dir, 'history'), exist_ok=True)
    with open(os.path.join(args.save_dir, 'history', 'cmd.log'), 'a') as f:
        cmd = 'python ' + ' '.join(sys.argv) + '\n'
        f.write(cmd)
    
    return args, Munch.fromDict(cfg)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.safe_dump(content, handle, indent=4, sort_keys=False)
