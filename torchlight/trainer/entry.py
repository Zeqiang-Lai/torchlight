import os
import sys
import datetime

from .module import Module
from .engine import Engine
from .config import write_yaml
from .util import auto_rename


class DataSource:
    def train_loader(self):
        raise NotImplementedError

    def valid_loader(self):
        return self.test_loader()

    def test_loader(self):
        return None


def backup_cmd(cfg, save_dir):
    os.makedirs(os.path.join(save_dir, 'history'), exist_ok=True)
    path = auto_rename(os.path.join(save_dir, 'history', 'config.yaml'))
    write_yaml(cfg, path)  # save a copy

    with open(os.path.join(save_dir, 'history', 'cmd.log'), 'a') as f:
        dt = datetime.datetime.now()
        dt = dt.strftime("%Y-%m-%d %H:%M:%S")
        save = dt + '\n' 
        save += '  cmd: python ' + ' '.join(sys.argv) + '\n'
        save += '  config: ' + path + '\n'
        f.write(save)
 

def run_lazy(args, cfg, module: Module, data_source: DataSource):
    cfg_engine = cfg.get('engine', {})
    engine = Engine(module, save_dir=args.save_dir, **cfg_engine)

    if args.mode == 'train':
        cfg_engine.update(engine.cfg._asdict())
        write_yaml(cfg, os.path.join(args.save_dir, 'config.yaml'))  # override the default config.yaml

        if args.resume:
            engine.resume(args.resume)
        engine.train(data_source.train_loader(), valid_loader=data_source.valid_loader())
    elif args.mode == 'test':
        test_loader = data_source.test_loader()
        assert test_loader is not None, 'test loader is not provided'
        engine.resume(args.resume)

        if isinstance(test_loader, dict):
            for name, dataloader in test_loader.items():
                engine.logger.info('Start testing {} ...'.format(name))
                engine.test(dataloader, name)
        else:
            engine.test(test_loader)

        cfg['engine'].update(engine.cfg._asdict())
        write_yaml(cfg, os.path.join(engine.test_log_dir, 'config.yaml'))
    else:
        engine.set_debug_mode()
        engine.train(data_source.train_loader(), valid_loader=data_source.valid_loader())

    backup_cmd(cfg, save_dir=args.save_dir)


def run(args, cfg, module, train_loader, valid_loader = None, test_loader = None):
    """ test_loader can be a dict of Dataloaders in the format of {name: loader}
        ```python
        run(args, cfg, module, train_loader, valid_loader, test_loader)
        run(args, cfg, module, train_loader, valid_loader, {'a': test_loader1})
        ``` 
    """
    class DataSource:
        def train_loader(self):
            return train_loader

        def valid_loader(self):
            return valid_loader

        def test_loader(self):
            return test_loader

    run_lazy(args, cfg, module, DataSource())
