import os

from .engine import Engine
from .config import write_yaml
from .utils.helper import auto_rename

def run(args, cfg, module, train_loader, valid_loader=None, test_loader=None):
    engine = Engine(module, save_dir=args.save_dir)
    engine.config(**cfg['engine'])
     
    if args.mode == 'train':
        cfg['engine'].update(engine.cfg._asdict())
        write_yaml(cfg, os.path.join(args.save_dir, 'config.yaml'))  # override the default config.yaml
        write_yaml(cfg, auto_rename(os.path.join(args.save_dir, 'history', 'config.yaml')))  # save a copy
        
        if args.resume:
            engine.resume(args.resume, base_dir=args.resume_dir)
        engine.train(train_loader, valid_loader=valid_loader)
    else:
        assert test_loader is not None, 'test loader is not provided'
        engine.resume(args.resume)
        engine.test(test_loader)

        cfg['engine'].update(engine.cfg._asdict())
        write_yaml(cfg, os.path.join(args.save_dir, 'config.yaml'))