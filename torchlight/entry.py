import os
from typing import Union

from torch.utils.data import DataLoader

from .core.module import Module
from .core.engine import Engine
from .core.config import write_yaml
from .utils.helper import auto_rename

def run(args, cfg, 
        module:Module, 
        train_loader:DataLoader, 
        valid_loader:DataLoader=None, 
        test_loader:Union[DataLoader, dict]=None):
    """ test_loader can be a dict of Dataloaders in the format of {name: loader}
        ```python
        run(args, cfg, module, train_loader, valid_loader, test_loader)
        run(args, cfg, module, train_loader, valid_loader, {'a': test_loader1})
        ``` 
    """
    
    engine = Engine(module, save_dir=args.save_dir)
    engine.config(**cfg['engine'])
     
    if args.mode == 'train':
        cfg['engine'].update(engine.cfg._asdict())
        write_yaml(cfg, os.path.join(args.save_dir, 'config.yaml'))  # override the default config.yaml
        write_yaml(cfg, auto_rename(os.path.join(args.save_dir, 'history', 'config.yaml')))  # save a copy
        
        if args.resume:
            engine.resume(args.resume, base_dir=args.resume_dir)
        engine.train(train_loader, valid_loader=valid_loader)
    elif args.mode == 'test':
        assert test_loader is not None, 'test loader is not provided'
        engine.resume(args.resume)
        
        if isinstance(test_loader, dict):
            for name, dataloader in test_loader.items():
                engine.logger.info('Start testing {} ...'.format(name))
                engine.test(dataloader, name)
        else:
            engine.test(test_loader)

        cfg['engine'].update(engine.cfg._asdict())
        write_yaml(cfg, os.path.join(args.save_dir, 'config.yaml'))
    else:
        engine.set_debug_mode()
        engine.train(train_loader, valid_loader=valid_loader)