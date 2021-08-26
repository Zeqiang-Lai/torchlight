import torch
from torchvision import datasets, transforms
import os
import torchlight
from torchlight.config import basic_args, write_yaml

from net import NetModule

if __name__ == '__main__':
    args, cfg = basic_args('MNIST Classification')
    
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=cfg['train_loader']['batch_size'], shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), num_workers=4, **cfg['test_loader'])

    module = NetModule(device=torch.device(args.device), **cfg['module'])
    engine = torchlight.Engine(module, save_dir=args.save_dir)
    engine.config(**cfg['engine'])
    
    cfg['engine'].update(engine.cfg._asdict())
    write_yaml(cfg, os.path.join(args.save_dir, 'config.yaml'))
    
    if args.mode == 'train':
        if args.resume:
            engine.resume(args.resume, base_dir=args.resume_dir)
        engine.train(train_loader, valid_loader=test_loader)
    else:
        engine.resume(args.resume)
        engine.test(test_loader)
