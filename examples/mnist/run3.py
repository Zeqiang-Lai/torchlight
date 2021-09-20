import torch
from torchvision import datasets, transforms
from torchlight import config
from torchlight.entry import run

from module import NetModule

if __name__ == '__main__':
    args, cfg = config.basic_args('MNIST')
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='MNIST', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=cfg['train_loader']['batch_size'], shuffle=True, num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='MNIST', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), num_workers=4, **cfg['test_loader'])

    module = NetModule(device=torch.device(args.device), **cfg['module'])
    
    run(args, cfg, module, train_loader=train_loader, valid_loader=test_loader, test_loader=test_loader)