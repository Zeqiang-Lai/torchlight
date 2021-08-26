import torch
from torchvision import datasets, transforms


import torchlight
from torchlight.config import read_yaml

from net import NetModule

if __name__ == '__main__':
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])), batch_size=64, shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=True, num_workers=4)

    device = torch.device('cuda')
    module = NetModule(cfg={'lr': 0.01}, device=device)
    engine = torchlight.Engine(module, save_dir='experiments/simple_l1')
    engine.config(read_yaml('config.yaml')['engine'])
    # engine.resume('epoch3')
    engine.train(train_loader, valid_loader=test_loader)
    engine.test(test_loader)