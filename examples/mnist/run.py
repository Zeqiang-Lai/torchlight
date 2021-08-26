from torchlight.utils.config.backend import read_yaml
import torch
from torchvision import datasets, transforms

from net import NetModule
import torchlight

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    module = NetModule()
    engine = torchlight.Engine(module, save_dir='experiments/simple_l1')
    engine.config(read_yaml('config.yaml')['engine'])
    # engine.resume('epoch3')
    engine.train(train_loader, valid_loader=test_loader)
    # engine.test(test_loader)