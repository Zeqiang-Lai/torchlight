# Torchlight

Torchlight is an ultra light-weight pytorch wrapper for fast prototyping of computer vision models.

## Installation

```shell
pip install .
pip install -e . # editable installation
```

## Examples

1. Define a torchlight module alongside you pytorch model

```python
class NetModule(torchlight.SimpleModule):
    def __init__(self, lr, device):
        self.device = device
        self.model = Net().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = F.nll_loss
        self.metrics = [accuracy]
        
    def on_step_end(self, input, output, target) -> Module.StepResult:
        metrics = {'accuracy': accuracy(output, target)}
        imgs = {'input': input}
        return Module.StepResult(metrics=metrics, imgs=imgs)
```

2. train with torchlight engine

```python
import torch
from torchvision import datasets, transforms
import torchlight
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
    module = NetModule(lr=0.01, device=device)
    engine = torchlight.Engine(module, save_dir='experiments/simple')
    engine.train(train_loader, valid_loader=test_loader)
```