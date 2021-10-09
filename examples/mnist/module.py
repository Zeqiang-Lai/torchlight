import torch.optim as optim
import torch.nn.functional as F

import torchlight
from torchlight.trainer import Module

from net import Net

def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(pred)

class NetModule(torchlight.SMSOModule):
    def __init__(self, lr, device):
        model = Net()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        super().__init__(model, optimizer)
        self.device = device
        self.model.to(device)
        self.criterion = F.nll_loss
        self.metrics = [accuracy]
    
    def _step(self, data, train, epoch, step):
        input, target = data
        input, target = input.to(self.device), target.to(self.device)
        output = self.model(input)
        loss = self.criterion(output, target)
        
        metrics = {'loss': loss.item(), 'accuracy': accuracy(output, target)}
        imgs = {'input': input}

        return loss, Module.StepResult(metrics=metrics, imgs=imgs)