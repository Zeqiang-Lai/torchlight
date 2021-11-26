from torchlight.nn.loss import FocalFrequencyLoss

import torch

if __name__ == '__main__':
    torch.random.manual_seed(2021)
    output = torch.rand(16, 3, 128, 128) 
    target = torch.rand(16, 3, 128, 128) 
    loss_fn = FocalFrequencyLoss()
    loss = loss_fn(output, target)
    print(loss.shape)
    print(loss)