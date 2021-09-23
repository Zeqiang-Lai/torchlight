import torch

def charbonnier_loss(x, y, eps=1e-3):
    diff = x - y
    loss = torch.mean(torch.sqrt((diff * diff) + (eps*eps)))
    return loss

