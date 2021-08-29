import torch
import torch.nn as nn


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            l = loss(predict, target)
            total_loss += l * weight
            # print(l.item(), weight)
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)


class SAMLoss(torch.nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, x1, x2, eps=1e-6):
        out = 0
        for i in range(x1.shape[0]):
            X = x1[i].squeeze()
            Y = x2[i].squeeze()
            tmp = (torch.sum(X*Y, axis=0) + eps) / (torch.sqrt(torch.sum(X **
                                                                         2, axis=0)) + eps) / (torch.sqrt(torch.sum(Y**2, axis=0)) + eps)
            out += torch.mean(torch.arccos(tmp))
        return out / x1.shape[0]


class CharbonnierLoss(nn.Module):
    def __init__(self):
        super(CharbonnierLoss,self).__init__()

    def forward(self,pre,gt):
        N = pre.shape[0]
        diff = torch.sum(torch.sqrt((pre - gt ).pow(2) + 0.001 **2)) / N
        return diff