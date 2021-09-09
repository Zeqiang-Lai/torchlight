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
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pre, gt):
        N = pre.shape[0]
        diff = torch.sum(torch.sqrt((pre - gt).pow(2) + self.eps**2)) / N
        return diff


class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)    # filter
        down = filtered[:, :, ::2, ::2]               # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4                  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
