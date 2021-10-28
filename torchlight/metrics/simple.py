import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ._util import enable_batch_input, torch2numpy, bandwise

__all__ = [
    'accuracy',
    'top_k_acc',
    'psnr',
    'ssim',
    'sam',
    'mpsnr',
    'mssim'
]


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# B, C, H, W  | C, H, W


@torch2numpy
@enable_batch_input()
def psnr(output, target, data_range=1):
    return peak_signal_noise_ratio(target, output, data_range=data_range)


@torch2numpy
@enable_batch_input()
def ssim(img1, img2, **kwargs):
    # CHW -> HWC
    img1 = img1.transpose(1,2,0)
    img2 = img2.transpose(1,2,0)
    return structural_similarity(img1, img2, multichannel=True, **kwargs)


@torch2numpy
@enable_batch_input()
def sam(img1, img2, eps=1e-8):
    tmp1 = np.sum(img1*img2, axis=0) + eps
    tmp2 = np.sqrt(np.sum(img1**2, axis=0)) + eps
    tmp3 = np.sqrt(np.sum(img2**2, axis=0)) + eps
    tmp4 = tmp1 / tmp2 / tmp3
    return np.mean(np.real(np.arccos(tmp4)))


@torch2numpy
@enable_batch_input()
@bandwise
def mpsnr(output, target, data_range=1):
    return peak_signal_noise_ratio(target, output, data_range=data_range)


@torch2numpy
@enable_batch_input()
@bandwise
def mssim(img1, img2, **kwargs):
    return structural_similarity(img1, img2, **kwargs)
