import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from functools import partial


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


def _cal_metric(img_gt, img_test, index_fn):
    assert img_test.shape == img_gt.shape

    if len(img_test.shape) == 4:
        b = img_test.shape[0]
        out = [index_fn(img_gt[i], img_test[i]) for i in range(b)]
        return out

    return index_fn(img_gt, img_test)


def psnr(img_test, img_gt, data_range=1):
    fn = partial(peak_signal_noise_ratio, data_range=data_range)
    return _cal_metric(img_gt, img_test, fn)


def ssim(img1, img2):
    fn = structural_similarity
    return _cal_metric(img1, img2, fn)


def _sam(img1, img2, eps=1e-8):
    tmp1 = np.sum(img1*img2, axis=0) + eps
    tmp2 = np.sqrt(np.sum(img1**2, axis=0)) + eps
    tmp3 = np.sqrt(np.sum(img2**2, axis=0)) + eps
    tmp4 = tmp1 / tmp2 / tmp3
    return np.mean(np.real(np.arccos(tmp4)))


def sam(img1, img2, eps=1e-8):
    fn = partial(_sam, eps=eps)
    return _cal_metric(img1, img2, fn)


class Bandwise(object):
    def __init__(self, metric_fn):
        self.index_fn = metric_fn

    def __call__(self, img1, img2):
        C = x.shape[-3]
        bwindex = []
        for ch in range(C):
            x = X[ch, :, :]
            y = Y[ch, :, :]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex
