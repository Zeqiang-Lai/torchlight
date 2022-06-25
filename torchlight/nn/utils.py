import torch.nn as nn
import torch.nn.init as init
import torch
from typing import Sequence

from .ops.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d


def init_params(net, init_type='kn'):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if init_type == 'kn':
                init.kaiming_normal_(m.weight, mode='fan_out')
            if init_type == 'ku':
                init.kaiming_uniform_(m.weight, mode='fan_out')
            if init_type == 'xn':
                init.xavier_normal_(m.weight)
            if init_type == 'xu':
                init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d)):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


class ReflectPadding:
    def __init__(self, ratio):
        self.ratio = ratio

    def pad(self, img):
        _, _, h, w = img.size()
        r = self.ratio
        h_pad = (h+r) // r * r - h
        w_pad = (w+r) // r * r - w
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + w_pad]

        # log for unpadding
        self.h = h
        self.w = w

        return img

    def unpad(self, img, scale=1):
        return img[..., :self.h * scale, :self.w * scale]


def freeze(module: nn.Module):
    for _, v in module.named_parameters():
        v.requires_grad = False


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device).float()
    if isinstance(data, Sequence):
        return [to_device(d, device=device) for d in data]
    if isinstance(data, dict):
        return {k: to_device(v, device=device) for k, v in data.items()}
    return data


def load_checkpoint(model, ckpt_path):
    model.load_state_dict(torch.load(ckpt_path)['module']['model'])
