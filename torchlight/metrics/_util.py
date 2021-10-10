import torch
import functools


def torch2numpy(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        return func(output, target, *args, **kwargs)
    return warpped


def bandwise(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        C = output.shape[-3]
        total = 0
        for ch in range(C):
            x = output[ch, :, :]
            y = target[ch, :, :]
            total += func(x, y, *args, **kwargs)
        return total / C
    return warpped


def enable_batch_input(reduce=True):
    def inner(func):
        @functools.wraps(func)
        def warpped(output, target, *args, **kwargs):
            if len(output.shape) == 4:
                b = output.shape[0]
                out = [func(output[i], target[i]) for i in range(b)]
                if reduce:
                    return sum(out) / len(out)
                return out
            return func(output, target, *args, **kwargs)
        return warpped
    return inner
