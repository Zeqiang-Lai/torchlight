import random

import numpy as np

from ._util import LockedIterator
from .functional import chw2hwc, hwc2chw
from .functional import hwc2chw, chw2hwc

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        out = data
        for transform in self.transforms:
            out = transform(out)
        return out


class SequentialSelect:
    def __pos(self, n):
        i = 0
        while True:
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out


class MinMaxNormalize:
    def __call__(self, array):
        amin = np.min(array)
        amax = np.max(array)
        return (array - amin) / (amax - amin)


class CenterCrop:
    def __init__(self, size):
        self.cropx = size[0]
        self.cropy = size[1]

    def __call__(self, img):
        x, y = img.shape[0], img.shape[1]
        startx = x//2-(self.cropx//2)
        starty = y//2-(self.cropy//2)
        return img[startx:startx+self.cropx, starty:starty+self.cropy, ...]


class RandCrop:
    def __init__(self, size):
        self.cropx = size[0]
        self.cropy = size[1]

    def __call__(self, img):
        x, y = img.shape[0], img.shape[1]
        x1 = random.randint(0, x - self.cropx)
        y1 = random.randint(0, y - self.cropy)
        return img[x1:x1+self.cropx, y1:y1+self.cropy, ...]


class HWC2CHW:
    def __call__(self, img):
        return hwc2chw(img)


class CHW2HWC:
    def __call__(self, img):
        return chw2hwc(img)


class MinSize:
    def __init__(self, size: int, keep_ratio=True):
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, img):
        import cv2
        x, y = img.shape[0], img.shape[1]
        if x > self.size and y > self.size:
            return img
        
        if self.keep_ratio:
            if x > y:
                y2 = self.size
                x2 = int(self.size * x / y)
            else:
                x2 = self.size
                y2 = int(self.size * y / x)
        else:
            x2, y2 = self.size, self.size
        
        return cv2.resize(img, (y2, x2), interpolation=cv2.INTER_CUBIC)
                
        
