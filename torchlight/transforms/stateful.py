import abc
import random
from typing import List

import numpy as np

from . import functional as F

__ALL__ = [
    'RandCrop',
    'FlipHorizontal',
    'FlipVertical',
    'BrightnessChange',
    'MultiplicativeColorChange',
    'Contrast',
    'Rotate',
]


class StatefulTransform(abc.ABC):
    """ A StatefulTransform transform each image with identical state 
        after initilization.  
    """

    def __init__(self):
        self.state = None

    def __call__(self, img):
        if self.state is None:
            self.state = self.get_state(img)
        if isinstance(self.state, tuple):
            return self.apply(img, *self.state)
        else:
            return self.apply(img, self.state)

    def reset(self):
        self.state = None

    @abc.abstractmethod
    def get_state(self, img):
        pass

    @abc.abstractmethod
    def apply(self, img, state):
        pass


class Compose(StatefulTransform):
    def __init__(self, transforms: List):
        super().__init__()
        self.transforms = transforms

    def get_state(self, img):
        return None

    def apply(self, img, state):
        out = img
        for transform in self.transforms:
            out = transform(out)
        return out
    
    def reset(self):
        for transform in self.transforms:
            if isinstance(transform, StatefulTransform):
                transform.reset()


class RandCrop(StatefulTransform):
    """ Assume [H,W,C] format """

    def __init__(self, crop_size):
        super().__init__()
        self.croph = crop_size[0]
        self.cropw = crop_size[1]

    def get_state(self, img):
        h, w = img.shape[0], img.shape[1]
        sh = random.randint(0, h - self.croph)  # start height
        sw = random.randint(0, w - self.cropw)  # start width
        return sh, sw

    def apply(self, img, sh, sw):
        return img[sh:sh+self.croph, sw:sw+self.cropw, :]


class FlipHorizontal(StatefulTransform):
    def get_state(self, img):
        enable = np.random.randint(0, 2) > 0.5
        return enable

    def apply(self, img, enable):
        if enable:
            return F.hflip(img)
        return img


class FlipVertical(StatefulTransform):
    def get_state(self, img):
        enable = np.random.randint(0, 2) > 0.5
        return enable

    def apply(self, img, enable):
        if enable:
            return F.vflip(img)
        return img


class BrightnessChange(StatefulTransform):
    def get_state(self, img):
        strength = np.random.normal(loc=0, scale=0.02)
        return strength

    def apply(self, img, strength):
        return F.brightness_change(img, strength)


class MultiplicativeColorChange(StatefulTransform):
    def get_state(self, img):
        strength = np.random.uniform(0.9, 1.1)
        return strength

    def apply(self, img, strength):
        return F.multiplicative_color_change(img, strength)


class Contrast(StatefulTransform):
    def get_state(self, img):
        alpha = np.random.uniform(-0.3, 0.3)
        return alpha

    def apply(self, img, alpha):
        return F.contrast_change(img, alpha)


# TODO: Fix
class Rotate(StatefulTransform):
    def get_state(self, img):
        k = np.random.randint(0, 4)
        return k

    def apply(self, img, k):
        return F.rotate(img, k)
