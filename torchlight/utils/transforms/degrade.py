import numpy as np
from scipy import ndimage
import scipy


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1),
                         np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


class AbstractBlur:
    def __call__(self, img):
        """ input: [w,h,c] """
        img_L = ndimage.filters.convolve(
            img, np.expand_dims(self.kernel, axis=2), mode='wrap')
        return img_L


class GaussianBlur(AbstractBlur):
    def __init__(self, ksize=8, sigma=3):
        self.kernel = fspecial_gaussian(ksize, sigma)


class UniformBlur(AbstractBlur):
    def __init__(self, ksize):
        self.kernel = np.ones((ksize, ksize)) / (ksize*ksize)


## -------------------- Downsample -------------------- ##

class KFoldDownsample:
    ''' k-fold downsampler:
        Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    '''

    def __init__(self, sf):
        self.sf = sf

    def __call__(self, img):
        """ input: [w,h,c] """
        st = 0
        return img[st::self.sf, st::self.sf, :]


class UniformDownsample:
    def __init__(self, sf):
        self.sf = sf
        self.blur = UniformBlur(sf)
        self.downsampler = KFoldDownsample(sf)

    def __call__(self, img):
        """ input: [w,h,c]
        """
        img = self.blur(img)
        img = self.downsampler(img)
        return img


class GaussianDownsample:
    def __init__(self, sf, ksize=8, sigma=3):
        self.sf = sf
        self.blur = GaussianBlur(ksize, sigma)
        self.downsampler = KFoldDownsample(sf)

    def __call__(self, img):
        """ input: [w,h,c]
        """
        img = self.blur(img)
        img = self.downsampler(img)
        return img
