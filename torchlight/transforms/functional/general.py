import numpy as np
import random
import cv2
from typing import List


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def normalize(img: np.ndarray, mean: List[float], std: List[float]):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    mean = mean.reshape((1, 1, -1))
    std = std.reshape((1, 1, -1))
    return (img - mean) / std


def crop_center(img, croph, cropw):
    h, w, _ = img.shape
    starth = h//2-(croph//2)
    startw = w//2-(cropw//2)
    return img[starth:starth+croph, startw:startw+cropw, :]


def rand_crop(img, croph, cropw):
    h, w, _ = img.shape
    h1 = random.randint(0, h - croph)
    w1 = random.randint(0, w - cropw)
    return img[h1:h1+croph, w1:w1+cropw, :]


def mod_crop(img, modulo):
    h, w, _ = img.shape
    h = h - (h % modulo)
    w = w - (w % modulo)
    img = img[0:h, 0:w, :]
    return img


def mod_resize(img, base, mode=cv2.INTER_CUBIC):
    ow, oh, _ = img.shape
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return cv2.resize(img, (w, h), interpolation=mode)


def imresize(image, height=None, width=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def hwc2chw(img):
    return img.transpose(2, 0, 1)


def chw2hwc(img):
    return img.transpose(1, 2, 0)


def vflip(img):
    return img[::-1, :, :]


def hflip(img):
    return img[:, ::-1, :]


def brightness_change(img, strength):
    return img + strength


def multiplicative_color_change(img, strength):
    return img * strength


def grayscale(img):
    dst = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
    dst[:, :, 0] = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    dst = np.repeat(dst, 3, axis=2)
    return dst


def blend(img1, img2, alpha=0.5):
    return img1 * alpha + img2 * (1-alpha)


def contrast_change(img, alpha):
    gs_2 = grayscale(img)
    img = blend(gs_2, img, alpha=alpha)
    return img


def rotate(img, k):
    return np.rot90(img, k)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.
    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    import cv2
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img
