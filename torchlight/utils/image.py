import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


class example:
    gray = scipy.misc.ascent()
    color = scipy.misc.face()


def imread_uint(path, mode):
    """ Read image from path.
        Args:
            path: Path of image.
            mode: Which way you would like to load image. Defaults to 'keep'.
                  choose from ['keep', 'color', 'gray'].

        Note:
            - keep: keep image original format, color `HW3`, gray `HW1`
            - color: convert gray image to `HW3` by repeating channel dim
            - gray: read color image as gray, all image `HW1`
    """
    if mode not in ['keep', 'color', 'gray']:
        raise ValueError('mode should be one of [keep, color, gray]')

    import cv2
    import numpy as np

    if mode == 'keep':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            return np.expand_dims(img, axis=2)  # HW -> HW1
        return img[:, :, :3]  # remove alpha channel if exists

    if mode == 'color':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)  # RGB
        return img

    if mode == 'gray':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=2)  # HW -> HW1
        return img


def imread_float(path, mode):
    img = imread_uint(path, mode)
    img = uint2float(img)
    return img


def imsave(path, img):
    """ Save image or list of images to path. 
        - Assume value ranges from float [0,1], or uint8 [0,255], in RGB format.
        - list of images will be stacked horizontally.

        Internally call `cv2.imwrite`.
    """
    import cv2
    if isinstance(img, list):
        img = np.hstack(img)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def imshow(img):
    plt.imshow(img)
    plt.show()


def hwc2chw(img):
    return img.transpose(2, 0, 1)


def chw2hwc(img):
    return img.transpose(1, 2, 0)


def uint2float(img):
    return img.astype('float32') / 255


def float2uint(img):
    return (img * 255).astype('uint8')


def rgb2bgr(img):
    """ input [H,W,C] """
    return img[:, :, ::-1]


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
