import numpy as np

# --------------------------------
# numpy(single) <--->  numpy(unit)
# --------------------------------


def uint2single(img):

    return np.float32(img/255.)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())


def uint162single(img):

    return np.float32(img/65535.)


def single2uint16(img):

    return np.uint8((img.clip(0, 1)*65535.).round())
