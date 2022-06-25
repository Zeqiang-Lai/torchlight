import numpy as np

__all__ = [
    'imread_uint',
    'imread_float',
    'imsave',
    'anisave',
    'filewrite',
    'yamlwrite',
    'txt'
]


def imread_uint(path, mode='keep'):
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
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
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


def imread_float(path, mode='keep'):
    img = imread_uint(path, mode)
    img = np.float32(img / 255.)
    return img


def imsave(path, img, bgr=False):
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
    if not bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def figsave(path, img, **kwargs):
    """ Save an image with `matplotlib.pyplot`
    """
    import matplotlib.pyplot as plt
    plt.imsave(path, img, **kwargs)


def anisave(img_list, filename='a.gif', fps=60):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    def animation_generate(img):
        ims_i = []
        im = plt.imshow(img, cmap='gray')
        ims_i.append([im])
        return ims_i

    ims = []
    fig = plt.figure()
    for img in img_list:
        ims += animation_generate(img)
    ani = animation.ArtistAnimation(fig, ims)
    ani.save(filename, fps=fps, writer='ffmpeg')


def filewrite(path, content):
    """ Unified interface for saving content to file according to extension.\\
        Support extensions:
            - yaml, yml
            - json
            - txt
    """
    if path.endswith('.yaml') or path.endswith('.yml'):
        yamlwrite(path, content)
    elif path.endswith('.txt') or path.endswith('.log'):
        txtwrite(path, content)
    else:
        raise ValueError('Unrecognized extension, support [yaml, yml, json, txt, log]')


def yamlwrite(path, obj):
    import yaml
    with open(path, 'w') as f:
        yaml.dump(obj, f)


def txtwrite(path, content):
    with open(path, 'w') as f:
        f.write(content+'\n')
