from multiprocessing import Manager

from torch.utils.data import Dataset

from .image_folder import find_all_images


class CacheDataset(Dataset):
    def __init__(self):
        super().__init__()
        self._cache = Manager().dict()

    def __getitem__(self, index):
        if index not in self._cache:
            self._cache[index] = self.get_data(index)
        return self._cache[index]

    def get_data(self, index):
        raise NotImplementedError


class ImageFolder(Dataset):
    def __init__(self, root, mode='keep'):
        """ Load any image recursively inside provided image folder.

        Args:
            root: path of the image folder
            mode: which way you would like to load image. Defaults to 'keep'.
                  choose from ['keep', 'color', 'gray'].

        Note:
            - keep: keep image original format, color `HW3`, gray `HW1`
            - color: convert gray image to `HW3` by repeating channel dim
            - gray: read color image as gray, all image `HW1`
        """
        if mode not in ['keep', 'color', 'gray']:
            raise ValueError(f'mode must be selected from [keep, color, gray]')

        super().__init__()
        self.img_paths = sorted(find_all_images(root))
        self.mode = mode

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = self.imread_uint(path, self.mode)
        img = img.astype('float32') / 255.0
        img = img.transpose((2, 0, 1))
        return img, path

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def imread_uint(path, mode):
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
            from imageio import imread
            img = np.array(imread(path))
            img = np.expand_dims(img, axis=2)  # HW -> HW1
            return img


class SingleImageDataset(ImageFolder):
    def __init__(self, root, mode='keep', crop_size=None):
        super().__init__(root, mode)
        self.crop_size = crop_size

    def __getitem__(self, index):
        img, path = super().__getitem__(index)
        if self.crop_size is not None:
            img = rand_crop(img, self.crop_size[0], self.crop_size[1])
        return img, path


def rand_crop(img, croph, cropw):
    import random
    _, h, w = img.shape
    h1 = random.randint(0, h - croph)
    w1 = random.randint(0, w - cropw)
    return img[:, h1:h1+croph, w1:w1+cropw]


class PairImageDataset(SingleImageDataset):
    def __init__(self, transform, root, mode='keep', crop_size=None):
        super().__init__(root, mode, crop_size)
        self.transform = transform

    def __getitem__(self, index):
        img, path = super().__getitem__(index)
        trsfm_img = self.transform(img.transpose((1, 2, 0)))
        trsfm_img = trsfm_img.transpose((2, 0, 1))
        return trsfm_img, img, path
