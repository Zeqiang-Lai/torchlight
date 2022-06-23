import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# helper functions

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def img_files(dir):
    files = sorted(os.listdir(dir))
    return [x for x in files if is_image_file(x)]


def full_path(basedir, filenames):
    return [os.path.join(basedir, x) for x in filenames]


def augmentation(aug, img):
    if aug == 1:
        img = img.flip(1)
    elif aug == 2:
        img = img.flip(2)
    elif aug == 3:
        img = torch.rot90(img, dims=(1, 2))
    elif aug == 4:
        img = torch.rot90(img, dims=(1, 2), k=2)
    elif aug == 5:
        img = torch.rot90(img, dims=(1, 2), k=3)
    elif aug == 6:
        img = torch.rot90(img.flip(1), dims=(1, 2))
    elif aug == 7:
        img = torch.rot90(img.flip(2), dims=(1, 2))
    return img

# datasets

class PairDatasetTrain(Dataset):
    def __init__(self, imgA_dir, imgB_dir, patch_size):
        super(PairDatasetTrain, self).__init__()

        self.pathsA = full_path(imgA_dir, img_files(imgA_dir))
        self.pathsB = full_path(imgB_dir, img_files(imgB_dir))

        lenA = len(self.pathsA)
        lenB = len(self.pathsB)
        if lenA != lenB:
            print(f'the number of imgs in A({lenA}) and B({lenB}) is not equal')
        self.size = min(lenA, lenB)  # use fewer one if not equal

        self.ps = patch_size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = index % self.size
        ps = self.ps

        pathA = self.pathsA[index]
        pathB = self.pathsB[index]

        imgA = Image.open(pathA)
        imgB = Image.open(pathB)

        # Reflect Pad in case image is smaller than patch_size
        h, w = imgB.size
        padh = ps-h if h < ps else 0
        padw = ps-w if w < ps else 0
        if padh != 0 or padw != 0:
            imgA = TF.pad(imgA, (0, 0, padh, padw), padding_mode='reflect')
            imgB = TF.pad(imgB, (0, 0, padh, padw), padding_mode='reflect')

        # Convert PIL image to tensor
        imgA = TF.to_tensor(imgA)
        imgB = TF.to_tensor(imgB)

        # Crop patch
        hh, ww = imgB.shape[1], imgB.shape[2]
        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        imgA = imgA[:, rr:rr+ps, cc:cc+ps]
        imgB = imgB[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        aug = random.randint(0, 8)
        imgA = augmentation(aug, imgA)
        imgB = augmentation(aug, imgB)

        return imgA, imgB


class PairDatasetVal(PairDatasetTrain):
    def __init__(self, imgA_dir, imgB_dir, patch_size=None):
        super().__init__(imgA_dir, imgB_dir, patch_size)

    def __getitem__(self, index):
        index = index % self.size
        ps = self.ps

        pathA = self.pathsA[index]
        pathB = self.pathsB[index]

        imgA = Image.open(pathA)
        imgB = Image.open(pathB)

        # Validate on center crop
        if self.ps is not None:
            imgA = TF.center_crop(imgA, (ps, ps))
            imgB = TF.center_crop(imgB, (ps, ps))

        imgA = TF.to_tensor(imgA)
        imgB = TF.to_tensor(imgB)

        return imgA, imgB


class PairDatasetTest(Dataset):
    def __init__(self, inp_dir):
        super(PairDatasetTest, self).__init__()

        self.paths_inp = full_path(inp_dir, img_files(inp_dir))
        self.size = len(self.paths_inp)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        path_inp = self.paths_inp[index]
        
        inp = Image.open(path_inp)
        inp = TF.to_tensor(inp)
        
        return inp, path_inp
